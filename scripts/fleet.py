#!/usr/bin/env python3
"""The docker container fleet -- the Kubernetes pod shape, ported to docker.

Every harness in this family shares this module. What differs between them is
the *shape* (client count, queue depth) and *which clients get reclaimed*;
bringing containers up, mapping clients to containers, checking progress and
scoring are the same everywhere.

Rules that have to hold:

* **Do not create workers with fork.** If the parent is PID 1, its death tears
  down the PID namespace and the kernel SIGKILLs every sibling -- and that death
  looks exactly like MPS propagation. Workers are children of a non-CUDA shim,
  started with subprocess.Popen.
* **Do not delete containers inside the measurement window.** Delete one and you
  can no longer separate "the reclaim killed it" from "the container went away".
* **`--pid=host` is required.** Client-to-container mapping is done by host pid;
  without it the mapping is empty and you cannot say who was reclaimed.
* **A client that never attached is not a bystander that survived.** Fewer
  clients than expected means the cell is VOID, not clean.
"""
import json
import os
import re
import subprocess
import time

IMG = os.environ.get("IMG", "nvcr.io/nvidia/pytorch:26.06-py3")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# Workload scripts and the prebuilt spin kernel; both default to inside this repo.
# queuefill.py raises FileNotFoundError without the kernel cubin, and then not a
# single offender attaches, which makes the whole cell VOID (measured: all 16
# workers died, 2 clients left).
WL = os.environ.get("WL", _HERE)
ART = os.environ.get("ART", os.path.join(_ROOT, "artifacts"))

POISON_KEYS = ("POISONED", "illegal memory", "IllegalAddress", "CUDNN_STATUS_",
               "CUBLAS_STATUS_", "cudaError")


def sh(cmd, timeout=180):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, timeout=timeout)
    return r.stdout or ""


class Fleet(object):
    def __init__(self, prefix, gpu_uuid, root, say, off_pct="25",
                 off_libvgpu=None):
        self.prefix = prefix
        self.gpu_uuid = gpu_uuid
        self.root = root
        self.say = say
        self.off_pct = off_pct
        self.off_libvgpu = off_libvgpu
        self.mps = prefix + "_mps"
        self.off = []
        self.vic = []

    # ------------------------------------------------------------- lifecycle
    def clean(self, keep_mps=False):
        """Remove containers. With keep_mps, the MPS daemon container stays.

        Starting a fresh daemon per rep is clean -- no cross-cell contamination --
        but it makes cumulative effects invisible. The 19ms -> 225ms monotonic
        degradation seen within a 40-client sequential reclaim means the damage
        accumulates inside one daemon, so measuring that axis requires keeping
        the daemon alive across reps.
        """
        out = sh(["docker", "ps", "-a", "--format", "{{.Names}}", "--filter",
                  "name=" + self.prefix + "_"], timeout=60).split()
        names = [n for n in out if not (keep_mps and n == self.mps)]
        if names:
            sh(["docker", "rm", "-f"] + names, timeout=300)

    def mps_alive(self):
        out = sh(["docker", "ps", "--format", "{{.Names}}", "--filter",
                  "name=^%s$" % self.mps], timeout=60).split()
        return self.mps in out

    def start_mps(self, reuse=False):
        if reuse and self.mps_alive():
            # Keep the daemon as it is, and do NOT wipe the pipe directory --
            # wiping it removes the live daemon's socket and every later call fails.
            self.say("reusing MPS daemon (%s)" % self.mps)
            return
        subprocess.run("rm -rf %s/mps %s/mpslog && mkdir -p %s/mps %s/mpslog"
                       % (self.root, self.root, self.root, self.root),
                       shell=True, timeout=60)
        self.say("starting MPS daemon container (%s)" % self.gpu_uuid)
        sh(["docker", "run", "-d", "--name", self.mps, "--runtime=nvidia",
            "--ipc=host", "--pid=host",
            "-e", "NVIDIA_VISIBLE_DEVICES=" + self.gpu_uuid,
            "-e", "CUDA_MPS_PIPE_DIRECTORY=/mps",
            "-e", "CUDA_MPS_LOG_DIRECTORY=/mpslog",
            "-v", self.root + "/mps:/mps", "-v", self.root + "/mpslog:/mpslog",
            IMG, "bash", "-c",
            "nvidia-cuda-mps-control -d && sleep infinity"], timeout=600)
        time.sleep(6)

    def _common(self):
        return ["--runtime=nvidia", "--ipc=host", "--pid=host",
                "-e", "NVIDIA_VISIBLE_DEVICES=" + self.gpu_uuid,
                "-e", "CUDA_MPS_PIPE_DIRECTORY=/mps",
                "-v", self.root + "/mps:/mps", "-v", WL + ":/wl:ro",
                "-v", ART + ":/artifacts:ro"]

    def start_victims(self, nvic, batch):
        """Start the victims FIRST, alone.

        Starting them together with the offenders produced victims that never
        completed a single step, and in that state there is no way to tell
        (a) slowed down by SM contention from (b) never dispatched at all. MPS
        time-slices once clients ask for more SMs than exist, so plain contention
        does not drive progress to zero -- which is exactly why the distinction
        matters.

        Bringing the victims up to full speed first and only then attaching the
        offenders makes the answer readable: either they keep stepping or they stop.
        """
        self.vic = ["%s_v%d" % (self.prefix, i) for i in range(1, nvic + 1)]
        for i, v in enumerate(self.vic):
            env = ["-e", "PYTHONUNBUFFERED=1", "-e", "ROLE=victim%d" % (i + 1),
                   "-e", "BATCH=%s" % batch, "-e", "NO_EXIT_ON_POISON=0",
                   "-e", "EXIT_MODE=exit"]
            sh(["docker", "run", "-d", "--name", v] + self._common() + env
               + [IMG, "bash", "-c",
                  "python3 /wl/victim_resnet.py 2>&1 | tee /tmp/w1.log"],
               timeout=600)
        time.sleep(3)

    def start_offenders(self, noff, nproc, qdepth, ksec):
        """Start the offenders. PID 1 of each container is pid1_shim.

        Same structure as the Kubernetes pods: PID 1 never touches CUDA, and the
        N workers are its children. Not fork -- if the parent is PID 1, its death
        collapses the namespace, the kernel SIGKILLs every sibling, and that
        death is indistinguishable from MPS propagation.
        """
        self.off = ["%s_o%d" % (self.prefix, i) for i in range(1, noff + 1)]
        for i, o in enumerate(self.off):
            env = ["-e", "PYTHONUNBUFFERED=1", "-e", "ART_DIR=/artifacts",
                   "-e", "WORK_DIR=/wl", "-e", "ROLE=off%d" % (i + 1),
                   "-e", "NPROC=%d" % nproc, "-e", "KERNEL_SEC=%s" % ksec,
                   "-e", "QUEUE_DEPTH=%s" % qdepth, "-e", "SPIN_SEC=%s" % ksec,
                   "-e", "NO_EXIT_ON_POISON=0", "-e", "EXIT_MODE=exit"]
            if self.off_libvgpu:
                # Confine only the offenders under a TPC mask.
                #
                # In bare docker a ResNet neighbour stalls permanently the moment
                # the offenders attach, while under Kubernetes the same 16
                # offenders leave it running at 17 steps/s. The only difference
                # is the vGPU library's TPC mask (PARTITION_PERCENTAGE=25), and
                # an MPS thread cap does not substitute for it (25% and 100%
                # measured identical).
                #
                # So the mask goes on the offenders only, holding their SM
                # footprint to 25%; the victims run unmasked. That is not the
                # Kubernetes topology (there every pod gets its own partition),
                # but it preserves the property that matters: the offenders
                # cannot take every SM. State this difference when reporting.
                env += ["-e", "CONFIG_DIR=/etc/block_gpu",
                        "-e", "LD_PRELOAD=/etc/block_gpu/lib/libvgpu.so"]
            if self.off_pct and self.off_pct != "100":
                # Approximates the Kubernetes TPC mask (PARTITION_PERCENTAGE=25).
                # **Different mechanism**: a TPC mask partitions SMs spatially,
                # this caps threads per client. A known non-correspondence of the
                # docker port.
                env += ["-e", "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=%s" % self.off_pct]
            mounts = []
            if self.off_libvgpu:
                mounts = ["-v", self.off_libvgpu + ":/etc/block_gpu:ro"]
            sh(["docker", "run", "-d", "--name", o] + self._common() + env + mounts
               + [IMG, "bash", "-c",
                  "python3 /wl/pid1_shim.py queuefill.py 2>&1 | tee /tmp/w1.log"],
               timeout=600)
        time.sleep(3)

    def wait_clients(self, cli, want, boot=600):
        self.say("expecting %d MPS clients" % want)
        t0, last, stable, pids = time.time(), -1, 0, []
        while time.time() - t0 < boot:
            _, pids = cli.ps(label="boot")
            if len(pids) != last:
                self.say("  MPS clients=%d" % len(pids))
                last, stable = len(pids), 0
            else:
                stable += 1
            if len(pids) >= want and stable >= 6:
                break
            time.sleep(5)
        self.say("  settled at %d clients (expected %d, stable %d polls)"
                 % (len(pids), want, stable))
        return pids

    # -------------------------------------------------------------- mapping
    def owner_map(self, pids):
        owner = {}
        for n in self.off + self.vic:
            out = sh(["docker", "top", n, "-eo", "pid"], timeout=60)
            cp = {int(x) for x in re.findall(r"^\s*(\d+)\s*$", out, re.M)}
            for p in pids:
                if p in cp:
                    owner[p] = n
        return owner

    # ------------------------------------------------------------- progress
    def logs(self, name):
        # With the workload as PID 1 its output goes to the container's stdout.
        # Read it with `docker logs` so it is still readable after the container
        # dies -- reading the file via `exec` yields nothing from a dead container.
        return sh(["docker", "logs", "--tail", "400", name], timeout=60)

    def last_step(self, name):
        """Last progress event. Victims emit `step`, queuefill offenders `qfill`/`done`.

        Offenders are neighbours too. Voiding a cell on the victims alone throws
        away the case where **an untouched offender took an illegal access while
        it was still running** (measured: one offender poisoned at done=3 while
        the victims in the same cell were still at step=0).
        """
        for ln in reversed(self.logs(name).splitlines()):
            if "EVT " not in ln:
                continue
            try:
                d = json.loads(ln.split("EVT ", 1)[1])
            except Exception:                                      # noqa: BLE001
                continue
            if d.get("ev") == "step":
                return d.get("step"), d.get("sps")
            if d.get("ev") in ("qfill", "done"):
                return d.get("done"), None
        return None, None

    def wait_victims_ready(self, min_step=200, timeout=300, poll=5.0):
        """Wait until the victims are actually training.

        ResNet-152 needs 20-50 s after container start before cuDNN autotune
        settles. Trusting a fixed `--steady 30` fires the reclaim while the
        victim has not completed its first step (measured: every POISONED line
        carried step 0, and the progress gate read step=None).

        A death in that state is "a neighbour died while starting up", not "a
        running neighbour was killed", so it cannot be the same event as the
        Kubernetes reproduction (killed at step 2610 while advancing +90/5 s).
        Claiming propagation requires passing this gate.
        """
        t0 = time.time()
        last = None
        while time.time() - t0 < timeout:
            cur = {v: self.last_step(v)[0] for v in self.vic}
            ok = all(isinstance(cur[v], int) and cur[v] >= min_step for v in self.vic)
            if cur != last:
                self.say("  victim readiness: %s (need step>=%d)"
                         % (", ".join("%s=%s" % (v, cur[v]) for v in self.vic),
                            min_step))
                last = cur
            if ok:
                return True
            time.sleep(poll)
        self.say("  victim readiness failed: %s" % last)
        return False

    def progress(self, names, tag, gap=5.0):
        """Check a set of containers for progress, as a delta over `gap` seconds.

        "The reclaim killed the neighbour" is evidence of propagation only if
        that neighbour was actually running when the reclaim fired. Each
        neighbour is judged on its own -- one of them being stopped does not
        invalidate another one's death.
        """
        a0 = {n: self.last_step(n)[0] for n in names}
        time.sleep(gap)
        live = {}
        for n in names:
            s1, sps = self.last_step(n)
            d = (s1 - a0[n]) if (isinstance(s1, int) and isinstance(a0[n], int)) else None
            live[n] = bool(d)
            self.say("  %s %s: prog=%s (+%s in %.0fs) sps=%s%s"
                     % (tag, n, s1, d, gap, sps, "  <-- NOT PROGRESSING" if not d else ""))
        return live

    def victim_progress(self, tag, gap=5.0):
        """Check the victims specifically, just before firing."""
        live = self.progress(self.vic, tag + " victim", gap=gap)
        return all(live.values())

    def score(self, names, tag):
        nerr = nill = 0
        for n in names:
            hits = [l for l in self.logs(n).splitlines()
                    if any(k in l for k in POISON_KEYS)]
            if hits:
                nerr += 1
                self.say("  %s %s: %s" % (tag, n, hits[-1][:150]))
                if any(("illegal memory" in h or "IllegalAddress" in h) for h in hits):
                    nill += 1
            else:
                self.say("  %s %s: no error" % (tag, n))
        return nerr, nill

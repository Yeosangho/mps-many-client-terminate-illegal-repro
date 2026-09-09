#!/usr/bin/env python3
"""
QUEUEFILL: keep a context's channel queue full of long kernels, nothing else.

The bystander in this experiment used to be a ResNet152 trainer, which drags
cuDNN, an autotuner and a large caching allocator into the measurement -- and
those produced failures of their own that had nothing to do with MPS.  It also
spends most of its time between kernels, so at any instant its queue is nearly
empty.

Here the tenant does no framework work at all.  It keeps QUEUE_DEPTH launches
of the prebuilt ~KERNEL_SEC kernel outstanding on one stream, topping the queue
up as each completes, so the context always has a backlog of long kernels
queued behind a resident one.  torch is used only to create the primary context
and own the device memory the kernel writes; no torch kernel is ever launched
except the calibration the loader needs.

Liveness is the completion counter: if the context is poisoned the next
synchronize raises and the counter stops.
"""
import collections
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import maybe_start_graph_assert
from common import (cuda_init_with_retry, ev, is_cuda_fault,
                    checksum_reference, checksum_verify)  # noqa: E402
from spin import SpinKernel, default_artifact  # noqa: E402

KERNEL_SEC = float(os.environ.get("KERNEL_SEC", os.environ.get("SPIN_SEC", "5.0")))
QUEUE_DEPTH = int(os.environ.get("QUEUE_DEPTH", "8"))
REPORT_EVERY = int(os.environ.get("REPORT_EVERY", "1"))
NO_EXIT_ON_POISON = os.environ.get("NO_EXIT_ON_POISON", "0") == "1"
EXIT_MODE = os.environ.get("EXIT_MODE", "exit")

_stop = {"v": False}


def _on_sig(signum, frame):
    ev("signal", signum=int(signum))
    _stop["v"] = True


def main():
    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, _on_sig)

    import torch

    ev("boot", torch=torch.__version__, pid=os.getpid(),
       kernel_sec=KERNEL_SEC, queue_depth=QUEUE_DEPTH)

    if not cuda_init_with_retry():
        return 44
    maybe_start_graph_assert()

    art = default_artifact(os.environ.get("ART_DIR", "/artifacts"))
    qk = SpinKernel(art)
    hz = qk.calibrate()
    props = torch.cuda.get_device_properties(0)

    stream = torch.cuda.current_stream()
    pending = collections.deque()

    def enqueue():
        qk.spin_seconds(KERNEL_SEC)
        e = torch.cuda.Event()
        e.record(stream)
        pending.append(e)

    # Fill the queue before announcing readiness, so the measured window starts
    # with the backlog already established rather than while it is building.
    for _ in range(QUEUE_DEPTH):
        enqueue()

    ck = checksum_reference()
    VERIFY_EVERY = int(os.environ.get("VERIFY_EVERY", "10"))

    ev("ready",
       checksum_ref=ck,
       gpu=props.name,
       sm_count=props.multi_processor_count,
       total_mem_mb=int(props.total_memory / 1024 / 1024),
       artifact=os.path.basename(art),
       sm_ghz=round(hz / 1e9, 4),
       kernel_sec=KERNEL_SEC,
       queue_depth=QUEUE_DEPTH)

    done = 0
    t_start = time.time()
    while not _stop["v"]:
        try:
            # Wait on the oldest launch, then top the queue back up: the depth
            # stays at QUEUE_DEPTH for the whole run.
            oldest = pending.popleft()
            oldest.synchronize()
            done += 1
            enqueue()
            if VERIFY_EVERY and done % VERIFY_EVERY == 0:
                ok, ref, got = checksum_verify()
                if ok is False:
                    ev("CORRUPT", done=done, ref=ref, got=got,
                       uptime=round(time.time() - t_start, 2))
                else:
                    ev("verify", done=done, ok=True)
            if done % REPORT_EVERY == 0:
                ev("qfill",
                   done=done,
                   depth=len(pending),
                   completed_kernels=qk.completed_spins,
                   uptime=round(time.time() - t_start, 2))
        except Exception as e:  # noqa: BLE001
            kind = "POISONED" if is_cuda_fault(e) else "ERROR"
            ev(kind, done=done, exc=type(e).__name__, msg=str(e)[:800],
               uptime=round(time.time() - t_start, 2))
            if NO_EXIT_ON_POISON:
                ev("holding_after_poison")
                while not _stop["v"]:
                    time.sleep(0.5)
                return 0
            if EXIT_MODE == "selfkill":
                ev("selfkill")
                os.kill(os.getpid(), signal.SIGKILL)
            os._exit(42 if kind == "POISONED" else 43)

    ev("stopping", done=done, uptime=round(time.time() - t_start, 2))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

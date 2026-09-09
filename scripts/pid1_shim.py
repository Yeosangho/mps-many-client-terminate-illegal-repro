#!/usr/bin/env python3
"""
A container PID 1 that never touches CUDA.

Why this exists
---------------
`term_all` signals the processes registered in the block-GPU shared region and
waits for them to exit.  When the container's PID 1 is itself one of those
processes, it finishes its own terminate_client first and exits -- and the
moment PID 1 leaves, the kernel tears down the PID namespace and SIGKILLs every
sibling that is still inside its own terminate_client, with GPU kernels still
resident.  That SIGKILL is what poisons the shared MPS context.

This shim keeps PID 1 out of CUDA entirely, so:
  * it never registers in the shared region -> term_all never signals it,
  * it stays alive while the real workers clean up -> the namespace holds,
  * only after every worker has exited does it exit itself.

Usage:  pid1_shim.py <workload script name under /work>
"""
import os
import signal
import subprocess
import sys
import time

WORK = os.environ.get("WORK_DIR", "/work")
NPROC = int(os.environ.get("NPROC", "1"))
BASE_ROLE = os.environ.get("ROLE", "worker")
GRACE = float(os.environ.get("SHIM_GRACE_SEC", "60"))

_kids = []
_stopping = {"v": False}


def _log(msg):
    sys.stdout.write("SHIM %s\n" % msg)
    sys.stdout.flush()


def _forward(signum, frame):
    if _stopping["v"]:
        return
    _stopping["v"] = True
    _log("got signal %d, forwarding to %d worker(s)" % (signum, len(_kids)))
    for p in _kids:
        try:
            p.send_signal(signum)
        except Exception:
            pass


def main():
    if len(sys.argv) < 2:
        _log("usage: pid1_shim.py <script>")
        return 2
    script = sys.argv[1]

    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, _forward)

    for i in range(NPROC):
        env = dict(os.environ)
        env["ROLE"] = "%s-p%d" % (BASE_ROLE, i)
        env["NPROC"] = "1"          # each worker is a single process
        env["PYTHONUNBUFFERED"] = "1"
        p = subprocess.Popen(["python3", os.path.join(WORK, script)], env=env)
        _kids.append(p)
        _log("started worker %d pid=%d role=%s" % (i, p.pid, env["ROLE"]))

    _log("pid1=%d is not a CUDA process; holding the PID namespace open" % os.getpid())

    # Wait for every worker.  Staying alive here is the whole point: the
    # namespace must outlive the workers' terminate_client.
    deadline = None
    while _kids:
        for p in list(_kids):
            if p.poll() is not None:
                _log("worker pid=%d exited rc=%s" % (p.pid, p.returncode))
                _kids.remove(p)
        if _stopping["v"] and deadline is None:
            deadline = time.time() + GRACE
        if deadline and time.time() > deadline:
            _log("grace expired, %d worker(s) still alive" % len(_kids))
            break
        time.sleep(0.05)

    _log("all workers gone, pid 1 exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main())

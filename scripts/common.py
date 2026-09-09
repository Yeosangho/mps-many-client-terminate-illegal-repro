"""Shared heartbeat/event emission so every workload lands on one timeline."""
import datetime
import json
import os
import sys


def _now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


ROLE = os.environ.get("ROLE", "unknown")


def ev(kind, **kw):
    rec = {"t": _now(), "role": ROLE, "ev": kind}
    rec.update(kw)
    sys.stdout.write("EVT " + json.dumps(rec, default=str) + "\n")
    sys.stdout.flush()


def is_cuda_fault(exc):
    """True when the exception looks like the shared context got poisoned."""
    s = ("%s: %s" % (type(exc).__name__, exc)).lower()
    needles = (
        "cuda error",
        "cublas",
        "cudnn",
        "unspecified launch failure",
        "an illegal memory access",
        "device-side assert",
        "invalid resource handle",
        "context is destroyed",
        "initialization error",
        "no cuda-capable device",
        "out of memory",  # MPS teardown often surfaces as an allocator failure
        "misaligned address",
        "launch timed out",
        "operating system call failed",
        "not permitted",
        "system not yet initialized",
        "client is not connected",
    )
    return any(n in s for n in needles)


def _is_device_unavailable(exc):
    """The transient MPS-server refusal, as opposed to a real fault."""
    s = ("%s" % exc).lower()
    return ("busy or unavailable" in s
            or "system not yet initialized" in s
            or "no cuda-capable device" in s)


def cuda_init_with_retry(attempts=None, delay=None):
    """First CUDA touch, retried.

    Right after a previous trial tears down, a fresh client can get
    "CUDA-capable device(s) is/are busy or unavailable" from the MPS server
    while it still has the old clients' state to release.  It clears on retry,
    so retry here instead of losing the whole trial to a startup error.
    """
    import time
    import torch
    # Tunable: right after a gpu-plugin restart the MPS server can refuse
    # clients for longer than the old fixed 32s budget.
    if attempts is None:
        attempts = int(os.environ.get("CUDA_INIT_ATTEMPTS", "8"))
    if delay is None:
        delay = float(os.environ.get("CUDA_INIT_DELAY", "4"))
    last = None
    for i in range(attempts):
        try:
            torch.zeros(1, device="cuda")
            torch.cuda.synchronize()
            if i:
                ev("cuda_init_retry_ok", attempt=i)
            return True
        except Exception as e:  # noqa: BLE001
            last = e
            ev("cuda_init_retry", attempt=i, msg=str(e)[:200])
            time.sleep(delay)
    # A cached refusal cannot be retried away in this process; re-exec so the
    # next attempt runs against a freshly loaded libcuda.
    budget = int(os.environ.get("CUDA_INIT_RELAUNCH", "2"))
    used = int(os.environ.get("_CUDA_INIT_RELAUNCH_USED", "0"))
    if used < budget and _is_device_unavailable(last):
        os.environ["_CUDA_INIT_RELAUNCH_USED"] = str(used + 1)
        ev("cuda_init_relaunch", attempt=used + 1, budget=budget,
           msg=str(last)[:200])
        sys.stdout.flush()
        try:
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except OSError as e:  # noqa: BLE001
            ev("cuda_init_relaunch_failed", msg="%s: %s" % (type(e).__name__, e))
    ev("cuda_init_failed", msg=str(last)[:400])
    return False


# --- silent-corruption check ------------------------------------------------
# Integer arithmetic so the result is bit-exact regardless of occupancy,
# algorithm selection or clocks: any difference is corruption, not noise.
_CK = {"ref": None, "buf": None}


def gpu_checksum(n=1 << 20):
    """Deterministic int64 checksum computed on the GPU."""
    import torch
    if _CK["buf"] is None:
        _CK["buf"] = torch.arange(n, device="cuda", dtype=torch.int64) % 9973
    a = _CK["buf"]
    return int(((a * a) % 10007).sum().item())


def checksum_reference():
    """Take the reference before anything is terminated."""
    _CK["ref"] = gpu_checksum()
    return _CK["ref"]


def checksum_verify():
    """(ok, ref, got). ok is None when no reference was taken."""
    if _CK["ref"] is None:
        return None, None, None
    got = gpu_checksum()
    return got == _CK["ref"], _CK["ref"], got


# --- on-demand graph assert (GRAPH_ASSERT=1) ------------------------------
# A watchdog kernel kept resident through a CUDA graph; the driver terminates
# this tenant by creating the trigger file, which makes the kernel assert and
# take the context down with CUDA 710 instead of a teardown with a resident
# kernel.  Started after the first CUDA touch so the graph is instantiated and
# the kernel is on the GPU before the real work begins.
def maybe_start_graph_assert():
    import os
    if os.environ.get("GRAPH_ASSERT", "0") != "1":
        return None
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("GRAPH_ONDEMAND_CUBIN", "/artifacts/wdog_ondemand.cubin")
    try:
        import graph_ondemand
        ok = graph_ondemand.start()
        ev("graph_assert_armed" if ok else "graph_assert_failed",
           mode="ondemand",
           trigger=os.environ.get("GRAPH_ONDEMAND_TRIGGER", "/nl/gtrig_dev0"),
           cubin=os.environ["GRAPH_ONDEMAND_CUBIN"],
           err=graph_ondemand.S.get("err"))
        return graph_ondemand if ok else None
    except Exception as e:                                   # noqa: BLE001
        ev("graph_assert_failed", mode="ondemand", err=str(e)[:200])
        return None

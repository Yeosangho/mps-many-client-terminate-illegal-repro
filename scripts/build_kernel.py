#!/usr/bin/env python3
"""
Build the long-running "spin" CUDA kernel ONCE with NVRTC and persist the
cubin so every later experiment run only has to cuModuleLoadData() it.

The pytorch runtime image has no nvcc/ptxas, but it does ship libnvrtc
(pulled in by the nvidia-cuda-nvrtc wheel), so we compile at build time and
save the binary artifact to the hostPath-mounted /artifacts directory.

Output:  /artifacts/spin_sm<CC>.cubin   (+ .meta.json describing how it was built)
"""
import ctypes
import ctypes.util
import glob
import hashlib
import json
import os
import sys
import time

ART_DIR = os.environ.get("ART_DIR", "/artifacts")

# The kernel: every thread busy-waits on the SM clock until `cycles` have
# elapsed.  A single launch therefore occupies its TPCs for a controllable
# wall-clock duration (we target ~5s, matching the earlier experiment's
# "5초 커널").  `flag` lets the host observe that it really ran, and
# `abort_flag` (a mapped host word) lets us cut a spin short if we ever need
# to; it is never written during the experiment itself.
KERNEL_SRC = r"""
extern "C" __global__
void spin_kernel(unsigned long long cycles,
                 volatile int *flag,
                 volatile int *abort_flag)
{
    unsigned long long t0 = clock64();
    // Loop is written so the compiler cannot hoist/eliminate it.
    while (true) {
        unsigned long long now = clock64();
        if (now - t0 >= cycles) break;
        if (abort_flag != 0 && *abort_flag != 0) break;
    }
    if (threadIdx.x == 0 && blockIdx.x == 0 && flag != 0) {
        *flag = *flag + 1;
    }
}
"""


def _find_nvrtc():
    cands = []
    # torch bundles it under site-packages/nvidia/cuda_nvrtc/lib
    for pat in (
        "/usr/local/lib/python*/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so*",
        "/opt/conda/lib/python*/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so*",
        "/usr/local/lib/python*/dist-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so*",
        "/usr/local/cuda*/lib64/libnvrtc.so*",
        "/usr/lib/x86_64-linux-gnu/libnvrtc.so*",
    ):
        cands.extend(sorted(glob.glob(pat)))
    try:
        import nvidia.cuda_nvrtc.lib as _l  # noqa
        cands.extend(sorted(glob.glob(os.path.join(os.path.dirname(_l.__file__), "libnvrtc.so*"))))
    except Exception:
        pass
    soname = ctypes.util.find_library("nvrtc")
    if soname:
        cands.append(soname)
    for c in cands:
        # skip the tiny "builtins" stub
        if "builtins" in os.path.basename(c):
            continue
        try:
            return ctypes.CDLL(c), c
        except OSError:
            continue
    raise RuntimeError("libnvrtc not found; tried: %s" % cands)


def _check_nvrtc(nvrtc, res, prog=None):
    if res == 0:
        return
    msg = "nvrtc error %d" % res
    if prog is not None:
        n = ctypes.c_size_t()
        nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
        buf = ctypes.create_string_buffer(n.value)
        nvrtc.nvrtcGetProgramLog(prog, buf)
        msg += "\n" + buf.value.decode(errors="replace")
    raise RuntimeError(msg)


def main():
    os.makedirs(ART_DIR, exist_ok=True)

    import torch  # noqa

    if not torch.cuda.is_available():
        print("FATAL: no CUDA device visible; build must run on a GPU pod", file=sys.stderr)
        return 2
    major, minor = torch.cuda.get_device_capability(0)
    cc = "%d%d" % (major, minor)
    name = torch.cuda.get_device_name(0)
    print("device: %s  (compute capability %d.%d)" % (name, major, minor))

    nvrtc, nvrtc_path = _find_nvrtc()
    print("nvrtc:  %s" % nvrtc_path)

    ver_major, ver_minor = ctypes.c_int(), ctypes.c_int()
    nvrtc.nvrtcVersion(ctypes.byref(ver_major), ctypes.byref(ver_minor))
    print("nvrtc version: %d.%d" % (ver_major.value, ver_minor.value))

    prog = ctypes.c_void_p()
    _check_nvrtc(nvrtc, nvrtc.nvrtcCreateProgram(
        ctypes.byref(prog), KERNEL_SRC.encode(), b"spin.cu", 0, None, None))

    # Try a real-architecture (cubin) build first so nothing has to be JIT'd at
    # load time.  Fall back to PTX if this NVRTC cannot target the arch.
    out_kind = None
    payload = None
    for arch, kind in (("sm_%s" % cc, "cubin"), ("compute_%s" % cc, "ptx")):
        opts = [("--gpu-architecture=%s" % arch).encode(), b"-default-device"]
        arr = (ctypes.c_char_p * len(opts))(*opts)
        res = nvrtc.nvrtcCompileProgram(prog, len(opts), arr)
        if res != 0:
            n = ctypes.c_size_t()
            nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
            buf = ctypes.create_string_buffer(n.value)
            nvrtc.nvrtcGetProgramLog(prog, buf)
            print("compile for %s failed:\n%s" % (arch, buf.value.decode(errors="replace")))
            continue
        n = ctypes.c_size_t()
        if kind == "cubin" and hasattr(nvrtc, "nvrtcGetCUBINSize"):
            _check_nvrtc(nvrtc, nvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(n)), prog)
            buf = ctypes.create_string_buffer(n.value)
            _check_nvrtc(nvrtc, nvrtc.nvrtcGetCUBIN(prog, buf), prog)
            payload, out_kind = buf.raw[: n.value], "cubin"
        else:
            _check_nvrtc(nvrtc, nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(n)), prog)
            buf = ctypes.create_string_buffer(n.value)
            _check_nvrtc(nvrtc, nvrtc.nvrtcGetPTX(prog, buf), prog)
            payload, out_kind = buf.raw[: n.value], "ptx"
        print("compiled for %s -> %s (%d bytes)" % (arch, out_kind, len(payload)))
        break
    if payload is None:
        print("FATAL: NVRTC could not compile the kernel for this device", file=sys.stderr)
        return 3

    out = os.path.join(ART_DIR, "spin_sm%s.%s" % (cc, out_kind))
    with open(out, "wb") as f:
        f.write(payload)

    meta = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device_name": name,
        "compute_capability": "%d.%d" % (major, minor),
        "artifact": os.path.basename(out),
        "artifact_kind": out_kind,
        "artifact_bytes": len(payload),
        "artifact_sha256": hashlib.sha256(payload).hexdigest(),
        "nvrtc_path": nvrtc_path,
        "nvrtc_version": "%d.%d" % (ver_major.value, ver_minor.value),
        "torch": torch.__version__,
        "source_sha256": hashlib.sha256(KERNEL_SRC.encode()).hexdigest(),
    }
    with open(os.path.join(ART_DIR, "spin_sm%s.meta.json" % cc), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print("WROTE %s" % out)

    # Smoke-test: load it back and run one short spin so a broken artifact is
    # caught at build time rather than in the middle of a trial.
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "workload"))
    from spin import SpinKernel  # noqa

    sk = SpinKernel(out)
    hz = sk.calibrate()
    print("calibrated SM clock: %.3f GHz" % (hz / 1e9))
    t0 = time.time()
    sk.spin_seconds(1.0, sync=True)
    print("1.0s spin measured %.3fs -> artifact OK" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())

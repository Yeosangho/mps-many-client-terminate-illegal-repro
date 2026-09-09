"""
Loader for the pre-built long-running spin kernel.

The cubin/ptx is compiled once by build/build_kernel.py and reused by every
run; here we only cuModuleLoadData() it into the PyTorch primary context and
launch it via the CUDA driver API.  No compilation happens at run time.
"""
import ctypes
import os
import time

CUDA_SUCCESS = 0


class _Drv:
    _lib = None

    @classmethod
    def lib(cls):
        if cls._lib is None:
            cls._lib = ctypes.CDLL("libcuda.so.1")
        return cls._lib


def _chk(res, what):
    if res != CUDA_SUCCESS:
        errstr = ctypes.c_char_p()
        try:
            _Drv.lib().cuGetErrorString(res, ctypes.byref(errstr))
            detail = errstr.value.decode() if errstr.value else ""
        except Exception:
            detail = ""
        raise RuntimeError("%s failed: CUresult=%d %s" % (what, res, detail))


class SpinKernel:
    """A single long-running kernel whose duration is set in SM clock cycles."""

    def __init__(self, artifact_path, device_index=0):
        import torch

        # Make sure a primary context exists and is current on this thread.
        torch.cuda.init()
        torch.cuda.set_device(device_index)
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()

        self.torch = torch
        self.artifact_path = artifact_path
        with open(artifact_path, "rb") as f:
            image = f.read()

        drv = _Drv.lib()
        self._module = ctypes.c_void_p()
        _chk(drv.cuModuleLoadData(ctypes.byref(self._module), image), "cuModuleLoadData")
        self._func = ctypes.c_void_p()
        _chk(drv.cuModuleGetFunction(ctypes.byref(self._func), self._module, b"spin_kernel"),
             "cuModuleGetFunction")

        props = torch.cuda.get_device_properties(device_index)
        self.sm_count = props.multi_processor_count
        # Observable side effects, kept in device memory owned by torch so the
        # allocator (and therefore the block memory accounting) sees them.
        self._flag = torch.zeros(1, dtype=torch.int32, device="cuda")
        self._abort = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.cycles_per_sec = None

    # -- launching -------------------------------------------------------
    def launch(self, cycles, blocks=None, threads=128, stream=None):
        drv = _Drv.lib()
        torch = self.torch
        if blocks is None:
            # Enough blocks to cover every SM the container can reach; the
            # TPC mask decides how many actually get work.
            blocks = max(1, self.sm_count)
        if stream is None:
            stream = torch.cuda.current_stream().cuda_stream

        c_cycles = ctypes.c_ulonglong(int(cycles))
        c_flag = ctypes.c_void_p(self._flag.data_ptr())
        c_abort = ctypes.c_void_p(self._abort.data_ptr())
        args = (ctypes.c_void_p * 3)(
            ctypes.cast(ctypes.pointer(c_cycles), ctypes.c_void_p),
            ctypes.cast(ctypes.pointer(c_flag), ctypes.c_void_p),
            ctypes.cast(ctypes.pointer(c_abort), ctypes.c_void_p),
        )
        _chk(drv.cuLaunchKernel(
            self._func,
            ctypes.c_uint(blocks), ctypes.c_uint(1), ctypes.c_uint(1),
            ctypes.c_uint(threads), ctypes.c_uint(1), ctypes.c_uint(1),
            ctypes.c_uint(0),
            ctypes.c_void_p(stream),
            args, None), "cuLaunchKernel")

    # -- calibration -----------------------------------------------------
    def calibrate(self, probe_cycles=200_000_000):
        """Return measured SM clock cycles per wall-clock second."""
        # warm up (module load / first launch cost)
        self.launch(1_000_000)
        self.torch.cuda.synchronize()
        t0 = time.time()
        self.launch(probe_cycles)
        self.torch.cuda.synchronize()
        dt = time.time() - t0
        self.cycles_per_sec = probe_cycles / max(dt, 1e-6)
        return self.cycles_per_sec

    def spin_seconds(self, seconds, blocks=None, threads=128, sync=False):
        if self.cycles_per_sec is None:
            self.calibrate()
        self.launch(int(self.cycles_per_sec * seconds), blocks=blocks, threads=threads)
        if sync:
            self.torch.cuda.synchronize()

    @property
    def completed_spins(self):
        return int(self._flag.item())


def default_artifact(art_dir="/artifacts"):
    import torch

    major, minor = torch.cuda.get_device_capability(0)
    cc = "%d%d" % (major, minor)
    for kind in ("cubin", "ptx"):
        p = os.path.join(art_dir, "spin_sm%s.%s" % (cc, kind))
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "no prebuilt spin kernel for sm_%s in %s -- run build/build_kernel.py once" % (cc, art_dir))

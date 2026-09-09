#!/usr/bin/env python3
"""
VICTIM: a plain ResNet152 training loop.

It is the innocent co-tenant.  Its only job is to keep submitting work to the
shared MPS server and to shout precisely when that stops working, so we can
tell "the neighbour survived" from "the neighbour was poisoned".

Exit codes
  0   asked to stop (SIGTERM/SIGINT) while healthy
  42  POISONED  - a CUDA/cuBLAS/cuDNN fault appeared mid-training
"""
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import maybe_start_graph_assert
from common import (cuda_init_with_retry, ev, is_cuda_fault,
                    checksum_reference, checksum_verify)  # noqa: E402

BATCH = int(os.environ.get("BATCH", "32"))
HEARTBEAT_EVERY = int(os.environ.get("HEARTBEAT_EVERY", "10"))
IMG = int(os.environ.get("IMG", "224"))

_stop = {"v": False}


def _on_sig(signum, frame):
    ev("signal", signum=int(signum))
    _stop["v"] = True


def main():
    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, _on_sig)

    import torch
    import torch.nn as nn
    import torchvision

    ev("boot", torch=torch.__version__, tv=torchvision.__version__, pid=os.getpid())

    CUDNN_BENCHMARK = os.environ.get("CUDNN_BENCHMARK", "1") == "1"
    torch.backends.cudnn.benchmark = CUDNN_BENCHMARK

    def _phase(name, t_prev):
        now = time.time()
        ev("phase", name=name, secs=round(now - t_prev, 2))
        return now

    t = time.time()
    if not cuda_init_with_retry():
        return 44
    t = _phase("cuda_init", t)
    maybe_start_graph_assert()

    # A block-GPU pod sees only its slice of the card; a plain docker container
    # sees all of it, and four tenants each sizing their allocator to the whole
    # B200 is how cuDNN ends up unable to place a workspace.
    frac = float(os.environ.get("MEM_FRACTION", "0"))
    if frac > 0:
        torch.cuda.set_per_process_memory_fraction(frac, 0)

    dev = torch.device("cuda")
    model = torchvision.models.resnet152(weights=None).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    lossfn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")

    x = torch.randn(BATCH, 3, IMG, IMG, device=dev).to(memory_format=torch.channels_last)
    y = torch.randint(0, 1000, (BATCH,), device=dev)
    torch.cuda.synchronize()
    t = _phase("model_build", t)

    # The first iterations carry cuDNN algorithm search and every one-off
    # allocation.  Doing them here means `ready` marks the start of steady
    # state rather than the start of startup.
    warmup = int(os.environ.get("WARMUP_STEPS", "0"))
    for i in range(warmup):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = lossfn(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if i == 0:
            torch.cuda.synchronize()
            t = _phase("first_step", t)
    if warmup:
        torch.cuda.synchronize()
        t = _phase("warmup_rest", t)

    # Reference taken while the GPU is still undisturbed.
    ck = checksum_reference()
    VERIFY_EVERY = int(os.environ.get("VERIFY_EVERY", "50"))

    props = torch.cuda.get_device_properties(0)
    ev("ready",
       checksum_ref=ck,
       gpu=props.name,
       total_mem_mb=int(props.total_memory / 1024 / 1024),
       sm_count=props.multi_processor_count,
       mem_fraction=frac,
       warmup_steps=warmup,
       cudnn_benchmark=CUDNN_BENCHMARK,
       batch=BATCH)

    step = 0
    t_start = time.time()
    last_hb = t_start
    while not _stop["v"]:
        try:
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
                loss = lossfn(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
            if VERIFY_EVERY and step % VERIFY_EVERY == 0:
                ok, ref, got = checksum_verify()
                if ok is False:
                    ev("CORRUPT", step=step, ref=ref, got=got,
                       uptime=round(time.time() - t_start, 2))
                else:
                    ev("verify", step=step, ok=True)
            if step % HEARTBEAT_EVERY == 0:
                # .item() forces a sync -> a poisoned context shows up here.
                lv = float(loss.detach().item())
                now = time.time()
                ev("step",
                   step=step,
                   loss=round(lv, 4),
                   sps=round(HEARTBEAT_EVERY / max(now - last_hb, 1e-9), 2),
                   uptime=round(now - t_start, 2))
                last_hb = now
        except Exception as e:  # noqa: BLE001
            kind = "POISONED" if is_cuda_fault(e) else "ERROR"
            ev(kind, step=step, exc=type(e).__name__, msg=str(e)[:800],
               uptime=round(time.time() - t_start, 2))
            # Do not try to clean up: a second CUDA call on a poisoned context
            # can hang and would mask the timing we are trying to measure.
            os._exit(42 if kind == "POISONED" else 43)

    ev("stopping", step=step, uptime=round(time.time() - t_start, 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

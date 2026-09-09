# `terminate_client` on a busy MPS server damages clients it never touched
<!-- REPRO-CMD -->
## Reproduce

```bash
# Runs on the HOST (it drives docker itself). Needs docker + an idle GPU.
python3 scripts/run_probe.py --mode illegal --arm ps_term \
    --gpu-uuid <GPU-UUID> --kill 6
```

Get the UUID with `nvidia-smi --query-gpu=index,uuid --format=csv`.
Swap `--arm term` to issue `terminate_client` without the preceding `ps`;
both arms reproduce.

18 MPS clients come up (2 offender containers x 8 queuefill workers, plus 2
ResNet neighbours). Six clients **from a single offender container** are then
reclaimed, one at a time, leaving the other offender container and both ResNet
tenants untouched. The line that matters:

```
RESULT ... live_neighbors=1 live_with_error=1 live_illegal=1
  round1 dkill_o2: CUDA error: an illegal memory access was encountered
```

`live_*` counts **only neighbours that were still making progress when the
reclaim fired** — measured five seconds before, in the same cell. A cell with
`live_neighbors=0` is void, not negative.


Terminating a few CUDA MPS clients is documented, supported and — with a handful
of clients — harmless. This repo asks a different question:

> **Does `terminate_client` stay harmless when the MPS server has *many* clients?**

### Choosing the tenant is most of the experiment

The tenants are `queuefill`: each keeps `QUEUE_DEPTH` long kernels **outstanding
at all times**, so a client is always holding a resident queue at the moment it
is reclaimed. That is deliberate, and it is the difference between a test and a
tautology.

An earlier version of this repo used a plain matmul tenant whose kernels retire.
Running 40 of those and finding no damage measures nothing — **4 would give the
same answer**, because a plain tenant is already known not to propagate under any
reclaim. The established boundary is:

| offender | `terminate_client` | `SIGKILL` / `os._exit(0)` |
|---|---|---|
| plain, retiring kernels | harmless | — |
| **`queuefill`** (resident queue) | harmless | **siblings get `cudaErrorIllegalAddress`** |
| **counted-barrier wedge** | **co-tenants get it with only 4 clients** | — |

`queuefill` therefore sits *at* the boundary: safe under the graceful reclaim,
lethal under an abrupt one. That is the only workload in which "does the client
count change anything?" is a real question — and the counted-barrier row is the
reason to suspect the answer is no, since **4 clients suffice there**.

## What it reports

```
pre-fire dkill_o2: prog=4 (+1 in 5s)        <- neighbour was running when we fired
pre-fire dkill_v1: prog=1330 (+0 in 5s)     <- this one was not; it cannot testify

RESULT dkcli mode=illegal arm=ps_term clients=18 killed=6 ok=6
       live_neighbors=1 live_with_error=1 live_illegal=1
       untouched_with_error=3 illegal=3
       worst_term_s=0.117 worst_ps_s=0.135 clients_after=0
```

Two counters, and the difference between them is the whole point:

| field | meaning |
|---|---|
| `untouched_with_error` / `illegal` | every container nobody terminated |
| **`live_with_error` / `live_illegal`** | **only those that were still making progress when the reclaim fired** |

Score the `live_*` pair. A neighbour that had already stopped is neither a
survivor nor a casualty — that cell is void. `live_neighbors=0` means no
untouched neighbour was running, so the cell says nothing either way.

`ok=` counts responses of exactly `0`. `clients_after=0` with `killed=6` out of
18 is the usual shape: reclaiming a few collapses the whole client table.

## Two rules this harness exists to enforce

Both were learned by getting them wrong.

**1. Terminates must be issued sequentially, and every response checked.**
The MPS control daemon accepts **one connection at a time**. Concurrent callers
are not queued — they are refused with
`Cannot send command to MPS control daemon process`, in about a millisecond,
having done nothing. A concurrent version of this experiment produced a fast,
clean-looking series in which **31 of 32 terminates had been refused**, and it
was very nearly written up as "terminate_client is fast and safe at scale". The
runner counts refusals and VOIDs the cell.

**2. Only clients that were never terminated may be scored.**
A terminated client's own `cudaErrorMpsClientTerminated` is its *intended*
ending. Counting it as damage turns a clean run into an apparent blast radius —
a mistake that produced a spurious propagation count in a sibling harness.

## Reading an ending

Three endings share exit code 43 and differ only in the message:

| message | meaning |
|---|---|
| `cudaErrorMpsClientTerminated` | the intended reclaim of that client |
| `cudaErrorMpsRpcFailure` | the MPS server went away — often the harness tearing down its own daemon |
| `cudaErrorIllegalAddress` | genuine propagation into a client nobody touched |

Never conclude from the exit code alone.

## Measurement notes

* Client progress is counted **device-side** and read without synchronising. A
  host-side counter cannot tell "the kernel ran" from "the launch was queued and
  never dispatched", so a stalled client reports as healthy.
* A poisoned context does not necessarily raise on the next launch, so the
  tenant asks explicitly — but **the probe must not be what kills the tenant**.
  `cudaPeekAtLastError` is not exposed on every torch build (missing on 26.06);
  using it unguarded killed all 36 bystanders in one run and was very nearly
  read as propagation. `torch.cuda.current_stream().query()` is non-blocking,
  present everywhere, and raises on a sticky error.
* **`queuefill` must never drain.** The tenant tracks outstanding work with CUDA
  events and `query()`; a `synchronize()` in that loop would empty the queue and
  destroy the very state under test.
* **The readiness gate matters.** MPS has a client limit (48 on Volta+ by
  default). A client that never attached is not a bystander that survived, so a
  cell where fewer clients attached than requested is VOID, not a clean result.
  If you are probing near the limit, that gate is the difference between a
  finding and an artefact.

## Status

**An earlier version of this harness never launched the ResNet victims at all.**
It referenced them in its readiness gate, its cleanup and its scoring, but no
code started them. Every "no damage at 40 clients" cell from that version
therefore ran with queuefill neighbours only: the tolerant side, which this
README already says will report no damage almost regardless. Those cells
measured nothing, and their results are void.

The harness here starts the victims first and waits until they are training
before anything else comes up.

What is established at the time of writing, with the fixed harness and a readiness
gate applied to **every untouched neighbour**:

| platform | population | reclaims | gated cells | neighbours damaged |
|---|---|---|---|---|
| Kubernetes | 18 clients + 2 ResNet | 6 then 10 | 2 | **2 / 2** |
| docker | 18 clients + 2 ResNet | 6 | 9 | **5 / 9 illegal, 6 / 9 any error** |
| docker | 40 clients + 2 ResNet | 40, sequential | 14 | 0 damaged, but **2 hung** — see `mps-terminate-hang-40client-repro` |

Both docker arms reproduce — `ps` before each `terminate_client`, and
`terminate_client` alone — so issuing `ps` is not an ingredient.

### Gate every untouched neighbour, not just the sensitive one

**This is the single easiest way to get a wrong answer here, and it fails in both
directions.**

Without a gate, a neighbour that had already stopped for unrelated reasons gets
counted as propagation. Four docker cells were nearly written up that way before
anyone checked that the ResNet victims had not completed a single training step.

But gating *only on the ResNet victims* is just as wrong. Under bare docker those
victims stop as soon as the offenders attach (see
`mps-neighbor-stall-on-attach-repro`), so every cell looks void — while an untouched
**offender** container, still making progress, is taking
`cudaErrorIllegalAddress` in the very same cell. Nine cells were discarded as "no
data" on exactly that reasoning; when the untouched offenders were added to the gate,
five of them were reproductions.

So: read the progress of each untouched container in a window before firing, and
score the ones that were moving. Progress events differ by workload — the ResNet
tenant emits `"ev": "step"`, the queuefill tenant emits `"ev": "qfill"` with a
`done` counter.

### The failure signature varies

`illegal memory access`, `illegal instruction`, and `CUDNN_STATUS_EXECUTION_FAILED`
have all appeared, sometimes on two neighbours of the same cell. Counting only
`illegal` understates the blast radius; count any CUDA-level error on a neighbour
that was running.

Repetition is not optional: this family of effects reproduces at rates around
1-in-3 to 1-in-10, so a single clean cell is not a negative result, it is an
underpowered one.

## Related

* `mps-counted-barrier-preempt-repro` — a *different* route to the same error:
  terminating a tenant parked on a counted CTA barrier kills every co-tenant,
  with only 4 clients on the GPU. There the ingredient is the barrier; here the
  question is whether population alone suffices.
* `mps-victim-stall-repro` — `SIGKILL` and a bare `os._exit(0)` on a tenant
  holding a resident queue poison its siblings.
* `mps-terminate-delay-repro` — why a terminate can appear to take tens of
  seconds, and why that matters: the caller escalates to `SIGKILL`.

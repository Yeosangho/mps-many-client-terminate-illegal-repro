#!/usr/bin/env python3
"""A CLI-only docker harness: two defects, two reclaim procedures.

    --mode delay    reclaim delay: does terminate_client exceed its budget (grace)?
    --mode illegal  neighbour poisoning: does an untouched neighbour take an
                    illegal memory access?

    --arm ps_term   per target: ps, then terminate_client, issued in sequence
    --arm term      terminate_client only (no ps)

The two modes differ **only in shape**. The reclaim procedure is the same code
(mpscore.reclaim) in both.

The shapes differ because the two defects need different conditions:

    delay   : NPROC=1, QUEUE_DEPTH=32, ~4 clients, 2 reclaims.
              The shape in which the delay reproduced under Kubernetes.
    illegal : NPROC=8, QUEUE_DEPTH=8, 18 clients, 6 reclaims then 10 more.
              The shape in which the poisoning reproduced; with few clients the
              defect does not appear at all.

Nothing but terminate_client touches the tenants. No signals are sent, no
container is killed, no orchestration agent is involved. What this docker port
cannot carry over is the vGPU library's LD_PRELOAD, its TPC MASK_VALUE
(PARTITION_PERCENTAGE=25), and the preStop/term_all signal path -- so a
reproduction here rules those three out as causes.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fleet import Fleet                                            # noqa: E402
from mpscore import MpsCli, classify, reclaim                       # noqa: E402

SHAPES = {
    # mode:      noff nvic nproc qdepth kill  observe cleanup_observe
    "delay":   dict(noff=2, nvic=2, nproc=1, qdepth="32", kill=None,
                    observe=45, cleanup_observe=0),
    "illegal": dict(noff=2, nvic=2, nproc=8, qdepth="8", kill=6,
                    observe=60, cleanup_observe=30),
}


def say(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["delay", "illegal"], required=True)
    ap.add_argument("--arm", choices=["ps_term", "term"], required=True)
    ap.add_argument("--gpu-uuid", default=os.environ.get("GPU_UUID", ""))
    ap.add_argument("--seq", type=int, default=1)
    ap.add_argument("--root", default=os.environ.get(
        "DK_ROOT", os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), "run")),
        help="where the MPS pipe directory and logs live; mounted into the containers")
    ap.add_argument("--steady", type=int, default=30)
    ap.add_argument("--boot", type=int, default=600)
    ap.add_argument("--grace", type=float, default=30.0,
                    help="per-call CLI budget, matched to the pod/container "
                         "termination grace period -- past it, production "
                         "escalates to SIGKILL")
    ap.add_argument("--ksec", default=os.environ.get("KERNEL_SEC", "5"))
    ap.add_argument("--batch", default=os.environ.get("BATCH", "32"))
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--reuse-mps", action="store_true",
                    help="keep the MPS daemon alive between reps. Needed to see "
                         "cumulative effects -- with a fresh daemon each cell "
                         "starts clean and nothing can accumulate")
    # Shape overrides.
    #
    # A CLI hang was caught on the **40th** of 40 sequential reclaims, blocked for
    # five hours (worker thread in __skb_wait_for_more_packets), roughly 1 in 6
    # trials; c1-c39 had returned normally in 19-394 ms. So that defect needs the
    # reclaim count to build up, and the default shapes (2 or 6 reclaims) have no
    # power to detect it.
    ap.add_argument("--noff", type=int)
    ap.add_argument("--nvic", type=int)
    ap.add_argument("--nproc", type=int)
    ap.add_argument("--kill", type=int)
    ap.add_argument("--qdepth")
    ap.add_argument("--off-libvgpu",
                    help="block_gpu config directory to layer onto the offenders; "
                         "its TPC mask caps their SM footprint and prevents the "
                         "neighbour stall")
    ap.add_argument("--off-pct", default="25",
                    help="CUDA_MPS_ACTIVE_THREAD_PERCENTAGE for the offenders, "
                         "approximating the Kubernetes TPC mask (25%%)")
    ap.add_argument("--min-step", type=int, default=200,
                    help="minimum step the victims must reach before firing")
    ap.add_argument("--vready", type=int, default=300,
                    help="how long to wait for victim readiness, in seconds")
    a = ap.parse_args()

    if not a.gpu_uuid:
        say("need a GPU UUID (--gpu-uuid, or the GPU_UUID environment variable)")
        return 2

    shape = dict(SHAPES[a.mode])
    for k in ("noff", "nvic", "nproc", "kill", "qdepth"):
        v = getattr(a, k)
        if v is not None:
            shape[k] = v
    prefix = "dk%s" % a.mode[:3]
    root = os.path.join(a.root, a.mode)
    os.makedirs(root, exist_ok=True)
    csv_path = os.path.join(root, "mpsctl_calls.csv")

    fleet = Fleet(prefix, a.gpu_uuid, root, say, off_pct=a.off_pct,
                  off_libvgpu=a.off_libvgpu)
    cli = MpsCli("dk_" + prefix, container=fleet.mps, csv_path=csv_path,
                 budget=a.grace, run_id="%s_%d" % (a.arm, a.seq))

    say("mode=%s arm=%s seq=%d shape=%s" % (a.mode, a.arm, a.seq, shape))
    fleet.clean(keep_mps=a.reuse_mps)
    fleet.start_mps(reuse=a.reuse_mps)

    # 1) Bring the victims up alone and wait until they are training.
    fleet.start_victims(shape["nvic"], a.batch)
    if not fleet.wait_victims_ready(min_step=a.min_step, timeout=a.vready):
        say("VOID: victims never reached step>=%d even with no offenders present "
            "-- this is a harness problem, not a result" % a.min_step)
        if not a.keep:
            fleet.clean(keep_mps=a.reuse_mps)
        return 2
    say("--- victim state before the offenders start ---")
    fleet.victim_progress("pre-offender")

    # 2) Now attach the offenders. Whether the victims stop here is the key
    #    observation: MPS time-slices past its SM budget, so plain contention
    #    slows a tenant but does not take it to zero. Zero is a dispatch stall,
    #    not contention.
    fleet.start_offenders(shape["noff"], shape["nproc"], shape["qdepth"], a.ksec)

    want = shape["noff"] * shape["nproc"] + shape["nvic"]
    pids = fleet.wait_clients(cli, want, boot=a.boot)
    say("--- victim state after the offenders started (before any reclaim) ---")
    live_after_off = fleet.victim_progress("post-offender")
    if not live_after_off:
        say("WARNING: the victims stopped before anything was reclaimed. "
            "This cell cannot testify to propagation -- attaching the offenders "
            "alone was enough to stall them.")
    if len(pids) < want:
        say("VOID: %d clients < %d expected -- a client that never attached is "
            "not a bystander that survived" % (len(pids), want))
        if not a.keep:
            fleet.clean(keep_mps=a.reuse_mps)
        return 2

    owner = fleet.owner_map(pids)
    off_pids = [p for p in pids if owner.get(p) in fleet.off]
    vic_pids = [p for p in pids if owner.get(p) in fleet.vic]
    say("ownership: offender %d / victim %d / unmapped %d"
        % (len(off_pids), len(vic_pids), len(pids) - len(off_pids) - len(vic_pids)))

    kill = shape["kill"] or len(off_pids)
    if len(off_pids) < kill:
        say("VOID: %d offender clients < %d to reclaim" % (len(off_pids), kill))
        if not a.keep:
            fleet.clean(keep_mps=a.reuse_mps)
        return 2

    say("steady state for %ds" % a.steady)
    time.sleep(a.steady)
    srv = cli.server()
    if srv is None:
        # Firing without a server pid makes the CLI read None as 0 and print
        # "Server 0 not found": nothing is issued, yet it records as if it were.
        say("VOID: get_server_list returned no server pid -- nothing issued")
        if not a.keep:
            fleet.clean(keep_mps=a.reuse_mps)
        return 2
    say("server=%s" % srv)

    # Pick the reclaim targets from **one offender container only**.
    #
    # Slicing off_pids[:kill] lets the targets span both offender containers, and
    # then no untouched offender is left. The only remaining neighbours are the
    # ResNet victims -- which stall as soon as the offenders attach -- so the
    # cell voids itself (measured: reps 2-5 all came out "0/2 progressing", and
    # an earlier campaign's 7/12 gate-pass rate had the same cause).
    #
    # Taking them from a single container always leaves the other one as a
    # neighbour that is still running.
    by_ctr = {}
    for p_ in off_pids:
        by_ctr.setdefault(owner[p_], []).append(p_)
    src = max(by_ctr, key=lambda c: len(by_ctr[c])) if by_ctr else None
    if src and len(by_ctr[src]) >= kill:
        targets = by_ctr[src][:kill]
        say("reclaiming %d targets from %s only (leaving %s as neighbours)"
            % (kill, src, ", ".join(c for c in by_ctr if c != src) or "none"))
    else:
        targets = off_pids[:kill]
        say("WARNING: could not take %d targets from one container, so they span "
            "several -- no untouched offender may be left" % kill)
    touched = {owner[p] for p in targets}

    # Check **every** untouched container for progress. Watching only the victims
    # loses the case where an untouched offender was poisoned while running.
    untouched_all = [n for n in fleet.off + fleet.vic if n not in touched]
    say("--- progress of untouched containers, just before firing ---")
    live_map = fleet.progress(untouched_all, "pre-fire")
    live = all(live_map.get(v, False) for v in fleet.vic if v in untouched_all)
    say("  pre-fire summary: %d of %d progressing (all victims progressing=%d)"
        % (sum(1 for x in live_map.values() if x), len(live_map), int(live)))

    r1 = reclaim(cli, targets, a.arm, "round1", say, srv=srv)
    if r1["ok"] == 0:
        say("VOID: none of the %d targets returned rc=0" % len(targets))

    say("observing for %ds" % shape["observe"])
    time.sleep(shape["observe"])
    say("--- untouched containers ---")
    nerr, nill = fleet.score(untouched_all, "round1")
    # Score the neighbours that were progressing separately: that set is the
    # evidence for propagation.
    live_names = [n for n, ok in live_map.items() if ok]
    lerr, lill = (0, 0)
    if live_names:
        lerr, lill = fleet.score(live_names, "round1 (progressing neighbours only)")
    _, after = cli.ps(label="after")
    over = [c for c in cli.calls
            if c["verb"] == "terminate_client" and c["elapsed_s"] >= a.grace * 0.5]
    say("RESULT dkcli mode=%s arm=%s seq=%d clients=%d killed=%d ok=%d "
        "victim_live_before=%d live_neighbors=%d live_with_error=%d live_illegal=%d "
        "untouched_with_error=%d illegal=%d "
        "worst_term_s=%.3f worst_ps_s=%.3f over_half_grace=%d clients_after=%d"
        % (a.mode, a.arm, a.seq, len(pids), len(targets), r1["ok"], int(live),
           len(live_names), lerr, lill,
           nerr, nill, r1["worst_term_s"], r1["worst_ps_s"], len(over), len(after)))

    # Only the illegal mode runs a second round. The victims are held back and
    # reclaimed last, otherwise a victim's death cannot be separated from its own
    # intended reclaim.
    if a.mode == "illegal":
        _, rest = cli.ps(label="round2")
        oth = [p for p in rest if owner.get(p) not in fleet.vic]
        vic_left = [p for p in rest if owner.get(p) in fleet.vic]
        say("remaining split: %d non-victim / %d victim" % (len(oth), len(vic_left)))
        if oth:
            say("--- victim progress before round 2 ---")
            live2 = fleet.victim_progress("pre-round2")
            r2 = reclaim(cli, oth, a.arm, "round2", say, srv=srv)
            say("observing for %ds" % shape["cleanup_observe"])
            time.sleep(shape["cleanup_observe"])
            aerr, aill = fleet.score(fleet.vic, "round2")
            _, after2 = cli.ps(label="after2")
            say("RESULT_R2 dkcli mode=%s arm=%s seq=%d reclaimed=%d ok=%d "
                "victim_live_before=%d victim_with_error=%d illegal=%d "
                "worst_term_s=%.3f clients_after=%d"
                % (a.mode, a.arm, a.seq, len(oth), r2["ok"], int(live2),
                   aerr, aill, r2["worst_term_s"], len(after2)))
        if vic_left:
            reclaim(cli, vic_left, a.arm, "cleanup(victim)", say, srv=srv)

    # Call summary -- the basis for any delay verdict.
    verbs = {}
    for c in cli.calls:
        verbs.setdefault(c["verb"], []).append(c["elapsed_s"])
    for v in sorted(verbs):
        xs = sorted(verbs[v])
        say("calls %-18s n=%3d median %.4fs max %.4fs"
            % (v, len(xs), xs[len(xs) // 2], xs[-1]))

    if not a.keep:
        say("removing containers")
        fleet.clean(keep_mps=a.reuse_mps)
    return 0


if __name__ == "__main__":
    sys.exit(main())

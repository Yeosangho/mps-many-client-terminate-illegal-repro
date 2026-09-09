#!/usr/bin/env python3
"""MPS CLI 전용 도커 하네스 -- 두 실험군, 두 회수 절차.

    --mode delay    종료 지연: terminate_client 가 예산(grace)을 넘겨 멈추는가
    --mode illegal  이웃 오염: 손대지 않은 이웃이 illegal memory access 로 죽는가

    --arm ps_term   대상마다 ps -> terminate_client 를 순차 발행
    --arm term      terminate_client 만 (ps 없음)

두 모드는 **모양만** 다르다. 회수 절차는 같은 코드(mpsctl.reclaim)를 쓴다.
k8s 쪽 대응은 각각:

    delay   x ps_term/term  <->  trial.py --arm cliterm (CLITERM_PS=1/0)
    illegal x ps_term/term  <->  bca_many.py --via cli_ps / cli

모양이 다른 이유는 두 결함의 재현 조건이 다르기 때문이다.

    delay   : NPROC=1, QUEUE_DEPTH=32, 클라 ~4개, 회수 2건.
              k8s 에서 지연이 재현된 레시피의 모양이다.
    illegal : NPROC=8, QUEUE_DEPTH=8, 클라 18개, 1라운드 6건 + 2라운드 10건.
              k8s 에서 illegal 이 재현된 모양이며, 2라운드에서 터졌다.
              클라가 적으면 이 결함은 애초에 안 나온다.

이 하네스는 회수 경로에 terminate_client 외에 아무것도 넣지 않는다. 신호를 보내지
않고, 컨테이너를 죽이지 않고, agent 도 거치지 않는다. 도커가 옮기지 못하는 것은
libvgpu LD_PRELOAD, TPC MASK_VALUE(PARTITION_PERCENTAGE=25), preStop/term_all 이며,
여기서 재현되면 그 셋은 용의선상에서 빠진다.
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
        help="MPS 파이프 디렉터리와 로그를 둘 곳. 컨테이너에 마운트된다")
    ap.add_argument("--steady", type=int, default=30)
    ap.add_argument("--boot", type=int, default=600)
    ap.add_argument("--grace", type=float, default=30.0,
                    help="CLI 호출 예산. 파드/컨테이너 종료 유예와 맞춘다 -- "
                         "프로덕션에서 이 시간을 넘기면 상위가 SIGKILL 로 올라간다")
    ap.add_argument("--ksec", default=os.environ.get("KERNEL_SEC", "5"))
    ap.add_argument("--batch", default=os.environ.get("BATCH", "32"))
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--reuse-mps", action="store_true",
                    help="rep 사이에 MPS 데몬을 유지한다. 누적 효과를 "
                         "보려면 필요하다 -- 매번 새 데몬이면 각 셀이 "
                         "깨끗한 상태에서 시작해 누적을 못 잰다")
    # 모양 오버라이드.
    #
    # peer 세션이 도커에서 40 클라 순차 종료의 **40번째**에서 CLI 가 5시간 무한
    # 대기하는 것을 잡았다(worker thread = __skb_wait_for_more_packets, 약 1/6).
    # c1~c39 는 19~394ms 로 정상이었다. 즉 이 결함은 회수 건수가 쌓여야 나온다.
    # 기본 모양(회수 2건/6건)으로는 검정력이 없다.
    ap.add_argument("--noff", type=int)
    ap.add_argument("--nvic", type=int)
    ap.add_argument("--nproc", type=int)
    ap.add_argument("--kill", type=int)
    ap.add_argument("--qdepth")
    ap.add_argument("--off-libvgpu",
                    help="오펜더에 얹을 block_gpu 설정 디렉터리. TPC 마스크로 "
                         "오펜더의 SM 점유를 가둬 이웃 정지를 막는다")
    ap.add_argument("--off-pct", default="25",
                    help="오펜더의 CUDA_MPS_ACTIVE_THREAD_PERCENTAGE. "
                         "k8s 의 TPC 마스크(25%%)를 근사한다. 없으면 "
                         "victim 이 굶어 첫 스텝도 못 찍는다")
    ap.add_argument("--min-step", type=int, default=200,
                    help="발사 전 victim 이 도달해야 하는 최소 step")
    ap.add_argument("--vready", type=int, default=300,
                    help="victim 준비 대기 상한(초)")
    a = ap.parse_args()

    if not a.gpu_uuid:
        say("GPU_UUID 를 주세요 (--gpu-uuid 또는 환경변수)")
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
    # 1) victim 만 먼저 띄우고, 정상 속도에 들 때까지 기다린다.
    fleet.start_victims(shape["nvic"], a.batch)
    if not fleet.wait_victims_ready(min_step=a.min_step, timeout=a.vready):
        say("VOID: 오펜더 없이도 victim 이 step>=%d 에 못 감 -- 하네스 문제"
            % a.min_step)
        if not a.keep:
            fleet.clean(keep_mps=a.reuse_mps)
        return 2
    say("--- 오펜더 기동 전 victim 상태 ---")
    fleet.victim_progress("오펜더전")

    # 2) 그 다음 오펜더를 붙인다. 여기서 victim 이 멈추는지가 핵심 관측이다.
    #    MPS 는 SM 할당량 초과 시 시분할로 전환되므로, 단순 경쟁이라면 느려질
    #    뿐 0 이 되지는 않는다. 0 이 되면 그것은 경쟁이 아니라 디스패치 정지다.
    fleet.start_offenders(shape["noff"], shape["nproc"], shape["qdepth"], a.ksec)

    want = shape["noff"] * shape["nproc"] + shape["nvic"]
    pids = fleet.wait_clients(cli, want, boot=a.boot)
    say("--- 오펜더 기동 후 victim 상태 (회수 전) ---")
    live_after_off = fleet.victim_progress("오펜더후")
    if not live_after_off:
        say("주의: 회수를 하기도 전에 victim 이 멈췄다. "
            "이 셀의 전파 판정은 성립하지 않는다 -- 오펜더 기동만으로 정지한 것이다.")
    if len(pids) < want:
        say("VOID: 클라 %d개 < 기대 %d -- 붙지 않은 클라는 살아남은 방관자가 아니다"
            % (len(pids), want))
        if not a.keep:
            fleet.clean(keep_mps=a.reuse_mps)
        return 2

    owner = fleet.owner_map(pids)
    off_pids = [p for p in pids if owner.get(p) in fleet.off]
    vic_pids = [p for p in pids if owner.get(p) in fleet.vic]
    say("소유: 오펜더 %d / victim %d / 미분류 %d"
        % (len(off_pids), len(vic_pids), len(pids) - len(off_pids) - len(vic_pids)))

    kill = shape["kill"] or len(off_pids)
    if len(off_pids) < kill:
        say("VOID: 오펜더 클라 %d개 < 회수 대상 %d" % (len(off_pids), kill))
        if not a.keep:
            fleet.clean(keep_mps=a.reuse_mps)
        return 2

    say("정상 상태 %ds" % a.steady)
    time.sleep(a.steady)
    srv = cli.server()
    if srv is None:
        # srv 없이 쏘면 CLI 가 None 을 0 으로 읽어 "Server 0 not found" 를 찍는다.
        # 발행은 한 건도 안 됐는데 성공처럼 기록되므로 여기서 멈춘다.
        say("VOID: get_server_list 가 서버 pid 를 못 줌 -- 발행 0건")
        if not a.keep:
            fleet.clean(keep_mps=a.reuse_mps)
        return 2
    say("server=%s" % srv)

    # 회수 대상은 **한 오펜더 컨테이너 안에서만** 고른다.
    #
    # 그냥 off_pids[:kill] 로 자르면 대상이 두 오펜더 컨테이너에 걸치고, 그러면
    # 손대지 않은 오펜더가 하나도 안 남는다. 남는 이웃이 ResNet victim 뿐인데
    # victim 은 오펜더 부착만으로 멈추므로 셀이 자동으로 VOID 가 된다
    # (실측: rep2~5 가 전부 "진행 중 0/2" 로 무효였고, 이전 캠페인의 게이트
    # 통과율 7/12 도 같은 이유였다).
    #
    # 한 컨테이너에서만 뽑으면 나머지 오펜더 컨테이너가 항상 이웃으로 남는다.
    by_ctr = {}
    for p_ in off_pids:
        by_ctr.setdefault(owner[p_], []).append(p_)
    src = max(by_ctr, key=lambda c: len(by_ctr[c])) if by_ctr else None
    if src and len(by_ctr[src]) >= kill:
        targets = by_ctr[src][:kill]
        say("회수 대상은 %s 에서만 %d개 (이웃으로 %s 남김)"
            % (src, kill, ", ".join(c for c in by_ctr if c != src) or "없음"))
    else:
        targets = off_pids[:kill]
        say("주의: 한 컨테이너에서 %d개를 못 채워 여러 컨테이너에 걸침 "
            "-- 손대지 않은 오펜더가 안 남을 수 있음" % kill)
    touched = {owner[p] for p in targets}
    # 손대지 않은 컨테이너 **전부**의 진행을 본다. victim 만 보면, victim 이
    # 정지한 셀에서 손대지 않은 오펜더가 illegal 을 맞은 사실을 놓친다.
    untouched_all = [n for n in fleet.off + fleet.vic if n not in touched]
    say("--- 발사 전 손대지 않은 컨테이너 진행 확인 ---")
    live_map = fleet.progress(untouched_all, "발사전")
    live = all(live_map.get(v, False) for v in fleet.vic if v in untouched_all)
    live_any = any(live_map.values())
    say("  발사전 요약: 진행 중 %d / %d  (victim 전부 진행=%d)"
        % (sum(1 for x in live_map.values() if x), len(live_map), int(live)))

    r1 = reclaim(cli, targets, a.arm, "1라운드", say, srv=srv)
    if r1["ok"] == 0:
        say("VOID 대상 %d개 중 성공(rc=0) 0건" % len(targets))

    say("관측 %ds" % shape["observe"])
    time.sleep(shape["observe"])
    say("--- 손대지 않은 컨테이너 ---")
    nerr, nill = fleet.score(untouched_all, "1라운드")
    # 진행 중이던 이웃만 따로 센다. 이것이 전파의 증거가 되는 집합이다.
    live_names = [n for n, ok in live_map.items() if ok]
    lerr, lill = (0, 0)
    if live_names:
        lerr, lill = fleet.score(live_names, "1라운드(진행중이던 이웃만)")
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

    # illegal 모드만 2라운드를 돈다. victim 을 빼고 남은 것만 회수해야
    # victim 사망이 전파인지 자기 회수인지 갈린다.
    if a.mode == "illegal":
        _, rest = cli.ps(label="round2")
        oth = [p for p in rest if owner.get(p) not in fleet.vic]
        vic_left = [p for p in rest if owner.get(p) in fleet.vic]
        say("정리 분할: 비-victim %d개 / victim %d개" % (len(oth), len(vic_left)))
        if oth:
            say("--- 2라운드 발사 전 victim 진행 확인 ---")
            live2 = fleet.victim_progress("2라운드전")
            r2 = reclaim(cli, oth, a.arm, "2라운드", say, srv=srv)
            say("관측 %ds" % shape["cleanup_observe"])
            time.sleep(shape["cleanup_observe"])
            aerr, aill = fleet.score(fleet.vic, "2라운드")
            _, after2 = cli.ps(label="after2")
            say("RESULT_R2 dkcli mode=%s arm=%s seq=%d reclaimed=%d ok=%d "
                "victim_live_before=%d victim_with_error=%d illegal=%d "
                "worst_term_s=%.3f clients_after=%d"
                % (a.mode, a.arm, a.seq, len(oth), r2["ok"], int(live2),
                   aerr, aill, r2["worst_term_s"], len(after2)))
        if vic_left:
            reclaim(cli, vic_left, a.arm, "정리(victim)", say, srv=srv)

    # 호출 요약 -- 지연 판정의 근거
    verbs = {}
    for c in cli.calls:
        verbs.setdefault(c["verb"], []).append(c["elapsed_s"])
    for v in sorted(verbs):
        xs = sorted(verbs[v])
        say("호출 %-18s n=%3d 중앙 %.4fs 최대 %.4fs"
            % (v, len(xs), xs[len(xs) // 2], xs[-1]))

    if not a.keep:
        say("컨테이너 정리")
        fleet.clean(keep_mps=a.reuse_mps)
    return 0


if __name__ == "__main__":
    sys.exit(main())

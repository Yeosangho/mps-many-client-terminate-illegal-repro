#!/usr/bin/env python3
"""MPS control CLI 코어 -- 두 실험군, 두 플랫폼이 **이 파일 하나**를 공유한다.

지연군(trial.py)과 illegal군(bca_many.py), k8s 와 도커가 각자 CLI 를 부르던 것을
여기로 모았다. 이유는 단순하다: 같은 절차를 두 번 구현하면 한쪽에서 고친 계측
함정이 다른 쪽에 안 들어간다. 실제로 발행 인터리빙이 두 하네스에서 달랐다.

**부르는 방식은 여기서 하나로 고정된다.**

    발행 = 단일 시퀀스. 대상마다  ps -> terminate_client  를 순서대로.
           한 대상의 (ps, terminate) 쌍 사이에 다른 호출이 끼지 않는다.
           시퀀스 전체를 락 한 번으로 감싸므로, 동시에 발화한 스레드가 있어도
           파드/컨테이너 단위로 줄을 선다.

**바뀌지 않는 것**: 무엇을 에러로 볼지, 어떤 모양으로 돌지는 실험군마다 다르다.
그건 호출자(trial.py / bca_many.py / run_probe.py)가 정한다.

여기 박아 둔 규칙은 전부 한 번씩 틀려 본 것들이다.

1. **순차 호출.** MPS control 데몬은 커넥션을 하나만 받는다. 동시에 부르면 큐에
   서는 게 아니라 "Cannot send command" 로 거절당하고, 호출자는 아무 일도 안 한 채
   1ms 만에 돌아온다. 락은 파일 락이다 -- 데몬은 노드 전체 공유라 스레드 락으로는
   다른 프로세스를 못 막는다(실측: 캠페인 2개가 싸워 거절 7건).
2. **응답 분류.** 성공은 "0" 뿐. 201=CUDA_ERROR_INVALID_CONTEXT,
   "Invalid process <pid>!", "Server 0 not found", "Cannot send command" 는
   전부 회수가 아니다(실측: 99건 중 73건만 실제 회수였다).
3. **ps 와 terminate 를 따로, 대기와 실행을 따로 잰다.** 합치면 멈춘 쪽을 지목할 수
   없고, 락 대기를 terminate 소요시간으로 오독한다.
4. **srv 를 못 얻으면 발행하지 않는다.** 그대로 쏘면 CLI 가 None 을 0 으로 읽어
   "Server 0 not found" 를 찍는데, 발행은 한 건도 안 됐으면서 기록엔 남는다.
"""
import contextlib
import csv
import errno
import fcntl
import os
import subprocess
import time

LOCK_DIR = os.environ.get("MPS_LOCK_DIR", "/var/lock")


@contextlib.contextmanager
def cli_lock(key, timeout=120.0, poll=0.01):
    """데몬(=GPU) 별 파일 락. 예산 안에 못 잡으면 held=False 로 알린다.

    무한정 기다리지 않는 이유: 락을 쥔 쪽이 멈추면 실험이 통째로 멈춘다.
    못 잡았다는 사실은 반드시 기록에 남긴다 -- 조용히 넘어가면 거절이 다시
    데이터에 섞인다.
    """
    path = os.path.join(LOCK_DIR, "mpsctl_%s.lock" % key)
    fd, held = None, False
    try:
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
        deadline = time.time() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held = True
                break
            except OSError as e:
                if e.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                if time.time() >= deadline:
                    break
                time.sleep(poll)
        yield held
    finally:
        if fd is not None:
            try:
                if held:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


def classify(resp):
    t = str(resp).strip()
    if t == "0":
        return "ok"
    if "Cannot send command" in t:
        return "refused"
    if "Invalid process" in t:
        return "invalid"
    if "not found" in t or "NO_SERVER" in t:
        return "nosrv"
    if t == "<TIMEOUT>":
        return "timeout"
    if t == "<NO_TERM>":
        # arm="ps" 는 terminate 를 아예 발행하지 않는다. 실패가 아니라
        # "쏘지 않았다"이므로 성공/실패 어느 쪽으로도 세면 안 된다.
        return "noop"
    if t == "":
        return "empty"
    if t.isdigit():
        return "rc" + t
    return "other"


def parse_ps(out):
    pids = []
    for ln in out.splitlines()[1:]:
        f = ln.split()
        if f and f[0].isdigit():
            pids.append(int(f[0]))
    return pids


class MpsCli(object):
    """CLI 한 건을 쏘고 재고 기록한다. 전송 방식은 둘 중 하나다.

        pipe_dir=... : 호스트에서 직접 (k8s 블록 MPS)
        container=...: docker exec 로 (도커의 사설 MPS)

    budget 은 파드/컨테이너의 종료 유예(grace)와 맞춘다. 프로덕션에서 의미 있는
    경계가 그것이기 때문이다 -- 그 안에 종료가 안 끝나면 상위가 SIGKILL 로
    올라가고, 상주 큐를 쥔 클라를 SIGKILL 하면 이웃이 오염된다.
    """

    def __init__(self, lock_key, pipe_dir=None, container=None,
                 csv_path=None, budget=30.0, run_id=""):
        assert pipe_dir or container, "pipe_dir 또는 container 가 필요합니다"
        self.lock_key = lock_key
        self.pipe_dir = pipe_dir
        self.container = container
        self.csv_path = csv_path
        self.budget = float(budget)
        self.run_id = run_id
        self.calls = []

    def _run(self, cmd):
        t0 = time.time()
        try:
            if self.container:
                r = subprocess.run(
                    ["docker", "exec", self.container, "sh", "-c",
                     "echo '%s' | nvidia-cuda-mps-control" % cmd],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, timeout=self.budget)
            else:
                env = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=self.pipe_dir)
                r = subprocess.run(
                    ["nvidia-cuda-mps-control"], input=cmd + "\n", env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, timeout=self.budget)
            out = (r.stdout or "").strip()
        except subprocess.TimeoutExpired:
            out = "<TIMEOUT>"
        except Exception as e:                                     # noqa: BLE001
            out = "<EXC %s>" % str(e)[:60]
        return round(time.time() - t0, 4), out

    def _record(self, cmd, el, out, label, note, wait_s, held):
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S.") + "%03d" % (
                   int(time.time() * 1000) % 1000),
               "run_id": self.run_id, "label": label,
               "verb": cmd.split()[0], "arg": " ".join(cmd.split()[1:]),
               "elapsed_s": el, "wait_s": round(wait_s, 4),
               "lock_held": int(held), "class": classify(out), "note": note,
               "response": out.replace("\n", " ")[:200]}
        self.calls.append(rec)
        if self.csv_path:
            try:
                new = not os.path.exists(self.csv_path)
                with open(self.csv_path, "a", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(rec.keys()))
                    if new:
                        w.writeheader()
                    w.writerow(rec)
            except Exception:                                      # noqa: BLE001
                pass
        return rec

    def one(self, cmd, label="", note=""):
        """락을 직접 잡고 한 건만 쏜다 (시퀀스 밖의 단발 호출용)."""
        tw = time.time()
        with cli_lock(self.lock_key, timeout=self.budget * 4) as held:
            wait_s = time.time() - tw
            el, out = self._run(cmd)
        self._record(cmd, el, out, label, note, wait_s, held)
        return el, out

    def ps(self, label="ps"):
        el, out = self.one("ps", label=label)
        return el, parse_ps(out)

    def server(self, label="get_server_list"):
        _, out = self.one("get_server_list", label=label)
        return next((t for t in out.split() if t.isdigit()), None)


def reclaim(cli, pids, arm, label, say, srv=None, verify_after=True):
    """회수 시퀀스 -- 두 실험군이 공유하는 **유일한** 발행 경로.

        arm="ps_term"  대상마다 ps 로 명부를 확인한 뒤 terminate_client
        arm="term"     terminate_client 만
        arm="ps"       ps 만. terminate 는 발행하지 않는다.

    세 arm 은 "mps ps + terminate_client 조합이 문제"라는 가설의 3분해다.
    ps 만 돌려도 같은 증상이 나오면 terminate 는 재료가 아니고, terminate 만
    돌려도 나오면 ps 가 재료가 아니며, 둘 다 아닌데 조합에서만 나오면 조합이다.
    arm="ps" 는 회수가 0건이므로 "성공 0건 -> VOID" 규칙을 그대로 적용하면 안 된다
    -- 호출자가 이 arm 을 대조군으로 따로 다뤄야 한다.

    시퀀스 전체를 락 한 번으로 감싼다. 그래야 한 대상의 (ps, terminate) 쌍 사이에
    다른 스레드의 호출이 끼지 않는다. 쌍이 섞이면 그 사이에 대상 상태가 바뀌어,
    지연을 재는 실험에서 무시할 수 없는 차이가 된다.
    """
    assert arm in ("ps_term", "term", "ps"), arm
    say("--- %s: %d개 (arm=%s) ---" % (label, len(pids), arm))
    res, tw = [], time.time()
    with cli_lock(cli.lock_key, timeout=cli.budget * 8) as held:
        seq_wait = time.time() - tw
        if not held:
            say("  주의: 락을 예산 안에 못 잡음 -- 거절될 수 있음")
        if srv is None:
            el, out = cli._run("get_server_list")
            cli._record("get_server_list", el, out, label, "", 0.0, held)
            srv = next((t for t in out.split() if t.isdigit()), None)
        if srv is None:
            say("VOID %s: get_server_list 가 서버 pid 를 못 줌 -- 발행 0건" % label)
            return {"res": [], "counts": {}, "ok": 0, "srv": None,
                    "worst_term_s": 0.0, "worst_ps_s": 0.0, "seq_wait_s": seq_wait}
        for pid in pids:
            note, ps_el = "", 0.0
            if arm in ("ps_term", "ps"):
                ps_el, ps_out = cli._run("ps")
                seen = parse_ps(ps_out)
                # 대상이 명부에 없는데 쏘면 그건 유령 회수다. 기록에 남겨야
                # "회수했다"로 오독하지 않는다.
                note = "n=%d tgt=%d" % (len(seen), 1 if pid in seen else 0)
                cli._record("ps", ps_el, ps_out, label + ":ps", note, 0.0, held)
            if arm == "ps":
                # 대조군: 호출 횟수와 발행 리듬은 ps_term 과 같게 두고
                # terminate 만 뺀다.
                el, out = 0.0, "<NO_TERM>"
            else:
                cmd = "terminate_client %s %s" % (srv, pid)
                el, out = cli._run(cmd)
                cli._record(cmd, el, out, label + ":term", note, 0.0, held)
            # 종료 직후 대상이 실제로 사라졌는지 확인한다.
            #
            # 가설: 201(CUDA_ERROR_INVALID_CONTEXT)은 **엉뚱한 컨텍스트를 지목해
            # 종료한 것**이고, 그래서 (a) 진짜 대상은 살아남고 (b) 애먼 컨텍스트가
            # 무너져 이웃 오염으로 나타난다. 맞다면 201 직후 ps 에 대상 pid 가
            # **그대로 남아 있어야** 한다.
            #
            # 성공(rc=0) 응답에도 같은 확인을 한다. "0 을 받았는데 안 죽었다"가
            # 있는지가 이 가설의 대조군이다.
            post = ""
            if verify_after:
                _pel, pout = cli._run("ps")
                pseen = parse_ps(pout)
                post = "after_n=%d after_tgt=%d" % (
                    len(pseen), 1 if pid in pseen else 0)
                cli._record("ps", _pel, pout, label + ":post", post, 0.0, held)

            res.append((pid, el, out, ps_el, (note + " " + post).strip()))
            if ps_el >= 1.0 or el >= 1.0 or classify(out) != "ok" \
                    or note.endswith("tgt=0") or post.endswith("after_tgt=1"):
                say("  [%s] pid=%d ps=%.3fs term=%.3fs %s %s -> %s"
                    % (label, pid, ps_el, el, note, str(out)[:60], post))
    counts = {}
    for r in res:
        k = classify(r[2])
        counts[k] = counts.get(k, 0) + 1
    say("%s 응답: %s" % (label, " ".join("%s=%d" % kv for kv in sorted(counts.items()))))
    ok = [r for r in res if classify(r[2]) == "ok"]
    worst_term = max((r[1] for r in ok), default=0.0)
    worst_ps = max((r[3] for r in res), default=0.0)
    survived = [r for r in res if str(r[4]).endswith("after_tgt=1")]
    say("%s worst_term=%.3fs worst_ps=%.3fs seq_wait=%.3fs (성공 %d/%d, 종료후 잔존 %d)"
        % (label, worst_term, worst_ps, seq_wait, len(ok), len(res), len(survived)))
    for r in survived:
        say("  ! 응답=%s 인데 pid=%d 가 ps 에 그대로 남음" % (classify(r[2]), r[0]))
    return {"res": res, "counts": counts, "ok": len(ok), "srv": srv,
            "worst_term_s": worst_term, "worst_ps_s": worst_ps,
            "seq_wait_s": seq_wait, "survived": len(survived)}

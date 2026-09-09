#!/usr/bin/env python3
"""The MPS control CLI core. Every harness in this family issues through here.

Two defects (reclaim delay, neighbour poisoning) and two platforms (Kubernetes,
docker) used to call the CLI from four different places. They were merged into
this one file for a simple reason: implement the same procedure twice and a
measurement trap fixed on one side does not reach the other. That happened --
the two harnesses interleaved their calls differently, and only one of them was
correct.

**The issuing procedure is fixed here.**

    One sequence. For each target: ps -> terminate_client, in order.
    No other call is interleaved between a target's (ps, terminate) pair,
    because the whole sequence is wrapped in a single lock acquisition. Threads
    that fire simultaneously therefore queue at pod/container granularity.

**What is NOT fixed here**: what counts as an error, and what shape to run. That
belongs to the caller, and it differs per defect.

Every rule below was learned by getting it wrong first.

1. **Issue sequentially.** The MPS control daemon accepts one connection at a
   time. Concurrent callers are not queued -- they are refused with "Cannot send
   command to MPS control daemon process" in about a millisecond, having done
   nothing. The lock is a *file* lock: the daemon is shared node-wide, so a
   thread lock cannot exclude another process (measured: two campaigns fighting
   produced 7 refusals).
2. **Classify every response.** Only "0" is success.
   201 = CUDA_ERROR_INVALID_CONTEXT, "Invalid process <pid>!",
   "Server 0 not found" and "Cannot send command" are all *not* reclaims
   (measured: only 73 of 99 responses were genuine reclaims).
3. **Time ps and terminate separately, and waiting separately from running.**
   Merge them and you cannot say which one stalled, and lock waiting gets
   misread as terminate duration.
4. **Do not issue if the server pid is unknown.** Passing None makes the CLI
   read it as 0 and print "Server 0 not found" -- nothing was issued, yet it
   lands in the record as if something was.
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
    """Per-daemon (per-GPU) file lock. Reports held=False if not acquired in time.

    It does not wait forever on purpose: if whoever holds the lock is stuck, the
    whole experiment stops. Failing to acquire must be recorded -- swallow it
    silently and refusals leak back into the data.
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
        # arm="ps" never issues a terminate at all. That is not a failure, it is
        # "did not fire", so it must count as neither success nor failure.
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
    """Issues one CLI call, times it, records it. Two transports:

        pipe_dir=... : directly on the host (a Kubernetes block MPS daemon)
        container=...: through `docker exec` (docker's private MPS daemon)

    `budget` is matched to the pod/container termination grace period, because
    that is the boundary that means something in production: if a reclaim has
    not finished within it, the layer above escalates to SIGKILL, and SIGKILL on
    a client holding a resident queue poisons its neighbours.
    """

    def __init__(self, lock_key, pipe_dir=None, container=None,
                 csv_path=None, budget=30.0, run_id=""):
        assert pipe_dir or container, "need either pipe_dir or container"
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
        """Take the lock and issue a single call (for one-offs outside a sequence)."""
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
    """The reclaim sequence -- the single issuing path shared by every harness.

        arm="ps_term"  per target: ps to read the client table, then terminate_client
        arm="term"     per target: terminate_client only
        arm="ps"       per target: ps only, no terminate at all

    The three arms decompose the hypothesis "it is the combination of mps ps and
    terminate_client". If ps alone shows the symptom, terminate is not an
    ingredient; if terminate alone shows it, ps is not; if neither does but the
    pair does, it is the combination. Note that arm="ps" reclaims nothing, so the
    caller must NOT apply the usual "zero successes -> VOID" rule to it: it is a
    control, not a failed cell.

    The whole sequence is wrapped in one lock acquisition so that no other
    thread's call lands between a target's ps and its terminate. If the pairs
    interleave, the target's state can change in between -- not a difference you
    can ignore when the thing being measured is latency.
    """
    assert arm in ("ps_term", "term", "ps"), arm
    say("--- %s: %d targets (arm=%s) ---" % (label, len(pids), arm))
    res, tw = [], time.time()
    with cli_lock(cli.lock_key, timeout=cli.budget * 8) as held:
        seq_wait = time.time() - tw
        if not held:
            say("  WARNING: lock not acquired within budget -- calls may be refused")
        if srv is None:
            el, out = cli._run("get_server_list")
            cli._record("get_server_list", el, out, label, "", 0.0, held)
            srv = next((t for t in out.split() if t.isdigit()), None)
        if srv is None:
            say("VOID %s: get_server_list returned no server pid -- nothing issued"
                % label)
            return {"res": [], "counts": {}, "ok": 0, "srv": None,
                    "worst_term_s": 0.0, "worst_ps_s": 0.0, "seq_wait_s": seq_wait}
        for pid in pids:
            note, ps_el = "", 0.0
            if arm in ("ps_term", "ps"):
                ps_el, ps_out = cli._run("ps")
                seen = parse_ps(ps_out)
                # If the target is not in the table and we fire anyway, that is a
                # reclaim of a ghost. Record it, or it reads later as a real one.
                note = "n=%d tgt=%d" % (len(seen), 1 if pid in seen else 0)
                cli._record("ps", ps_el, ps_out, label + ":ps", note, 0.0, held)
            if arm == "ps":
                # Control arm: same call count and same rhythm as ps_term, with
                # only the terminate removed.
                el, out = 0.0, "<NO_TERM>"
            else:
                cmd = "terminate_client %s %s" % (srv, pid)
                el, out = cli._run(cmd)
                cli._record(cmd, el, out, label + ":term", note, 0.0, held)
            # Check whether the target actually disappeared.
            #
            # Hypothesis under test: 201 (CUDA_ERROR_INVALID_CONTEXT) means the
            # daemon named the *wrong* context, so (a) the real target survives
            # and (b) some innocent context is torn down instead, which is what
            # surfaces as neighbour poisoning. If that were so, the target pid
            # would still be in `ps` right after a 201.
            #
            # The same check runs on successful (rc=0) responses too: "got 0 but
            # it did not die" is the control for this hypothesis.
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
    say("%s responses: %s"
        % (label, " ".join("%s=%d" % kv for kv in sorted(counts.items()))))
    ok = [r for r in res if classify(r[2]) == "ok"]
    worst_term = max((r[1] for r in ok), default=0.0)
    worst_ps = max((r[3] for r in res), default=0.0)
    survived = [r for r in res if str(r[4]).endswith("after_tgt=1")]
    say("%s worst_term=%.3fs worst_ps=%.3fs seq_wait=%.3fs (ok %d/%d, still present %d)"
        % (label, worst_term, worst_ps, seq_wait, len(ok), len(res), len(survived)))
    for r in survived:
        say("  ! response=%s but pid=%d is still in ps" % (classify(r[2]), r[0]))
    return {"res": res, "counts": counts, "ok": len(ok), "srv": srv,
            "worst_term_s": worst_term, "worst_ps_s": worst_ps,
            "seq_wait_s": seq_wait, "survived": len(survived)}

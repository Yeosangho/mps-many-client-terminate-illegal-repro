#!/usr/bin/env python3
"""도커 컨테이너 함대 -- k8s 파드 모양을 도커로 옮긴다.

두 실험군이 이 모듈을 공유한다. 다른 것은 **모양(클라 수, 큐 깊이)** 과
**회수 대상 선택**뿐이고, 컨테이너 기동/클라 매핑/진행 확인/채점은 같다.

지켜야 하는 것들:

* **fork 로 워커를 만들지 않는다.** 부모가 PID 1 이면 부모 사망 시 PID namespace
  가 붕괴해 커널이 자식 전원을 SIGKILL 하고, 그 사망이 MPS 전파로 둔갑한다.
  워커는 `docker exec -d` 로 각각 띄운다.
* **측정 창 안에서 컨테이너를 지우지 않는다.** 지우면 "회수가 죽였다"와
  "컨테이너가 죽었다"가 구분되지 않는다.
* **`--pid=host`** 가 필요하다. 호스트 pid 로 MPS 클라와 컨테이너를 매핑해야
  누구를 회수했는지 말할 수 있다.
* **붙지 않은 클라는 살아남은 방관자가 아니다.** 기대 수에 못 미치면 VOID.
"""
import json
import os
import re
import subprocess
import time

IMG = os.environ.get("IMG", "nvcr.io/nvidia/pytorch:26.06-py3")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# 워크로드 스크립트와 사전 빌드된 spin 커널. 기본값은 이 리포 안이다.
# queuefill.py 는 커널 cubin 이 없으면 FileNotFoundError 로 죽고, 그러면
# 오펜더가 한 개도 안 붙어 셀이 통째로 VOID 가 된다.
WL = os.environ.get("WL", _HERE)
ART = os.environ.get("ART", os.path.join(_ROOT, "artifacts"))
# queuefill.py 는 사전 빌드된 spin 커널(cubin)을 요구한다. 안 마운트하면
# 워커가 전부 FileNotFoundError 로 죽고 victim 만 남아, 클라 수가 기대에 못 미쳐
# 셀이 VOID 가 된다(실측: 16개 워커 전멸, 클라 2개).

POISON_KEYS = ("POISONED", "illegal memory", "IllegalAddress", "CUDNN_STATUS_",
               "CUBLAS_STATUS_", "cudaError")


def sh(cmd, timeout=180):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, timeout=timeout)
    return r.stdout or ""


class Fleet(object):
    def __init__(self, prefix, gpu_uuid, root, say, off_pct="25",
                 off_libvgpu=None):
        self.prefix = prefix
        self.gpu_uuid = gpu_uuid
        self.root = root
        self.say = say
        self.off_pct = off_pct
        self.off_libvgpu = off_libvgpu
        self.mps = prefix + "_mps"
        self.off = []
        self.vic = []

    # ------------------------------------------------------------- lifecycle
    def clean(self, keep_mps=False):
        """컨테이너 정리. keep_mps 면 MPS 데몬 컨테이너는 남긴다.

        rep 마다 데몬을 새로 띄우면 셀 간 오염이 없어 깨끗하지만, **누적 효과를
        볼 수 없다.** 40클라 순차 종료에서 관측된 19ms -> 225ms 단조 악화는
        한 데몬 안에서 회수를 거듭할수록 나빠진다는 뜻이므로, 데몬을 유지한 채
        rep 을 반복해야 그 축을 잴 수 있다.
        """
        out = sh(["docker", "ps", "-a", "--format", "{{.Names}}", "--filter",
                  "name=" + self.prefix + "_"], timeout=60).split()
        names = [n for n in out if not (keep_mps and n == self.mps)]
        if names:
            sh(["docker", "rm", "-f"] + names, timeout=300)

    def mps_alive(self):
        out = sh(["docker", "ps", "--format", "{{.Names}}", "--filter",
                  "name=^%s$" % self.mps], timeout=60).split()
        return self.mps in out

    def start_mps(self, reuse=False):
        if reuse and self.mps_alive():
            # 데몬을 그대로 쓴다. 파이프 디렉터리도 지우지 않는다 -- 지우면
            # 살아 있는 데몬의 소켓이 사라져 이후 모든 호출이 실패한다.
            self.say("MPS 데몬 재사용 (%s)" % self.mps)
            return
        subprocess.run("rm -rf %s/mps %s/mpslog && mkdir -p %s/mps %s/mpslog"
                       % (self.root, self.root, self.root, self.root),
                       shell=True, timeout=60)
        self.say("MPS 데몬 컨테이너 기동 (%s)" % self.gpu_uuid)
        sh(["docker", "run", "-d", "--name", self.mps, "--runtime=nvidia",
            "--ipc=host", "--pid=host",
            "-e", "NVIDIA_VISIBLE_DEVICES=" + self.gpu_uuid,
            "-e", "CUDA_MPS_PIPE_DIRECTORY=/mps",
            "-e", "CUDA_MPS_LOG_DIRECTORY=/mpslog",
            "-v", self.root + "/mps:/mps", "-v", self.root + "/mpslog:/mpslog",
            IMG, "bash", "-c",
            "nvidia-cuda-mps-control -d && sleep infinity"], timeout=600)
        time.sleep(6)

    def _common(self):
        return ["--runtime=nvidia", "--ipc=host", "--pid=host",
                "-e", "NVIDIA_VISIBLE_DEVICES=" + self.gpu_uuid,
                "-e", "CUDA_MPS_PIPE_DIRECTORY=/mps",
                "-v", self.root + "/mps:/mps", "-v", WL + ":/wl:ro",
                "-v", ART + ":/artifacts:ro"]

    def start_victims(self, nvic, batch):
        """victim 만 먼저 띄운다.

        오펜더와 동시에 띄우면 victim 이 첫 스텝도 못 찍는 일이 있었는데, 그것이
        (a) SM 경쟁으로 느려진 것인지 (b) 디스패치가 아예 안 되는 정지인지
        구분할 수 없었다. MPS 는 SM 할당량을 넘기면 시분할로 전환되므로 단순 경쟁
        으로 진행이 0 이 되지는 않는다 -- 그래서 이 구분이 중요하다.

        victim 을 먼저 정상 속도까지 올려놓고 나서 오펜더를 붙이면, 그 뒤 victim 이
        멈추는지 여부가 곧 답이 된다.
        """
        self.vic = ["%s_v%d" % (self.prefix, i) for i in range(1, nvic + 1)]
        for i, v in enumerate(self.vic):
            env = ["-e", "PYTHONUNBUFFERED=1", "-e", "ROLE=victim%d" % (i + 1),
                   "-e", "BATCH=%s" % batch, "-e", "NO_EXIT_ON_POISON=0",
                   "-e", "EXIT_MODE=exit"]
            sh(["docker", "run", "-d", "--name", v] + self._common() + env
               + [IMG, "bash", "-c",
                  "python3 /wl/victim_resnet.py 2>&1 | tee /tmp/w1.log"],
               timeout=600)
        time.sleep(3)

    def start_offenders(self, noff, nproc, qdepth, ksec):
        """오펜더를 띄운다. 워크로드는 컨테이너의 PID 1 = pid1_shim.

        k8s 파드와 같은 구조다: PID 1 은 CUDA 를 안 쓰고, 워커 N 개는 그 자식이다.
        fork 가 아니다 -- 부모가 PID 1 이면 부모 사망 시 namespace 가 붕괴해
        커널이 자식 전원을 SIGKILL 하고, 그 사망이 MPS 전파로 둔갑한다.
        """
        self.off = ["%s_o%d" % (self.prefix, i) for i in range(1, noff + 1)]
        for i, o in enumerate(self.off):
            env = ["-e", "PYTHONUNBUFFERED=1", "-e", "ART_DIR=/artifacts",
                   "-e", "WORK_DIR=/wl", "-e", "ROLE=off%d" % (i + 1),
                   "-e", "NPROC=%d" % nproc, "-e", "KERNEL_SEC=%s" % ksec,
                   "-e", "QUEUE_DEPTH=%s" % qdepth, "-e", "SPIN_SEC=%s" % ksec,
                   "-e", "NO_EXIT_ON_POISON=0", "-e", "EXIT_MODE=exit"]
            if self.off_libvgpu:
                # 오펜더만 TPC 마스크 아래로 가둔다.
                #
                # 도커에서 이웃이 오펜더 부착만으로 영구 정지하는데, k8s 에서는
                # 같은 16 오펜더 아래 이웃이 17 sps 로 계속 돈다. 유일한 차이가
                # libvgpu 의 TPC 마스크(PARTITION_PERCENTAGE=25)였고, MPS 스레드
                # 캡으로는 대체되지 않는 것을 확인했다(캡 25%와 100%가 동일).
                #
                # 그래서 오펜더에만 마스크를 씌워 SM 점유를 25%로 가둔다. victim 은
                # 마스크 없이 전체를 쓴다 -- k8s 처럼 파드마다 서로 다른 파티션을
                # 주는 것은 아니지만, "오펜더가 모든 SM 을 못 먹는다"는 핵심 성질은
                # 같다. 이 차이는 결과 해석에 명시해야 한다.
                env += ["-e", "CONFIG_DIR=/etc/block_gpu",
                        "-e", "LD_PRELOAD=/etc/block_gpu/lib/libvgpu.so"]
            if self.off_pct and self.off_pct != "100":
                # k8s 의 TPC 마스크(PARTITION_PERCENTAGE=25)를 근사한다.
                # **기구가 다르다** -- TPC 마스크는 SM 을 공간 분할하고 이것은
                # 클라당 스레드 수를 제한한다. 도커 재현의 알려진 비대응 지점.
                env += ["-e", "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=%s" % self.off_pct]
            mounts = []
            if self.off_libvgpu:
                mounts = ["-v", self.off_libvgpu + ":/etc/block_gpu:ro"]
            sh(["docker", "run", "-d", "--name", o] + self._common() + env + mounts
               + [IMG, "bash", "-c",
                  "python3 /wl/pid1_shim.py queuefill.py 2>&1 | tee /tmp/w1.log"],
               timeout=600)
        time.sleep(3)

    def wait_clients(self, cli, want, boot=600):
        self.say("기대 클라 %d개" % want)
        t0, last, stable, pids = time.time(), -1, 0, []
        while time.time() - t0 < boot:
            _, pids = cli.ps(label="boot")
            if len(pids) != last:
                self.say("  MPS clients=%d" % len(pids))
                last, stable = len(pids), 0
            else:
                stable += 1
            if len(pids) >= want and stable >= 6:
                break
            time.sleep(5)
        self.say("  최종 클라 %d개 (기대 %d, 안정 %d회)" % (len(pids), want, stable))
        return pids

    # -------------------------------------------------------------- mapping
    def owner_map(self, pids):
        owner = {}
        for n in self.off + self.vic:
            out = sh(["docker", "top", n, "-eo", "pid"], timeout=60)
            cp = {int(x) for x in re.findall(r"^\s*(\d+)\s*$", out, re.M)}
            for p in pids:
                if p in cp:
                    owner[p] = n
        return owner

    # ------------------------------------------------------------- progress
    def logs(self, name):
        # PID 1 로 띄우면 워크로드 출력이 컨테이너 stdout 으로 나간다.
        # 컨테이너가 죽은 뒤에도 읽을 수 있도록 docker logs 를 쓴다 --
        # exec 로 파일을 읽으면 죽은 컨테이너에서는 아무것도 못 얻는다.
        return sh(["docker", "logs", "--tail", "400", name], timeout=60)

    def last_step(self, name):
        """마지막 진행 이벤트. victim 은 step, queuefill 오펜더는 qfill/done.

        오펜더도 이웃이다. victim 만 보고 셀을 무효 처리하면, victim 이 정지한
        셀에서 **손대지 않은 오펜더가 illegal 을 맞은 사실**을 통째로 놓친다
        (실측: dkill_o2 가 done=3 상태에서 illegal, 같은 셀의 victim 은 step=0).
        """
        for ln in reversed(self.logs(name).splitlines()):
            if "EVT " not in ln:
                continue
            try:
                d = json.loads(ln.split("EVT ", 1)[1])
            except Exception:                                      # noqa: BLE001
                continue
            if d.get("ev") == "step":
                return d.get("step"), d.get("sps")
            if d.get("ev") in ("qfill", "done"):
                return d.get("done"), None
        return None, None

    def wait_victims_ready(self, min_step=200, timeout=300, poll=5.0):
        """victim 이 **실제로 학습을 돌고 있을 때까지** 기다린다.

        왜 필요한가: ResNet152 는 컨테이너 기동 후 cuDNN autotune 까지 20~50초가
        걸린다. `--steady 30` 만 믿고 발사하면 victim 이 아직 첫 step 도 못 찍은
        상태에서 회수가 나간다. 실측: POISONED 줄의 step 이 전부 0 이었고 진행
        게이트는 step=None 을 읽었다.

        그 상태의 사망은 "정상 진행 중이던 이웃이 죽었다"가 아니라 "기동 중이던
        이웃이 죽었다"라서, k8s 재현(step 2610 에서 +90/5s 진행 중 사망)과 같은
        사건이라고 말할 수 없다. 전파를 주장하려면 이 게이트를 통과해야 한다.
        """
        t0 = time.time()
        last = None
        while time.time() - t0 < timeout:
            cur = {v: self.last_step(v)[0] for v in self.vic}
            ok = all(isinstance(cur[v], int) and cur[v] >= min_step for v in self.vic)
            if cur != last:
                self.say("  victim 준비: %s (기준 step>=%d)"
                         % (", ".join("%s=%s" % (v, cur[v]) for v in self.vic),
                            min_step))
                last = cur
            if ok:
                return True
            time.sleep(poll)
        self.say("  victim 준비 실패: %s" % last)
        return False

    def progress(self, names, tag, gap=5.0):
        """임의 컨테이너 묶음의 진행을 5초 증분으로 확인한다.

        "회수했더니 이웃이 죽었다"는 회수 직전에 그 이웃이 실제로 돌고 있었을
        때만 전파의 증거다. 이웃마다 따로 판정해야 한다 -- 한 이웃이 멈춰 있다고
        다른 이웃의 사망까지 무효가 되는 것은 아니다.
        """
        a0 = {n: self.last_step(n)[0] for n in names}
        time.sleep(gap)
        live = {}
        for n in names:
            s1, sps = self.last_step(n)
            d = (s1 - a0[n]) if (isinstance(s1, int) and isinstance(a0[n], int)) else None
            live[n] = bool(d)
            self.say("  %s %s: prog=%s (+%s in %.0fs) sps=%s%s"
                     % (tag, n, s1, d, gap, sps, "  <-- 진행 없음" if not d else ""))
        return live

    def victim_progress(self, tag, gap=5.0):
        """발사 직전 victim 이 진행 중인지 증분으로 확인한다.

        "회수했더니 이웃이 죽었다"는 회수 직전에 이웃이 실제로 돌고 있었을 때만
        전파의 증거다. 증분이 0이면 그 라운드는 판정이 성립하지 않는다.
        """
        live = self.progress(self.vic, tag + " victim", gap=gap)
        return all(live.values())

    def score(self, names, tag):
        nerr = nill = 0
        for n in names:
            hits = [l for l in self.logs(n).splitlines()
                    if any(k in l for k in POISON_KEYS)]
            if hits:
                nerr += 1
                self.say("  %s %s: %s" % (tag, n, hits[-1][:150]))
                if any(("illegal memory" in h or "IllegalAddress" in h) for h in hits):
                    nill += 1
            else:
                self.say("  %s %s: 에러 없음" % (tag, n))
        return nerr, nill

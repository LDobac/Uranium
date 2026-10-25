# M6 — 시간 축 코어 2차 비교 (PyTorch, LSTM vs GTrXL, 032-R17 ⑥)

- 상태: **학습 중** (2026-10-24 시작)
- spec: [032](../spec/032-model-architecture.md) 032-R17 ⑥, 032-AC7
- 1차 비교(Burn, 5,300 update, GTrXL 채택): [`m6_core_ablation.md`](m6_core_ablation.md)

1차 리포트와 같은 규칙으로 쓴다. 배경·예상은 실험 전에 적고 고치지 않는다. 진행 기록은 아래에 시각과 함께 덧붙이기만 한다.

## 1. 배경 (실험 전 정리, 2026-10-24)

### 1.1 왜 다시 하나
- 2026-10-24 사용자 결정: BC 본학습 전에 LSTM·GTrXL 비교를 다시 한다. 코어마다 **10,000 update**(1차 5,300의 약 1.9배), 장비는 1차와 같게 학습 PC(CUDA, **유휴일 때만**)와 BC250 1호기(Linux ROCm, 상시).
- 1차 비교 뒤 학습 구현이 Burn에서 PyTorch로 바뀌었다(AMB-000-4). 1차 체크포인트는 새 코드에서 읽지 않는다.

### 1.2 1차와 달라진 조건
| 항목 | 1차 (10-20~23) | 2차 |
|---|---|---|
| 학습 구현 | Burn | PyTorch 2.14 (+ 2단계 최적화, 같은 수식, 032-R19) |
| 길이·스케줄 | S1 5,300, warmup 265 + cosine | S1 **10,000**, warmup 500 + cosine (둘 다 5%) |
| CUDA 정밀도 | TF32 matmul | bf16 혼합 (032-R8) |
| Linux 학습 호스트 | Vulkan f32 | ROCm f32 |
| 데이터 | 2,044 파일 | 동결 데이터 2,069 파일 (scr_ladder shard 25개 복원, I-D1) |
| win_open 헤드 | 없음 | 있음 (`bc.toml`과 같게. 입력이 detach돼 나머지 학습에는 영향 없음, 041-R19) |
| 설정 | `config/bc_core_ablation.toml` | `config/bc_core_ablation_pt.toml` (`bc.toml`과 S1 길이·warmup·보존만 다름, 테스트로 검사) |
| run id | `core-abl-lstm` / `core-abl-gtrxl` | `core-abl-lstm-pt` / `core-abl-gtrxl-pt` |

평가는 1차와 같게 잡았다(032-R17 ②~⑤): val 500게임(고정 seed 셔플), 맞대결 100판, T0 상대 600판, 확률적 샘플링, 리플레이 전부 저장. plan은 `config/plans/core-ablation-pt-{h2h,t0}.toml`(1차와 같은 seed·대진, 체크포인트만 `ckpt-10000`).

### 1.3 실험 전 예상 (Claude, 측정 아님)
- 1차는 val 지표 전부, 맞대결(42 : 13), T0 승률(약 10% : 2.7%) 모두 GTrXL이 앞섰다. 차이가 seed 잡음보다 커 보여서 2차에서도 GTrXL이 앞설 가능성이 높다고 본다.
- 길게 학습하면 LSTM이 따라붙을 여지는 있다. 1차의 5,300 update는 S1 한 epoch(39,096)의 14%였고, 2차 10,000도 26%라서 둘 다 학습 초반이다. 학습 초반에 수렴이 빠른 쪽(GTrXL, T 방향 병렬·attention)이 유리했을 수 있다.
- 장비 비율과 정밀도 차이(CUDA bf16 vs ROCm f32)가 1차보다 크다. bf16 autocast는 같은 입력에서 출력 차이가 중앙값 1.5%였다(`pytorch_opt.md`, 032-R8 측정). 두 run의 장비 비율이 크게 다르면 비교에 섞인다.

### 1.4 예상 시간
학습 PC 2.0 s/update, BC250 1호기 10 s/update(GTrXL, `pytorch_opt.md` 3절) 기준. LSTM은 2단계 최적화 측정에 들어가지 않아 아래 진행 기록에서 잰다.
- 두 run 합계 20,000 update. BC250 1호기 단독이면 약 56시간(약 2.3일), 학습 PC 단독이면 약 11시간.
- 함께 쓰면 처리량 `a/2.0 + 1/10` update/s(a = 학습 PC 가용률): a = 50%면 약 17시간, a = 25%면 약 27시간.

## 2. 실행 절차

큐 파일 `<shared>/queue/core-ablation-pt.toml`:
```toml
[defaults]
datasets = ["replays/shards/stardata", "replays/shards/community", "replays/shards/scr_ladder", "replays/shards/personal"]
config = "config/bc_core_ablation_pt.toml"

[[run]]
id = "core-abl-lstm-pt"
core = "lstm"

[[run]]
id = "core-abl-gtrxl-pt"
core = "gtrxl"
```
```bash
# BC250 1호기 (상시)
setsid nohup target/release/uranium-train worker --queue <shared>/queue/core-ablation-pt.toml \
  --backend rocm --priority 1 --log runs/worker-abl-pt.log > runs/worker-abl-pt.out 2>&1 &
```
```powershell
# 학습 PC: 유휴일 때만 실행, 쓸 때 Ctrl+C (체크포인트·push·잠금 해제 뒤 BC250 1호기가 이어받는다)
target\release\uranium-train worker --queue <shared>/queue/core-ablation-pt.toml --backend cuda --priority 10 --log runs\worker-abl-pt.log
```
- 장비별 update 수: `uranium-train devices --run core-abl-lstm-pt` / `core-abl-gtrxl-pt`.
- 산출물(지우지 않음): 체크포인트 `<shared>/checkpoints/core-abl-{lstm,gtrxl}-pt/`(전부 보존), 장비 구간 `<shared>/runs/core-abl-{lstm,gtrxl}-pt/devices.jsonl`, 작업자 로그 각 장비 저장소 `runs/worker-abl-pt.log`.

## 3. 결과

(학습·평가 뒤)

## 4. 진행 기록

시각은 학습 PC 기준이다. 항목을 지우거나 고치지 않고 아래에 덧붙인다.

- 2026-10-24 — 사용자 결정으로 2차 비교 시작. spec 032-R17 ⑥, 설정 `config/bc_core_ablation_pt.toml`, plan 2개, 설정 동기화 테스트 추가(작업 브랜치 `core-ablation-pt`).
- 2026-10-24 21:22 — BC250 1호기를 작업 브랜치로 맞추고 `cargo build --release -p uranium-train`(J-3의 Linux 쪽) + `cargo test --release -p uranium-data --test bc` 25개 통과(새 설정 동기화 검사 포함). 로그 BC250 1호기 `~/build-logs/build_20261024_abl.log`.
- 2026-10-24 21:24~21:29 — **표본(BC250 1호기 ROCm f32, 실제 설정·실데이터, 6 update씩)**. 공용 드라이브 run `core-abl-smoke-lstm-pt`·`core-abl-smoke-gtrxl-pt`(지우지 않음, 학습 PC 재개 확인용). 시작 줄: 2,069 파일·33,183게임, 맵 사전계산 ok, `total_steps 10000`, `micro_lanes 1`·`prefetch 2`, run_ident config `ed216b8e2fe5ef55`·dataset `2069:86b35ab6a7a464aa`. 두 run 모두 rc 0, 손실 유한, `ckpt-6` push.

  | | LSTM | GTrXL |
  |---|---|---|
  | update 시간 (step 2~6) | 11.1~11.2 s | 9.9~10.5 s |
  | 첫 update (데이터 대기 포함) | 11.5 s + 대기 10.2 s | 13.4 s + 대기 10.1 s |
  | 시스템 메모리 최대 (10초 간격, GTT 포함) | 10.8 GB (GTT 4.2 GB) | 11.5 GB (GTT 5.0 GB) |
  | swap | 없음 | 없음 |

  LSTM ROCm(MIOpen 끔, `core.py`)도 GTrXL과 비슷한 속도라 1.4절 예상 시간은 그대로 쓴다. 로그 BC250 1호기 `~/build-logs/abl_pt_smoke_20261024/`(`lstm.log`, `gtrxl.log`, `mem.log`), 스크립트 `~/abl_pt_smoke.sh`.
- 남은 사전 확인(학습 PC 유휴일 때): 메인 작업 사본 `cargo build --release -p uranium-train`(J-3의 CUDA 쪽), 표본 run을 학습 PC CUDA에서 이어 받기(ROCm → CUDA 재개, 061-R10). BC250 1호기 작업자는 먼저 시작한다(재개에 문제가 있어도 BC250 1호기 체크포인트는 유효).
- 2026-10-24 21:3x — **학습 시작**: BC250 1호기 작업자(master 1ebb654, `~/abl_pt_worker.sh`: ROCm 실행 환경 + `uranium-train worker --backend rocm --priority 1`, 로그 `runs/worker-abl-pt.{log,out}`)가 큐 첫 run `core-abl-lstm-pt`를 잡았다. step 1·2 14.0·11.8 s. 학습 PC 작업자는 유휴일 때 시작한다(사용자 확인).
- 2026-10-24 21:35~21:41 — **학습 PC 사전 확인**(사용자: "지금 PC 유휴 상태, 진행해"): 메인 작업 사본 `cargo build --release -p uranium-train`(J-3의 CUDA 쪽, 18초). 표본 run 두 개를 CUDA에서 `--steps 9`로 이어 받음(ROCm → CUDA 재개, 061-R10): 둘 다 `ckpt-6`에서 재개, 같은 run_ident, `ckpt-9` push, rc 0. 학습 PC 설정 `micro_lanes 2`·`prefetch 1`. update 시간 LSTM 146.1(첫 update, 준비 시간)·5.3·2.7 s, GTrXL 19.2·2.7·2.7 s. 로그 학습 PC 저장소 `runs/abl_pt_smoke/{lstm,gtrxl}.cuda.log`.
- 2026-10-24 21:42~21:53 — **발견**: 학습 PC 작업자가 `core-abl-gtrxl-pt`를 잡았지만 자식 학습을 띄우지 못하고 "URANIUM_PYTHON is not set"으로 재시작을 반복했다(60·120·240·480초, 잠금은 잡지 않음). 학습 PC `.env`의 `URANIUM_PYTHON`이 따옴표 없는 Windows 경로라 Rust dotenvy가 읽지 못했다(Python 쪽 읽기는 됨 → 표본은 통과). 값을 작은따옴표로 감쌌다(원본 `.env.bak-20261024-pyquote`). Python 쪽도 같은 경로로 읽는 것을 확인.
- 2026-10-24 21:54 — **학습 PC 합류**: 새 PowerShell 창에서 작업자(`--backend cuda --priority 10`, 로그 `runs/worker-abl-pt.log`)가 `core-abl-gtrxl-pt`를 시작. step 1 84.4 s(torch.compile 준비), step 3~5 1.84~1.92 s. 두 run이 동시에 돈다: LSTM은 BC250 1호기, GTrXL은 학습 PC. 학습 PC를 쓸 때는 그 창에서 Ctrl+C.
- 2026-10-24 21:57~22:09 — **발견: BC250 1호기 LSTM 학습 프로세스 종료**: step 116 근처에서 자식이 시그널로 끝남(`Exited(None)`, 첫 체크포인트 전이라 step 0부터 재시작). 재시작 run의 메모리 기록: 학습 프로세스 상주가 update마다 늘어 약 25 update 만에 RAM 15.2 GB를 다 쓰고 swap 6.3 GB, update 11.5 → 15 s. 커널 로그는 sudo가 없어 못 봤지만 OOM으로 판단. 작업자에 SIGINT → `ckpt-48` push 뒤 정지. 로그 BC250 1호기 `~/build-logs/abl_pt_mem_20261024.log`.
- 2026-10-24 22:14~22:36 — 원인 좁히기(BC250 1호기, 표본 run `core-abl-smoke-gtrxl-pt`): GTrXL도 같은 증가(50 update에 사용 15.2 GB, swap 6.1 GB) → 코어 무관. 미리 읽기 1 + 엔티티 재계산 16도 GTT만 5.5 → 3.9 GB로 줄 뿐 상주가 10.2 GB + swap 2 GB로 76 update에서 한계. 학습을 빼고 데이터 공급·collate만 돌린 시험(update 대신 8 s 휴식)에서도 상주 3.5 → 13.6 GB(40 update), GTT 0.25 GB 그대로 → **데이터 공급 쪽 누적**. 학습 PC도 학습 프로세스 커밋 18.3 GB(작업 집합 5.6 GB)로 같은 현상.
- 2026-10-24 22:45~23:08 — 공급 전용 15 update 분해: 기준 3.7 → 7.6 GB(update당 약 +280 MB), batch 압축 끔 13~14 GB로 크지만 증가 없음, 공급 스레드 1·미리 읽기 0은 증가, `MALLOC_ARENA_MAX=2` 3.6 → 5.4 GB. **원인: batch 압축(041-R18)의 `zstd::bulk::compress`가 결과 버퍼를 원본 크기(step당 약 1.9 MB)로 잡아 둠** → batch(2,048 step)마다 약 4 GB가 할당된 채 남음. 압축 결과를 압축 크기로 줄이는 수정 뒤 15 update 3.3 → 4.4 GB, 50 update 3.25 → 5.4 GB(update당 약 44 MB, 중간에 평평한 구간). 로그 BC250 1호기 `~/memdiag/`(`feedonly.py`, `bisect*/`).
- 2026-10-24 22:3x~22:55 — **발견: 학습 PC Ctrl+C로 작업자가 멈추지 않음**(사용자 보고). 원인: Windows의 "Ctrl+C 무시" 속성이 부모에서 상속됨 → 학습 자식의 정지 핸들러가 불리지 않음. 작업자 창을 Claude 도구 프로세스에서 `Start-Process`로 열어 그 속성을 물려받은 것으로 보임(같은 방식으로 띄운 시험 파이썬은 venv 실행기·기본 인터프리터 모두 `stop_requested=False`, 속성 해제 시 `True`). 작업자·학습 자식·`python -m uranium` 시작 시 속성을 푸는 수정. 그 사이 학습 PC 작업자는 정시 체크포인트 `ckpt-991` push 직후 강제 종료(손실 없음). 학습 PC 데이터 대기(update 2.4~3.8 s 중 0.8~2.6 s)는 메모리 수정 뒤 다시 본다.
- 메모: 이 리포트 4절 21:24 표본의 메모리 최대(10.8 / 11.5 GB)와 `reports/pytorch_opt.md`의 BC250 1호기 메모리 수치(2.6·2.7·2.23절, 060-R3 판단 근거)는 압축 버그가 있던 때의 12 update 이하 측정이다. 수정 뒤 실제 학습에서 다시 잰다.
- 2026-10-24 23:30~23:44 — 압축기·버퍼 재사용 시험(공급 전용): 50 update 3.29 → 5.81 GB로 용량 수정만 한 것(5.40 GB)과 비슷해 반영하지 않음. 100 update에서 step 42 이후 5.8~6.1 GB로 평평 → 남은 증가는 게임 중후반 batch 크기 몫.
- 2026-10-24 23:5x — **실수와 복구**: 학습 PC 단계별 시간을 재려고 실제 run에 `bc --run core-abl-gtrxl-pt --steps 1352`로 40 update를 돌렸는데, `--steps`에 닿으면 run에 `DONE`을 써서 두 작업자가 GTrXL을 끝난 run으로 건너뜀(BC250 1호기 "no run to take"). 사용자 지적 뒤 `DONE`을 `DONE.mistaken-20261024`로 이름만 바꿔 되돌림. 그 사이 학습 PC 작업자는 LSTM을 잡아 `ckpt-94`부터 학습(BC250 1호기와 잠금 경쟁에서 이김).
- 2026-10-24 23:3x / 2026-10-25 00:0x — 학습 PC 단계 기록(40 update, 대기 0인 구간): forward + backward 1.73 s, 호스트→GPU 동기 복사 0.34 s, 레인 묶음 준비 대기 0.18 s → update 안에서 약 22% GPU 쉼. 실제 run 대기는 update 셋 중 하나꼴로 약 1 s. 원인과 수정은 `pytorch_opt.md` 5절(페이지 폴트, Windows 할당자 mimalloc: 2.75 → 2.05 s, pinned 복사까지 1.94 s).
- 2026-10-25 00:2x — BC250 1호기 GTrXL(`ckpt-1352`부터)도 메모리가 계속 늘어(13분에 상주 + swap 8.4 → 10.1 GB, update당 약 25 MB) `ckpt-1496`에서 Ctrl+C, 작업자 스크립트에 `MALLOC_ARENA_MAX=2`(glibc 할당 영역 수 제한, 결과 무관)를 넣어 재시작. 큐 순서대로 LSTM(`ckpt-287`, 학습 PC가 멈춘 지점)을 잡음. 학습 PC는 mimalloc 반영 뒤 재시작해 GTrXL을 잡는다.
- 2026-10-25 00:35~01:01 — **BC250 1호기 메모리 증가 원인 분해**(`MALLOC_ARENA_MAX=2`로 재시작한 LSTM run, `/proc/<pid>/smaps` 10분 간격, 상주 + swap MB): glibc 메인 힙 2,439 → 2,439, 파일 매핑 431 → 431, 60 MB 미만 익명 매핑 560 → 393, 60 MB 이상 익명 매핑 2,766 → 3,342(step 318 → 411). 큰 매핑 목록: 64 MB glibc arena heap 23 → 25개, 128 MB 공급 버퍼 8개로 일정 → 늘어난 것은 arena heap뿐(버퍼가 해제되지 않는 것이 아님). 학습 프로세스 상주는 step 320 6.2 GB, 342 6.45, 367 6.66, 390 6.63, 415 6.61, 439 6.61 GB로 약 70 update 동안 평평, swap 0(같은 시점 설정 없이는 8.6 GB + swap 1.6 GB로 증가 중). 판단: 늘던 몫은 batch마다 새로 생기는 공급 스레드에 glibc가 arena를 나눠 준 탓의 단편화로 보고, arena 수 제한을 `.env` 키 `URANIUM_MALLOC_ARENA_MAX`(시작 시 `mallopt`, 060-R3)로 정식화. 앞서 공급 전용 시험에서 같은 설정이 3.6 → 5.4 GB로 늘었던 것은 압축 버퍼 수정 전이고 15 update라 batch 크기 증가 구간이다. 로그 BC250 1호기 `~/memdiag/smaps_cat.log`, `~/build-logs/abl_pt_worker_mem2_20261024.log`.
- 2026-10-25 01:0x — 사용자: 학습을 계속 진행하고, 문제가 생기면 고친 뒤 재개. 감시는 문제 신호(작업자 종료·재시작, swap 증가, 데이터 대기, step 정지)만 알리게 둔다.
- 2026-10-25 02:0x~02:2x — BC250 1호기 LSTM step 693 → 740 사이에 큰 익명 매핑이 한 번 3.3 → 5.0 GB로 뜀(64 MB arena 25 → 약 52개, 128 MB 공급 버퍼 8 → 12개), 그 뒤 step 785까지 평평(상주 8.3 GB, swap 0, 남은 RAM 1.75 GB). 누수가 아니라 큰 batch 구간의 정점을 glibc가 쥔 것으로 봄. 감시 기준을 남은 RAM 800 MB·swap 1 GB로 앞당김(넘으면 update마다 `malloc_trim` 검토).
- 2026-10-25 02:47 — **일시 정지**(다른 세션 요청, 사용자 허가: 032-R20 구조 변형 5종 GPU 테스트·프로파일링): 학습 PC GTrXL `ckpt-6048`, BC250 1호기 LSTM `ckpt-935`에서 Ctrl+C로 정지(체크포인트 push, 잠금 해제, 작업자 종료). 큐에는 변형 5종(`arch-abl-*-pt`)이 이어서 추가됨. 재개는 그 세션의 요청 뒤, 큐 7개 run 전부.
- 2026-10-25 09:22 — `core-abl-lstm-pt` 10,000 update 완료(`ckpt-10000`, DONE). 장비 비율 학습 PC CUDA bf16 9,258(92.6%) / BC250 1호기 ROCm f32 742(7.4%). 학습 PC는 이어서 큐의 `arch-abl-terrain-pt`(032-R20)를 0 step부터 시작.
- 2026-10-25 13:04 — **학습 PC 작업자 정지**(다른 세션 요청, 사용자 지시: 이 GPU에서 WSL FlexAttention 측정): `arch-abl-terrain-pt` `ckpt-7463`에서 Ctrl+C(체크포인트 push, 잠금 해제, DONE 없음). BC250 1호기 `core-abl-gtrxl-pt`(step 약 8,620)는 계속. 학습 PC 재개는 그 세션 요청 뒤(Windows 또는 WSL).
- 2026-10-25 13:5x — **학습 PC 작업자 재개**(다른 세션 요청: WSL FlexAttention 측정 끝, Windows·WSL 차이가 오차 범위라 Windows에서): 같은 실행 환경(공급 스레드 6, 미리 읽기 2, `URANIUM_H2D=pinned`, 우선순위 10)으로 `arch-abl-terrain-pt` `ckpt-7463`부터, 약 1.8 s/update, 대기 0.
- 2026-10-25 14:3x — **결정**(사용자): 코어 비교 평가는 큐의 7개 모델(LSTM, GTrXL, 구조 변형 5개, 모두 `ckpt-10000`) 전체로, 7개 학습이 모두 끝난 뒤 한다. 맞대결은 서로 다른 21쌍 × 100판(맵 2 × 50, 25판마다 슬롯 교대: LSTM–GTrXL은 `core-ablation-pt-h2h.toml`, 나머지 20쌍은 `arch-abl-pt-league.toml`), T0는 모델당 300판(종족 3 × 맵 2 × 50: `core-ablation-pt-t0.toml` + 새 `arch-abl-pt-t0.toml`), val 500게임 채점도 7개 모두. spec 032-R20 ③ 반영(T0를 고른 후보만 → 7개 모두).
- 2026-10-25 17:25 — `core-abl-gtrxl-pt` 10,000 update 완료(`ckpt-10000`, DONE). 장비 비율 학습 PC CUDA bf16 5,904(59.0%) / BC250 1호기 ROCm f32 4,096(41.0%). LSTM(CUDA 92.6% / ROCm 7.4%)과 비율이 크게 달라 bf16·f32 수치 차이가 비교에 섞일 수 있다(공정성 주의, 결과 보고에 함께 적는다). 원인은 일시 정지 뒤 재개 때 두 run의 장비를 바꿔 배정한 것(빈 run이 있으면 양보 요청이 없어 LSTM이 BC250 1호기에 오래 머무는 것을 피하려고). 코어 비교 두 run 학습 끝. BC250 1호기는 이어서 `arch-abl-pool-pt`, 학습 PC는 `arch-abl-terrain-pt` 완료(14:54) 뒤 `arch-abl-ctx-pt`를 학습 중. 평가는 7개 학습이 모두 끝난 뒤(032-R20 ③).
- 2026-10-25 18:40 — 학습 PC 작업자가 Ctrl+C로 정상 정지(`arch-abl-ctx-pt` `ckpt-7482`, 사용자 PC 사용으로 봄). 23:4x 사용자 "유휴 상태, 재개" → 같은 실행 환경으로 재개, ctx 7482부터 약 1.95 s/update. 그동안 BC250 1호기는 `arch-abl-pool-pt`를 계속(step 약 1,970).
- 2026-10-26 00:38 — **BC250 1호기 작업자 정지**(사용자 지시 "Bc250의 학습을 중지해"): `arch-abl-pool-pt` `ckpt-2277`에서 SIGINT(체크포인트 push, 잠금 해제, DONE 없음). 남은 run은 학습 PC 작업자가 큐 순서대로 맡는다(ctx 뒤 pool 2277부터, 그다음 outer, all).
- 2026-10-26 00:55 — BC250 1호기 작업자 재개(사용자 지시): `~/abl_pt_worker.sh`로 다시 시작, `arch-abl-pool-pt` `ckpt-2277`부터 약 11~13 s/update.
- 2026-10-26 03:13~04:1x — **BC250 2호기 투입**(사용자 지시: "수리 후 복귀, bc250_1과 동일하게 셋팅 후 학습에 투입"; 10-18의 2호기 학습 금지는 이 지시로 폐지). 셋업: 저장소 master, uv(1호기에서 복사)로 torch 환경에 `uranium` 확장과 pytest 설치, `cargo build --release -p uranium-train`, `.env`에 학습 키 8개(1호기와 같은 값, 원본 `.env.bak-20261026-training`), 맵 사전계산 1,248개 복사, 작업자 스크립트 `~/abl_pt_worker.sh`(1호기와 같음). `pytest -m gpu` 10 통과·2 건너뜀. 작업자(우선순위 1)가 `arch-abl-all-pt`를 0 step부터: 첫 실행이라 데이터셋 복사(shard 1,913개)·스캔(캐시 없음)으로 첫 update까지 약 40분, 처음 보는 모양마다 MIOpen 첫 호출로 step 1~8은 283→43 s, step 12 이후 11.8~13.2 s/update(모두 켬 변형, 기준 대비 약 +14%). 남은 RAM 4.9 GB, swap 0. 장비 셋이 동시에 학습(학습 PC outer, 1호기 pool, 2호기 all). 학습 PC가 계속 유휴면 7개 전체 끝은 10-26 14시쯤(학습 PC가 outer 뒤 pool, 그다음 all을 양보받아 마무리).
- 2026-10-26 04:21 — **결정**(사용자): 작업자 우선순위 학습 PC(10) > BC250 1호기(2) > BC250 2호기(1). "bc250_1이 유휴면 bc250_2의 잡을 가져온다" — 빈 run이 없을 때 우선순위가 더 낮은 작업자의 run을 양보받는 기존 동작(061-R14)으로 구현. BC250 1호기 작업자를 `ckpt-3357`에서 Ctrl+C 뒤 `--priority 2`로 재시작(`~/abl_pt_worker.sh`), pool을 이어 받음. 예상: 06:15쯤 학습 PC가 outer를 끝내고 pool을 양보받으면 1호기가 2호기의 all을 양보받고, 09:20쯤 학습 PC가 pool을 끝내면 all을 양보받는다. 7개 전체 끝은 그대로 14:15쯤.

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

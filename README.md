# Uranium — 모방 학습(BC)·코어 비교 실행 안내

스타크래프트1 테란 봇. 이 문서는 **BC 학습과 코어 비교를 돌릴 때 필요한 것만** 모았다. 옵션 상세는 `docs/commands.md`, 규칙은 `spec/`(032·041·060·061), 경위와 결과는 `reports/`에 있다.

## 1. 순서

| 단계 | 내용 | 상태 |
|---|---|---|
| 0 | 데이터 재build·검증, 벤치, 대전 경로 시험 | 완료 (`reports/m6_rebuild.md` R-5) |
| 1 | **코어 비교**: LSTM·GTrXL을 S1 5,300 update씩 학습 → 평가 → 사용자가 코어 선택 | 시작 대기 |
| 2 | **BC 본학습 S1**: 전체 맵 38,584 update | 1 뒤 |
| 3 | **BC S2**: 대상 맵(투혼·서킷브레이커) 11,319 update, S1에서 시작 + 비교용 ablation (A) | 2 뒤 |
| 4 | 평가(오프라인 지표, T0 대전) → RL | 3 뒤 |

## 2. 장비 운영 규칙 (060-R8)

- **Linux Vulkan 호스트**: 작업자를 24시간 띄워 둔다(우선순위 1).
- **CUDA 호스트**: 쉬는 동안만 사용자가 작업자를 직접 켠다(우선순위 10). PC를 써야 하면 **Ctrl+C** → `ckpt-N Pushed` 출력 후 종료될 때까지 기다린다(약 30초).
- 작업자는 큐에서 끝나지 않은 run을 순서대로 잡는다. 두 run이면 두 호스트가 하나씩 동시에 학습한다. 빈 run이 없으면 CUDA 호스트가 Linux 쪽 run을 양보받아 이어 가고, CUDA 호스트를 멈추면 Linux 호스트가 마지막 체크포인트부터 다시 잡는다. 수동으로 넘길 필요 없다.
- 같은 run을 큐 밖에서 `bc --run`으로 직접 돌리지 않는다.

## 3. 준비 (호스트마다 한 번, 코드가 바뀌면 다시)

`.env`(저장소 루트, git 미추적)에 필요: `URANIUM_SHARED_DIR`, `URANIUM_MAPS_DIR`, `URANIUM_CACHE_DIR`, `URANIUM_AUTOTUNE_CACHE_DIR`.

```powershell
# CUDA 호스트 (Windows PowerShell)
git pull --ff-only
cargo build --release -p uranium-train -p uranium-runner -p uranium-agent --features uranium-train/cuda,uranium-runner/cuda,uranium-agent/cuda
# 새 창에서 CUDA_PATH가 비어 있으면(첫 update에서 libdevice 오류):
$env:CUDA_PATH = [Environment]::GetEnvironmentVariable('CUDA_PATH','Machine')
```
```bash
# Linux Vulkan 호스트
git pull --ff-only
cargo build --release -p uranium-train
```
- 작업자가 도는 중에는 다시 빌드하지 않는다. 먼저 작업자를 멈추고(Ctrl+C / `kill -INT <pid>`) 빌드한 뒤 다시 띄운다.

## 4. 코어 비교 (032-R17)

**① 큐 파일** `<shared>/queue/core-ablation.toml` (없으면 만든다)
```toml
[defaults]
datasets = ["replays/shards/stardata", "replays/shards/community", "replays/shards/scr_ladder", "replays/shards/personal"]
config = "config/bc_core_ablation.toml"

[[run]]
id = "core-abl-lstm"
core = "lstm"

[[run]]
id = "core-abl-gtrxl"
core = "gtrxl"
```

**② 작업자 실행**
```bash
# Linux Vulkan 호스트: 터미널을 닫아도 계속 돌게 (포그라운드 터미널에서 돌리지 않는다)
setsid nohup target/release/uranium-train worker --queue <shared>/queue/core-ablation.toml \
  --backend vulkan --priority 1 --log runs/worker.log > runs/worker.out 2>&1 &
# 멈출 때: kill -INT <worker pid>
```
```powershell
# CUDA 호스트: 쉬는 동안 실행, 쓸 때 Ctrl+C
target\release\uranium-train worker --queue <shared>/queue/core-ablation.toml --backend cuda --priority 10 --log runs/worker.log
```

**③ 진행 확인**
- 누가 무엇을 돌리는지: `<shared>/workers/<host>.json`(1분마다 갱신)
- 장비별 update 수·비율: `uranium-train devices --run core-abl-lstm` (gtrxl도 같이)
- 체크포인트: `<shared>/checkpoints/core-abl-*/` (1시간마다, 끝나면 `ckpt-5300`과 `DONE`)

**④ 둘 다 `DONE`이 생기면 평가** (CUDA 호스트)
```powershell
$DATA = "--dataset replays/shards/stardata --dataset replays/shards/community --dataset replays/shards/scr_ladder --dataset replays/shards/personal".Split(' ')
# 오프라인 지표: val 같은 500게임
target\release\uranium-train bc-eval @DATA --checkpoint <shared>/checkpoints/core-abl-lstm  --split val --games 500 --core lstm  --config config/bc_core_ablation.toml --backend cuda
target\release\uranium-train bc-eval @DATA --checkpoint <shared>/checkpoints/core-abl-gtrxl --split val --games 500 --core gtrxl --config config/bc_core_ablation.toml --backend cuda
# 대전: 맞대결 100판, T0 600판 (게임은 config/hosts.toml의 hostd에서. hostd가 없는 항목은 빼 둔다)
target\release\uranium-env run --plan config/plans/core-ablation-h2h.toml --backend cuda --out runs/core-ablation-h2h
target\release\uranium-env run --plan config/plans/core-ablation-t0.toml  --backend cuda --out runs/core-ablation-t0
```
- 결과는 `runs/core-ablation-h2h/summary.md`, `runs/core-ablation-t0/summary.md`, bc-eval JSON 출력.
- 보고: `reports/m6_core_ablation.md` 4절에 옮기고(장비별 update 비율 포함), **사용자가 코어를 고른다.**

**예상 기간** (실측 LSTM: CUDA 약 11초/update, Linux Vulkan 약 101초/update. GTrXL은 비슷하거나 조금 느림)
- CUDA 호스트가 계속 쉬면 두 run 합계 약 1.5일. Linux 호스트 혼자면 run당 약 6일. 실제는 CUDA 호스트를 쉬게 둔 시간에 따라 그 사이.

## 5. BC 본학습 (041) — 코어 선택 뒤

**큐 파일** `<shared>/queue/bc.toml` (`core`는 고른 코어로)
```toml
[defaults]
datasets = ["replays/shards/stardata", "replays/shards/community", "replays/shards/scr_ladder", "replays/shards/personal"]
config = "config/bc.toml"

[[run]]
id = "bc-s1"
core = "<lstm|gtrxl>"
stage = "s1"

[[run]]
id = "bc-s2"
core = "<lstm|gtrxl>"
stage = "s2"
init_run = "bc-s1"        # bc-s1에 DONE이 생긴 뒤에만 잡힌다

[[run]]
id = "bc-ablation-a"      # 비교용: 대상 맵만으로 처음부터 (041-AC1·AC3)
core = "<lstm|gtrxl>"
stage = "ablation-a"
```
- 작업자 명령은 4절 ②와 같고 `--queue <shared>/queue/bc.toml`만 바꾼다. Linux 서비스 파일 `tools/worker/uranium-worker.service`도 이 큐를 쓴다.
- 평가: `uranium-train bc-eval $DATA --checkpoint <shared>/checkpoints/bc-s2 --split test --backend cuda` (ablation-a도 같이), T0 대전 `config/plans/bc-vs-t0.toml`.

## 6. 남은 확인 (운영 중에 함께)

- [ ] 코어 비교 시작 (사용자)
- [ ] **Windows 실제 콘솔 Ctrl+C**: CUDA 호스트 작업자 창에서 Ctrl+C → `ckpt-N Pushed` → 종료 → Linux 호스트가 이어받는지 확인
- [ ] **Linux 호스트 메모리·스왑**: Linux 호스트가 학습하는 동안 `free -m`, `vmstat 5`를 몇 번 기록
- [ ] **24시간 연속 실행(060-AC3)**: 코어 비교 첫 24시간에 작업자·run이 멈추지 않았는지 `workers/<host>.json`·로그로 확인
- [ ] 코어 비교 결과 보고 → 코어 선택 → BC 본학습 큐 작성

## 7. 문제가 생기면

- **PC가 꺼져 잠금이 남음**: 같은 PC에서 다시 띄우면 이어 간다. 다른 PC 작업자는 heartbeat가 10분 멈춘 것을 보고 자동으로 인수한다.
- **재개가 거부됨**(설정·데이터 지문 다름): 설정이나 데이터가 바뀐 것이다. 같은 run을 억지로 잇지 말고 원인을 확인한다.
- **shard를 다시 만들었으면** `uranium-train manifest <shard 폴더>`로 manifest를 다시 만든다.
- 상세: `docs/commands.md` "감시·작업자", "BC" 절.

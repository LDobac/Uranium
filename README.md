# Uranium — 모방 학습(BC) 실행 안내

스타크래프트1 테란 봇. 이 문서는 **BC 학습을 돌릴 때 필요한 것만** 모았다. 옵션 상세는 `docs/commands.md`, 규칙은 `spec/`(032·041·060·061), 경위와 결과는 `reports/`에 있다. 학습·추론은 Python + PyTorch(`python -m uranium`), 데이터 공급·저장소·게임 연동은 Rust(`uranium._core`)다(2026-10-23 전환, `reports/pytorch_port.md`).

## 1. 순서

| 단계 | 내용 | 상태 |
|---|---|---|
| 0 | 데이터 재build·검증, 벤치, 대전 경로 시험 | 완료 (`reports/m6_rebuild.md` R-5) |
| 1 | **코어 비교**: LSTM·GTrXL을 S1 5,300 update씩 학습 → 평가 → 사용자가 코어 선택 | 완료 (2026-10-23, **GTrXL 채택**, `reports/m6_core_ablation.md` 4절) |
| 1.5 | **PyTorch 전환**: 학습·추론 이식 | 1단계 완료 (`reports/pytorch_port.md`), 2단계 최적화 다음 |
| 2 | **BC 본학습 S1**: 전체 맵, GTrXL. 39,096 update(2026-10-23 `bc-epoch`) | PyTorch로 새로 시작 |
| 3 | **BC S2**: 대상 맵(투혼·서킷브레이커), S1에서 시작 + 비교용 ablation (A) | 2 뒤 |
| 4 | 평가(오프라인 지표, T0 대전) → RL | 3 뒤 |

## 2. 장비 운영 규칙 (060-R8)

- **Linux ROCm 호스트**: 작업자를 24시간 띄워 둔다(우선순위 1).
- **CUDA 호스트**: 쉬는 동안만 사용자가 작업자를 직접 켠다(우선순위 10). PC를 써야 하면 **Ctrl+C** → `ckpt-N Pushed` 출력 후 종료될 때까지 기다린다(약 30초).
- 작업자는 큐에서 비어 있는 첫 run을 잡아 끝날 때까지 놓지 않는다. 잡을 run이 없을 때만 CUDA 호스트가 Linux 쪽 run을 양보받는다. CUDA 호스트를 멈추면 그 run은 다른 작업자가 비어 있을 때 마지막 체크포인트부터 다시 잡는다. 수동으로 넘길 필요 없다.
- 본학습 큐(4절, 2026-10-23 결정 I-D3 ②)에서는 CUDA 호스트가 S1, Linux 호스트가 ablation (A)를 잡는다. CUDA 호스트를 멈춘 동안 S1은 쉬고(Linux 호스트는 (A)를 계속), 다시 켜면 이어 간다. S1이 끝나면 CUDA 호스트가 S2를 잡고, 그것도 끝나면 Linux 호스트의 (A)를 양보받아 마무리한다. **CUDA 호스트 작업자를 먼저 켠다**(Linux 호스트가 먼저 켜지면 S1을 잡는다).
- 같은 run을 큐 밖에서 `python -m uranium bc --run`으로 직접 돌리지 않는다.

## 3. 준비 (호스트마다 한 번, 코드가 바뀌면 다시)

`.env`(저장소 루트, git 미추적)에 필요: `URANIUM_SHARED_DIR`, `URANIUM_MAPS_DIR`, `URANIUM_CACHE_DIR`, `URANIUM_PYTHON`(아래 venv의 python). Linux ROCm 호스트는 `MIOPEN_USER_DB_PATH`(고정 로컬 폴더)도.

학습 호스트는 `URANIUM_MAPS_DIR`에 build 맵 사전계산(v2)을 복사해 둔다. 없으면 `bc`가 시작 검사(B-2)에서 "map hash(es) ... have no map precompute v2"로 끝나고 작업자가 재시작을 반복한다.
```bash
cp -n <shared>/datasets/replays/build/maps/*.toml <URANIUM_MAPS_DIR>/     # Windows: Copy-Item
```

**Linux ROCm 호스트**(GPU와 RAM을 나눠 씀) `.env` 메모리 설정. 값을 바꾸지 않으므로 장비 사이 재개에 영향이 없다.
```bash
URANIUM_BC_PREFETCH=0        # 041-R16 미리 읽기 깊이 (2로 두면 데이터 대기가 사라지지만 RAM을 더 씀)
URANIUM_BC_ENTITY_CHUNK=16   # 041-R17 엔티티 인코더 조각 재계산
URANIUM_BC_PACK_BATCH=1      # 041-R18 호스트 batch 압축
```

**설치·갱신**(Python 3.14 venv + PyTorch 2.14 + `uranium` 패키지)
```powershell
# CUDA 호스트 (Windows PowerShell, uv는 scoop)
git pull --ff-only
uv venv --python 3.14 .venv                                   # 처음 한 번
uv pip install --python .venv\Scripts\python.exe "torch==2.14.*" --index-url https://download.pytorch.org/whl/cu130   # 처음 한 번
uv pip install --python .venv\Scripts\python.exe numpy safetensors pytest   # 처음 한 번
uv pip install --python .venv\Scripts\python.exe -e .          # 코드가 바뀔 때마다 (Rust 확장 빌드 포함)
cargo build --release -p uranium-train                         # 작업자·감시 바이너리
```
```bash
# Linux ROCm 호스트: 그 GPU용으로 빌드한 PyTorch가 든 venv에 설치
git pull --ff-only
uv pip install --python <venv>/bin/python -e .
cargo build --release -p uranium-train
```
- 작업자가 도는 중에는 다시 설치·빌드하지 않는다. 먼저 작업자를 멈추고(Ctrl+C / `kill -INT <pid>`) 갱신한 뒤 다시 띄운다.
- 평가·RL용 게임 호스트는 OpenBW·BWAPI(`tools/openbw/build.sh`)·브리지(`build_bridge.sh`)도 같은 커밋으로 다시 빌드하고 hostd를 재시작한다(Windows는 `.ps1`).

## 4. BC 본학습 (041)

**큐 파일** `<shared>/queue/bc.toml`. 코어는 **GTrXL**(2026-10-23 결정).
```toml
[defaults]
datasets = ["replays/shards/stardata", "replays/shards/community", "replays/shards/scr_ladder", "replays/shards/personal"]
config = "config/bc.toml"

[[run]]
id = "bc-s1-pt"
core = "gtrxl"
stage = "s1"

[[run]]
id = "bc-s2-pt"
core = "gtrxl"
stage = "s2"
init_run = "bc-s1-pt"     # bc-s1-pt에 DONE이 생긴 뒤에만 잡힌다

[[run]]
id = "bc-ablation-a-pt"   # 비교용: 대상 맵만으로 처음부터 (041-AC1·AC3)
core = "gtrxl"
stage = "ablation-a"
```

**작업자 실행**(자식 학습은 `URANIUM_PYTHON -m uranium bc --run ...`)
```bash
# Linux ROCm 호스트: 터미널을 닫아도 계속 돌게 (포그라운드 터미널에서 돌리지 않는다)
setsid nohup target/release/uranium-train worker --queue <shared>/queue/bc.toml \
  --backend rocm --priority 1 --log runs/worker.log > runs/worker.out 2>&1 &
# 멈출 때: kill -INT <worker pid>
```
```powershell
# CUDA 호스트: 쉬는 동안 실행, 쓸 때 Ctrl+C
target\release\uranium-train worker --queue <shared>/queue/bc.toml --backend cuda --priority 10 --log runs/worker.log
```
- Linux 서비스 파일 `tools/worker/uranium-worker.service`도 이 큐를 쓴다.

**진행 확인**
- 누가 무엇을 돌리는지: `<shared>/workers/<host>.json`(1분마다 갱신)
- 장비별 update 수·비율: `uranium-train devices --run bc-s1-pt`
- 체크포인트: `<shared>/checkpoints/bc-s1-pt/` 등(1시간마다, 끝나면 `DONE`)

**예상 속도**(PyTorch 1단계 실측, GTrXL, 32레인 × T64): Linux ROCm 호스트 약 20 s/update. CUDA 호스트 값은 `reports/pytorch_port.md`. 2단계 최적화 뒤 다시 잰다.

**평가**
- 오프라인 지표: `python -m uranium bc-eval $DATA --checkpoint <shared>/checkpoints/bc-s2-pt --split test`(ablation-a도 같이). 엔티티 조각 재계산 없이 돌므로 메모리가 큰 CUDA 호스트에서 돌린다.
- T0 대전 `python -m uranium run --plan config/plans/bc-vs-t0.toml`(plan의 체크포인트 경로를 새 run으로). 다른 호스트의 게임을 쓰려면 에이전트를 돌리는 PC의 `.env` `URANIUM_AGENT_LISTEN`/`URANIUM_AGENT_PUBLIC_ADDR`를 LAN 주소로 둔다(기본 127.0.0.1이면 다른 호스트 판이 모델 없이 0초에 끝난다). GTrXL은 큰 장면에서 추론이 무거우므로 plan의 `wall_timeout_s`를 넉넉히 둔다(코어 비교는 7,200초).

## 5. 코어 비교 (032-R17) — 완료

- 결과(2026-10-23): 맞대결 100판 GTrXL 42승·LSTM 13승·무 45, T0 상대 승률 GTrXL 약 10%(274판)·LSTM 2.7%(300판), val 500게임 모든 지표 GTrXL 우세 → **GTrXL 채택**. 상세·경위는 `reports/m6_core_ablation.md`.
## 6. 문제가 생기면

- **PC가 꺼져 잠금이 남음**: 같은 PC에서 다시 띄우면 이어 간다. 다른 PC 작업자는 heartbeat가 10분 멈춘 것을 보고 자동으로 인수한다.
- **재개가 거부됨**(설정·데이터 지문 다름): 설정이나 데이터가 바뀐 것이다. 같은 run을 억지로 잇지 말고 원인을 확인한다.
- **작업자가 시작하지 않음 "URANIUM_PYTHON is not set"**: `.env`에 venv의 python 경로를 적는다.
- **shard를 다시 만들었으면** `uranium-train manifest <shard 폴더>`로 manifest를 다시 만든다. manifest에 없는 shard를 출처 폴더에 두지 않는다(공유 제공 PC의 로컬 모드는 폴더 목록을 읽어 지문이 달라진다, 061-R12).
- 상세: `docs/commands.md` "감시·작업자", "BC" 절.

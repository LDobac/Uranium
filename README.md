# Uranium — 모방 학습(BC) 실행 안내

스타크래프트1 테란 봇. 이 문서는 **BC 학습을 돌릴 때 필요한 것만** 모았다. 옵션 상세는 `docs/commands.md`, 규칙은 `spec/`(032·041·060·061), 경위와 결과는 `reports/`에 있다. 본학습 착수 전 남은 결정·작업은 `reports/m6_pre_bc_review.md` I절.

## 1. 순서

| 단계 | 내용 | 상태 |
|---|---|---|
| 0 | 데이터 재build·검증, 벤치, 대전 경로 시험 | 완료 (`reports/m6_rebuild.md` R-5) |
| 1 | **코어 비교**: LSTM·GTrXL을 S1 5,300 update씩 학습 → 평가 → 사용자가 코어 선택 | 완료 (2026-10-23, **GTrXL 채택**, `reports/m6_core_ablation.md` 4절) |
| 2 | **BC 본학습 S1**: 전체 맵, GTrXL. update 수는 `bc-epoch`로 다시 계산(이전 값 38,584) | 착수 준비 (`m6_pre_bc_review.md` I절) |
| 3 | **BC S2**: 대상 맵(투혼·서킷브레이커), S1에서 시작 + 비교용 ablation (A) | 2 뒤 |
| 4 | 평가(오프라인 지표, T0 대전) → RL | 3 뒤 |

## 2. 장비 운영 규칙 (060-R8)

- **Linux Vulkan 호스트**: 작업자를 24시간 띄워 둔다(우선순위 1).
- **CUDA 호스트**: 쉬는 동안만 사용자가 작업자를 직접 켠다(우선순위 10). PC를 써야 하면 **Ctrl+C** → `ckpt-N Pushed` 출력 후 종료될 때까지 기다린다(약 30초). 2026-10-21 코어 비교에서 이 흐름으로 Linux 호스트가 이어받는 것을 확인했다.
- 작업자는 큐에서 끝나지 않은 run을 순서대로 잡는다. 두 run이면 두 호스트가 하나씩 동시에 학습한다. 빈 run이 없으면 CUDA 호스트가 Linux 쪽 run을 양보받아 이어 가고, CUDA 호스트를 멈추면 Linux 호스트가 마지막 체크포인트부터 다시 잡는다. 수동으로 넘길 필요 없다.
- 같은 run을 큐 밖에서 `bc --run`으로 직접 돌리지 않는다.

## 3. 준비 (호스트마다 한 번, 코드가 바뀌면 다시)

`.env`(저장소 루트, git 미추적)에 필요: `URANIUM_SHARED_DIR`, `URANIUM_MAPS_DIR`, `URANIUM_CACHE_DIR`, `URANIUM_AUTOTUNE_CACHE_DIR`.

학습 호스트는 `URANIUM_MAPS_DIR`에 build 맵 사전계산(v2)을 복사해 둔다. 없으면 `bc`가 시작 검사(B-2)에서 "map hash(es) ... have no map precompute v2"로 끝나고 작업자가 재시작을 반복한다. 재build로 사전계산이 바뀌었으면 다시 복사한다.
```bash
cp -n <shared>/datasets/replays/build/maps/*.toml <URANIUM_MAPS_DIR>/     # Windows: Copy-Item
```

**GTrXL을 Linux Vulkan 호스트에서 학습할 때** `.env`에 메모리 설정을 둔다(GPU와 RAM을 나눠 쓰는 호스트, 2026-10-20 실측). 없으면 커널 OOM이나 엔티티 512개 장면의 2 GiB GPU 할당 실패로 멈춘다. 설정 지문은 바뀌지 않으므로 장비 사이 재개에 영향이 없다.
```bash
URANIUM_BC_PREFETCH=0        # 041-R16 미리 읽기 깊이
URANIUM_BC_ENTITY_CHUNK=16   # 041-R17 엔티티 인코더 조각 재계산
URANIUM_BC_PACK_BATCH=1      # 041-R18 호스트 batch 압축
```

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
- 평가·RL용 게임 호스트는 OpenBW·BWAPI(`tools/openbw/build.sh`)·브리지(`build_bridge.sh`)도 같은 커밋으로 다시 빌드하고 hostd를 재시작한다(Windows는 `.ps1`).

## 4. BC 본학습 (041)

**큐 파일** `<shared>/queue/bc.toml`. 코어는 **GTrXL**이다(2026-10-23 결정). `--core` 기본값은 아직 `lstm`이므로 run마다 `core = "gtrxl"`을 꼭 적는다.
```toml
[defaults]
datasets = ["replays/shards/stardata", "replays/shards/community", "replays/shards/scr_ladder", "replays/shards/personal"]
config = "config/bc.toml"

[[run]]
id = "bc-s1"
core = "gtrxl"
stage = "s1"

[[run]]
id = "bc-s2"
core = "gtrxl"
stage = "s2"
init_run = "bc-s1"        # bc-s1에 DONE이 생긴 뒤에만 잡힌다

[[run]]
id = "bc-ablation-a"      # 비교용: 대상 맵만으로 처음부터 (041-AC1·AC3)
core = "gtrxl"
stage = "ablation-a"
```

**작업자 실행**
```bash
# Linux Vulkan 호스트: 터미널을 닫아도 계속 돌게 (포그라운드 터미널에서 돌리지 않는다)
setsid nohup target/release/uranium-train worker --queue <shared>/queue/bc.toml \
  --backend vulkan --priority 1 --log runs/worker.log > runs/worker.out 2>&1 &
# 멈출 때: kill -INT <worker pid>
```
```powershell
# CUDA 호스트: 쉬는 동안 실행, 쓸 때 Ctrl+C
target\release\uranium-train worker --queue <shared>/queue/bc.toml --backend cuda --priority 10 --log runs/worker.log
```
- Linux 서비스 파일 `tools/worker/uranium-worker.service`도 이 큐를 쓴다.

**진행 확인**
- 누가 무엇을 돌리는지: `<shared>/workers/<host>.json`(1분마다 갱신)
- 장비별 update 수·비율: `uranium-train devices --run bc-s1`
- 체크포인트: `<shared>/checkpoints/bc-s1/` 등(1시간마다, 끝나면 `DONE`)

**예상 기간**(코어 비교 실측, GTrXL): CUDA 호스트 약 12 s/update, Linux Vulkan 호스트 약 108 s/update. S1 약 38,600 update는 CUDA 호스트가 쉬지 않으면 약 5.4일, Linux 호스트만이면 약 48일이다. S2·(A)는 각 약 11,300 update(CUDA 약 1.6일씩). 실제 기간은 CUDA 호스트를 학습에 쓸 수 있는 시간에 거의 비례한다.

**평가**
- 오프라인 지표는 **CUDA 호스트에서** 돌린다: `uranium-train bc-eval $DATA --checkpoint <shared>/checkpoints/bc-s2 --split test --core gtrxl --backend cuda`(ablation-a도 같이). Linux Vulkan 호스트에서는 엔티티 512개 장면에서 2 GiB 할당에 실패한다(엔티티 조각 재계산이 `bc-eval`에는 없음).
- T0 대전 `config/plans/bc-vs-t0.toml`. 다른 호스트의 게임을 쓰려면 에이전트를 돌리는 PC의 `.env` `URANIUM_AGENT_LISTEN`/`URANIUM_AGENT_PUBLIC_ADDR`를 LAN 주소로 둔다(기본 127.0.0.1이면 다른 호스트 판이 모델 없이 0초에 끝난다). GTrXL은 큰 장면에서 추론이 무거우므로 plan의 `wall_timeout_s`를 넉넉히 둔다(코어 비교는 7,200초).

## 5. 코어 비교 (032-R17) — 완료

- 결과(2026-10-23): 맞대결 100판 GTrXL 42승·LSTM 13승·무 45, T0 상대 승률 GTrXL 약 10%(274판)·LSTM 2.7%(300판), val 500게임 모든 지표 GTrXL 우세 → **GTrXL 채택**. 상세·경위는 `reports/m6_core_ablation.md`.
- 체크포인트 `<shared>/checkpoints/core-abl-{lstm,gtrxl}/`, 대전·채점 출력과 리플레이 `<shared>/artifacts/m6_core_ablation/final/`(전부 보존).
- 다시 돌릴 때: 큐 `<shared>/queue/core-ablation.toml`(`config/bc_core_ablation.toml`), 평가는 `final/run_final.ps1`(맞대결 → T0 → val 채점, 고정 빌드·plan 사용)을 참고한다.

## 6. 남은 확인

- [x] 코어 비교 완료·코어 선택(GTrXL, 2026-10-23)
- [x] Windows 실제 콘솔 Ctrl+C(2026-10-21 코어 비교 중 확인)
- [x] Linux 호스트 메모리·스왑(041-R16~R18 설정으로 해결)
- [x] 24시간 연속 실행(060-AC3, 코어 비교 LSTM run: CUDA 오류 1회 → 자동 재시작 → 완주)
- [ ] 본학습 착수 전 결정·작업(데이터 동결 시점, `--core` 기본값, 학습 장비 재빌드, `[steps]` 재계산, 표본 end-to-end, 큐 작성): `reports/m6_pre_bc_review.md` I절

## 7. 문제가 생기면

- **PC가 꺼져 잠금이 남음**: 같은 PC에서 다시 띄우면 이어 간다. 다른 PC 작업자는 heartbeat가 10분 멈춘 것을 보고 자동으로 인수한다.
- **재개가 거부됨**(설정·데이터 지문 다름): 설정이나 데이터가 바뀐 것이다. 같은 run을 억지로 잇지 말고 원인을 확인한다.
- **shard를 다시 만들었으면** `uranium-train manifest <shard 폴더>`로 manifest를 다시 만든다. manifest에 없는 shard를 출처 폴더에 두지 않는다(공유 제공 PC의 로컬 모드는 폴더 목록을 읽어 지문이 달라진다, 061-R12).
- 상세: `docs/commands.md` "감시·작업자", "BC" 절.

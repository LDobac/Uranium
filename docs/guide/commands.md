# 명령어 상세

옵션·동작·운영 주의. 요구사항은 괄호 안 spec ID.

목차: [1 설치·설정](#1-설치설정) · [2 벤치·프로파일](#2-벤치프로파일) · [3 게임 실행](#3-게임-실행) · [4 교사·DAgger·증류 데이터](#4-교사dagger증류-데이터) · [5 분기 게임·경로 측정](#5-분기-게임경로-측정) · [6 리플레이](#6-리플레이) · [7 BC](#7-bc) · [8 감시·작업자](#8-감시작업자) · [9 RL](#9-rl) · [10 단서 진단](#10-단서-진단) · [11 계획 JEPA S0](#11-계획-jepa-s0) · [12 래더 수집](#12-래더-수집) · [13 Windows](#13-windows) · [14 여러 호스트](#14-여러-호스트) · [15 문제 해결](#15-문제-해결)

## 1. 설치·설정

### .env·작업 파일
- `cp .env.example .env` 후 값 입력. 새 키는 `.env.example`에도. 프로세스 환경변수 우선.
- Rust 바이너리는 시작 시 읽음, 셸 스크립트는 `tools/env.sh`(Windows `tools\env.ps1`). Windows 경로는 작은따옴표.
- `URANIUM_ENV_FILE`(프로세스 환경변수로 지정): `<repo>/.env` 대신 그 파일을 읽음. Rust 바이너리·`python -m uranium`·셸 스크립트 공통, 파일이 없으면 아무것도 안 읽음([001-R9](../spec/2026_10_15_22_24_001-system-architecture.md)).
- 학습 호스트 필수: `URANIUM_SHARED_DIR`, `URANIUM_MAPS_DIR`, `URANIUM_CACHE_DIR`, `URANIUM_PYTHON`(venv python, 작업자·감시가 학습 자식을 띄울 때 씀). Linux ROCm 호스트는 `MIOPEN_USER_DB_PATH`(고정 로컬 폴더)도.
- 작업 로그·생성 파일·일회용 스크립트: `<repo>/local/work/<작업>/`. 호스트별 스크립트: `<repo>/local/scripts/`. 게임 데이터: `<repo>/local/sc_data/`(`openbw/`, `maps/`).

### Git LFS
- `releases/**/model.safetensors`는 Git LFS. 장비마다 한 번 `git lfs install`, 받은 뒤 포인터만 있으면 `git lfs pull`. 가중치를 안 쓰는 장비는 포인터로 둬도 됨.

### Python·PyTorch ([AMB-000-4](../spec/README.md), [001-R8](../spec/2026_10_15_22_24_001-system-architecture.md))
- 학습·추론은 Python 3.14 + PyTorch 2.14(`python/uranium`). `uranium` 패키지 = Python + Rust 확장 `uranium._core`(crate `uranium-py`, maturin). Rust를 고치면 같은 설치 명령으로 다시 설치.
- 호스트마다 venv → torch(Linux ROCm은 그 GPU용 빌드) → 저장소 루트에서 `-e .` 설치. 명령은 아래 블록.

```powershell
# CUDA 호스트 (Windows PowerShell, uv는 scoop)
git pull --ff-only
uv venv --python 3.14 .venv                                   # 처음 한 번
uv pip install --python .venv\Scripts\python.exe "torch==2.14.*" --index-url https://download.pytorch.org/whl/cu130   # 처음 한 번
uv pip install --python .venv\Scripts\python.exe numpy safetensors pytest   # 처음 한 번
uv pip install --python .venv\Scripts\python.exe -e ".[compile]"   # 코드가 바뀔 때마다 (Rust 확장 빌드 포함, [compile] = triton-windows: CUDA 기본 torch.compile에 필요)
cargo build --release -p uranium-train                         # 작업자·감시 바이너리
```
```bash
# Linux ROCm 호스트: 그 GPU용으로 빌드한 PyTorch가 든 venv에 설치
git pull --ff-only
uv pip install --python <venv>/bin/python -e .
cargo build --release -p uranium-train
```
- 작업자가 도는 중에는 재설치·빌드 금지. 먼저 멈추고(Ctrl+C / `kill -INT <pid>`) 갱신 → 다시 띄움([8 감시·작업자](#8-감시작업자) "코드 교체").
- 평가·RL용 게임 호스트는 OpenBW·BWAPI·브리지도 같은 커밋으로 다시 빌드하고 hostd 재시작.
- 직접 빌드한 torch 보호: torchvision·torchaudio 등 torch 의존 패키지는 공식 torch를 끌어옴 → 소스 빌드 또는 `--no-deps`(GPU 연산자는 따로 빌드). 설치 뒤 `python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"`로 버전·GPU 아키텍처 확인. `uranium`은 torch에 의존하지 않음, `[compile]`(triton-windows)은 Windows에서만 설치.
- Linux ROCm 호스트 실행 환경: 그 호스트 rocBLAS 경로를 `LD_LIBRARY_PATH`에, `HSA_ENABLE_SDMA=0`.
- Linux ROCm 호스트의 ROCm 패키지 버전이 바뀌면([rocm_10_1](../report/investigation/2026_11_17_16_27_rocm_10_1.md)): 그 GPU용 rocBLAS를 같은 버전 소스로 다시 빌드, torch도 그 ROCm에 맞춰 다시 빌드(같은 소스 커밋·옵션, 한 호스트에서 빌드해 wheel 복사, 절차 리포트 9절). torch를 아직 다시 빌드하지 않은 동안만(다른 MIOpen 버전으로 빌드된 torch) `.env`에 `PYTORCH_SKIP_CUDNN_COMPATIBILITY_CHECK=1`(pytest는 `.env`를 읽지 않으므로 셸에 export). 직접 쓴 커널 캐시는 torch 빌드 때의 HIP 버전으로 구분돼 torch를 다시 빌드하면 새로 빌드됨. 확인: `-m gpu` 테스트, `bench suite`, `bench bc`.
- 정밀도([032-R8](../spec/2026_10_15_22_24_032-model-architecture.md)): CUDA bf16 혼합, ROCm f32. 커널 선택([041-R15](../spec/2026_10_15_22_24_041-imitation-learning.md)): cuDNN benchmark는 CUDA만, ROCm은 MIOpen find-db(`MIOPEN_USER_DB_PATH`)만(ROCm benchmark는 모양마다 수 초~수 분 탐색).
- Burn: 마지막 커밋 태그 `burn-final`. Burn 체크포인트(`model.bpk`)는 보관만, 읽지 않음([061-R10](../spec/2026_10_15_22_24_061-shared-storage.md)).

### OpenBW·브리지·T1 봇 ([spec 011](../spec/2026_10_15_22_24_011-bridge-openbw.md), [spec 021](../spec/2026_10_15_22_24_021-opponent-pool.md))
- OpenBW/BWAPI: `tools/openbw/build.sh` → `third_party/bwapi/build/bin/BWAPILauncher`. 패치 `third_party/patches/`(0001~0015) 순서대로. OpenBW·BWAPI·T1 봇의 URL·커밋·sha256·패치 이름은 `tools/pins.txt`(`<키> <값>` 줄, 두 셸이 같이 읽음).
- 브리지: `tools/openbw/build_bridge.sh` → `bridge/openbw/build/liburanium_bridge.so`. shim 500줄 이하. shim·Rust 쪽은 C ABI라 늘 함께 빌드. 브리지 .so·에이전트·hostd는 모든 PC에서 같은 커밋으로 함께 빌드([010-R12](../spec/2026_10_15_22_24_010-game-protocol.md) 프로토콜 버전 2), 옛 shim과 새 라이브러리 섞지 않음.
- Windows: `tools\openbw\build.ps1` → `tools\openbw\build_bridge.ps1` → `tools\oss\build_<bot>.ps1`([13 Windows](#13-windows)).
- T1 봇: `tools/oss/build_<bot>.sh`(`ualbertabot`, `steamhammer`, `zzzkbot`, `stardust`, `mcrave`, `iron`) → `<URANIUM_OSS_BOTS_DIR>/<bot>/{module.so, VERSION}`. 상대 ID `<bot>@<버전>`([021-R3](../spec/2026_10_15_22_24_021-opponent-pool.md)).
  - `-DNDEBUG`(원본 Release·Windows 빌드와 같음). 플래그가 바뀌면 오브젝트 재생성(`build/obj/<bot>/.flags`).
  - 원본을 다시 빌드하면 모든 설정 변형 폴더(`<bot><tag>/`)에 새 `module.so` 복사 + 원본과 sha256 일치 확인(변형 = 설정 파일만 다른 원본 복사본).
- T1 설정 변형(spec 021 "T1 설정 변형"): 원본 빌드·배포 뒤 `python3 -I tools/oss/make_variant.py --bot <bot> --variant <tag> --file <설정 파일> (--replace <이전> <이후> | --lines A-B --text <새 줄>)` → `<URANIUM_OSS_BOTS_DIR>/<bot><tag>/`, `VERSION` `<bot><tag>@<원본 버전>-<tag>`.
  - 바이너리 해시 일치 확인, [oss_bot_porting](../report/survey/2026_10_16_00_18_oss_bot_porting.md)에 적을 값(파일·줄·이전 → 이후·sha256) 출력.
  - 게임 호스트마다 같은 인자로(Windows는 venv python). 예시는 스크립트 머리말.

### 게임 데이터·표
- 추출([011-R13](../spec/2026_10_15_22_24_011-bridge-openbw.md)): `cargo install casc --version 0.1.6 --locked` 후 `cargo run -p casc-extract -- <SC:R Data> local/sc_data/openbw` → `.env` `URANIUM_DATA_DIR`.
- BWAPI 타입 표([013-R1](../spec/2026_10_15_22_24_013-version-mapping.md)): `cargo run -p bwapi-tables -- third_party/bwapi/bwapi crates/uranium-env/tables`.
- 맵 사전계산(v2): 학습 호스트는 build 산출물을 `URANIUM_MAPS_DIR`에 복사. 없으면 `bc` 시작 검사에서 "map hash(es) ... have no map precompute v2"로 끝나고 작업자가 재시작 반복.
```bash
cp -n <shared>/datasets/replays/build/maps/*.toml <URANIUM_MAPS_DIR>/     # Windows: Copy-Item
```

### 구현 스위치 (AMB-000-4, [032-R19](../spec/2026_10_15_22_24_032-model-architecture.md))
- 같은 수식, 장비별 빠른 쪽. 근거 [pytorch_opt](../report/bench/2026_10_24_07_03_pytorch_opt.md).
- 미지정이면 백엔드 기본값(`python/uranium/device.py` `IMPL_DEFAULTS`). 표의 "학습" = BC 학습 진입점(`bc`·`bench bc`·`bc-profile`)에서만 쓰는 `TRAIN_DEFAULTS`(추론 프로세스는 컴파일이 느려지는 스위치를 켜지 않음). 환경변수·`.env`가 이김.
- 적용 값은 체크포인트 메타 `kernels`에 남음(041-R15). 층 모양이 가정과 다르면 그 최적화는 자동으로 라이브러리 경로.

| 키 | 내용 | CUDA 호스트 | Linux ROCm 호스트 |
|---|---|---|---|
| `URANIUM_LOC_ROWS` | 위치 헤드 사용 행: 호스트 라벨에서, 버킷 | `pow2` | `exact` |
| `URANIUM_LOC_OUT` | 위치 헤드 1×1 출력 | `conv` | `mulsum` |
| `URANIUM_BIAS_GRAD` | 거리 bias gather·표 gradient | `embedding` | `table_t` |
| `URANIUM_CONV_WGRAD` | stem·stride-2·64채널 이하 ResBlock conv 가중치 gradient | `vendor` | `bmm` |
| `URANIUM_LINEAR_WGRAD` | Linear 가중치 gradient(f32만) | `vendor` | `splitk` |
| `URANIUM_ENTITY_ATTN` | 엔티티 attention 직접 작성 커널([060-R1](../spec/2026_10_15_22_24_060-infra.md)) | `fused` | `fused` |
| `URANIUM_INFER_GRAPH` | 추론 `act`를 CUDA/HIP graph로 | `on` | `on` |
| `URANIUM_INFER_GRAPH_MAX_BATCH` | graph 최대 배치(넘으면 eager) | 제한 없음 | `4`([060-R3](../spec/2026_10_15_22_24_060-infra.md) 추론 2.5 GiB) |
| `URANIUM_INFER_COMPILE` | 추론 `act` torch.compile(그래프 안) | `default` | `off` |
| `URANIUM_H2D` | 호스트→GPU 복사를 page-locked에서 비동기로(BC는 collate 스레드에서 준비) | 학습 `pinned` | `sync` |
| `URANIUM_CHANNELS_LAST` | 공간 인코더·위치 헤드 activation·conv 가중치 channels-last | 학습 `on` | `off` |
| `URANIUM_SPATIAL_BF16` | 관측 평면을 호스트에서 bf16으로(bf16 autocast만, 비트 동일) | 학습 `on` | `off`(f32) |
| `URANIUM_STEM_PAD` | stem conv 입력 채널을 8의 배수로 0 채움 | 학습 `on` | `off` |
| `URANIUM_BC_PAD_ENTITIES` | BC 학습 엔티티 패딩 단위(엔티티 상한 512의 약수; 평가·RL·추론은 설정값 그대로; CUDA에서 바꾸면 폭 종류가 늘어 torch.compile graph가 늘 수 있음, [041](../spec/2026_10_15_22_24_041-imitation-learning.md) 구현 메모) | 설정값(64) | 학습 `32` |
| `URANIUM_COMPILE` | BC forward torch.compile(Triton 필요); `reduce-overhead` = CUDA graph 재생 | 학습 `reduce-overhead` | `off` |
| `URANIUM_RL_COMPILE` | RL 학습기 forward·KL 참조 forward torch.compile(값은 `URANIUM_COMPILE`과 같음, Triton 필요). graph 모드(`reduce-overhead`)는 바깥 LSTM 모델의 실게임 RL에서 추론 서버를 오래 멈출 수 있어 권장하지 않음 | `off` | `off` |
| `URANIUM_RL_CHANNELS_LAST` | RL 학습기·KL 참조 모델만 channels-last + stem 채움(추론 서버 모델은 그대로) | `off` | `off` |
| `URANIUM_ADAM`, `URANIUM_COMPILE_DYNAMIC`, `URANIUM_COMPILE_RNG`, `URANIUM_INFER_STREAM` | 기본값 유지(`INFER_STREAM=own`은 Linux ROCm 호스트에서 멈춤, `COMPILE_DYNAMIC=on`은 재컴파일 못 없앰) | - | - |

- 장비별 기본값 다시 고르기(모델 구조 변경 뒤 등): `python -m uranium bench bc --groups 1 --updates 12`를 스위치 값만 바꿔 비교, 손실 동등성은 `bench loss`(30 update).
- 직접 작성 커널: 첫 사용 때 빌드해 `URANIUM_KERNEL_DIR`(기본 `~/.cache/uranium/kernels`). Windows는 `cl`이 PATH에 없으면 최신 Visual Studio vcvars64를 불러옴.

### 호스트 메모리 설정 (`.env`)
- CUDA 호스트(RAM을 평소 작업과 나눠 씀): `URANIUM_BC_MICRO_LANES=2`, `URANIUM_BC_PREFETCH=1`, `URANIUM_BC_PACK_BATCH=1`, `URANIUM_BC_FILE_AHEAD=2`.
- Linux ROCm 호스트(GPU와 RAM 공유): 아래. 결과 값을 바꾸지 않으므로 장비 사이 재개에 영향 없음. 근거 [m6_core_ablation_pt](../report/experiment/2026_10_24_21_29_m6_core_ablation_pt.md) 4절, [pytorch_opt](../report/bench/2026_10_24_07_03_pytorch_opt.md).
```bash
URANIUM_BC_PREFETCH=2                    # 041-R16 미리 읽기 깊이 (데이터 대기 0)
URANIUM_BC_ENTITY_CHUNK=0                # 041-R17 엔티티 재계산 끔 (AMB-041-8, 2026-10-24)
URANIUM_BC_PACK_BATCH=1                  # 041-R18 호스트 batch 압축
URANIUM_MALLOC_MMAP_THRESHOLD=1048576    # 다 쓴 batch 메모리를 OS에 돌려줌 (상주 10 → 6.3 GiB)
URANIUM_MALLOC_ARENA_MAX=2               # glibc arena 수 제한 (긴 run에서 update마다 늘던 메모리를 막음)
URANIUM_BC_MICRO_LANES=1                 # 041-R3 레인 그룹 (AMB-041-7)
URANIUM_BC_FEED_THREADS=2                # 공급 스레드 (이 호스트는 3이면 학습과 CPU를 다툼)
```
- 각 키 설명은 [7 BC](#7-bc) "메모리·공급 환경변수".

## 2. 벤치·프로파일
- 벤치 스위트([060-R2](../spec/2026_10_15_22_24_060-infra.md)): `python -m uranium bench suite [--suite kernels,model,infer] [--filter <regex>] [--precision f32|bf16] [--budget 10] [--out docs/report/bench/<YYYY_MM_DD_HH_mm>_bench_<backend>.md] [--json local/work/bench/<name>.json]`
  - 커널·모델·추론 케이스 forward / forward+backward 중앙값·p95, CPU f32 대비 오차, 최대 GPU 할당.
  - 반복은 표준오차 < 평균 1%까지(워밍업 제외, 케이스당 `--budget`초까지).
- 실데이터 BC update(060-R2, [AMB-041-7](../spec/README.md)): `python -m uranium bench bc (--dataset <rel> | --shards <dir>)... [--groups 1,2,4,8] [--updates 20] [--seed 1] [--phases] [--ops 25] [--json <log>]`
  - 그룹 크기마다 같은 초기 가중치·batch로 update 시간, 데이터 대기, 최대 GPU 할당·상주 메모리([060-R3](../spec/2026_10_15_22_24_060-infra.md)), update별 손실. 첫 update는 워밍업(중앙값 제외).
- 최적화 동등성([AMB-000-4](../spec/README.md)): 기준·후보를 같은 `--seed`·데이터로 `bench bc --updates 100 --json` → `python -m uranium bench loss <기준.json> <후보.json> [--group N] [--window 20]`(구간 평균 손실 상대 차이. 그룹 크기가 다르면 dropout 난수열이 다름).
- BC 구간별·연산별 시간([060-AC4](../spec/2026_10_15_22_24_060-infra.md)): `python -m uranium bc-profile (--dataset <rel> | --shards <dir>)... [--backend cuda|rocm|cpu] [--core gtrxl|lstm] [--batches 3] [--ops 25] [--trace <chrome trace json>]`. 학습 중에는 `URANIUM_BC_PROFILE=1`로 update마다 구간 시간 로그.
- 장치 비교([032-AC1](../spec/2026_10_15_22_24_032-model-architecture.md)·AC8): `pytest -m gpu tests/python`(CPU f32 대비, CUDA bf16은 잠정 허용치).
- RL learner 벤치: [9 RL](#9-rl).

## 3. 게임 실행

### hostd·스케줄러 ([spec 020](../spec/2026_10_15_22_24_020-env-orchestration.md))
- hostd: `uranium-hostd serve`. 설정 키 `.env.example` hostd 절.
- 동시 게임 수([020-R10](../spec/2026_10_15_22_24_020-env-orchestration.md)): 게임 호스트에서 `uranium-env calibrate` → `<URANIUM_CACHE_DIR>/hostd_calibration.toml`(`URANIUM_HOSTD_MAX_GAMES` 없을 때 사용).
- 스케줄러(crate `uranium-runner`, 바이너리 `uranium-env`):
  - 스크립트 플레이어만: `uranium-env run --plan config/plans/<plan>.toml [--out runs/<dir>]`
  - 모델 플레이어 포함(추론은 Python, [AMB-000-4](../spec/README.md)): `python -m uranium run --plan <plan> [--out runs/<dir>] [--backend cuda|rocm|cpu] [--max-batch 32] [--max-wait-ms 5]`(백엔드 기본은 그 PC의 GPU).
  - 출력: `games.jsonl`, `results.jsonl`, `summary.md`(Wilson 95%, 정책별 합산), `sessions.md`. 호스트 목록 `URANIUM_HOSTS_FILE`(예시 `config/hosts.example.toml`).
- hostd·스케줄러는 같은 커밋으로 빌드(제어 프로토콜 v3).
- 멈춘 Python 프로세스(Linux): `kill -USR1 <pid>` → 모든 스레드 스택을 그 프로세스 stderr(작업자·학습 로그)에 쓰고 계속 돔(root 불필요).

### plan 키
- 모델 정책([020-R11](../spec/2026_10_15_22_24_020-env-orchestration.md)): `{ kind = "model", checkpoint = "<URANIUM_SHARED_DIR 기준 경로>", sample = "argmax"|"stochastic" }`. Burn `model.bpk`는 읽지 않음([061-R10](../spec/2026_10_15_22_24_061-shared-storage.md)).
- `save_replay = true` → `<out>/replays/g<id>.rep`([020-R1](../spec/2026_10_15_22_24_020-env-orchestration.md), hostd와 같은 커밋).
- `surrender = "off" | "shadow" | "on"`(기본 `shadow`), `surrender_chat`(기본 `"gg"`, 빈 문자열 = 채팅 없이 퇴장, 79바이트 이하): 모델 GG 항복([031-R12](../spec/2026_10_15_22_24_031-action-space.md), [020-R14](../spec/2026_10_15_22_24_020-env-orchestration.md)).
  - `shadow`: 판정만 기록, summary Model sessions 표에 판정 성립 세션 수·그중 승리 수(오항복).
  - `on`: 판정 성립 시 채팅 1초 뒤 퇴장(패배). win_open 헤드 있는 체크포인트 필요([032-R18](../spec/2026_10_15_22_24_032-model-architecture.md)).
  - 에이전트 서버 단독: `--surrender on --surrender-chat gg`. 테스트 전용 `URANIUM_TEST_SURRENDER_FRAME=<프레임>`: `on`에서 그 프레임 뒤 첫 결정에 무조건 항복.
- `zero_last_failed = true`: `last_failed` 입력 0 고정([030-R10](../spec/2026_10_15_22_24_030-observation.md)). BC 체크포인트 평가 plan(코어 비교, bc-vs-t0)에서 켜고 RL은 끔. 단독 실행은 `--zero-last-failed`.
- 맵 사전계산 형식 `version = 2`(자원 목록 포함, [spec 030](../spec/2026_10_15_22_24_030-observation.md)). 옛 `<hash>.toml`·`<hash>.resources.toml`은 없는 것으로 봄 → 재build·`uranium-replay mapgen`. 대상 맵 사전계산 `uranium-replay mapgen <맵 파일>...`([AMB-031-4](../spec/README.md)).
- 플레이어 이름([020-R16](../spec/2026_10_15_22_24_020-env-orchestration.md)): 리플레이에서 `p<i>-<run 폴더 이름>`(모델), `p<i>-T0-<종족>`, OSS 봇 이름(24바이트 자름). plan 플레이어 `name`으로 바꿈.
- 예시 plan: 스모크 `config/plans/smoke-model.toml`(체크포인트 `python -m uranium random-checkpoint --out <dir> [--core lstm|gtrxl] [--win-open]`), [041-AC2](../spec/2026_10_15_22_24_041-imitation-learning.md) `config/plans/bc-vs-t0.toml`.

### 평가 작업 eval-worker ([020-R15](../spec/2026_10_15_22_24_020-env-orchestration.md))
긴 평가를 추론 장비가 빠졌다 합류하며 이어 감.
- 공용 드라이브 `eval/<job>/job.toml`: `stages = ["plans/01-<plan>.toml", ...]`(차례대로), `hosts = "hosts.toml"`(둘 다 job 폴더 기준) + plan 사본·hosts 파일.
- hosts 파일은 모든 작업자가 같이 읽음 → LAN 주소로(127.0.0.1은 작업자마다 다른 PC, 시작 때 경고). 1분마다 다시 읽고 끊긴 호스트도 재확인.
- 장비마다: `python -m uranium eval-worker --job eval/<job> --priority <n> [--backend cuda|rocm] [--local-hostd <hosts 파일의 이 PC 주소> --hostd <uranium-hostd 경로>]`
  - CUDA 호스트(Windows): `tools\eval\eval_worker.ps1 -Job eval/<job> -LocalHostd <hosts 파일의 이 PC 주소>`(우선순위 기본 10). hostd를 별도 콘솔 창으로 띄워(Ctrl+C 안 닿음) 판을 다 마친 뒤 끔. 로그 `--log <파일>`(스크립트는 `runs\eval-worker.log`).
  - Linux ROCm 호스트 대기 작업자: ROCm 실행 환경(`LD_LIBRARY_PATH`·`HSA_ENABLE_SDMA=0`)으로 `--backend rocm --priority 2`처럼.
- 판 배정 권한(`locks/eval-<job>.json`, [061-R16](../spec/2026_10_15_22_24_061-shared-storage.md))을 쥔 작업자만 배정·추론, 모델도 그동안만 올림. 더 높은 우선순위가 오면 새 배정을 멈추고 권한을 넘긴 뒤 진행 중 판은 마침(판 도중 추론 장비 안 바뀜).
- 권한을 받으면 모델마다 모든 (배치, 엔티티 수) 모양 워밍업 뒤 첫 판(`--no-warmup`으로 끔). 공용 드라이브를 못 읽는 동안은 새 배정만 멈춤.
- Ctrl+C 1회: 진행 중 판 기록 후 종료(그 PC hostd는 `leaving/` 표시로 새 판 안 받음). 2회: 진행 중 판 `StopGame` → 같은 seed로 재실행 예약 후 종료. 3회: 즉시. 다시 실행하면 합류.
- `--hostd` 작업자는 끝날 때 그 hostd의 판(다른 작업자 판 포함)이 0이 될 때까지 기다린 뒤 hostd 끔(Linux hostd 출력 `<log>.hostd`).
- 다시 하는 판: 브리지 미접속([020-R4](../spec/2026_10_15_22_24_020-env-orchestration.md)), hostd 끊김, 강제 정지. 시간 초과·crash는 결과로 남음. hostd 판 번호 `시도 × 10,000,000 + 단계 × 100,000 + 판 번호`. 재시도 시도가 `max_attempts`(`job.toml`, 기본 3, [AMB-020-6](../spec/README.md))에 이르면 결과 없이 끝내고 경고.
- 기록: `games/<host>.jsonl`(판마다 `infer_host`·`backend`)·`results/<host>.jsonl`, 진행 중 `inflight/<host>.json`, 리플레이 `<stage>/replays/g<판 번호>.rep`, 작업자 상태 `workers/<host>.eval.json`(학습 작업자 파일과 별도).
- 집계: `python -m uranium eval-status --job eval/<job> [--no-write]` → 단계별 진행 표, `<stage>/games.jsonl`·`results.jsonl`(plan 판 번호, 마지막 시도만)·`summary.md`(추론 backend별 판 수 포함)·`sessions.md` 재작성. 리그전 `tools/ablation/league_table.py --run <job>/<stage> <plan>`.

### 러시·교사 점검 plan (050-R6, [050-R8](../spec/2026_10_15_22_24_050-evaluation.md))
- 조건 통제 러시 점검([050-R6](../spec/2026_10_15_22_24_050-evaluation.md), [020-R20](../spec/2026_10_15_22_24_020-env-orchestration.md)·R21): `python tools/eval/rush_check_plan.py` → `config/plans/rush5_check.toml`(128판, 생성기 시드 목록). 점검 job용 모델 교체 plan: `python tools/eval/rush_check_plan.py --checkpoint <ckpt>/agents/main --name <단계 이름> --out eval/<job>/plans/01-rush.toml`(venv python, `uranium._core` 필요).
- 정찰 반응형 테란 교사(050-R8, [021-R8](../spec/2026_10_15_22_24_021-opponent-pool.md)): `uranium-env run --plan <plan> --out runs/<dir>`로(스크립트·OSS 봇만, GPU 불필요, 게임 호스트에 `URANIUM_MAPS_DIR` 사전계산 필요 — 대상 맵에서 없으면 세션 거부)
  - `config/plans/teacher_check.toml`(050-R6과 같은 128판·시드, p0 = T0 `terran_react`, `python tools/eval/rush_check_plan.py --teacher`로 재작성)
  - `config/plans/teacher_macro_check.toml`(운영형 4상대 × 10판), `config/plans/teacher_t0zerg_check.toml`(T0 표준 저그 10판)
  - T0 테란 세션 `results.jsonl` 행에 `scripted`(키: [050 "T0 테란 세션 지표"](../spec/2026_10_15_22_24_050-evaluation.md), 교사 `react`는 021-R8 기록).
  - `summary.md` "T0 Terran sessions" 표: matchup·맵별 막음(`scv_min_1m > 4`)·승·4:00 전 벙커·선제 벙커·①②④′ 발화·4:00까지 확장·8:00 SCV ≥ 16/≥ 14·첫 규칙(상대별 기준 050-R8).
  - 적중·빗나감 조건별 집계: plan 시드로 생성기(`uranium._core`)를 돌려 나눔(`scripted` 행).

### 테란·프로토스전 교사 ([spec 022](../spec/2026_11_14_08_18_022-matchup-teacher.md), [021-R9](../spec/2026_10_15_22_24_021-opponent-pool.md))
- plan `p0 = { kind = "t0", race = "terran", build = "terran_tp" }`(상대 ID `t0-terran-tp`). 로비 상대가 저그면 그 세션은 021-R8 교사(`terran_react`)([022-R2](../spec/2026_11_14_08_18_022-matchup-teacher.md)).
- 맵 사전계산은 021-R8 교사와 같이 읽음. 결과 `scripted.tp`([022-R10](../spec/2026_11_14_08_18_022-matchup-teacher.md): 첫 R 시각·규칙, 리파이너리·팩토리·머신샵·시즈 연구·첫 탱크·앞마당 지시 시각, 첫 웨이브, 끝 가지).
- 022 교사 판 구분은 `t0_build`가 아니라 `scripted.tp` 유무(로비 저그 판은 `t0_build = "terran_tp"`여도 `scripted.react`만).
- 점검([022-R11](../spec/2026_11_14_08_18_022-matchup-teacher.md)·[022-AC2](../spec/2026_11_14_08_18_022-matchup-teacher.md)): `python tools/eval/rush_check_plan.py --teacher-tp` → `config/plans/teacher_tp_check.toml`(교사 대 T0 표준 프로토스·테란 2맵 × 32 = 128판 + T1 UAlbertaBot P·Stardust·UAlbertaBot T·Iron 2맵 × 10 = 80판(기록만), 60분 상한, 리플레이 저장). 실행 `uranium-env run --plan config/plans/teacher_tp_check.toml --out <폴더>`(T1은 [050-R9](../spec/2026_10_15_22_24_050-evaluation.md) NDEBUG 규칙). 합격 프로토스·테란 각 ≥ 44/64, 한 번에 잼(개발 조정은 다른 seed).
- botgen·보충·재레이블·t1: [4 교사·DAgger·증류 데이터](#4-교사dagger증류-데이터).

### 에이전트 서버·기타
- 에이전트 서버: 스크립트 봇 `uranium-agent serve --listen 127.0.0.1:7600 --out runs/x.jsonl`, 모델 `python -m uranium serve --listen 127.0.0.1:7600 --out runs/x.jsonl --policy model --checkpoint <dir> [--backend <b>] [--sample argmax]`, 요약 `uranium-agent summarize <jsonl> --out <md>`.
- 셀프매치: `tools/openbw/selfmatch.sh <map> <games> [parallel] [seed_base] [agent args...]`. `SAME_SEED=1` = 같은 seed, `URANIUM_MAX_FRAMES` = 길이 제한.
- OpenBW LAN 매칭([011-R16](../spec/2026_10_15_22_24_011-bridge-openbw.md)): Linux `$XDG_RUNTIME_DIR/uranium/` LOCAL 소켓(경로 108바이트 제한), Windows `URANIUM_HOSTD_PORT_BASE`부터 TCP 포트 쌍.
- SC:R 실전([spec 012](../spec/2026_10_15_22_24_012-bridge-remastered.md), 설치·절차 [scr.md](scr.md)): `tools\scr\build.ps1`(테스트·빌드·모드 폴더 설치) → `python -m uranium scr --checkpoint <ckpt> [--backend cuda] [--sample stochastic] [--listen 127.0.0.1:7700] [--out runs/scr/sessions.jsonl]`(realtime: Act 기한 관측 + 24프레임 [012-R7](../spec/2026_10_15_22_24_012-bridge-remastered.md), `last_failed` 0 [012-R14](../spec/2026_10_15_22_24_012-bridge-remastered.md), 항복 기본 `on` [012-R11](../spec/2026_10_15_22_24_012-bridge-remastered.md)) → `tools\scr\run.ps1`(samase64 해시 확인 후 실행) → 게임 시작은 사람.
- 액션 시나리오([031-AC1](../spec/2026_10_15_22_24_031-action-space.md)): hostd를 띄운 뒤 `URANIUM_TEST_HOSTD=<hostd 주소> [URANIUM_MAPS_DIR=<투혼 자원 파일이 있는 폴더>] cargo test -p uranium-env --test openbw_actions -- --ignored --nocapture`
  - 사전계산은 라이브 에이전트 규칙으로 읽음(없으면 CC 자리 규칙이 BWAPI와 다를 수 있음, `uranium-replay mapgen <투혼 .scx>`).
  - 투혼에서 스크립트 테란 p0가 [034](../spec/2026_10_15_22_24_034-action-table.md)의 101개 action_type을 실제 마스크·executor로 한 번 이상 실행, `CommandFailed` 없음 확인. 허용 거절: 유닛이 선 건물 자리 Unbuildable_Location, 한 Act 여러 지출 명령 자원 부족. 시야 밖 한 번 본 미네랄 Smart_Unit·Gather_Unit([011-R20](../spec/2026_10_15_22_24_011-bridge-openbw.md)), 비무장 유닛 Attack_Pos → Move([AMB-041-1](../spec/README.md))도 확인. p1은 가만히 있는 테란.
  - 다 덮으면 `StopGame`(게임 약 19분, 벽시계 약 1분). 출력 `target/tmp/openbw_actions_031_ac1.{md,rep}`.
  - hostd가 다른 PC면 `URANIUM_TEST_AGENT_LISTEN=0.0.0.0:0`과 `URANIUM_AGENT_PUBLIC_ADDR`.

## 4. 교사·DAgger·증류 데이터
공통: plan 생성기는 venv python. build 전 브리지·`uranium-replay`는 같은 커밋([011-R10](../spec/2026_10_15_22_24_011-bridge-openbw.md)). records 이어 하기는 같은 폴더의 `<출처>*.jsonl`을 모두 읽으므로 다시 build하려면 새 records 폴더·새 shard 폴더.

### 교사 데이터 절단·021-R8 교사 botgen
- 절단([040-R21](../spec/2026_10_15_22_24_040-replay-dataset.md)): `uranium-replay build <dir> --src botgen --records <jsonl> --until-frame 11520` → 8:00(11,520 프레임) 뒤 스텝은 안 씀(레코드 `until_frame`).
- `--time-limit`: 길이가 plan `max_frames` 이상인 판(시간 상한 퇴장)은 [040-R8](../spec/2026_10_15_22_24_040-replay-dataset.md) ③ 디싱크 신호를 보지 않고 판정 불가(`result` 0·`result_source` `none`)(040-R21).
- 교사 botgen([040-R14](../spec/2026_10_15_22_24_040-replay-dataset.md)·[050-R8](../spec/2026_10_15_22_24_050-evaluation.md)): `python tools/eval/rush_check_plan.py --botgen` → `config/plans/teacher_botgen_rush.toml`(러시 4상대 × 2맵 × 200)·`teacher_botgen_macro.toml`(운영형 5상대 × 2맵 × 40, 판 번호 해시 시드).
  1. `uranium-env run --plan <plan> --out <폴더>`
  2. 리플레이·plan·결과를 `<shared>/datasets/replays/raw/botgen/teacher-<날짜>/<plan>/`로
  3. `uranium-replay build <그 폴더> --src botgen --out <shared>/datasets/replays/shards/botgen/teacher-<날짜> --records <shared>/datasets/replays/records/botgen_teacher-<날짜>.jsonl --until-frame 11520 --time-limit 14400`(`URANIUM_MAPS_DIR` = `datasets/replays/build/maps`)
  4. `uranium-train manifest <shared>/datasets/replays/shards/botgen`

### [022](../spec/2026_11_14_08_18_022-matchup-teacher.md) 교사 botgen·보충
- botgen plan([022-R12](../spec/2026_11_14_08_18_022-matchup-teacher.md)·[044-R8](../spec/2026_11_14_08_18_044-self-game-distillation.md), 점검 통과 뒤): `python tools/eval/rush_check_plan.py --botgen-tp` → `config/plans/teacher_botgen_tp.toml`(점검과 같은 상대, T0 2맵 × 150 + T1 2맵 × 25 = 800판, 10분 상한).
- build는 plan(stage) 폴더 하나씩: `uranium-replay build <폴더>/replays --src botgen --games <폴더>/games.jsonl --until-frame 14400 --time-limit 14400 --out <shared>/datasets/replays/shards/botgen/teacher-tp-<날짜>`
  - `--games`: 교사(plan players[0], hostd 이름 `p0-...`) 시점만 데이터(TvT 상대 테란 시점 제외), 상대 로비 종족은 games.jsonl에서. players[0]가 hostd 이름 `p0-T0-terran-terran_tp`인 022 교사가 아닌 판(plan `name` 덮어쓰기 포함)은 거부. `--games` 없는 botgen build는 022 교사 판을 `filtered`.
  - 교사 + dagger 안 종족 비율(저그 : 프로토스 : 테란 = 50 : 30 : 20)은 학습 혼합이 정함(plan에 없음).
- seed(두 plan, [022-R2](../spec/2026_11_14_08_18_022-matchup-teacher.md) 종족 고정·Random 없음): `splitmix64(1,000,000 + n) >> 32`, 점검 n = 1..208, botgen 1,001..1,800. 서로·`--out-dir` 아래 모든 plan(`protocol/` 포함)·botgen·점검 seed·아직 안 만든 DAgger·증류 라운드 블록(1..10, 증류 보충)·교사 라운드 t1 블록·022 botgen 보충 블록·교사 개발 seed 77,001..77,264와 겹치면 생성기 멈춤.
- 보충(044-R8·022-R12): t1 `mix-fractions --teacher-races`가 프로토스·테란을 `short`로 내면(교사 + dagger 안 30%·20%에 ± 3%p 아래) 저그를 줄이지 않고 모자란 종족 상대 022 botgen 판만 더 둠.
  - `python tools/eval/rush_check_plan.py --botgen-tp --extra-p N --extra-t M --part K` → `config/plans/teacher_botgen_tp_x<K>.toml`(plan 이름 `teacher-botgen-tp-x<K>`, 프로토스 N판·테란 M판, 종족마다 022-R12 구성 비율대로 × N / 400·× M / 400). N·M은 16의 배수, 한 part N + M ≤ 992. 값은 `teacher_races.extra_botgen_tp`, part 나눔은 `after_top_up.parts`.
  - seed: 같은 키의 판 번호 10,000 + (K − 1) × 1,000 + i(K = 1..10, 보충마다 새 K), 같은 겹침 검사, 이미 있는 보충 plan은 다른 seed로 덮어쓰지 않음.
  - build는 botgen plan과 같이(`--src botgen --games … --until-frame 14400 --time-limit 14400`) 같은 `botgen/teacher-tp-<날짜>` 출처로. `filtered` 비율만큼 더 둠.

### DAgger ([041-R22](../spec/2026_10_15_22_24_041-imitation-learning.md), 출처 `dagger`)
- 판 생성 plan: `python tools/eval/rush_check_plan.py --dagger --round R [--half] [--checkpoint <ckpt>]` → `config/plans/dagger_r<R>.toml`(기본 모델 `checkpoints/bc-mix-e10/ckpt-1053` stochastic, 10분 상한, 리플레이 저장).
  - 기본 러시 4상대 × 2맵 × 40 + 운영형 5상대 × 2맵 × 28 = 600판, `--half` × 20·× 14 = 300판.
  - seed 판 번호는 라운드마다 600 폭: 라운드 R은 n = (R − 1) × 600 + 1부터(2라운드 `--half` 601..900). 다른 plan(`--out-dir` 안 전부)과 겹침 검사.
- 기억 기록 `--memory-log`(모델 2 경로 [041-R29](../spec/2026_10_15_22_24_041-imitation-learning.md) ④, [041-R28](../spec/2026_10_15_22_24_041-imitation-learning.md) ①): plan에 `memory_log = true`. 기억에 기록하는 결정마다 (라이브 관측 프레임, action, delay bucket, 선택 유닛 id)를 `memory/g<id>.jsonl`(평가 job은 `<stage>/memory/`, `replays/g<id>.rep` 옆, 다시 둔 판은 덮어씀)에. 끝 줄(결정 수·마지막 프레임·자기 플레이어 id)은 `GameEnd` 때만(오류·끊김이면 없음 → 재레이블 incomplete). 기본 꺼짐, `save_replay` 필요, 판마다 모델 플레이어 하나.
- 재레이블: `uranium-replay build <폴더> --src dagger --relabel terran_react --out <shared>/datasets/replays/shards/dagger/<라운드> --records <shared>/datasets/replays/records/dagger_<라운드>.jsonl --until-frame 11520 --time-limit 14400`
  - 레이블 = 테란 시점 관측에 그림자로 돌린 [021-R8](../spec/2026_10_15_22_24_021-opponent-pool.md) 교사 명령(라이브 프레임 L의 재생 프레임 L − 2를 먹이고 명령은 L + 2). 모델 명령은 디싱크 검사만. 5분 미만 판도 씀. records에 `relabel`.
- 기억 붙이기 `--memory-log <판 생성 폴더>/memory`(041-R28, 재레이블 없는 build에도 가능 → 분기 줄기): 스텝마다 [040-R22](../spec/2026_10_15_22_24_040-replay-dataset.md) `memory_action`(shard schema 7).
  - 스텝 i 관측(재생 프레임 obs_i) = 라이브 live_i = obs_i + 2(게임 시작 1은 1). live_{i−1} ≤ L < live_i인 모델 결정이 순서대로(그 관측에서 내린 결정은 다음 스텝).
  - 판 전체 비움: 기록 없음, 끝 줄 없음, 끝 줄 플레이어가 이 시점 아님(`other_player`), 끝 프레임이 리플레이 길이와 8프레임 넘게 다름, 선택 유닛이 재생 프레임(L − 2) 자기 유닛 아닌 결정 > 1%. records `memory`: `status` ok·missing·incomplete·other_player·unit_mismatch, `reason`, `decisions`·`checked`·`mismatched`.
  - 리플레이 파일 이름으로 기록을 찾음 → 같은 이름 리플레이가 둘이면 거부: plan(stage) 폴더 하나씩 build.
  - records에 남은 시점은 상태와 무관하게 끝난 것으로 봄 → 고쳐 다시 build하면 새 records·shard 폴더.
- ignored 테스트: `URANIUM_TEST_DAGGER_REP=<모델 판 .rep> cargo test -p uranium-replay --lib relabel_is_deterministic -- --ignored`(두 번 같음·교사 액션만), `URANIUM_TEST_TEACHER_REP=<교사 botgen 판 .rep> cargo test -p uranium-replay --lib relabel_of_a_teacher -- --ignored --nocapture`(botgen 레이블 일치율, `URANIUM_TEST_VERBOSE=1`이면 차이 목록).
- 모델 2 경로 라운드(041-R29 ④):
  - plan `python tools/eval/rush_check_plan.py --dagger --round R [--half] --memory-log --checkpoint <ckpt> --out config/plans/dagger_m2_r<R>.toml`. seed는 Checkpoint 1 경로 같은 라운드와 같음(겹침 검사에서 같은 라운드 `dagger_r<R>.toml`만 뺌 → 이 파일이 생긴 뒤 `dagger_r<R>.toml`을 다시 만들려면 `--out-dir`을 따로).
  - 판 생성 job의 `<stage>/{replays,memory}`를 `<shared>/datasets/replays/raw/dagger_m2/r<R>/`로 복사.
  - 재레이블 `--out <shared>/datasets/replays/shards/dagger_m2/r<R> --records <shared>/datasets/replays/records/dagger_m2/dagger_r<R>.jsonl --memory-log <raw>/memory`(records 폴더를 따로 둬 Checkpoint 1 records와 분리) → `uranium-train manifest <shared>/datasets/replays/shards/dagger_m2`.
  - 학습 설정 `config/bc_m2_dagger_r<R>.toml`(`chain_start_sources = ["botgen"]`, `keep_sources = ["botgen", "dagger"]`: shard 출처 태그는 `dagger`). 몫 `uranium-train mix-fractions --teacher-share <몫>`(r1 실현 몫 0.5954).
  - `mix-fractions`는 교사(+ dagger) 몫을 `--teacher-share`(없으면 0.5) ± 3%로 판정(출력 `s2_mix.teacher_share_target`; `bc-epoch`의 `s2_mix`는 늘 0.5 기준).
  - 데이터: 사람 4출처 + `botgen` + `dagger_m2`만(`replays/shards/dagger` 제외).

### 그림자 재레이블 terran_tp ([022-R9](../spec/2026_11_14_08_18_022-matchup-teacher.md), 041-R22)
- `uranium-replay build <모델 판 폴더> --src dagger --relabel terran_tp --games <판 생성 폴더>/games.jsonl ...`(`--games` 필수, 나머지는 `terran_react` 재레이블과 같음, shard `dagger_m2/t<R>`).
- 모델(players[0]) 시점만(TvT 상대 시점 제외). 상대 로비 종족은 리플레이 헤더가 아니라 games.jsonl.
- 재생 없이 `filtered`(이유 `lobby Zerg opponent ...`·`lobby Random opponent ...`·`no games.jsonl line ...` 등): 저그([022-AC3](../spec/2026_11_14_08_18_022-matchup-teacher.md))·Random(022-R2, 저그로 나온 판 포함), games.jsonl 줄 없음, 로비·헤더 종족 다름, players[0] 이름(`p0-...`·`uranium-p0`)이 정확히 하나가 아님, players[0] 로비 종족이 테란 아님.
- 게임 id는 파일 이름(`g<id>.rep`, `g<id>-s<seed>-p0.rep`)에서 → 같은 id 두 파일이면 시작 안 함(plan(stage) 폴더 하나씩).
- ignored 테스트: `URANIUM_TEST_TEACHER_TP_REP=<022 교사 botgen 판 .rep> cargo test -p uranium-replay --lib relabel_of_a_tp_teacher -- --ignored --nocapture`(botgen 레이블 일치율 ≥ 95%, 022-AC3).

### 자기 판 증류 ([spec 044](../spec/2026_11_14_08_18_044-self-game-distillation.md), [041-R31](../spec/2026_10_15_22_24_041-imitation-learning.md))
- 판 생성 plan([044-R1](../spec/2026_11_14_08_18_044-self-game-distillation.md)): `python tools/eval/rush_check_plan.py --distill --round R [--half] --checkpoint <ckpt>` → `config/plans/distill_m2_d<R>.toml`(plan 이름 `distill-m2-d<R>`). 기본 체크포인트 없음(d1 = 041-R29 ④ 선택 S, d2 = d1 모델).
  - 모델 stochastic 대 T0 표준 프로토스 2맵 × 120 + T0 표준 테란 2맵 × 60 + T1 UAlbertaBot P·Stardust·UAlbertaBot T·Iron 각 2맵 × 30 = 600판, `--half`(d2) 블록마다 절반 = 300판.
  - 종족 고정, 10분 상한, 리플레이 저장, `memory_log` 늘 켬(`--memory-log`는 받아도 같음). T1은 [050-R9](../spec/2026_10_15_22_24_050-evaluation.md) T1 규칙(NDEBUG 해시, 상대 크래시 판 무효).
  - seed `splitmix64(80,000 + n) >> 32`, 라운드마다 600 폭(d2 `--half` n = 601..900). botgen·점검·dagger seed, `--out-dir` 아래 하위 폴더까지 모든 plan(`config/plans/protocol/` 050-R9 템플릿, 특히 `t0_tp`)과 겹치면 멈춤.
- 결정 스텝 build([044-R2](../spec/2026_11_14_08_18_044-self-game-distillation.md), [040-R24](../spec/2026_10_15_22_24_040-replay-dataset.md)): `uranium-replay build <raw>/replays --src distill --memory-log <raw>/memory --until-frame 14400 --time-limit 14400 --out <shared>/datasets/replays/distill_pre/d<R> --records <records 폴더>/distill_d<R>.jsonl`
  - 스텝 = 게임 시작 관측 + 기록된 모델 결정마다 하나(재생 프레임 L − 2, NoOp 포함). 리플레이 명령은 040-R8 검사만. 라이브 1·3 결정은 둘 다 재생 1이라 한 스텝(보통 스텝 수 = 결정 수 − 1).
  - 레이블 칸 모두 `unknown`([044-R3](../spec/2026_11_14_08_18_044-self-game-distillation.md) 레이블러가 채움). 스텝 i `memory_action` = [obs_{i−1}, obs_i)의 결정(자기 결정은 다음 스텝 기억).
  - 041-R28 ④·⑤ 실패 시점(기록 없음·불완전·다른 플레이어(TvT 상대)·유닛 불일치)은 `filtered`(사유 `decision_log <status> …`).
  - 거부: `--memory-log`·BC shard 폴더 밖 `--out`·`--time-limit`·이름이 `distill`로 시작하는 records 중 하나라도 없음. `stage` B, APM·5분 기준 없음. records `memory.decision_steps` = 결정 프레임 스텝 수, `coverage`는 리플레이(모델) 명령 기준.
  - 레이블러와의 약속: 스텝 식별자 (`meta.replay_sha256`, `meta.perspective_slot`, `raw_frame.frame`), 판 안 프레임은 겹치지 않고 증가(첫 스텝 1, 라이브 = 프레임 + 2, 프레임 1은 라이브 1). 레이블러는 `action_label` 전체(delay 포함)만 바꾸고 `action_label.frame == raw_frame.frame` 유지, `raw_frame`·`privileged`·`memory_action`·`meta`(`src` = `distill`, `stage` = B, 결과) 그대로. 모든 스텝에 `memory_action` 있음.
  - d1 같은 큰 배치 전 표본 build로([044-R12](../spec/2026_11_14_08_18_044-self-game-distillation.md)): `URANIUM_TEST_DISTILL_SHARD=<shard_*.bin.zst> URANIUM_TEST_DISTILL_MEMORY=<memory 폴더> URANIUM_TEST_DISTILL_REPLAYS=<replays 폴더> cargo test -p uranium-replay --lib distill_shard_matches_its_logs -- --ignored`(테스트 `distill_shard_matches_its_logs_044_r2`: 스텝 프레임·`unknown` 칸·기억이 결정 기록과 맞는지).
- 정책 레이블(044-R3, 040-R24): `python -m uranium distill-label --checkpoint <shared>/checkpoints/m2-s2e2/ckpt-11387 --mode sample --label-seed <s> --in <shared>/datasets/replays/distill_pre/d<R> --out <shared>/datasets/replays/shards/distill_m2/d<R> --records <records 폴더>/label_d<R>.jsonl [--config config/bc_m2.toml] [--backend cuda]` → `uranium-train manifest <shared>/datasets/replays/shards/distill_m2`
  - 입력은 BC 로더(`uranium._core`)와 같음([030](../spec/2026_10_15_22_24_030-observation.md) feature·[034](../spec/2026_10_15_22_24_034-action-table.md) 마스크, 관측 전 `memory_action`으로 기억 기록·레이블은 기억에 안 씀, 맵 사전계산, `last_failed` 0). z·대칭·종족 가림·체인 시작 관측 없음, 시점 첫 스텝부터 core 상태 이음.
  - `--mode sample`: 스텝마다 분포에서 하나(온도 1, 헤드 순서·마스크는 라이브와 같음). 잡음은 (시점 seed, 스텝 프레임, 헤드 표집 순번, 열) 정수 해시(`uranium.distill.noise`, `label_policy.noise` = `rowhash-1`), 시점 seed는 `_core.distill_game_seed`(SHA-256 `044-R3/<replay_sha256>/<perspective_slot>/<label seed>`).
  - 레인 수·`micro_lanes`·`pad_entities`·`seq_len`과 무관하게 같은 레이블(CPU 비트 동일; GPU는 같은 배치 모양일 때만).
  - `--mode argmax`는 도구 옵션: BC shard 폴더 안 `--out` 거부.
  - 헤드 값 → [040](../spec/2026_10_15_22_24_040-replay-dataset.md) 레이블(delay = 버킷 하한, 선택 = 고른 순서 유닛 id, 대상 유닛 id, 위치 = 셀의 타일). 이어서 [040-R19](../spec/2026_10_15_22_24_040-replay-dataset.md) 무효 20% 필터(build와 같은 함수). 입력 파일마다 같은 상대 경로로 [040-R17](../spec/2026_10_15_22_24_040-replay-dataset.md) 검증 복사(`URANIUM_SHARD_STAGING_DIR`), records 줄 덧붙임.
  - records: 시점마다 한 줄(`status` written·filtered, 040-R19 수, `label_policy`: 체크포인트 경로·sha256, 방식, 레이블·시점 seed, 장치·정밀도, git 커밋, 무효 수, 설정·지문). 요약은 표준 출력과 `<records 이름>.summary.json`.
  - records 이름이 `distill`로 시작하면 거부. 모양이 build records와 달라(`status`) `dataset-stats`에 넣지 않음.
  - 새 records면 `--out`에 shard가 없어야 함. 기존 records를 주면 이어 함([044-R5](../spec/2026_11_14_08_18_044-self-game-distillation.md) ③ 보충, 멈춘 run): records에 있는 입력 파일은 건너뛰고 같은 체크포인트·방식·레이블 seed로(다르면 오류). 배치 모양·장치는 달라도 됨. 출력 파일이 이미 있는데 records 줄이 없으면 지우고 다시.
  - 입력 shard 머리(schema·feature·action·레이블러 버전)가 지금 build와 다르면 거부, 범위 밖 헤드 값은 오류. 보충 뒤 manifest 재생성.
- 학습 몫(044-R5): 설정 `config/bc_m2_distill_d<R>.toml`(`data.seed` d1 = 15·d2 = 16, `keep_sources = ["botgen", "dagger"]`, 증류 출처 `distill`은 `[data.subsample.fraction_by_source] distill`). 세기:
  `uranium-train mix-fractions --config config/bc_m2_distill_d<R>.toml --share teacher=0.45,distill=0.2 --dataset-group replays/shards/stardata,replays/shards/community,replays/shards/scr_ladder,replays/shards/personal --dataset-group replays/shards/botgen,replays/shards/dagger_m2,replays/shards/distill_m2`
  - 교사 + dagger 전부(epoch = 교사 스텝 / 0.45), 증류는 epoch 20%에 가장 가까운 게임 수(모자라면 전부 = 1.0), 사람은 나머지 35%를 [041-R26](../spec/2026_10_15_22_24_041-imitation-learning.md) 종족 비율로.
  - 실패(0 아닌 코드): 세 몫(각 ± 3%p, `s2_mix.teacher_ok`·`s2_mix.sources.distill.ok`·`s2_mix.human.ok`)이나 사람 종족 비율(`race_ok`) 중 하나(증류 부족 포함).
  - 출력 `fractions`(사람 `fraction`·`fraction_by_race`), `source_fractions.fraction_by_source`, `s2.updates`·`s2_warmup`을 설정에 옮김. 자리 표시(0) 설정은 로더가 거부 → 세기는 `fraction_by_race`·`fraction_by_source` 두 표를 뺀 사본으로(증류 실행 스크립트 몫).
  - `--share` 없이 `fraction_by_source` 있는 설정은 거부. `bc-epoch`의 `s2_mix`는 교사 0.5 기준이라 증류 설정에서 `teacher_ok = false`(판정은 `mix-fractions --share`).
- 증류 부족 보충(044-R5 ③): 교사 + dagger를 줄이거나 작은 몫을 받지 않고 같은 라운드 증류 판을 더 둠.
  - `mix-fractions --share`가 ± 3%p 아래로 실패한 출처마다 `source_fractions.short.<출처>`(`missing_steps`, `steps_per_game`, `selected_share`, `extra_games` = 올림(모자란 스텝 ÷ 판당 스텝 ÷ 선택 몫)) + stderr 안내. 목표 아래지만 범위 안이면 `below_target`(참고, 보충 안 함). build `filtered` 비율만큼 더.
  - `python tools/eval/rush_check_plan.py --distill --round R --extra N --part K --checkpoint <그 라운드 모델>`(N은 20의 배수로 올림, 044-R1 구성 비율, `config/plans/distill_m2_d<R>_x<K>.toml`, seed 게임 번호 100,000 + (R − 1) × 10,000 + (K − 1) × 1,000 + i, 라운드 판(n 1..6,000)·다른 plan과 겹침 검사).
  - 같은 라운드 `raw/distill/d<R>/`·`distill_pre/d<R>`·`shards/distill_m2/d<R>`로 build·레이블 → 몫 다시 셈.

### 교사 라운드 t1 (044-R8)
- 증류 라운드 뒤(d1 뒤, d2를 했으면 d2 뒤) 한 번, [044-R6](../spec/2026_11_14_08_18_044-self-game-distillation.md) 규칙으로 판정.
- 자기 판 plan(022-R12): `python tools/eval/rush_check_plan.py --teacher-round --checkpoint <그 라운드 모델>` → `config/plans/dagger_m2_t1.toml`(plan 이름 `dagger-m2-t1`, `--round`·`--half` 없음).
  - T0 표준 프로토스 2맵 × 60 + T0 표준 테란 2맵 × 30 + T1 UAlbertaBot P·Stardust·UAlbertaBot T·Iron 2맵 × 15 = 300판. 종족 고정, 10분 상한, 리플레이 저장, `memory_log` 늘 켬, `zero_last_failed = false`, 기본 체크포인트 없음.
  - seed `splitmix64(3,000,000 + n) >> 32`, n = 1..300(블록 1..600 예약). 모든 plan·botgen·점검·DAgger·증류 블록·증류 보충·022 botgen 보충·교사 개발 seed와 겹치면 멈춤(`--teacher-tp`·`--botgen-tp`도 이 블록을 피함).
  - build: 그림자 재레이블(`--games <판 생성 폴더>/games.jsonl`, `--memory-log <판 생성 폴더>/memory`, `--until-frame 14400 --time-limit 14400`) → `dagger_m2/t1`.
- 몫: 증류 없음(교사 + dagger 65 / 사람 35). `uranium-train mix-fractions --config <t1 설정> --teacher-share 0.65 --teacher-races Z=0.5,P=0.3,T=0.2`. 데이터 사람 4출처 + `botgen`(021-R8·022 교사) + `dagger_m2`(r1·r2·r3·t1), `distill_m2` 제외(`--teacher-races`를 `--share` 없이 쓰면 증류 게임 있는 데이터 거부).
  - `--teacher-races`(022-R12, [AMB-022-7](../spec/README.md); 세 종족·합 1, `--share`와 병용 가능): 교사 + dagger 게임을 줄이지 않고 그 안 종족별 몫 판정(`s2_mix.teacher_races`, 각 ± 3%p).
  - 출력 `teacher_races`: 종족별·출처별 스텝, `anchor`(몫 대비 스텝 최다 종족)와 기준 목표 스텝, 실패 종족 `short.<종족>`(모자란 스텝, 판당 스텝·선택 몫, 판 수 추정; 범위 안 미달은 `below_target`), 프로토스·테란 부족 시 `extra_botgen_tp`(종족별 보충 판 수, 16의 배수 올림, 다른 종족 보충 뒤 범위 밖으로 밀리는 종족 포함). stderr 안내(`--botgen-tp --extra-p N --extra-t M`) + 0 아닌 코드.
  - 보충은 위 `--botgen-tp --extra-p N --extra-t M --part K` → build → 다시 셈. `teacher_races.after_top_up` = 보충 뒤 예상 스텝·몫·판정, `--part` 나눔(한 번 992판까지).
  - `teacher_races.warnings`·stderr `WARNING`(보충해도 ± 3%p 불가, 992판 초과, 저그 부족(프로토스·테란 보충으로 못 올림)): 그대로 두지 말고 보고.
- 학습 설정 `config/bc_m2_teacher_t1.toml`(자리 표시: 시작 = 마지막 증류 라운드의 마지막 체크포인트, `data.seed` 16(d2를 했으면 17), 몫·`[steps]`은 위 세기로, 이전 증류 데이터 제외).
- 판정 `python tools/eval/distill_verdict.py --round t1 ...`(`pass`·`stop`, S = r2 `m2-dagger-r2/ckpt-1408`의 job), 측정 지점 `t1`(run `m2-teacher-t1`). 통과하면 t1이 D(RL 출발점), 지점 `D` = run `m2-teacher-t1`.

## 5. 분기 게임·경로 측정

### 분기 게임 줄기 판 build ([051-R13](../spec/2026_11_12_12_39_051-branch-games.md) ④)
- `uranium-replay build <job>/<단계>/replays --src branch --out <shared>/eval/branch-<날짜>/shards/trunks --records <shared>/eval/branch-<날짜>/build/branch_build.jsonl --memory-log <job>/<단계>/acts --time-limit 11568`(재레이블 없음, 레이블은 리플레이 명령 그대로)
- 거부: `--time-limit` 없음(줄기 plan `max_frames` = H + 48, 줄기는 모두 시간 상한으로 끝나 없으면 [040-R8](../spec/2026_10_15_22_24_040-replay-dataset.md) 디싱크 검사가 시점을 버릴 수 있음), records 이름이 `branch`로 시작하지 않음(이어 하기 검사), records 폴더가 `branches.jsonl`·`points.jsonl`이 있는 [051-R12](../spec/2026_11_12_12_39_051-branch-games.md) 출력 폴더. records는 자기 폴더(`build/`, 미리 만듦).
- 스텝 기억을 라이브 세션 것으로 채우고, 기록된 결정마다(NoOp 포함) 재생 프레임 L − 2에 관측·기억 전용 스텝(`unknown` 레이블, BC 가중치 0, delay는 다음 스텝까지·마지막은 판 끝까지, records `memory.decision_steps`)을 더해 결정 시점 덤프 행 보장([041-R28](../spec/2026_10_15_22_24_041-imitation-learning.md)).
- BC 학습 데이터 아님: 재레이블 없는 `--memory-log`는 `--src branch`만, BC shard 폴더(`<shared>/datasets/replays/shards`) 밖 `--out` 필수(APM·5분 기준 없음).
- 기록 파일: `g<id>.jsonl`이 있으면 그것, 없으면 [051-R2](../spec/2026_11_12_12_39_051-branch-games.md) 행동 기록 `g<id>-p<players 인덱스>.jsonl`(T0 세션 기록은 건너뜀), 시점 BWAPI 플레이어 = 끝 줄 플레이어인 기록. 정렬·판 단위 전부 아니면 없음·유닛 확인·records `memory`는 DAgger 재레이블과 같음. 기록이 다른 플레이어 것인 관점(테란전 상대)은 쓰지 않음(records `filtered`, 053-R5).
- 분석: `python -m uranium diag plan-branch --dir <shared>/eval/branch-<날짜> [--dump <줄기 plan-dump> --s0 <plan-s0 폴더>] [--q-head <머리>]` → `<dir>/analysis/plan-branch.md`·`analysis.json`(짝 효과·줄기 bootstrap·split-half, M3u·M2c, 무효·거절·missed·희석; `--help`).
  - `--q-head`: M3u 채점 머리. 기본 = 예측기에 `held_8m` 머리가 있으면 그것, 없으면 `scv8`(S0b)([051-R13](../spec/2026_11_12_12_39_051-branch-games.md) ②). `--m3u-outcome`(비교할 분기 측정 결과, 기본 `held_8m`)과 다름.

### 분기 게임 B1 실행 순서 ([spec 051](../spec/2026_11_12_12_39_051-branch-games.md))
- B1 체크포인트 `m2-gn-t1e2/ckpt-2844`([051-R1](../spec/2026_11_12_12_39_051-branch-games.md) ②, 사용자 결정 2026-11-17, `config/branch_b1.toml`). 누출 수정 최종 가중치 F가 다르면 S0b·줄기 덤프를 F로 다시(게임은 그대로).
- 기호: `<S>` = `URANIUM_SHARED_DIR`, `<D>` = 실행 날짜, `<T>` = `eval/branch-b1-<D>/trunks`(줄기 job, 공용 드라이브 기준), `<O>` = `<S>/eval/branch-b1-<D>`(분기 job = 051-R12 출력 폴더).
- 줄기·분기 판을 두는 모든 PC는 같은 커밋의 hostd·에이전트·`_core`.
- 경로 측정([050-R9](../spec/2026_10_15_22_24_050-evaluation.md)) 게임과 hostd 자리·추론 GPU 공유 금지([051-R14](../spec/2026_11_12_12_39_051-branch-games.md)): 측정 전에 분기 작업자를 Ctrl+C 한 번으로 멈추고, 끝나면 같은 명령으로 재시작(남은 판부터).
1. 줄기 plan(아무 PC, venv python): `python tools/eval/branch_plan.py trunk --config config/branch_b1.toml --out <S>/<T>/plans/01-trunk.toml`(맵마다 첫 목표 적중 10·빗나감 10, stage `branch-b1-trunk`). `<S>/<T>/job.toml`: `stages = ["plans/01-trunk.toml"]`, `hosts = "hosts.toml"`, hosts는 경로 측정과 겹치지 않는 게임 호스트(`[[host]] addr = "<LAN 주소>:<포트>"`, `max_games`).
2. 줄기 판(추론 PC): `python -m uranium eval-worker --job <T> --priority <n> [--backend cuda|rocm]`(게임 호스트마다 hostd, 그 PC hostd를 띄우면 `--local-hostd`·`--hostd`) → `python -m uranium eval-status --job <T>`(병합 `games.jsonl`·`results.jsonl`). 40판 GPU 약 0.5시간.
3. 시점·분기 plan(아무 PC): `python tools/eval/branch_plan.py branches --trunk-job <T> --stage branch-b1-trunk --config config/branch_b1.toml --out <O> --hosts hosts.toml --replacements <S>/<T>/plans/02-trunk-r.toml`
   - `--trunk-job`·`branch_collect.py --job`은 있으면 그대로, 없으면 공용 드라이브 기준.
   - 출력 `<O>/points.jsonl`·`plans/branches_plan.jsonl`·`plans/<이름>.toml`·`branch_plan.json`(쓴 줄기·뺀 이유·칸)·`job.toml`, hosts는 `<O>/hosts.toml`에 따로.
   - 첫 시점이 2:45 뒤라 두 규칙이 같은 결정에 걸리면 둘째 시점은 첫 시점 + 30초 이상 첫 결정으로(`b1-p2`의 `after_rule = "b1-p1"`, `after_s = 30`).
   - 둘째 시점이 4:00 전에 없거나(`rule did not fire`) 다른 검사에 걸린 줄기는 교체, (맵, 조건) 칸이 모자라면 교체 plan 작성 → `<T>/job.toml` `stages`에 덧붙여 2번 재실행(남은 판만) → 3번을 모든 줄기 stage(`--stage branch-b1-trunk --stage branch-b1-trunk-r ...`)와 새 `--replacements <S>/<T>/plans/03-trunk-r2.toml`로 재실행(교체 stage 이름 `-r`, `-r2`, … 자동). 칸이 다 차면 교체 plan 없음.
4. 분기 판(추론 PC): `python -m uranium eval-worker --job eval/branch-b1-<D> --priority <n> [--backend ...]` → `python -m uranium eval-status --job eval/branch-b1-<D>` → `python tools/eval/branch_collect.py --job eval/branch-b1-<D>`(`<O>/branches.jsonl`, `branch_collect.json`: 무효 사유별·대안별 수·빠진 판). plan 경로는 공용 드라이브 기준이라 작업자마다 `URANIUM_SHARED_DIR` 필요. 1,280판 4~6시간(경로 측정 틈에 나눠).
5. 줄기 shard(Linux CPU 호스트, 저장소 루트, `URANIUM_MAPS_DIR=<S>/datasets/replays/build/maps`, `mkdir -p <O>/build`): 줄기 stage마다 같은 `--out`·records로 `target/release/uranium-replay build <S>/<T>/<stage>/replays --src branch --out <O>/shards/trunks --records <O>/build/branch_build.jsonl --memory-log <S>/<T>/<stage>/acts --time-limit 11568 --jobs 4`. shard는 출처 폴더 아래 한 단계(`<O>/shards/trunks/shard_*.bin.zst`)여야 덤프가 `--shards <O>/shards`로 읽음(`<O>/shards`에 바로 쓰면 [040-R13](../spec/2026_10_15_22_24_040-replay-dataset.md) 깊이 검사로 거부).
6. 덤프(CPU로 충분): `python -m uranium diag plan-dump --checkpoint <S>/checkpoints/m2-gn-t1e2/ckpt-2844 --shards <O>/shards --splits train,val,test --config config/bc.toml --backend cpu --out <O>/dump`. 기본 `--splits val,test`는 줄기 대부분을 뺌(manifest 없는 줄기 shard는 모두 `train`) → 셋 다 줌. 줄기 4판 CPU 약 2분 반.
7. 분석: `python -m uranium diag plan-branch --dir <O> --dump <O>/dump --s0 <S>/eval/jepa-plan-feas-<날짜>/s0b`(`--s0` = S0b 예측기 A `models/A_k8.pt`·`A_k32.pt`가 있는 `plan-s0` 출력 폴더, B1 체크포인트 `m2-gn-t1e2/ckpt-2844`로 다시 학습한 것 — 아래 "둘째 실행 S0b" 끝. 11-12 S0b(`m2-s1/ckpt-39096`)는 sha256이 달라 쓰지 않음) → `<O>/analysis/plan-branch.md`·`analysis.json`, 리포트 `docs/report/experiment/<YYYY_MM_DD_HH_mm>_branch_b1.md`([051-AC7](../spec/2026_11_12_12_39_051-branch-games.md)).
- [051-AC6](../spec/2026_11_12_12_39_051-branch-games.md) 작은 확인(Linux CPU 호스트 하나, CPU 추론): 1번에 `--hit 1 --miss 0`(맵마다 1판), 3번에 `--hit 1 --miss 0 --alts none,force_bunker --repeats 1 --verify-state`, 작업자 `--backend cpu --no-warmup --max-batch 4 --local-hostd <주소> --hostd target/release/uranium-hostd`.
  - `--verify-state`는 plan 판 모두에 앞부분 추론을 켬 → B1 본 plan에는 주지 않음(시험은 판 1~2개 stage 따로).
  - 여러 판이 한 배치면 앞부분 추론 배치 구성이 줄기와 달라 `state_check`가 비트 일치 안 할 수 있음(3판 동시 최대 차이 1e-6~1e-5, 혼자 `--max-batch 1`이면 일치).
  - 8:00 판 하나 약 1분(3판 동시, 앞부분 추론 포함 1~1.5분). 확인 기록 `<S>/eval/branch-ac6-20261113/`.

### 탐색 판 ([spec 054](../spec/2026_11_18_08_58_054-exploration-games.md))
- plan(venv python, `uranium._core` 필요): `python tools/eval/explore_plan.py --checkpoint <F> --zero-last-failed <true|false> --surrender <off|shadow|on> [--games 1000 --first 1 --name explore-pilot --out <파일>]`. 두 설정은 F의 050 평가 plan 값([054-R1](../spec/2026_11_18_08_58_054-exploration-games.md)), 기본값 없음.
- 상대 몫·맵 반반은 판 수에 맞춰 최대 나머지로 반올림(160의 배수면 정확). 시드는 탐색 블록(키 7,000,000, 게임 번호 1~10,100), 다음 plan은 `--first`를 앞 plan 뒤로.
- plan `[explore]`([020-R22](../spec/2026_10_15_22_24_020-env-orchestration.md) ⑤)이 있으면 `players[0]` 모델 세션이 판 시드로 시점 3개를 정하고 행동 기록 줄 `explore`, 끝 줄 `explore`(`taus`·`missed`), 세션 결과 `explore`를 남긴다. 실행은 평가 작업(`python -m uranium eval-worker --job <job>`), 경로 측정과 hostd 자리·추론 GPU 공유 금지([054-R7](../spec/2026_11_18_08_58_054-exploration-games.md)).

### 경로 측정·결정 단위 측정
- 경로 측정 프로토콜(050-R9, [measurement_protocol](measurement_protocol.md)): Checkpoint 1 job 재집계 `python tools/eval/protocol.py tables --lineage ck1 --point <지점>`, 새 평가 job `python tools/eval/protocol.py job --lineage <ck1|m2> --point <지점> --checkpoint <ckpt> --metrics <지표,...> --job <이름>`, 결과 표 `python tools/eval/tables.py <job>... [--json <파일>]`(러시·운영형·T1·승률; 명령 스트림 `uranium-replay cmddump`), 교사 보관분 행동별 채점 `tools/eval/per_action.py`·`per_action_table.py`. 호스트 목록 git 미추적 `tools/eval/hosts.toml`(예시 `hosts.example.toml`).
- 결정 단위 측정(G0, 분석 전용):
  - 저장된 레이블·관측: `uranium-replay granularity <shard 폴더…> --matchup TvZ --stage S2 --families fighting_spirit,circuit_breaker --lo 2400 --out <out>/human.jsonl.gz`
  - 리플레이 자체 명령([040](../spec/2026_10_15_22_24_040-replay-dataset.md) 레이블러, 5분 기준·디싱크 검사 없음, `.env` 런처·브리지 필요): `uranium-replay granularity --replays --list <리플레이 목록> --source <태그> --lo 2400 --out <out>/<태그>.jsonl.gz`
  - 표: `python tools/eval/granularity.py --human … --teacher <dump> <plan> --model <dump> <plan> <job 폴더> <stage 번호> --out <md> --json <json>`(2:30~4:30 지연 분포·12기 포화율·같은 프레임 연타·SCV 미세 조작·명령 빈도, venv python, plan 판은 시드 조건으로 이른/늦은).
  - 결과·12기 포화·사람 러시 규칙·막음 구분: [model2](../report/plan/2026_11_09_22_18_model2.md) 3절. 지연 = 다음 명령 레이블까지 프레임, 연타 = 앞 레이블 뒤 2프레임 이내, 명령 빈도 = 창 안 1,000프레임당 레이블.
  - SCV 미세 조작: 본진 근처 적 병력이 있는 5초 구간마다 전투 명령(move·attack·smart·repair·hold·stop·patrol, 중립 유닛 우클릭은 채취) 받은 SCV 수, 벙커 완성 → 첫 탑승(Load_Unit·마린 벙커 우클릭), 첫 피격 → 첫 수리(Repair_Unit·SCV 벙커 우클릭, 창 안 없으면 censored).
  - plan 판: "이른" = 상대 첫 목표가 우리 본진인 seed 조건([050-R6](../spec/2026_10_15_22_24_050-evaluation.md)). 리플레이 이름 `g<n>`으로 seed(`seeds[n − 1]`, matchup 순)에 맞춤. `--opponents` 기본 `t0 zerg_rush,ualbertabot12poollings`(빈 값 = 전부).

## 6. 리플레이
`uranium-replay`([spec 040](../spec/2026_10_15_22_24_040-replay-dataset.md)):
- `inspect <rep>`, `convert <rep> <out>`(SC:R → 1.16.1, 맵 CHK 정규화, [spec 013](../spec/2026_10_15_22_24_013-version-mapping.md)), `check-screp <dir> --screp <exe>`([040-AC1](../spec/2026_10_15_22_24_040-replay-dataset.md)), `ophist <dir>`, `play <rep>`(재생·디싱크 신호), `mapgen`, `roundtrip <dir>`([013-R7](../spec/2026_10_15_22_24_013-version-mapping.md)).
- `shard-stats <dir>`: 하위 폴더 포함. GameInfo `is_enemy`가 정확히 1명이 아닌 게임이나 같은 (리플레이, 시점) 게임이 두 번([040-R20](../spec/2026_10_15_22_24_040-replay-dataset.md))이면 실패.
- `cmddump <rep>...`: 리플레이마다 명령 스트림 JSON 한 줄(`tools/eval/tables.py`용).
  - 형식 `{"rep", "frames", "players": [{"id", "race"}], "cmds": [[frame, player, opcode, payload hex]]}`.
- `opsdump <rep>... [--perspective <slot>] [--until-frame 14400] [--stride 8] [--field-check]`([052-R7](../spec/2026_11_13_17_19_052-ops-metrics.md)): OpenBW 재생으로 우리 시점(세션 행 `slot`, 없으면 테란 플레이어마다)에서 8프레임마다 [052](../spec/2026_11_13_17_19_052-ops-metrics.md) `ops`·`rush_combat` 집계.
  - 시점마다 JSON 한 줄 `{"rep", "perspective", "frames", "ops", "rush_combat"}`, 실패면 둘 다 null + `error`. 10:00과 러시 전투 창 끝 중 늦은 쪽에서 끊음.
  - `--field-check`([052-AC2](../spec/2026_11_13_17_19_052-ops-metrics.md) 리허설): `field_check` = `depleted`(창 안 우리 SCV가 반납한 필드의 초기량 − 마지막 잔량 합), `fields`, `unknown_fields`, `scvs`(창 안 마지막 관측 SCV 수), `scvs_max`, `carrying`(마지막 관측 미네랄 운반 중), `lost_cargo`(운반 중 죽은 SCV마다 8). 기준 |`depleted` − 8 × `carrying` − `lost_cargo` − `mined`| ≤ 8 × `scvs_max`.
- `granularity <shard 폴더…> | --replays <rep…>`: 결정 단위 측정 G0·[AMB-031-7](../spec/README.md)용 판마다 레이블·압축 상태 `.jsonl.gz`(분석 전용, [5](#경로-측정결정-단위-측정)).
- `dataset-stats <records.jsonl 또는 폴더>... --out docs/report/survey/<YYYY_MM_DD_HH_mm>_dataset_stats.md`: 맵·매치업·src·stage·분할별 게임·시점 수, 디싱크 폐기율, `result_source` 분포([040-R6](../spec/2026_10_15_22_24_040-replay-dataset.md), [040-AC2](../spec/2026_10_15_22_24_040-replay-dataset.md), [040-AC4](../spec/2026_10_15_22_24_040-replay-dataset.md)).
- `build <dir> --src <tag> --records <jsonl> [--jobs N --limit N --part i --parts N --ladder-records <collect.jsonl>]`: 필터 → 013-R7 왕복 검증 → 재생 → 레이블 → shard.
  - 기본 출력 `<URANIUM_SHARED_DIR>/datasets/replays/shards/<src>`, 원본 `datasets/replays/raw/<src>/`([040-R13](../spec/2026_10_15_22_24_040-replay-dataset.md)), 맵 사전계산은 `URANIUM_MAPS_DIR`에 씀.
  - 쓰기([040-R17](../spec/2026_10_15_22_24_040-replay-dataset.md)): `URANIUM_SHARD_STAGING_DIR`(로컬) → 공용 경로 `<이름>.tmp` 복사·fsync → 캐시 없이 다시 읽은 SHA-256 일치 → rename. 불일치 3회 재시도, 실패하면 오류로 멈추고 로컬 사본 남김.
  - 승패 `result`([040-R8](../spec/2026_10_15_22_24_040-replay-dataset.md)): 브리지 승자 → 첫 LEAVE 이탈자 패배 → 기록 승자(`--ladder-records`, `--src scr_ladder`이면 기본 `datasets/replays/ladder/collect.jsonl`) → 판정 불가(0).
  - 재개: records 폴더에서 이름이 `<src>`로 시작하는 `.jsonl`을 모두 읽어 건너뜀. 표본·시험 records는 하위 폴더에.
  - 중복(040-R20): 같은 SHA-256 파일은 출처 전체 정렬 목록(`--part` 분할 전) 첫 파일만, 나머지 `duplicate file in this source (040-R20): same file as global index <n>`. 파트마다 출처 전체 크기 목록을 읽음. 일부 파트만 돌리면 원본이 안 돌린 파트에 있는 리플레이는 빠짐.
  - `config/map_exclude.toml` 맵(디싱크 10% 초과, [040-R16](../spec/2026_10_15_22_24_040-replay-dataset.md))은 재생 전 제외. 대상 맵은 넣을 수 없음.
  - 브리지와 `uranium-replay`는 같은 커밋([011-R10](../spec/2026_10_15_22_24_011-bridge-openbw.md)).
- 교사·DAgger·증류·분기 build 옵션: [4](#4-교사dagger증류-데이터), [5](#5-분기-게임경로-측정).

## 7. BC
학습 `python -m uranium`(PyTorch 학습기 + Rust 공급·실행 루프, [AMB-000-4](../spec/README.md)), 데이터 확인 `uranium-train`. 결과·경위: [m6_bc_eval](../report/experiment/2026_10_29_15_37_m6_bc_eval.md), [bc_ablation](../report/experiment/2026_10_29_15_37_bc_ablation.md), [m6_core_ablation](../report/experiment/2026_10_18_18_43_m6_core_ablation.md).

### bc 명령
`python -m uranium bc (--dataset <rel> | --shards <dir>)... (--out <ckpt> | --run <id> [--take-over]) [--backend cuda|rocm|cpu] [--core lstm|gtrxl] [--s2 | --stage s2] [--resume <ckpt> | --init <ckpt|run>] [--steps N]`
- 체크포인트([061-R10](../spec/2026_10_15_22_24_061-shared-storage.md)): `model.safetensors`(f32, 정책 버전 = 이 파일 SHA-256), `optim-main.safetensors`, `meta.json`, `train_state.json`, `run_ident.json`, `config/`.
- `--core` 기본 `gtrxl`([032-R17](../spec/2026_10_15_22_24_032-model-architecture.md); `bc`·`bc-profile`·`random-checkpoint`, 큐 `core`). LSTM 체크포인트는 `--core lstm`(코어가 다르면 읽을 때 오류).
- `--dataset replays/shards/<src>`는 manifest 검증, `--shards`는 안 함. 하위 폴더까지 읽음([041-R9](../spec/2026_10_15_22_24_041-imitation-learning.md)).
- `URANIUM_SHARED_LOCAL=1`([061-R12](../spec/2026_10_15_22_24_061-shared-storage.md)): `--dataset`을 캐시로 받지 않고 공용 경로에서 manifest 검증 후 바로 읽음(`uranium-train` `--dataset`도 같음).
- 폴더 배치([041-R13](../spec/2026_10_15_22_24_041-imitation-learning.md)): 출처 폴더(`<파트>/shard_*`) 또는 출처들의 상위 폴더(`<출처>/<파트>/shard_*`)만. 다른 깊이(출처 바로 아래 shard, 더 깊음, 혼합)는 파일 이름과 함께 오류. 파일 순서 키 `<출처>/<파트>/<파일>` → `--dataset`/`--shards`, 적은 순서, 출처별/상위 폴더, 저장 위치·대소문자와 무관하게 게임 순서·첫 배치·지문 같음. 같은 출처·파일 두 번(대소문자·링크만 다른 경로 포함)은 실패.
- `--run`([061](../spec/2026_10_15_22_24_061-shared-storage.md) run, `checkpoints/<id>/`): 잠금·heartbeat·push, 시작 시 `LATEST`에서 재개(데이터 위치 포함). 저장은 `checkpoint_every` update 또는 `checkpoint_minutes`마다, Ctrl+C는 저장 후 종료.
- 다른 PC 잠금(유효·만료 모두)이면 종료 코드 75. `--take-over`는 상대 PC가 멈춘 것을 확인한 뒤에만: 즉시 인수·경고, 상대가 살아 있으면 다음 heartbeat에서 push 없이 멈춤([061-R5](../spec/2026_10_15_22_24_061-shared-storage.md)).
- 단계: 기본 S1(전 소스 train 분할), `--s2`·`--stage s2` = 대상 맵 train 분할([041-R12](../spec/2026_10_15_22_24_041-imitation-learning.md)·R13). S2는 `--init`으로 가중치만([041-R4](../spec/2026_10_15_22_24_041-imitation-learning.md)). `--init <run>`은 `DONE`이 있어야 하고 `DONE`이 가리키는 마지막 체크포인트([061-R13](../spec/2026_10_15_22_24_061-shared-storage.md)). 길이 `config/bc.toml` `[steps]`([041-R14](../spec/2026_10_15_22_24_041-imitation-learning.md)).
- 같은 run을 큐 밖에서 `python -m uranium bc --run`으로 직접 돌리지 않음([8](#8-감시작업자)).
- 로그 `no_result_games` = 시작한 게임 중 판정 불가(`result_source = none`)로 value 가중치 0인 수([041-R7](../spec/2026_10_15_22_24_041-imitation-learning.md)).
- 하이퍼파라미터 `config/bc.toml`(GPU 메모리: `micro_lanes`, `pad_entities`, `grad_checkpointing`), 코어 비교 `config/bc_core_ablation.toml`, 보상 `config/reward.toml`. `grad_clip` = 전 파라미터 한 벡터 전역 L2 norm, 레인 그룹 gradient 합친 뒤 optimizer step 전 한 번([041-R3](../spec/2026_10_15_22_24_041-imitation-learning.md), RL `config/rl.toml`도 같음).

### 일부 게임만·혼합 몫 ([041-R21](../spec/2026_10_15_22_24_041-imitation-learning.md), [041-R26](../spec/2026_10_15_22_24_041-imitation-learning.md))
- 설정 `[data.subsample]`(`fraction` = 남길 몫, `keep_sources` = 통째로 두는 출처): 단계 선택 뒤 게임마다 `replay_sha256`·`data.seed` 해시로 고름. `bc`·`bc-epoch`·`feed-digest`가 같은 게임 → `[steps]`는 `uranium-train bc-epoch --config <설정>` 값.
- `[data.subsample.fraction_by_race]` `T`·`P`·`Z` = 그 상대 종족 사람 게임 몫(없으면 `fraction`).
- `bc-epoch` 출력 `by_source`(단계별 출처·상대 종족별 게임·스텝). `[data.subsample]`이 있으면 `s2_mix`: 교사(`keep_sources`) 몫 50 ± 3%(`teacher_ok`), 사람 몫 안 종족 비율 대 S2 사람 데이터 비율 ± 3%p(`race_ok`).
- `uranium-train mix-fractions --config <혼합 설정> [--teacher-share 0.5]`: 설정의 `data.seed`·`keep_sources`로 두 조건을 맞추는 `fraction_by_race` 계산 + 그 값으로 센 S2 epoch(`[steps] s2`·warmup·`s2_mix`). seed마다 다시 셈(예 `config/bc_m2_mix_e1.toml`). 조건 불충족이면 0 아닌 코드.
  - 데이터: `--dataset`·`--shards`, `--dataset-group replays/shards/a,replays/shards/b`(manifest로 받거나 확인한 묶음), `--group a,b,c`(그대로 읽는 shard 폴더 묶음, 묶음마다 훑기 캐시 따로 → 사람 4출처와 교사를 나누면 교사 shard가 바뀌어도 사람 캐시 재사용, `bc-epoch`도 같음).
  - `--share`·`--teacher-races`: [4](#4-교사dagger증류-데이터).
- 체인 시작 관측([041-R30](../spec/2026_10_15_22_24_041-imitation-learning.md)): `[data] chain_start_sources = ["botgen"]`(허용 `botgen`·`dagger`) → 그 출처의 체인 스텝(프레임 ≤ 앞 스텝 + 1이고 앞 레이블이 NoOp 아님)은 체인 시작 스텝 RawFrame으로 관측(그 events 비우고, 체인 스텝 자신의 events는 다음 체인 아닌 스텝에). 레이블·delay·스텝 수·`[steps]` 그대로, 키가 있을 때만 설정 지문에(경로 2 `config/bc_m2_mix_e1.toml`).
  - 채점은 설정 키와 무관: `bc-eval --chain-start-sources botgen`. `tools/eval/per_action.py`는 기본 두 방식(기존 필드 = 현재 관측: Checkpoint 1 경로 연장 규칙 입력, 경로 2는 기록만; `chain_start` 블록: 경로 2 연장 규칙 입력, 041-R30 ⑤), `--chain-start none`이면 현재 관측만.
- 교사 경로 빈 파인튜닝: `python -m uranium bc --run bc-ft-empty-s1 --s2 --init <shared>/checkpoints/bc-s2e2-pt/ckpt-11387 --config config/bc_ft_empty.toml --dataset replays/shards/stardata --dataset replays/shards/community --dataset replays/shards/scr_ladder --dataset replays/shards/personal --backend rocm`(교사 파인튜닝은 `config/bc_ft_teacher.toml` + 교사 shard `--dataset`).

### 메모리·공급 환경변수
모두 batch 내용·순서를 바꾸지 않음 → 값이 다른 호스트끼리 같은 run 이어 하기 가능. 시작 줄 `{"prefetch": N, ...}`, `entity_chunk`·`pack_batch`·`micro_lanes`로 확인. 호스트별 값은 [1](#호스트-메모리-설정-env).
- `URANIUM_BC_PREFETCH`([041-R16](../spec/2026_10_15_22_24_041-imitation-learning.md), 기본 2): batch 하나 약 4 GB, 깊이 + 2개까지 메모리. Linux ROCm 호스트도 `URANIUM_MALLOC_MMAP_THRESHOLD=1048576`과 함께 2(최대 상주 약 10 → 6.3 GiB, 데이터 대기 0). `0` = 데이터 스레드 없이 batch 1개(대기 약 4~6초/update). 다음 레인 그룹 collate는 늘 작업 스레드에서 미리.
- `URANIUM_BC_FILE_AHEAD`([041-R16](../spec/2026_10_15_22_24_041-imitation-learning.md), 기본 0 = 끔, CUDA 호스트 2, Linux ROCm 호스트는 두지 않음(끔)): 다음 shard 파일을 읽기 스레드가 미리 디코드해 남는 게임만 보관하는 파일 수. 켜면 디코드 중인 파일 1개(필터 전, 순간 약 1.7 GiB)와 보관 파일(통째로 남는 출처는 하나당 약 0.65 GiB)이 추가로 메모리에 있음. 시작 줄 `file_ahead`로 확인.
- `URANIUM_BC_ENTITY_CHUNK`([041-R17](../spec/2026_10_15_22_24_041-imitation-learning.md), 기본 0 = 끔, `torch.utils.checkpoint`), `URANIUM_BC_PACK_BATCH`([041-R18](../spec/2026_10_15_22_24_041-imitation-learning.md), 기본 0): Linux ROCm 호스트 `0`·`1`([AMB-041-8](../spec/README.md): 재계산 끄면 약 15% 빠름, 최악 모양 GPU 5.7 GiB; 16이면 1.8 GiB). 재계산 dropout 난수 열만 다름.
- `URANIUM_BC_MICRO_LANES`(041-R3, [AMB-041-7](../spec/README.md)): 한 번에 forward·backward하는 레인 수(없으면 설정 `micro_lanes`). CUDA 호스트 2, Linux ROCm 호스트 1. dropout 난수 묶음만 다름.
- `URANIUM_BC_FEED_THREADS`(기본 3): 레인별 feature 추출·레인 그룹 압축 해제 스레드. 늘리면 학습 스레드와 CPU를 다퉈 느려짐.
- `URANIUM_BC_PROFILE=1`: 구간별 시간 출력.

### 시작 검사·재개
- `--dataset` pull·검증은 파일마다 `[pull]`/`[verify]` 진행 줄을 stderr에(감시 30분 멈춤 판정 대비).
- 맵 사전계산: `URANIUM_MAPS_DIR` 필수(기본 없음). 시작 때(잠금 전) 고른 게임의 맵 해시마다 현재 버전 `<map_hash>.toml` 확인, 없으면 빠진 해시·게임 수 나열 후 실패. 로더도 오류. `bc-eval`·`label-check`도 같음. 라이브 에이전트는 투혼·서킷브레이커 사전계산이 없으면 세션 거부, 그 외 맵은 경고 후 채널 0([spec 031](../spec/2026_10_15_22_24_031-action-space.md)).
- shard 훑기 캐시: `<URANIUM_CACHE_DIR>/scans/scan-<지문>.json`(게임별 meta·스텝 수·privileged 없는 스텝 수). 지문 = 파일별 manifest SHA-256(없으면 크기·수정 시각). `bc`·`bc-eval`·`label-check`·`split-stats`·`bc-epoch` 공용.
- `--shards` 폴더(상위 폴더면 그 아래 출처 폴더)에 `manifest.json`이 있으면 캐시·데이터 지문이 manifest를 따름(검증은 안 함). **shard를 다시 만들고 manifest를 두면 다른 데이터로 재개가 허용될 수 있음** → 반드시 `manifest <dir>`로 재생성하거나 옛 `manifest.json` 삭제. manifest 없는 사본은 크기 기준 지문이라 `--dataset`으로 시작한 run을 이어 받지 못함.
- 재개 검사: `train_state.json` `extra.run`에 stage(`S1`/`S2`/`AblationA`), 설정 지문(결과 영향 설정 화이트리스트 `ResultSettings`, `[steps]` 반영; 실행 방식은 `EXECUTION_ONLY`; 새 필드는 둘 중 하나로 분류해야 테스트 통과), 데이터 지문. 하나라도 다르면 잠금 풀고 거부(우회 없음, 새 run 또는 `--init`).
- 보존([061-R6](../spec/2026_10_15_22_24_061-shared-storage.md)): `keep_checkpoints`(없으면 5, 0 = 전부; `config/bc_core_ablation.toml`은 0). 삭제 실패는 경고만.
- 압축 보관([061-R17](../spec/2026_10_15_22_24_061-shared-storage.md)): 묶기 `<venv python> tools/archive/archive_shared.py pack --shared <shared> --plan <json>`, 단위 찾기 `<shared>/archive/INDEX.json`, 풀기 `<venv python> tools/archive/archive_shared.py restore --shared <shared> --unit <단위> [--only <경로>]`(manifest 대조, 기존 파일 덮어쓰지 않음).
- 디스크([061-R8](../spec/2026_10_15_22_24_061-shared-storage.md)): 시작·저장마다 공용 드라이브·로컬 캐시(로컬 `--out`이면 그 폴더) 물리 디스크 80% 이상이면 경고(학습 계속).

### 평가·데이터 확인
- `python -m uranium bc-eval (--dataset <rel> | --shards <dir>)... --checkpoint <ckpt|run> [--split test|val] [--games N --seed 1] [--backend cuda|rocm|cpu] [--out <json>]`: [041](../spec/2026_10_15_22_24_041-imitation-learning.md) 지표, 서렌더 재현 [041-AC4](../spec/2026_10_15_22_24_041-imitation-learning.md) 포함, `--out`이면 파일에도.
  - `--games N`: 분할 전체를 고정 seed(`--seed`, 기본 1)로 섞어 앞 N게임(체크포인트·코어 무관 같은 게임, 032-R17 ②).
  - 단위: `delay_mae_frames` = 프레임(bucket 값), `delay_mae_buckets` = bucket 번호, `nll.<head>` = 그 헤드를 쓰는 스텝(`nll_steps.<head>`)만의 평균. 출력 `units`에 설명.
  - 엔티티 재계산 없이 돌므로 메모리 큰 CUDA 호스트에서. 예: `python -m uranium bc-eval $DATA --checkpoint <shared>/checkpoints/bc-s2-pt --split test`.
- `python -m uranium bc-profile`: [2](#2-벤치프로파일).
- T0 대전: `python -m uranium run --plan config/plans/bc-vs-t0.toml`(plan 체크포인트 경로를 새 run으로). 다른 호스트 게임을 쓰면 에이전트 PC `.env` `URANIUM_AGENT_LISTEN`/`URANIUM_AGENT_PUBLIC_ADDR`를 LAN 주소로(기본 127.0.0.1이면 다른 호스트 판이 모델 없이 0초에 끝남). GTrXL은 큰 장면 추론이 무거우므로 `wall_timeout_s` 넉넉히(코어 비교 7,200초).
- `uranium-train`(모두 `--dataset`/`--shards`):
  - `bc-epoch`: 1 epoch update 수(게임 시작 NoOp 스텝 포함).
  - `split-stats`: `splits`(소스·맵 그룹·분할별 게임 수), `sources`(소스별 `privileged` 없는 게임·스텝 수·비율, `result_source`별 게임 수 [040-R8](../spec/2026_10_15_22_24_040-replay-dataset.md)).
  - `eval-games --split <train|val|test> --games <n> [--seed 1] [--target-maps]`: `bc-eval --split <s> --games <n> --seed <k>`가 채점하는 게임, 게임마다 JSON 한 줄(키 `map_family`·`matchup`·`perspective_slot`·`replay_sha256`·`split`·`src`, 이름순). `--target-maps` = 그 분할의 대상 맵 게임만 같은 방식으로 섞음. 훑기 캐시만 읽음(스텝 안 읽음).
  - `feed-digest [--stage s1|s2] [--batches 3] [--group 1] [--pack]`: 첫 BC batch 레인 그룹별 SHA-256, batch마다 JSON 한 줄. `python -m uranium feed-digest --shards <dir>`(같은 인자)와 같아야 함(Rust·Python 공급 일치, [AMB-000-4](../spec/README.md)).
  - `label-check [--games N]`([034-AC3](../spec/2026_10_15_22_24_034-action-table.md)): `target_causes`(대상 없음·위치 거절 원인: 이전 목격 여부·경과 프레임, 건물 종류별 막는 조건, 실제로 지어졌는지), `per_game`(게임별 무효 수).
  - `manifest <dir> [--verify]`([061-R7](../spec/2026_10_15_22_24_061-shared-storage.md)): 만들 때 하위 shard를 끝까지 디코드, 실패하면 나열하고 쓰지 않음([040-R17](../spec/2026_10_15_22_24_040-replay-dataset.md)).

### 큐 운영 ([060-R8](../spec/2026_10_15_22_24_060-infra.md))
- Linux ROCm 호스트: 작업자 24시간(우선순위 1). CUDA 호스트: 쉬는 동안만 사용자가 켬(우선순위 10). PC를 써야 하면 **Ctrl+C** → `ckpt-N Pushed` 출력 후 종료까지 대기(약 30초).
- Linux ROCm 호스트가 여럿이면 `--priority`를 서로 다르게, CUDA 호스트 10보다 낮게(예: 2와 1; CUDA 호스트 > 첫째 > 둘째).
- 작업자는 비어 있는 첫 run을 끝날 때까지 쥠. 빈 run이 없을 때만 자기보다 우선순위 낮은 작업자의 run을 양보받음([061-R14](../spec/2026_10_15_22_24_061-shared-storage.md)). 멈춘 호스트의 run은 다른 작업자가 비어 있을 때 마지막 체크포인트부터(수동 인계 불필요).
- 큐 순서대로 잡히므로 특정 run을 잡을 호스트의 작업자를 먼저 켬(아래 예시: CUDA 호스트를 먼저 켜야 S1을 CUDA가 잡음. 이후 S1 → S2, 끝나면 Linux 쪽 ablation을 양보받아 마무리).
- 큐 파일 예(`<shared>/queue/bc.toml`, 코어 GTrXL, 구조 변형은 `config/bc.toml` `[model]`, [032](../spec/2026_10_15_22_24_032-model-architecture.md) §2.7):
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
- 작업자 실행(자식 `URANIUM_PYTHON -m uranium bc --run ...`):
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
- 진행 확인: `<shared>/workers/<host>.json`(1분마다), `uranium-train devices --run bc-s1-pt`(장비별 update 수·비율), `<shared>/checkpoints/<run>/`(1시간마다, 끝나면 `DONE`).
- 속도(모두 켬 모델, 32레인 × T64, 데이터 대기 포함): Linux ROCm 호스트 약 12~13 s/update, CUDA 호스트 약 1.3 s/update([pytorch_opt](../report/bench/2026_10_24_07_03_pytorch_opt.md) 6절).

## 8. 감시·작업자
([060-R6](../spec/2026_10_15_22_24_060-infra.md), [061-R13](../spec/2026_10_15_22_24_061-shared-storage.md)~R15)

### supervise
`uranium-train supervise [--hang-minutes 30 --max-restarts 5 --window-minutes 60 --delay-seconds 60 --progress-file <file> --fatal-pattern <regex>... --log <file>] -- <bc|rl 인자>`
- 자식 `URANIUM_PYTHON -m uranium <bc|rl 인자>`(`.env`에 없으면 시작 안 함). 비정상 종료·멈춤이면 같은 run 재시작.
- 재시작 안 함: 종료 코드 0·130·75(잠금 못 얻음: 다른 호스트 잠금, 획득 경쟁 패배, 양보 예약)·76(양보). 재시작: 70(장치 치명 오류, 저장 없음, [042-R20](../spec/2026_10_15_22_24_042-rl-training.md)).
- 멈춤 판정([AMB-042-20](../spec/README.md)): `--progress-file`이 있으면 그 파일 변경만 진행(stderr 줄 아님), 없으면 stderr 줄. RL은 `--progress-file <logs>/train.csv` + `--hang-minutes` **60** 권장(train.csv 갱신 간격 최대 약 26분 관찰).
- `--fatal-pattern <regex>`(여러 번): 자식 stderr 줄이 맞으면 죽이고 비정상 종료처럼 재시작(횟수 제한 그대로). 예: `--fatal-pattern "illegal memory access" --fatal-pattern "Sticky error"`.
- Linux: 자식은 별도 프로세스 그룹 + `PR_SET_PDEATHSIG`(SIGINT). 포그라운드 Ctrl+C는 감시가 한 번만 전달 → 저장 후 종료. 감시가 강제 종료돼도 자식은 SIGINT로 저장·push·잠금 해제(그 뒤 stderr 로그는 남지 않음).
- `bc`·`rl`의 Ctrl+C 처리는 pull·모델 로드보다 먼저 설치 → 시작 중 Ctrl+C도 잠금이 있으면 해제(pull은 다음에 이어 받음).
- Windows: 같은 콘솔이라 Ctrl+C가 자식에게 직접. 감시·자식은 물려받은 "Ctrl+C 무시" 설정을 시작 때 지움([042-R10](../spec/2026_10_15_22_24_042-rl-training.md)).

### worker
`uranium-train worker --queue <shared>/queue/<이름>.toml --backend <cuda|rocm|cpu> --priority <n> [--poll-seconds 60] [--log <file>]`(자식 `URANIUM_PYTHON -m uranium bc --run ...`)
- 큐: `[defaults] datasets=[...] config="..."`, `[[run]] id, config, core, stage(s1|s2|ablation-a), init_run, datasets, steps`. 예시는 [7](#큐-운영-060-r8).
- `checkpoints/<id>/DONE` 없는 run을 순서대로. `init_run`이 있으면 그 run에 `DONE`이 생긴 뒤에만(061-R13).
- 양보([061-R14](../spec/2026_10_15_22_24_061-shared-storage.md)): 빈 run이 없으면 우선순위 낮은 작업자의 run에 `locks/<id>.yield`. 다른 run을 잡거나 작업자가 끝나면 자기 요청 삭제. 직접 실행한 `bc --run`에는 안 씀.
  - 만료: 같은 요청 내용을 자기 단조 시계로 30분 관측하면 삭제(mtime 무관). 작업자가 하나도 없고 요청 host가 사라졌으면 `locks/<id>.yield`를 직접 삭제.
- 다른 호스트 잠금 자동 인수([061-R5](../spec/2026_10_15_22_24_061-shared-storage.md)): 같은 heartbeat 내용을 자기 단조 시계로 10분 연속 관측했을 때만(학습 중에는 관측 안 함, 돌아온 뒤 다시 10분). 관측 간격 > 5분이면 끊긴 것 → `--poll-seconds` < 300(넘으면 시작 거부).
- 같은 호스트에서 큐의 run을 수동(`bc --run`)으로 돌리지 않음(같은 host 잠금은 작업자가 자기 것으로 보고 다시 잡음).
- 강제 종료(kill -9, 전원 차단) 뒤 `workers/<host>.json`이 `run=X`로 남음 → 그 host에서 X를 수동 실행하면 양보 요청을 받을 수 있음. 수동 실행 전 상태 파일 삭제 또는 작업자 재기동.
- 공용 드라이브 접근 실패: "모름"으로 보고 종료하지 않고 `--poll-seconds`마다 재확인. run 단위(잠금·`DONE`·양보 요청을 못 읽은 run만 건너뜀, 큐·`checkpoints/` 자체가 안 되면 아무것도 안 잡음). 그동안 자식 실패는 재시작 횟수에 안 넣음(061-R13).
- 상태 `workers/<host>.json` 1분마다(학습 중에도). 장비 구간 `runs/<id>/devices.jsonl`, 집계 `uranium-train devices --run <id>`([061-R15](../spec/2026_10_15_22_24_061-shared-storage.md)).
- Windows CUDA 호스트(우선순위 10): 사용자가 직접 실행, Ctrl+C로 멈춤(저장·push·잠금 해제 약 30초). 자동 시작 없음.
- Linux ROCm 호스트(ROCm f32, 우선순위 1): `tools/worker/uranium-worker.service`(systemd user, `KillSignal=SIGINT`, `TimeoutStopSec=900`, `<shared>/queue/bc.toml` 사용) 또는 `setsid nohup ... &`, 멈출 때 `kill -INT <pid>`. learner로 정하지 않은 호스트에서는 띄우지 않음.
- 코드 교체: 작업자를 먼저 멈추고(Ctrl+C, 서비스는 `systemctl --user stop uranium-worker`) 빌드·설치(`uv pip install -e .`) 뒤 띄움. 도는 중에 바꾸면 옛·새 코드가 섞임.

## 9. RL
([spec 042](../spec/2026_10_15_22_24_042-rl-training.md))

### rl 명령
`python -m uranium rl --run <id> --bc <BC ckpt> [--init <RL ckpt>] --logs <dir> [--config config/rl.toml] [--algo vtrace|ppo] [--backend cuda|rocm|cpu] [--z-table <jsonl>] [--ac1-race terran|protoss|zerg] [--max-agent-steps N] [--max-updates N] [--max-minutes N] [--checkpoint-minutes 30] [--max-batch 32] [--max-wait-ms 5] [--take-over]`
- 학습기·추론 PyTorch, 리그·배정·세그먼트 큐·체크포인트·잠금 Rust(`uranium-league`), 한 프로세스([AMB-000-4](../spec/README.md)). 종료 코드 75 = 다른 호스트 잠금([061-R5](../spec/2026_10_15_22_24_061-shared-storage.md)).
- 체크포인트 `agents/<agent>/`의 `model.safetensors`·`optim-main.safetensors`·`meta.json`, 잠금 [spec 061](../spec/2026_10_15_22_24_061-shared-storage.md), 게임 호스트 `URANIUM_HOSTS_FILE`. 로그 `train.csv`·`games.csv`(042-R2). Burn RL 체크포인트 안 읽음.
- `--init`([AMB-042-13](../spec/README.md)): Main만 RL 체크포인트 폴더(`<shared>/checkpoints/<run>/ckpt-<step>`, 그 안 `agents/main`) 또는 그 `agents/main`에서 가중치만(옵티마이저 새로, update 0). `--bc`는 KL 참조, Exploiter 둘은 BC에서 시작·리셋. 모델 설정이 `--bc`와 다르면 시작 오류. run에 LATEST가 있으면(재시작·`supervise` 재실행 포함) `--init` 무시. 출발 출처는 `meta.json` `train.start`(`from`·`path`·`abs_path`, `--init`이면 `update`·`step`), 재개해도 유지.
- [042-AC1](../spec/2026_10_15_22_24_042-rl-training.md)·[033-AC2](../spec/2026_10_15_22_24_033-reward.md): `--max-agent-steps 10000000`(run 누적, 도달하면 정상 종료, [042-R12](../spec/2026_10_15_22_24_042-rl-training.md)).
- `--logs`가 상대 경로면 현재 폴더 기준 절대 경로(추론 서버 모델 폴더 `<logs>/infer/<agent>`).
- learner 레인 그룹: `.env` `URANIUM_RL_MICRO_LANES`(0 = 전체, 없으면 `config/rl.toml` `micro_lanes`). gradient 합산 순서만 다름([032-R19](../spec/2026_10_15_22_24_032-model-architecture.md)).
- 리플레이: `[replays] sample_every`(기본 100) 배수 게임과 승자 불일치 게임만 `runs/<run>/replays/`([042-R11](../spec/2026_10_15_22_24_042-rl-training.md)).
- 시작·체크포인트마다 디스크 80% 경고([061-R8](../spec/2026_10_15_22_24_061-shared-storage.md)), `runs/*/replays/` 합계 100GB 초과 경고(지우지 않음, [060-R7](../spec/2026_10_15_22_24_060-infra.md)). 보존 `keep_checkpoints`(없으면 5, 0 = 전부).
- 정지 감지([042-R20](../spec/2026_10_15_22_24_042-rl-training.md), [AMB-042-20](../spec/README.md)): `[actors] dead_learner_minutes`(기본 15, 0 = 끔). `Finished` 판 20판 이상인데 그 시간 동안 update 없음, 또는 끝난 판 10판 연속 학습 세션 결정 0 → 체크포인트 없이 오류 종료(supervise가 LATEST에서 재시작). CUDA·HIP 치명 오류(illegal memory access 등)는 한 줄 남기고 저장 없이 코드 70.
- z 테이블: `uranium-train z-table`([033-R8](../spec/2026_10_15_22_24_033-reward.md)).
- 운영 절차(재개·끊김 복구·결과): [checkpoint1_rl](../report/release/2026_11_08_13_35_checkpoint1_rl.md), [m6_ac1](../report/experiment/2026_11_02_23_37_m6_ac1.md).

### 설정 블록 (`config/rl*.toml`)
- 러시 사다리·붕괴 항복([AMB-042-15](../spec/README.md)): `[curriculum] c1_ladder`·`ladder_pick = "f_var"`, `[[ladder]]`(`kind = "t0"`+`build` 또는 `kind = "t1"`+`name`, `race`), `[concede] enabled`([042-R14](../spec/2026_10_15_22_24_042-rl-training.md), 학습 게임만). `games.csv` 마지막 열 `end_reason`(`Concede` = 붕괴 항복). 열이 다른 옛 `games.csv`는 `games-<unix>.csv`로 옮기고 새로. 예 `config/rl_rush1.toml`.
- 사다리 그룹·정찰 보상([AMB-042-16](../spec/README.md)): `[curriculum] ladder_groups = { rush = 0.5, macro = 0.5 }` + `[[ladder]] group`(그룹을 몫대로 먼저, 그 안에서 f_var), `[league] win_rate_decay`, `[reward] pseudo = "scout"`(build·units 채널이 r_scout·r_survive, [033-R11](../spec/2026_10_15_22_24_033-reward.md), `[reward.scout]`·`[reward.survive]`).
  - `"scout"`에 `--z-table`을 주거나 다른 모드 체크포인트를 재개하면 시작 오류. 그룹 이름은 영문·숫자·`_`·`-`만.
  - `games.csv` `end_reason` 뒤 열: [042 "정찰·대응 지표"](../spec/2026_10_15_22_24_042-rl-training.md#정찰대응-지표-2026-11-03-amb-042-16). 같은 지표가 모든 모델 세션 기록(`results.jsonl`·`sessions.jsonl`) `model`에도. 예 `config/rl_rush2.toml`(운영형 그룹 목록은 자리표시).
- 신호 밀도·KL([AMB-042-17](../spec/README.md)):
  - `[league] model_game_max_s`: 모델 상대 RL 판 게임 초 상한(0 = 60분, 0 또는 60~3600, 도달하면 `TimeLimitDraw` 무승부 r_win 0, [042-R15](../spec/2026_10_15_22_24_042-rl-training.md)).
  - `ladder_pick = "band"` 또는 그룹별 표(`{ rush = "band", macro = "f_var" }`, 없는 그룹은 f_var) + `[curriculum.band]` `center`·`width`·`exclude_below`.
  - `[kl]` `init`·`floor`·`half_life_updates`(update는 run 안에서 0부터, 재개는 이어 감), `[kl.head_scale]`(헤드별 배수, 기본 1.0), `[loss] td_lambda`, `[sil]`(자기 모방·막은 판 재학습, [042-R16](../spec/2026_10_15_22_24_042-rl-training.md), 기본 꺼짐, 켜는 것은 사용자 결정).
  - `train.csv` `queue_wait_s` 뒤 열: [042-R2 "진단 로그"](../spec/2026_10_15_22_24_042-rl-training.md). 예 `config/rl_rush3.toml`.
- 러시 미니게임([AMB-042-18](../spec/README.md)): `[league.rush_judge]` `groups`(사다리 그룹 이름, 기본 `[]` = 꺼짐, `ladder_groups`에 있어야 함)·`at_s`(기본 480, 60~3600)·`min_scv`(기본 8, ≥ 1).
  - 그 그룹 RL 학습 판만 `min(86400, at_s × 24)` 프레임에 끝남(모델 상대 판·평가 게임 제외). 그때 `TimeLimitDraw`면 마지막 관측 자기 SCV ≥ `min_scv` → 승, 아니면 패(관측·세션 보고 없으면 결과 없음, [042-R17](../spec/2026_10_15_22_24_042-rl-training.md)). 그 전에 끝난 판은 원래 결과.
  - `[sil] wins_only = true`: 자기 모방이 승 판만(기본 false). `games.csv` 판정 판 `end_reason = RushJudge`, 끝 열 `judge_scv`. 예 `config/rl_rush4.toml`.
- 종족별 참조 KL([044-R9](../spec/2026_11_14_08_18_044-self-game-distillation.md), [042-R22](../spec/2026_10_15_22_24_042-rl-training.md) ①): `[kl.reference]` `zerg`·`terran`·`protoss`·`random` = 참조 체크포인트 폴더(`--bc`와 같은 모델 폴더).
  - 상대 경로는 `URANIUM_SHARED_DIR` 기준(`[league] anchor`와 같음). `--bc`는 작업 폴더 기준이라 다름.
  - 판의 참조 = 학습 세션 시작 `GameInfo` 상대 로비 종족(로비 Random은 Random·Unknown → `random`), 판 안 고정. 키 없는 종족은 `--bc`.
  - 같은 가중치·설정 키는 모델 공유(`--bc`와 같으면 `--bc` 그대로, 비트 동일), 다른 참조는 한 번만 올려 세 learner 공유. 모델 설정이 `--bc`와 다르면 시작 오류. 참조가 둘 이상이면 참조별 레인 그룹, 그룹마다 참조 forward 한 번. Exploiter 리셋은 `--bc`.
  - `train.csv` 끝(공간 통계 뒤) `kl_ref_zerg`·`kl_ref_terran`·`kl_ref_protoss`·`kl_ref_random`(그 종족 레인 step당 KL 평균, 없으면 빈 값). 이 빌드로 재개하면 이전 `train.csv`를 `train-<unix>.csv`로 두고 새 파일.
  - 체크포인트 `meta.json` `train.kl_reference`에 종족별 참조(`path`·`abs_path`·가중치 `sha256`). 예: `config/rl.toml` `[kl.head_scale]` 뒤 주석.

### learner 벤치 (032-R19)
`python -m uranium bench rl --checkpoint <BC ckpt> [--config config/rl.toml] [--algo vtrace|ppo] [--backend cuda|rocm|cpu] [--learner new,reference] [--groups 1,2,4] [--off pinned,spatial_bf16] [--trim] [--updates 10] [--lanes 32] [--entities 64:192] [--templates 8] [--seed 1] [--phases] [--ops 25] [--trace <json>] [--isolate] [--references 1] [--json <log>] [--out <md>]`
- 게임 없이 고정 합성 세그먼트(레인 템플릿을 망이 샘플한 라벨로)로 `Learner.update`: update 시간·최대 GPU 할당(learner만)·상주 메모리, update별 손실·KL·gradient norm. 첫 update 워밍업 제외.
- `--learner reference` = 최적화 전 learner(`uranium.rl.reference`). `--off` = 새 learner 준비 단계 하나씩 끄기; `--off pinned`는 page-locked 준비와 CUDA 비동기 read-back을 함께 끔. `--trim` = [032](../spec/2026_10_15_22_24_032-model-architecture.md) §2.7 엔티티 패딩 자르기(learner 기본 끔: 허용 오차 안에서만 같고 속도 이득 없음).
- `--entities lo:hi` = 템플릿 엔티티 수 범위. [060-R3](../spec/2026_10_15_22_24_060-infra.md) 최악 모양 `--groups 1 --entities 512:512`(N_MAX 512).
- 상주 메모리는 프로세스 누적 최대 → run별은 `--isolate`(learner × 그룹마다 자식 프로세스).
- `--references K`(044-R9, [060-AC7](../spec/2026_10_15_22_24_060-infra.md) ②): 서로 다른 KL 참조 K개(1~4), 레인 로비 종족을 zerg·terran·protoss·random 순으로(새 learner만). 경로 2 수정 RL([044-R11](../spec/2026_11_14_08_18_044-self-game-distillation.md)) 최악 모양: Linux ROCm 호스트에서 `--checkpoint <m2-s2e2 ckpt> --config config/rl_m2.toml --groups 1 --entities 512:512 --isolate --references 2`(참조 S와 `m2-s2e2`). 늘어나는 메모리는 참조 모델 하나분(f32 약 0.1 GiB)이어야 함.
- `--phases`는 표시마다 동기화 → 단계 분해용, 속도 비교에 쓰지 않음.
- 동등성: `python -m uranium bench rl-parity <a.json> [<b.json>] --base-run reference:1 --cand-run new:2`(항목별 최대 절대·상대 차이).

## 10. 단서 진단
근거 [jepa_survey](../report/survey/2026_11_08_19_26_jepa_survey.md) 7·10절. 읽기 전용: teacher forcing만, 정답은 privileged 열에서 오프라인, 모델 입력은 자기 시야. 데이터 경로는 인자로만(`--dataset` 공용 드라이브 기준, `--shards` 폴더).
- 덤프: `python -m uranium diag dump --checkpoint <ckpt|run|none> (--dataset <rel> | --shards <dir>)... --out <dir> [--splits val,test] [--games N] [--backend cuda|rocm|cpu] [--lanes-per-group 8] [--row-until 8640] [--act-until 5760] [--act-every 1] [--read-until F] [--core-points none|gate|full] [--cf [--cf-starts T:1280,2080;P:1720,3030;Z:1800,3240]]`
  - 모델은 trunk + 행동 종류 헤드만(`forward_seq`의 `dists["action"]`). 판마다 순환 상태를 이어 한 번.
  - 행 특징·정답은 `--row-until`(기본 6:00) 전, 모델 pass·활성값은 `--act-until`(기본 4:00) 전, 활성값은 적 기준 건물 토큰이 있는 행만. `--read-until` = 그 프레임까지만 읽음(`--row-until`·`--act-until` 이상, 행 같고 parts 안 순서만 다름).
  - 기준 건물: 테란 배럭, 프로토스 게이트웨이, 저그 스포닝 풀. 적 종족은 첫 행 privileged 일꾼·본진 수로.
  - 지점: 토큰, e_pool, sp_pool, c, h_t, GTrXL 하위(`gate`: x_in, 층마다 a·g1·f·out; `full`은 core_in, a_pre, f_pre도).
  - `--cf`: 판 전체 반사실 pass 6개(이른·늦은 착공, 최대 관측 수 ±1 스칼라만, ±1 스칼라+토큰), 각자 순환 상태. 미완성 건물 명령 id는 종족별 관측값(`UNFINISHED_ORDER`), 실제 분포는 manifest `focus_orders`. `--act-until` 뒤 시작 chunk 확률은 비움(`intervene`은 그보다 늦은 창 거부).
  - 출력 `rows_pNNN.npz`, `acts_pNNN.npz`, `games.json`, `manifest.json`(체크포인트 sha256, 공용 드라이브 경로는 `<shared>/...`).
- 회귀: `python -m uranium diag probes <dump> [--races T,P,Z] [--stages token,e_pool,sp_pool,c,x_in,L1.g1,L1.out,h,hc] [--gain resid|joint] [--boot 500] [--cap 0] [--no-mlp]`
  - 종족별. 정답: HP 비율, 착공 시각, 러시·운영 분류, 적 최대 관측 수, 3:00 전 기준 건물 2개.
  - 지점마다 ridge·MLP(64), 판 단위 5겹 CV, 판 bootstrap 구간, 정답 섞기 대조군, 방해 변수 기준선.
  - `resid`(기본): 방해 변수 ridge 잔차에 지점 ridge(벌점 따로). `joint`: [지점, 방해 변수] 한 ridge(`python/uranium/diag/probe.py`).
- 반사실 개입: `python -m uranium diag intervene <dump(--cf)> [--window 2160,5760]`: 1:30~4:00 마스크 가능 step에서 판마다 P(벙커 ≥ 1), 마린 기대 주문 수, 두 번째 배럭 P. 이른 − 늦은, +1 − −1 평균과 판 bootstrap 구간.
- 죽은 채널([032-R25](../spec/2026_10_15_22_24_032-model-architecture.md)): `python -m uranium diag spatial --checkpoint <ckpt> [--checkpoint <ckpt> ...] (--dataset|--shards)... --out <json> [--split val] [--batches 2] [--skip-batches 2] [--precision auto|f32] [--const-threshold 1e-4]`
  - 공간 단계(stem·stride-2 conv·ResBlock 안팎·bottleneck·s8), 엔티티 FFN, 헤드 ResMlp, 코어 FFN마다 `pos`(0보다 큰 비율)·`dead`(한 번도 안 켜진 채널 수). `S8` 채널·`s_pool` 차원 중 묶음 안 sd가 문턱([AMB-041-14](../spec/README.md)) 아래인 수 `const`와 정밀도(bf16 autocast 반올림 잡음 1e-4 근처 가능).
  - 학습기 연동([041-R25](../spec/2026_10_15_22_24_041-imitation-learning.md)·[042-R21](../spec/2026_10_15_22_24_042-rl-training.md)): `python/uranium/model/monitor.py` `relu_stats(points="spatial")`·`packed`(열 순서 `SPEC_COLUMNS`). 계약(update 전 s0, 상태 저장 없음, no_grad·eval, RNG 미사용, [032-AC15](../spec/2026_10_15_22_24_032-model-architecture.md))은 그 파일 머리말.
- 코어 분해: `python -m uranium diag core-split <dump> --checkpoint <ckpt>`: 미완성 기준 건물 행 지점별 HP R², GRU 게이트마다 (1−z)·x·z·h~ 분산 비율과 R², 변수별 x_in→h_t 잔존 비율.
- 요약: `python -m uranium diag summary --point <경로 지점> [--probes <json>] [--intervene <json>] [--spatial <json>] [--core-split <json>] --out <json>`(스키마 `uranium.diag.summary/1`, `python/uranium/diag/cue/summary.py`).
- 경로 측정([050-R9](../spec/2026_10_15_22_24_050-evaluation.md))의 `cue_diag`(교사 보관분 + 사람 TvT·TvP)·`dead_channels`(학습 로그 `sp_*`, `tools/eval/spatial_logs.py`)는 `python tools/eval/protocol.py diag ... [--run]`이 고정 설정으로 만듦. 확인 `check-diag`, 표 `tools/eval/diag_table.py`([measurement_protocol](measurement_protocol.md) 4). `diag spatial`은 프로토콜 밖([AMB-050-5](../spec/README.md)).
- 설정: 기본 = 10.1 저그 진단(활성값 4:00까지, 판당 상한 없음). 테란·프로토스 사람 리플레이(10.2.3)는 `dump --act-until 8640 --act-every 4`, `probes --cap 60 --boot 300`.
- 재현 기준값(Checkpoint 1, 교사 보관분 433판, Linux ROCm f32): 미완성 풀 HP R² 토큰 0.991·e_pool 0.868·h_t 0.227, s8 양수 0%·128/128, `const` S8 128/128·`s_pool` 256/256.

## 11. 계획 JEPA S0
측정 도구([jepa_planning](../report/plan/2026_11_11_14_56_jepa_planning.md) 6절). 모델·spec·계획 모듈은 바꾸지 않음. 덤프(GPU 1회) → 예측기 학습·표(CPU). 코드: `python/uranium/diag/planning/dump.py`(덤프), `pred.py`(데이터·예측기), `stats.py`(판 bootstrap·AUC·R²·Spearman), `s0.py`(학습·표·판정), `labels.py`(053 결과 정답).

### plan-dump
`python -m uranium diag plan-dump --checkpoint <ckpt> (--dataset <rel> | --shards <dir>)... --out <dir> [--splits val,test] [--games N] [--seed 1] [--backend cuda|rocm|cpu] [--lanes-per-group 8] [--row-until 14400] [--read-until F] [--gk 8,32] [--classes <json>] [--skip-views <json>]`
- `diag dump`와 같은 경로(trunk + 행동 종류 헤드, `action_seq`), 같은 forward에서 JEPA 정답 y = LN([e_pool, s_pool])·g_k(h')·win_open도([032-R24](../spec/2026_10_15_22_24_032-model-architecture.md) 학습 전용 출력 읽기만).
- `--row-until`(기본 10:00) 전 모든 실제 step이 한 행: h', y, privileged(정답용), 내 상태 34열(SCV·병력·건물 종류별 완성+생산 중, 서플라이(BWAPI 반 단위: 마린 = 2), 병력 합, 자기 시야 스칼라에서), 레이블 행동 종류·지연 bucket·가중치, 행동 종류 log-prob, type_mask, win_open, g_8·g_32, 프레임, 판 안 결정 번호.
- `games.json`: 결과, 8:00 SCV ≥ 8, 빌드 분류(privileged 규칙, `--classes` 값도), 리플레이 sha256, 자기 시작 슬롯(`scalar_cat[2]`, sha256과 함께 관점), 맵 계열, 상대 종족.
- 행은 레인 그룹마다 `rows/<필드>.bin`(호스트 메모리 일정). 행당 약 5.2 KB(g_k 포함), `--gk ''`면 약 3.2 KB. `--read-until` 기본 = `--row-until`.
- `--skip-views <관점 표>`(`--drop`과 같은 JSON): `bot` 목록 스트림은 행 안 씀(`games.json` `skipped`). 레인 그룹이 모두 그런 스트림이면 forward도 건너뜀(나머지 행은 전체 덤프와 같음). 모델 판을 모델 명령으로 다시 build해 테란전 봇 관점이 길어질 때 씀.

### plan-s0
`python -m uranium diag plan-s0 --src <이름>:<human|teacher|model>:<train|eval|split>=<덤프> ... --out <dir> [--stage all|train|report] [--ks 8,32] [--steps 3000] [--batch 512] [--lr 1e-3] [--seed 0] [--threads N] [--eval-frac 0.2] [--split-seed 1] [--boot 1000] [--drop <src>=<json>] [--keep-bot-perspective] [--maps 0,1] [--m3b-races Z] [--m3b-window 2880,5040] [--m3b-macro macro] [--tag <이름>] [--th-s-ratio 2 --th-r-min 0.2 --th-r-noise 0.1 --th-m2b 0.9 --th-m2a 0.5 --th-m3a 0.01 --th-own-k 8] [--v1 | --chains drop|keep --chain-kinds teacher,model --own-input on|off --own-target delta|level --s-partner other-game|same-game --k-unit dec|frame]`
- 설계(`planning/pred.py`의 `Design`, jepa_planning 6.8): 기본 = 체인 행 빼기(교사·모델 판), 모든 예측기에 o_t 입력, ô 정답 = o_{t+k} − o_t, S 대조 상대 = 같은 출처 다른 판, k 단위 = 결정. `--v1` = 첫 실행 설계·판정표(다른 설계 플래그·`--th-own-k`와 함께면 거부). 설계는 모델 파일 지문에 들어가 다른 설계로 `--stage report`면 거부.
- 체인 행: 같은 판 앞 결정(dec − 1)이 1프레임 이내 명령(NoOp 아닌 레이블, 무효 포함)인 행. 교사·모델 판에서만 기본 제외(사람은 제외 안 함). 제외 행도 t+k 정답·H 오라클 행동으로는 남음. 수는 `[data] chained rows` 줄.
- `--k-unit frame`: k = 프레임, t+k = k프레임 이상 뒤 첫 행, H 오라클 = 그 사이 결정 행동 비율, g_k 기준선 없음. 결정 k의 프레임 간격은 출처마다 다름(사람 k = 8 약 82프레임, 교사·모델 약 143프레임).
- 역할: `train` 모든 판 학습, `eval` 모든 판 평가, `split` 고정 seed 판 단위 분할(평가 20%, 한 리플레이 두 관점은 같은 쪽). 분할은 (`--split-seed`, `--src` 이름 crc32) → 이름을 바꾸면 분할도 바뀜. 모델 파일에 판·역할 지문 → 다른 분할(`--eval-frac`·`--split-seed`·`--drop`·`--maps`)로 `--stage report`면 거부.
- 관점: 모델 판 shard 테란전은 두 관점 모두 스트림. `model` 출처에 테란 상대 스트림이 있으면 `--drop <src>=config/plan_s0/jepa_diag_tp_views.json`(뷰마다 [리플레이 sha256, 자기 시작 슬롯], `bot`·`model`) 또는 `--keep-bot-perspective` 필요. 목록에 없는 스트림이 있으면 거부.
- S 대조 상대 행: 기본 같은 출처 다른 판의 쓸 수 있는 행(유효 레이블, 뺀 체인 행 아님) 하나, 생성기 하나(seed 12345)가 `--src` 순서로. `--s-partner same-game`(`--v1`) = 같은 판, 전역 순서(`--src` 순서, 판 id).
- 표준화: 정답 sd 바닥(내 상태 0.5개, privileged 0.05), 학습 행 상수 열 제외. ô 머리 지표 10열(SCV, 서플라이 2, 배럭, 벙커, CC, 디폿, 정제소, 팩토리, 병력) + 34열 `o_all`.
- 내 상태 지표: 수준 R²(`o`·`o_all`), 변화 R²(`od`·`od_all`, 보고 기준), `cut` = 1 − err_X / err_U(짝 bootstrap).
- k마다 U·A·A+d·H·S(같은 구조·step·seed)와 M3c용 A(사람만, 모델 판만)를 `<out>/models/`에. 있는 모델은 재학습 안 함(`--retrain`). 같은 스레드 수면 비트 동일.
- 표 `<out>/s0.json`·`s0.md`(`--tag`면 `s0-<tag>.*`): M0, M1(전체·운영 행동 행, 출처 묶음별), M2a·M2b, M3a(시간대별), M3b·M3c, M3d, 6.5 판정표(채점 대상 판 결과·8:00 SCV 따로). 구간은 판 단위 bootstrap 95%. 문턱 기본값은 기본안(미확정).
  - `--v1`이 아니면 1·2행 M1에 내 상태 기준(k = `--th-own-k`, pooled 운영 행동 행 A 변화 오차 감소 구간 > 0이고 S 대조 감소의 `--th-s-ratio`배 이상), 4행은 ŷ와 함께 A·H 변화 오차 감소도. `--th-own-k 0` = 내 상태 기준 끔.
- 메모리: 행당 약 3.2 KB를 출처별 미리 잡은 배열에(사본 없음). 130만 행 약 4.2 GB, 최대 약 5~6 GB.

### 모델 판 출처·결정 행 ([053](../spec/2026_11_18_08_58_053-planning-predictor-data.md)-R5·R6)
- 원본 `<shared>/datasets/replays/raw/`: `dagger_m2/{r1,r2,r3,t1,t2,t2z}`, `distill/d1/{distill-m2-d1,distill-m2-d1-x1}`, `distill/d2/distill-m2-d2`. 폴더마다 `replays/`, `memory/`(기억 기록 `g<id>.jsonl`), `results.jsonl`.
- build(재레이블 없음, 출처마다 `<R>` = 원본 폴더, 저장소 루트, `URANIUM_MAPS_DIR` 필요): `uranium-replay build <R>/replays --src branch --out <O>/shards/<출처>/<출처> --records <O>/build/branch_<출처>.jsonl --memory-log <R>/memory --time-limit 14400 --until-frame 14400 --jobs N`
  - 기억 기록 관점(`players[0]`, 모델)만 씀. 테란전 상대 관점은 records `filtered`(`not the logged model's view`, [041-R28](../spec/2026_10_15_22_24_041-imitation-learning.md)). BC 데이터 아님([040-R25](../spec/2026_10_15_22_24_040-replay-dataset.md)), `--out`은 BC shard 폴더 밖.
  - 확인: records `memory.status` = `ok`, `memory.decision_steps`(재생 L − 2에 더한 결정 스텝).
  - 시간(Windows CPU 표본 15판, 2026-11-18): 판당 job 1개 약 6초(TvZ 러시)~11초(TvT·TvP 10분). 3,800판 `--jobs 6` 약 2시간(추정).
- 덤프: `plan-dump --shards <O>/shards/<출처> --splits train,val,test ...`(manifest 없는 shard는 모두 `train`, 그래도 build가 판을 val·test로도 나눔). CPU 2스레드 약 5.6행/초라 본 덤프는 Linux ROCm 호스트.
- 결정 행: `python -m uranium diag plan-decisions --dump <덤프> --job <R> [--job ...] --records <O>/build/branch_<출처>.jsonl`(`--records` 필수) → `<덤프>/decisions/`(`rows.npz`, `games.json`, `manifest.json`).
  - 기록된 결정(라이브 L)마다 재생 프레임 max(L − 2, 1)의 첫 행. 같은 행 프레임에 두 번째 결정(라이브 1·3)은 행 없음(`collisions`). 10:00 뒤 결정은 `beyond`.
  - `--job` = `replays/`와 `memory/` 또는 `acts/`(분기 줄기 행동 기록, T0 기록은 건너뜀)가 있는 폴더, `results.jsonl`이 있으면 판별 정책 sha256 앞자리(`policy_sha`).
  - 판 상태: `ok`, `no_record`(records에 없음), `no_log`, `incomplete`, `record_<memory.status>`(records), `other_view`(같은 리플레이의 다른 스트림), `no_replay`, `duplicate_replay`. `ok`만 씀(053-R5 ⑥).
- `plan-s0 --src <이름>:model:split:decision=<덤프>`: 그 출처는 결정 행만 예시(행동·지연 = 기록된 결정, NoOp 결정 행 포함, 명령 레이블 행은 없음). k = NoOp이 아닌 결정 수(051-R9 ① 단위). 행동 기록(`acts/`)의 거절 결정도 세지 않음, 기억 기록은 거절 여부가 없어 NoOp만 뺌. `--k-unit frame`과 함께면 거부. H 창 = t부터 t+k 결정 행 앞까지. 정책 확률 `Data.propensity` = 덤프 π, 덤프 체크포인트가 둔 판(`policy_sha`)만, 나머지 NaN.
  - 체인 행 빼기(`--chains drop`)는 결정 행 출처에 적용하지 않음(053-R5 ③, 사용자 결정 2026-11-18). 다른 출처는 그대로.
- `plan-s0 --mmap <로컬 폴더>`: h·y·priv·logp를 `<폴더>/<지문 앞 16자>/<필드>.bin` 로컬 사본으로 메모리 매핑, 나머지 필드는 RAM. 덤프는 파일 읽기로만(매핑 없음). 같은 출처·선택이면 사본 재사용, 다른 선택은 다른 하위 폴더.
  - 거부: 폴더가 `URANIUM_SHARED_DIR` 안(실제 경로 비교). 공용 드라이브가 다른 드라이브 문자·마운트 경로로 붙어 있으면 막지 못함.
  - `s0.json` `memory`·`s0.md` 끝 줄: 프로세스 최대 RSS(Linux VmHWM, Windows 최대 working set), mmap 사본 폴더·재사용·새로 쓴 필드.
  - 053-AC10(1,000만 행 1,000 step)은 단계 2 전체 덤프 때 잼. 2026-11-18 Windows 표본(119만 행, 2스레드): RAM 최대 private 7.7 GB → mmap 4.3 GB, 사본 3.3 GB.

### 053 결과 정답·섞기·검증 ([spec 053](../spec/2026_11_18_08_58_053-planning-predictor-data.md) R2~R4)
`plan-s0` 새 플래그(기본 = S0b 동작, 같은 스레드 수면 모델·표 비트 동일): `[--q-heads held_8m,held_s,alive_10m,win] [--results <src>=<results.jsonl>[,…] | --labels <src>=<labels.json>] [--side <src>=model|scripted] [--mix human=0.2 [--op-x 5 --rare-share 0.001 --rare-x 20]] [--val-frac 0.1 [--val-every 500 --patience 4 --max-steps 20000 --val-rows 50000]] [--split-nets [--score-wd 0.01]]`
- 결과 정답(053-R2): `--q-heads` 이름 목록(기본 `win,scv8`; 쓸 수 있는 이름 `held_8m`·`held_s`·`alive_10m`·`win`·`scv8`). `held_8m`·`held_s`·`alive_10m`이 있으면 교사·모델 출처마다 `--results` 또는 `--labels` 필수, 사람 출처는 덤프에서(`held_8m`·`win`만). `scv8`은 늘 덤프 값.
  - 봇 줄: `--side` 필드(기본 교사 `scripted`, 모델 `model`; 교사 botgen 결과는 교사 T0 세션 지표가 `scripted`에 있음)가 null이 아닌 줄. 판 id마다 마지막 시도(job 형식 `<job>/results/*.jsonl`은 판 id의 시도 번호, plan 폴더 형식은 마지막 줄). 리플레이: job 형식 `<job>/<단계 이름>/replays/g<n>.rep`, plan 폴더 형식 `<results 폴더>/replays/g<id>.rep` → sha256 → 덤프 `games.json` `sha`.
  - 정의: `held_8m` = `scv_8m` ≥ 8 또는 (`won`이고 `end_frame` < 11,520), 8:00 전 패 0, 그 밖 없음. `held_s` = `scv_min_1m` > 4. `alive_10m` = 14,400 전 패 0, 승 또는 끝 ≥ 14,400이면 1. `win` = 승패 난 판만. 잇지 못한 판·`skipped` 관점은 없음(0 아님), 없는 정답은 손실·표에서 뺌.
  - coverage(출처·머리별 정답 판 수·비율·1 수, 학습·검증·평가 판 수) → `<out>/coverage.json`, `[data] label coverage` 줄, `s0.md` 표.
- `python -m uranium diag plan-label --dump <덤프> --kind human|teacher|model [--results <results.jsonl>[,…]] [--side model|scripted] --out <labels.json>`: 같은 정답을 파일로(판 id → 머리 값, `join`·`coverage`·결과 파일별 줄 수·재시도·리플레이 없음). `plan-s0 --labels`가 읽음(`--results`와 같은 모델).
- 섞기(053-R3, `--mix`가 있을 때만): 배치를 출처 종류 몫으로(지정 안 한 종류는 남은 몫을 행 수 비례, 행 없는 종류는 빼고 다시 맞춤), 종류 안 운영 행동 행 × `--op-x`, 학습 행 몫 < `--rare-share`인 운영 행동은 대신 min(`--rare-x`, `--rare-share` ÷ 몫). 목표·기대·실현 몫(뽑은 행 기준) → `<out>/mix.json`, `s0.md` 표.
- 검증(053-R4, `--val-frac` > 0일 때만): 학습 판(`train` 출처 전체, `split` 출처의 평가 아닌 판)의 몫을 판 단위(리플레이 sha256, 두 관점 같은 쪽)로 검증 판으로, seed는 `--split-seed`와 `<src>/val` crc32. `--val-every` step마다 검증 손실(행 균등, 최대 `--val-rows`행), 망마다 최저 step 가중치를 남기고 `--patience`번 연속 안 낮아지면 그 망만 멈춤, 최대 `--max-steps`(`--steps` 대신). 곡선·고른 step → `<out>/val/<모델>.json`, `s0.md` 표. 보고 지표는 평가 판만.
- `--split-nets`: 세계 망(ŷ·p̂·ô, Adam) + 결과 망(q̂, AdamW `--score-wd`) 따로, 같은 배치. 검증 손실 세계 = y + p + o, 결과 = 결과 머리 합. 없으면 한 망·검증 손실은 둘의 합.
- 결과 머리·정답(`q_digest`)·분할은 모델 지문에, 섞기·검증·망 설정은 모델 `training`에 → 다르면 `--stage report`·재사용 거부(`--retrain`).

### 첫 실행 순서 (덤프 Linux ROCm 호스트, 학습·표 Linux CPU 호스트)
1. 표본(수 분, 저장소 루트): 아래 덤프에 `--games 4 --limit-batches 20`, 출력 `local/work/plan_s0/smoke/...` → 같은 호스트 `plan-s0 ... --steps 50 --boot 20 --out local/work/plan_s0/smoke/s0`.
2. 덤프(Linux ROCm 호스트, 저장소 루트, 사용자가 비워 둔 장비에서 학습·측정 job 없을 때). `<ckpt>` = `<shared>/checkpoints/m2-s1/ckpt-39096`, `<D>` = `local/work/plan_s0/m2-s1__ckpt-39096`:
     ```
     export LD_LIBRARY_PATH=<그 호스트의 rocBLAS 경로> HSA_ENABLE_SDMA=0 MALLOC_ARENA_MAX=2 URANIUM_BC_FEED_THREADS=5
     mkdir -p local/work/plan_s0
     python -c "import os, sys, json; from uranium.measure import per_action as p; json.dump(p.load_classes('auto', 'orig'), open('local/work/plan_s0/teacher_classes.json', 'w'))"
     python -m uranium diag plan-dump --checkpoint <ckpt> --dataset replays/shards/stardata --dataset replays/shards/community --dataset replays/shards/scr_ladder --dataset replays/shards/personal --splits train --games 400 --seed 1 --config config/bc.toml --backend rocm --lanes-per-group 4 --out <D>/human_train
     python -m uranium diag plan-dump --checkpoint <ckpt> --dataset replays/shards/stardata --dataset replays/shards/community --dataset replays/shards/scr_ladder --dataset replays/shards/personal --splits val --games 150 --seed 1 --config config/bc.toml --backend rocm --lanes-per-group 4 --out <D>/human_val
     python -m uranium diag plan-dump --checkpoint <ckpt> --shards <cache>/diag_holdout/by_name/hold_orig --shards <cache>/diag_holdout/by_name/hold_macro2 --splits val,test --config config/bc.toml --backend rocm --lanes-per-group 4 --classes local/work/plan_s0/teacher_classes.json --out <D>/teacher
     python -m uranium diag plan-dump --checkpoint <ckpt> --shards <shared>/eval/jepa-diag-tp-20261109/shards/tp --splits train,val,test --config config/bc.toml --backend rocm --lanes-per-group 4 --out <D>/model
     cp -r <D> <shared>/eval/jepa-plan-feas-<날짜>/dump.part && mv <shared>/eval/jepa-plan-feas-<날짜>/dump.part <shared>/eval/jepa-plan-feas-<날짜>/dump
     ```
   - `hold_orig`·`hold_macro2`는 `protocol.py diag`가 그 호스트에 만든 연결 폴더([measurement_protocol](measurement_protocol.md) 4). `teacher_classes.json` = `per_action.py` 러시·운영형 분류([041-R27](../spec/2026_10_15_22_24_041-imitation-learning.md)) {sha256: 분류}.
   - 사람 `--splits train` = S1 학습 분할(모든 맵, `EvalStream`에 대상 맵 선택 없음). 맵 계열은 `games.json` `map`(0·1 = 대상 맵). val 150판 = val500(같은 seed) 앞 150판.
   - 모델 판 shard 520 스트림 중 160은 테란전 봇 관점 → 덤프는 모두 읽고 `plan-s0 --drop model=config/plan_s0/jepa_diag_tp_views.json`이 뺌.
3. 학습·표(Linux CPU 호스트, 저장소 루트, 덤프를 공용 드라이브에서 같은 `<D>`로 먼저 복사):
     ```
     python -m uranium diag plan-s0 --src human-train:human:train=<D>/human_train --src human-val:human:eval=<D>/human_val --src teacher:teacher:split=<D>/teacher --src model:model:split=<D>/model --drop model=config/plan_s0/jepa_diag_tp_views.json --threads 4 --out <D>/s0
     python -m uranium diag plan-s0 <같은 --src·--drop> --threads 4 --stage report --m3b-races T,P --tag tp --out <D>/s0
     cp -r <D>/s0 <shared>/eval/jepa-plan-feas-<날짜>/s0
     ```
   - 둘째 줄 = 모델 판(테란·프로토스 상대만) M3b·M3c 표(모델 판 shard에 저그 상대 없음).
   - 판정표: 1·2행은 "M3b(모델 판)"이 필요 → `s0-tp.md`. 5행(M3c 착각)은 저그 교사 보관분이 있는 `s0.md`. 3·4·6행은 둘 다 같음.
4. 비교 체크포인트(M1만): `<ckpt2>` = `<shared>/checkpoints/m2-lam-base/ckpt-3000`, `<D2>` = `local/work/plan_s0/m2-lam-base__ckpt-3000`로 2의 덤프 네 줄을 같은 판 수로, 3의 첫 줄을 `<D2>`·`--no-m3c`로. 두 `s0.md` M1 표(운영 행동 행, k = 8·32) 비교.
- 시간(추정): 덤프 행당 약 2.3 ms(Linux ROCm, 레인 그룹 4). 표본 합계 약 130만 행(사람 학습 400판 약 51만, val 150판 약 19만, 교사 433판 최대 29만, 모델 판 최대 34만) → 1~2시간(shard 훑기 포함), 4의 비교 덤프 같은 시간 추가. 디스크 약 7 GB(`--gk ''` 약 4 GB). 학습·표: 예측기 14개 × 3,000 step 약 40분~1시간 + 표 20~30분.

### 둘째 실행 S0b
- 위 1~4는 이제 둘째 설계로 돔. 첫 실행 재현은 `plan-s0`에 `--v1`(첫 실행 모델 파일로 `--stage report --v1`도 가능).
- `<J>` = `<shared>/eval/jepa-diag-tp-20261109`(모델 판 원본), `<O>` = `<shared>/eval/jepa-plan-feas-20261112`. 사람·교사 덤프는 첫 실행 것(`<shared>/eval/jepa-plan-feas-20261111/dump/`)을 쓰고 모델 판만 다시.
1. 모델 판 다시 build(Linux CPU 호스트, 저장소 루트, 약 20~25분). 레이블 = 리플레이의 모델 자신 명령(`--relabel` 없음 → 출처 태그 `botgen`, [040-R14](../spec/2026_10_15_22_24_040-replay-dataset.md)):
     ```
     set -a; . ./.env; set +a   # .env가 셸 문법(KEY=값, 공백 없음)이어야 한다. 아니면 URANIUM_SHARED_DIR을 직접 export
     J=$URANIUM_SHARED_DIR/eval/jepa-diag-tp-20261109; O=$URANIUM_SHARED_DIR/eval/jepa-plan-feas-20261112
     export URANIUM_MAPS_DIR=$URANIUM_SHARED_DIR/datasets/replays/build/maps
     mkdir -p $O/records/model_own $O/logs
     for part in games extra; do
       target/release/uranium-replay build $J/tp-$part/replays --src botgen --out $O/shards/model_own/$part \
         --records $O/records/model_own/botgen_$part.jsonl --jobs 6 --until-frame 11520 --time-limit 14400 < /dev/null > $O/logs/build_$part.log 2>&1
     done
     ```
   - `botgen` 5분 기준(`dagger`만 예외) → 5:00 전 종료 리플레이는 `filtered`(shorter than 5 minutes). 테란전 봇 관점 일부는 무효 레이블 20% 이상([040-R19](../spec/2026_10_15_22_24_040-replay-dataset.md))으로 빠짐(봇 관점은 어차피 관점 표가 뺌). 관점 키는 레이블과 무관해 `config/plan_s0/jepa_diag_tp_views.json` 그대로 맞음.
   - 확인: `$O/logs/build_*.log` 마지막 줄 `by_outcome`, records `filtered` 사유.
2. 모델 판 덤프(Linux ROCm 호스트, 약 15~30분). S0 전용 사본(`local/work/s0/repo`를 origin/master로, Rust 변경 없으면 그 사본 `_core` 그대로)에서:
     ```
     export LD_LIBRARY_PATH=<그 호스트의 rocBLAS 경로> HSA_ENABLE_SDMA=0 MALLOC_ARENA_MAX=2 URANIUM_BC_FEED_THREADS=5
     R=$PWD; cd $R/local/work/s0/repo && git fetch -q origin && git checkout -q --detach origin/master && export PYTHONPATH=$PWD/python
     D=$R/local/work/s0b/m2-s1__ckpt-39096
     <venv python> -m uranium diag plan-dump --checkpoint <shared>/checkpoints/m2-s1/ckpt-39096 --shards <O>/shards/model_own --splits train,val,test \
       --skip-views config/plan_s0/jepa_diag_tp_views.json --config config/bc.toml --backend rocm --lanes-per-group 4 --out $D/model_own
     mkdir -p <O>/dump && cp -r $D/model_own <O>/dump/model_own.part && mv <O>/dump/model_own.part <O>/dump/model_own
     ```
3. 학습·표(Linux CPU 호스트, 같은 S0 사본, 본 표 약 25~30분 + `tp` 표 약 7분). 첫 실행 덤프가 `local/work/s0/m2-s1__ckpt-39096`에 없으면 `<shared>/eval/jepa-plan-feas-20261111/dump/`에서 복사:
     ```
     R=$PWD; cd $R/local/work/s0/repo && git fetch -q origin && git checkout -q --detach origin/master && export PYTHONPATH=$PWD/python
     D1=$R/local/work/s0/m2-s1__ckpt-39096; D=$R/local/work/s0b/m2-s1__ckpt-39096
     mkdir -p $D && cp -r <O>/dump/model_own $D/model_own
     SRC="--src human-train:human:train=$D1/human_train --src human-val:human:eval=$D1/human_val --src teacher:teacher:split=$D1/teacher --src model:model:split=$D/model_own --drop model=config/plan_s0/jepa_diag_tp_views.json --threads 4"
     <venv python> -m uranium diag plan-s0 $SRC --out $D/s0b
     <venv python> -m uranium diag plan-s0 $SRC --stage report --m3b-races T,P --tag tp --out $D/s0b
     cp -r $D/s0b <O>/s0b.part && mv <O>/s0b.part <O>/s0b
     ```
   - 판정표는 첫 실행 3과 같음. 내 상태 기준(k = 8) = "M1 내 상태 변화 오차 감소" 표 pooled·운영 행동 행 A 감소.
4. 선택 비교(각 3과 같은 시간, `$SRC` 그대로): 레이블 고침만 `--v1 --out $D/s0b-v1`, 프레임 k `--k-unit frame --ks 144,576 --th-own-k 144 --out $D/s0b-frame`(결정 k = 8·32에 가까운 6초·24초).
- B1 분석용 다시 학습([051-R1](../spec/2026_11_12_12_39_051-branch-games.md) ②, 사용자 결정 2026-11-17): 체크포인트를 `<shared>/checkpoints/m2-gn-t1e2/ckpt-2844`로, 폴더 이름을 `m2-gn-t1e2__ckpt-2844`로. 예측기 입력 h'가 체크포인트마다 달라 첫 실행 2의 사람·교사 덤프도 이 체크포인트로 다시. 출력 `<shared>/eval/jepa-plan-feas-<날짜>/s0b`.

## 12. 래더 수집
([spec 043](../spec/2026_10_16_22_48_043-ladder-replay-collector.md), Windows, SC:R 로그인 상태)
- `uranium-ladder probe [--leaderboard <id> | --toon <t> --gateway <g> [--gamemode <m> --season <s>] | --match-id <link> | --download <url>]`([m4_ladder_probe](../report/milestone/2026_10_16_23_08_m4_ladder_probe.md)), `shape <json> [--pointer /a/0]`.
- 하루 1회 `uranium-ladder collect --plan config/ladder/kr_asia.toml [--limit N] [--dry-run]` → `raw/scr_ladder/`, 집계 `uranium-ladder status`. 0.5 req/s·일 1,000개([043-R4](../spec/2026_10_16_22_48_043-ladder-replay-collector.md)). 랭크 A 이상 = `min_bucket = 6`.
- 스키마 실패 재시도 `collect --plan ... --retry-failed [사유 접두어, 기본 schema:]`([043-R9](../spec/2026_10_16_22_48_043-ladder-replay-collector.md), 같은 속도·한도).
- 로그인된 SC:R 클라이언트 로컬 웹 API 사용. 포트는 `tasklist`·`netstat -ano`(관리자 불필요), `URANIUM_SCR_API_PORT`로 고정 가능.
- 리플레이는 약 30일 뒤 사라지고 프로필엔 최근 25경기 → 매일 실행([AMB-043-8](../spec/README.md)).
- 첫 실행은 `raw\` 전체 SHA-256 인덱스(몇 분), 이후 `ladder\raw_index.json`. 일일 한도에 닿으면 종료, 다시 실행하면 `collect.jsonl` 보고 이어 감.
- `datasets\replays\ladder\collect.jsonl`에 toon 평문 → 공용 드라이브 밖으로 옮기지 않음([AMB-043-6](../spec/README.md)). 원본 응답 플레이어 이름은 출력·커밋 금지([043-R11](../spec/2026_10_16_22_48_043-ladder-replay-collector.md), PowerShell 5.1은 `Get-Content -Encoding UTF8`).
- `failed (schema)` 20번 연속이면 멈춤 → `crates\uranium-ladder\src\schema.rs`만 고침(043-R9·R10).
- 데이터는 [040](../spec/2026_10_15_22_24_040-replay-dataset.md)의 예외로 내부 학습 전용, 재배포 금지([AMB-043-1](../spec/README.md)).

## 13. Windows
([001-R8](../spec/2026_10_15_22_24_001-system-architecture.md), [011-R16](../spec/2026_10_15_22_24_011-bridge-openbw.md), [020-R8](../spec/2026_10_15_22_24_020-env-orchestration.md)) 명령은 PowerShell, 스크립트 `powershell -ExecutionPolicy Bypass -File <script>`. 검증 [m3_windows](../report/milestone/2026_10_16_11_33_m3_windows.md), [m3_remote](../report/milestone/2026_10_16_12_28_m3_remote.md).
- 준비: Visual Studio 2026(또는 2022) "C++를 사용한 데스크톱 개발", CMake 3.16+, Git, Rust MSVC(`rustup default stable-x86_64-pc-windows-msvc`). CMake 생성기 기본 `Visual Studio 18 2026`, VS 2022는 `URANIUM_CMAKE_GENERATOR="Visual Studio 17 2022"`.
- `.gitattributes` 이전 clone이면 패치 재체크아웃: `Remove-Item third_party\patches\*.patch; git checkout -- third_party/patches`.
- `.env` Windows 경로는 작은따옴표. 셸 스크립트 로더 `tools\env.ps1`(`. tools\env.ps1; Import-UraniumEnv <repo>`). PowerShell 변수 이름은 대소문자 무구분(`$T` = `$t`).
- 할당자: Windows `uranium._core`는 mimalloc([032-R19](../spec/2026_10_15_22_24_032-model-architecture.md), 스위치 없음, 결과 동일; Windows 힙 페이지 폴트로 BC 공급이 막히던 문제 해결, [pytorch_opt](../report/bench/2026_10_24_07_03_pytorch_opt.md) 5절).
- 게임 데이터: `casc` CLI 설치에 libclang 필요(LLVM·관리자 권한이 없으면 PyPI `libclang` 휠의 `libclang.dll`을 sha256 대조 뒤 `LIBCLANG_PATH`로). 추출 `content_sha256`·맵 sha256은 `config/assets.toml`과 대조([011-R13](../spec/2026_10_15_22_24_011-bridge-openbw.md)). 맵은 SC:R `Maps\ladder`에서 `2024Season1\(4)Fighting_Spirit_1.3.scx` → `fighting_spirit_1.3.scx`, `Season 10\(4)Circuit Breaker.scx` → `circuit_breaker_1.0.scx`.
- 빌드 산출물:
  - OpenBW+BWAPI `third_party\bwapi\build\bin\Release\{BWAPILauncher.exe, BWAPI.dll, OpenBWData.dll}`(`URANIUM_BWAPI_DIR=third_party\bwapi\build`)
  - 브리지 `bridge\openbw\build\Release\uranium_bridge.dll`(export `gameInit`·`newAIModule`만)
  - T1 봇 `<URANIUM_OSS_BOTS_DIR>\<bot>\{module.dll, module.pdb, VERSION, bwapi-data\}`(Linux와 같은 커밋·패치, `tools\oss\common.ps1`이 봇마다 CMake 프로젝트 → MSVC)
  - Rust `cargo build --release -p uranium-hostd -p uranium-runner -p uranium-agent -p uranium-train`
- hostd: LAN 매칭은 게임마다 `URANIUM_HOSTD_PORT_BASE`(기본 17000)부터 루프백 TCP 포트 2개(011-R16, LISTEN 중 포트 건너뜀). 런처는 게임마다 Job Object(kill-on-close) → hostd가 죽으면 함께 끝남([020-AC3](../spec/2026_10_15_22_24_020-env-orchestration.md)). 봇 크래시는 WinDbg `cdbX64`에 `-y <module.pdb 폴더>`.

## 14. 여러 호스트
([spec 020](../spec/2026_10_15_22_24_020-env-orchestration.md)) 학습 호스트 = 스케줄러(에이전트 서버 내장, `python -m uranium run` 또는 `uranium-env run`) + 로컬 hostd. 원격 Linux 게임 호스트 = hostd만. 원격 브리지는 학습 호스트 에이전트에 접속.
- 원격 게임 호스트:
  1. `git pull --ff-only` → `tools/openbw/build.sh` → `tools/openbw/build_bridge.sh` → `cargo build --release -p uranium-hostd -p uranium-runner -p uranium-agent` → `tools/oss/build_<bot>.sh`(패치가 바뀐 봇은 먼저 `git -C third_party/<bot> reset --hard && git -C third_party/<bot> clean -fdx`).
  2. 게임 데이터를 `local/sc_data/`(`openbw/`, `maps/`)에 복사 → `config/assets.toml`로 검증.
  3. `.env`: `URANIUM_DATA_DIR`·`URANIUM_MAPS_DIR`·`URANIUM_CACHE_DIR`·`URANIUM_BWAPI_DIR`·`URANIUM_BRIDGE_LIB`·`URANIUM_OSS_BOTS_DIR`, `URANIUM_HOSTD_LISTEN=<LAN 주소>:7700`.
  4. `uranium-env calibrate` 한 번 → `mkdir -p local/work/hostd && setsid nohup target/release/uranium-hostd serve > local/work/hostd/hostd.log 2>&1 < /dev/null &`.
  - 원격 `pkill -f uranium-hostd`는 SSH 명령 자신도 죽임 → `pkill -x uranium-hostd`.
- 학습 호스트: `config/hosts.toml`(git 미추적, 형식 `config/hosts.example.toml`)에 로컬·원격 hostd. `.env` `URANIUM_HOSTS_FILE`, `URANIUM_AGENT_LISTEN=0.0.0.0:7600`, `URANIUM_AGENT_PUBLIC_ADDR=<원격에서 닿는 주소>:7600`(로컬 게임도 이 주소).
  - Windows는 들어오는 TCP 7600 방화벽 허용(관리자): `New-NetFirewallRule -DisplayName 'Uranium agent' -Direction Inbound -Action Allow -Protocol TCP -LocalPort 7600 -Program <python 또는 uranium-env.exe> -Profile Any`.
- 응답 없는 hostd는 경고 후 제외, 도중 끊긴 hostd 게임은 Aborted([020-R4](../spec/2026_10_15_22_24_020-env-orchestration.md)). 호스트를 바꾸려면 run 정지 → hosts 파일 수정 → 재시작. `eval-worker`는 hosts 파일을 1분마다 다시 읽음.

## 15. 문제 해결
- **PC가 꺼져 잠금이 남음**: 같은 PC에서 다시 띄우면 이어 감. 다른 PC 작업자는 heartbeat 10분 정지를 보고 자동 인수.
- **재개 거부**(설정·데이터 지문 다름): 설정·데이터가 바뀐 것. 억지로 잇지 말고 원인 확인([7](#시작-검사재개)).
- **"URANIUM_PYTHON is not set"**: `.env`에 venv python 경로.
- **shard 재생성 후**: `uranium-train manifest <shard 폴더>`로 manifest 재생성. manifest에 없는 shard를 출처 폴더에 두지 않음(공유 제공 PC의 로컬 모드는 폴더 목록을 읽어 지문이 달라짐, [061-R12](../spec/2026_10_15_22_24_061-shared-storage.md)).
- **"map hash(es) ... have no map precompute v2"**: [1](#게임-데이터표) 맵 사전계산 복사.
- **다른 호스트 판이 0초에 끝남**: 에이전트 `.env` `URANIUM_AGENT_LISTEN`/`URANIUM_AGENT_PUBLIC_ADDR`를 LAN 주소로([7](#평가데이터-확인)).
- RL 끊김·멈춤: [9](#9-rl) 정지 감지, [8](#supervise).

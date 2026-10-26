# 명령어 상세

CLAUDE.md에는 자주 쓰는 명령만 두고, 옵션과 동작 설명은 여기에 둔다. 요구사항은 괄호 안 spec ID를 본다.

## 빌드·설정
- 설정: `cp .env.example .env` 후 값 입력. Rust 바이너리는 시작 시, 셸 스크립트는 `tools/env.sh`(Windows `tools\env.ps1`)로 읽는다. Windows 경로는 작은따옴표로 감싼다.
- OpenBW/BWAPI(011): `tools/openbw/build.sh` → `third_party/bwapi/build/bin/BWAPILauncher`. 패치는 `third_party/patches/`(0001~0013)를 순서대로 적용한다.
- 브리지(011): `tools/openbw/build_bridge.sh` → `bridge/openbw/build/liburanium_bridge.so`. shim은 500줄 이하. shim과 Rust 쪽은 C ABI로 묶여 있어 항상 함께 빌드한다. 2026-10-19 `uranium_bridge_end` 인자 4개·프로토콜 버전 2(010-R12)부터는 브리지 .so·에이전트·hostd를 모든 PC에서 함께 다시 빌드하고, 옛 shim과 새 라이브러리를 섞지 않는다.
- Windows(001-R8, 아래 "Windows" 절): `tools\openbw\build.ps1` → `tools\openbw\build_bridge.ps1` → `tools\oss\build_<bot>.ps1`.
- T1 봇(021): `tools/oss/build_<bot>.sh`(`ualbertabot`, `steamhammer`, `zzzkbot`, `stardust`, `mcrave`, `iron`) → `<URANIUM_OSS_BOTS_DIR>/<bot>/{module.so, VERSION}`. 상대 ID는 `<bot>@<버전>`(021-R3).
- 게임 데이터 추출(011-R13): `cargo install casc --version 0.1.6 --locked` 후 `cargo run -p casc-extract -- <SC:R Data> <출력>` → `.env` `URANIUM_DATA_DIR`.
- BWAPI 타입 표(013-R1): `cargo run -p bwapi-tables -- third_party/bwapi/bwapi crates/uranium-env/tables`.
- Python(AMB-000-4, 001-R8): 학습·추론은 Python 3.14 + PyTorch 2.14(`python/uranium`). 호스트마다 venv를 만들고(`uv venv --python 3.14 .venv`) PyTorch를 설치한 뒤(CUDA 호스트 `uv pip install --python <venv python> "torch==2.14.*" --index-url https://download.pytorch.org/whl/cu130`, Linux ROCm 호스트는 그 GPU용으로 빌드한 PyTorch) 저장소 루트에서 `uv pip install --python <venv python> -e .`로 `uranium` 패키지와 Rust 확장 `uranium._core`(crate `uranium-py`, maturin)를 설치한다. Rust를 고치면 같은 명령으로 다시 설치한다. `.env`의 `URANIUM_PYTHON`에 그 venv의 python을 적는다(작업자·감시가 학습 자식을 띄울 때 씀).
- 정밀도(032-R8): CUDA는 bf16 혼합, Linux ROCm은 f32. 커널 선택(041-R15): cuDNN benchmark는 CUDA에서만 켜고, ROCm은 MIOpen find-db(`MIOPEN_USER_DB_PATH`)만 쓴다(ROCm에서 benchmark를 켜면 모양이 바뀔 때마다 수 초~수 분 탐색).
- 구현 스위치(AMB-000-4 2단계, 032-R19: 같은 수식, 장비별로 빠른 쪽, 근거 `reports/pytorch_opt.md`): 값을 정하지 않으면 백엔드 기본값(`python/uranium/device.py` `IMPL_DEFAULTS`)이 들어가고, 환경변수·`.env`가 이긴다. 체크포인트 메타 `kernels`에 적용 값이 남는다(041-R15). 층 모양이 가정과 다르면 해당 최적화는 자동으로 라이브러리 경로로 돌아간다.

  | 키 | 내용 | CUDA 호스트 | Linux ROCm 호스트 |
  |---|---|---|---|
  | `URANIUM_LOC_ROWS` | 위치 헤드 사용 행: 호스트 라벨에서, 버킷 | `pow2` | `exact` |
  | `URANIUM_LOC_OUT` | 위치 헤드 1×1 출력 | `conv` | `mulsum` |
  | `URANIUM_BIAS_GRAD` | 거리 bias gather·표 gradient | `embedding` | `table_t` |
  | `URANIUM_CONV_WGRAD` | stem·stride-2·64채널 이하 ResBlock conv 가중치 gradient | `vendor` | `bmm` |
  | `URANIUM_LINEAR_WGRAD` | Linear 가중치 gradient(f32만) | `vendor` | `splitk` |
  | `URANIUM_ENTITY_ATTN` | 엔티티 attention 직접 작성 커널(060-R1) | `fused` | `fused` |
  | `URANIUM_COMPILE` | BC forward torch.compile(Triton 필요) | `default` | `off` |
  | `URANIUM_INFER_GRAPH` | 추론 `act`를 CUDA/HIP graph로 | `on` | `on` |
  | `URANIUM_INFER_GRAPH_MAX_BATCH` | graph로 돌리는 최대 배치(넘으면 eager) | 제한 없음 | `1`(060-R3 추론 2 GiB) |
  | `URANIUM_INFER_COMPILE` | 추론 `act` torch.compile(그래프 안) | `default` | `off` |
  | `URANIUM_H2D`, `URANIUM_ADAM`, `URANIUM_COMPILE_DYNAMIC`, `URANIUM_COMPILE_RNG`, `URANIUM_INFER_STREAM` | 비교 후 기본값 유지(`INFER_STREAM=own`은 Linux ROCm 호스트에서 멈춤) | - | - |

  - 호스트 설정(`.env`, 실측 근거 같은 리포트): CUDA 호스트(RAM을 평소 작업과 나눠 씀) `URANIUM_BC_MICRO_LANES=2`, `URANIUM_BC_PREFETCH=1`, `URANIUM_BC_PACK_BATCH=1`. Linux ROCm 호스트 `URANIUM_BC_MICRO_LANES=1`, `URANIUM_BC_FEED_THREADS=2`, `URANIUM_MALLOC_ARENA_MAX=2`(glibc arena 수 제한: 긴 run에서 arena 단편화로 update마다 메모리가 늘던 것을 막음, 결과 동일, `reports/m6_core_ablation_pt.md` 4절)와 아래 README의 메모리 설정.
  - 장비별 기본값을 다시 고를 때(모델 구조를 바꾼 뒤 등)는 `python -m uranium bench bc --groups 1 --updates 12`를 스위치 값만 바꿔 비교하고, 손실 동등성은 `bench loss`(30 update)로 본다.
  - 직접 작성 커널은 첫 사용 때 빌드해 `URANIUM_KERNEL_DIR`(기본 `~/.cache/uranium/kernels`)에 둔다. Windows는 `cl`이 PATH에 없으면 최신 Visual Studio의 vcvars64 환경을 불러온다.
- Linux ROCm 호스트 실행 환경: 그 호스트용 rocBLAS 경로를 `LD_LIBRARY_PATH`에 두고 `HSA_ENABLE_SDMA=0`.
- 직접 빌드한 PyTorch를 덮어쓰지 않게: torchvision·torchaudio처럼 torch에 의존하는 패키지를 그냥 설치하면 공식 torch를 끌어와 직접 빌드한 torch를 바꿀 수 있다. 필요하면 소스 빌드하거나 `--no-deps`로 설치하고(GPU 연산자는 따로 빌드), 설치 뒤 `python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"`로 소스 빌드 버전과 그 GPU 아키텍처가 남아 있는지 확인한다. `uranium` 패키지는 torch에 의존하지 않고, `[compile]`(triton-windows)은 Windows에서만 설치된다.
- 마지막 Burn 커밋은 태그 `burn-final`이다. Burn 체크포인트(`model.bpk`)는 보관만 하고 읽지 않는다(061-R10).

## 벤치·프로파일
- 벤치 스위트(060-R2): `python -m uranium bench suite [--suite kernels,model,infer] [--filter <regex>] [--precision f32|bf16] [--budget 10] [--out reports/bench_<backend>.md] [--json reports/raw/<name>.json]`. 커널·모델·추론 케이스의 forward / forward+backward 중앙값·p95, CPU f32 대비 오차, 최대 GPU 할당. 반복 횟수는 표준오차가 평균의 1% 아래가 될 때까지 늘린다(워밍업 제외, 케이스당 `--budget`초까지).
- 실데이터 BC update(060-R2, AMB-041-7): `python -m uranium bench bc (--dataset <rel> | --shards <dir>)... [--groups 1,2,4,8] [--updates 20] [--seed 1] [--phases] [--ops 25] [--json <log>]`. 그룹 크기마다 같은 초기 가중치·같은 batch로 update 시간, 데이터 대기, 최대 GPU 할당·상주 메모리(060-R3), update별 손실을 기록한다. 첫 update는 워밍업이라 중앙값에서 뺀다.
- 최적화 후보의 동등성(AMB-000-4 2단계): 기준과 후보를 같은 `--seed`·데이터로 `bench bc --updates 100 --json`한 뒤 `python -m uranium bench loss <기준.json> <후보.json> [--group N] [--window 20]`. 구간 평균 손실의 상대 차이를 낸다(그룹 크기가 다르면 dropout 난수열이 달라진다).
- BC update 구간별 시간과 연산별 장치 시간(060-AC4): `python -m uranium bc-profile (--dataset <rel> | --shards <dir>)... [--backend cuda|rocm|cpu] [--core gtrxl|lstm] [--batches 3] [--ops 25] [--trace <chrome trace json>]`. 학습 중에는 `URANIUM_BC_PROFILE=1`로 update마다 구간 시간을 로그에 쓴다.
- 장치 비교(032-AC1·AC8): `pytest -m gpu tests/python`(CPU f32 대비. CUDA bf16은 잠정 허용치).

## 게임 실행
- hostd(020): `uranium-hostd serve`. 설정 키는 `.env.example` hostd 절.
- 동시 게임 수 측정(020-R10): 게임 호스트에서 `uranium-env calibrate` → `<URANIUM_CACHE_DIR>/hostd_calibration.toml`(`URANIUM_HOSTD_MAX_GAMES`가 없을 때 사용).
- 스케줄러(020): `uranium-env run --plan config/plans/<plan>.toml [--out runs/<dir>]`(스크립트 플레이어만) 또는 `python -m uranium run --plan <plan> [--out runs/<dir>] [--backend cuda|rocm|cpu --max-batch N --max-wait-ms M]`(모델 플레이어 포함) → `games.jsonl`, `results.jsonl`, `summary.md`(Wilson 95%, 정책별 합산), `sessions.md`. 호스트 목록은 `URANIUM_HOSTS_FILE`(예시 `config/hosts.example.toml`). crate는 `uranium-runner`, 바이너리 이름은 `uranium-env`.
  - plan의 `save_replay = true` 게임은 `<out>/replays/g<id>.rep`에 모인다(020-R1, hostd와 같은 커밋으로 빌드).
  - plan의 `surrender = "off" | "shadow" | "on"`(기본 `shadow`)과 `surrender_chat`(기본 `"gg"`, 빈 문자열 = 채팅 없이 퇴장, 79바이트 이하)은 모델 플레이어의 GG 항복이다(031-R12, 020-R14). `shadow`는 판정만 기록하고 나가지 않으며 summary의 Model sessions 표에 판정 성립 세션 수와 그중 승리 수(오항복)를 적는다. `on`은 판정이 성립하면 채팅 후 1초 뒤 퇴장한다(패배). 판정에는 win_open 헤드가 있는 체크포인트가 필요하다(032-R18, 본학습 설정부터). 에이전트 서버 단독 실행은 `--surrender on --surrender-chat gg`. 통합 확인용으로 `URANIUM_TEST_SURRENDER_FRAME=<프레임>`을 주면 `on` 모드에서 그 프레임 이후 첫 결정에 판정과 무관하게 항복한다(테스트 전용).
  - plan의 `zero_last_failed = true`는 모델 플레이어의 `last_failed` 입력을 0으로 고정한다(030-R10). BC 체크포인트 평가 plan(코어 비교, bc-vs-t0)에서 켜고 RL은 끈다. 에이전트 서버 단독 실행은 `--zero-last-failed`.
  - 맵 사전계산 파일 형식이 바뀌었다(`version = 2`, 자원 목록 포함, 030 §2). 옛 `<hash>.toml`·`<hash>.resources.toml`은 없는 것으로 보므로 재build와 `uranium-replay mapgen`으로 다시 만든다.
  - 모델 정책(020-R11): 플레이어 `{ kind = "model", checkpoint = "<URANIUM_SHARED_DIR 기준 경로>", sample = "argmax"|"stochastic" }`. 모델 플레이어가 있는 plan은 추론을 Python이 맡으므로 `python -m uranium run --plan <plan> [--out <dir>] [--backend cuda|rocm|cpu] [--max-batch 32] [--max-wait-ms 5]`로 돌린다(AMB-000-4, 백엔드 기본은 그 PC의 GPU). `uranium-env run`은 스크립트 플레이어만 있는 plan용이다. 대상 맵 사전계산 `uranium-replay mapgen <맵 파일>...`(AMB-031-4). 스모크 `config/plans/smoke-model.toml`(체크포인트 `python -m uranium random-checkpoint --out <dir> [--core lstm|gtrxl] [--win-open]`), 041-AC2 `config/plans/bc-vs-t0.toml`. Burn 체크포인트(`model.bpk`)는 읽지 않는다(061-R10).
- 평가 작업(020-R15, 긴 평가를 추론 장비가 빠졌다 합류하며 이어 가기): 공용 드라이브에 `eval/<job>/job.toml`(`stages = ["plans/01-<plan>.toml", ...]` 차례대로, `hosts = "hosts.toml"`, 둘 다 job 폴더 기준)과 plan 사본·hosts 파일을 둔다. 장비마다 `python -m uranium eval-worker --job eval/<job> --priority <n> [--backend cuda|rocm] [--local-hostd <hosts 파일의 이 PC 주소> --hostd <uranium-hostd 경로>]`. 판 배정 권한(`locks/eval-<job>.json`, 061-R16)을 쥔 작업자만 새 판을 배정하고 추론한다. 우선순위가 더 높은 작업자가 오면 쥔 쪽은 새 판 배정을 멈추고 권한을 넘긴 뒤 진행 중인 판을 마친다(판 도중에 추론 장비가 바뀌지 않음). 모델은 권한을 쥔 동안에만 올린다.
  - Ctrl+C 1회: 진행 중인 판을 끝까지 기록하고 종료(그 PC의 hostd는 `leaving/` 표시로 새 판을 받지 않음). 2회: 진행 중인 판을 `StopGame`으로 끝내고 같은 seed로 다시 하게 한 뒤 종료. 3회: 즉시 종료. 다시 실행하면 합류한다.
  - job의 hosts 파일은 모든 작업자가 같이 읽으므로 호스트마다 LAN 주소를 적는다(127.0.0.1은 작업자마다 다른 PC가 됨, 시작 때 경고).
  - CUDA 호스트(Windows)는 `tools\eval\eval_worker.ps1 -Job eval/<job> -LocalHostd <hosts 파일의 이 PC 주소>`(우선순위 기본 10): 작업자가 hostd를 그 주소로 listen하게 별도 콘솔 창으로 띄워 Ctrl+C가 닿지 않게 하고, 판을 다 마친 뒤 끈다. 로그는 `--log <파일>`(스크립트는 `runs\eval-worker.log`). Linux ROCm 호스트의 대기 작업자는 ROCm 실행 환경(작업자 스크립트와 같은 `LD_LIBRARY_PATH`·`HSA_ENABLE_SDMA=0`)으로 `--backend rocm --priority 2`처럼 띄운다.
  - 다시 하는 판: 브리지 미접속(020-R4), hostd 끊김, 강제 정지. 시간 초과·게임 crash는 결과로 남는다. hostd 판 번호는 `시도 × 10,000,000 + 단계 × 100,000 + 판 번호`.
  - 기록: 장비별 `games/<host>.jsonl`(판마다 `infer_host`·`backend`)·`results/<host>.jsonl`, 진행 중 `inflight/<host>.json`, 리플레이 `<stage>/replays/g<판 번호>.rep`. `python -m uranium eval-status --job eval/<job> [--no-write]`가 단계별 진행 표를 보이고 `<stage>/games.jsonl`·`results.jsonl`(plan 판 번호, 마지막 시도만)·`summary.md`(추론 backend별 판 수 포함)·`sessions.md`를 다시 쓴다. 리그전 집계는 `tools/ablation/league_table.py --run <job>/<stage> <plan>`.
  - 플레이어 이름(020-R16): 리플레이에서 `p<i>-<run 폴더 이름>`(모델), `p<i>-T0-<종족>`, OSS 봇 이름(24바이트에서 자름). plan 플레이어의 `name`으로 바꿀 수 있다. hostd와 스케줄러는 같은 커밋으로 빌드한다(제어 프로토콜 v3).
- 멈춘 Python 프로세스 들여다보기(Linux): `kill -USR1 <pid>`면 모든 Python 스레드의 스택을 그 프로세스의 stderr(작업자·학습 로그)에 쓰고 계속 돈다(root 불필요).
- 액션 시나리오(031-AC1): hostd를 띄운 뒤 `URANIUM_TEST_HOSTD=<hostd 주소> [URANIUM_MAPS_DIR=<투혼 자원 파일이 있는 폴더>] cargo test -p uranium-env --test openbw_actions -- --ignored --nocapture`. 맵 사전계산은 라이브 에이전트와 같은 규칙으로 읽는다(없으면 CC 자리 규칙이 BWAPI와 달라질 수 있음, `uranium-replay mapgen <투혼 .scx>`로 만든다). 투혼에서 스크립트 테란(p0)이 테크를 다 올리며 034의 101개 action_type을 실제 마스크·executor로 한 번 이상 실행하고, 게임이 받아들였는지(`CommandFailed` 없음) 확인한다. 거절은 정당화된 것(유닛이 선 건물 자리 Unbuildable_Location, 한 Act 여러 지출 명령의 자원 부족)만 허용하고, 시야 밖 한 번 본 미네랄에 Smart_Unit·Gather_Unit도 확인한다(011-R20). 상대(p1)는 가만히 있는 테란이다. 비무장 유닛(드랍십·베슬) Attack_Pos가 Move로 바뀌는지도 확인한다(AMB-041-1). 다 덮으면 `StopGame`으로 끝낸다(게임 약 19분, 벽시계 약 1분). 표와 리플레이는 `target/tmp/openbw_actions_031_ac1.{md,rep}`에 쓴다. hostd가 다른 PC면 `URANIUM_TEST_AGENT_LISTEN=0.0.0.0:0`과 `URANIUM_AGENT_PUBLIC_ADDR`를 준다.
- SC:R 실전(012, 설치·절차는 `docs/scr.md`): `tools\scr\build.ps1`(브리지 테스트·빌드·모드 폴더 설치) → `python -m uranium scr --checkpoint <ckpt> [--backend cuda] [--sample stochastic] [--listen 127.0.0.1:7700] [--out runs/scr/sessions.jsonl]`(realtime 세션: Act 기한 관측 + 24프레임 012-R7, `last_failed` 0 012-R14, 항복 기본 `on` 012-R11) → `tools\scr\run.ps1`(samase64 해시 확인 후 SC:R 실행) → 게임 시작은 사람이 한다.
- 셀프매치(M2): `tools/openbw/selfmatch.sh <map> <games> [parallel] [seed_base] [agent args...]`. `SAME_SEED=1`이면 같은 seed, `URANIUM_MAX_FRAMES`로 길이 제한.
- 에이전트 서버: 스크립트 봇 `uranium-agent serve --listen 127.0.0.1:7600 --out runs/x.jsonl`, 모델 `python -m uranium serve --listen 127.0.0.1:7600 --out runs/x.jsonl --policy model --checkpoint <dir> [--backend <b>] [--sample argmax]`, 요약 `uranium-agent summarize <jsonl> --out <md>`.
- OpenBW LAN 매칭(011-R16): Linux는 `$XDG_RUNTIME_DIR/uranium/`의 LOCAL 소켓(108바이트 경로 제한), Windows는 `URANIUM_HOSTD_PORT_BASE`부터 TCP 포트 쌍.

## 리플레이(040)
`uranium-replay`:
- `inspect <rep>`, `convert <rep> <out>`(SC:R → 1.16.1, 맵 CHK 정규화 포함, 013), `check-screp <dir> --screp <exe>`(040-AC1), `ophist <dir>`, `play <rep>`(재생·디싱크 신호), `shard-stats <dir>`(하위 폴더 포함. GameInfo `is_enemy`가 정확히 1명이 아닌 게임(B-12)이나 같은 (리플레이, 시점) 게임이 두 번 있으면(040-R20) 실패), `mapgen`, `roundtrip <dir>`(013-R7).
- `dataset-stats <records.jsonl 또는 폴더>... --out reports/dataset_stats.md`: build 기록으로 맵·매치업·src·stage·분할별 게임·시점 수, 디싱크 폐기율, `result_source` 분포를 쓴다(040-R6, 040-AC2, 040-AC4).
- `build <dir> --src <tag> --records <jsonl> [--jobs N --limit N --part i --parts N --ladder-records <collect.jsonl>]`: 필터 → 013-R7 왕복 검증 → 재생 → 레이블 → shard. 기본 출력 `<URANIUM_SHARED_DIR>/datasets/replays/shards/<src>`. 원본은 `datasets/replays/raw/<src>/`(040-R13). 맵 사전계산은 `URANIUM_MAPS_DIR`에 쓴다.
  - shard는 `URANIUM_SHARD_STAGING_DIR`(로컬)에 먼저 쓰고, 공용 경로에 `<이름>.tmp`로 복사해 fsync 뒤 클라이언트 캐시를 거치지 않고 다시 읽은 SHA-256을 맞춘 뒤 rename한다. 불일치는 3회 재시도, 그래도 실패하면 build가 오류로 멈추고 로컬 사본이 남는다(040-R17).
  - 승패(`result`)는 브리지 승자 → 첫 LEAVE 이탈자 패배 → 기록 승자(`--ladder-records`, `--src scr_ladder`이면 `datasets/replays/ladder/collect.jsonl` 기본) → 판정 불가(0) 순서다(040-R8).
  - 재개는 records 폴더에서 이름이 `<src>`로 시작하는 `.jsonl`을 모두 읽어 건너뛴다. 표본·시험 records는 하위 폴더에 둔다.
  - 내용(SHA-256)이 같은 파일은 출처 전체 정렬 목록(`--part` 분할 전) 기준 첫 파일만 처리하고 나머지는 `duplicate file in this source (040-R20): same file as global index <n>`으로 거른다. 크기가 같은 파일만 읽어 해시하므로 파트마다 출처 전체의 크기 목록을 읽는다. 출처의 일부 파트만 돌리면, 원본(첫 파일)이 돌리지 않은 파트에 있는 리플레이는 빠진다.
  - `config/map_exclude.toml`의 맵(디싱크 10% 초과, 040-R16)은 재생 전에 제외된다. 대상 맵은 넣을 수 없다.
  - build 전에 브리지와 `uranium-replay`를 같은 커밋으로 빌드한다(011-R10).

## BC(041)
학습은 `python -m uranium`(PyTorch 학습기 + Rust 데이터 공급·실행 루프, AMB-000-4), 데이터 확인 명령은 `uranium-train`:
- `python -m uranium bc (--dataset <rel> | --shards <dir>)... (--out <ckpt> | --run <id> [--take-over]) [--backend cuda|rocm|cpu] [--core lstm|gtrxl] [--s2 | --stage s2] [--resume <ckpt> | --init <ckpt|run>] [--steps N]`
  - 체크포인트(061-R10): `model.safetensors`(f32, 정책 버전 = 이 파일의 SHA-256), `optim-main.safetensors`, `meta.json`, `train_state.json`, `run_ident.json`, `config/`.
  - `--core` 기본값은 채택 코어 `gtrxl`이다(032-R17, 2026-10-23 변경. `bc`·`bc-profile`·`random-checkpoint`, 큐의 `core`). 코어 비교의 LSTM 체크포인트를 다룰 때는 `--core lstm`을 적는다(코어가 다르면 체크포인트를 읽을 때 오류로 멈춘다).
  - `--dataset replays/shards/<src>`는 manifest를 검증하고, `--shards`는 검증하지 않는다. 하위 폴더까지 읽는다(041-R9).
  - 파일 순서는 파일마다 `<출처>/<파트>/<파일>` 키로 정한다(041-R13). 폴더는 build 배치여야 한다: 출처 폴더(`<파트>/shard_*`)나 출처들을 담은 상위 폴더(`<출처>/<파트>/shard_*`). 다른 깊이(출처 폴더 바로 아래 shard, 더 깊은 하위 폴더, 두 깊이 혼합)는 파일 이름과 함께 오류로 멈춘다. 이 배치에서는 같은 shard 파일이면 `--dataset`/`--shards`, 폴더를 적은 순서, 출처 폴더를 하나씩 주든 상위 폴더를 주든, 저장 위치(캐시·공용 드라이브·다른 마운트)·대소문자 표기와 상관없이 게임 순서·첫 배치·지문이 같다. 같은 출처나 같은 파일을 두 번 지정하면(대소문자·링크만 다른 경로 포함) 실패한다.
  - `--run`은 061 run(`checkpoints/<id>/`): 잠금·heartbeat·push, 시작 시 `LATEST`에서 재개(데이터 위치 포함, 061-R10). 체크포인트는 `checkpoint_every` update 또는 `checkpoint_minutes`마다, Ctrl+C는 저장 후 종료.
  - 다른 PC의 잠금이 있으면(유효·만료 모두) 종료 코드 75로 거부된다. `--take-over`는 상대 PC가 멈춘 것을 확인한 뒤에만 쓴다: 잠금 상태와 무관하게 즉시 인수하고 경고를 출력하며, 상대가 살아 있으면 그쪽은 다음 heartbeat에서 push 없이 멈춘다(061-R5).
  - 기본 S1(전 소스 train 분할), `--s2`·`--stage s2`는 대상 맵 train 분할(041-R12·R13). S2는 `--init`으로 가중치만 가져온다(041-R4). `--init`에 run 폴더를 주면 그 run에 `DONE`이 있어야 하고 `DONE`이 가리키는 마지막 체크포인트를 쓴다(061-R13). 학습 길이는 `config/bc.toml` `[steps]`(041-R14). `URANIUM_BC_PROFILE=1`이면 구간별 시간 출력.
  - 미리 읽기 깊이 `URANIUM_BC_PREFETCH`(041-R16, 기본 2): batch 하나가 약 4 GB이고 깊이 + 2개까지 메모리에 있다. RAM을 GPU와 나눠 쓰는 Linux ROCm 호스트도 `URANIUM_MALLOC_MMAP_THRESHOLD=1048576`과 함께 2로 둔다(glibc가 다 쓴 batch 블록을 OS에 돌려줘 최대 상주 메모리 약 10 → 6.3 GiB, 데이터 대기 0, `reports/pytorch_opt.md` 2.7). `0`은 데이터 스레드 없이 batch 1개만 둔다(데이터 대기 약 4~6초/update). 다음 레인 그룹의 collate는 항상 작업 스레드에서 미리 한다. batch 내용·순서는 바뀌지 않으므로 깊이가 다른 호스트끼리 같은 run을 이어 해도 된다. 시작 때 `{"prefetch": N, ...}` 줄로 확인한다.
  - 엔티티 인코더 재계산 `URANIUM_BC_ENTITY_CHUNK`(041-R17, 기본 0 = 끔, `torch.utils.checkpoint`)와 호스트 batch 압축 `URANIUM_BC_PACK_BATCH`(041-R18, 기본 0): RAM을 GPU와 나눠 쓰는 Linux ROCm 호스트는 `.env`에 `0`과 `1`로 둔다(AMB-041-8, 2026-10-24: 재계산을 끄면 update가 약 15% 빠르고 최악 모양 GPU 5.7 GiB. 16이면 GPU 1.8 GiB). GPU 메모리와 batch의 호스트 메모리가 줄고, 값은 바뀌지 않으므로(재계산의 dropout 난수 열만 다름) 설정이 다른 호스트끼리 같은 run을 이어 해도 된다. 시작 줄의 `entity_chunk`·`pack_batch`로 확인한다.
  - 레인 그룹 크기 `URANIUM_BC_MICRO_LANES`(041-R3, AMB-041-7): 한 번에 forward·backward하는 레인 수(없으면 설정 `micro_lanes`). CUDA 호스트 2, Linux ROCm 호스트 1(2026-10-24 사용자 결정). gradient는 같고 dropout 난수 묶음만 달라서 호스트끼리 달라도 같은 run을 이어 한다. 시작 줄의 `micro_lanes`로 확인한다.
  - 공급 스레드 `URANIUM_BC_FEED_THREADS`: batch 하나의 레인별 feature 추출과 압축 batch의 레인 그룹 압축 해제를 나눠 맡는 스레드 수(기본 3. 늘리면 학습 스레드와 CPU를 다퉈 오히려 느려진다, `reports/pytorch_opt.md` 2.21). batch는 바뀌지 않는다.
  - `--dataset` pull·검증은 파일마다 `[pull]`/`[verify]` 진행 줄을 stderr에 쓴다(감시 30분 멈춤 판정 대비).
  - 맵 사전계산(B-2): `URANIUM_MAPS_DIR`가 필수다(기본값 없음). 시작할 때(잠금 전) 고른 게임의 맵 해시마다 현재 버전 `<map_hash>.toml`이 있는지 shard를 한 번 훑어 확인하고, 없으면 빠진 해시와 게임 수를 나열하고 실패한다. 로더도 사전계산이 없는 게임은 오류로 처리한다. `bc-eval`·`label-check`도 같다. 라이브 게임(에이전트)은 투혼·서킷브레이커에서 사전계산을 얻지 못하면 세션을 거부하고, 그 외 맵은 경고 후 채널 0으로 진행한다(031).
  - 시작 검사용 shard 훑기 결과(게임별 meta·스텝 수·privileged 없는 스텝 수)는 `<URANIUM_CACHE_DIR>/scans/scan-<지문>.json`에 캐시한다. 지문은 파일별 manifest SHA-256(manifest가 없으면 크기·수정 시각)이라 데이터가 바뀌면 새로 훑는다. `bc`·`bc-eval`·`label-check`·`split-stats`·`bc-epoch`가 같이 쓴다.
  - `--shards` 폴더(상위 폴더를 주면 그 아래 출처 폴더들)에 `manifest.json`이 있으면 스캔 캐시와 데이터 지문이 manifest를 따른다. `--shards`는 manifest를 검증하지 않는다. 그래서 shard를 다시 만들고 manifest를 그대로 두면 지문이 옛 데이터와 같아져 **다른 데이터로 재개가 허용될 수 있다**. shard를 다시 만들면 반드시 `manifest <dir>`로 manifest를 다시 만들거나 옛 `manifest.json`을 지운다. manifest가 없는 사본은 지문이 크기 기준이라 `--dataset`으로 시작한 run을 이어 받지 못한다(재개 거부).
  - 재개 검사(B-3): 체크포인트 `train_state.json`의 `extra.run`에 stage(`S1`/`S2`/`AblationA`), 설정 지문(결과에 영향을 주는 설정만 담은 화이트리스트 `ResultSettings`, `[steps]` 반영. 실행 방식 설정은 `EXECUTION_ONLY`에 따로 두고, 새 필드는 둘 중 하나로 분류해야 테스트가 통과한다), 데이터 지문(파일별 manifest SHA-256, manifest가 없으면 크기)을 남긴다. 재개할 때 하나라도 다르면 잠금을 풀고 거부한다(우회 옵션 없음, 새 run 또는 `--init`).
  - 체크포인트 보존(061-R6): 설정 `keep_checkpoints`(없으면 5, 0이면 전부 보존). `config/bc_core_ablation.toml`은 0. 정리(삭제) 실패는 경고만 하고 학습을 계속한다.
  - 디스크 경고(061-R8): 학습 시작과 체크포인트 저장마다 공용 드라이브·로컬 캐시(로컬 `--out`이면 그 폴더)의 물리 디스크 사용률이 80% 이상이면 경고한다. 학습은 멈추지 않는다.
  - 로그 줄의 `no_result_games`는 지금까지 시작한 게임 중 승패 판정 불가(`result_source = none`)로 value 가중치 0인 게임 수다(041-R7).
- `python -m uranium bc-eval (--dataset <rel> | --shards <dir>)... --checkpoint <ckpt|run> [--split test|val] [--games N --seed 1] [--backend cuda|rocm|cpu] [--out <json>]`(041 지표, 서렌더 재현 041-AC4 포함, `--out`이면 JSON을 파일에도 씀), `python -m uranium bc-profile`(벤치 절).
- `uranium-train`: `bc-epoch`(1 epoch update 수, 게임 시작 NoOp 스텝 포함), `split-stats`, `label-check [--games N]`(034-AC3; 출력에 `target_causes`(대상 없음·위치 거절 원인: 이전 목격 여부와 경과 프레임, 건물 종류별 막는 조건, 실제로 지어졌는지)와 `per_game`(게임별 무효 수) 포함, 액션 감사 F23·F24), `manifest <dir> [--verify]`(061-R7. 만들 때 하위 폴더의 shard를 모두 끝까지 디코드해 보고, 실패하면 실패한 shard를 나열하고 쓰지 않는다, 040-R17). 모두 `--dataset`/`--shards`를 받는다.
  - `bc-eval --games N`은 분할 전체를 고정 seed(`--seed`, 기본 1)로 섞어 앞의 N게임을 고른다. 체크포인트·코어와 무관하게 같은 게임이다(032-R17 ②). 단위(E-9): `delay_mae_frames`는 프레임(bucket 값), `delay_mae_buckets`는 bucket 번호, `nll.<head>`는 그 헤드를 쓰는 스텝(`nll_steps.<head>`)만의 평균이다. 출력 `units`에 같은 설명이 들어 있다.
  - `split-stats` 출력: `splits`(소스·맵 그룹·분할별 게임 수), `sources`(소스별 `privileged` 없는 게임·스텝 수와 비율(E-10), `result_source`별 게임 수(040-R8)).
- 하이퍼파라미터 `config/bc.toml`(GPU 메모리는 `micro_lanes`, `pad_entities`, `grad_checkpointing`), 코어 비교 `config/bc_core_ablation.toml`, 보상 `config/reward.toml`. `grad_clip`은 모든 파라미터 gradient를 한 벡터로 본 전역 L2 norm 기준이고, 레인 그룹 gradient를 합친 뒤 optimizer step 전에 한 번 적용한다(041-R3, RL `config/rl.toml`도 같음).

## 감시·작업자(060-R6, 061-R13~R15)
- `uranium-train supervise [--hang-minutes 30 --max-restarts 5 --window-minutes 60 --delay-seconds 60 --progress-file <file> --log <file>] -- <bc|rl 인자>`: 자식은 `URANIUM_PYTHON -m uranium <bc|rl 인자>`다(`.env`에 없으면 시작하지 않는다). 비정상 종료·멈춤이면 같은 run을 재시작한다. 종료 코드 0·130·75(잠금을 얻지 못함: 다른 호스트 잠금, 획득 경쟁에서 짐, 양보 예약)·76(양보)은 재시작하지 않는다.
  - Linux: 자식 학습은 별도 프로세스 그룹으로 뜨고 `PR_SET_PDEATHSIG`(SIGINT)가 걸린다. 터미널 포그라운드 Ctrl+C도 감시 프로세스가 한 번만 전달하므로 저장 후 끝난다. 감시 프로세스가 강제 종료돼도 자식은 SIGINT를 받아 저장·push·잠금 해제 후 끝난다(060-R6). 이때 자식의 stderr 파이프는 닫혀 있어서 그 뒤 로그는 남지 않는다(학습 코드는 stderr 쓰기 실패를 무시한다). `bc`·`rl`의 Ctrl+C 처리는 데이터셋 pull·모델 로드보다 먼저 설치되므로, 시작 중 Ctrl+C도 잠금이 있으면 해제하고, 없으면 그냥 끝난다(pull은 다음에 이어서 받는다). Windows: 같은 콘솔이라 Ctrl+C가 자식에게 직접 간다. 감시 프로세스와 자식은 시작할 때 부모에게서 물려받은 "Ctrl+C 무시" 설정을 지운다(042-R10, 지우기 전에는 Ctrl+C가 자식에게 닿지 않아 저장 없이 계속 학습했다).
- `uranium-train worker --queue <shared>/queue/<이름>.toml --backend <cuda|rocm|cpu> --priority <n> [--poll-seconds 60] [--log <file>]`(자식 `URANIUM_PYTHON -m uranium bc --run ...`)
  - 큐: `[defaults] datasets=[...] config="..."`, `[[run]] id, config, core, stage(s1|s2|ablation-a), init_run, datasets, steps`.
  - 끝나지 않은(`checkpoints/<id>/DONE` 없음) 잡을 수 있는 run을 순서대로 돌린다. `init_run`이 있는 run은 `init_run`에 `DONE`이 생긴 뒤에만 잡는다(061-R13). 빈 run이 없으면 우선순위가 낮은 호스트의 작업자가 돌리는 run에 양보 요청(`locks/<id>.yield`)을 쓰고, 다른 run을 잡거나 작업자가 끝나면 자기 요청을 지운다(061-R14). 직접 실행한 `bc --run`에는 양보 요청을 쓰지 않는다.
  - 양보 요청 만료: 작업자가 같은 요청 내용을 자기 단조 시계로 30분 동안 관측하면 만료로 보고 지운다(파일 mtime 무관). 작업자가 하나도 없을 때 요청한 host가 사라졌으면 `locks/<id>.yield`를 직접 지운다.
  - 다른 호스트 잠금의 자동 인수: 작업자가 자기 단조 시계로 같은 heartbeat 내용을 10분 동안 연속 관측했을 때만 한다(호스트 시계·파일 mtime 무관, 061-R5). 학습하는 동안에는 관측하지 않으므로, 돌아온 뒤 다시 10분을 관측한다. 관측 간격이 5분(만료의 절반)을 넘으면 관측이 끊긴 것으로 보므로 `--poll-seconds`는 300 미만이어야 한다(넘으면 작업자가 시작을 거부한다).
  - 같은 호스트에서 큐에 있는 run을 수동으로(`bc --run`) 실행하지 않는다. 같은 host의 잠금은 작업자가 자기 것으로 보고 다시 잡는다.
  - 알려진 한계: 작업자를 강제 종료(kill -9, 전원 차단 등)하면 상태 파일 `workers/<host>.json`이 `run=X`인 채로 남는다. 그 host에서 X를 `bc --run`으로 수동 실행하면 다른 작업자는 이를 작업자가 돌리는 run으로 보고 양보 요청을 보낼 수 있다(061-R14). 수동 실행 전에 남은 상태 파일을 지우거나, 그 host의 작업자를 다시 띄워 상태 파일을 갱신한다.
  - 공용 드라이브를 읽거나 쓰지 못하면 상태를 "모름"으로 보고 종료하지 않으며 `--poll-seconds`마다 다시 본다. 모름은 run 단위다: 잠금·`DONE`·양보 요청을 읽지 못한 run만 그 주기에 건너뛰고, 공용 드라이브 자체(큐, `checkpoints/`)에 접근할 수 없으면 아무 run도 잡지 않는다. 공용 드라이브가 없는 동안 자식이 실패한 것은 재시작 횟수에 넣지 않는다(061-R13).
  - 상태 파일 `workers/<host>.json`은 1분마다(자식 학습 중에도) 갱신한다. 장비 구간은 `runs/<id>/devices.jsonl`, 집계는 `uranium-train devices --run <id>`(host별 update 수·비율, 061-R15).
  - Windows CUDA 호스트(우선순위 10): 사용자가 직접 실행하고 Ctrl+C로 멈춘다(저장·push·잠금 해제, 약 30초). 자동 시작 구성은 없다(060-R6).
  - Linux ROCm 호스트(ROCm f32, 우선순위 1): `tools/worker/uranium-worker.service`(systemd user, `KillSignal=SIGINT`, `TimeoutStopSec=900`) 또는 `setsid nohup ... &`, 멈출 때 `kill -INT <pid>`. learner로 정하지 않은 호스트에서는 띄우지 않는다.
  - 코드 교체: 작업자를 먼저 멈추고(Ctrl+C, Linux 서비스는 `systemctl --user stop uranium-worker`) 다시 빌드·설치(`uv pip install -e .`)한 뒤 띄운다. 작업자가 도는 중에 바꾸면 돌고 있는 자식 학습은 옛 코드로 계속 돌고 다음 자식부터 새 코드가 떠서 두 버전이 섞인다.

## RL(042)
- `python -m uranium rl --run <id> --bc <BC ckpt> --logs <dir> [--config config/rl.toml] [--algo vtrace|ppo] [--backend cuda|rocm|cpu] [--z-table <jsonl>] [--ac1-race terran|protoss|zerg] [--max-agent-steps N] [--max-updates N] [--max-minutes N] [--checkpoint-minutes 30] [--max-batch 32] [--max-wait-ms 5] [--take-over]`
  - 학습기·추론은 PyTorch, 리그·게임 배정·세그먼트 큐·체크포인트·잠금은 Rust(`uranium-league`)가 한 프로세스에서 돈다(AMB-000-4). 종료 코드 75 = 다른 호스트의 잠금(061-R5).
  - 체크포인트(`agents/<agent>/`의 `model.safetensors`·`optim-main.safetensors`·`meta.json`)·잠금은 061, 게임 호스트는 `URANIUM_HOSTS_FILE`. 로그 `train.csv`·`games.csv`(042-R2). Burn RL 체크포인트는 읽지 않는다.
  - 042-AC1·033-AC2는 `--max-agent-steps 10000000`(run 누적, 도달하면 정상 종료, 042-R12).
  - 리플레이: `[replays] sample_every`(기본 100)의 배수 게임과 승자 불일치 게임만 `runs/<run>/replays/`에 남긴다(042-R11).
  - 학습 시작과 체크포인트마다 디스크 사용률(80% 경고, 061-R8)과 `runs/*/replays/` 합계(예산 100GB 초과 시 경고, 지우지 않음, 060-R7)를 확인한다. 보존 개수는 `config/rl.toml`의 `keep_checkpoints`(없으면 5, 0이면 전부).
- z 테이블 `uranium-train z-table`(033-R8).

## 래더 리플레이 수집(043, Windows, SC:R 로그인 상태)
- `uranium-ladder probe [--leaderboard <id> | --toon <t> --gateway <g> [--gamemode <m> --season <s>] | --match-id <link> | --download <url>]`(`reports/m4_ladder_probe.md`), `shape <json> [--pointer /a/0]`.
- 하루 1회 `uranium-ladder collect --plan config/ladder/kr_asia.toml [--limit N] [--dry-run]` → `raw/scr_ladder/`, 집계 `uranium-ladder status`. 0.5 req/s·일 1,000개(043-R4). 랭크 A 이상 = `min_bucket = 6`. 스키마 실패 경기 재시도 `collect --plan ... --retry-failed [사유 접두어, 기본 schema:]`(043-R9, 같은 속도·일일 한도).
- 운영 메모(043): 로그인된 SC:R 클라이언트의 로컬 웹 API를 쓴다. 포트는 `tasklist`·`netstat -ano`로 찾고(관리자 권한 불필요) `URANIUM_SCR_API_PORT`로 고정할 수 있다. 리플레이 파일은 약 30일 뒤 사라지고 프로필에는 최근 25경기만 있어 매일 돌린다(AMB-043-8). 첫 실행은 `raw\` 전체 SHA-256 인덱스를 만들어 몇 분 걸리고 이후 `ladder\raw_index.json`을 쓴다. 일일 한도에 닿으면 그날 수집을 마치고, 다시 실행하면 `collect.jsonl`을 보고 이어 간다. 기록 `datasets\replays\ladder\collect.jsonl`에는 toon이 평문으로 있으니 공용 드라이브 밖으로 옮기지 않는다(AMB-043-6). 원본 응답의 플레이어 이름은 출력·커밋하지 않는다(043-R11, PowerShell 5.1은 `Get-Content -Encoding UTF8`). 응답 형식이 바뀌어 `failed (schema)`가 20번 연속이면 멈추고, `crates\uranium-ladder\src\schema.rs`만 고친다(043-R9·R10). 데이터 정책은 040의 예외로 내부 학습 전용, 재배포하지 않는다(AMB-043-1).

## Windows (001-R8, 011-R16, 020-R8)
명령은 PowerShell 기준, 스크립트는 `powershell -ExecutionPolicy Bypass -File <script>`. 검증 결과 `reports/m3_windows.md`·`reports/m3_remote.md`.
- 준비: Visual Studio 2026(또는 2022)의 "C++를 사용한 데스크톱 개발", CMake 3.16+, Git, Rust MSVC 툴체인(`rustup default stable-x86_64-pc-windows-msvc`). 빌드 스크립트의 CMake 생성기 기본값은 `Visual Studio 18 2026`, VS 2022는 `URANIUM_CMAKE_GENERATOR="Visual Studio 17 2022"`. `.gitattributes`가 생기기 전에 clone했다면 패치를 다시 체크아웃한다(`Remove-Item third_party\patches\*.patch; git checkout -- third_party/patches`). `.env`의 Windows 경로는 작은따옴표, 셸 스크립트 로더는 `tools\env.ps1`(`. tools\env.ps1; Import-UraniumEnv <repo>`). PowerShell 변수 이름은 대소문자를 구분하지 않는다(`$T`와 `$t`는 같은 변수).
- 할당자: Windows의 확장 `uranium._core`는 mimalloc을 쓴다(032-R19, 스위치 없음, 결과 동일). Windows 힙이 BC 공급의 큰 버퍼를 매번 OS에 돌려줘 페이지 폴트로 데이터 공급이 막히던 것을 없앴다(`reports/pytorch_opt.md` 5절).
- 게임 데이터: `casc` CLI 설치에 libclang이 필요하다(LLVM이 없고 관리자 권한도 없으면 PyPI `libclang` 휠의 `libclang.dll`을 sha256 대조 뒤 `LIBCLANG_PATH`로). 추출 결과 `content_sha256`과 맵 sha256은 `config/assets.toml`과 대조한다(011-R13). 맵은 SC:R `Maps\ladder`에서 `2024Season1\(4)Fighting_Spirit_1.3.scx` → `fighting_spirit_1.3.scx`, `Season 10\(4)Circuit Breaker.scx` → `circuit_breaker_1.0.scx`.
- 빌드 산출물: OpenBW+BWAPI `third_party\bwapi\build\bin\Release\{BWAPILauncher.exe, BWAPI.dll, OpenBWData.dll}`(`URANIUM_BWAPI_DIR=third_party\bwapi\build`), 브리지 `bridge\openbw\build\Release\uranium_bridge.dll`(export는 `gameInit`·`newAIModule`만), T1 봇 `<URANIUM_OSS_BOTS_DIR>\<bot>\{module.dll, module.pdb, VERSION, bwapi-data\}`(Linux와 같은 커밋·패치, `tools\oss\common.ps1`이 봇마다 CMake 프로젝트를 만들어 MSVC로 빌드), Rust `cargo build --release -p uranium-hostd -p uranium-runner -p uranium-agent -p uranium-train`.
- hostd: LAN 매칭은 게임마다 `URANIUM_HOSTD_PORT_BASE`(기본 17000)부터 루프백 TCP 포트 2개(011-R16, LISTEN 중인 포트는 건너뜀). 런처는 게임마다 Job Object(kill-on-close)에 들어가 hostd가 죽으면 함께 끝난다(020-AC3). 봇 크래시는 WinDbg `cdbX64`에 `-y <module.pdb 폴더>`.

## 여러 호스트로 게임 돌리기 (020 호스트 구성)
학습 호스트가 스케줄러(에이전트 서버 내장, `python -m uranium run` 또는 `uranium-env run`)와 로컬 hostd를, 원격 Linux 게임 호스트는 hostd만 돌린다. 원격 게임의 브리지는 학습 호스트의 에이전트에 접속한다.
- 원격 게임 호스트: `git pull --ff-only` → `tools/openbw/build.sh` → `tools/openbw/build_bridge.sh` → `cargo build --release -p uranium-hostd -p uranium-runner -p uranium-agent` → `tools/oss/build_<bot>.sh`(패치가 바뀐 봇은 먼저 `git -C third_party/<bot> reset --hard && git -C third_party/<bot> clean -fdx`). 게임 데이터를 복사해 `config/assets.toml`로 검증, `.env`에 `URANIUM_DATA_DIR`·`URANIUM_MAPS_DIR`·`URANIUM_CACHE_DIR`·`URANIUM_BWAPI_DIR`·`URANIUM_BRIDGE_LIB`·`URANIUM_OSS_BOTS_DIR`, `URANIUM_HOSTD_LISTEN=<LAN 주소>:7700`. `uranium-env calibrate` 한 번 뒤 `setsid nohup target/release/uranium-hostd serve > /tmp/hostd.log 2>&1 < /dev/null &`. 원격에서 `pkill -f uranium-hostd`는 SSH 명령 자신도 죽이므로 `pkill -x uranium-hostd`.
- 학습 호스트: `config/hosts.toml`(git 미추적, 형식 `config/hosts.example.toml`)에 로컬·원격 hostd를 적고 `.env`에 `URANIUM_HOSTS_FILE`, `URANIUM_AGENT_LISTEN=0.0.0.0:7600`, `URANIUM_AGENT_PUBLIC_ADDR=<원격에서 닿는 주소>:7600`(로컬 게임도 이 주소를 쓴다). Windows는 들어오는 TCP 7600 방화벽 허용이 필요하다(관리자 권한, `New-NetFirewallRule -DisplayName 'Uranium agent' -Direction Inbound -Action Allow -Protocol TCP -LocalPort 7600 -Program <python 또는 uranium-env.exe> -Profile Any`). 응답하지 않는 hostd는 경고 후 빼고, 도중에 끊긴 hostd의 게임은 Aborted(020-R4). 호스트를 바꾸려면 run을 멈추고 hosts 파일을 고친 뒤 다시 시작한다. 평가 작업(`eval-worker`)은 hosts 파일을 1분마다 다시 읽고 끊긴 호스트도 다시 확인한다.

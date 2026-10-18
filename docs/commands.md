# 명령어 상세

CLAUDE.md에는 자주 쓰는 명령만 두고, 옵션과 동작 설명은 여기에 둔다. 요구사항은 괄호 안 spec ID를 본다.

## 빌드·설정
- 설정: `cp .env.example .env` 후 값 입력. Rust 바이너리는 시작 시, 셸 스크립트는 `tools/env.sh`(Windows `tools\env.ps1`)로 읽는다. Windows 경로는 작은따옴표로 감싼다.
- OpenBW/BWAPI(011): `tools/openbw/build.sh` → `third_party/bwapi/build/bin/BWAPILauncher`. 패치는 `third_party/patches/`(0001~0013)를 순서대로 적용한다.
- 브리지(011): `tools/openbw/build_bridge.sh` → `bridge/openbw/build/liburanium_bridge.so`. shim은 500줄 이하.
- Windows(001-R8, `docs/windows.md`): `tools\openbw\build.ps1` → `tools\openbw\build_bridge.ps1` → `tools\oss\build_<bot>.ps1`.
- T1 봇(021): `tools/oss/build_<bot>.sh`(`ualbertabot`, `steamhammer`, `zzzkbot`, `stardust`, `mcrave`, `iron`) → `<URANIUM_OSS_BOTS_DIR>/<bot>/{module.so, VERSION}`. 상대 ID는 `<bot>@<버전>`(021-R3).
- 게임 데이터 추출(011-R13): `cargo install casc --version 0.1.6 --locked` 후 `cargo run -p casc-extract -- <SC:R Data> <출력>` → `.env` `URANIUM_DATA_DIR`.
- BWAPI 타입 표(013-R1): `cargo run -p bwapi-tables -- third_party/bwapi/bwapi crates/uranium-env/tables`.
- CUDA 실행은 `CUDA_PATH`가 필요하다(없으면 커널 컴파일 시 libdevice 오류). CUDA 빌드(`--features cuda`)는 CUDA PC에서만 한다.

## 벤치
- Burn/Vulkan(060-R2): `cargo run -p bench --release -- --backend vulkan [--quick] [--filter <name>]` → `reports/bench_bc250.md`.
- 모델(032-AC2/AC3): `cargo run -p bench --release [--features cuda] -- --backend <vulkan|cuda> --filter net_ [--f16]`.

## 게임 실행
- hostd(020): `uranium-hostd serve`. 설정 키는 `.env.example` hostd 절.
- 동시 게임 수 측정(020-R10): 게임 호스트에서 `uranium-env calibrate` → `<URANIUM_CACHE_DIR>/hostd_calibration.toml`(`URANIUM_HOSTD_MAX_GAMES`가 없을 때 사용).
- 스케줄러(020): `uranium-env run --plan config/plans/<plan>.toml [--out runs/<dir>] [--backend cuda|vulkan|flex --max-batch N --max-wait-ms M]` → `games.jsonl`, `results.jsonl`, `summary.md`(Wilson 95%, 정책별 합산), `sessions.md`. 호스트 목록은 `URANIUM_HOSTS_FILE`(예시 `config/hosts.example.toml`). crate는 `uranium-runner`, 바이너리 이름은 `uranium-env`.
  - plan의 `save_replay = true` 게임은 `<out>/replays/g<id>.rep`에 모인다(020-R1, hostd와 같은 커밋으로 빌드).
  - 모델 정책(020-R11): 플레이어 `{ kind = "model", checkpoint = "<URANIUM_SHARED_DIR 기준 경로>", sample = "argmax"|"stochastic" }`. 추론 백엔드 기본 `URANIUM_AGENT_BACKEND`. 대상 맵 사전계산 `uranium-replay mapgen <맵 파일>...`(AMB-031-4). 스모크 `config/plans/smoke-model.toml`(체크포인트 `uranium-agent random-checkpoint --out <dir>`), 041-AC2 `config/plans/bc-vs-t0.toml`.
- 셀프매치(M2): `tools/openbw/selfmatch.sh <map> <games> [parallel] [seed_base] [agent args...]`. `SAME_SEED=1`이면 같은 seed, `URANIUM_MAX_FRAMES`로 길이 제한.
- 에이전트 서버: `uranium-agent serve --listen 127.0.0.1:7600 --out runs/x.jsonl [--policy model --checkpoint <dir> --backend <b> --sample argmax]`, 요약 `uranium-agent summarize <jsonl> --out <md>`.
- OpenBW LAN 매칭(011-R16): Linux는 `$XDG_RUNTIME_DIR/uranium/`의 LOCAL 소켓(108바이트 경로 제한), Windows는 `URANIUM_HOSTD_PORT_BASE`부터 TCP 포트 쌍.

## 리플레이(040)
`uranium-replay`:
- `inspect <rep>`, `convert <rep> <out>`(SC:R → 1.16.1, 맵 CHK 정규화 포함, 013), `check-screp <dir> --screp <exe>`(040-AC1), `ophist <dir>`, `play <rep>`(재생·디싱크 신호), `shard-stats <dir>`(GameInfo `is_enemy`가 정확히 1명이 아닌 게임이 있으면 실패, B-12), `mapgen`, `roundtrip <dir>`(013-R7).
- `dataset-stats <records.jsonl 또는 폴더>... --out reports/dataset_stats.md`: build 기록으로 맵·매치업·src·stage·분할별 게임·시점 수, 디싱크 폐기율, `result_source` 분포를 쓴다(040-R6, 040-AC2, 040-AC4).
- `build <dir> --src <tag> --records <jsonl> [--jobs N --limit N --part i --parts N --ladder-records <collect.jsonl>]`: 필터 → 013-R7 왕복 검증 → 재생 → 레이블 → shard. 기본 출력 `<URANIUM_SHARED_DIR>/datasets/replays/shards/<src>`. 원본은 `datasets/replays/raw/<src>/`(040-R13). 맵 사전계산은 `URANIUM_MAPS_DIR`에 쓴다.
  - shard는 `URANIUM_SHARD_STAGING_DIR`(로컬)에 먼저 쓰고, 공용 경로에 `<이름>.tmp`로 복사해 SHA-256을 다시 읽어 맞춘 뒤 rename한다. 불일치는 3회 재시도, 그래도 실패하면 build가 오류로 멈추고 로컬 사본이 남는다(040-R17).
  - 승패(`result`)는 브리지 승자 → 첫 LEAVE 이탈자 패배 → 기록 승자(`--ladder-records`, `--src scr_ladder`이면 `datasets/replays/ladder/collect.jsonl` 기본) → 판정 불가(0) 순서다(040-R8).
  - 재개는 records 폴더에서 이름이 `<src>`로 시작하는 `.jsonl`을 모두 읽어 건너뛴다. 표본·시험 records는 하위 폴더에 둔다.
  - `config/map_exclude.toml`의 맵(디싱크 10% 초과, 040-R16)은 재생 전에 제외된다. 대상 맵은 넣을 수 없다.
  - build 전에 브리지와 `uranium-replay`를 같은 커밋으로 빌드한다(011-R10).

## BC(041)
`uranium-train`:
- `bc (--dataset <rel> | --shards <dir>)... (--out <ckpt> | --run <id> [--take-over]) [--backend vulkan|cuda|flex] [--core lstm|gtrxl] [--s2 | --stage s2] [--resume <ckpt> | --init <ckpt|run>] [--steps N]`
  - `--dataset replays/shards/<src>`는 manifest를 검증하고, `--shards`는 검증하지 않는다. 하위 폴더까지 읽는다(041-R9).
  - `--run`은 061 run(`checkpoints/<id>/`): 잠금·heartbeat·push, 시작 시 `LATEST`에서 재개(데이터 위치 포함, 061-R10). 체크포인트는 `checkpoint_every` update 또는 `checkpoint_minutes`마다, Ctrl+C는 저장 후 종료.
  - 다른 PC의 잠금이 있으면(유효·만료 모두) 종료 코드 75로 거부된다. `--take-over`는 상대 PC가 멈춘 것을 확인한 뒤에만 쓴다: 잠금 상태와 무관하게 즉시 인수하고 경고를 출력하며, 상대가 살아 있으면 그쪽은 다음 heartbeat에서 push 없이 멈춘다(061-R5).
  - 기본 S1(전 소스 train 분할), `--s2`·`--stage s2`는 대상 맵 train 분할(041-R12·R13). S2는 `--init`으로 가중치만 가져온다(041-R4). `--init`에 run 폴더를 주면 그 run에 `DONE`이 있어야 하고 `DONE`이 가리키는 마지막 체크포인트를 쓴다(061-R13). 학습 길이는 `config/bc.toml` `[steps]`(041-R14). `URANIUM_BC_PROFILE=1`이면 구간별 시간 출력.
  - `--dataset` pull·검증은 파일마다 `[pull]`/`[verify]` 진행 줄을 stderr에 쓴다(감시 30분 멈춤 판정 대비).
- `bc-epoch`(1 epoch update 수), `bc-eval --checkpoint <ckpt|run> [--split test|val] [--games N]`(041 지표), `bc-profile --backend <b> [--batches N]`, `split-stats`, `label-check [--games N]`(034-AC3), `manifest <dir> [--verify]`(061-R7). 모두 `--dataset`/`--shards`를 받는다.
- 하이퍼파라미터 `config/bc.toml`(GPU 메모리는 `micro_lanes`, `pad_entities`, `grad_checkpointing`), 코어 비교 `config/bc_core_ablation.toml`, 보상 `config/reward.toml`.

## 감시·작업자(060-R6, 061-R13~R15)
- `uranium-train supervise [--hang-minutes 30 --max-restarts 5 --window-minutes 60 --delay-seconds 60 --progress-file <file> --log <file>] -- <bc|rl 인자>`: 비정상 종료·멈춤이면 같은 run을 재시작한다. 종료 코드 0·130·75(잠금을 얻지 못함: 다른 PC 잠금, 획득 경쟁에서 짐, 양보 예약)·76(양보)은 재시작하지 않는다.
  - Linux: 자식 학습은 별도 프로세스 그룹으로 뜨고 `PR_SET_PDEATHSIG`(SIGINT)가 걸린다. 터미널 포그라운드 Ctrl+C도 감시 프로세스가 한 번만 전달하므로 저장 후 끝난다. 감시 프로세스가 강제 종료돼도 자식은 SIGINT를 받아 저장·push·잠금 해제 후 끝난다(060-R6). Windows: 같은 콘솔이라 Ctrl+C가 자식에게 직접 간다(기존 동작).
- `uranium-train worker --queue <shared>/queue/<이름>.toml --backend <cuda|vulkan> --priority <n> [--poll-seconds 60] [--log <file>]`
  - 큐: `[defaults] datasets=[...] config="..."`, `[[run]] id, config, core, stage(s1|s2|ablation-a), init_run, datasets, steps`.
  - 끝나지 않은(`checkpoints/<id>/DONE` 없음) 잡을 수 있는 run을 순서대로 돌린다. `init_run`이 있는 run은 `init_run`에 `DONE`이 생긴 뒤에만 잡는다(061-R13). 빈 run이 없으면 우선순위가 낮은 PC의 작업자가 돌리는 run에 양보 요청(`locks/<id>.yield`)을 쓰고, 다른 run을 잡거나 작업자가 끝나면 자기 요청을 지운다(061-R14). 직접 실행한 `bc --run`에는 양보 요청을 쓰지 않는다.
  - 양보 요청 만료: 작업자가 같은 요청 내용을 자기 단조 시계로 30분 동안 관측하면 만료로 보고 지운다(파일 mtime 무관). 작업자가 하나도 없을 때 요청한 host가 사라졌으면 `locks/<id>.yield`를 직접 지운다.
  - 다른 PC 잠금의 자동 인수: 작업자가 자기 단조 시계로 같은 heartbeat 내용을 10분 동안 연속 관측했을 때만 한다(PC 시계·파일 mtime 무관, 061-R5). 학습하는 동안에는 관측하지 않으므로, 돌아온 뒤 다시 10분을 관측한다.
  - 공용 드라이브를 읽거나 쓰지 못하면 상태를 "모름"으로 보고 종료하지 않으며 `--poll-seconds`마다 다시 본다. 모름은 run 단위다: 잠금·`DONE`·양보 요청을 읽지 못한 run만 그 주기에 건너뛰고, 공용 드라이브 자체(큐, `checkpoints/`)에 접근할 수 없으면 아무 run도 잡지 않는다. 공용 드라이브가 없는 동안 자식이 실패한 것은 재시작 횟수에 넣지 않는다(061-R13).
  - 상태 파일 `workers/<host>.json`은 1분마다(자식 학습 중에도) 갱신한다. 장비 구간은 `runs/<id>/devices.jsonl`, 집계는 `uranium-train devices --run <id>`(host별 update 수·비율, 061-R15).
  - Windows CUDA 학습 PC(우선순위 10): 쉴 때 직접 실행, Ctrl+C로 멈춘다(저장·push·잠금 해제, 약 30초). 자동 시작 구성은 없다(060-R6).
  - Linux Vulkan 호스트(Vulkan f32, 우선순위 1): 상시 실행. `tools/worker/uranium-worker.service`(systemd user, `KillSignal=SIGINT`, `TimeoutStopSec=900`) 또는 `setsid nohup ... &`, 멈출 때 `kill -INT <pid>`. learner로 정하지 않은 호스트에서는 띄우지 않는다.

## RL(042)
- `uranium-train rl --run <id> --bc <BC ckpt> --logs <dir> [--config config/rl.toml] [--algo vtrace|ppo] [--backend cuda|vulkan|flex] [--z-table <jsonl>] [--ac1-race terran|protoss|zerg] [--max-agent-steps N] [--max-updates N] [--max-minutes N] [--take-over]`
  - 체크포인트·잠금은 061, 게임 PC는 `URANIUM_HOSTS_FILE`. 로그 `train.csv`·`games.csv`(042-R2).
  - 042-AC1·033-AC2는 `--max-agent-steps 10000000`(run 누적, 도달하면 정상 종료, 042-R12).
  - 리플레이: `[replays] sample_every`(기본 100)의 배수 게임과 승자 불일치 게임만 `runs/<run>/replays/`에 남긴다(042-R11).
- z 테이블 `uranium-train z-table`(033-R8).

## 래더 리플레이 수집(043, Windows, SC:R 로그인 상태)
- `uranium-ladder probe [--leaderboard <id> | --toon <t> --gateway <g> [--gamemode <m> --season <s>] | --match-id <link> | --download <url>]`(`reports/m4_ladder_probe.md`), `shape <json> [--pointer /a/0]`.
- 하루 1회 `uranium-ladder collect --plan config/ladder/kr_asia.toml [--limit N] [--dry-run]` → `raw/scr_ladder/`, 집계 `uranium-ladder status`. 0.5 req/s·일 1,000개(043-R4). 랭크 A 이상 = `min_bucket = 6`.

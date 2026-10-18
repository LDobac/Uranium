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
- `inspect <rep>`, `convert <rep> <out>`(SC:R → 1.16.1, 맵 CHK 정규화 포함, 013), `check-screp <dir> --screp <exe>`(040-AC1), `ophist <dir>`, `play <rep>`(재생·디싱크 신호), `shard-stats <dir>`, `mapgen`.
- `build <dir> --src <tag> --records <jsonl> [--jobs N --limit N --part i --parts N]`: 필터 → 재생 → 레이블 → shard. 기본 출력 `<URANIUM_SHARED_DIR>/datasets/replays/shards/<src>`. 원본은 `datasets/replays/raw/<src>/`(040-R13). 맵 사전계산은 `URANIUM_MAPS_DIR`에 쓴다.
  - 재개는 records 폴더에서 이름이 `<src>`로 시작하는 `.jsonl`을 모두 읽어 건너뛴다. 표본·시험 records는 하위 폴더에 둔다.
  - `config/map_exclude.toml`의 맵(디싱크 10% 초과, 040-R16)은 재생 전에 제외된다. 대상 맵은 넣을 수 없다.
  - build 전에 브리지와 `uranium-replay`를 같은 커밋으로 빌드한다(011-R10).

## BC(041)
`uranium-train`:
- `bc (--dataset <rel> | --shards <dir>)... (--out <ckpt> | --run <id> [--take-over]) [--backend vulkan|cuda|flex] [--core lstm|gtrxl] [--s2 | --stage s2] [--resume <ckpt> | --init <ckpt|run>] [--steps N]`
  - `--dataset replays/shards/<src>`는 manifest를 검증하고, `--shards`는 검증하지 않는다. 하위 폴더까지 읽는다(041-R9).
  - `--run`은 061 run(`checkpoints/<id>/`): 잠금·heartbeat·push, 시작 시 `LATEST`에서 재개(데이터 위치 포함, 061-R10). 체크포인트는 `checkpoint_every` update 또는 `checkpoint_minutes`마다, Ctrl+C는 저장 후 종료.
  - 기본 S1(전 소스 train 분할), `--s2`·`--stage s2`는 대상 맵 train 분할(041-R12·R13). S2는 `--init`으로 가중치만 가져온다(041-R4). 학습 길이는 `config/bc.toml` `[steps]`(041-R14). `URANIUM_BC_PROFILE=1`이면 구간별 시간 출력.
- `bc-epoch`(1 epoch update 수), `bc-eval --checkpoint <ckpt|run> [--split test|val] [--games N]`(041 지표), `bc-profile --backend <b> [--batches N]`, `split-stats`, `label-check [--games N]`(034-AC3), `manifest <dir> [--verify]`(061-R7). 모두 `--dataset`/`--shards`를 받는다.
- 하이퍼파라미터 `config/bc.toml`(GPU 메모리는 `micro_lanes`, `pad_entities`, `grad_checkpointing`), 코어 비교 `config/bc_core_ablation.toml`, 보상 `config/reward.toml`.

## 감시·작업자(060-R6, 061-R13~R15)
- `uranium-train supervise [--hang-minutes 30 --max-restarts 5 --window-minutes 60 --delay-seconds 60 --progress-file <file> --log <file>] -- <bc|rl 인자>`: 비정상 종료·멈춤이면 같은 run을 재시작한다. 종료 코드 0·130·75(다른 PC가 잡음)·76(양보)은 재시작하지 않는다.
- `uranium-train worker --queue <shared>/queue/<이름>.toml --backend <cuda|vulkan> --priority <n> [--log <file>]`
  - 큐: `[defaults] datasets=[...] config="..."`, `[[run]] id, config, core, stage(s1|s2|ablation-a), init_run, datasets, steps`.
  - 끝나지 않은(`checkpoints/<id>/DONE` 없음) 잡을 수 있는 run을 순서대로 돌린다. 빈 run이 없으면 우선순위가 낮은 PC에 양보 요청(`locks/<id>.yield`)을 쓴다. 장비 구간은 `runs/<id>/devices.jsonl`.
  - 학습 PC(CUDA, 우선순위 10): 쉴 때 실행, Ctrl+C로 멈춘다(저장·push·잠금 해제, 약 30초).
  - BC250 1호기(Vulkan f32, 우선순위 1): 상시 실행. `tools/worker/uranium-worker.service`(systemd user, `KillSignal=SIGINT`) 또는 `setsid nohup ... &`, 멈출 때 `kill -INT <pid>`.
  - BC250 2호기는 learner로 쓰지 않는다.

## RL(042)
- `uranium-train rl --run <id> --bc <BC ckpt> --logs <dir> [--config config/rl.toml] [--algo vtrace|ppo] [--backend cuda|vulkan|flex] [--z-table <jsonl>] [--ac1-race terran|protoss|zerg] [--max-agent-steps N] [--max-updates N] [--max-minutes N] [--take-over]`
  - 체크포인트·잠금은 061, 게임 PC는 `URANIUM_HOSTS_FILE`. 로그 `train.csv`·`games.csv`(042-R2).
  - 042-AC1·033-AC2는 `--max-agent-steps 10000000`(run 누적, 도달하면 정상 종료, 042-R12).
  - 리플레이: `[replays] sample_every`(기본 100)의 배수 게임과 승자 불일치 게임만 `runs/<run>/replays/`에 남긴다(042-R11).
- z 테이블 `uranium-train z-table`(033-R8).

## 래더 리플레이 수집(043, Windows, SC:R 로그인 상태)
- `uranium-ladder probe [--leaderboard <id> | --toon <t> --gateway <g> [--gamemode <m> --season <s>] | --match-id <link> | --download <url>]`(`reports/m4_ladder_probe.md`), `shape <json> [--pointer /a/0]`.
- 하루 1회 `uranium-ladder collect --plan config/ladder/kr_asia.toml [--limit N] [--dry-run]` → `raw/scr_ladder/`, 집계 `uranium-ladder status`. 0.5 req/s·일 1,000개(043-R4). 랭크 A 이상 = `min_bucket = 6`.

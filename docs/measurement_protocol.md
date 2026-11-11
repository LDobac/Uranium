# 경로 측정 프로토콜 (050-R9, 050-AC5)

Checkpoint 1 경로(BC → 교사 혼합 → DAgger → RL)의 지점마다, 다음 모델 경로를 **같은 조건·같은 지표**로 재는 방법이다. 조건은 Checkpoint 1을 잴 때 쓴 값으로 고정했다(`reports/checkpoint1.md` 7절, `reports/checkpoint1_rl.md`).

## 파일

| 파일 | 내용 |
|---|---|
| `config/protocol/lineage.toml` | 경로 지점(`s1e1` … `rl-s3`), 지점별 지표 메뉴, 지표별 고정 조건, plan 템플릿의 조건 해시, Checkpoint 1 결과 위치(`ck1_outputs`), Checkpoint 1 보충 측정 칸(`ck1_supplement`), T1 플랫폼(`t1_platforms`) |
| `config/plans/protocol/*.toml` | Checkpoint 1 측정 plan 사본 13개. 바뀌는 것은 이름(`@NAME@`)과 체크포인트(`@CHECKPOINT@`)뿐 |
| `config/protocol/t1_bots.toml` | 허용하는 OSS 봇 바이너리 sha256(NDEBUG 빌드) |
| `tools/eval/protocol.py` | job 생성·조건 검사·기록, Checkpoint 1 job 재집계, teacher-forced 명령, 봇 해시 확인, 비교 |
| `tools/eval/tables.py` | 평가 job 결과 표(러시·운영형·T1·승률) |
| `tools/eval/per_action.py`, `per_action_table.py` | 교사 보관분 행동별 채점과 표 |
| `tools/eval/diag_table.py` | 진단(`cue_diag`·`dead_channels`) 출력 폴더의 표 |
| `tools/eval/spatial_logs.py` | 학습 로그의 죽은 채널 기록(041-R25) 요약 |
| `tools/eval/s1e2_loop.py` | `s1e2` 체크포인트마다 val 500·T0 100판(041-R20 ③) |
| `tools/eval/hosts.toml` | 게임 호스트 목록(git 미추적, 예시 `hosts.example.toml`) |

- 명령 스트림은 `uranium-replay cmddump`로 뽑는다(`cargo build --release -p uranium-replay`, 다른 위치면 `.env`의 `URANIUM_REPLAY_BIN`).
- DAgger 판 생성 plan은 이 프로토콜 밖이다. 기존대로 `tools/eval/rush_check_plan.py --dagger`로 만든다(`docs/commands.md`).

## 지점과 지표

- 지점: `s1e1`, `s1e2`, `s2e2`(BC 기준선), `mix-e1` … `mix-e10`, `r1`(Checkpoint 1), `r2`, `rl-s1`(Checkpoint 1 RL) … `rl-s3`. `python tools/eval/protocol.py list`로 본다.
- `metrics`는 Checkpoint 1 경로에서 잰 지표다. 이 지표는 바로 비교할 수 있다. `optional`은 Checkpoint 1 값이 없는 지표다. 그중 050-R9 지점 표가 요구하는 칸은 `ck1_supplement`로 Checkpoint 1 쪽도 보충 측정할 수 있다.
- 지표

| 지표 | 내용 | 고정 조건 |
|---|---|---|
| `val500` | 대상 맵 val 500게임 teacher forcing | seed 1, ROCm f32, `config/bc.toml` |
| `val500_race` | `val500`을 상대 종족별로(050-R10 ②) | `val500`과 같은 실행. bc-eval 출력의 `by_enemy_race`(종족별 step·top-1·NLL)를 쓴다. 이 필드가 없는 옛 출력은 `check-bc-eval`이 거부한다 |
| `test` | 대상 맵 test 전체(1,199게임) | CUDA, BC 기준선 |
| `per_action` | 교사 보관분(2026-11-06 교사 판 val 165게임) 행동별 top-1 | CUDA, `--holdout orig` |
| `rush` | 050-R6 러시 128판 | seed 목록 45,001…, stochastic, `zero_last_failed` false, 60분 |
| `macro` | 운영형 40판 | seed_base 46000, stochastic, false, 10분 |
| `tvz` | T0 저그 32판 | seed 42069…42100, BC 계열 true / RL false |
| `t0_100`·`t0_300`·`t0_600` | T0 3종족 | seed_base 42000 / 41300 / 41000, true(BC 계열만) |
| `t0_tp` | T0 표준 테란·프로토스 각 32판(050-R10 ①) | T0 300의 맵 블록마다 첫 16 seed: T 41301–41316·41351–41366, P 41401–41416·41451–41466. stochastic, 60분, true |
| `t1` | T1 봇 6종 × 50판 | seed_base 46000, BC 계열 true / RL false, NDEBUG 봇, 플랫폼 고정 |
| `ac1` | 042-AC1 200판 | seed_base 42000(argmax 42200), BC 기준선은 a·b·c 세 plan |
| `cue_diag` | 단서 진단(050-R9): 적 생산 건물 HP·착공의 지점별 ridge R²(h_t까지), 적 최대 관측 생산 건물 수, 반사실 개입 Δ | 교사 보관분 433판(저그, `diag dump --cf`, `probes --races Z --boot 500`, `intervene`)과 사람 TvT·TvP(val·test 각 500게임, `probes --races T,P --cap 60 --boot 300`), ROCm f32·레인 그룹 4(아래 4) |
| `dead_channels` | 죽은 채널 통계(032-R25·041-R25) | 그 run의 학습 로그 `sp_*` 기록, update 600부터 체크포인트까지. Checkpoint 1 경로는 재지 않음(학습 기록 없음) |

- `zero_last_failed`는 plan마다 Checkpoint 1 값 그대로다. 러시·운영형은 BC·RL 모두 false다. TvZ·T1·T0·AC1(a)는 BC 계열이 true다. 그래서 TvZ·T1·AC1은 계열(`bc`/`rl`)마다 plan이 따로 있다.
- T1은 지점의 Checkpoint 1 상대와 같은 플랫폼에서만 돈다(사용자 결정 2026-11-09). 봇 바이너리가 플랫폼마다 다르기 때문이다.
  - BC 기준선(`s2e2`): Linux ROCm 호스트
  - Checkpoint 1(`r1`): Windows 호스트
  - Checkpoint 1 RL(`rl-s1`): 둘 다(실제로 섞여 돌았다)
  - `t1_platforms`가 없는 지점은 T1을 받지 않는다.

## 재는 법

1. **게임 지표**(새 경로)
   - job 만들기: `python tools/eval/protocol.py job --lineage m2 --point s2e2 --checkpoint checkpoints/m2-s2e2/ckpt-N --metrics rush,macro,tvz --job m2-s2e2-<날짜>`
   - `zero_last_failed`가 다른 단계는 한 job에 둘 수 없다. 그래서 값마다 job을 나눠 `<job>-rush-macro`, `<job>-tvz`처럼 만든다.
   - 각 job에 `protocol.json`이 남는다. 들어가는 것:
     - 체크포인트 sha256
     - git 커밋. 측정 파일이 커밋 안 됐으면 그 경로와 diff 해시
     - plan 내용 해시와 조건 해시
     - 호스트별 플랫폼과 봇 해시
   - 돌리기: 평가 작업자(`docs/commands.md` "평가 작업").
   - 표: `python tools/eval/tables.py <shared>/eval/<job> --json <job>.json`. 단계마다 다음을 함께 적는다.
     - 추론 backend별 판 수와 정밀도(CUDA bf16 혼합, ROCm f32)
     - `protocol.json`의 체크포인트 sha256, 미커밋 표시, 호스트 플랫폼
   - 확인: `python tools/eval/protocol.py verify <shared>/eval/<job> --lineage m2 --point s2e2`. 다음 중 하나라도 어긋나면 실패한다.
     - `protocol.json`이 있는지
     - plan 파일이 기록된 해시와 같은지
     - 미커밋 측정 파일로 만든 job인지
     - plan 조건, 행의 `zero_last_failed`, 행의 체크포인트 id(sha256 앞 12자리), T1 플랫폼
2. **teacher-forced 지표**(새 경로)
   - `python tools/eval/protocol.py bc-eval --metric val500 --lineage m2 --point s2e2 --checkpoint <ckpt>`가 고정 명령을 출력한다. 그 backend 장비에서 돌린다.
   - 출력은 `protocol.py check-bc-eval <json> --metric val500`로 확인한다(split·게임 수·selection seed).
   - 행동별: `python tools/eval/per_action.py --checkpoint <ckpt> --out <json>` → `python tools/eval/per_action_table.py BC=<json> m2-e1=<json> ...`
3. **Checkpoint 1 경로**(`--lineage ck1`): 이미 잰 칸은 다시 재지 않는다(050-R9).
   - 이미 있는 칸: `ck1_outputs`의 기존 출력을 쓴다.
     - 게임 칸은 `python tools/eval/protocol.py tables --lineage ck1 --point r1 [--json <파일>]`로 다시 집계해 두 held·bunker 정의를 낸다. 게임은 새로 돌리지 않는다.
     - `ck1_outputs` 값은 job 폴더, job 폴더 목록, `{ job, stage }`(job의 한 단계), `{ job, as_plan }`(큰 job 안에서 seed·맵·상대로 찾은 프로토콜 plan의 판) 중 하나다.
     - 예: BC 기준선의 `t0_tp`는 T0 300 job(`bc-e2-final`) 안의 64판(T 12/32, P 5/32), e10의 `tvz`는 T0 100 job의 69~100번 판(18/32), Checkpoint 1의 `t0_tp`는 `jepa-diag-tp-20261109`의 `t0-tp` 단계(T 9/32, P 2/32)다.
     - teacher-forced 값은 기존 출력이다. 행동별 출력 사본은 `<shared>/artifacts/per_action/ck1/`에 있다.
   - 새로 재는 것: `cue_diag`(모든 지점, 아래 4. `dead_channels`는 학습 기록이 없어 재지 않음)와, 050-R9 지점 표에 있지만 Checkpoint 1이 재지 않은 칸(`ck1_supplement`)이다. 이 칸은 새 경로와 같은 `job`·`bc-eval` 명령에 `--lineage ck1`로 잰다("비교가 필요하면 Checkpoint 1 쪽을 같은 plan으로 보충 측정한다").
     - `s2e2`: `t0_600`(041-AC2 T0 600은 계보 밖 `bc-s2-pt`만 쟀다), `val500_race`
     - `mix-e10`(채택 epoch): `t0_tp`
     - `r1`(Checkpoint 1): `test`, `val500_race`, `per_action`
     - `r2`: `per_action`, `t0_tp`
4. **진단**(`cue_diag`·`dead_channels`, 모든 지점. 사용자 결정 2026-11-10, AMB-050-4·5·6)
   - `cue_diag`는 보관분 두 개로 잰다. 출력은 `<out>/<teacher|human>/`이다.
     - `teacher`(교사 보관분): 2026-11-06 교사 판(`teacher-20261106-a`~`d`)과 `teacher-20261107-macro2`의 val + test, 433판, 모두 TvZ다. Checkpoint 1 진단(`reports/jepa_survey.md` 10.1), U1 대조, λ 선택 run과 같은 데이터다. 저그 HP·착공 단서와 반사실 개입 Δ를 잰다.
       - 공용 드라이브 `datasets/replays/shards/botgen`에서 그 파트만 연결한 폴더를 실행 호스트의 `<cache>/diag_holdout/<hold_orig|hold_macro2>/botgen`에 만든다(`<cache>` = `URANIUM_CACHE_DIR`, 없으면 `runs`). `per_action.py`의 `orig`와 같은 방식이다.
       - 원본과 연결 폴더의 shard 파일 수·지문(파일 이름·크기)이 `lineage.toml`과 다르면 거부한다.
       - 이 보관분의 저그 풀 행에서는 적 최대 관측 수가 늘 1이라 `diag probes`가 건너뛴다. 확인은 그 건너뜀이 났는지만 본다.
     - `human`(사람 TvT·TvP): 적 최대 관측 생산 건물 수(`count|focus_rows`, 배럭·게이트웨이)를 잰다(AMB-050-4). `reports/jepa_survey.md` 10.2.3 방식이다.
       - 데이터: 사람 shard 4개(val500과 같은 `--dataset`)의 val·test에서 각 500게임(032-R17 ② 고정 seed 1). 데이터셋 manifest 지문(2,069 파일)이 다르면 거부한다.
       - 판 수는 사용자 결정이다(사용자 확인 2026-11-10). `diag dump`에 상대 종족 필터가 없어 모든 상대의 1,000게임을 읽는다. val500 종족 비율로 TvT 약 150, TvP 약 500 관점이다(조사의 전체 패스는 497·1,142, 두 체크포인트 약 4.2시간).
       - 설정: 조사의 테란·프로토스 설정(`--act-until 8640 --act-every 4`, `probes --cap 60 --boot 300`), 반사실 pass 없음, ROCm f32, 레인 그룹 4. `probes --races T,P`이고 `count|focus_rows`에 값이 있어야 한다.
       - `--read-until 8640`: 판을 6:00까지만 읽는다. 쓰는 행·활성값은 모두 6:00 전이라 같고, 사람 판(15~20분)의 나머지 step을 읽지 않는다. 레인이 판을 더 빨리 바꾸므로 parts 안 행 순서만 다르다. 교사 보관분에는 쓰지 않는다(λ run과 같은 행 순서를 지켜야 probe 값이 같다).
       - 데이터 지문(판·행·활성값 행 수)은 첫 측정(2026-11-11 r1) 값으로 고정했다: 판 1,000·행 763,946·활성값 행 85,872. 그 밖에 `check-diag`는 옆의 끝난 출력 폴더(`<shared>/artifacts/diag/*/<set>`)와 판·행·활성값 행 수, 과제별 행·판 수를 맞춰 보고 다르면 거부한다(데이터만으로 정해지는 값이라 체크포인트와 무관).
   - `dead_channels`는 그 run의 학습 로그 `sp_*` 기록(041-R25·042-R21)만 쓴다(AMB-050-5). 기록은 update 600부터 그 체크포인트의 update(BC: `ckpt-N` = update N, RL: agent `main`의 로그 전체)까지다.
     - Checkpoint 1 경로 칸은 "재지 않음(학습 기록 없음)"이다. `--lineage ck1`의 `dead_channels`는 거부한다.
     - `diag spatial`은 프로토콜에서 뺐다. 필요하면 따로 쓰는 도구다.
   - `s1e2` 지점의 단서 진단(`cue_diag`)은 run의 최종 체크포인트(`DONE`이 있고 가장 늦은 `ckpt-N`)만 받는다(AMB-050-6). `dead_channels`는 이 제한이 없다.
   - 명령: `python tools/eval/protocol.py diag --lineage m2 --point s1e1 --checkpoint checkpoints/m2-s1/ckpt-N --train-log <BC stderr 로그 | RL train.csv>`가 이 호스트 경로로 명령을 출력만 한다(아무것도 쓰지 않음). Linux ROCm 호스트에서 `--run`을 붙이면 차례로 실행한다. Checkpoint 1 경로는 `--metrics cue_diag`.
     - `--run`은 먼저 `<out>/running.json`을 쓰고, 단계마다 걸린 시간·오류를 남긴다. 덤프는 이 호스트 `<cache>/diag_dump/<run>__<ckpt>/<set>/dump`에 쓰고, 끝나면 `<out>/<set>/dump.part`로 복사한 뒤 이름을 바꾼다(λ 스크립트와 같음).
     - 다 끝나면 `running.json`이 `protocol.json`이 된다. 체크포인트 sha256, git 커밋과 미커밋 측정 파일, backend·정밀도, 보관분, 학습 로그 이름·sha256, 실행한 명령(`python`, `<shared>`·`<cache>` 표기), 단계 기록, 확인 결과가 들어간다.
     - `protocol.json`이 있는 폴더는 거부한다. `running.json`에 실패한 단계가 없으면(다른 호스트가 재는 중일 수 있음) 거부하고, 그 호스트가 없어졌으면 `--takeover`로 다시 잰다. 실패한 시도의 폴더는 `<out>.failed-<k>`로, 캐시의 덤프는 `<run>__<ckpt>.failed-<k>`로 이름을 바꿔 남기고 다시 잰다.
     - 두 호스트로 나누기(`--part`): `--run --part gpu`(Linux ROCm 호스트)는 덤프와 그 복사만 하고 `running.json`을 남긴다. 이어 CPU 호스트에서 같은 git 커밋으로 `--run --part cpu`(`dead_channels`면 `--train-log`도)를 하면 덤프를 그 호스트 `<cache>`로 가져와(`fetch`) probes·intervene·summary·학습 로그 단계를 돌리고 `protocol.json`을 쓴다. gpu 단계가 안 끝났거나, 커밋·체크포인트·지표가 다르면 거부한다. 단계마다 호스트를 남긴다. 기본값 `--part all`은 한 호스트에서 전부 돈다.
       - probes(`--device cpu`)는 호스트마다 값이 조금 다르다. 같은 덤프(r1 교사)에서 Linux CPU 호스트와 Linux ROCm 호스트의 CPU를 비교하면 ridge는 2e-8까지, MLP(64)는 0.005까지 차이 났다(CPU 행렬 곱 라이브러리 차이). λ run probes는 Linux CPU 호스트에서 돌았으므로 그 호스트에서 돌린다. Linux ROCm 호스트의 CPU에서는 약 10배 느리다(교사 probes 7,033초 대 707초).
     - 이 호스트 캐시에 지점마다 덤프가 남는다. 교사 덤프는 약 0.4 GB(λ run, `--cf` 없이), 반사실 pass가 있는 교사 덤프도 약 0.4 GB, 사람 1,000게임 덤프는 약 1.0 GB다(r1). `--part cpu` 호스트 캐시에도 같은 크기가 남는다. 공용 드라이브에도 같은 크기가 복사된다.
     - 출력하는 명령(경로는 실행 호스트 것, `<out>` = `<shared>/artifacts/diag/<run>__<ckpt>`):
       ```
       python -m uranium diag dump --checkpoint <ckpt> --shards <cache>/diag_holdout/hold_orig/botgen --shards <cache>/diag_holdout/hold_macro2/botgen --splits val,test --config config/bc.toml --backend rocm --lanes-per-group 4 --row-until 8640 --act-until 5760 --act-every 1 --core-points gate --cf --cf-starts 'T:1280,2080;P:1720,3030;Z:1800,3240' --out <cache>/diag_dump/<run>__<ckpt>/teacher/dump
       python -m uranium diag probes <cache>/diag_dump/<run>__<ckpt>/teacher/dump --races Z --stages token,e_pool,sp_pool,c,x_in,L1.g1,L1.out,h,hc --gain resid --boot 500 --cap 0 --seed 7 --device cpu --out <out>/teacher/probes.json
       python -m uranium diag intervene <cache>/diag_dump/<run>__<ckpt>/teacher/dump --races Z --window 2160,5760 --boot 2000 --seed 3 --out <out>/teacher/intervene.json
       cp -r <cache>/diag_dump/<run>__<ckpt>/teacher/dump <out>/teacher/dump.part && mv <out>/teacher/dump.part <out>/teacher/dump
       python -m uranium diag summary --point <지점> --probes <out>/teacher/probes.json --intervene <out>/teacher/intervene.json --out <out>/teacher/summary.json
       python -m uranium diag dump --checkpoint <ckpt> --dataset replays/shards/stardata --dataset replays/shards/community --dataset replays/shards/scr_ladder --dataset replays/shards/personal --splits val,test --games 500 --seed 1 --config config/bc.toml --backend rocm --lanes-per-group 4 --row-until 8640 --act-until 8640 --act-every 4 --core-points gate --read-until 8640 --out <cache>/diag_dump/<run>__<ckpt>/human/dump
       python -m uranium diag probes <cache>/diag_dump/<run>__<ckpt>/human/dump --races T,P --stages token,e_pool,sp_pool,c,x_in,L1.g1,L1.out,h,hc --gain resid --boot 300 --cap 60 --seed 7 --device cpu --out <out>/human/probes.json
       cp -r <cache>/diag_dump/<run>__<ckpt>/human/dump <out>/human/dump.part && mv <out>/human/dump.part <out>/human/dump
       python -m uranium diag summary --point <지점> --probes <out>/human/probes.json --out <out>/human/summary.json
       python tools/eval/spatial_logs.py <학습 로그> --checkpoint <ckpt> --from-update 600 [--until-update N] --out <out>/spatial_log.json
       ```
     - 시간(2026-11-11 r1 실측): `--part gpu`(Linux ROCm 호스트) 64분 = 교사 덤프 2,054초(lane 그룹마다 forward 7번, 그룹·pass당 약 0.26초로 GPU가 정한다) + 사람 1,000게임 덤프 1,772초 + 복사·데이터 확인. `--part cpu`(Linux CPU 호스트) 23분 = 교사 probes 712초 + 사람 probes 634초 + 가져오기·intervene·summary. 다음 지점의 gpu 단계와 겹쳐 돌리면 지점마다 ROCm 호스트 약 64분이다.
       - 덤프의 모델 pass는 trunk와 행동 종류 헤드만 돈다(`UraniumNet.action_seq`, `forward_seq`의 `dists["action"]`과 같은 값). 위치 헤드는 위치를 쓰는 행 수만큼 역합성곱을 돌리는데, ROCm 백엔드는 그 행 수가 처음 나올 때마다 커널을 새로 찾았다(한 번에 약 70초, 사람 판에서 잦음). 2026-11-11 첫 사람 덤프가 느렸던 원인이고, 4시간 멈춤도 이 탐색 중으로 본다(재현은 안 됨).
       - ROCm forward는 같은 입력에서도 실행마다 활성값이 fp16 1 ULP, p(행동)가 6e-6까지 다르다. 덤프를 다시 하면 probe 값도 그만큼 흔들린다.
   - 확인: `python tools/eval/protocol.py check-diag <out> [--metrics ...] [--checkpoint <ckpt>]`. 다음 중 하나라도 다르면 실패한다.
     - `protocol.json`: 끝난 기록, 그 지표, git 커밋, 체크포인트 sha256(050-R9 기록 규칙)
     - 덤프(보관분마다): 데이터 폴더, 분할, 판 수·seed(`human`), 설정 파일, backend·정밀도, 레인 그룹, `row_until`·`act_until`·`act_every`·`read_until`(`human`만 8640), 반사실 pass(교사만)와 착공 시각, 고정된 데이터 지문(교사: 판 433·행 176,873·활성값 행 36,516, 사람: 판 1,000·행 763,946·활성값 행 85,872), 보고 지점(`token`, `e_pool`, `x_in`, `L1.out`, `h`)의 활성값
     - probes: 같은 체크포인트, gain 방식·bootstrap·cap·MLP, 종족, 과제별 행·판 수(교사: 미완성 풀 17,097행·154판, 토큰 있는 행 36,516행·351판), 값이 있어야 하는 과제(`human`의 `count|focus_rows`), 건너뛰어야 하는 과제(교사의 `count|focus_rows`)
     - intervene(교사): 같은 체크포인트, 창, 종족
     - `spatial_log.json`: 시작 update 600, 기록이 있음, 경보 층 11개
   - 표: `python tools/eval/diag_table.py m2-s1=<out> ck1-s1=<out> ... [--metrics ...] [--json <파일>] [--md <파일>]`. 폴더마다 먼저 `check-diag`를 하고, 다르면 거부한다. Checkpoint 1 폴더는 `--metrics cue_diag`로 따로 내거나 `--no-check`로 함께 낸다(죽은 채널 칸은 "재지 않음").
     - 단서(보관분·종족·과제마다): 지점별 ridge R²(분류는 AUC)와 판 bootstrap 95% 구간, h_t MLP(64), 방해 변수만의 R², h_t의 방해 변수 대비 향상. 교사 과제는 `hp|unf_token`, `start|unf_token`, `start|all_token_rows`, 사람 과제는 `count|focus_rows`, `hp|unf_token`, `start|all_token_rows`, `focus2_by_300|all_token_rows`
     - 반사실 개입(교사): 효과(`start`, `start_unf`, `count_sc`, `count_tok`) × 지표(`bunker_any`, `marine_sum`, `rax2_any`, `label_p`)의 평균 차이와 구간
     - 죽은 채널: 기록 수, 경보 줄·기록·규칙을 넘은 기록, 가장 많이 죽은 층, 최대 `const` S8·`s_pool`, 최소 양수 비율
   - `spatial_logs.py`: BC는 stderr 로그의 JSON 줄과 경보 줄을 읽는다. RL은 `train.csv` 열을 읽는데, RL 경보 줄은 RL 프로세스 stderr에 있어 세지 않고 규칙을 넘은 기록만 센다. 채널 수는 체크포인트의 모델 설정(`meta.json`)에서 읽는다. λ run 스크립트에서 옮겼고 `m2-lam-base` 로그에서 같은 값이 나온다.
   - λ run 출력(`<shared>/eval/m2-lambda-20261110/`)은 반사실 pass 없이 덤프했고 `protocol.json`도 없어 `check-diag`를 통과하지 않는다. 실제 pass의 활성값은 `--cf`와 무관하므로 교사 probe 수치는 비교할 수 있다.
5. **`s1e2` 체크포인트마다**(041-R20 ③, 050-R9 "`m2-s1e2` 체크포인트마다"): `python tools/eval/s1e2_loop.py`
   - Checkpoint 1 방식(`reports/m6_bc_eval.md` 4절) 그대로다. 채점기가 빌 때마다 아직 안 잡힌 가장 새 체크포인트를 잡고, 그 사이는 건너뛰고, 최종(run의 `DONE`)은 반드시 잰다. 기준인 S1 최종(`s1e1` 지점, `--reference`)을 먼저 잰다. 학습은 멈추지 않는다.
   - val 500(Linux ROCm 호스트): `python tools/eval/s1e2_loop.py val --run m2-s1e2 --reference checkpoints/m2-s1/ckpt-N --watch`
     - 체크포인트마다 `protocol.py`의 고정 명령을 돌린다: `python -m uranium bc-eval --dataset replays/shards/stardata --dataset replays/shards/community --dataset replays/shards/scr_ladder --dataset replays/shards/personal --checkpoint <shared>/checkpoints/<run>/ckpt-N --split val --games 500 --seed 1 --backend rocm --config config/bc.toml --bunker-before 5760 --out <shared>/artifacts/bc_e2_val/val500/<run>__ckpt-N.json`
     - 2시간 제한(Checkpoint 1 규칙)을 둔다. 호스트가 여럿이면 `<...>.claim` 파일로 나눠 잡는다.
     - 결과 옆 `<...>.protocol.json`에 체크포인트 sha256, git, backend·정밀도, 종료 코드, 시도 번호, `check-bc-eval` 결과를 남긴다. 확인이 어긋나도 기록하고 다음으로 간다.
     - 실패·시간 초과·중단(Ctrl-C, 기록 전 오류)은 그 시도의 파일을 `<...>.failed-<k>.*`로 바꿔 남기고(결과 → 로그 → 기록 → claim 순), 그 체크포인트가 아직 잡을 대상이면 다시 돈다. 체크포인트마다 모두 3번까지다(메인 임시 결정: λ 채점 때 한 ROCm 호스트에서 val500이 3번 죽은 뒤 성공).
     - 기록 없는 claim은 그 채점기가 사라진 것으로 보고(claim에 호스트·pid·시간 제한을 적는다) "stale claim" 기록을 쓴 뒤 같은 방식으로 남기고 다시 돈다(시도 1번으로 센다): 시간 제한 + 15분보다 오래됐거나, 이 호스트의 claim인데 그 프로세스가 없을 때. 기다리는 동안 어느 체크포인트가 어느 호스트에서 도는지 출력한다.
     - 이미 있는 결과는 덮어쓰지 않는다(041-R29 ⑨). 손으로 만든 결과도 잡힌 것으로 본다.
     - 끝날 때 기준·최종에 종료 코드 0 기록이 없으면 그 목록과 함께 0이 아닌 코드로 끝난다. 기록 없이 손으로 만든 결과는 NOTE로 알린다.
     - `--dry-run`은 모든 체크포인트의 명령을 출력만 한다.
   - T0 100판: `python tools/eval/s1e2_loop.py games --run m2-s1e2 --reference ... --hosts tools/eval/hosts.toml --watch --worker "--priority 2 --backend rocm --local-hostd <주소> --hostd <uranium-hostd>"`
     - 체크포인트마다 `protocol.py job --lineage m2 --point s1e2 --metrics t0_100`으로 `<shared>/eval/<run>-games/<run>__ckpt-N/` job을 만든다(`--games-root`로 바꿈). 다른 호스트가 먼저 만든 job은 잡힌 것으로 본다.
     - 그 job을 이 호스트의 `eval-worker`로 돌리고 `DONE`을 기다린 뒤 다음 체크포인트로 간다. `--worker`가 없으면 job만 만들고, 작업자는 따로 띄운다. `--worker`가 있으면 다른 호스트가 만든 끝나지 않은 job에도 작업자로 붙는다.
   - 진행: `s1e2_loop.py list --run m2-s1e2 --reference ...`
   - 표: `s1e2_loop.py table --run m2-s1e2 --reference ... [--json] [--md]`
     - val: top-1, 헤드별 NLL(action·delay·selected·target_unit·target_loc), win_open MSE, step 수
     - T0 100: 종족별 승/패/무, 합계, Wilson 95%
     - 조건이 고정값과 다른 val 출력과 `verify`에 걸린 job은 표에 MISMATCH로 표시한다.
   - 진단은 최종 체크포인트에서 `protocol.py diag --point s1e2`로 한 번 잰다(AMB-050-6). 이 반복은 진단을 돌리지 않는다.

## 거부하는 경우

- Checkpoint 1 경로에서 이미 잰 칸의 재측정(그 지점의 `ck1_supplement`와 진단 지표만 받는다)
- 지점 메뉴에 없는 지표, 다른 계열의 plan(예: BC 지점에 `tvz_rl`)
- 템플릿 조건 해시나 고정값(판 수·표집·`zero_last_failed`·`max_frames`)이 `lineage.toml`과 다를 때. 이름·체크포인트(p0·p1의 모델 모두)가 자리표시가 아닐 때
- `m2` 경로인데 run 이름이 `m2-`로 시작하지 않을 때
- 체크포인트 가중치 파일이 없을 때
- T1 job의 호스트 중 `platform`이 없거나 지점의 `t1_platforms`가 아닌 것이 있을 때
- job에 OSS 봇이 있는데, 호스트의 봇 해시를 읽을 수 없거나 `t1_bots.toml`에 없을 때. 호스트 목록에 `local = true`를 주거나, `ssh = "<별칭>"`과 `oss_bots_dir`(그 호스트의 `URANIUM_OSS_BOTS_DIR`)을 주거나, `--bot-hashes <파일>`(`<주소> <봇> <sha256>` 줄)을 준다.
- 측정 파일(`tools/eval`, `config/plans/protocol`, `config/protocol`)에 커밋 안 된 변경이 있을 때. `--allow-dirty`면 경로와 diff 해시를 `protocol.json`에 기록하고 진행한다. `verify`는 그런 job을 표시한다.
- git 작업 사본이 아닐 때
- 진단: `bc-eval`·`job`에 진단 지표를 줄 때(`diag`만), 보관분 shard·데이터셋 manifest가 고정 지문과 다를 때, 출력 폴더에 끝난 `protocol.json`이 이미 있을 때(측정은 폴더마다 한 번, 다른 폴더는 `--out`), Checkpoint 1 경로의 `dead_channels`, `--train-log` 없는 `dead_channels`, `s1e2`의 최종이 아닌 체크포인트, `check-diag`가 다른 설정·데이터·체크포인트나 끝나지 않은 기록을 찾을 때
- `tables.py`에서 조건이 어떤 프로토콜 plan과도 맞지 않는 단계. `--kind <단계 이름>=<rush|macro|t1|winrate>`를 주면 집계한다.

## 지표 정의

- 막음은 두 정의를 항상 같이 낸다.
  - `held_8m`: 8:00 SCV ≥ 8(`scv_8m`). 8:00 전에 끝난 판은 실패다. DAgger 시험과 최종 게이트가 썼다(`test_table.py`).
  - `held_s`: 1:00 이후 SCV 최솟값 > 4(`scv_min_1m`, 050-R7 세션 막음). 교사 클론 이전 게이트가 썼다.
  - 예외: 옛 `rush_summary.py`는 8:00 전에 이긴 판을 `scv_8m ≥ 8` 통과로 셌다. 이 값은 JSON의 `held_8m_winpass`로만 낸다. Checkpoint 1 job 중 둘이 다르다: e10 이른 15 → 18, r1 늦은 37 → 42, r2 늦은 37 → 44.
- 벙커도 두 정의를 같이 낸다.
  - `bunker_order_<t>`: 명령 스트림의 첫 Build_Bunker 지시 프레임 < t
  - `bunker_unit_<t>`: 세션 행 `bunker_s` < t(벙커 유닛이 생긴 시각)
  - 러시는 3:00·4:00, 운영형은 4:00이다. 경계는 미만이다(180초·240초는 들지 않는다).
- 정찰: 3:00 전, SCV 1기를 고른 뒤 첫 건물 위치에서 맨해튼 거리 1,200픽셀 넘게 보낸 이동 명령. 8:00까지의 명령만 본다.
- 운영형 경제: CC 지시 ≤ 5:00, `scv_8m` ≥ 16
  - 8:00 전에 끝난 판은 `scv_8m` ≥ 16 실패로 센다. Checkpoint 1 모델 표(`test_table.py`)와 같다.
  - Checkpoint 1 임시 결정 "섬멸 승으로 8:00 전에 끝난 판은 통과"는 교사 세션 표(`summary.md`의 T0 Terran sessions)에만 적용됐다. 이 규칙의 값은 JSON의 `scv_8m_16_winpass`로만 낸다(r1 31 → 37, r2 34 → 37, e10 26 → 32, Checkpoint 1 RL 37 → 37).
- 무효 판: 우리 launcher는 0으로, 상대 launcher는 0이 아닌 값으로 끝난 Aborted 시도가 있는 판(상대 크래시)이다. 그 판의 나중 기록(재시도나 Surrender 승)도 세지 않는다. 우리 쪽도 비정상으로 끝난 시도는 보통 재시도로 본다.
- 판 번호: `시도 × 10,000,000 + 단계 × 100,000 + n`에서 시도를 떼고 센다. 한 판의 모델 행이 여럿이면 시도 번호가 가장 큰 행을 쓰고, 같으면 나중 행을 쓴다.
- `scv_4m` p50은 4:00 전에 끝난 판을 0으로 센다(`test_table.py`). `g3.py` 방식(그런 판 제외)은 JSON의 `scv_4m_p50_known`이다.

## 비교

- 같은 지점의 두 경로를 `python tools/eval/protocol.py compare <A.json> <B.json>`로 비교한다. 지표마다 개수와 Fisher 정확 검정 p를 낸다.
- 표에는 Wilson 95% 구간을 같이 적는다(050-R2). 표집 잡음은 재현되지 않는다. 64판 조건 하나의 잡음은 ±12%p 정도다.
- 의도한 차이(혼합 방식, 벙커 완화, 기억 필드 등)는 비교표에 따로 적는다.
- `per_action`은 CUDA bf16이라 다시 돌리면 칸이 조금 달라질 수 있다. e10 재채점에서는 같은 165게임·94,433 step에 Attack_Unit top-1이 81.2 → 80.8%였고, 벙커·전체 top-1은 같았다.

## 옛 스크립트와의 관계

`runs/`(git 미추적)의 Checkpoint 1 스크립트를 옮겼다. 같은 job이면 같은 수가 나온다(2026-11-09 재집계). 예외는 아래와 같다.

| 옛 스크립트 | 새 위치 |
|---|---|
| `runs/dagger/test_table.py`, `runs/tclone_macro/macro_table.py`, `runs/bc_ft_teacher/rush_summary.py`, `runs/mix/g3.py` | `tables.py`(rush, macro) |
| `runs/dagger/t1_table.py`, `ac1_table.py` | `tables.py`(t1, winrate) |
| `runs/dagger/make_test_job.py`, `runs/mix/make_jobs.py`, `runs/tclone/make_jobs.py` | `protocol.py job` |
| `runs/mix/per_action.py`, `runs/mix/table.py` | `per_action.py`, `per_action_table.py` |
| `rl-cmddump` | `uranium-replay cmddump` |
| λ run `scripts/spatial_logs.py` | `spatial_logs.py`(같은 값) |
| λ run `scripts/summary.py`의 probe 표·공간 표 | `diag_table.py`(같은 칸, `m2-lam-base` probe 값 같음) |
| λ run 호스트 스크립트(`worker.sh` 등의 dump·probes 줄) | `protocol.py diag`(교사 보관분 설정 같고 `--cf` 추가, 사람 보관분 추가) |
| 호스트 `~/e2val/val_watch.sh`, `~/e2games/games_watch.sh`(Checkpoint 1 `s1e2`) | `s1e2_loop.py val`, `games` |

- 예외 1: e10 운영형 `scv_8m ≥ 16`은 `tables.py`로 26이다. Checkpoint 1 리포트의 25는 옛 `macro_table.py`가 40번 판에서 모델 행 대신 모델 정보가 없는 마지막 행을 읽은 값이다(리포트에 정정 주석).
- 예외 2: 재시도 판.
  - 옛 `test_table.py`·`ac1_table.py`는 `game_id // 100000`으로 단계를 골라 재시도 판의 행을 빠뜨렸다. `t1_table.py`(`% 100000`)·`rush_summary.py`·평가 작업자 `summary.md`는 재시도 판을 셌다.
  - 재시도 모델 행이 있는 Checkpoint 1 job은 둘이다: `bc-check-20261105`(6판), `ac1-20261101/baseline/ac1-base-lf0`(1판).
  - 두 job의 리포트 값(`rush_summary.py`, `summary.md`)은 이미 재시도 판을 포함한다. `tables.py` 값도 같다.
- 예외 3: `rush_summary.py`의 `scv_8m` 승리 통과는 위 "지표 정의"대로 `held_8m_winpass`로만 낸다.

## 교사 보관분 러시·운영형 나누기 (041-R27)

- `per_action.py`는 `--holdout orig|full`이면 판마다 상대 종류(러시·운영형)를 붙여 따로도 센다. 전체 값(`per_action`)은 그대로다. `--classes none`이면 나누지 않는다.
- 판 분류: 판을 만든 plan의 `games.jsonl` `opponent`로 정한다.
  - 러시 4상대(T0 `zerg_rush`, `ualbertabot12poollings`, `ualbertabot9pool10lings`, `ualbertabot9poolhydra`): plan `teacher_botgen_rush`
  - 운영형 5상대(`ualbertabot2hatchhydra`, `ualbertabot3hatchmuta`, `steamhammer2hatchmuta`, T0 `zerg_muta`, T0 표준 저그): plan `teacher_botgen_macro`·`macro2`·`macro3`(040-R23)
  - 리플레이 파일 SHA-256이 shard의 `replay_sha256`이라 이것으로 판을 잇는다. 원본은 `<shared>/datasets/replays/raw/botgen/teacher-*/teacher_botgen_*/`, 캐시는 `<URANIUM_CACHE_DIR>/per_action_holdout/teacher_classes.json`이다.
  - 모르는 상대 이름이나 분류가 없는 판이 나오면 멈춘다(새 plan의 상대는 `per_action.py`에 더한다).
- 보관분(교사 판 val, 2026-11-10): `orig` 165판 = 러시 129 + 운영형 36. macro2 37판, macro3 44판은 모두 운영형이다. `full`이면 러시 129 + 운영형 117이다.
- 출력에 더해지는 것
  - `by_class.rush`·`by_class.macro`: 판 수, step, 행동별 행, `bunker_p_before_4m`
  - `r27.rush_bunker_top1`: 혼합 연장 규칙(041-R21 ⑥)의 벙커 top-1. 경로 2부터 러시 상대 판에서만 센다.
  - `r27.macro_bunker_p_before_4m`: 운영형 판에서 4:00 전(관측 프레임 < 5,760) 채점 step의 p(Build_Bunker) 평균. 기록만 한다.
    - 이 값은 `full` 보관분의 운영형 117판으로 잰다(메인 임시 결정 2026-11-10, `orig`만이면 36판). `--holdout orig`이면 나머지 botgen 파트(macro2·macro3)를 이 값만을 위해 한 번 더 채점한다. `r27.macro_holdout`에 범위를 적는다.
    - 그 밖의 값(`per_action`, `by_class`, 러시 벙커 top-1)은 채점한 보관분(프로토콜 값 `orig`) 그대로다.
- `per_action_table.py`는 나눈 출력이면 러시 벙커 top-1과 열 사이 변화, 운영형 p(벙커)를 행으로 더한다. 나누지 않은 열은 `-`다.
- Checkpoint 1 경로의 `bc-mix` 벙커 top-1은 러시·운영형을 합친 보관분 값이다. 경로 2와 비교할 때 정의 차이를 표에 적는다.
- `uranium._core`의 `Batch.lane_games`(판의 `replay_sha256`)가 필요하다. 이 변경 이후로 빌드한 확장이어야 한다.

## 회귀 게이트 (050-R10)

- 명령: `python tools/eval/regression_gate.py --baseline-val <m2-s2e2 val500 json> --candidate-val <지점 val500 json> --baseline-games <m2-s2e2 t0_tp job> --candidate-games <지점 t0_tp job> [--json <파일>]`
- 입력
  - val500 출력: `protocol.py bc-eval --metric val500_race`(또는 `val500`, 같은 출력 파일)가 내는 명령으로 돌린다. 이 명령에 `--bunker-before 5760`이 들어 있다. 이 옵션은 `bunker_before` 필드만 더하고 나머지 출력은 같다. `check-bc-eval`은 이 필드가 없으면 경고한다(3a가 MISSING).
  - 게임: 프로토콜 plan `t0_tp` 단계가 있는 평가 job. 명령 스트림은 `tables.py`처럼 `uranium-replay cmddump`로 뽑는다.
- 검사(문턱은 AMB-050-3 결정, 경계는 통과)

| 검사 | 내용 | 통과 |
|---|---|---|
| 1 | T0 표준 테란·프로토스 각 32판 승률(승 / 유효 판, 무승부는 승 아님, 상대 크래시 판 무효) | 기준선 승률 이상 |
| 2 | val500 상대 종족별 action top-1, 행동 종류 NLL(`nll.action`, 사용자 확인 2026-11-10) | top-1 ≥ 기준선 − 0.5%p, NLL ≤ 기준선 + 0.01 |
| 3a | 사람 TvT·TvP val 판의 4:00 전 p(Build_Bunker) step 평균(teacher forcing, step 평균은 메인 임시 결정 2026-11-10) | 기준선 이하 |
| 3b | T0 테란·프로토스 판의 `bunker_order_4m`·`bunker_unit_4m` 비율 | 기준선 이하 |

- 판정은 테란·프로토스 상대만 한다. 하나라도 FAIL이면 그 지점 결과와 함께 사용자에게 보고한다. 학습은 멈추지 않는다. 입력이 없는 검사는 MISSING이다. T0 job에 결과도 무효 표시도 없는 판이 남아 있으면(끝나지 않은 job) 1·3b는 MISSING이다.
- 게임 칸에는 Wilson 95% 구간과 Fisher p를 함께 낸다. 32판은 조건당 약 ±12%p 잡음이 있어 작은 차이는 사용자가 판단한다.

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
| `val500_race` | `val500`을 상대 종족별로(050-R10 ②) | `val500`과 같은 실행. bc-eval 출력에 `by_enemy_race`(종족별 step·top-1·NLL)가 있어야 한다. 지금 bc-eval은 이 필드를 쓰지 않으므로 `python/uranium/bc/eval.py`에 추가가 필요하다. 없으면 `check-bc-eval`이 거부한다 |
| `test` | 대상 맵 test 전체(1,199게임) | CUDA, BC 기준선 |
| `per_action` | 교사 보관분(2026-11-06 교사 판 val 165게임) 행동별 top-1 | CUDA, `--holdout orig` |
| `rush` | 050-R6 러시 128판 | seed 목록 45,001…, stochastic, `zero_last_failed` false, 60분 |
| `macro` | 운영형 40판 | seed_base 46000, stochastic, false, 10분 |
| `tvz` | T0 저그 32판 | seed 42069…42100, BC 계열 true / RL false |
| `t0_100`·`t0_300`·`t0_600` | T0 3종족 | seed_base 42000 / 41300 / 41000, true(BC 계열만) |
| `t0_tp` | T0 표준 테란·프로토스 각 32판(050-R10 ①) | T0 300의 맵 블록마다 첫 16 seed: T 41301–41316·41351–41366, P 41401–41416·41451–41466. stochastic, 60분, true |
| `t1` | T1 봇 6종 × 50판 | seed_base 46000, BC 계열 true / RL false, NDEBUG 봇, 플랫폼 고정 |
| `ac1` | 042-AC1 200판 | seed_base 42000(argmax 42200), BC 기준선은 a·b·c 세 plan |
| `cue_diag`, `dead_channels` | 단서 진단, 죽은 채널 통계 | 다른 작업 단위가 채운다(지금은 자리만) |

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
   - 새로 재는 것: `cue_diag`·`dead_channels`(모든 지점, 다른 작업 단위의 진단 도구)와, 050-R9 지점 표에 있지만 Checkpoint 1이 재지 않은 칸(`ck1_supplement`)이다. 이 칸은 새 경로와 같은 `job`·`bc-eval` 명령에 `--lineage ck1`로 잰다("비교가 필요하면 Checkpoint 1 쪽을 같은 plan으로 보충 측정한다").
     - `s2e2`: `t0_600`(041-AC2 T0 600은 계보 밖 `bc-s2-pt`만 쟀다), `val500_race`
     - `mix-e10`(채택 epoch): `t0_tp`
     - `r1`(Checkpoint 1): `test`, `val500_race`, `per_action`
     - `r2`: `per_action`, `t0_tp`

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

- 예외 1: e10 운영형 `scv_8m ≥ 16`은 `tables.py`로 26이다. Checkpoint 1 리포트의 25는 옛 `macro_table.py`가 40번 판에서 모델 행 대신 모델 정보가 없는 마지막 행을 읽은 값이다(리포트에 정정 주석).
- 예외 2: 재시도 판.
  - 옛 `test_table.py`·`ac1_table.py`는 `game_id // 100000`으로 단계를 골라 재시도 판의 행을 빠뜨렸다. `t1_table.py`(`% 100000`)·`rush_summary.py`·평가 작업자 `summary.md`는 재시도 판을 셌다.
  - 재시도 모델 행이 있는 Checkpoint 1 job은 둘이다: `bc-check-20261105`(6판), `ac1-20261101/baseline/ac1-base-lf0`(1판).
  - 두 job의 리포트 값(`rush_summary.py`, `summary.md`)은 이미 재시도 판을 포함한다. `tables.py` 값도 같다.
- 예외 3: `rush_summary.py`의 `scv_8m` 승리 통과는 위 "지표 정의"대로 `held_8m_winpass`로만 낸다.

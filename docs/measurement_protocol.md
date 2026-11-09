# 경로 측정 프로토콜

Checkpoint 1 경로(BC → 교사 혼합 → DAgger → RL)의 지점마다, 다음 모델 경로를 **같은 조건·같은 지표**로 재는 방법이다. 조건은 Checkpoint 1을 잴 때 쓴 값으로 고정했다(`reports/checkpoint1.md` 7절, `reports/checkpoint1_rl.md`).

## 파일

| 파일 | 내용 |
|---|---|
| `config/protocol/lineage.toml` | 경로 지점(`s1e1` … `rl-s3`), 지점별 지표 메뉴, 지표별 고정 조건, plan 템플릿의 조건 해시, Checkpoint 1 결과 위치(`ck1_outputs`) |
| `config/plans/protocol/*.toml` | Checkpoint 1 측정 plan 사본. 바뀌는 것은 이름(`@NAME@`)과 체크포인트(`@CHECKPOINT@`)뿐 |
| `config/protocol/t1_bots.toml` | 허용하는 OSS 봇 바이너리 sha256(NDEBUG 빌드) |
| `tools/eval/protocol.py` | job 생성·조건 검사·기록, teacher-forced 명령, 봇 해시 확인, 비교 |
| `tools/eval/tables.py` | 평가 job 결과 표(러시·운영형·T1·승률) |
| `tools/eval/per_action.py`, `per_action_table.py` | 교사 보관분 행동별 채점과 표 |
| `tools/eval/hosts.toml` | 게임 호스트 목록(git 미추적, 예시 `hosts.example.toml`) |

명령 스트림은 `uranium-replay cmddump`로 뽑는다(`cargo build --release -p uranium-replay`, 다른 위치면 `URANIUM_REPLAY_BIN`).

## 지점과 지표

- 지점: `s1e1`, `s1e2`, `s2e2`(BC 기준선), `mix-e1` … `mix-e10`, `r1`(Checkpoint 1), `r2`, `rl-s1`(Checkpoint 1 RL) … `rl-s3`. `python tools/eval/protocol.py list`로 본다.
- `metrics`는 Checkpoint 1 경로에서 잰 지표다. 이 지표는 바로 비교할 수 있다. `optional`은 그 지점에서 재도 되지만 Checkpoint 1 값이 아직 없는 지표다.
- 지표

| 지표 | 내용 | 고정 조건 |
|---|---|---|
| `val500` | 대상 맵 val 500게임 teacher forcing | seed 1, ROCm f32, `config/bc.toml` |
| `test` | 대상 맵 test 전체(1,199게임) | CUDA, BC 기준선 |
| `per_action` | 교사 보관분(2026-11-06 교사 판 val 165게임) 행동별 top-1 | CUDA, `--holdout orig` |
| `rush` | 050-R6 러시 128판 | seed 목록 45,001…, stochastic, `zero_last_failed` false, 60분 |
| `macro` | 운영형 40판 | seed_base 46000, stochastic, false, 10분 |
| `tvz` | T0 저그 32판 | seed 42069…42100, BC 계열 true / RL false |
| `t0_100`·`t0_300`·`t0_600` | T0 3종족 | seed_base 42000 / 41300 / 41000, true(BC 계열만) |
| `t1` | T1 봇 6종 × 50판 | seed_base 46000, BC 계열 true / RL false, NDEBUG 봇 |
| `ac1` | 042-AC1 200판 | seed_base 42000(argmax 42200), BC 기준선은 a·b·c 세 plan |
| `cue_diag`, `dead_channels` | 단서 진단, 죽은 채널 통계 | 다른 작업 단위가 채운다(지금은 자리만) |

- `zero_last_failed`는 plan마다 Checkpoint 1 값 그대로다. 러시·운영형은 BC·RL 모두 false, TvZ·T1·T0·AC1(a)는 BC 계열 true다. 그래서 TvZ·T1·AC1은 계열(`bc`/`rl`)마다 plan이 따로 있다.

## 재는 법

1. **게임 지표**
   - job 만들기: `python tools/eval/protocol.py job --lineage m2 --point s2e2 --checkpoint checkpoints/m2-s2e2/ckpt-N --metrics rush,macro,tvz --job m2-s2e2-<날짜>`
   - `zero_last_failed`가 다른 단계는 한 job에 둘 수 없다. 그래서 값마다 job을 나눠 `<job>-rush-macro`, `<job>-tvz`처럼 만든다.
   - 각 job에 `protocol.json`이 남는다. 체크포인트 sha256, git 커밋(측정 파일 미커밋 여부 포함), plan 내용 해시와 조건 해시, 호스트별 봇 해시가 들어간다.
   - 돌리기: 평가 작업자(`docs/commands.md` "평가 작업").
   - 표: `python tools/eval/tables.py <shared>/eval/<job> --json <job>.json`. 단계마다 추론 backend별 판 수와 정밀도(CUDA bf16 혼합, ROCm f32)도 적는다.
   - 확인: `python tools/eval/protocol.py verify <shared>/eval/<job> --lineage m2 --point s2e2`. plan 조건, 행의 `zero_last_failed`, 행의 체크포인트 id(sha256 앞 12자리)를 대조한다.
2. **teacher-forced 지표**
   - `python tools/eval/protocol.py bc-eval --metric val500 --lineage m2 --point s2e2 --checkpoint <ckpt>`가 고정 명령을 출력한다. 그 backend 장비에서 돌린다.
   - 출력은 `protocol.py check-bc-eval <json> --metric val500`로 확인한다(split·게임 수·selection seed).
   - 행동별: `python tools/eval/per_action.py --checkpoint <ckpt> --out <json>` → `python tools/eval/per_action_table.py BC=<json> m2-e1=<json> ...`
3. **Checkpoint 1 쪽 값**: `lineage.toml`의 `ck1_outputs`에 있는 job을 같은 `tables.py`로 다시 집계한다. Checkpoint 1 지점을 새 지표로 보충 측정할 때는 `--lineage ck1`로 job을 만든다. 체크포인트가 기록된 것과 다르면 거부한다.

## 거부하는 경우

- 지점 메뉴에 없는 지표, 다른 계열의 plan(예: BC 지점에 `tvz_rl`)
- 템플릿 조건 해시나 고정값(판 수·표집·`zero_last_failed`·`max_frames`)이 `lineage.toml`과 다를 때
- `m2` 경로인데 run 이름이 `m2-`로 시작하지 않거나, `ck1` 경로인데 기록된 체크포인트가 아닐 때
- 체크포인트 가중치 파일이 없을 때
- job에 OSS 봇이 있는데, 호스트의 봇 해시를 읽을 수 없거나 `t1_bots.toml`에 없을 때. 호스트 목록에 `local = true` 또는 `ssh = "<별칭>"`을 주거나, `--bot-hashes <파일>`(`<주소> <봇> <sha256>` 줄)을 준다.
- 측정 파일(`tools/eval`, `config/plans/protocol`, `config/protocol`)에 커밋 안 된 변경이 있을 때(`--allow-dirty`면 기록만 하고 진행)

## 지표 정의

- 막음은 두 정의를 항상 같이 낸다.
  - `held_8m`: 8:00 SCV ≥ 8(`scv_8m`, 8:00 전에 끝난 판은 실패). DAgger 시험과 최종 게이트가 썼다.
  - `held_s`: 1:00 이후 SCV 최솟값 > 4(`scv_min_1m`, 050-R7 세션 막음). 교사 클론 이전 게이트가 썼다.
- 벙커도 두 정의를 같이 낸다.
  - `bunker_order_<t>`: 명령 스트림의 첫 Build_Bunker 지시 프레임 < t
  - `bunker_unit_<t>`: 세션 행 `bunker_s` < t(벙커 유닛이 생긴 시각)
  - 러시는 3:00·4:00, 운영형은 4:00이다.
- 정찰: 3:00 전, SCV 1기를 고른 뒤 첫 건물 위치에서 맨해튼 거리 1,200픽셀 넘게 보낸 이동 명령. 8:00까지의 명령만 본다.
- 운영형 경제: CC 지시 ≤ 5:00, `scv_8m` ≥ 16
- 무효 판: 상대 launcher가 0으로 끝나지 않은 Aborted 시도가 있는 판(상대 크래시)이다. 그 판의 나중 기록(재시도나 Surrender 승)도 세지 않는다.
- 판 번호: `시도 × 10,000,000 + 단계 × 100,000 + n`에서 시도를 떼고 센다(재시도 판 포함). 한 판의 기록이 여럿이면 마지막 모델 행을 쓴다.

## 비교

- 같은 지점의 두 경로를 `python tools/eval/protocol.py compare <A.json> <B.json>`로 비교한다. 지표마다 개수와 Fisher 정확 검정 p를 낸다.
- 표에는 Wilson 95% 구간을 같이 적는다(050-R2). 표집 잡음은 재현되지 않는다. 64판 조건 하나의 잡음은 ±12%p 정도다.
- 의도한 차이(혼합 방식, 벙커 완화, 기억 필드 등)는 비교표에 따로 적는다.

## 옛 스크립트와의 관계

`runs/`(git 미추적)의 Checkpoint 1 스크립트를 옮겼다. 같은 job이면 같은 수가 나온다(2026-11-09 재집계, 아래 예외 2건).

| 옛 스크립트 | 새 위치 |
|---|---|
| `runs/dagger/test_table.py`, `runs/tclone_macro/macro_table.py`, `runs/bc_ft_teacher/rush_summary.py`, `runs/mix/g3.py` | `tables.py`(rush, macro). `scv_4m` p50은 4:00 전에 끝난 판을 0으로 센다(test_table). g3 방식(그런 판 제외)은 JSON의 `scv_4m_p50_known` |
| `runs/dagger/t1_table.py`, `ac1_table.py` | `tables.py`(t1, winrate) |
| `runs/dagger/make_test_job.py`, `runs/mix/make_jobs.py`, `runs/tclone/make_jobs.py` | `protocol.py job` |
| `runs/mix/per_action.py`, `runs/mix/table.py` | `per_action.py`, `per_action_table.py` |
| `rl-cmddump` | `uranium-replay cmddump` |

- 예외 1: e10 운영형 `scv_8m ≥ 16`은 `tables.py`로 26이다. Checkpoint 1 리포트의 25는 옛 `macro_table.py`가 한 판(40번)에서 모델 행 대신 모델 정보가 없는 행을 읽은 값이다.
- 예외 2: 재시도 판. 옛 `test_table.py`·`t1_table.py`는 `game_id // 100000`으로 단계를 골라 재시도 판의 행을 빠뜨렸다. Checkpoint 1 표에 쓴 job 중에는 BC 러시 점검(`bc-check-20261105`, 6판)이 해당한다. 그 job의 리포트 값은 재시도 판을 센 `rush_summary.py` 값이어서 `tables.py` 값과 같다.

# 경로 측정 프로토콜 (050-R9, 050-AC5)

- spec: [050-R9][s050], [050-AC5][s050].
- 목적: Checkpoint 1 경로(BC → 교사 혼합 → DAgger → RL)의 지점마다 다음 모델 경로를 같은 조건·지표로 잰다.
- 조건: Checkpoint 1 측정값으로 고정([checkpoint1] 7절, [checkpoint1_rl]).

## 파일

| 파일 | 내용 |
|---|---|
| `config/protocol/lineage.toml` | 지점(`s1e1` … `rl-s3`), 지점별 지표 메뉴, 지표별 고정 조건, plan 템플릿 조건 해시, Checkpoint 1 결과 위치(`ck1_outputs`), Checkpoint 1 보충 측정 칸(`ck1_supplement`), T1 플랫폼(`t1_platforms`) |
| `config/plans/protocol/*.toml` | Checkpoint 1 측정 plan 사본 13개 + `t0_tp_rl`(`t0_tp`에서 `zero_last_failed`만 false, 수정 RL 점검용). 바뀌는 것은 이름(`@NAME@`)·체크포인트(`@CHECKPOINT@`)뿐 |
| `config/protocol/t1_bots.toml` | 허용 OSS 봇 바이너리 sha256(NDEBUG 빌드) |
| `tools/eval/protocol.py` | job 생성·조건 검사·기록, Checkpoint 1 job 재집계, teacher-forced 명령, 봇 해시 확인, 비교 |
| `tools/eval/tables.py` | 평가 job 결과 표(러시·운영형·T1·승률), 모든 단계의 [052][s052] `ops`·`rush_combat` 표, 테란·프로토스 상대 단계의 교사 모드 판 비율([050-R10][s050]) |
| `tools/eval/regression_gate.py` | 050-R10 회귀 게이트(아래 "회귀 게이트") |
| `tools/eval/distill_verdict.py` | [044-R6][s044] 증류 라운드 판정(아래 "증류 라운드 판정") |
| `tools/eval/per_action.py`, `per_action_table.py` | 교사 보관분 행동별 채점·표 |
| `tools/eval/diag_table.py` | 진단(`cue_diag`·`dead_channels`) 출력 폴더의 표 |
| `tools/eval/spatial_logs.py` | 학습 로그의 죽은 채널 기록([041-R25][s041]) 요약 |
| `tools/eval/ops_sets.py` | eval job 밖 비교 집합(교사 botgen, 사람 리플레이)의 `opsdump` 값([052-R7][s052]) |
| `tools/eval/branch_plan.py`, `branch_collect.py` | 분기 게임([051][s051]) 줄기·분기 plan, 분기 결과 모음. 프로토콜 밖([commands]) |
| `tools/eval/granularity.py` | 결정 단위 측정 표([031][s031] "결정 단위"). 프로토콜 밖([commands]) |
| `tools/eval/rush_check_plan.py` | 러시 점검·교사·botgen·DAgger·증류 판 plan 생성. 프로토콜 밖([commands]) |
| `python/uranium/measure/` | 위 `tools/eval/*.py`의 코드(같은 이름 모듈). `tools/eval/*.py`는 그 모듈의 `main`만 부르는 실행 파일이라 `uranium`을 설치한 venv python으로 실행 |
| `python/uranium/measure/lineage.py`, `eval_common.py`, `branch_common.py` | 위 도구들이 함께 쓰는 함수(`lineage.py`: `lineage.toml` 읽기·plan 조건 해시·템플릿) |
| `tools/eval/eval_worker.ps1` | Windows CUDA 호스트의 `eval-worker` 실행([commands]) |
| `tools/eval/hosts.toml` | 게임 호스트 목록(git 미추적, 예시 `hosts.example.toml`) |

- 명령 스트림: `uranium-replay cmddump`(`cargo build --release -p uranium-replay`, 다른 위치면 `.env`의 `URANIUM_REPLAY_BIN`).
- 052 운영 지표(`ops`·`rush_combat`, [052-R7·R8][s052]): 같은 바이너리 `uranium-replay opsdump`.
  - 리플레이를 우리 슬롯 시점에서 8프레임마다 멈춰 다시 센다(OpenBW 재생, `.env`의 `URANIUM_BWAPI_DIR`·`URANIUM_BRIDGE_LIB`·`URANIUM_DATA_DIR`·`URANIUM_MAPS_DIR`).
  - 결과 `<cache>/<job>/<stage>/g<n>.ops.json`, 다음 집계가 다시 씀. 경로 비교는 양쪽 모두 재생 값.
  - `tables.py --ops live` = 세션 행 라이브 값(run 감시용), `--ops off` = 표 뺌.
  - 학습 중이 아닌 CPU 호스트에서 `--ops-jobs N`으로 나눠 돌림.
- DAgger 판 생성 plan은 프로토콜 밖: `tools/eval/rush_check_plan.py --dagger`([commands]).

## 지점과 지표

- 지점: `s1e1`, `s1e2`, `s2e2`(BC 기준선), `mix-e1` … `mix-e10`, `r1`(Checkpoint 1), `r2`, `rl-s1`(Checkpoint 1 RL) … `rl-s3`. 목록 `python tools/eval/protocol.py list`.
- Checkpoint 1 대응 없는 경로 2 지점([050-R9][s050] 지점 표 2026-11-14 행, [044-R6·R7·R11][s044]): `ck1_checkpoint` 없음 → `--lineage ck1` 거부.
  - `r3`: 경로 2 DAgger 3라운드(사용자 결정 2026-11-14, [041-R29][s041] ④). 메뉴 = `r2`(Checkpoint 1 대응 모델 후보라 test·val 500 종족별·T1(Windows)도 `optional`).
  - `d1`, `d2`: 증류 라운드(run `m2-distill-d<R>`만, `run_pattern`). val 500(종족별), 교사 hold-out, T0 표준 테란·프로토스 각 32판, TvZ 32판, 러시 128판 + 운영형 40판, 교사 모드 판 비율, 진단.
  - `D`: 경로 2 수정 모델([044-R7][s044], 통과 라운드의 체크포인트). Checkpoint 1 대응 모델 행과 같은 test 1,199, val 500(종족별), T1 300판, T0 표준 테란·프로토스. 진단은 그 라운드 지점과 같은 체크포인트·출력 폴더.
  - `rl-fix`: 경로 2 수정 RL 점검([044-R11][s044], 150만 step마다). T0 표준 테란·프로토스(`t0_tp_rl`), TvZ(`tvz_rl`), 러시 128 + 운영형 40, 끝에 T1 300(`t1_rl`). 단서 진단·죽은 채널은 최종 체크포인트만(`diag_final_only = ["cue_diag", "dead_channels"]`, 사용자 결정 2026-11-14). run 이름 `m2-rl-fix-<번호>`만(사용자 결정 2026-11-14).
  - 네 지점 모두 050-R10 게이트 적용, 교사 모드 판 비율 기록.
  - `gn`: GroupNorm 이어 학습 모델([050-R9][s050] 지점 표 2026-11-17 행, 사용자 결정 2026-11-17, [gn_resume](../report/experiment/2026_11_17_10_05_gn_resume.md) 3절). run `m2-gn-t1`·`m2-gn-t1e<번호>`만. [Checkpoint 2 표](../report/release/2026_11_17_09_05_checkpoint2.md)(3절) 전체: test 1,199, val 500(종족별), T1 300판(Windows·Linux), T0 표준 테란·프로토스 각 32판, TvZ 32판, 러시 128판 + 운영형 40판, 교사 모드 판 비율, 진단. 050-R10 게이트는 적용하지 않음, 그 지표(종족별 val, T0 표준 테란·프로토스, 교사 모드 판 비율)는 측정·기록만(사용자 결정 2026-11-17).
  - `t2`(run `m2-teacher-t2`, [044-R8][s044] ⑤)·`rl-fix-3`(run `m2-rl-fix-3` 끝 체크포인트, [044-R11][s044]): `gn`과 같은 표 + `zerg_build`·`tp_style`. 050-R10 게이트 미적용, 판정은 [044-R13][s044] 합격 기준(gn 대비, 사용자 결정 2026-11-17).
- `metrics` = Checkpoint 1 경로에서 잰 지표(바로 비교 가능). `optional` = Checkpoint 1 값 없는 지표. 그중 050-R9 지점 표가 요구하는 칸은 `ck1_supplement`로 Checkpoint 1 쪽 보충 측정 가능.

| 지표 | 내용 | 고정 조건 |
|---|---|---|
| `val500` | 대상 맵 val 500게임 teacher forcing | seed 1, ROCm f32, `config/bc.toml` |
| `val500_race` | `val500` 상대 종족별([050-R10][s050] ②) | `val500`과 같은 실행. bc-eval 출력 `by_enemy_race`(종족별 step·top-1·NLL). 이 필드 없는 옛 출력은 `check-bc-eval`이 거부 |
| `test` | 대상 맵 test 전체(1,199게임) | CUDA, BC 기준선 |
| `per_action` | 교사 보관분(2026-11-06 교사 판 val 165게임) 행동별 top-1 | CUDA, `--holdout orig` |
| `rush` | [050-R6][s050] 러시 128판 | seed 목록 45,001…, stochastic, `zero_last_failed` false, 60분 |
| `macro` | 운영형 40판 | seed_base 46000, stochastic, false, 10분 |
| `tvz` | T0 저그 32판 | seed 42069…42100, BC 계열 true / RL false |
| `t0_100`·`t0_300`·`t0_600` | T0 3종족 | seed_base 42000 / 41300 / 41000, true(BC 계열만) |
| `t0_tp` | T0 표준 테란·프로토스 각 32판(050-R10 ①) | T0 300 맵 블록마다 첫 16 seed: T 41301–41316·41351–41366, P 41401–41416·41451–41466. stochastic, 60분, true |
| `t0_tp`(RL) | 같은 64판, RL 점검용 plan `t0_tp_rl` | 같은 seed·조건, `zero_last_failed` false(050-R9 RL 규칙) |
| `teacher_mode` | 교사 모드 판 비율(050-R10, 기록만). job이 아니라 `tables.py`가 T0·T1 단계의 테란·프로토스 상대 판마다 셈(아래 "지표 정의") | 명령 스트림(`uranium-replay cmddump`) |
| `zerg_build`·`tp_style` | 저그전 교사 빌드 판·022 교사 방식 판(050-R10 2026-11-17, [044-R13][s044] 판정). `teacher_mode`처럼 `tables.py`가 셈 | 명령 스트림 |
| `t1` | T1 봇 6종 × 50판 | seed_base 46000, BC 계열 true / RL false, NDEBUG 봇, 플랫폼 고정 |
| `ac1` | [042-AC1][s042] 200판 | seed_base 42000(argmax 42200), BC 기준선은 a·b·c 세 plan |
| `cue_diag` | 단서 진단(050-R9): 적 생산 건물 HP·착공의 지점별 ridge R²(h_t까지), 적 최대 관측 생산 건물 수, 반사실 개입 Δ | 교사 보관분 433판(저그, `diag dump --cf`, `probes --races Z --boot 500`, `intervene`) + 사람 TvT·TvP(val·test 각 500게임, `probes --races T,P --cap 60 --boot 300`), ROCm f32·레인 그룹 4(아래 4) |
| `dead_channels` | 죽은 채널 통계([032-R25][s032]·[041-R25][s041]) | 그 run 학습 로그 `sp_*` 기록, update 600부터 체크포인트까지. Checkpoint 1 경로는 재지 않음(학습 기록 없음) |

- `zero_last_failed`: plan마다 Checkpoint 1 값 그대로. 러시·운영형은 BC·RL 모두 false, TvZ·T1·T0·AC1(a)는 BC 계열 true → TvZ·T1·AC1은 계열(`bc`/`rl`)마다 plan 따로.
- T1은 지점의 Checkpoint 1 상대와 같은 플랫폼에서만(사용자 결정 2026-11-09, 봇 바이너리가 플랫폼마다 다름).
  - BC 기준선(`s2e2`): Linux ROCm 호스트
  - Checkpoint 1(`r1`): Windows 호스트
  - Checkpoint 1 RL(`rl-s1`): 둘 다(실제로 섞여 돌았음)
  - `r2`: Windows. 경로 2 Checkpoint 1 대응 모델은 r1·r2 중 041-R29 ④로 고르므로 r2도 그 모델 행 칸(test, val 500 종족별, T1, T0 표준 테란·프로토스)을 `optional`로 받음.
  - `D`·`rl-fix`: Checkpoint 1 대응 없음 → 빈 호스트(Windows·Linux 모두, 사용자 결정 2026-11-14). `gn`도 같게 둠(사용자 결정 2026-11-17). job `protocol.json`에 호스트 플랫폼, `tables.py`에 단계별 추론 backend·정밀도. `compare`는 두 쪽 backend나 T1 플랫폼이 다르면 `NOTE` 줄.
  - `t1_platforms` 없는 지점은 T1 안 받음.

## 재는 법

1. **게임 지표**(새 경로)
   - job: `python tools/eval/protocol.py job --lineage m2 --point s2e2 --checkpoint checkpoints/m2-s2e2/ckpt-N --metrics rush,macro,tvz --job m2-s2e2-<날짜>`
   - `zero_last_failed`가 다른 단계는 한 job 불가 → 값마다 `<job>-rush-macro`, `<job>-tvz`. 한 지표가 두 값에 걸치면(BC 지점 `ac1`만) `<job>-ac1-lf0`, `<job>-ac1-real`.
   - job마다 `protocol.json`: 체크포인트 sha256, git 커밋(측정 파일 미커밋이면 경로·diff 해시), plan 내용 해시·조건 해시, 호스트별 플랫폼·봇 해시.
   - 돌리기: 평가 작업자([commands] "평가 작업").
   - 표: `python tools/eval/tables.py <shared>/eval/<job> --json <job>.json`.
     - 단계마다 [052][s052] `ops`·`rush_combat` 표(matchup별 중앙값·평균, 러시 진입 판의 SCV 동원·손실·벙커·집중도, `held_s`·`held_8m`별).
     - 단계마다 함께 적음: 추론 backend별 판 수·정밀도(CUDA bf16 혼합, ROCm f32); `protocol.json`의 체크포인트 sha256·미커밋 표시·호스트 플랫폼; 결과·ledger 파일의 깨진 줄(공용 드라이브 쓰기 손상: NUL 줄, 잘린 줄).
     - 깨진 줄: NUL만 있는 줄은 무시, 읽을 수 없는 줄은 건너뛰고 `SKIPPED` 파일:줄(JSON `skipped_lines`). 회귀 게이트는 깨진 줄이 있는 쪽이 있으면 게임 판정(1, 3b) 모두 MISSING.
   - 확인: `python tools/eval/protocol.py verify <shared>/eval/<job> --lineage m2 --point s2e2`. 실패 조건(하나라도): `protocol.json` 없음, plan 파일이 기록 해시와 다름, 미커밋 측정 파일로 만든 job, plan 조건·행의 `zero_last_failed`·행의 체크포인트 id(sha256 앞 12자리)·T1 플랫폼 불일치.
2. **teacher-forced 지표**(새 경로)
   - `python tools/eval/protocol.py bc-eval --metric val500 --lineage m2 --point s2e2 --checkpoint <ckpt>` → 고정 명령 출력, 그 backend 장비에서 실행.
   - 확인 `protocol.py check-bc-eval <json> --metric val500`(split·게임 수·selection seed).
   - 행동별: `python tools/eval/per_action.py --checkpoint <ckpt> --out <json>` → `python tools/eval/per_action_table.py BC=<json> m2-e1=<json> ...`.
     - 기존 필드 = 현재 관측 점수(Checkpoint 1 비교·Checkpoint 1 경로 연장 규칙, 경로 2는 기록만).
     - `chain_start` 블록 = 체인 시작 관측 점수([041-R30][s041] ⑤, 경로 2 연장 규칙 입력, 사용자 결정 2026-11-12). `--chain-start none`이면 생략.
     - 경로 2 표는 `per_action_table.py --chain-start`.
3. **Checkpoint 1 경로**(`--lineage ck1`): 이미 잰 칸은 다시 재지 않음([050-R9][s050]).
   - 있는 칸 = `ck1_outputs` 기존 출력.
     - 게임 칸: `python tools/eval/protocol.py tables --lineage ck1 --point r1 [--json <파일>]`로 재집계(두 held·bunker 정의). 게임 새로 안 돌림.
     - `ck1_outputs` 값: job 폴더, job 폴더 목록, `{ job, stage }`(job의 한 단계), `{ job, as_plan }`(큰 job 안에서 seed·맵·상대로 찾은 프로토콜 plan의 판) 중 하나.
     - 예: BC 기준선 `t0_tp` = T0 300 job(`bc-e2-final`) 안 64판(T 12/32, P 5/32), e10 `tvz` = T0 100 job의 69~100번 판(18/32), Checkpoint 1 `t0_tp` = `jepa-diag-tp-20261109`의 `t0-tp` 단계(T 9/32, P 2/32).
     - teacher-forced 값 = 기존 출력. 행동별 출력 사본 `<shared>/artifacts/per_action/ck1/`.
   - 새로 재는 것: `cue_diag`(모든 지점, 아래 4. `dead_channels`는 학습 기록 없어 안 잼) + 050-R9 지점 표에 있지만 Checkpoint 1이 안 잰 칸(`ck1_supplement`). 새 경로와 같은 `job`·`bc-eval` 명령에 `--lineage ck1`("비교가 필요하면 Checkpoint 1 쪽을 같은 plan으로 보충 측정한다").
     - `s2e2`: `t0_600`([041-AC2][s041] T0 600은 계보 밖 `bc-s2-pt`만 쟀음), `val500_race`
     - `mix-e10`(채택 epoch): `t0_tp`
     - `r1`(Checkpoint 1): `test`, `val500_race`, `per_action`
     - `r2`: `per_action`, `t0_tp`
4. **진단**(`cue_diag`·`dead_channels`, 모든 지점. 사용자 결정 2026-11-10, [AMB-050-4·5·6][amb])
   - `cue_diag` 보관분 두 개, 출력 `<out>/<teacher|human>/`.
     - `teacher`(교사 보관분): 2026-11-06 교사 판(`teacher-20261106-a`~`d`) + `teacher-20261107-macro2`의 val + test, 433판, 모두 TvZ. Checkpoint 1 진단([jepa_survey] 10.1)·U1 대조·λ 선택 run과 같은 데이터. 저그 HP·착공 단서, 반사실 개입 Δ.
       - 공용 드라이브 `datasets/replays/shards/botgen`에서 그 파트만 연결한 폴더를 실행 호스트 `<cache>/diag_holdout/by_name/<hold_orig|hold_macro2>`에 만듦(`<cache>` = `URANIUM_CACHE_DIR`, 없으면 `runs`). `per_action.py`의 `orig`와 같은 방식.
       - 폴더 이름이 판 순서를 정함: shard 리더는 `<폴더 이름>/<파트>/shard_*` 순 → `hold_macro2` 먼저. U1 대조·λ run 덤프(`<data>/hold_orig`, `<data>/hold_macro2`)와 같은 순서여야 같은 가중치에서 같은 값(판 id로 probe 겹·bootstrap을 나눔). `check-diag`는 판 순서 지문(`expect_game_rows_sha256`, 판 id 순 판별 행 수의 sha256)도 봄.
       - 원본·연결 폴더의 shard 파일 수·지문(파일 이름·크기)이 `lineage.toml`과 다르면 거부.
       - 이 보관분 저그 풀 행은 적 최대 관측 수가 늘 1 → `diag probes`가 건너뜀. 확인은 건너뜀 발생 여부만.
     - `human`(사람 TvT·TvP): 적 최대 관측 생산 건물 수(`count|focus_rows`, 배럭·게이트웨이, [AMB-050-4][amb]). [jepa_survey] 10.2.3 방식.
       - 데이터: 사람 shard 4개(val500과 같은 `--dataset`)의 val·test 각 500게임([032-R17][s032] ② 고정 seed 1). 데이터셋 manifest 지문(2,069 파일) 다르면 거부.
       - 판 수 = 사용자 결정(사용자 확인 2026-11-10). `diag dump`에 상대 종족 필터가 없어 모든 상대 1,000게임을 읽음. val500 종족 비율로 TvT 약 150, TvP 약 500 관점(조사 전체 패스는 497·1,142, 두 체크포인트 약 4.2시간).
       - 설정: 조사의 테란·프로토스 설정(`--act-until 8640 --act-every 4`, `probes --cap 60 --boot 300`), 반사실 pass 없음, ROCm f32, 레인 그룹 4. `probes --races T,P`, 프로토스 `count|focus_rows`에 값이 있어야 함.
       - 테란(TvT) `count|focus_rows`(적 최대 관측 배럭 수) = "이 데이터로 측정 불가"(사용자 확인 2026-11-11, AMB-050-4 후속). 6:00 전 상대 배럭이 거의 늘 1개(Checkpoint 1 r1: probe 행 6,557개 중 6,497개가 1, 나머지 60개는 한 판). probe는 돌지만 `check-diag`가 요구하지 않고 표는 "측정 불가"(`lineage.toml` `report.not_measurable`). 지점 비교는 프로토스(게이트웨이 수)만.
       - `--read-until 8640`: 판을 6:00까지만 읽음(쓰는 행·활성값은 같음, parts 안 행 순서만 다름). 교사 보관분에는 안 씀(λ run과 같은 행 순서 유지).
       - 데이터 지문(판·행·활성값 행 수) 고정값(첫 측정 2026-11-11 r1): 판 1,000·행 763,946·활성값 행 85,872. `check-diag`는 옆 끝난 출력 폴더(`<shared>/artifacts/diag/*/<set>`)와 판·행·활성값 행 수, 과제별 행·판 수도 맞춰 보고 다르면 거부.
   - `dead_channels`: 그 run 학습 로그 `sp_*` 기록([041-R25][s041]·[042-R21][s042])만([AMB-050-5][amb]). update 600부터 그 체크포인트 update까지(BC: `ckpt-N` = update N, RL: agent `main` 로그 전체).
     - Checkpoint 1 경로 칸 = "재지 않음(학습 기록 없음)". `--lineage ck1`의 `dead_channels` 거부.
     - `diag spatial`은 프로토콜에서 뺌(필요하면 따로 쓰는 도구).
   - 최종 체크포인트 제한: `s1e2`의 `cue_diag`는 run 최종 체크포인트(`DONE`이 있고 가장 늦은 `ckpt-N`)만([AMB-050-6][amb]), `dead_channels`는 제한 없음. `rl-fix`는 `cue_diag`·`dead_channels` 모두 최종만(사용자 결정 2026-11-14, RL 체크포인트 `ckpt-N/agents/<agent>`의 `ckpt-N` 기준).
   - 명령: `python tools/eval/protocol.py diag --lineage m2 --point s1e1 --checkpoint checkpoints/m2-s1/ckpt-N --train-log <BC stderr 로그 | RL train.csv>` → 이 호스트 경로로 명령만 출력(아무것도 안 씀). Linux ROCm 호스트에서 `--run`이면 차례로 실행. Checkpoint 1 경로는 `--metrics cue_diag`.
     - `--run`: 먼저 `<out>/running.json`, 단계마다 걸린 시간·오류 기록. 덤프는 이 호스트 `<cache>/diag_dump/<run>__<ckpt>/<set>/dump` → 끝나면 `<out>/<set>/dump.part`로 복사 후 이름 바꿈(λ 스크립트와 같음).
     - 끝나면 `running.json` → `protocol.json`: 체크포인트 sha256, git 커밋·미커밋 측정 파일, backend·정밀도, 보관분, 학습 로그 이름·sha256, 실행 명령(`python`, `<shared>`·`<cache>` 표기), 단계 기록, 확인 결과.
     - 거부·재시도: `protocol.json` 있는 폴더 거부. `running.json`에 실패 단계가 없으면(다른 호스트가 재는 중일 수 있음) 거부, 그 호스트가 없어졌으면 `--takeover`. 실패 시도 폴더는 `<out>.failed-<k>`, 캐시 덤프는 `<run>__<ckpt>.failed-<k>`로 이름 바꿔 남기고 다시 잼.
     - 두 호스트로 나누기(`--part`):
       - `--run --part gpu`(Linux ROCm 호스트): 덤프·복사만, `running.json` 남김.
       - 이어 CPU 호스트에서 같은 git 커밋으로 `--run --part cpu`(`dead_channels`면 `--train-log`도): 덤프를 그 호스트 `<cache>`로 가져와(`fetch`) probes·intervene·summary·학습 로그 단계, `protocol.json` 씀.
       - gpu 단계 미완료, 커밋·체크포인트·지표 불일치면 거부. 단계마다 호스트 기록. 기본 `--part all` = 한 호스트에서 전부.
       - `--part cpu`는 Linux CPU 호스트에서(메인 임시 결정 2026-11-11). 이유: probes(`--device cpu`)는 호스트마다 조금 다름(ridge 2e-8, MLP(64) 0.005까지, CPU 행렬 곱 라이브러리 차이)이고 λ run probes가 Linux CPU 호스트였음. Linux ROCm 호스트 CPU는 약 10배 느림(교사 probes 7,033초 대 707초). `check-diag`가 단계별 호스트 출력.
     - 디스크: 지점마다 덤프가 캐시에 남음. 교사 약 0.4 GB(λ run, `--cf` 없이), 반사실 pass 있는 교사도 약 0.4 GB, 사람 1,000게임 약 1.0 GB(r1). `--part cpu` 호스트 캐시·공용 드라이브에도 같은 크기.
     - 출력 명령(경로는 실행 호스트 것, `<out>` = `<shared>/artifacts/diag/<run>__<ckpt>`):
       ```
       python -m uranium diag dump --checkpoint <ckpt> --shards <cache>/diag_holdout/by_name/hold_orig --shards <cache>/diag_holdout/by_name/hold_macro2 --splits val,test --config config/bc.toml --backend rocm --lanes-per-group 4 --row-until 8640 --act-until 5760 --act-every 1 --core-points gate --cf --cf-starts 'T:1280,2080;P:1720,3030;Z:1800,3240' --out <cache>/diag_dump/<run>__<ckpt>/teacher/dump
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
     - 시간(2026-11-11 r1 실측):
       - `--part gpu`(Linux ROCm 호스트) 64분 = 교사 덤프 2,054초(lane 그룹마다 forward 7번, 그룹·pass당 약 0.26초, GPU가 정함) + 사람 1,000게임 덤프 1,772초 + 복사·데이터 확인.
       - `--part cpu`(Linux CPU 호스트) 23분 = 교사 probes 712초 + 사람 probes 634초 + 가져오기·intervene·summary.
       - 다음 지점 gpu 단계와 겹치면 지점마다 ROCm 호스트 약 64분.
       - 덤프 모델 pass는 trunk·행동 종류 헤드만(`UraniumNet.action_seq`, `forward_seq`의 `dists["action"]`과 같은 값). ROCm은 위치 헤드 행 수가 처음 나올 때마다 커널을 새로 찾음(약 70초) → 2026-11-11 첫 사람 덤프가 느렸던 원인(4시간 멈춤도 이것으로 봄, 재현 안 됨).
       - 덤프를 다시 하면 probe 값이 조금 흔들림: ROCm forward가 실행마다 다름(활성값 fp16 1 ULP, p(행동) 6e-6까지).
   - 확인: `python tools/eval/protocol.py check-diag <out> [--metrics ...] [--checkpoint <ckpt>]`. 하나라도 다르면 실패:
     - `protocol.json`: 끝난 기록, 그 지표, git 커밋, 체크포인트 sha256(050-R9 기록 규칙)
     - 덤프(보관분마다): 데이터 폴더, 분할, 판 수·seed(`human`), 설정 파일, backend·정밀도, 레인 그룹, `row_until`·`act_until`·`act_every`·`read_until`(`human`만 8640), 반사실 pass(교사만)·착공 시각, 고정 데이터 지문(교사: 판 433·행 176,873·활성값 행 36,516, 사람: 판 1,000·행 763,946·활성값 행 85,872), 보고 지점(`token`, `e_pool`, `x_in`, `L1.out`, `h`) 활성값
     - probes: 같은 체크포인트, gain 방식·bootstrap·cap·MLP, 종족, 과제별 행·판 수(교사: 미완성 풀 17,097행·154판, 토큰 있는 행 36,516행·351판), 값이 있어야 하는 과제(`human` 프로토스 `count|focus_rows`, 테란은 측정 불가라 요구 안 함), 건너뛰어야 하는 과제(교사 `count|focus_rows`)
     - intervene(교사): 같은 체크포인트, 창, 종족
     - `spatial_log.json`: 시작 update 600, 기록 있음, 경보 층 11개
   - 표: `python tools/eval/diag_table.py m2-s1=<out> ck1-s1=<out> ... [--metrics ...] [--json <파일>] [--md <파일>]`. 폴더마다 먼저 `check-diag`, 다르면 거부. Checkpoint 1 폴더는 `--metrics cue_diag`로 따로 또는 `--no-check`로 함께(죽은 채널 칸 "재지 않음").
     - 단서(보관분·종족·과제마다): 지점별 ridge R²(분류는 AUC)·판 bootstrap 95% 구간, h_t MLP(64), 방해 변수만의 R², h_t의 방해 변수 대비 향상. 교사 과제 `hp|unf_token`, `start|unf_token`, `start|all_token_rows`, 사람 과제 `count|focus_rows`, `hp|unf_token`, `start|all_token_rows`, `focus2_by_300|all_token_rows`
     - 반사실 개입(교사): 효과(`start`, `start_unf`, `count_sc`, `count_tok`) × 지표(`bunker_any`, `marine_sum`, `rax2_any`, `label_p`)의 평균 차이·구간
     - 죽은 채널: 기록 수, 경보 줄·기록·규칙을 넘은 기록, 가장 많이 죽은 층, 최대 `const` S8·`s_pool`, 최소 양수 비율
   - `spatial_logs.py`: BC는 stderr 로그의 JSON 줄·경보 줄. RL은 `train.csv` 열(RL 경보 줄은 RL 프로세스 stderr라 안 세고 규칙을 넘은 기록만 셈). 채널 수는 체크포인트 모델 설정(`meta.json`). λ run 스크립트에서 옮김, `m2-lam-base` 로그에서 같은 값.
   - λ run 출력(`<shared>/eval/m2-lambda-20261110/`)은 반사실 pass 없는 덤프·`protocol.json` 없음 → `check-diag` 불통과. 실제 pass 활성값은 `--cf` 무관이라 교사 probe 수치는 비교 가능.
5. **`s1e2` 체크포인트마다**([041-R20][s041] ③, [050-R9][s050] "`m2-s1e2` 체크포인트마다"): 체크포인트 하나씩.
   - val 500은 `protocol.py bc-eval --metric val500 --lineage m2 --point s1e2 --checkpoint <ckpt>` 명령 → `check-bc-eval`, T0 100판은 `protocol.py job --lineage m2 --point s1e2 --metrics t0_100`.
   - 진단은 최종 체크포인트에서 `protocol.py diag --point s1e2`로 한 번([AMB-050-6][amb]).

## 거부하는 경우

- Checkpoint 1 경로에서 이미 잰 칸의 재측정(그 지점의 `ck1_supplement`와 진단 지표만 받음)
- 지점 메뉴에 없는 지표, 다른 계열의 plan(예: BC 지점에 `tvz_rl`)
- 템플릿 조건 해시나 고정값(판 수·표집·`zero_last_failed`·`max_frames`)이 `lineage.toml`과 다름. 이름·체크포인트(p0·p1의 모델 모두)가 자리표시가 아님
- `m2` 경로인데 run 이름이 `m2-`로 시작 안 함, `run_pattern` 있는 지점(`d1`·`d2`·`t1`·`D`·`gn`·`rl-fix`)에서 run 이름 불일치
- Checkpoint 1 대응 없는 지점(`r3`·`d1`·`d2`·`t1`·`D`·`gn`·`rl-fix`)의 `--lineage ck1`
- 체크포인트 가중치 파일 없음
- T1 job 호스트 중 `platform`이 없거나 지점 `t1_platforms`가 아닌 것
- job에 OSS 봇이 있는데 호스트 봇 해시를 읽을 수 없거나 `t1_bots.toml`에 없음. 해결: 호스트 목록에 `local = true`, 또는 `ssh = "<별칭>"`과 `oss_bots_dir`(그 호스트의 `URANIUM_OSS_BOTS_DIR`), 또는 `--bot-hashes <파일>`(`<주소> <봇> <sha256>` 줄)
- 측정 파일(`python/uranium/measure`, `tools/eval`, `config/plans/protocol`, `config/protocol`)에 미커밋 변경. `--allow-dirty`면 경로·diff 해시를 `protocol.json`에 기록하고 진행, `verify`가 그 job 표시
- git 작업 사본이 아님
- 진단: `bc-eval`·`job`에 진단 지표(`diag`만), 보관분 shard·데이터셋 manifest가 고정 지문과 다름, 출력 폴더에 끝난 `protocol.json`이 이미 있음(측정은 폴더마다 한 번, 다른 폴더는 `--out`), Checkpoint 1 경로 `dead_channels`, `--train-log` 없는 `dead_channels`, `s1e2`의 최종 아닌 체크포인트, `check-diag`가 다른 설정·데이터·체크포인트나 끝나지 않은 기록을 찾음
- `tables.py`에서 조건이 어떤 프로토콜 plan과도 안 맞는 단계. `--kind <단계 이름>=<rush|macro|t1|winrate>`를 주면 집계

## 지표 정의

- 막음(두 정의 항상 같이):
  - `held_8m`: 8:00 SCV ≥ 8(`scv_8m`). 8:00 전에 끝난 판은 실패. DAgger 시험·최종 게이트(`test_table.py`).
  - `held_s`: 1:00 이후 SCV 최솟값 > 4(`scv_min_1m`, [050-R7][s050] 세션 막음). 교사 클론 이전 게이트.
  - 예외: 옛 `rush_summary.py`는 8:00 전 승리 판을 `scv_8m ≥ 8` 통과로 셈 → JSON `held_8m_winpass`로만. Checkpoint 1 job 중 다른 것 둘: e10 이른 15 → 18, r1 늦은 37 → 42, r2 늦은 37 → 44.
- 벙커(두 정의 같이):
  - `bunker_order_<t>`: 명령 스트림 첫 Build_Bunker 지시 프레임 < t
  - `bunker_unit_<t>`: 세션 행 `bunker_s` < t(벙커 유닛 생긴 시각)
  - 러시 3:00·4:00, 운영형 4:00. 경계는 미만(180초·240초 불포함).
- 정찰(두 정의 같이, [050-R9][s050] 2026-11-14 개정, 사용자 결정 2026-11-14). 규칙: 3:00 전, 유닛 1기를 고른 뒤 첫 건물 위치에서 맨해튼 거리 1,200픽셀 넘게 보낸 땅 이동 명령. 8:00까지 명령만.
  - `scout_3m`: 기존 값. 우클릭(0x14)·대상 지정 명령(0x15)의 y를 바이트 1~2, 대상을 2~3에서 읽어(실제 y 2~3, 대상 4~5) 땅 우클릭을 못 세고 사실상 한 유닛 `Move`만 셈. [041-R22][s041] ④·[041-R29][s041] ④ 정찰 문턱이 이 값 기준 → 판정은 이 값.
  - `scout_3m_fixed`: 바이트를 바로 읽은 값. 땅 우클릭(대상 0 또는 0xFFFF)과 `Move`를 셈. 기록·비교용. 선택 유닛 종류는 명령 스트림에 없어 SCV 여부를 안 봄 → 건물 1기 선택 후 멀리 땅 우클릭(랠리)도 정찰로 셈.
  - 둘 다 명령 스트림 캐시(`g<n>.json`)에서 집계마다 계산(캐시 삭제 불필요). `compare`는 두 JSON 중 한쪽이라도 이 키가 없으면(옛·새 JSON 비교 포함) 건너뜀.
- 명령 스트림 모델 플레이어: 테란이 하나면 그 플레이어, TvT면 세션 행 `slot` 자리 플레이어(cmddump `players` 순서 = 세션 slot, TvP 판에서 확인). 2026-11-14 전에는 첫 테란을 읽어 `t0_tp` 테란전에서 모델이 slot 1인 판(32판 중 30판)은 상대 T0 명령을 셌음. 영향은 회귀 게이트 3b 테란전 `bunker_order_4m`뿐(러시·운영형은 저그전이라 같음).
- 교사 모드 판([050-R10][s050]·[044][s044] 용어, 원인 분석 판별 규칙): 0~6분 분당 명령 < 150이고 4:00 전 리파이너리 지시 없는 판.
  - 분당 명령 = 모델 플레이어 명령 스트림 항목(선택 명령 포함) 중 프레임 < 8,640인 수 ÷ 6. 6:00 전에 끝나도 6으로 나눔. 비교는 정수(항목 수 < 900).
  - 리파이너리 지시 = 프레임 < 5,760의 Build(0x0c) 중 유닛 종류 110.
  - 표: 상대 종족(plan p1 종족, 테란·프로토스)마다 명령 스트림 있는 유효 판 중 교사 모드 판 수, 모드별 승·무·패와 10:00 전 패(`end_frame` < 14,400). 상대 크래시 판 제외. 명령 스트림 없는 판은 `no_stream`. JSON `counts`의 `teacher_mode vs <종족>` 줄로 `compare`가 Fisher p.
  - 원인 분석 값과 같음(2026-11-14 확인): 테란전·프로토스전 교사 모드 판 경로 2 s2e2 0·0/32, mix-e7 18·19/32, r1 25·30/32, Checkpoint 1 r1 22·27/32. 모드별 승·10분 안 패도 같음.
  - 2026-11-17부터 기록 필드만(과거 값·t1 판정 재현). gn·t1은 분당 명령 조건이 늘 참([teacher_leak_fix](../report/experiment/2026_11_17_20_08_teacher_leak_fix.md)).
- 방식 판별([050-R10][s050] 2026-11-17, 사용자 결정 2026-11-17, [044-R13][s044] 판정):
  - `zerg_build`: 0~6분 분당 명령 < 150(항목 수 < 900)이고 프레임 < 7,200(5:00)의 Build(0x0c) 중 리파이너리(유닛 종류 110)도 팩토리(113)도 없는 판. 저그전 교사(021-R8) 빌드.
  - `tp_style`: 0~6분 분당 명령 < 120(항목 수 < 720)이고 `zerg_build` 아닌 판. 022 교사 방식.
  - 나머지: 사람 방식. 분당 명령 세는 법은 교사 모드 판과 같음.
  - 표: T0·T1 단계, 상대 종족(테란·프로토스)마다 세 방식 판 수와 승·무·패. 상대 크래시 판 제외, 명령 스트림 없는 판 `no_stream`.
  - 기준 값(gn T0 / T1, 테란 + 프로토스): `zerg_build` 10/64 · 26/196, `tp_style` 50/64 · 150/196.
- 운영형 경제: CC 지시 ≤ 5:00, `scv_8m` ≥ 16
  - 8:00 전에 끝난 판은 `scv_8m` ≥ 16 실패(Checkpoint 1 모델 표 `test_table.py`와 같음).
  - Checkpoint 1 임시 결정 "섬멸 승으로 8:00 전에 끝난 판은 통과"는 교사 세션 표(`summary.md`의 T0 Terran sessions)에만 적용 → JSON `scv_8m_16_winpass`로만(r1 31 → 37, r2 34 → 37, e10 26 → 32, Checkpoint 1 RL 37 → 37).
- 무효 판(상대 크래시): 우리 launcher 0, 상대 launcher 0 아닌 값으로 끝난 Aborted 시도가 있는 판. 그 판의 나중 기록(재시도, Surrender 승)도 안 셈. 우리 쪽도 비정상 종료한 시도는 보통 재시도로 봄.
- 판 번호: `시도 × 10,000,000 + 단계 × 100,000 + n`에서 시도를 떼고 셈. 한 판 모델 행이 여럿이면 시도 번호가 가장 큰 행, 같으면 나중 행.
- `scv_4m` p50: 4:00 전에 끝난 판은 0(`test_table.py`). `g3.py` 방식(그런 판 제외)은 JSON `scv_4m_p50_known`.

## 비교

- 같은 지점 두 경로: `python tools/eval/protocol.py compare <A.json> <B.json>` → 지표마다 개수·Fisher 정확 검정 p.
- 표에 Wilson 95% 구간 같이([050-R2][s050]). 표집 잡음은 재현 안 됨. 64판 조건 하나 잡음 ±12%p 정도.
- 의도한 차이(혼합 방식, 벙커 완화, 기억 필드 등)는 비교표에 따로 적음.
- `per_action`은 CUDA bf16이라 재실행 시 칸이 조금 다를 수 있음(e10 재채점: 같은 165게임·94,433 step, Attack_Unit top-1 81.2 → 80.8%, 벙커·전체 top-1 같음).

## 옛 스크립트와의 관계

- `runs/`(git 미추적) Checkpoint 1 스크립트의 새 위치. 같은 job이면 같은 수(2026-11-09 재집계), 예외는 아래.

| 옛 스크립트 | 새 위치 |
|---|---|
| `runs/dagger/test_table.py`, `runs/tclone_macro/macro_table.py`, `runs/bc_ft_teacher/rush_summary.py`, `runs/mix/g3.py` | `tables.py`(rush, macro) |
| `runs/dagger/t1_table.py`, `ac1_table.py` | `tables.py`(t1, winrate) |
| `runs/dagger/make_test_job.py`, `runs/mix/make_jobs.py`, `runs/tclone/make_jobs.py` | `protocol.py job` |
| `runs/mix/per_action.py`, `runs/mix/table.py` | `per_action.py`, `per_action_table.py` |
| `rl-cmddump` | `uranium-replay cmddump` |
| λ run `scripts/spatial_logs.py` | `spatial_logs.py`(같은 값, `m2-lam-base` 로그로 확인) |
| λ run `scripts/summary.py`의 probe 표·공간 표 | `diag_table.py`(같은 칸, `m2-lam-base` probe 값 같음) |
| λ run 호스트 스크립트(`worker.sh` 등의 dump·probes 줄) | `protocol.py diag`(교사 보관분 설정 같고 `--cf` 추가, 사람 보관분 추가) |

- 예외 1: e10 운영형 `scv_8m ≥ 16`은 `tables.py`로 26. Checkpoint 1 리포트의 25는 옛 `macro_table.py`가 40번 판에서 모델 행 대신 모델 정보 없는 마지막 행을 읽은 값(리포트에 정정 주석).
- 예외 2: 재시도 판.
  - 옛 `test_table.py`·`ac1_table.py`는 `game_id // 100000`으로 단계를 골라 재시도 판 행을 빠뜨림. `t1_table.py`(`% 100000`)·`rush_summary.py`·평가 작업자 `summary.md`는 셌음.
  - 재시도 모델 행 있는 Checkpoint 1 job 둘: `bc-check-20261105`(6판), `ac1-20261101/baseline/ac1-base-lf0`(1판). 두 job 리포트 값(`rush_summary.py`, `summary.md`)은 이미 재시도 판 포함, `tables.py` 값도 같음.
- 예외 3: `rush_summary.py`의 `scv_8m` 승리 통과는 "지표 정의"대로 `held_8m_winpass`로만.

## 교사 보관분 러시·운영형 나누기 (041-R27)

- `per_action.py`는 `--holdout orig|full`이면 판마다 상대 종류(러시·운영형)를 붙여 따로도 셈([041-R27][s041]). 전체 값(`per_action`)은 그대로. `--classes none`이면 안 나눔.
- 판 분류 = 판을 만든 plan의 `games.jsonl` `opponent`:
  - 러시 4상대(T0 `zerg_rush`, `ualbertabot12poollings`, `ualbertabot9pool10lings`, `ualbertabot9poolhydra`): plan `teacher_botgen_rush`
  - 운영형 5상대(`ualbertabot2hatchhydra`, `ualbertabot3hatchmuta`, `steamhammer2hatchmuta`, T0 `zerg_muta`, T0 표준 저그): plan `teacher_botgen_macro`·`macro2`·`macro3`([040-R23][s040])
  - 판 연결: 리플레이 파일 SHA-256 = shard `replay_sha256`. 원본 `<shared>/datasets/replays/raw/botgen/teacher-*/teacher_botgen_*/`, 캐시 `<URANIUM_CACHE_DIR>/per_action_holdout/teacher_classes.json`.
  - 모르는 상대 이름·분류 없는 판이 나오면 멈춤(새 plan 상대는 `per_action.py`에 추가).
  - TvT·TvP 교사([022][s022]) botgen 폴더 `teacher-tp-*`(`-v2` 포함)는 분류 안 하고 `orig`·`full`·운영형 추가 채점 어디에도 안 넣음(041-R27은 저그 교사 보관분, [044-R8][s044]로 더해지는 이 판들은 러시·운영형 분류 없음). `full`은 botgen manifest 등록 파트 중 이 폴더를 뺀 파트를 `orig`처럼 연결 폴더(`<URANIUM_CACHE_DIR>/per_action_holdout_full/botgen`)로 읽고, 디스크 파트 폴더·파일 크기가 manifest와 다르면 멈춤.
- 보관분(교사 판 val, 2026-11-10): `orig` 165판 = 러시 129 + 운영형 36. macro2 37판, macro3 44판은 모두 운영형. `full` = 러시 129 + 운영형 117.
- 출력에 더해지는 것:
  - `by_class.rush`·`by_class.macro`: 판 수, step, 행동별 행, `bunker_p_before_4m`
  - `r27.rush_bunker_top1`: 혼합 연장 규칙([041-R21][s041] ⑥)의 벙커 top-1. 경로 2부터 러시 상대 판에서만 셈.
  - `r27.macro_bunker_p_before_4m`: 운영형 판 4:00 전(관측 프레임 < 5,760) 채점 step의 p(Build_Bunker) 평균, 기록만.
    - `full` 보관분 운영형 117판으로 잼(메인 임시 결정 2026-11-10, `orig`만이면 36판). `--holdout orig`이면 나머지 botgen 파트(macro2·macro3)를 이 값만 위해 한 번 더 채점. `r27.macro_holdout`에 범위.
    - 그 밖의 값(`per_action`, `by_class`, 러시 벙커 top-1)은 채점한 보관분(프로토콜 값 `orig`) 그대로.
- `per_action_table.py`: 나눈 출력이면 러시 벙커 top-1과 열 사이 변화, 운영형 p(벙커)를 행으로 더함. 나누지 않은 열은 `-`.
- Checkpoint 1 경로 `bc-mix` 벙커 top-1은 러시·운영형 합친 보관분 값 → 경로 2와 비교 때 정의 차이를 표에 적음.
- `uranium._core`의 `Batch.lane_games`(판 `replay_sha256`) 필요 → 이 변경 이후 빌드한 확장.

## 회귀 게이트 (050-R10)

- 명령: `python tools/eval/regression_gate.py --baseline-val <m2-s2e2 val500 json> --candidate-val <지점 val500 json> --baseline-games <m2-s2e2 t0_tp job> --candidate-games <지점 t0_tp job> [--json <파일>]`
- 입력([050-R10][s050]):
  - val500 출력: `protocol.py bc-eval --metric val500_race`(또는 `val500`, 같은 출력 파일)가 내는 명령. `--bunker-before 5760` 포함(`bunker_before` 필드만 더하고 나머지 출력은 같음). 이 필드 없으면 `check-bc-eval` 경고(3a MISSING).
  - 게임: 프로토콜 plan `t0_tp` 단계가 있는 평가 job. 명령 스트림은 `tables.py`처럼 `uranium-replay cmddump`.
- 검사(문턱 [AMB-050-3][amb] 결정, 경계는 통과)

| 검사 | 내용 | 통과 |
|---|---|---|
| 1 | T0 표준 테란·프로토스 각 32판 승률(승 / 유효 판, 무승부는 승 아님, 상대 크래시 판 무효) | 기준선 승률 이상 |
| 2 | val500 상대 종족별 action top-1, 행동 종류 NLL(`nll.action`, 사용자 확인 2026-11-10) | top-1 ≥ 기준선 − 0.5%p, NLL ≤ 기준선 + 0.01 |
| 3a | 사람 TvT·TvP val 판의 4:00 전 p(Build_Bunker) step 평균(teacher forcing, step 평균은 메인 임시 결정 2026-11-10) | 기준선 이하 |
| 3b | T0 테란·프로토스 판의 `bunker_order_4m`·`bunker_unit_4m` 비율 | 기준선 이하 |

- 판정은 테란·프로토스 상대만. 하나라도 FAIL이면 그 지점 결과와 함께 사용자에게 보고, 학습은 멈추지 않음. 입력 없는 검사는 MISSING. T0 job에 결과도 무효 표시도 없는 판이 남으면(끝나지 않은 job) 1·3b MISSING.
- 적용 지점(2026-11-14 추가): 증류 라운드, 수정 모델 D, 수정 RL 점검. RL 점검 T0 job은 `t0_tp_rl` 단계, 기준선은 그대로 `m2-s2e2`의 `t0_tp`.
- 3b 테란전 `bunker_order_4m`: 2026-11-14 전 판정의 이 칸은 상대 T0 명령("지표 정의" 모델 플레이어 항목). 경로 2 s2e2·e7·r1·r2 모두 1/32, 고친 값 10·19·18·23/32.

## 증류 라운드 판정 (044-R6)

- 명령: `python tools/eval/distill_verdict.py --round d1 --baseline-val <m2-s2e2 val500 json> --candidate-val <라운드 val500 json> --baseline-games <m2-s2e2 t0_tp job> --candidate-games <라운드 t0_tp job> --candidate-tvz <라운드 TvZ job> --candidate-rush <라운드 러시 job> --s-games <S의 t0_tp job> --s-rush <S의 러시 job> [--verdict <shared>/eval/m2-lineage/d1_first_test.verdict] [--json <파일>]`
- S = [041-R29][s041] ④ 선택 결과(Checkpoint 1 대응 모델). 같은 job이 TvZ·T0 단계를 함께 가져도 됨(`-tvz-t0_tp`).
- 판정([AMB-044-7][amb], 경계는 통과, [044-R6][s044])

| 판정 | 조건 |
|---|---|
| `pass` | [050-R10][s050] 게이트 전부 PASS(1, 2, 3a, 3b) + TvZ 32판 ≥ 16승 + 늦은 막음(`held_8m`) ≥ 24/64 + 막음 합(이른 + 늦은 `held_8m`) ≥ S의 막음 합 − 6 |
| `d2` | d1만. 통과하지 못했지만 T0 프로토스 승 ≥ S + 4이고 TvZ 유지(≥ 16승, 늦은 막음 ≥ 24/64) |
| `stop` | TvZ 유지 실패 또는 T0 프로토스 승 < S + 4. d2가 통과하지 못하면 늘 `stop`(d2가 마지막) |

- 판정은 이 셋뿐(사용자 결정 2026-11-14). 게이트 검사 MISSING이거나 칸이 비면(판 빠짐, val 출력 없음) 그 조건 실패.
- 판정 파일 보호: 입력 job 중 하나라도 끝나지 않았거나(`DONE` 없음) 깨진 줄이 있으면 판정·파일 없이 거부. 기준선 job은 `s2e2` 지점, S의 두 job은 선택 후보 지점(`r1`·`r2`·`r3`·`s2e2`) 하나·체크포인트 하나의 `protocol.json`이어야 함.
- 막음·TvZ는 세션 행으로 셈(`tables.py`와 같은 규칙, 상대 크래시 판 무효). 승 수는 무효 판 뺀 절대 수.
- 교사 모드 판 비율(라운드와 S의 `t0_tp`)은 판정 파일에 기록만(d1·d2. t1부터 라운드 쪽은 판정 조건, 아래).
- 라운드 job `protocol.json`은 그 라운드 지점(`d1`·`d2`·`t1`)·체크포인트 하나여야 함, 아니면 거부.
- 교사 라운드 t1([044-R8][s044], 사용자 결정 2026-11-14: 증류 라운드 뒤 한 번, 044-R6 규칙):
  - `--round t1`, 판정 파일 `<shared>/eval/m2-lineage/t1_first_test.verdict`. 판정은 `pass`·`stop`뿐(t1이 마지막이라 `d2` 없음, 사용자 확인 2026-11-14).
  - 라운드 job = 지점 `t1`(run `m2-teacher-t1`). S(`--s-games`·`--s-rush`) = 041-R29 ④ 선택 S = r2(`m2-dagger-r2/ckpt-1408`)의 job.
  - t1이 통과하면 t1 체크포인트가 D(지점 `D`가 run `m2-teacher-t1`도 받음).
- t1부터 바뀐 판정(044-R6·R7·050-R10 2026-11-15 개정, 사용자 위임·Fable 자문 2026-11-15). d1·d2는 전과 같고 이미 낸 판정 파일은 다시 만들지 않음.
  - 050-R10 ② 종족별 `nll.action`은 기록만: 결과 칸 `RECORD`, 기준 + 0.03 넘으면 `WARN`, 게이트를 떨어뜨리지 않음. top-1 −0.5%p, 1(T0 승), 3a·3b(4:00 전 벙커)는 그대로. 종족별 val 기준은 그대로 후보와 같은 백엔드로 잰 `m2-s2e2`.
  - `pass`에 교사 모드 판("지표 정의", `tables.py` `teacher_mode`) ≤ 3/32를 TvT·TvP 따로 추가(라운드 `t0_tp` job, 무효 판 뺀 절대 수 ≤ 3). 칸에 값이 없으면(그 종족 판 없음, 명령 스트림 없는 판, 결과 없는 판, 깨진 줄) 실패. S의 교사 모드는 기록만.
  - 판정 뒤 [044-R7][s044] RL 출발점 D 산출(멈추지 않음): `pass`면 D = t1. 아니어도 TvZ ≥ 16/32, 늦은 막음 ≥ 24/64, 막음 합(이른 + 늦은) ≥ 60, T0 테란 + 프로토스 승 합 ≥ 20/64, 교사 모드 TvT·TvP 각 ≤ 3/32를 모두 만족하면 D = t1, 그 밖에는 D = d1(`checkpoints/m2-distill-d1/ckpt-2000`). 조건별 값·통과 여부·고른 체크포인트를 출력, `--json`의 `d_choice`, 판정 파일 `# 044-R7 ...` 줄에 남김(첫 줄은 그대로 `pass`·`stop`).
  - t1의 S job이 `r2` 지점이 아니면 거부.
  - `--verdict`를 주면 다음 경우도 판정·파일 없이 거부: 라운드 job에 `protocol.json`이 하나도 없어 D 체크포인트를 모름, 라운드 T0 테란·프로토스 유효 판에 명령 스트림이 없거나 셀 수 없음. `--verdict` 없이 돌리면 그 교사 모드 칸은 값 없음(실패)으로 보임.
- 판정 파일: 첫 줄 판정, 나머지 `# ` 줄(조건별 값). 있으면 덮어쓰지 않음([041-R29][s041] ⑨).
- 어느 판정이든 결과를 사용자에게 보고, 다음 단계는 사용자가 정함(자동 정지 없음).
- 게임 칸에 Wilson 95% 구간·Fisher p 함께. 32판은 조건당 약 ±12%p 잡음, 작은 차이는 사용자 판단.

[s022]: ../spec/2026_11_14_08_18_022-matchup-teacher.md
[s031]: ../spec/2026_10_15_22_24_031-action-space.md
[s032]: ../spec/2026_10_15_22_24_032-model-architecture.md
[s040]: ../spec/2026_10_15_22_24_040-replay-dataset.md
[s041]: ../spec/2026_10_15_22_24_041-imitation-learning.md
[s042]: ../spec/2026_10_15_22_24_042-rl-training.md
[s044]: ../spec/2026_11_14_08_18_044-self-game-distillation.md
[s050]: ../spec/2026_10_15_22_24_050-evaluation.md
[s051]: ../spec/2026_11_12_12_39_051-branch-games.md
[s052]: ../spec/2026_11_13_17_19_052-ops-metrics.md
[amb]: ../spec/README.md
[checkpoint1]: ../report/release/2026_11_08_12_21_checkpoint1.md
[checkpoint1_rl]: ../report/release/2026_11_08_13_35_checkpoint1_rl.md
[jepa_survey]: ../report/survey/2026_11_08_19_26_jepa_survey.md
[commands]: commands.md

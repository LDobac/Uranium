# 054 — Exploration Games (탐색 판)

- **Status**: Approved (2026-11-18, 사용자 확인 2026-11-18. 내용은 사용자 결정 2026-11-18, 근거 [jepa_predictor_data](../report/investigation/2026_11_18_08_58_jepa_predictor_data.md) 8·9절. AMB-054-1 Resolved 2026-11-18: 기본안 확정)
- **Related**: [020](2026_10_15_22_24_020-env-orchestration.md)(plan·평가 작업 R15·R22), [021](2026_10_15_22_24_021-opponent-pool.md)(T0·T1), [031](2026_10_15_22_24_031-action-space.md)(031-R4 마스크), [032](2026_10_15_22_24_032-model-architecture.md)(032-R16 상태), [034](2026_10_15_22_24_034-action-table.md)(`type_mask`), [041](2026_10_15_22_24_041-imitation-learning.md)(041-R28 기억 기록), [044](2026_11_14_08_18_044-self-game-distillation.md)(044-R13 ⑤ F), [050](2026_10_15_22_24_050-evaluation.md), [051](2026_11_12_12_39_051-branch-games.md)(051-R2 행동 기록, R4 상태 되살리기, R7 강제, R14 장비), [053](2026_11_18_08_58_053-planning-predictor-data.md)(053-R8 학습)

## Purpose
행동 효과를 배우려면 같은 상황에서 다른 행동을 고른 기록과 그 확률(propensity)이 필요하다([점검](../report/investigation/2026_11_18_08_58_jepa_predictor_data.md) 8절). 모델이 평소처럼 두는 판에서 몇 결정만 정한 확률로 다른 운영 행동을 두게 해, 053-R8 학습용 개입 데이터를 만든다.

## Scope & Non-goals
- **범위**: OpenBW lockstep, `players[0]` = F(044-R13 ⑤) stochastic, T0·공개 OSS(T1) 상대, 탐색 시점·선택·기록, 추론 응답 `type_logp`, plan 키 `explore`.
- **Non-goals**
  - 평가·게이트·경로 측정 표(050)에 넣는 일.
  - 판을 다시 두는 분기(051). 한 판 안에서 상태를 되살려 한 결정만 바꾼다.
  - RL 학습 데이터로 쓰는 일. 053 예측기 학습만.
  - F 전 판(gn 등). 탐색 판은 F만 둔다(053-R1).

## 용어
| 용어 | 뜻 |
|---|---|
| 탐색 시점 | 판마다 정한 3개 시각 τ₁·τ₂·τ₃ |
| 탐색 결정 | τ 뒤 첫 결정 중 열린 운영 행동이 2개 이상인 결정 |
| 운영 행동 | `diag plan-dump` `op_actions`(034 그룹 build·addon·train·research·upgrade·cancel + Lift·Land) |
| a0 | 탐색 결정에서 정책이 평소대로 뽑은 행동 종류 |
| `cands` | 후보: 열린 운영 행동 중 π 상위 4개(K = 4), a0 제외 |

## Requirements
| ID | 요구사항 |
|---|---|
| 054-R1 | **MUST** 판 구성. `players[0]` = F, `stochastic`. 상대 종족 판 기준 저그 50 : 프로토스 30 : 테란 20, 종족마다 T0 스크립트 봇 50 · 공개 OSS 봇(T1) 50. T0 빌드: 저그 = 021 표준·`zerg_rush` 반반, 프로토스·테란 = 021 표준. T1 = 021 그 종족 OSS 목록 균등. 맵 투혼·서킷브레이커 반반. plan 설정(`zero_last_failed`, `surrender`)은 F의 050 평가 설정과 같게. 10:00 상한(`max_frames = 14400`). 리플레이·기억 기록(041-R28 `memory_log`)·행동 기록(051-R2 `act_log`) 켬 |
| 054-R2 | **MUST** 탐색 시점. 판마다 3개, 판 seed만으로 정한다(같은 seed → 같은 시각). 창 [2:00, 9:00)(2,880~12,960프레임), 시각끼리 60초 이상 떨어짐. 각 시점은 프레임 ≥ τ인 첫 결정 중 열린 운영 행동(034 `type_mask`)이 2개 이상인 결정에서 걸린다. 한 결정에는 한 시점만(먼저 걸린 시점, 나머지는 다음 맞는 결정으로). 9:00 전에 맞는 결정이 없으면 그 시점은 걸리지 않고 사유를 남긴다 |
| 054-R3 | **MUST** 선택. 탐색 결정에서 ① feature 추출 뒤 자기 슬롯 Core 상태를 저장한다(051-R3 시점과 같음). ② 평소대로 추론해 a0를 뽑는다. ③ `cands` = 열린 운영 행동을 π 내림차순으로 a0를 빼고 상위 K = 4개. ④ 확률 ε = 0.5로: 저장한 상태를 되살리고(051-R4 `SetState`) `cands`에서 균등하게 뽑은 c로 `type_mask`를 {c}만 남겨(051-R7 ① `force`와 같은 좁히기) 다시 추론한다. 그 결과가 보낼 결정이다. ⑤ 아니면 a0를 그대로 보낸다(대조). ⑥ 위치·유닛·지연은 모델이 뽑는다. 기억 기록·행동 기록은 보낸 결정으로 남긴다. ⑦ 동전·c 뽑기는 판 seed와 시점 번호로 정한 난수(결정론) |
| 054-R4 | **MUST** 기록. 탐색 결정마다 행동 기록 줄에 `explore`: 시점 번호, τ, 프레임, ε, K, 후보(id, π) 목록(`cands` 전체), a0, 동전(`forced`/`control`), c(대조면 null), 보낸 결정의 executor 거절 여부, propensity = a0 조건부 확률(`forced` ε / \|cands\|, `control` 1 − ε). 걸리지 않은 시점은 끝 줄에 사유와 함께. 판 결과는 세션 결과(053-R2) |
| 054-R5 | **MUST** 추론 응답 `type_logp`. 세션이 요청한 결정(탐색 결정, 051-R2 `policy_top` 결정)에 한해 응답에 마스크 적용 행동 종류 log-prob(034 행동 수 길이, 닫힌 것 −∞)를 더한다. 값은 헤드의 마스크 적용 log-softmax(032 `heads.action_logp`)와 같다. 다른 `act` 출력은 바뀌지 않는다 |
| 054-R6 | **MUST** 규모. 시범 1,000판 먼저. 시범 데이터로 053-R8 예측기를 학습해 떼어 둔 데이터(탐색 판 20%, B2 떼어 둔 줄기)에서 나아지면 10,000판으로 늘린다. 나아짐 판단과 확대는 결과를 보고 사용자가 정한다 |
| 054-R7 | **MUST** 운영. 평가 작업(020-R15)으로 돌린다. 경로 측정·평가 게임과 hostd 자리·추론 GPU를 함께 쓰지 않는다(051-R14 규칙). 결과는 050 표·게이트에 섞지 않는다 |

### 다른 spec에 넣은 변경(2026-11-18 각 spec에 반영)
| spec | 넣을 문구 |
|---|---|
| 020-R22 | ⑤ 추가: "`explore`(표, 선택): 054 탐색 판. 키 `points = 3`, `window_s = [120, 540]`, `gap_s = 60`, `k = 4`, `epsilon = 0.5`, `min_open = 2`. `act_log = true`·`memory_log = true` 필수, 없으면 plan 로드 오류". ④ 문구 "세 키가 없는 plan"을 "R22 키(`explore` 포함)가 없는 plan"으로 |
| 031-R4 | 예외 문구에 "054 탐색 결정의 `force` 좁히기(한 결정)"를 더한다. 닫힌 비트를 열지 않고 `unit_cap`·`loc_class_mask`·executor 규칙 그대로 |
| 032-R16 | 끝에 추가: "(054-R5) 세션이 요청한 결정에 한해 응답에 `type_logp`를 더한다. `act` 출력은 바뀌지 않는다" |
| 050 | "분기 게임(051)" 항목에 054 탐색 판을 더한다 |

## Interfaces
**plan 예**
```toml
name = "explore-pilot"
save_replay = true
memory_log = true
act_log = true
max_frames = 14400
seeds = [ ... ]
[explore]
points = 3
window_s = [120, 540]
gap_s = 60
k = 4
epsilon = 0.5
min_open = 2
[[matchup]]
map = "fighting_spirit"
games = <n>
p0 = { kind = "model", checkpoint = "checkpoints/<F>", sample = "stochastic" }
p1 = { kind = "t0", race = "zerg", build = "zerg_rush" }
```

**행동 기록 `explore` 필드(054-R4)**
```json
"explore":{"point":1,"tau":4123,"frame":4130,"epsilon":0.5,"k":4,
 "cands":[[37,0.21],[20,0.08],[41,0.05],[44,0.02]],"a0":0,"coin":"forced","c":20,
 "rejected":false,"propensity":0.125}
```
- 출력: `<shared>/eval/explore-<날짜>/`(061), 평가 작업 폴더(리플레이, 세션 결과, `acts/`, `memory/`).

## Acceptance Criteria
- 054-AC1 (CPU 단위 테스트, Rust) 시점: 같은 seed 같은 τ, 창·간격 경계, 열린 운영 행동 < 2이면 다음 결정으로, 한 결정 한 시점, 9:00 전 없음 → 사유. 후보: a0 제외 상위 4, 동전·c 결정론, propensity `forced` ε / \|cands\|·`control` 1 − ε.
- 054-AC2 (CPU pytest) 되살리기: 저장 → act → 되살리기 → act의 둘째 출력이 저장 직후 act와 비트 단위로 같다(051-AC2와 같은 배치 조건). 다른 슬롯 불변.
- 054-AC3 (CPU pytest) `type_logp`: 헤드의 마스크 적용 log-prob와 비트 단위로 같고, 요청 없는 결정의 응답·`act` 출력은 바뀌지 않는다.
- 054-AC4 (작은 end-to-end, Linux CPU 호스트, CPU 추론) 20판: 판마다 시점 3개 기록(걸림 또는 사유), `forced` 결정의 보낸 행동 = c, `control` = a0, 기억 기록·행동 기록·세션 결과 완전, 053-R5 build·덤프가 끝까지 돈다.

## Risks & Open Questions
- **F 대기**: F가 정해지기 전에는 판을 두지 않는다(코드·AC4는 gn으로 시험 가능).
- **효과 희석**: 한 결정 강제는 정책이 30초 안에 같은 행동으로 되돌릴 수 있다(B1 희석, [branch_b1](../report/experiment/2026_11_18_01_10_branch_b1.md)). 053-R8 학습 결과로 본다.
- **OSS 비결정성**: 같은 seed로 판을 다시 둘 수 없다. 이 spec은 다시 두기를 쓰지 않으므로 무관하다.
- **추론 비용**: `forced` 탐색 결정(판마다 평균 1.5개)만 추론이 한 번 더. 영향 TBD(시범).

### 모호함 (AMB-054-1 Resolved 2026-11-18: 기본안 확정, 사용자 결정 2026-11-18)
> **AMB-054-1 (Resolved 2026-11-18: 기본안 확정, 사용자 결정 2026-11-18)**: 상대 세부와 plan 설정.
> - 결정: 종족 안 T0 50 · T1 50, 저그 T0 = 021 표준·`zerg_rush` 반반, 프로토스·테란 T0 = 021 표준, T1 = 021 그 종족 OSS 목록 균등, 맵 반반, `zero_last_failed`·`surrender`는 F의 050 평가 설정과 같게(054-R1).

## Related
- 근거: [jepa_predictor_data](../report/investigation/2026_11_18_08_58_jepa_predictor_data.md), [branch_b1](../report/experiment/2026_11_18_01_10_branch_b1.md).
- 학습: [053](2026_11_18_08_58_053-planning-predictor-data.md) R8. 분기 B2: [051](2026_11_12_12_39_051-branch-games.md) R16.

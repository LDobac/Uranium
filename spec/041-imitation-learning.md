# 041 — Imitation Learning (BC)

- **Status**: Approved (2026-10-15, 2026-10-17 개정: R11 혼합 선택 레이블 — AMB-041-1)
- **Related**: [032](032-model-architecture.md), [040](040-replay-dataset.md), [042](042-rl-training.md), [050](050-evaluation.md)

## Purpose
리플레이 데이터셋으로 정책을 지도학습한다. 결과물은 RL의 초기 정책이자 KL 정규화의 teacher다.

## 맵 전략
"맵 고정"은 평가·RL 범위에만 적용하고, 데이터 범위에는 적용하지 않는다.

| 단계 | 데이터 | 목적 |
|---|---|---|
| S1 사전학습 | 전체 맵 TvX | 맵 무관 지식(유닛 상성, 마이크로, 빌드, 매크로). 맵 조건은 공간 채널과 map 임베딩으로 처리 |
| S2 파인튜닝 | 투혼·서킷브레이커만 + 대칭 증강 | 맵 특화(초크, 심시티, 확장 경로, 거리 감각) |

## Requirements
| ID | 요구사항 |
|---|---|
| 041-R1 | **MUST** 손실 = 헤드별 cross-entropy 합이다. 헤드 가중치는 action_type 1.0, delay 0.5, queued 0.2, selected 1.0, target_unit 1.0, target_loc 1.0으로 한다. 해당 action_type에서 쓰지 않는 헤드는 마스크로 0 처리한다 |
| 041-R2 | **MUST** 학습은 시퀀스 단위(T=64, truncated BPTT)로 한다. Core 상태는 같은 게임 시점의 연속 구간 사이에서 이어 쓴다 |
| 041-R3 | **MUST** 옵티마이저는 AdamW로 하고, warmup + cosine 스케줄, grad clip 1.0을 쓴다. 하이퍼파라미터는 설정 파일로 관리한다 |
| 041-R4 | **MUST** S1→S2 전환 시 S1 체크포인트를 보존하고, S2는 학습률을 1/10로 낮춘다 |
| 041-R5 | **MUST** ablation: (A) 대상 맵만 학습 vs (B) S1+S2를 비교하고 결과를 `reports/bc_ablation.md`에 기록한다 |
| 041-R6 | **SHOULD** NoOp과 Move 같은 빈도 높은 클래스에는 클래스 균형 가중치를 적용할지 실험한다 |
| 041-R7 | **MUST** value head(win)도 BC 단계에서 학습한다. 정답은 리플레이의 실제 승패(테란 시점 승 +1 / 패 −1)이고, 손실은 MSE(가중치 0.5, 설정 가능)다. 입력에는 재생 모드의 `PrivilegedSummary`(011-R12)를 쓴다. build·units·value 헤드는 BC에서 학습하지 않는다 |
| 041-R8 | **MUST** z 조건은 해당 리플레이 자신의 z를 쓰고, 샘플의 25%는 `z_present=false`로 비워서 학습한다. RL(033-R2)과 같은 비율이다 |
| 041-R9 | **MUST** 데이터 로더는 저장 shard와 재생 스트리밍(040-R12)을 하나의 인터페이스로 읽는다. 두 경로의 출력은 동일하다(040-AC6) |
| 041-R10 | **MUST** 손실 가중치, 학습률 등 하이퍼파라미터는 설정 파일로 관리하고 체크포인트에 함께 기록한다. 튜닝 대상이며 정책 결정 사항이 아니다 |
| 041-R11 | **MUST** 혼합 선택 레이블(2026-10-17, AMB-041-1): 게임은 명령을 선택된 유닛마다 따로 적용하고, 받을 수 없는 유닛은 건너뛴다(OpenBW `action_*`). 그래서 레이블의 `selected`에서 해당 액션의 선택 자격([034](034-action-table.md) `select_cap`)이 없는 유닛을 빼고 학습한다. 남는 유닛이 없으면(게임에서 아무 일도 일어나지 않는 명령) 무효 레이블(weight 0, 031-R4)이다. 에이전트 기억(030 §4 직전 행동)에도 뺀 뒤의 유닛을 기록한다. 이 정책은 학습 시점 변환이며 shard에는 원래 선택을 저장한다 |

## 평가 지표
- 오프라인 지표:
  - 대상 맵 test 셋 기준 action_type top-1/top-5 정확도
  - delay MAE
  - target_loc 타일 오차
  - selected F1
- 온라인 지표: BC 정책 vs T0 스크립트 봇 승률(종족×맵). 측정 방법은 [050](050-evaluation.md)을 따른다.

## Acceptance Criteria
- 041-AC1: 대상 맵 test action_type top-1 ≥ 50%다.
- 041-AC2: BC 정책이 T0 봇 상대로 각 매치업에서 승률 ≥ 50%다.
- AC1과 AC2의 수치는 초기값이다(2026-10-15 확정). M5 기준선 측정 후 조정을 제안할 수 있고, 조정하려면 사용자 확인이 필요하다(050과 같은 방식).
- 041-AC3: (B)가 (A)보다 대상 맵 test 지표가 같거나 높다. 그렇지 않으면 맵 전략을 재검토한다.

## 구현 메모 (비규범)
- GPU 메모리(2026-10-17, [`reports/m5_bc_memory.md`](../reports/m5_bc_memory.md)): `config/bc.toml`의 `micro_lanes`만큼 레인을 묶어 forward/backward를 하고, gradient를 합쳐 AdamW step을 한 번 한다. 손실은 배치 전체 유효 스텝 수로 나누므로 `lanes` 전체를 한 번에 처리한 것과 같은 업데이트다(R2·R3 그대로). 최대 메모리는 그룹 크기를 따른다(기본 1 = 레인 1 × T64 × N512).
- `pad_entities`는 배치 N을 그 배수로 올리는 마스크 패딩이고, `grad_checkpointing`은 Burn balanced checkpointing이다. 둘 다 결과를 바꾸지 않는다.

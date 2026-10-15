# 033 — Reward

- **Status**: Approved (2026-10-15)
- **Related**: [042](042-rl-training.md), [040](040-replay-dataset.md)

## Purpose
RL 단계의 보상 함수를 정의한다. 최종 목적은 승리이며, 보상 설계는 이 목적에 정렬되어야 한다.

## 보상 구성
| 항목 | 정의 | 가중치 (초기) | 스케줄 |
|---|---|---|---|
| r_win | 종료 시 승 +1, 패 −1, 무 0. 60분 시간 초과는 무승부 0([020](020-env-orchestration.md)) | 1.0 | 고정 |
| r_build | z의 빌드 순서와 현재 빌드 순서의 Levenshtein 거리 변화(감소 시 +) | 0.3 | 리그 진행에 따라 0으로 감쇠 |
| r_units | z의 유닛 구성 누적 통계와 Hamming/L1 거리 변화 | 0.3 | 동일 |
| r_value | (적 손실 자원 − 아군 손실 자원) / 1000 차분 | 0.05 | 커리큘럼 **C0 단계에서만** 사용([042](042-rl-training.md) R5), C1부터 0 |

## Requirements
| ID | 요구사항 |
|---|---|
| 033-R1 | **MUST** 최종 평가와 체크포인트 선택은 r_win 기준으로만 한다 |
| 033-R2 | **MUST** pseudo-reward(r_build, r_units)는 z가 주어진 에피소드에서만 활성화한다. z는 리플레이 분포에서 샘플링하며, 일부 에피소드는 z 없이 진행한다(기본 25%) |
| 033-R3 | **MUST** 보상 항목별 값은 분리해 로깅하고, 항목마다 별도 value head로 학습한다. 대응은 win↔r_win, build↔r_build, units↔r_units, value↔r_value다([032](032-model-architecture.md) §2.6) |
| 033-R6 | **MAY** value head는 학습 전용 privileged 입력(상대 실제 상태 요약)을 쓸 수 있다. 정책 헤드와 추론 경로에는 절대 연결하지 않는다(032-R11) |
| 033-R4 | **MUST** 보상 가중치와 감쇠 스케줄(감쇠 속도 등)은 튜닝 대상이다. 설정 파일로 관리하고, 각 체크포인트에 사용한 값을 기록한다 |
| 033-R5 | **MUST NOT** 맵핵 정보(보이지 않는 적 정보)를 보상 계산에 쓰지 않는다. r_value의 적 손실은 관측된 파괴 이벤트만 센다 |

## Acceptance Criteria
- 033-AC1: 보상 함수 단위 테스트를 통과한다. 합성 시나리오에서 부호와 크기가 맞는지 확인한다.
- 033-AC2: pseudo-reward를 끈 ablation 대비 초기 1M 스텝의 승률 곡선을 비교해 [042](042-rl-training.md)에 기록한다.

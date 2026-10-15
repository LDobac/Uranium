# 030 — Observation Features

- **Status**: Approved (2026-10-15)
- **Related**: [010](010-game-protocol.md), [031](031-action-space.md), [032](032-model-architecture.md), [034](034-action-table.md), [040](040-replay-dataset.md)

## Purpose
RawFrame과 에이전트 메모리를 모델 입력 텐서로 변환하는 규칙과 **텐서 계약**을 정의한다. 학습(OpenBW), 리플레이(BC), 실전(SC:R)에서 **같은 코드 경로**를 쓴다.

## Scope & Non-goals
- 범위:
  - 엔티티, 공간, 스칼라, 전략 조건(z) 텐서
  - 마스크용 capability 텐서(정의는 [034](034-action-table.md))
  - 정규화 규칙
- 비범위:
  - 모델 내부 구조([032](032-model-architecture.md))
  - 보상([033](033-reward.md))

## Requirements
| ID | 요구사항 |
|---|---|
| 030-R1 | **MUST** feature 추출은 `uranium-env::features`의 순수 함수로 구현한다: `(RawFrame, &mut AgentMemory, &MapInfo) -> ObsTensors` |
| 030-R2 | **MUST** 자기 시야 밖 정보는 쓰지 않는다. 적 건물 기억은 "마지막 목격 상태 + 경과 프레임"으로만 표현한다 |
| 030-R3 | **MUST** 리플레이 BC 데이터 생성 시 관측자 시점이 아닌 **테란 플레이어 시점**으로 시야를 필터링한다 |
| 030-R4 | **MUST** 대칭 증강 변환(회전/반전)을 feature 수준에서 지원하고, 액션 좌표에도 같은 변환을 적용한다 |
| 030-R5 | **SHOULD** feature 스키마에 버전을 붙이고 체크포인트에 저장한다 |
| 030-R6 | **MUST** 공간 텐서는 **128×128 빌드타일(32px) 해상도**로 한다. 맵이 128보다 작은 축은 zero-pad하고 `valid_map` 채널로 표시한다. 각 축이 128을 넘는 맵은 BC 데이터에서 제외한다([040](040-replay-dataset.md)) |
| 030-R7 | **MUST** 엔티티 수 상한은 N_max=512이다. 배치 안에서는 최대 N으로 동적 패딩하고 `entity_mask`로 표시한다 |
| 030-R8 | **MUST** 마스크 계산용 텐서(`type_mask`, `unit_cap`, `loc_class_mask`)는 feature와 같은 시점의 상태에서 계산한다. 의미는 [034](034-action-table.md)를 따른다 |
| 030-R9 | **MUST** 학습 전용 privileged feature는 [010](010-game-protocol.md)의 `PrivilegedSummary`로 만들고 `PrivilegedTensors`로 분리한다. 추론용 `ObsTensors`에는 절대 포함하지 않는다 |

## 1. 텐서 계약
| 이름 | shape / dtype | 내용 |
|---|---|---|
| `entity_cat` | [B,N,4] i32 | unit_type(234), owner(3: self/enemy/neutral), order(190), last_action_type(A, 아군만, 없으면 0) |
| `entity_cont` | [B,N,F_c≈76] f32 | §2 |
| `entity_mask` | [B,N] bool | 패딩이 아니면 true |
| `entity_cell` | [B,N] i32 | 128×128 타일 인덱스 `y*128+x` (scatter용) |
| `entity_pos` | [B,N,2] f32 | 픽셀 좌표 (distance bias 계산용, [032](032-model-architecture.md)) |
| `spatial` | [B,C≈26,128,128] f32 | §3 |
| `scalar` | [B,F_s≈512] f32 | §4 |
| `scalar_cat` | [B,3] i32 | enemy_race(4: T/P/Z/unknown), map_family, start_loc(8) |
| `z_build` | [B,20] i32 | 전략 빌드 순서(action_type ID, 0=패딩) — §5 |
| `z_units` | [B,U] f32 | 전략 유닛 구성(bag-of-units, 테란 유닛·건물 타입별 목표 비율) — §5 |
| `z_present` | [B] bool | z 조건 사용 여부 |
| `type_mask` | [B,A] bool | [034](034-action-table.md) |
| `unit_cap` | [B,N,K≈48] bool | [034](034-action-table.md) |
| `loc_class_mask` | [B,L=8,128,128] bool | [034](034-action-table.md) |

## 2. 엔티티 연속 feature (`entity_cont`)
포함 순서: 아군 → 보이는 적 → 기억된 적 건물 → 중립(자원). N_max를 넘으면 중립부터 잘라낸다.

| feature | 인코딩 | 차원 |
|---|---|---|
| hp, shield, energy | 최대값 대비 비율 + sqrt(절대값)/16 | 6 |
| 위치 | x/(mapW·32), y/(mapH·32) + Fourier(8주파수 × 2축 × sin/cos) | 34 |
| vel, angle | 정규화 연속값 | 3 |
| cooldowns | ground/air/spell 쿨다운 / 최대 | 3 |
| progress | 건설·생산·연구 진행도 [0,1], build_queue 길이/5 | 2 |
| flags | 완성, 시즈, 버로우, 클록, 탐지됨, 리프트, 운반, 탑승, idle, 피격, 애드온 보유, 건설 중 SCV | 12 |
| 상태이상 | Lockdown, Irradiate, DefensiveMatrix, Stasis, Plague, Ensnare, Maelstrom, Blind, Parasite, Stim 남은 시간, (클록·버로우는 flags) | 10 |
| 스킬 자원 | spider_mine_count/3, nuke_armed | 2 |
| memory | is_remembered, frames_since_seen/2048 | 2 |
| resources | 미네랄 잔량/1500, 가스 잔량/5000 | 2 |

## 3. 공간 채널 (`spatial`, 128×128)
| 구분 | 채널 | 수 |
|---|---|---|
| 정적 | height one-hot(3), walkable, buildable, start_loc, base_loc, choke(사전계산), wall_candidate(사전계산, 힌트 전용), valid_map | 10 |
| 시야 | visible_now, explored, enemy_last_seen_age | 3 |
| 지형 동적 | creep, own_building_footprint, enemy_building_footprint(기억 포함) | 3 |
| 밀도 | own_ground, own_air, enemy_ground, enemy_air (log(1+count)) | 4 |
| 위험 | hazard: Psionic Storm, Dark Swarm, Disruption Web, 핵 조준점, EMP·Irradiate·기타 투사체 (관측된 것만) | 5 |
| 기타 | scanner_active(아군 스캔 범위) | 1 |

`wall_candidate`는 맵 사전계산 결과다. 정책에 주는 **힌트일 뿐** 강제 위치가 아니다([031](031-action-space.md)).

## 4. 스칼라 (`scalar`)
- 시간: frame/86400, log(1+frame)/12
- 자원: 미네랄과 가스 log(1+x)/10, supply used/total/max
- 업그레이드 레벨 63(정규화), 테크 47(완료 여부), 연구·업그레이드 진행 중 여부
- 아군 유닛·건물 타입별 수(테란 subset): 완성 수와 생산 중 수
- 관측된 적 유닛 타입별 **최대 동시 관측 수**(전 종족 subset)
- 직전 액션: action_type one-hot, delay bucket, 실행 실패 여부(031-R10)

## 5. 전략 조건 z (확정: 포함)
| 상황 | z 출처 |
|---|---|
| BC | 해당 리플레이 테란 플레이어의 실제 빌드 순서(처음 20개 Build/Train/Research/Upgrade 액션) + 게임 종료까지의 유닛 구성 |
| RL | 리플레이 z 분포에서 샘플링. 25%는 `z_present=false` ([033](033-reward.md)) |
| 실전 | 사용자가 지정한 z(전략 프리셋), 또는 상대 종족별 기본 z 분포에서 샘플링 |

## Acceptance Criteria
- 030-AC1: 같은 리플레이 프레임을 OpenBW 경로와 SC:R 경로로 추출했을 때 feature가 비트 단위로 일치한다(013-AC2 기반).
- 030-AC2: 시야 필터 테스트: 안 보이는 적 유닛이 엔티티, 공간, 스칼라 어디에도 새지 않는다.
- 030-AC3: 512 엔티티 기준 feature + 마스크 추출이 CPU에서 ≤ 1ms다.
- 030-AC4: 대칭 증강 변환 후 역변환하면 원본과 일치한다. 액션 좌표도 포함한다.
- 030-AC5: `ObsTensors` 타입에 privileged 필드가 없음을 컴파일 타임에 보장한다(타입 분리).

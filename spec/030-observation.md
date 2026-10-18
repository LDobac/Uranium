# 030 — Observation Features

- **Status**: Approved (2026-10-15. 2026-10-19 개정(BC 전 리뷰 결정, 사용자 확정): R2 맵 공개 자원 예외(010-R5, E-3), R4 `entity_cell` 재계산·flip 기준 크기(B-6), R10 평가 대전 `last_failed` 0 고정(E-5), R11 Random 상대 종족 모름(E-6), §3 hazard 반지름·채널(E-1), F_c = 76)
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
| 030-R2 | **MUST** 자기 시야 밖 정보는 쓰지 않는다. 적 건물 기억은 "마지막 목격 상태 + 경과 프레임"으로만 표현한다. 예외: 맵에서 공개적으로 알 수 있는 중립 자원(미네랄 필드·베스핀 간헐천)은 시야 밖이어도 마지막 관측 상태(한 번도 못 봤으면 맵 초기값)로 엔티티에 넣는다(010-R5 예외, 2026-10-19 사용자 결정 E-3, AMB-030-2) |
| 030-R3 | **MUST** 리플레이 BC 데이터 생성 시 관측자 시점이 아닌 **테란 플레이어 시점**으로 시야를 필터링한다 |
| 030-R4 | **MUST** 대칭 증강 변환을 feature 수준에서 지원하고, 액션 좌표에도 같은 변환을 적용한다. 학습에 쓰는 변환은 원본, 좌우 뒤집기, 상하 뒤집기, 둘 다(180° 회전) **4종**이다. 가로·세로를 바꾸는(transpose) 4종은 쓰지 않는다(2026-10-18 사용자 결정). 변환은 엔티티 위치·속도·각도, 공간 채널, 마스크, 액션 좌표(footprint 사각형 기준)와 `scalar_cat.start_loc`에 함께 적용한다. `start_loc`은 변환한 자기 시작 위치(자원 건물 4×3 footprint의 왼쪽 위 타일)에 가장 가까운 맵 시작 위치의 인덱스로 바꿔 공간 채널과 일치시킨다. **알려진 한계**: 좌우 뒤집기에서는 애드온이 본체 왼쪽에 붙은 것처럼 보인다(실제 게임은 항상 오른쪽). 사용자가 좌우 뒤집기 유지를 선택했다(2026-10-18). **변환 기준(2026-10-19 명시, 코드 `crates/uranium-env/src/features/augment.rs`)**: 뒤집기 기준 크기는 128 패딩이 아니라 **실제 맵 크기**(W×H 타일)다. 픽셀 좌표는 x → W·32 − x(상하는 H), 타일·footprint 사각형 [x, x+fw)는 x → W − x − fw로 바꾼다. 공간 채널과 위치 마스크는 유효 영역(W×H)만 옮기고 패딩 영역은 0으로 둔다. `entity_pos`·위치 feature는 변환한 픽셀로 다시 계산하고, **`entity_cell`도 변환한 픽셀에서 다시 계산한다**(타일 = 픽셀 ÷ 32 내림, 맵 범위로 자름. 추출 때와 같은 규칙). 타일 인덱스를 따로 뒤집으면 중심이 32의 배수인 짝수 폭 건물에서 1타일 어긋나기 때문이다(B-6, AMB-030-2) |
| 030-R5 | **SHOULD** feature 스키마에 버전을 붙이고 체크포인트에 저장한다 |
| 030-R6 | **MUST** 공간 텐서는 **128×128 빌드타일(32px) 해상도**로 한다. 맵이 128보다 작은 축은 zero-pad하고 `valid_map` 채널로 표시한다. 각 축이 128을 넘는 맵은 BC 데이터에서 제외한다([040](040-replay-dataset.md)) |
| 030-R7 | **MUST** 엔티티 수 상한은 N_max=512이다. 배치 안에서는 최대 N으로 동적 패딩하고 `entity_mask`로 표시한다 |
| 030-R8 | **MUST** 마스크 계산용 텐서(`type_mask`, `unit_cap`, `loc_class_mask`)는 feature와 같은 시점의 상태에서 계산한다. 의미는 [034](034-action-table.md)를 따른다 |
| 030-R9 | **MUST** 학습 전용 privileged feature는 [010](010-game-protocol.md)의 `PrivilegedSummary`로 만들고 `PrivilegedTensors`로 분리한다. 추론용 `ObsTensors`에는 절대 포함하지 않는다 |
| 030-R10 | **MUST** 직전 액션 실패 여부(`last_failed`, §4, 031-R10) 스칼라(2026-10-19 사용자 결정, E-5, AMB-030-2): BC 모델을 그대로 쓰는 평가 대전(코어 비교 032-R17, 041-AC2 등)에서는 설정 플래그로 **0으로 고정**한다. 이유: 리플레이 재생에는 명령 실패 이벤트가 없어 BC 학습 데이터에서 이 값이 항상 0이다. 평가 대전에서 처음 보는 값 1이 들어가면 학습하지 않은 입력이라 잡음이 된다. RL부터는 실제 값을 쓴다. 플래그를 쓰는 코드에는 이 이유를 주석으로 남긴다 |
| 030-R11 | **MUST** 상대 종족(`scalar_cat.enemy_race`, 2026-10-19 사용자 결정, E-6, AMB-030-2): 로비에서 종족을 고른 상대는 처음부터 그 종족이다. 로비에서 **Random**인 상대는 첫 적 유닛을 관측하기 전까지 `unknown`이고, 관측한 뒤 그 유닛의 종족으로 정한다(000 AMB-000-3). 리플레이(BC)와 라이브 게임에 같은 규칙을 쓴다 |

## 1. 텐서 계약
| 이름 | shape / dtype | 내용 |
|---|---|---|
| `entity_cat` | [B,N,4] i32 | unit_type(234), owner(3: self/enemy/neutral), order(190), last_action_type(A, 아군만, 없으면 0) |
| `entity_cont` | [B,N,F_c=76] f32 | §2 |
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
포함 순서: 아군 → 보이는 적 → 기억된 적 건물 → 중립(자원). N_max를 넘으면 중립부터 잘라낸다. 중립 자원은 지금 보이는 것과, 시야 밖이면 마지막 관측 상태(못 봤으면 맵 초기값)를 넣는다(R2 예외). 시야 밖 자원은 memory feature(`is_remembered`, 경과 프레임)를 기억된 적 건물과 같은 방식으로 채운다. **자원 출처(2026-10-19 기술 결정, AMB-030-3)**: 맵 초기값(자원 종류·위치·초기량)은 맵 사전계산(`MapPrecompute`, CHK 해시별, CHK 유닛 배치에서 추출, 031 맵 사전계산)에서 읽는다. BC와 라이브 게임이 같은 사전계산을 쓴다(라이브는 같은 CHK로 만든 사전계산 또는 031 `uranium-replay mapgen` 결과). 시야 밖 자원 엔티티와 관측된 자원·레이블 target의 대응은 **위치(타일) 매칭**으로 한다(자원은 움직이지 않는다). 한 번도 보지 못한 맵 공개 자원은 게임 시작 시점에 맵에서 안 것으로 보고 `seen_frame = 0`으로 둔다(경과 프레임 feature = 현재 프레임 ÷ 2048, 다른 기억 엔티티와 같은 상한 없는 선형 값, 2026-10-19 기술 결정). 브리지·프로토콜(010 Interfaces)은 바꾸지 않는다.

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
| 시야 | visible_now, explored, last_seen_age (이전 이름 enemy_last_seen_age) | 3 |
| 지형 동적 | creep, own_building_footprint, enemy_building_footprint(기억 포함) | 3 |
| 밀도 | own_ground, own_air, enemy_ground, enemy_air (log(1+count)) | 4 |
| 위험 | hazard: Psionic Storm, Dark Swarm, Disruption Web, 핵 조준점, 기타(EMP·Irradiate·Plague·Ensnare·그 밖의 투사체) (관측된 것만) | 5 |
| 기타 | scanner_active(아군 스캔 범위) | 1 |

`last_seen_age`는 타일을 자기 시야로 마지막으로 본 뒤 지난 시간이다(정찰 기록, 032 §2.4의 장기 정보). 지금 보이는 타일은 0, 한 번도 보지 못한 타일은 1, 그 외는 log(1+경과 프레임)/log(1+86400)이다. 적 유닛의 과거 위치는 담지 않는다. 단기 이동은 코어가 담당한다 (AMB-030-1 Resolved, 2026-10-16: 사용자가 Claude에게 결정을 위임).

hazard 채널(2026-10-19 사용자 결정, E-1, AMB-030-2): 관측된 효과마다 효과 위치를 중심으로 **그 효과의 실제 반지름**(1.16.1 게임 데이터 기준 영향 범위, 효과 종류별)의 원과 겹치는 타일을 1로 칠한다(고정 16px이 아님). Plague와 Ensnare도 hazard 채널(기타)에 넣는다. 효과별 반지름 값은 구현 때 게임 데이터에서 옮겨 이 절에 기록한다.

`wall_candidate`는 맵 사전계산 결과다. 정책에 주는 **힌트일 뿐** 강제 위치가 아니다([031](031-action-space.md)).

## 4. 스칼라 (`scalar`)
- 시간: frame/86400, log(1+frame)/12
- 자원: 미네랄과 가스 log(1+x)/10, supply used/total/max
- 업그레이드 레벨 63(정규화), 테크 47(완료 여부), 연구·업그레이드 진행 중 여부
- 아군 유닛·건물 타입별 수(테란 subset): 완성 수와 생산 중 수
- 관측된 적 유닛 타입별 **최대 동시 관측 수**(전 종족 subset)
- 직전 액션: action_type one-hot, delay bucket, 실행 실패 여부(031-R10, 평가 대전 처리는 R10)

## 5. 전략 조건 z (확정: 포함)
| 상황 | z 출처 |
|---|---|
| BC | 해당 리플레이 테란 플레이어의 실제 빌드 순서(처음 20개 Build/Train/Research/Upgrade 액션) + 게임 종료까지의 유닛 구성 |
| RL | 리플레이 z 분포에서 샘플링. 25%는 `z_present=false` ([033](033-reward.md)) |
| 실전 | 사용자가 지정한 z(전략 프리셋), 또는 상대 종족별 기본 z 분포에서 샘플링 |

## 6. 정규화 상수 (구현 기록, 2026-10-19)
위 절에 값이 없던 상수를 코드(`crates/uranium-env/src/features/extract.rs`, `norm` 모듈과 추출 함수)에서 옮겨 적는다. 동작은 바꾸지 않았다. BC 본학습 뒤에 바꾸면 재학습이 필요하므로, 바꿀 때는 spec 개정과 사용자 확인을 먼저 한다.

| 입력 | 정규화 |
|---|---|
| 체력·실드·에너지 | 최대값 대비 비율(최대 1)과 √값 ÷ 16 두 개 |
| 속도 vx·vy (1/256 px/frame) | ÷ (256 × 8) |
| 방향 angle | ÷ 256 |
| 스펠 쿨다운 | ÷ 64, 최대 1 |
| 상태 타이머(락다운 등 8종), 스팀 | ÷ 256, 최대 1 |
| 생산 큐 길이 | ÷ 5 |
| 스파이더 마인 | ÷ 3 |
| 마지막으로 본 뒤 경과 프레임(엔티티) | ÷ 2048 |
| 미네랄 필드 / 가스 잔량 | ÷ 1500 / ÷ 5000 |
| `last_seen_age` 공간 채널(AMB-030-1) | 0 = 지금 보임, 1 = 본 적 없음, 그 밖은 ln(1 + age) ÷ ln(1 + 86400), 최대 1 |
| 공간 개수 채널 | ln(1 + n) |
| 스칼라 시간(프레임 f) | f ÷ 86400, ln(1 + f) ÷ 12 |
| 미네랄·가스(자기·적) | ln(1 + v) ÷ 10 |
| 서플라이 사용·전체 | ÷ 400 |
| 업그레이드 레벨 | ÷ 3 |
| 유닛·건물 개수(완료·건설 중·적 최대 관측·적 보유·적 생산 중) | ln(1 + n) ÷ 4 |

## Acceptance Criteria
- 030-AC1: 같은 리플레이 프레임을 OpenBW 경로와 SC:R 경로로 추출했을 때 feature가 비트 단위로 일치한다(013-AC2 기반).
- 030-AC2: 시야 필터 테스트: 안 보이는 적 유닛이 엔티티, 공간, 스칼라 어디에도 새지 않는다.
- 030-AC3: 512 엔티티 기준 feature + 마스크 추출이 CPU에서 ≤ 1ms다.
- 030-AC4: 대칭 증강 변환 후 역변환하면 원본과 일치한다. 액션 좌표도 포함한다.
- 030-AC5: `ObsTensors` 타입에 privileged 필드가 없음을 컴파일 타임에 보장한다(타입 분리).
- 030-AC6: 학습용 4종 변환에서 건물 footprint 크기가 바뀌지 않고(위치 마스크·라벨이 같은 footprint), 변환한 `start_loc` 인덱스가 변환한 공간 채널의 자기 시작 위치와 일치하며, 같은 입력·변환이면 출력이 같다(결정론).

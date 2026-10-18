# 031 — Action Space & Executor

- **Status**: Approved (2026-10-15, 2026-10-17 개정: 라이브 게임 맵 사전계산 키(AMB-031-4), executor 거부 시 동작(031-R4, AMB-031-5). 2026-10-19: 맵 사전계산 생성 경로 표기를 실제(`uranium-replay`)로 정정)
- **Related**: [010](010-game-protocol.md), [030](030-observation.md), [032](032-model-architecture.md), [034](034-action-table.md), [040](040-replay-dataset.md)

## Purpose
정책이 내는 **의미 단위 액션**의 구조와, 이를 프로토콜 `Command`로 바꾸는 최소 규칙 계층(executor)을 정의한다. action_type 전체 목록과 인자 테이블은 [034](034-action-table.md)에 있다.

## 확정 사항 (2026-10-15)
| 항목 | 결정 |
|---|---|
| 정책 구조 | 단일 end-to-end 정책. 매크로와 마이크로를 같은 헤드 체인으로 처리 |
| executor 범위 | **최소** (AMB-031-1 Resolved) |
| APM 제한 | **없음** (AMB-031-2 Resolved) |
| delay | 1–32프레임 비균등 bucket 유지 |
| 선택 상한 | **12기**. 게임 동시 선택 제한과 같다 |

## 액션 구조 (자기회귀 순서)
```
1 action_type  ∈ A (~100, 034)
2 delay        ∈ 16 buckets {1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32} 프레임 (다음 관측까지)
3 queued       ∈ {0,1}                         (action_table.uses_queued)
4 selected     ⊆ 아군 entities, 1..max_select(≤12), END 토큰 (action_table.uses_select)
5 target_unit  ∈ entities                     (target = "unit")
6 target_loc   ∈ 128×128 빌드타일              (target = "loc")
```
- 타깃 종류가 다른 액션은 **서로 다른 action_type**으로 분리한다(예: `Attack_Unit` / `Attack_Pos`). 그래서 5번과 6번 헤드는 상호 배타다.
- 각 action_type이 쓰는 헤드, 선택 자격, 타깃 자격, 위치 클래스는 [034](034-action-table.md) 테이블이 결정한다.

## Executor 규칙 (최소)
| ID | 규칙 |
|---|---|
| 031-R1 | **MUST** `executor = pick_building` 액션(Train, Research, Upgrade, ScannerSweep): 가능한 건물 중 큐가 가장 짧은 건물을 고른다(Scan은 energy 최대 Comsat). tie-break는 unit id 오름차순으로 결정론적으로 한다 |
| 031-R2 | **MUST** `executor = pick_worker` 액션(Build_*): 대상 타일까지 가장 가까운 비운반·비건설 SCV를 고른다. 운반 중인 SCV만 있으면 그중 가장 가까운 SCV를 고른다. **건설 위치는 정책이 고른 타일(좌상단)을 그대로 쓰고, 위치 보정은 하지 않는다.** 실행 시점에 위치가 무효면 명령 실패로 기록하고 재시도하지 않는다 |
| 031-R3 | **MUST** 선택 상한은 12기다. selected 헤드는 최대 12 step 뒤에 END로 끝난다. 액션 하나는 Command 하나로 변환하며 분할하지 않는다. 12기를 넘는 군대에 명령하려면 여러 액션을 낸다 |
| 031-R4 | **MUST** 무효 액션 마스크는 [034](034-action-table.md)의 정의를 따른다. 마스크는 BC, RL, 실전에서 동일하다. 모델 정책(RL·평가·실전)에서 executor가 액션을 거부하면 명령 없이 샘플된 delay만큼 진행하고, 거부 건수를 이유별로 세션 기록에 남긴다(2026-10-17, AMB-031-5). 에이전트 기억(030 §4 직전 행동)에는 거부 여부와 관계없이 샘플된 액션과 선택을 기록한다. BC 레이블 처리(041-R11)와 같은 규칙이다 |
| 031-R5 | ~~APM 제한~~ — **삭제(2026-10-15)**. APM 제한을 두지 않는다. SC:R 네트워크 턴당 커맨드 버퍼 한도는 기술적 제약이며 브리지가 큐잉으로 처리한다([012](012-bridge-remastered.md)) |
| 031-R6 | ~~자동 채취·자동화~~ — **삭제(2026-10-15)**. 새 SCV의 채취를 포함해 모든 유닛 지시는 정책이 직접 내린다 |
| 031-R7 | **MUST** executor는 결정론적이어야 한다. 같은 입력이면 같은 Command를 낸다. BC 레이블 역변환이 가능해야 하기 때문이다([040](040-replay-dataset.md)) |
| 031-R8 | **MUST** 타깃 스킬(`max_select = 1`)은 시전 유닛을 1기만 선택한다. BW에서는 여러 시전 유닛을 선택하면 전원이 시전해 에너지를 낭비한다 |
| 031-R9 | **MUST** `delay`는 이 액션 실행 후 다음 관측까지 진행할 프레임 수다. lockstep 모드에서는 정확히 그 프레임만큼 진행한다. realtime 모드에서는 그 프레임을 목표로 다음 관측을 요청한다 |
| 031-R10 | **MUST** 명령 실패(위치 무효, 대상 소멸 등)는 이벤트(`CommandFailed`, 010)로 기록한다. 다음 관측의 스칼라 feature(직전 액션 실패 여부, 030 §4)로 제공한다 |
| 031-R11 | **MUST** 32프레임을 넘는 무행동 구간은 `NoOp(delay=32)`을 연속으로 이어 표현한다. 마지막 남은 구간은 가장 가까운 하위 bucket의 NoOp으로 채운다. 리플레이 레이블 생성([040](040-replay-dataset.md))과 RL 실행 모두 같은 규칙을 쓴다 |

## 맵 사전계산 (모든 맵)
- 시작 위치별 본진·앞마당 베이스 위치, 초크 포인트, 확장 순서, 입구 벽 후보 타일.
- BC 데이터는 `uranium-replay build`가 리플레이를 재생할 때 맵마다 `<URANIUM_MAPS_DIR>/<CHK 해시>.toml`을 만든다(알고리즘은 `uranium-env` `mapgen::precompute`). 라이브 대상 맵의 자원 파일은 `uranium-replay mapgen`이 만든다(아래). (2026-10-19 정정: 이전 표기 `tools/mapgen`·`assets/maps/`는 실제와 달랐다)
- 대상 맵(투혼, 서킷브레이커)만이 아니라 BC 데이터에 나오는 **모든 맵**에 같은 맵 무관 알고리즘으로 적용한다. S1(전체 맵)과 S2(대상 맵)의 입력 분포를 같게 하기 위해서다 (AMB-031-3 Resolved, 2026-10-16).
- **관측 채널(`base_loc`, `choke`, `wall_candidate`)로만 제공한다. executor가 위치를 강제하거나 보정하지 않는다.**
- **라이브 게임의 사전계산 (확정 2026-10-17, AMB-031-4)**
  - BC shard의 사전계산 키는 리플레이 CHK의 SHA-256이고, 라이브 게임의 `GameInfo.map_hash`는 맵 파일(.scx)의 SHA-256이다(010). 키가 달라 그대로는 라이브 게임에서 사전계산을 찾지 못한다.
  - RL·평가 대상 맵 2개(`config/assets.toml`)만 지원한다. `uranium-replay mapgen <scx>...`가 맵 파일에서 CHK를 꺼내 자원 배치를 `<URANIUM_MAPS_DIR>/<맵 파일 SHA-256>.resources.toml`로 저장한다(오프라인, 맵마다 한 번).
  - 에이전트 세션은 게임 시작 시 `<맵 파일 SHA-256>.toml`(사전계산)이 있으면 그것을 쓰고, 없으면 `.resources.toml`과 `GameInfo.static_map`으로 BC와 **같은 알고리즘**(`mapgen::precompute`)을 돌려 만든다. 리플레이 경로와 입력(정적 맵 + CHK 자원)이 같으므로 결과도 같다.
  - 그 외 맵은 사전계산 없이(`base_loc`·`choke`·`wall_candidate` 채널 0) 진행하고 경고를 남긴다. 브리지·프로토콜은 바꾸지 않는다(001-R3).
  - 맵 계열(`map_family`, 040-R15)은 BC와 같은 `config/map_families.toml` 규칙(제목 패턴)으로 `GameInfo.map_name`에서 정한다. 판정 코드는 `uranium-env` 라이브러리에 둔다.

## Acceptance Criteria
- 031-AC1: [034](034-action-table.md)의 모든 action_type이 OpenBW에서 실제로 실행되는지 시나리오 테스트로 확인한다.
- 031-AC2: 마스크된 액션이 실행되지 않는다(property test).
- 031-AC3: 리플레이 사람 명령을 의미 단위 액션으로 역변환할 때 커버리지 ≥ 95%다. 미분류는 `Unknown`으로 기록하고 학습에서 제외한다.
- 031-AC4: Build 액션 실행 시 건설 위치가 정책이 고른 타일과 정확히 같다.

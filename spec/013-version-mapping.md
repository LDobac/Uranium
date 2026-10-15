# 013 — Version Mapping (1.16.1 ↔ Remastered)

- **Status**: Draft
- **Related**: [010](010-game-protocol.md), [011](011-bridge-openbw.md), [012](012-bridge-remastered.md)

## Purpose
OpenBW(1.16.1)와 SC:R 사이의 차이를 한곳에 모으고, 두 백엔드가 같은 의미의 RawFrame과 Command를 주고받도록 정규화 규칙을 정의한다.

## Scope & Non-goals
- 범위: ID 체계, 구조체 레이아웃, 명령 인코딩, 타이밍, 유닛 한도, 리플레이 포맷 차이.
- 비범위: 그래픽과 UI 차이(HD 그래픽 등). 게임 로직에 영향이 없으므로 다루지 않는다.

## 차이표 (초안, M1c/M7에서 검증)
| 항목 | 1.16.1 / OpenBW | SC:R | 정규화 |
|---|---|---|---|
| unit/order/tech/upgrade/weapon ID | BWAPI enum | dat 파일 기준 동일 ID 예상 | 정규 ID = BWAPI enum, 차이 발견 시 테이블 |
| 유닛 구조체 | BWAPI 추상화 | 64비트 메모리 레이아웃(포인터 8바이트) | 브리지 내부 `UnitView` 접근자, 오프셋 테이블 |
| 선택 명령 | 0x09/0x0A/0x0B (unit index 16bit) | 0x63/0x64/0x65 확장 | proto `Command`는 단일 형식, 브리지에서 인코딩 |
| 우클릭/타깃 명령 | 0x14/0x15 | 0x60/0x61 | 동일 |
| 유닛 한도 | 1700 | 1700 (확장 한도 옵션 확인 필요) | proto는 u32 ID 사용 |
| 프레임 속도 | Fastest 42ms | Fastest 42ms (멀티 latency 설정 차이) | 정규 시간 단위 = 프레임 |
| 명령 지연 | lockstep 즉시 반영 | 네트워크 turn latency (Low/High) | realtime 모드에서 latency 프레임만큼 delay 보정 |
| 게임 로직 | 1.16.1 | 1.16.1 기반 + 일부 버그 수정 | 차이 목록 유지, 학습 영향 평가 |
| 리플레이 | PKWare 압축 섹션 | zlib 압축 + 추가 섹션 (1.21+) | [040](040-replay-dataset.md) 파서가 양쪽 지원 |

## Requirements
| ID | 요구사항 |
|---|---|
| 013-R1 | **MUST** 버전별 매핑은 데이터 파일(`crates/uranium-adapter/tables/*.toml`)로 관리하고, 각 테이블에 버전 키를 둔다 |
| 013-R2 | **MUST** 정규 ID 집합의 각 원소는 두 백엔드 모두에서 왕복 변환된다(전단사). 불가능한 항목은 명시적으로 `Unsupported`로 처리한다 |
| 013-R3 | **MUST** realtime 모드에서는 명령 지연(turn latency)을 GameInfo에 포함해 에이전트에 알린다. delay 헤드 해석 시 이 값을 보정한다 |
| 013-R4 | **SHOULD** 같은 리플레이를 OpenBW와 SC:R에서 재생해 프레임 N의 RawFrame을 비교하는 골든 테스트를 둔다 |
| 013-R5 | **MUST** 게임 로직 차이(버그 수정 등)가 학습한 정책에 영향을 줄 수 있으면 목록에 기록하고 [050](050-evaluation.md)에서 평가한다 |

## Acceptance Criteria
- 013-AC1: 테란·프로토스·저그 유닛, 테크, 업그레이드 전체의 매핑 테이블 왕복 테스트를 통과한다.
- 013-AC2: 골든 테스트 10개 장면에서 유닛 수, 위치(±0px), HP, order가 일치한다.
- 013-AC3: SC:R 명령 인코딩을 SC:R 리플레이에 기록된 명령 바이트와 비교했을 때 일치한다.

## Risks & Open Questions
- 학습용 OpenBW 데이터 파일(`arr/*.dat`, iscript, tileset)은 SC:R CASC에서 추출한다(011-R13). 따라서 OpenBW 시뮬레이션은 SC:R 데이터 파일 기준이 되며, 1.16.1 원본과 다를 수 있다. 차이가 발견되면 이 문서에 기록한다.
- SC:R 패치로 생긴 밸런스나 로직 변경이 있는지 확인해야 한다. 알려진 차이는 대부분 버그 수정이며, 목록화가 필요하다.
- SC:R 리플레이를 OpenBW로 재생할 수 있는지 확인해야 한다. 가능하면 SC:R 리플레이도 BC 데이터로 쓸 수 있다([040](040-replay-dataset.md)).

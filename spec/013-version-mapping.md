# 013 — Version Mapping (1.16.1 ↔ Remastered)

- **Status**: Draft. 단, **"리플레이 명령 인코딩 (M4)" 절과 013-R6~R8은 Approved (2026-10-16, M4 계획 승인. 2026-10-17 개정: 맵 CHK 정규화, opcode 0x5D)**. 나머지는 M7에서 확정한다. 2026-10-19 표기 정정: 013-R1에 현재 테이블 위치를 적고 `uranium-adapter`를 M7 예정으로 표시.
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

## 리플레이 명령 인코딩 (M4, Approved 2026-10-16)
SC:R 리플레이를 OpenBW(1.16.1 로더, `third_party/openbw/replay.h`)로 재생하려면 컨테이너와 명령을 1.16.1 형식으로 바꿔야 한다(040-R10). 포맷 레퍼런스는 공개 파서 screp(icza/screp v1.13.8)다.

### 컨테이너
| 형식 | 판별 | 섹션 | 압축 |
|---|---|---|---|
| Legacy (1.16.1) | 오프셋 12 = `reRS`, 오프셋 28 ≠ `0x78` | ReplayID(4) · Header(0x279) · Commands · MapData | PKWare implode |
| Modern (1.18–1.20) | 오프셋 12 = `reRS`, 오프셋 28 = `0x78` | Legacy 4개 + PlayerNames(0x300) + 확장 섹션 | zlib(청크별, 비압축 청크 허용) |
| Modern 1.21+ | 오프셋 12 = `seRS` | 위와 같음. ReplayID 다음에 4바이트 길이 필드가 하나 더 있다 | zlib |

- 섹션 구조는 `crc32, 청크 수, (청크 길이, 데이터)*`이고, 청크 출력은 최대 8,192바이트다.
- 변환본은 ReplayID를 `reRS`로 쓰고 4개 섹션(Header·Commands·MapData 길이 접두 포함)만 남긴다. 청크는 **비압축**(입력 길이 = 출력 길이)으로 쓰고, crc32는 원본 데이터로 다시 계산한다. OpenBW 로더는 비압축 청크를 그대로 복사하므로 implode 압축기가 필요 없다.
- PlayerNames와 확장 섹션(SKIN·LMTS·BFIX·CCLR·GCFG·서드파티)은 버린다. 게임 진행에 영향이 없다.
- **맵 CHK 정규화** (2026-10-17 사용자 확인, `reports/m4_scr_desync.md`): OpenBW 맵 로더는 CHK `VER` 59·63·205만 받고 `STR `을 필수로 읽는다. SC:R 맵 형식 VER 64·206은 각각 63·205에 확장 문자열 표 `STRx`(u32 개수·u32 오프셋)를 쓰는 형식이다. 변환본의 맵 섹션에서
  - `VER` 64 → 63, 206 → 205로 바꾼다.
  - `STRx`를 `STR `(u16 개수·u16 오프셋)으로 다시 쓴다. 문자열 내용과 순서는 그대로 두고, 범위를 벗어난 오프셋은 빈 문자열로 둔다. 개수나 크기가 u16을 넘으면 변환 불가(`Unsupported`)다. `STRx`가 있으면 기존 `STR `은 버린다.
  - 그 밖의 섹션과, 길이가 잘못된 뒤쪽 바이트(맵 보호 기법)는 그대로 복사한다.
  - 맵 해시(`GameMeta.map_hash`)와 맵 패밀리 판정은 원본 CHK 기준이다.

### 명령 (SC:R 1.21 확장 → 1.16.1)
| SC:R opcode | 1.16.1 opcode | SC:R 필드 | 변환 |
|---|---|---|---|
| `0x63` Select121 | `0x09` Select | count(u8), (tag u16, pad u16)×count | pad를 태그 상위 16비트로 합쳐 아래 태그 규칙으로 변환 |
| `0x64` SelectAdd121 | `0x0A` SelectAdd | 위와 같음 | pad를 태그 상위 16비트로 합쳐 아래 태그 규칙으로 변환 |
| `0x65` SelectRemove121 | `0x0B` SelectRemove | 위와 같음 | pad를 태그 상위 16비트로 합쳐 아래 태그 규칙으로 변환 |
| `0x60` RightClick121 | `0x14` RightClick | x u16, y u16, tag u16, pad u16, unit u16, queued u8 | pad를 태그 상위 16비트로 합쳐 아래 태그 규칙으로 변환 |
| `0x61` TargetedOrder121 | `0x15` TargetedOrder | x u16, y u16, tag u16, pad u16, unit u16, order u8, queued u8 | pad를 태그 상위 16비트로 합쳐 아래 태그 규칙으로 변환 |
| `0x62` Unload121 | `0x29` Unload | tag u16, pad u16 | pad를 태그 상위 16비트로 합쳐 아래 태그 규칙으로 변환 |
| 그 밖의 opcode | 같음 | 1.16.1과 같은 배치 | 그대로 복사 |

- **유닛 태그 규칙** (M4 스파이크 실측, 2026-10-16. 1.21 형식의 pad는 screp 설명과 달리 항상 0이 아니며 재사용 카운터의 상위 비트다):
  - SC:R 1.21 태그(32비트) = `tag u16 | pad u16 << 16`. `0`과 `0xFFFF`는 "유닛 없음"이다.
  - 유닛 한도가 1,700(LMTS 섹션 없음 포함)이면 `인덱스 = 하위 11비트`, `재사용 = 나머지`다. 유닛 한도가 3,400 등 확장(`LMTS` 섹션 0x0C의 units 값)이면 `인덱스 = 하위 13비트`, `재사용 = 나머지`다.
  - 1.16.1 태그 = `인덱스' | (재사용 & 0x1F) << 11`. 확장 한도에서는 플레이어 유닛이 배열 위쪽부터 배정되므로 `인덱스 ≥ 한도 - 1700`이면 `인덱스' = 인덱스 - (한도 - 1700)`이다. 맵 배치 유닛(미네랄 등)은 아래쪽 인덱스라 그대로 둔다. 인덱스는 1부터 시작한다.
  - 변환 결과 인덱스가 1,700을 넘으면 그 리플레이를 `Unsupported`로 폐기한다(013-R2).
  - 1.16.1 태그는 BWAPI `getReplayID()`와 같다(011-R5).
- **플레이어 번호** (M4 스파이크 실측): 명령의 플레이어 바이트는 헤더 플레이어 구조체의 `ID` 필드다. BWAPI 플레이어 번호는 이름이 있는(점유된) 헤더 칸을 배열 순서대로 0, 1, 2…로 다시 매긴 값이며, 배열 위치나 `slot`·`ID` 필드와 다를 수 있다(BWAPI GameInfo와 이름 대조로 확인). 재생 시점(`URANIUM_REPLAY_PERSPECTIVE`, hostd `ReplaySpec.perspective_slot`)은 이 BWAPI 번호로 넘긴다.
- **opcode 처리 규칙** (2026-10-16 사용자 지시: 알 수 없는 opcode를 단순 폐기하지 않는다). OpenBW 로더(`actions.h` `read_action`)가 처리하지 않는 명령은 재생 전체를 멈추므로 변환 시 다음처럼 나눈다.
  - OpenBW가 처리하는 opcode: 그대로 둔다.
  - OpenBW가 처리하지 않고 게임 상태에 영향이 없는 opcode(게임 속도 0x0F, 일시정지·재개 0x10·0x11, 지연 0x55, keep-alive·sync·로비 0x05–0x08·0x37–0x45·0x48·0x54·0x56·0x5B, SC:R `0x5D`): 제거한다. `0x5D`는 screp에도 정의가 없지만 래더 리플레이 2,000개에서 페이로드 4바이트로 읽으면 모든 블록이 정확히 끝나고 값이 항상 0이며 시작 프레임에만 나온다(2026-10-17 사용자 확인). 영향이 있었다면 040-R8 디싱크 검사가 걸러낸다.
  - 게임 상태에 영향이 있는데 OpenBW가 처리하지 않는 opcode(예: 0x1D)와 길이를 모르는 새 opcode: 리플레이를 버리지 않고 "확인 대기"로 분류해 opcode·프레임을 리포트한다. 정체(레퍼런스 파서·OpenBW 소스)를 확인해 OpenBW 실행에 문제가 없으면 파서와 이 표에 추가하고 다시 처리한다.
- 헤더 플레이어 `ID`에 없는 플레이어 바이트(옵저버, 128 이상)의 명령은 변환 시 제거한다. OpenBW 로더가 "player id not found"로 재생을 멈추며, 이 명령들은 게임 상태에 영향이 없다(M4 스파이크 실측).

## Requirements
| ID | 요구사항 |
|---|---|
| 013-R6 | **MUST** SC:R 리플레이(Modern, Modern 1.21+)는 위 "리플레이 명령 인코딩" 표에 따라 1.16.1 컨테이너와 명령으로 변환한 뒤 OpenBW로 재생한다(040-R10). 변환은 `uranium-replay`의 Rust 코드로만 하고 OpenBW는 수정하지 않는다 |
| 013-R7 | **MUST** 변환은 손실 없이 왕복 검증한다. 변환본을 다시 파싱한 명령 목록(프레임, 플레이어, opcode 의미, 인자)이 원본 파싱 결과와 같아야 한다 |
| 013-R8 | **MUST** 변환 불가(`Unsupported`)·디싱크로 폐기한 리플레이 수와 사유를 리포트한다(040-AC5) |
| 013-R1 | **MUST** 버전별 매핑은 데이터 파일로 관리하고, 각 테이블에 버전 키를 둔다. 현재(M4~M6) 테이블은 `crates/uranium-env/tables/*.toml`과 `crates/uranium-replay/tables/*.toml`에 있다. SC:R 매핑 테이블은 `crates/uranium-adapter/tables/*.toml`(M7 예정, 미생성)에 둔다 |
| 013-R2 | **MUST** 정규 ID 집합의 각 원소는 두 백엔드 모두에서 왕복 변환된다(전단사). 불가능한 항목은 명시적으로 `Unsupported`로 처리한다 |
| 013-R3 | **MUST** realtime 모드에서는 명령 지연(turn latency)을 GameInfo에 포함해 에이전트에 알린다. delay 헤드 해석 시 이 값을 보정한다 |
| 013-R4 | **SHOULD** 같은 리플레이를 OpenBW와 SC:R에서 재생해 프레임 N의 RawFrame을 비교하는 골든 테스트를 둔다 |
| 013-R5 | **MUST** 게임 로직 차이(버그 수정 등)가 학습한 정책에 영향을 줄 수 있으면 목록에 기록하고 [050](050-evaluation.md)에서 평가한다 |

## Acceptance Criteria
- 013-AC1: 테란·프로토스·저그 유닛, 테크, 업그레이드 전체의 매핑 테이블 왕복 테스트를 통과한다.
- 013-AC2: 골든 테스트 10개 장면에서 유닛 수, 위치(±0px), HP, order가 일치한다.
- 013-AC3: SC:R 명령 인코딩을 SC:R 리플레이에 기록된 명령 바이트와 비교했을 때 일치한다.
- 013-AC4 (M4): SC:R 리플레이 100개 이상에서 변환 왕복(013-R7)이 통과하고, 변환본이 OpenBW 로더에서 오류 없이 열린다.

## Risks & Open Questions
- 학습용 OpenBW 데이터 파일(`arr/*.dat`, iscript, tileset)은 SC:R CASC에서 추출한다(011-R13). 따라서 OpenBW 시뮬레이션은 SC:R 데이터 파일 기준이 되며, 1.16.1 원본과 다를 수 있다. 차이가 발견되면 이 문서에 기록한다.
- SC:R 패치로 생긴 밸런스나 로직 변경이 있는지 확인해야 한다. 알려진 차이는 대부분 버그 수정이며, 목록화가 필요하다.
- SC:R 리플레이를 OpenBW로 재생할 수 있는지 확인해야 한다. 가능하면 SC:R 리플레이도 BC 데이터로 쓸 수 있다([040](040-replay-dataset.md)).

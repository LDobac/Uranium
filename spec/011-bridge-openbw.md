# 011 — Bridge: OpenBW

- **Status**: Approved (2026-10-15, 2026-10-16 개정: 011-R13 내용 해시, 011-R15 첫 관측 프레임 1, 011-AC6 원격 판정 기준, 011-AC8 통과)
- **Related**: [010](010-game-protocol.md), [013](013-version-mapping.md), [020](020-env-orchestration.md)

## Purpose
OpenBW 포크 BWAPI 4.2(`github.com/OpenBW/bwapi`) 위에서 동작하는 AIModule `.so` 브리지를 정의한다. OpenBW 게임 상태를 RawFrame으로 직렬화하고, Command를 BWAPI 호출로 실행한다.

## Scope & Non-goals
- 범위: Linux 헤드리스 실행, lockstep 스텝, 1v1 LAN 매칭, 리플레이 저장, **리플레이 재생 모드**(BC 데이터 생성, [040](040-replay-dataset.md)).
- 비범위: GUI 디버깅. GUI 디버깅은 `OPENBW_ENABLE_UI=1` 빌드로 선택적으로만 지원한다. (변경 이력: 2026-10-15 'Windows 실행'을 비범위에서 제외, 011-R16)

## 구조
- `bridge/openbw/shim.cpp`: `BWAPI::AIModule` 구현. `onStart/onFrame/onEnd/onUnit*` 이벤트를 C ABI로 Rust에 전달한다. 약 500줄 이내로 유지한다.
- `bridge/openbw/src/lib.rs` (`staticlib`): BWAPI 상태 덤프 구조체를 받아 RawFrame으로 변환하고, TCP 송수신과 Command 디코딩을 한다.
- 빌드는 CMake로 하며 shim과 Rust staticlib을 링크해 `liburanium_bridge.so`를 만든다.

## Requirements
| ID | 요구사항 |
|---|---|
| 011-R1 | **MUST** `BWAPI_CONFIG_AI__AI=<path>/liburanium_bridge.so`로 로드되어야 하고, 접속 대상 에이전트 주소는 환경변수 `URANIUM_AGENT_ADDR`(`host:port`)로 받는다. 에이전트는 다른 PC일 수도 있고 **같은 PC(`127.0.0.1`/`localhost`)** 일 수도 있다. 두 경우 모두 같은 TCP 경로를 쓰며 코드 분기는 없다(001-R2) |
| 011-R2 | **MUST** 헤드리스(`OPENBW_ENABLE_UI=0`)로 동작한다. 게임 속도는 `setLocalSpeed(0)`, `setFrameSkip` 기본 0이다 |
| 011-R3 | **MUST** lockstep: `onFrame`에서 다음 관측 프레임이면 RawFrame을 보내고 `Act`가 올 때까지 블로킹한다. 그 사이 프레임은 즉시 반환한다 |
| 011-R4 | **MUST** 1v1 매칭은 `auto_menu=LAN`, `OPENBW_LAN_MODE=TCP` 또는 `LOCAL`로 한다. 두 런처는 같은 맵과 같은 `bwapi.ini`를 써야 한다 |
| 011-R5 | **MUST** 정책 관측(RawFrame)은 항상 **자기 시야로 필터링**한다(맵핵 금지). OpenBW 브리지(lockstep)는 `CompleteMapInformation`을 **항상 켠다**. 이는 결정론적 유닛 ID(`getReplayID`, 리플레이 명령의 유닛 ID와 동일)를 얻기 위해서다. RawFrame 생성 시에는 `isVisible(self)`와 탐지 여부로 적 유닛·이벤트를 직접 거른다. `PrivilegedSummary`는 `URANIUM_PRIVILEGED=1`일 때만 송신한다. (2026-10-15 사용자 승인으로 변경. 이전 문구는 '학습용 상대 정보 생성 시에만 켠다') |
| 011-R16 | **MUST** Windows에서도 동작한다. 브리지는 MSVC로 빌드한 `.dll` AI 모듈이다. OpenBW는 Windows에서 `LOCAL` 소켓을 지원하지 않으므로 LAN 매칭은 `OPENBW_LAN_MODE=TCP`로 하고 게임마다 포트 2개(각 클라이언트 listen 1개)를 쓴다. 포트는 hostd가 할당한다(020-R8) |
| 011-R15 | **MUST** 유닛 ID는 `getReplayID()`를 쓴다. `getID()`는 프로세스마다 다른 포인터 순서로 배정되어 재현이 불가능하다. 한 관측 안의 유닛·이펙트·이벤트는 결정론적 순서로 정렬한다. `CompleteMapInformation`은 `onStart`에서 켜지므로 프레임 0에서는 모든 유닛의 replay id가 0이다. 따라서 **첫 관측은 프레임 1**(replay id가 배정된 뒤)에 보낸다(2026-10-16 사용자 확인) |
| 011-R6 | **MUST** 정규 ID는 BWAPI enum 값과 같다. OpenBW 브리지의 ID 변환은 항등 매핑이다 |
| 011-R7 | **MUST** 게임 종료 시 리플레이(`.rep`)를 저장하고 경로를 GameEnd에 포함한다. 저장 여부는 설정으로 끌 수 있다 |
| 011-R8 | **MUST** 에이전트 연결이 끊기면 `leaveGame()`을 호출하고 비정상 종료 코드로 끝낸다 |
| 011-R9 | **SHOULD** 시작 위치를 고정 또는 랜덤으로 고를 수 있게 한다(`BWAPI_CONFIG_AUTO_MENU__MAP` + 맵 시드). 이 기능은 평가 재현용이다 |
| 011-R10 | **MUST** **리플레이 재생 모드**: BWAPI 리플레이 모드로 `.rep`를 재생하고, 환경변수 `URANIUM_REPLAY_PERSPECTIVE=<player_slot>`으로 지정한 플레이어(테란) **시점의 시야로 필터링**한 RawFrame을 만든다(030-R3). 상대(PrivilegedSummary·승자 판정 대상)는 `URANIUM_REPLAY_ENEMY=<player_slot>`으로 지정하고, 이때 GameInfo 플레이어 목록은 시점 플레이어와 상대 두 명만 담는다(활동 없는 빈 칸 제외, 실전 1:1과 같은 형태). 미지정이면 유닛을 가진 다른 플레이어 중 id가 가장 낮은 쪽을 상대로 한다. GameInfo 플레이어는 모든 모드에서 id 순이다(결정론, 2026-10-17 위임 결정) |
| 011-R11 | **MUST** 재생 모드의 `Hello.mode`는 `Lockstep`이고 별도 플래그 `replay=true`를 쓴다. 에이전트는 `cmds`가 비어 있는 `Act{advance_frames}`만 보낸다. 명령이 들어 있으면 브리지가 에러로 거부한다. 행동 레이블은 브리지가 아니라 [040](040-replay-dataset.md) 리플레이 파서가 만든다 |
| 011-R13 | **MUST** 게임 데이터는 MPQ 3종 또는 **SC:R CASC에서 추출한 풀린 파일 폴더**(`URANIUM_DATA_DIR`) 중 하나로 공급한다. 폴더 방식은 OpenBW 패치 `third_party/patches/0001-openbw-loose-data-files.patch`로 지원한다. 추출 도구는 `tools/casc-extract`다(2026-10-15). 추출물 검증 값은 **경로에 무관한 내용 해시**(`content_sha256`)다. 파일마다 `<상대 경로(/ 구분)>\t<크기>\t<sha256>\n` 줄을 만들어 경로 순으로 이어 붙이고, 그 문자열의 SHA-256을 낸다. `config/assets.toml`은 이 값을 고정한다. `manifest.toml` 파일 자체의 해시는 추출 원본의 절대 경로가 들어가 호스트마다 달라지므로 쓰지 않는다(2026-10-16 변경) |
| 011-R14 | **MUST** 서드파티 소스(OpenBW, BWAPI)는 커밋을 고정해 `tools/openbw/build.sh`가 받는다. 우리 수정은 `third_party/patches/NNNN-<repo>-<desc>.patch`로만 관리하고 빌드 시 적용한다 |
| 011-R12 | **SHOULD** 재생 모드에서도 `PrivilegedSummary`(상대 실제 상태)를 생성할 수 있게 한다. BC 단계의 value head 사전학습 입력으로 쓴다 |

## 환경변수 (2026-10-15 정리)
| 변수 | 의미 | 기본값 |
|---|---|---|
| `URANIUM_AGENT_ADDR` | 에이전트 서버 `host:port` (011-R1) | 필수 |
| `URANIUM_GAME_ID` | `Hello.game_id` (010-R9) | 0 |
| `URANIUM_SEED` | 설정 시 `onStart`에서 `setRandomSeed` (011-AC4) | 미설정 |
| `URANIUM_MAX_FRAMES` | 도달 시 `TimeLimitDraw`로 종료 (020) | 86400 |
| `URANIUM_PRIVILEGED` | `1`이면 lockstep 학습 모드에서 `PrivilegedSummary` 송신 (011-R5) | 0 |
| `URANIUM_DATA_DIR` | 추출 데이터 폴더 (011-R13) | 미설정 |
| `URANIUM_REPLAY_PERSPECTIVE` | 재생 모드 시점 (011-R10) | – |
| `URANIUM_REPLAY_ENEMY` | 재생 모드 상대 (011-R10) | 유닛을 가진 다른 플레이어 중 최소 id |

- 구현 단계: 재생 모드(R10–R12, AC5)는 **M4**에서 구현한다. M2는 대전 모드만 구현한다.
- 재생 모드 실행(2026-10-16): hostd가 `rep_bytes`를 게임 작업 디렉터리에 `.rep`로 쓰고, 런처 1개를 `BWAPI_CONFIG_AUTO_MENU__MAP=<그 파일>`로 띄운다. SC:R 리플레이는 013 변환본을 넘긴다. 시야 필터 시점은 `URANIUM_REPLAY_PERSPECTIVE` 슬롯이다.
- 유닛 참조(2026-10-18, 030-R1·010-R5 버그 수정): `UnitState.order_target`·`loaded_in`은 **같은 관측의 유닛 목록에 있는 유닛만** 가리키고, 그 밖이면 `None`이다. BWAPI는 죽었거나 추적하지 않는 유닛을 가리키는 참조에 초기화되지 않은 replayID·`exists()`를 남겨 같은 리플레이 재생마다 값이 달랐다(040-R8 "replays differ" 오판, `reports/m4_desync_nondet.md`). 시야 밖 유닛 id 노출도 함께 막힌다.

## Acceptance Criteria
- 011-AC1: 투혼과 서킷브레이커 각각에서 스크립트 에이전트 vs 스크립트 에이전트 100판을 크래시 없이 완주한다.
- 011-AC2: 프레임당 브리지 오버헤드는 RawFrame 생성 기준 200 유닛에서 ≤ 0.3ms다.
- 011-AC3: 단일 게임의 lockstep 처리량은 에이전트 응답 시간을 제외하고 ≥ 5,000 frames/s다. 실측값은 M1(b)에서 기록한다.
- 011-AC4: 같은 seed와 같은 Act 시퀀스로 두 번 실행했을 때 GameEnd 프레임과 승자가 같다.
- 011-AC5: 재생 모드에서 같은 리플레이를 두 번 재생했을 때 RawFrame 해시 시퀀스가 같다(040-R8). 지정 시점 밖의 적 유닛이 RawFrame에 없다.
- 011-AC6: 에이전트를 같은 PC(`127.0.0.1`)에 둔 구성과 원격 PC에 둔 구성 모두에서 게임이 크래시 없이 완주한다. 같은 PC 구성은 011-AC1로, 원격 구성은 020-AC1 실행(원격 게임 호스트 → 학습 PC 에이전트)으로 판정한다(2026-10-16 사용자 결정으로 변경. 이전: 두 구성 모두 011-AC1 시나리오)
- 011-AC8 (Windows 세션 검증): Windows에서 011-AC1(두 맵 각 100판 완주)과 011-AC4(seed 결정론)를 재현한다. — **통과 (2026-10-16, `reports/m3_windows.md`)**
- 011-AC7 (M2 검증, 브리지 필요): 에이전트가 응답을 최대 1초 지연해도 LAN 상대 클라이언트가 강제 퇴장(드롭)되지 않는다. 퇴장되면 대응 방안을 이 spec에 추가하고 사용자 확인을 받는다.

## Risks & Open Questions
- **LAN 동기화:** lockstep으로 우리 브리지가 멈추면 LAN 상대 클라이언트도 대기한다. OpenBW에 지연 타임아웃(드롭)이 있는지 M1(b)에서 확인한다(011-AC7).
- OpenBW 포크는 BWAPI 4.2다. BWAPI 4.4 대상 오픈소스 봇과는 바이너리 호환되지 않는다. 대응은 [021](021-opponent-pool.md)에서 다룬다.
- 맵 파일: 투혼과 서킷브레이커의 1.16.1 호환 `.scx`/`.scm` 버전을 확보하고 해시를 고정해야 한다. MPQ 3종(Stardat, Broodat, Patch_rt)은 사용자가 직접 제공한다.

- **MPQ·맵 확보 (확정 2026-10-15, AMB-011-1)**: 두 경로를 병행한다.
  - (a) 사용자가 구버전(1.16.1) 또는 2017년 무료 배포 1.18 설치본을 확보한다.
  - (b) Claude가 M1 사전 작업으로 SC:R 설치본(CASC)에서 OpenBW에 필요한 데이터를 추출·변환할 수 있는지 조사한다. 결과는 `reports/mpq_extraction.md`에 기록한다.
  - 먼저 성공한 경로를 쓰고, 파일 해시를 `config/assets.toml`에 고정한다.
  - 맵 파일은 AMB-000-3(최신 공개 버전)을 따르며, 1.16.1 CHK 호환성을 M1(b)에서 검증한다.

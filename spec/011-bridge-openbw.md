# 011 — Bridge: OpenBW

- **Status**: Approved (2026-10-15, 2026-10-16 개정: 011-R13 내용 해시, 011-R15 첫 관측 프레임 1, 011-AC6 원격 판정 기준, 011-AC8 통과, 2026-10-18 개정: 011-R17 퇴장 사유 구분·011-R18 드롭 시간(011-AC7 대응, 사용자 결정). 2026-10-19 개정: 011-R19 GameEnd 최종 관측(010-R12, 사용자 결정). 2026-10-19 액션 감사 개정: 011-R20 명령 실행(사용자 결정 2026-10-19, 액션 감사 F2·F3·F17). 2026-10-21 개정: 011-R21 항복 실행(사용자 결정, AMB-031-6). 2026-10-29 개정: 011-R22 패배한 원격 클라이언트의 남은 명령, 011-AC10(사용자 결정))
- **Related**: [010](010-game-protocol.md), [013](013-version-mapping.md), [020](020-env-orchestration.md)

## Purpose
OpenBW 포크 BWAPI 4.2(`github.com/OpenBW/bwapi`) 위에서 동작하는 AIModule `.so` 브리지를 정의한다. OpenBW 게임 상태를 RawFrame으로 직렬화하고, Command를 BWAPI 호출로 실행한다.

## Scope & Non-goals
- 범위: Linux 헤드리스 실행, lockstep 스텝, 1v1 LAN 매칭, 리플레이 저장, **리플레이 재생 모드**(BC 데이터 생성, [040](040-replay-dataset.md)).
- 비범위: GUI 디버깅. GUI 디버깅은 `OPENBW_ENABLE_UI=1` 빌드로 선택적으로만 지원한다. (변경 이력: 2026-10-15 'Windows 실행'을 비범위에서 제외, 011-R16)

## 구조
- `bridge/openbw/shim.cpp`: `BWAPI::AIModule` 구현. `onStart/onFrame/onEnd/onUnit*` 이벤트를 C ABI로 Rust에 전달한다. 약 500줄 이내로 유지한다(2026-10-19 011-R20 추가 뒤 약 570줄).
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
| 011-R17 | **MUST** GameEnd 사유: 상대가 섬멸되지 않고 게임을 떠나면(자발적 퇴장 또는 OpenBW 드롭) `Surrender`, 섬멸이면 `Elimination`으로 보고한다. 판정은 `onPlayerLeft(상대)` 시점에 직전 프레임까지 보던 상대 건물이 아직 존재하는지로 한다. 근접전 패배는 건물이 모두 파괴돼야 하고, 퇴장하면 건물이 중립으로 남기 때문이다(BWAPI는 섬멸된 플레이어에도 `onPlayerLeft`를 보내므로 이벤트만으로는 구분할 수 없다). 시간 제한은 기존대로 `TimeLimitDraw`다. 두 세션이 서로 상대 퇴장으로 승리를 보고한 게임의 처리는 020-R13 (2026-10-18 사용자 결정) |
| 011-R18 | **MUST** OpenBW 드롭 시간: 동기화가 끊긴 LAN 클라이언트를 끊는 OpenBW 타임아웃(원본 60초 고정)을 환경변수 `URANIUM_OPENBW_DROP_TIMEOUT_MS`로 바꿀 수 있게 한다(패치 `0013-openbw-drop-timeout.patch`). 미설정이면 원본과 같은 60초다. hostd는 모든 런처에 이 값을 넘기며 기본값은 600,000(10분)이다(020, `.env`로 변경). lockstep 학습은 에이전트 응답을 기다리며 블로킹하므로 학습 호스트 멈춤이 60초를 넘으면 두 클라이언트가 서로를 끊어 버린다(2026-10-18 관측, 011-AC7) (2026-10-18 사용자 결정) |
| 011-R19 | **MUST** 최종 관측(010-R12, 사용자 결정 2026-10-19, E-8): shim의 `onEnd`는 종료 프레임의 자기 시야 상태와 마지막 step 뒤 이벤트로 `UrFrame`을 만들어 `uranium_bridge_end(winner, reason, frame, last)`로 넘긴다. 실패 상태·리플레이 모드에서는 NULL이다 |
| 011-R20 | **MUST** 명령 실행(사용자 결정 2026-10-19, `reports/m6_action_audit.md` F2·F3·F17). ① **대상 해석**: `Command`의 유닛 대상 id는 이번 관측에 넣은 유닛에서 찾고, 없으면 맵의 중립 미네랄 필드·베스핀 간헐천 전체(시야 밖 포함)에서 찾는다. 034의 "시야 밖이지만 한 번 본 자원" 대상을 실행하기 위한 것이며, 맵에서 공개된 자원뿐이므로 010-R5 예외 범위 안이다. 적·다른 플레이어 유닛은 관측 유닛에서만 찾는다(맵핵 금지). ①-1 **사라진 자원(메인 기술 결정 2026-10-19, 맵핵 금지)**: 대상 id가 한 번 존재했던 맵 자원인데 지금은 없으면(채굴 완료, 정제소가 덮은 간헐천) Move·RightClick·Gather는 그 자원의 마지막 위치로의 이동으로, SetRally는 그 위치로의 랠리로 실행한다. 스타1에서 안개 속 잔상을 클릭하면 그 자리로 가는 것과 같고, `CommandFailed`가 "시야 밖 자원이 없어졌다"는 정보를 알려 주는 부채널을 막는다. ② **즉시 실패**: 그 밖에 대상 id를 찾지 못하거나, 위치가 필요한 명령에 위치가 없으면(`UR_NONE`) 게임에 아무것도 보내지 않고 `CommandFailed`(Unit_Does_Not_Exist 또는 Invalid_Parameter)를 낸다. 이전에는 Move·RightClick이 (-1,-1) 이동으로 떨어져 실패 이벤트 없이 무시됐다. ③ **그룹 명령**: `Command::Unit`은 선택 유닛 전체에 한 번의 명령으로 보낸다. 같은 명령을 받는 유닛은 BWAPI `Game::issueCommand`(선택 1회 + 명령 1회)로 함께 보내 엔진의 그룹 이동(`calc_group_move`)이 사람 명령과 같게 적용된다. 그룹으로 명령할 수 없는 유닛(떠 있는 건물 등)은 따로 보낸다. 유닛마다 명령이 달라지는 보정 — 비무장 유닛 Attack_Pos → move(AMB-041-1), 적을 공격할 수 없는 유닛의 적 우클릭 → follow, 메딕은 대상 위치로 어택땅(게임은 각각 Follow·HealMove로 바꾸지만 BWAPI는 우클릭을 거절, F17) — 은 명령별로 묶어 보낸다. ④ **실패 판정**: 선택 중 1기 이상이 명령을 받으면 성공이고, 받은 유닛이 없을 때만 `CommandFailed`(첫 거절 유닛의 BWAPI 오류)를 낸다(게임도 명령을 받을 수 없는 선택 유닛만 건너뛴다, 041-R11). 유닛별 거절 수는 `CommandFailed`에 넣지 않는다(010 프로토콜 변경은 별도 결정). BWAPI `Unitset`은 포인터 해시 순서로 돌기 때문에 게임이 보는 선택 순서는 풀 순서와 다를 수 있으나, OpenBW의 그룹 처리(`action_order`, `calc_group_move`)는 선택 순서에 의존하지 않는다(011-AC4 결정론 재확인: 같은 seed·같은 Act 10분 게임 2회, 관측 1,800개 체크섬·GameEnd 동일, 2026-10-19 `same_seed_same_acts_same_game_011_ac4`) |
| 011-R12 | **SHOULD** 재생 모드에서도 `PrivilegedSummary`(상대 실제 상태)를 생성할 수 있게 한다. BC 단계의 value head 사전학습 입력으로 쓴다 |
| 011-R21 | **MUST** 항복 실행(2026-10-21 사용자 결정, AMB-031-6, 010-R13). shim은 `surrender`가 있는 `Act`를 받으면 ① 문자열이 비어 있지 않으면 `Broodwar->sendText(text)`(OpenBW용 BWAPI 포크에서 OpenBW는 `NetMode`가 항상 0이라 LAN 게임도 싱글 플레이 분기로 가서 채팅이 로컬 출력만 되고, 멀티플레이 분기는 비어 있다(`fixme`). 패치 `0014-bwapi-multiplayer-chat.patch`가 그 출력과 함께 게임 채팅 액션 0x5C(보낸 슬롯 + 80바이트)를 보내 상대 클라이언트와 저장 리플레이에 남긴다), ② 채팅이 네트워크로 나가도록 24프레임(약 1초) 동안 관측을 보내지 않고 진행, ③ `leaveGame()`. 떠난 쪽 세션의 `GameEnd`는 winner = 상대, reason = `Surrender`(자기 퇴장 플래그)로 보고한다. 남은 쪽은 기존 011-R17대로 `Surrender`와 자기 승리를 보고하므로 두 보고가 일치하고 020-R13에 따라 일반 패배로 집계된다. `final_frame`(011-R19)은 퇴장 직전 프레임 기준이다 |
| 011-R22 | **MUST** 패배한 원격 클라이언트의 남은 명령(2026-10-29 사용자 결정, `reports/m6_bc_eval.md` 3절). OpenBW `sync.h`는 원격 클라이언트의 플레이어가 점유 상태가 아닌데(패배·퇴장) 그 클라이언트의 예약 명령이 실행되면 연결을 끊는다(`kill_client`). 진 쪽이 섬멸 직전에 보낸 명령(지연 2~3프레임)이 남아 있으면, 앞서 가는 이긴 쪽 프로세스가 진 쪽을 끊는다. 그러면 진 쪽 프로세스는 이를 상대 드롭(퇴장 사유 6)으로 처리해, 자기 패배를 끝내기 전에 승리(Surrender)나 승자 없음을 보고하고 리플레이에도 가짜 드롭을 남긴다. 패치 `0015-openbw-keep-defeated-client.patch`는 이 경우 로컬 클라이언트와 같이 그 클라이언트의 예약 명령만 버리고 연결은 유지한다. 스스로 나간 플레이어(011-R21 항복, T1 봇 GG, `max_frames` 퇴장)도 같은 분기를 탄다: 남은 쪽은 퇴장 명령(87)을 실행하고 연결은 끊지 않는다. 연결은 상대 프로세스가 끝날 때(BWAPILauncher 종료, 0003) 또는 동기화가 없는 채로 드롭 시간(011-R18)이 지날 때 끊긴다. 그동안 남은 쪽은 나간 쪽의 마지막 프레임까지만 진행한다(lockstep, 패치 전 정상 판과 같은 대기) |

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
| `URANIUM_OPENBW_DROP_TIMEOUT_MS` | 동기화 끊긴 클라이언트 드롭 시간, OpenBW 패치 0013 (011-R18) | 60000 (hostd는 600000을 넘김) |

- 구현 단계: 재생 모드(R10–R12, AC5)는 **M4**에서 구현한다. M2는 대전 모드만 구현한다.
- 재생 모드 실행(2026-10-16): hostd가 `rep_bytes`를 게임 작업 디렉터리에 `.rep`로 쓰고, 런처 1개를 `BWAPI_CONFIG_AUTO_MENU__MAP=<그 파일>`로 띄운다. SC:R 리플레이는 013 변환본을 넘긴다. 시야 필터 시점은 `URANIUM_REPLAY_PERSPECTIVE` 슬롯이다.
- 유닛 참조(2026-10-18, 030-R1·010-R5 버그 수정): `UnitState.order_target`·`loaded_in`은 **같은 관측의 유닛 목록에 있는 유닛만** 가리키고, 그 밖이면 `None`이다. BWAPI는 죽었거나 추적하지 않는 유닛을 가리키는 참조에 초기화되지 않은 replayID·`exists()`를 남겨 같은 리플레이 재생마다 값이 달랐다(040-R8 "replays differ" 오판, `reports/m4_desync_nondet.md`). 시야 밖 유닛 id 노출도 함께 막힌다.

## Acceptance Criteria
- 011-AC1: 투혼과 서킷브레이커 각각에서 스크립트 에이전트 vs 스크립트 에이전트 100판을 크래시 없이 완주한다.
- 011-AC2: 프레임당 브리지 오버헤드는 RawFrame 생성 기준 200 유닛에서 ≤ 0.3ms다.
- 011-AC3: 단일 게임의 lockstep 처리량은 에이전트 응답 시간을 제외하고 ≥ 5,000 frames/s다. 실측값은 M1(b)에서 기록한다.
- 011-AC4: 같은 seed와 같은 Act 시퀀스로 두 번 실행했을 때 GameEnd 프레임과 승자가 같다.
- 011-AC5: 재생 모드에서 같은 리플레이를 두 번 재생했을 때 RawFrame 해시 시퀀스가 같다(040-R8). 지정 시점 밖의 적 유닛이 RawFrame에 없다.
- 011-AC6: 에이전트를 같은 PC(`127.0.0.1`)에 둔 구성과 원격 PC에 둔 구성 모두에서 게임이 크래시 없이 완주한다. 같은 PC 구성은 011-AC1로, 원격 구성은 020-AC1 실행(원격 게임 호스트 → 학습 호스트 에이전트)으로 판정한다(2026-10-16 사용자 결정으로 변경. 이전: 두 구성 모두 011-AC1 시나리오)
- 011-AC7 (M2 검증, 브리지 필요): 에이전트가 응답을 최대 1초 지연해도 LAN 상대 클라이언트가 강제 퇴장(드롭)되지 않는다. 퇴장되면 대응 방안을 이 spec에 추가하고 사용자 확인을 받는다.
  - **관측과 대응 (2026-10-18, 사용자 결정):** OpenBW `sync.h`의 `timeout_func`는 60초 동안 동기화가 없는 클라이언트를 끊고 "플레이어 나감"(action 87)을 넣는다. RL 스모크(Windows CUDA 학습 호스트, 동시 3게임)에서 학습 호스트 추론이 약 240초 멈추자, T0 클라이언트가 60초 뒤 모델을 끊고 모델 클라이언트도 재개 후 T0를 끊어, 두 세션이 같은 프레임(약 1분)에 서로 승리를 보고했다(28판 중 3판). 그 게임은 섬멸 패배로 학습에 들어갔다. 대응: ① 드롭 시간을 늘린다(011-R18, 학습 기본 10분). ② 퇴장을 섬멸과 구분해 보고하고(011-R17), 두 세션이 모두 상대 퇴장으로 승리를 보고하면 Aborted로 처리해 학습에서 회수한다(020-R13, 042-R3). 상대가 한 세션뿐인 게임(T1 봇)에서는 끊김과 봇의 GG 퇴장을 구분할 수 없어 ①로만 막는다. 멈춤 원인은 미확인이다(같은 시각 CUDA 드라이버 예외 기록). 추론 배치가 5초를 넘으면 시각과 함께 로그를 남긴다.
- 011-AC8 (Windows 세션 검증): Windows에서 011-AC1(두 맵 각 100판 완주)과 011-AC4(seed 결정론)를 재현한다. — **통과 (2026-10-16, `reports/m3_windows.md`)**
- 011-AC9: `surrender` Act 뒤 두 세션 `GameEnd`가 같은 승자(상대)를 보고하고 게임 결과가 Aborted가 아니며, 저장 리플레이에 채팅 명령(문자열 일치)과 퇴장 명령이 있다.
- 011-AC10 (011-R22): 거의 매 프레임 명령을 보내는 플레이어가 섬멸되는 게임(모델 argmax가 p0인 대 T0, 투혼. hostd는 launcher 슬롯 0만 리플레이를 저장한다) 20판 이상에서 두 세션 `GameEnd`의 승자가 모두 일치하고(winner disagree 0), 이긴 쪽 보고는 그대로(T0 승리, `Elimination`)이며, 진 쪽이 저장한 리플레이에 상대의 퇴장(0x57) 명령이 없고, 드롭 시간 초과가 없다. 011-R21 항복 판(011-AC9)도 다시 확인한다.

## Risks & Open Questions
- **명령 부채널(011-R20)**: 명령 성공·실패 이벤트는 시야 밖 상태를 알려 줄 수 있다. 사라진 자원은 이동으로 바꿔 막았다. 시야 밖 기억 적 건물 대상(액션 감사 F18, 사용자 결정 대기)을 열 때도 같은 규칙이 필요하다. 사라진 자원의 replay id가 다른 유닛에 재사용되어 보이면 그 유닛이 대상이 된다(드묾).
- **LAN 동기화:** lockstep으로 우리 브리지가 멈추면 LAN 상대 클라이언트도 대기한다. OpenBW는 60초 동기화가 없으면 드롭한다(2026-10-18 확인, 011-AC7 대응 → 011-R17·R18).
- OpenBW 포크는 BWAPI 4.2다. BWAPI 4.4 대상 오픈소스 봇과는 바이너리 호환되지 않는다. 대응은 [021](021-opponent-pool.md)에서 다룬다.
- 맵 파일: 투혼과 서킷브레이커의 1.16.1 호환 `.scx`/`.scm` 버전을 확보하고 해시를 고정해야 한다. MPQ 3종(Stardat, Broodat, Patch_rt)은 사용자가 직접 제공한다.

- **MPQ·맵 확보 (확정 2026-10-15, AMB-011-1)**: 두 경로를 병행한다.
  - (a) 사용자가 구버전(1.16.1) 또는 2017년 무료 배포 1.18 설치본을 확보한다.
  - (b) Claude가 M1 사전 작업으로 SC:R 설치본(CASC)에서 OpenBW에 필요한 데이터를 추출·변환할 수 있는지 조사한다. 결과는 `reports/mpq_extraction.md`에 기록한다.
  - 먼저 성공한 경로를 쓰고, 파일 해시를 `config/assets.toml`에 고정한다.
  - 맵 파일은 AMB-000-3(최신 공개 버전)을 따르며, 1.16.1 CHK 호환성을 M1(b)에서 검증한다.

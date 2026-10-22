# 012 — Bridge: Remastered (samase plugin)

- **Status**: **Approved** (2026-10-22 사용자 승인. M7 스파이크 결과와 2026-10-22 사용자 결정 AMB-012-1·2·4·5·6, AMB-001-5, AMB-030-4·5를 반영했다. 012-R15 시간 제한 없음, 012-R16 접속 실패 시 비활성 포함)
- **Related**: [001](001-system-architecture.md), [010](010-game-protocol.md), [013](013-version-mapping.md), [030](030-observation.md), [050](050-evaluation.md)
- **근거 리포트**: `reports/m7_scr_spike.md`(실게임 측정), `reports/m7_scr_field_mapping.md`(필드 대응표)

## Purpose
StarCraft: Remastered 프로세스에 samase 플러그인 DLL로 로드되는 브리지를 정의한다. 브리지는 samase가 실행 중에 분석해 넘겨주는 API(samase_scarf 기반)로 게임 상태를 읽고, 명령을 **네트워크 커맨드 경로**로 보낸다. OpenBW 브리지(011)와 같은 프로토콜(010)과 같은 의미의 관측을 에이전트에 보낸다.

## Scope & Non-goals
- 범위:
  - SC:R **1.23.10.13515 x64** 하나(AMB-012-2). x86 클라이언트도 허용했지만 x64가 스파이크를 통과했으므로 x64만 지원한다.
  - samase 로더 **0.10.28-64**(`samase64.exe`, SHA-256 `8cc8d84d2c8c533ee048760b13cf49ed369502a821840f6e35ebe537afec668c`, 플러그인 API version 44).
  - 1v1 싱글 플레이 커스텀 게임과 LAN(UDP) 게임. 상대는 SC:R 내장 AI 또는 사람이다(AMB-000-2).
  - realtime 모드(001-R5).
- 비범위:
  - 게임 시작 자동화. 로비 입장과 게임 시작은 사람이 한다. samase `--map` 자동 시작은 1.23.10에서 "Loading map failed: Code 8000000d"로 실패해 쓰지 않는다(스파이크).
  - 게임 나가기 자동화(AMB-012-4). 항복은 채팅 뒤 멈춘다(012-R11).
  - 배틀넷 공개 게임·래더, 안티치트 우회(AMB-012-3).
  - `PrivilegedSummary`. realtime에서는 만들지 않는다(010-R10).
  - 명령 실패(`CommandFailed`) 이벤트. 보류 항목이다(AMB-012-6, 012-R14).

## 구조
- `bridge/scr`: Rust `cdylib`, 타깃 `x86_64-pc-windows-msvc`. 루트 Cargo workspace에 넣지 않고 자기 workspace를 갖는다. 그래야 Linux 학습 호스트의 `cargo build/test --workspace`에 영향이 없다(001-R8).
  - 의존성: `crates/uranium-proto`(경로), `samase_plugin`(MIT), aise의 `bw_dat`(Apache-2.0, 구조체 정의). 외부 두 개는 git 리비전을 고정한다.
  - 버전별 표는 `bridge/scr/tables/*.toml`에 둔다(013-R1, AMB-001-5).
- 배치: samase 모드 폴더 `<mod>/samase/`에 DLL과 `plugins` 목록 파일을 두고, `samase64.exe <mod>`로 SC:R을 띄운다.
  - 설정(에이전트 주소 등)은 001-R9에 따라 저장소 `.env`에서 읽는다. 실행 스크립트(`tools/scr/run.ps1`)가 `.env`를 읽어 환경변수로 넘기고, 플러그인은 프로세스 환경변수를 읽는다.
  - 설치·실행 절차는 `docs/scr.md`에 둔다.
- 스레드:
  - **게임 스레드**: samase `hook_step_objects`(after) 프레임 훅에서 다음을 한다.
    - 매 프레임 기록: 유닛 집합 비교로 이벤트를 쌓고, 게임 종료를 판정한다.
    - 관측할 프레임이면 관측 스냅샷을 만든다.
    - 받아 둔 `Act`가 있으면 명령을 `send_command`로 보낸다.
  - **I/O 스레드**: 에이전트와 TCP로 주고받는다. 게임 스레드와는 lock-free 이중 버퍼·큐로만 주고받는다.
- 연결 단위: 게임 하나당 연결 하나다.
  - 게임의 첫 프레임 훅에서 접속해 `Hello`·`GameInfo`를 보낸다.
  - `GameEnd`를 보낸 뒤 연결을 닫는다. 다음 게임은 다시 접속한다.

## Requirements
| ID | 요구사항 |
|---|---|
| 012-R1 | **MUST** 플러그인은 게임 스레드를 블로킹하지 않는다. 관측 스냅샷은 lock-free 이중 버퍼로 I/O 스레드에 넘기고, `Act`는 I/O 스레드가 큐에 넣으면 게임 스레드가 다음 프레임 훅에서 꺼낸다 |
| 012-R2 | **MUST** 상태는 samase 플러그인 API로만 읽는다(스파이크에서 모두 확인). `game`, `players`, `first_active_unit`, `first_hidden_unit`, `unit_array_len`, `selections`·`client_selection`, `map_tile_flags`(하위 8비트 = 플레이어별 안 보임, 8–15비트 = 미탐색), `first_active_bullet`, `local_player_id`, `is_multiplayer`, `is_replay`, `read_vars`(타일셋·`ScMainState` 등). 구조체 필드는 `bw_dat` 정의와 013 표로만 읽는다. 플러그인 자신이 시그니처 스캔이나 주소 탐색을 하지 않는다 |
| 012-R3 | **MUST** 관측 필터는 OpenBW 브리지와 같은 의미다(010-R5, 011 shim `visible_to_self`). 자기 유닛은 모두 넣는다. 다른 유닛은 다음을 모두 만족해야 넣는다: 살아 있음, `sprite.visibility_mask`에 자기 비트, 점유 타일 중 하나가 보임, 클록·버로우면 탐지됨. 효과는 중심 타일이 보이는 것만 넣는다(AMB-030-4). 판정 세부는 `reports/m7_scr_field_mapping.md` §2를 따른다 |
| 012-R4 | **MUST** 명령은 플레이어 입력과 같은 네트워크 커맨드로만 보낸다(samase `send_command`). 유닛 구조체를 고치거나 게임 함수로 order를 직접 내리는 방식(`issue_order` 등)은 쓰지 않는다. 한 `Command`는 `0x63` Select(최대 12기씩, 011과 같은 분할)를 보낸 뒤 명령 하나를 보내고, 한 `Act`의 명령은 같은 프레임 훅에서 차례로 보낸다 |
| 012-R5 | **MUST** 네이티브 값(order·명령 바이트·유닛 태그 등)의 변환은 `bridge/scr/tables`의 013 표와 `bw_dat` 구조체로만 한다. 코드에 버전별 상수를 흩뿌리지 않는다 |
| 012-R6 | **MUST** 지원하지 않는 환경이면 플러그인은 훅을 걸지 않는 비활성 상태로 로드되고, 로그만 남긴다. 판별 기준: x64 `StarCraft.exe` SHA-256이 고정 목록(AMB-012-2)과 다르거나, 플러그인 API version이 44 미만이거나, 012-R2의 API 중 하나라도 없을 때 |
| 012-R7 | **MUST** realtime 명령 유효 기한(AMB-012-5): 에이전트는 `Act.valid_until_frame` = 관측 프레임 + 24(약 1초)를 붙인다. 브리지는 `Act`를 꺼낸 프레임이 그 값을 넘으면 명령을 버리고 건수를 기록한다(010-R7). 지연 보정은 하지 않는다 |
| 012-R8 | **MUST** 관측 시점: 프레임 f의 관측에 대한 `Act`(`advance_frames` = d)를 받은 뒤, 다음 관측은 max(f + d, f + 2, `Act`를 적용한 프레임) 프레임에 한다. 에이전트 응답을 기다리는 관측은 언제나 하나뿐이다. 응답이 늦으면 게임은 계속 진행되고 관측만 늦어진다 |
| 012-R9 | **SHOULD** 디버그 오버레이(선택 유닛, 의도된 타깃)는 기본으로 끈다 |
| 012-R10 | **MUST** 게임 종료 시 `GameEnd.final_frame`(010-R12)을 OpenBW 브리지와 같은 의미로 보낸다: 마지막 `RawFrame` 이후 이벤트와 종료 시점의 자기 시야 상태(012-R3 필터 적용)(사용자 결정 2026-10-19, E-8) |
| 012-R11 | **MUST** 항복(2026-10-21 사용자 결정, AMB-031-6. 2026-10-22 개정, AMB-012-4): `Act.surrender`(010-R13)를 받으면 012-R4 경로로 채팅을 보낸다(비어 있지 않을 때, `0x5C` + 자기 플레이어 번호 1바이트 + 80바이트 문구, 0 채움, 스파이크에서 동작 확인). 그 뒤 브리지는 **명령을 더 실행하지 않고** `GameEnd`(winner = 상대, reason = `Surrender`)를 보내고 연결을 닫는다. **게임에서 나가지는 않는다.** 게임 창은 사람이 닫는다. 이유: 스파이크에서 `0x57` 나가기 명령은 게임이 처리했지만 로컬 클라이언트가 게임에서 빠지지 않았고, samase API에는 클라이언트를 게임에서 빼는 수단이 없다. SC:R 실전 설정의 항복 모드 기본값은 `On`, 채팅 기본값은 `"gg"`이고 설정으로 바꿀 수 있다(020-R14와 같은 키) |
| 012-R12 | **MUST** `RawFrame`·`GameInfo`·`GameEnd` 각 필드는 OpenBW 브리지와 같은 의미로 채운다(`reports/m7_scr_field_mapping.md` §1). 학습 데이터에 굳어진 OpenBW 쪽 특성도 그대로 따른다: Discover는 Create와, Evade는 Destroy와 같은 프레임에 나간다. `UNDER_ATTACK`은 항상 0이다. 다른 플레이어 유닛의 내부 정보는 OpenBW처럼 보내고 에이전트가 지운다(AMB-030-5). 일부러 다르게 둔 것은 013 차이 목록에 적는다 |
| 012-R13 | **MUST** `GameInfo`: `map_hash`는 OpenBW와 같은 정의(게임이 연 맵 파일 전체의 SHA-256, `Game.map_path`)로 만든다. 맵 사전계산 키(031, AMB-031-4)가 같아야 하기 때문이다. `latency_frames`는 싱글 1, LAN 2로 보낸다(스파이크 실측: 보낸 프레임 대비 처리 프레임). 브리지는 자기 첫 명령이 처리된 프레임으로 실제 지연을 재서 다르면 로그와 `BridgeStats`에 남긴다. 에이전트는 이 값을 기록에만 쓴다(AMB-012-5, 013-R3) |
| 012-R14 | **MUST NOT** `CommandFailed` 이벤트를 만들지 않는다. 에이전트는 SC:R 세션에서 `last_failed`를 0으로 고정한다(030-R10과 같은 처리). RL 모델을 SC:R에 배포하기 전에 방법을 다시 정한다(**[SPEC DEFERRED(M7)] AMB-012-6**: M7 안에서 RL 모델을 SC:R에 올리기 전에 다시 묻는다. 후보는 BWAPI 명령 가능 검사 이식, 게임 오류 메시지·order 변화로 근사) |
| 012-R15 | **MUST** 게임 종료 판정: `Game.victory_state`가 3이면 승리, 1·2·4·6이면 패배다. `Player.player_type`이 10·11이 되거나 중립(7)으로 바뀐 플레이어는 나간 것이다. 승자 판정은 011과 같은 규칙(둘 다 패배면 무승부)이다. 실게임에서 이 전이를 확인하는 것은 012-AC2에 포함한다. 브리지는 시간 제한으로 게임을 끝내지 않는다(020의 60분 무승부는 OpenBW 학습·평가 규칙) |
| 012-R16 | **MUST** 에이전트에 접속할 수 없으면 그 게임에서 봇은 아무 명령도 내리지 않는다. 화면 문구(`print_text`)와 로그로 알리고, 게임은 그대로 진행된다 |
| 012-R17 | **MUST** 유닛 id는 SC:R 유닛 태그(배열 index 13비트 + 재사용 카운터, 배열 길이 3400, 스파이크 실측)를 u32로 쓴다. 에이전트에게는 불투명한 값이다. 명령의 대상은 같은 태그로 유닛을 찾아 보낸다 |

### 에이전트 측 (uranium-agent)
| ID | 요구사항 |
|---|---|
| 012-R18 | **MUST** `Hello.mode = Realtime`인 세션에서 에이전트는 다음과 같이 동작한다. `PrivilegedSummary`를 기대하지 않는다. 012-R7 유효 기한을 붙인다. `last_failed`를 0으로 고정한다(012-R14). 항복 모드 기본값은 `On`이다(031-R12) |
| 012-R19 | **MUST** 같은 에이전트 바이너리가 백엔드 플래그만 바꿔 OpenBW와 SC:R 브리지를 모두 받는다(001-AC1). SC:R 실전 실행은 `uranium-agent`의 하위 명령으로 한다(체크포인트·샘플링·listen 주소 지정). 추론 장치는 지연 측정 뒤 정한다(AMB-001-1) |

## 명령 인코딩 (SC:R)
- 유닛을 지정하는 명령은 SC:R 1.21 확장 opcode를 쓴다. 레이아웃은 [013](013-version-mapping.md) 표를 따른다.
  - Select `0x63`(count + u32 태그 × count)
  - RightClick `0x60`(x u16, y u16, u32 태그, u16 unit_type, queued u8)
  - TargetedOrder `0x61`(위와 같고 order u8 추가)
  - Unload `0x62`(u32 태그)
- 그 밖의 명령(Build, Train, Research, Upgrade, Siege, Unsiege, Stim, Lift, Land, Hold, Stop, ReturnCargo, Cloak, Decloak, Cancel 계열, 채팅 `0x5C`)은 1.16.1과 같은 opcode와 레이아웃이다.
- 스파이크에서 Select·RightClick·채팅을 실게임으로 확인했다. 나머지 명령은 012-AC5에서 확인한다.
- `Command` → 바이트 대응은 `reports/m7_scr_field_mapping.md` §3을 따른다(011 shim이 BWAPI로 내는 것과 같은 명령).
- 명령 순서는 `Select(units) → Order`다. 사람이 봇 클라이언트의 선택을 바꾸는 경우는 고려하지 않는다(봇 전용 클라이언트).

## Acceptance Criteria
- 012-AC1 (**통과 2026-10-22**): samase 0.10.28-64가 SC:R 1.23.10.13515 x64에서 units, players, game, selections, send_command, step_game hook, visibility를 모두 제공한다(`reports/m7_scr_spike.md`).
- 012-AC2: SC:R vs 내장 AI 1v1 10판(싱글 커스텀, 투혼·서킷브레이커, 상대 종족 섞음)을 크래시 없이 완주한다. 각 판의 SC:R 리플레이를 SC:R에서 다시 재생했을 때 결과(승패·끝난 프레임)가 같아야 하고, 브리지가 보고한 `GameEnd`와도 같아야 한다.
- 012-AC3: 같은 리플레이를 SC:R(플러그인 재생 관측 덤프)과 OpenBW(013 변환 후 재생)로 돌려, 같은 플레이어 시점의 `RawFrame`을 필드별로 비교한다. 10개 장면에서 013-AC2 기준(유닛 수, 위치 ±0px, HP, order)을 만족한다.
- 012-AC4: 실제 브리지가 게임 스레드에 더하는 시간이 프레임당 p99 ≤ 0.5ms다(관측 프레임 포함, 30분 이상 게임 1판). 참고로 스파이크 탐침은 p99 0.018~0.034ms였다.
- 012-AC5: 034 액션 테이블의 명령 종류를 SC:R 실게임에서 한 번 이상 보내, 게임이 의도한 order로 처리했는지 확인한다(011의 031-AC1 시나리오와 같은 목록, 테란 기준).
- 012-AC6: realtime 실행에서 `PrivilegedSummary` 송신 0건(010-AC5), 유효 기한을 넘긴 `Act` 건수와 관측→명령 처리 지연(프레임) 분포를 `BridgeStats`·로그로 보고한다.
- 012-AC7: 항복 시험: 강제 항복을 걸면 채팅이 상대 화면에 뜨고, `GameEnd(Surrender)` 뒤 명령이 0건이다.

## Risks & Open Questions
- **samase 로더는 비공개 바이너리이고 제3자 저장소에서 받는다.** SHA-256을 고정했다. 다른 빌드로 바꾸려면 사용자 확인을 받는다.
- **계정 위험**: 사용자가 본계정 사용과 위험 감수를 결정했다(2026-10-22).
- **자동 업데이트**: 배틀넷 런처가 SC:R을 패치하면 012-R6에 따라 비활성이 된다. 지원 빌드 추가는 그때 사용자에게 묻는다(AMB-012-2).
- **명령 실패 신호 없음**(AMB-012-6): RL 모델이 `last_failed` 입력을 쓰므로 RL 모델을 SC:R에 배포하기 전에 정해야 한다.
- **지연**: LAN에서 관측부터 명령 처리까지 추론 시간 + 2프레임이 걸린다. 보정하지 않는다(AMB-012-5). 012-AC6 분포를 보고 다시 정할 수 있다.
- **사람이 시작해야 함**: 012-AC2와 050 E-SCR-Smoke(최소 18판)는 판마다 사람이 게임을 시작해야 한다.
- 맵 이름 인코딩(CP949/UTF-8), 로비 Random 종족 위치는 구현 때 확인한다(필드 대응표 §4-8).
- **확정 기록**:
  - AMB-012-1 (2026-10-22): 대안이 필요 없다. samase가 실행 중 분석해 API를 제공하므로 디스크 exe를 직접 분석하지 않는다.
  - AMB-012-2 (2026-10-22): SC:R 빌드를 1.23.10.13515로 고정한다. x64 `StarCraft.exe` SHA-256은 `ce3ab05dc9a6aa35418947e5d95e19651ad6fbcd5e2c1856cd729c5b7c31281c`이다. x86(SHA-256 `32dbbdd001dd381cb1b3a719b7ad1fc918a9d4bc99661c675e00254efecca827`)도 허용했지만, x64가 통과해 x64만 쓴다. 기존 samase 환경은 없었다(설치 절차는 `docs/scr.md`). 실전 PC는 CUDA 학습 호스트(Windows)와 같다.
  - AMB-012-3 (2026-10-15): 로컬/LAN 커스텀 게임만 지원한다.
  - AMB-012-4 (2026-10-22): 항복은 채팅 후 멈춘다(012-R11).
  - AMB-012-5 (2026-10-22): 지연을 보정하지 않는다. 유효 기한은 24프레임이다(012-R7).
- 추론 위치(SC:R PC 로컬 vs 원격)는 [001](001-system-architecture.md) AMB-001-1에서 지연을 잰 뒤 정한다.

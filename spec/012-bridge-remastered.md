# 012 — Bridge: Remastered (samase plugin)

- **Status**: Draft
- **Related**: [010](010-game-protocol.md), [013](013-version-mapping.md), [050](050-evaluation.md)

## Purpose
StarCraft: Remastered 프로세스에 samase 플러그인 DLL로 로드되는 브리지를 정의한다. samase_scarf로 찾은 메모리 주소와 함수로 게임 상태를 읽고, 명령을 **네트워크 커맨드 경로**로 송신한다.

## Scope & Non-goals
- 범위: 64비트 SC:R, 1v1 커스텀/로비 게임, realtime 모드.
- 비범위:
  - 배틀넷 래더 자동화와 로비 자동 입장. 게임 시작까지는 사람이 조작한다.
  - 안티치트 우회. 사용 환경은 로컬/LAN 테스트 또는 허용된 환경으로 한정한다.

## 구조
- `bridge/scr`: `crate-type = ["cdylib"]`, 타깃 `x86_64-pc-windows-msvc`.
- samase 로더가 플러그인을 주입하면, 초기화 시 samase가 넘겨주는 API 테이블로 게임 훅을 등록한다. API 테이블은 samase_scarf 분석 결과에 기반한다.
- 게임 프레임 훅(step_game 이후)에서 상태를 스냅샷하고, 별도 I/O 스레드가 TCP 송수신을 맡는다. 게임 스레드는 블로킹하지 않는다.

## Requirements
| ID | 요구사항 |
|---|---|
| 012-R1 | **MUST** 플러그인은 게임 스레드를 블로킹하지 않는다. 상태 스냅샷은 lock-free 이중 버퍼로 I/O 스레드에 넘긴다 |
| 012-R2 | **MUST** 상태 읽기 대상: 유닛 리스트(first_active_unit 체인), 플레이어 자원·서플라이·업그레이드·테크, 시야/탐색 마스크, 크립 맵, 게임 프레임, 게임 종료 상태 |
| 012-R3 | **MUST** 자기 플레이어의 시야 마스크로 적 유닛을 필터링한다. 보이지 않는 적, 클록·버로우 상태에서 탐지되지 않은 적은 제외한다. 결과는 010-R5와 같은 의미여야 한다 |
| 012-R4 | **MUST** 명령은 플레이어 입력과 같은 **네트워크 커맨드**(select + order 계열)로 송신한다. 유닛 구조체를 직접 수정하는 방식은 멀티플레이 디싱크를 일으키므로 금지한다 |
| 012-R5 | **MUST** 네이티브 ID와 구조체 오프셋은 [013](013-version-mapping.md) 테이블로만 변환하고, 코드에 상수를 흩뿌리지 않는다 |
| 012-R6 | **MUST** 지원하지 않는 SC:R 빌드면 플러그인이 비활성 상태로 로드되고, 게임에 영향을 주지 않으며, 로그만 남긴다. SC:R 빌드 판별은 exe 해시 또는 samase_scarf 분석 실패로 한다 |
| 012-R7 | **MUST** realtime 모드: 에이전트 응답이 `valid_until_frame`을 넘으면 그 명령은 버린다(010-R7) |
| 012-R8 | **SHOULD** 관측 송신 주기는 에이전트가 요청한 delay를 따르되, 최소 간격은 2프레임으로 한다 |
| 012-R9 | **SHOULD** 디버그 오버레이(선택 유닛, 의도된 타깃)는 기본으로 끈다 |
| 012-R10 | **MUST** 게임 종료 시 `GameEnd.final_frame`(010-R12)을 OpenBW 브리지와 같은 의미로 보낸다: 마지막 `RawFrame` 이후 이벤트와 종료 시점의 자기 시야 상태(012-R3 필터 적용). M7 구현 때 반영한다(사용자 결정 2026-10-19, E-8) |
| 012-R11 | **MUST** 항복(2026-10-21 사용자 결정, AMB-031-6, M7 구현. 2026-10-22 개정, AMB-012-4): `Act.surrender`(010-R13)를 받으면 012-R4 네트워크 커맨드 경로로 채팅을 보낸다(비어 있지 않을 때, `0x5C` + 보낸 사람 번호 + 80바이트 문구). 그 뒤 브리지는 **명령을 더 실행하지 않고** `GameEnd`(winner = 상대, reason = `Surrender`)를 보내고 에이전트 연결을 닫는다. **게임에서 나가지는 않는다.** 게임 창은 사람이 닫는다. 이유: 스파이크에서 `0x57` 나가기 명령은 게임이 처리했지만 로컬 클라이언트가 게임에서 빠지지 않았고, samase API에는 클라이언트를 게임에서 빼는 수단이 없다(`reports/m7_scr_spike.md`). SC:R 실전 설정의 항복 모드 기본값은 `On`, 채팅 기본값은 `"gg"`이고 설정으로 바꿀 수 있다(020-R14와 같은 키) |

## 명령 인코딩 (SC:R)
SC:R 확장 커맨드를 사용한다. 정확한 바이트 레이아웃은 [013](013-version-mapping.md)에 둔다.
- `0x63 Select (Remastered)`, `0x64 SelectAdd`, `0x65 SelectRemove`: 유닛 ID 확장 형식
- `0x60 RightClick (Remastered)`, `0x61 TargetedOrder (Remastered)`, `0x62 Unload (Remastered)`
- `Build`, `Train`, `Research`, `Upgrade`, `Siege`, `Unsiege`, `Stim`, `Lift`, `Land`, `Hold`, `Stop`, `ReturnCargo`, `Cloak`, `Decloak`: 1.16.1과 같은 opcode. M1(c)에서 검증한다

명령 전송 순서는 `Select(units) → Order`이며 한 Act 안에서 원자적으로 큐잉한다. 이후 플레이어가 수동으로 선택을 바꾸는 경우는 고려하지 않는다(봇 전용 클라이언트).

## Acceptance Criteria
- 012-AC1 (M1c): 현재 SC:R exe에 samase_scarf 분석을 돌렸을 때 필요한 주소/함수(아래 목록)를 모두 찾는다. 실패하면 대안 경로를 확정하고 이 spec을 갱신한다.
  - 필요 목록: units, players, game, selections, send_command, step_game hook, visibility
- 012-AC2: SC:R vs 내장 AI 1v1 10판을 크래시와 디싱크 없이 완주한다. 리플레이 재생 결과도 일치해야 한다.
- 012-AC3: 같은 장면에서 OpenBW RawFrame과 SC:R RawFrame의 필드별 골든 비교가 허용 오차 이내다. 장면은 동일 리플레이 재생으로 만든다([013](013-version-mapping.md)).
- 012-AC4: 게임 스레드에 추가되는 지연은 프레임당 p99 ≤ 0.5ms다.

## Risks & Open Questions
- **samase_scarf README에 따르면 현재 배포 실행 파일 형식은 분석이 불가능하다.** 대안을 M1(c)에서 순서대로 검토한다.
  1. 실행 중 프로세스 메모리 덤프에서 언패킹된 이미지로 분석하기
  2. 분석이 가능한 SC:R 버전으로 고정해 오프라인 환경에서 쓰기
  3. 시그니처 스캔을 수동으로 유지보수하기

- **확정 (2026-10-22, AMB-012-1)**: 대안이 필요 없다. M7 스파이크에서 samase 로더 0.10.28-64(플러그인 API 44)가 SC:R 1.23.10.13515 x64를 실행 중 분석해, 012-AC1 목록을 플러그인 API로 모두 제공했다. 디스크 exe를 직접 분석하지 않는다. 명령 송신과 시야 비트도 실게임에서 확인했다(`reports/m7_scr_spike.md`).
>
- **확정 (2026-10-22, AMB-012-2)**:
  - 대상 SC:R 빌드는 **1.23.10.13515 하나로 고정**한다(x86_64 `StarCraft.exe` SHA-256 `ce3ab05dc9a6aa35418947e5d95e19651ad6fbcd5e2c1856cd729c5b7c31281c`, x86 `StarCraft.exe` SHA-256 `32dbbdd001dd381cb1b3a719b7ad1fc918a9d4bc99661c675e00254efecca827`). 64비트가 아니라 32비트(x86) 클라이언트를 써도 된다(사용자 확인 2026-10-22). 둘 중 무엇을 쓸지는 스파이크 결과로 정하고, 정해지면 범위와 `bridge/scr` 타깃을 그에 맞춘다. 이 해시가 아닌 빌드에서는 012-R6에 따라 비활성으로 로드된다. 패치가 나오면 지원 빌드 추가 여부를 그때 사용자에게 묻는다.
  - 사용자가 쓰던 samase 환경은 없다. 로더 확보와 설치 절차는 M7 스파이크에서 조사해 `docs/`에 정리한다.
  - 실전 PC는 CUDA 학습 호스트(Windows)와 같은 PC다(추론 위치 AMB-001-1은 지연 측정 뒤 결정).
- **확정 (2026-10-15, AMB-012-3)**: 실행 환경은 로컬/LAN 커스텀 게임에 한정한다([000](000-overview.md)). 배틀넷 연결 상태의 공개 게임과 래더는 지원하지 않는다.
- SC:R 패치마다 오프셋이 바뀐다. 자동 분석이 실패하면 지원 빌드 목록을 관리한다.
- SC:R 유닛 수 한도와 확장 유닛 ID 형식(16비트 unit index)을 확인해야 한다.
- 추론 위치(SC:R PC 로컬 vs 학습 호스트 원격)는 [001](001-system-architecture.md)에서 정한다.

# M7 스파이크 — samase·samase_scarf로 SC:R 1.23.10.13515 연동 가능성

- 시작: 2026-10-22
- 관련 spec: [012](../spec/012-bridge-remastered.md) (Draft), [013](../spec/013-version-mapping.md), [001](../spec/001-system-architecture.md) AMB-001-1, 012-AC1
- 장비: 학습 PC(Windows, CUDA). SC:R 실전 PC와 같다(AMB-012-2). 착수 시점에 학습 작업 없음(사용자 확인)

## 1. 배경과 예상 (실험 전)
- M7의 첫 단계는 012-AC1이다. 설치된 SC:R exe에서 브리지에 필요한 주소·함수를 모두 찾아야 한다. 필요 목록: units, players, game, selections, send_command, step_game hook, visibility.
- 대상 빌드는 1.23.10.13515 하나로 고정했다(2026-10-22, AMB-012-2). exe SHA-256은 `ce3ab05d…281c`(012 본문)다.
- 예상:
  - 디스크의 exe는 samase_scarf로 직접 분석할 수 없다(README). 실행 중 언팩된 이미지로 분석해야 한다(012 대안 1).
  - samase 로더가 13515를 지원하면, 로더 안의 분석 결과를 플러그인 API로 그대로 받을 수 있다. 이 경우 주소 테이블을 우리가 직접 관리하지 않아도 된다.
  - 게임 종료·승패·채팅·나가기는 전용 API가 없을 가능성이 크다. Game 구조체 필드나 명령 바이트로 해결해야 한다.

## 2. 문헌 조사 (2026-10-22, 실행 없음)
| 항목 | 결과 | 확인 수준 |
|---|---|---|
| samase 로더 | 소스 비공개. 공식 배포는 staredit.net 스레드 첨부(2017, 오래됨)뿐이다. 최신 64비트 빌드는 제3자 GitHub 저장소 `BoneHorror/sezam`의 `bin/samase64.exe`(2026-08-04 갱신, 2,765,312바이트, 라이선스 표기 없음)에 있고, 2026-02 스레드에서 개발자 빌드로 안내됐다 | 확인 |
| 실행 방식 | `samase64.exe <모드 디렉터리>`로 SC:R을 띄우고 그 안에 들어간다. 플러그인 DLL은 `<모드>/samase/`에 두고 `samase/plugins` 파일에 나열한다. 백신 오탐 보고가 있다 | 확인 |
| 플러그인 API | `samase_plugin`(MIT, `VERSION = 44`), 진입점 `samase_plugin_init`. 제공 항목: game·players·first_active_unit·units·selections·client_selection·map_tile_flags·local_player_id·is_replay·is_multiplayer, `hook_step_objects`, `send_command`, `hook_ingame_command`, `print_text`, `read_vars`/`write_vars` | 확인 |
| API에 없는 것 | 게임 종료·승패, 나가기, 채팅, 프레임 카운터 전용 API. Game 구조체 필드와 `send_command` 명령 바이트로 처리해야 할 것으로 보인다 | 추정 |
| samase_scarf | Apache-2.0. 테스트 비교 데이터가 1.23.10a~g(32/64비트)까지 있고, 최신은 `12310g`(2024-11 추가)다. 13515가 12310g인지는 불확실하지만, 그 뒤로 새 버전 데이터가 없다. 64비트에서는 game·players 포인터가 XOR로 가려져 있어 고정 주소로 읽을 수 없다 | 확인(버전 대응은 추정) |
| 참고 코드 | neivv/mtl `rally.rs`(자체 명령을 `send_command`로 보내고 `hook_ingame_command`로 받음, 네트워크 명령 경로의 실제 예), neivv/aise·aice(초기화·구조체 래퍼) | 확인 |
| 같은 빌드 사례 | 제3자 저장소 `StarCraft-Remastered-BWAPI-Bridge`(2026-10-19)가 SC:R 1.23.10.13515 **x86 전용**으로 BWAPI 봇을 돌린다고 한다. 바이너리만 있고 소스는 없다 | 확인(내용 검증 안 함) |
| 위험 | 개발자가 2017년에 "블리자드 계정 정지 가능성"을 경고했다. exe에 탐지·종료 보호 동작이 있다. LAN 게임의 안티치트 여부, 배틀넷 주기 로그인(30일 언급), 자동 업데이트 차단 방법은 확인하지 못했다 | 미확인 |

## 3. 진행 기록
- 2026-10-22: AMB-012-2를 확정했다(빌드 고정, 기존 samase 환경 없음, 실전 PC = 학습 PC). 위 문헌 조사를 마쳤다. samase64.exe를 받아 실행할지는 사용자 결정을 기다린다.

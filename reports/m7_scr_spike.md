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
- 2026-10-22 18시대: 사용자 결정 — samase64.exe를 받아 써 보고, 본계정 위험을 감수한다.
  - 받은 파일: 2,765,312바이트, SHA-256 `8cc8d84d2c8c533ee048760b13cf49ed369502a821840f6e35ebe537afec668c`, 코드 서명 없음.
  - VirusTotal 해시 조회는 API 키가 없어 하지 못했다. 대신 Windows Defender 단일 파일 검사를 했고 위협은 없었다.
  - 탐침 플러그인 `uranium_scr_probe.dll`을 만들었다(저장소 밖 스파이크 폴더, samase_plugin + aise `bw_dat` 구조체). 하는 일:
    - API 항목이 실제로 해석되는지 기록한다.
    - 프레임 훅에서 프레임·자원·서플라이·플레이어별 유닛 수·내 유닛 위치와 타일 플래그를 읽는다.
    - 적 유닛의 타일 비트로 시야 비트 배치를 확인한다.
    - 훅 시간 p50·p99를 잰다(012-AC4).
    - 240프레임에 SCV 하나를 `0x63` Select와 `0x60` RightClick으로 움직이고, 48프레임 뒤 위치·order를 확인한다.
    - 보내거나 받은 명령 바이트를 모두 기록한다(013-AC3 자료).
  - 실행은 보류했다. 같은 PC에서 래더 리플레이 일일 수집(17:54 시작)이 기존 SC:R 인스턴스를 쓰고 있어서, 수집이 끝난 뒤 실행한다.
- 2026-10-22: 사용자 확인 — x64가 아니라 x86 클라이언트를 써도 된다. 대비책으로 x86도 준비했다.
  - x86 `samase.exe`: 2,126,848바이트, SHA-256 `3c83a07b203a62fb0dbd05270c7c277481157a7a583e92950a4dc858dde87e76`, Defender 검사 깨끗함.
  - 탐침 i686 빌드: 같은 코드다. SC:R x86에서도 진입점은 `samase_plugin_init`이고, aise의 `samase_shim`은 1.16.1 전용이다.
  - 순서: x64부터 해 보고, 막히면 x86으로 바꾼다.
- 2026-10-22 21:23: 래더 수집이 끝났다. 수집에 쓰던 SC:R 인스턴스를 닫고 samase64로 실행했다.
- 21:23·21:24·21:28: `--map` 시도 3회가 모두 실패했다. 대화상자 문구는 "Loading map failed: Code 8000000d, Game build is 13515, Loader version is 0.10.28-64"였다. 래더 투혼(괄호·공백이 있는 원래 경로, 단순 경로로 복사한 사본)과 기본 맵 Astral Balance 모두 같은 결과였다. 로더와 플러그인 로드는 정상이었으므로 `--map` 옵션만의 문제로 본다. 원인은 조사하지 않았다(브리지 운영은 사람이 게임을 시작하는 방식, 012 비범위).
- 21:41: `--map` 없이 띄우고 사용자가 싱글 플레이 커스텀 게임(투혼, 테란 vs 컴퓨터 1)을 시작했다. 결과:

| 확인 항목 | 결과 |
|---|---|
| 로더 | samase 0.10.28-64, 플러그인 API version 44(max_func_id 80). 빌드 13515를 인식했다 |
| 012-AC1 목록 | units(`first_active_unit`·`units`·`unit_array_len`), players, game, selections·client_selection, send_command, step_game 훅(`hook_step_objects`), visibility(`map_tile_flags`)를 **모두 찾았다**. local_player_id·is_multiplayer·is_replay·print_text·first_hidden_unit도 있다 |
| 상태 읽기 | 첫 프레임 훅 frame 0, local_player 3, multiplayer 0, replay 0, 맵 128×128이었다. 미네랄 50, 서플라이 8/20(서플라이 값은 실제의 2배 단위), 플레이어별 유닛 수 [0]=8~9(컴퓨터), [3]=5(SCV 4 + CC), [11]=122(중립)였다. SCV는 unit_id 7, 체력 15360(= 60 × 256) |
| 시야 비트 | `map_tile_flags` 하위 8비트는 플레이어별 "안 보임"이고 8–15비트는 "미탐색"이다. 내 SCV 타일 `0x0201f7f7`에서 비트 3(나)은 두 영역 모두 꺼져 있었다(보임·탐색됨). 적 유닛 9개의 타일은 비트 3이 모두 켜져 있었다(안 보임). 012-R3 필터에 쓸 수 있다 |
| 유닛 한도 | `unit_array_len` = **3400**(확장 한도)이다. 유닛 태그는 인덱스 13비트 + 재사용 카운터다(013 태그 규칙, aise `bw_dat`의 `to_unique_id`와 같다) |
| 명령 송신 (012-R4) | 240프레임에 SCV(uid `0x2ccb`)를 `63 01 cb 2c 00 00`(Select121)로 선택하고 `60 98 0f 08 0f 00 00 00 00 e4 00 00`(RightClick121: x 3992, y 3848, 대상 없음, unit_type 0xE4, 큐 0)으로 보냈다. 게임이 241프레임에 이 두 명령을 플레이어 3의 인게임 명령으로 처리했다. 288프레임에는 order 6(Move)으로 (3953,3845)까지 갔고, 479프레임에는 (3956,3845)에서 멈춰 있었다 |
| 명령 바이트 | 013 표의 SC:R 1.21 형식(Select121: count + u32 태그, RightClick121 12바이트)과 같다(013-AC3 1차 확인) |
| 프레임 훅 지연 (012-AC4 참고) | 탐침 수준(유닛 순회 + 주기적 로그)에서 2,400프레임 동안 p50 0.006ms, p99 0.020ms, 최대 0.546ms(로그 쓰기 프레임)였다. 실제 브리지는 직렬화가 더해지므로 다시 잰다 |
| 그 밖 | 클라이언트는 프레임마다 `0x37` 동기 명령(7바이트)을 보낸다. `hook_send_command` 기록 상한(2,000줄)을 이것이 대부분 차지했으므로, 다음 탐침에서는 0x37을 걸러야 한다 |

- 결론: **012-AC1 통과**(x64, samase 0.10.28-64). AMB-012-1(분석 실패 시 대안)은 필요 없다. x86은 시험하지 않았다(x64가 동작해서).
- 로그 보관(학습 PC 스파이크 폴더): `logs/probe-x64-20261022-run1.log`, `logs/samase-20261022.log`. 탐침 소스는 스파이크 폴더 `probe/`에 있다.
- 21:54 2차 탐침(LAN): 0x37 동기 명령을 거르고, 승패 상태(`victory_state`·`player_lose_type`·`ScMainState`)가 바뀔 때마다 기록하게 했다. 사용자가 LAN(UDP) 게임(투혼, 컴퓨터 1)을 혼자 시작해 채팅 "gg"를 치고 게임을 끝냈다.
  - `is_multiplayer` = 1. 240프레임에 보낸 Select·RightClick을 게임이 **242프레임**에 처리했다. **LAN 명령 지연은 2프레임**이다(싱글은 1프레임).
  - 사람이 친 채팅은 `hook_send_command`에 잡히지 않았다. 자동 저장 리플레이(22:00:21)에는 `0x5C`가 1개 있었다. 클라이언트는 채팅을 다른 함수로 보내는 것으로 보인다.
  - 사람이 게임을 끝낸 뒤에는 프레임 훅이 돌지 않아 승패 상태 변화가 기록되지 않았다(479프레임에서 로그가 끝남).
  - 로그: `logs/probe-x64-20261022-run2-lan.log`
- 22:02 3차 탐침(LAN): 480프레임에 `send_command`로 채팅 `5C 03 "uranium gg" + 0 채움`(82바이트), 720프레임에 나가기 `57 01`을 보냈다. 게임 루프 시작 훅에서도 승패 상태를 기록하게 했다.
  - 채팅: 게임이 482프레임에 플레이어 3의 명령으로 처리했고 화면에 표시됐다(사용자 확인). **항복 채팅(012-R11)은 `send_command` 0x5C로 된다.**
  - 나가기: 게임이 722프레임에 명령으로 처리했지만, **로컬 클라이언트는 게임에서 빠지지 않았다**(사용자 확인). 8,400프레임까지 우리 유닛 5기가 플레이어 3 소유로 남았고 `victory_state`도 0이었다. samase API에는 클라이언트의 게임 상태·다음 화면 변수가 없고, samase_scarf에도 관련 분석이 없다. 원래 BWAPI는 나가기 명령을 직접 보내지 않고 클라이언트 게임 상태를 끝내 점수 화면으로 넘긴다(그러면 게임이 나가기를 보낸다).
  - 프레임 훅 지연 7,200프레임: p99 0.018~0.034ms, 최대 0.11~1.99ms.
  - 로그: `logs/probe-x64-20261022-run3-lan.log`
- OpenBW 경로 필드별 대응표(서브 에이전트 조사, BWAPI 소스 근거): [`m7_scr_field_mapping.md`](m7_scr_field_mapping.md)
- 남은 확인(브리지 구현 때):
  - 게임 종료·승패: BWAPI는 `victory_state` 3을 승리, 1·2·4·6을 패배로 보고, `player_type`이 10·11이거나 중립(7)으로 바뀐 것을 나감으로 본다(대응표 §2). 실게임 전이는 아직 기록하지 못했다.
  - 로컬 클라이언트를 게임에서 빼는 방법(위 3차 결과, 사용자 결정 필요).
  - 실제 브리지의 훅 지연.

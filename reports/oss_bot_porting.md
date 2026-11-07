# T1 오픈소스 봇 포팅 (021 1·2단계)

- 일자: 2026-10-16
- 대상 환경: OpenBW BWAPI 4.2 포크, Linux GCC. Windows MSVC는 `docs/windows.md`에서 다룬다.
- 결정: AMB-021-1 단계적 접근. 2026-10-16에 재확인했다. Steamhammer·McRave 포팅에 Stardust·ZZZKBot·Iron을 추가했다.
- 빌드: `tools/oss/build_<bot>.sh` → `<URANIUM_OSS_BOTS_DIR>/<bot>/{module.so, VERSION, bwapi-data/}`
  - 공통 헬퍼: `tools/oss/common.sh`
  - 제공 기능: 고정 커밋 clone, zip sha256 검증, 패치 적용, 헤더 의존성까지 보는 증분 병렬 컴파일, 링크
  - 6개 봇 모두 소스를 지운 상태에서 다시 받아 빌드되는 것을 확인했다.

## 봇 목록 (021-R2 기록 항목)
| 봇 | 종족 | 라이선스 | 원본 BWAPI | 고정 | 패치 | 상대 ID (021-R3) |
|---|---|---|---|---|---|---|
| UAlbertaBot | T/P/Z | MIT | 4.4 (클라이언트 실행 파일) | 커밋 `558899d8` | 0005 | `ualbertabot@558899d` |
| Steamhammer 5.3.6 | Z 주력·P·T | MIT | 4.4 | 소스·설정 zip sha256 | 0006 | `steamhammer@5.3.6` |
| ZZZKBot | Z | LGPL-3 | 4.2.0 | 커밋 `7183e37b` | 0007 | `zzzkbot@7183e37` |
| Stardust | P | MIT + 대회 출전 금지 조건 (우리 용도에는 해당 없음) | 저자 OpenBW 포크 (4.2 기반 확장) | 커밋 `22d93d7a` | 0008 | `stardust@22d93d7` |
| McRave | P/Z/T | MIT | 4.4 | 커밋 `7d1719a2` | 0009 | `mcrave@7d1719a` |
| Iron (AIIDE 2017판) | T | MIT/X11 | 4.1.2 | AIIDE 2019 봇 아카이브 zip sha256 | 0010 | `iron@aiide2017` |

- 모든 봇이 맵핵 없이 동작한다(021-R5). 봇 설정이나 코드에서 `CompleteMapInformation`을 켜는 곳이 없다.
- 봇 소스는 커밋하지 않는다. `third_party/`는 `patches/`만 추적한다.

## 봇별 포팅 내역
공통 작업은 세 가지다. `Dll.cpp`(Windows 진입점)를 `linux/UraniumModule.cpp`로 대체하고, MSVC 전용 함수의 호환 헤더(`linux/compat.h`)를 넣고, `#include` 경로 구분자(`\`)를 고쳤다.

| 봇 | 주요 수정 | 비고 |
|---|---|---|
| UAlbertaBot | BOSS·SparCraft 초기화를 어댑터로 옮김, 설정 경로 이동 | 1단계에서 완료 |
| Steamhammer | include 구분자 1곳 | 설정 JSON은 릴리스 zip에서 가져온다. 사용하지 않는 변형 파일 `OpsBoss with immobile.cpp`는 제외한다 |
| ZZZKBot | `getRandomSeed()` 호출 제거 | OpenBW의 BWAPI는 이 함수를 구현하지 않고 예외를 던진다. 그래서 시작 직후 게임이 끝났다. 300프레임 미만 의심 게임 표시로 발견했다 |
| Stardust | 우리 4.2 헤더로 빌드. 저자 포크의 확장 API는 `IS_OPENBW`·계측 매크로 안에만 있어서 끈다. GCC 템플릿 구문·clock 타입 수정. vendored zstd 정적 빌드 | 정밀 채취 최적화 백엔드는 BW 내부 타이머를 프레임 단위로 예측하다가 우리 엔진에서 크래시했다(빈 집합 역참조). 저자가 남겨 둔 단순 백엔드(미네랄 고정)로 바꿨다. 채취 효율이 원본보다 약간 낮을 수 있다 |
| McRave | MSVC만 허용하는 구문 약 60곳 수정: 임시 객체의 비 const 참조, void 삼항 연산자, 반환값 누락, 불완전 타입 조회, `windows.h` | 59:59 자동 퇴장을 제거했다. 대회 시간 제한용 동작인데, 남겨 두면 우리 규칙(60분 무승부, 020-R3)에서 우리 승리로 잘못 기록된다 |
| Iron | include 대소문자(Windows는 대소문자를 구분하지 않음), `__pragma`·`strncpy_s` 호환, BWEM 템플릿 조회 순서, `std::find` 한정 | VS 프로젝트에 없는 옛 파일 `behavior/wall.cpp`는 제외한다 |

- `-fpermissive`로 GCC가 경고로 낮춘 부분 중 동작에 영향을 주는 것을 따로 점검했다. 반환값이 없는 함수(McRave 4곳)는 GCC에서 정의되지 않은 동작이라 고쳤다.
- 2026-11-07: Linux 빌드에 `-DNDEBUG`가 없어 봇 원본 Release 프로젝트(모든 봇의 Release `.vcxproj`가 `NDEBUG`)와 Windows 빌드(CMake Release)에서는 꺼져 있는 assert가 살아 있었다. 그 결과 Iron은 `myassert.cpp:22`(`onAssertThrowFailed`의 `assert(false)` — Release에서는 예외를 던져 프레임마다의 `exceptionHandler`가 잡고 계속한다), Steamhammer는 `RC/Source/Input.cpp:75`(`insist` = `assert`, Evaluator 입력 범위 검사)에서 프로세스가 죽었다(BC 기준선 T1 평가 `<shared>/eval/t1-bc-20261107/`에서 Iron 18판·Steamhammer 5판, Linux 호스트 세 대 모두). `tools/oss/common.sh`가 모든 봇에 `-DNDEBUG`를 넣고, 플래그가 바뀌면 오브젝트를 다시 빌드하게 고쳤다. 다른 봇에서 `NDEBUG`가 바꾸는 것은 표준 `assert` 계열(rapidjson·nlohmann json·bitsery 포함, 조건에 부작용 없음)뿐이라 Windows 빌드와 같아진다(UAlbertaBot `BOSS_ASSERT`·`SPARCRAFT_ASSERT`, Iron `assert_`(`_DEBUG`)는 영향 없음). 상대 ID(`iron@aiide2017`·`steamhammer@5.3.6`)는 바꾸지 않는다(021-R3은 원본 버전, 이제 Linux·Windows 바이너리가 같은 동작). 고치기 전 Linux 결과에는 상대 크래시가 우리 승으로 섞여 있을 수 있다. 재현·확인은 아래 "NDEBUG 확인" 절.

## 검증 (021-AC2, `config/plans/m3-t1.toml`)
- 조건: T0 테란 대 각 봇·종족. 판수는 아래 표와 같고 맵당 절반씩이다. 동시 5게임, 벽시계 13분.
- 결과: **100판 모두 완주.** 중간 종료 0, 300프레임 미만 의심 게임 0, 시야 위반 0, 세션 오류 0, 명령 실패율 3.16%

| 봇 (종족) | 판수 | T0 승 | 봇 승 | 봇 쪽 계산 시간 p50 (ms/프레임) |
|---|---|---|---|---|
| UAlbertaBot (P/Z/T) | 30 | 6 | 24 | 1.6 |
| Steamhammer (Z/P/T) | 30 | 6 | 24 | 2.3 |
| McRave (P 5, Z 3, T 2) | 10 | 0 | 10 | 10.3 (최대 16.4) |
| Stardust (P) | 10 | 0 | 10 | 2.1 |
| ZZZKBot (Z) | 10 | 0 | 10 | 0.6 |
| Iron (T) | 10 | 0 | 10 | 1.2 |

- ms/프레임은 게임 벽시계 시간을 프레임 수로 나눈 값이다. 우리 T0와 엔진 시간이 포함되지만, 봇 사이의 차이는 대부분 봇의 계산량에서 나온다.
- McRave는 사용자 결정(2026-10-16)으로 종족별 판수를 줄였다. 1판에 다른 봇보다 10~40배 느리기 때문이다.
- 스모크에서 Steamhammer와 McRave가 60분 무승부까지 간 경우가 있었다. 본 검증 100판에서는 모두 전멸로 끝났다.

## 결정론 (같은 seed로 봇별 2판)
| 봇 | 결과 | 원인 |
|---|---|---|
| ZZZKBot | 동일 | — |
| UAlbertaBot, Steamhammer, Iron | 다름 | 난수를 시각이나 `std::random_device`로 시작한다 |
| Stardust, McRave | 다름 | 실제 경과 시간(`steady_clock` 등)으로 계산량을 조절한다. CPU 부하에 따라 판단이 달라진다 |

- 우리 쪽(T0·브리지·엔진)은 결정론적이다. 차이는 상대 봇 내부에서 생긴다.
- 학습 상대로서는 판마다 다르게 두는 것이 오히려 다양성이 된다. 다만 020-AC2("같은 seed·상대·정책이면 재현")를 OSS 봇에도 요구할지는 정해지지 않았다.
  - 난수 시드는 패치로 고정할 수 있다.
  - 시간 기반 조절은 근본적으로 고정하기 어렵다.
- 결정(2026-10-16, AMB-021-3): 그대로 둔다. 020-AC2는 T0·셀프플레이 상대로 검증한다.

## McRave 속도 분석 (perf, 10초 샘플)
| 비중 | 위치 | 내용 |
|---|---|---|
| 95% | `McRaveModule::onFrame` | 매 프레임 봇 로직 전체 |
| 44% | `Combat::onFrame` → `BWEB::Path::generateAS` | 전투 유닛마다 매 프레임 A* 경로 탐색 |
| 10% | `BWEB::Path::generateJPS` | 다른 경로 탐색 |
| 9% | `Grids::onFrame` / `updateVisibility` | 맵 격자 전체 갱신. 칸마다 BWAPI 가상 호출 |

- 실시간(42ms/프레임)에서는 충분히 빠르다. lockstep 학습에서는 상대를 기다려야 해서 McRave 대전이 처리량을 크게 낮춘다.
- 줄이려면 봇 로직을 고쳐야 하는데, 그러면 원본과 다른 봇이 된다. 그래서 하지 않았다. 리그(M6)에서 출전 비율로 조절하는 것을 권장한다(미확정).

## NDEBUG 확인 (2026-11-07, Linux 호스트 C)
- 배경: BC 기준선 T1 평가에서 상대 봇이 판 도중 죽은 판(Iron 18·Steamhammer 5)이 `Surrender`·우리 승으로 기록됐다. 예상: assert를 끄면 원본·Windows처럼 Iron은 예외를 잡고 계속 두고, Steamhammer는 범위 밖 입력을 그대로 쓰고 계속 둔다.
- 재현: T0 테란 상대로는 24판 모두 정상 종료(T0가 대부분 일찍 져서 그 상황까지 가지 않음)라, 크래시가 난 조건과 같은 BC 모델 상대(`bc-s2e2-pt/ckpt-11387` stochastic, CPU 추론)로 Iron 12·Steamhammer 테란 6·저그 6판(맵 반반, 15분 상한, seed_base 47100)을 고치기 전·뒤 바이너리로 돌렸다.

| 빌드 | Iron | Steamhammer 테란 | Steamhammer 저그 | 우리 기록 |
|---|---|---|---|---|
| 고치기 전 | assert 크래시 6/12 | assert 크래시 1/6 | 0/6 | 크래시 7판 모두 `Surrender` 승 |
| 고친 뒤 (`-DNDEBUG`) | 0/12 | 0/6 | 0/6 | 24판 정상 종료(Elimination 패 12, 시간 상한 무승부 12), 우리 승 0 |

- 새 빌드로 나머지 봇도 다시 빌드하고 T0 테란 상대 스모크(McRave 제외 봇·종족 9종 × 2판, 맵 반반) 18판 모두 정상 종료
- 산출물(추적 안 함): 호스트 C `~/ndebug-fix/`(plan·`before`·`before_model`·`after_model`·`smoke`·빌드 로그).
- 배포: 호스트 C만 새 바이너리(설정 변형 폴더 포함). 다른 Linux 게임 호스트는 이 커밋을 pull한 뒤 `tools/oss/build_<bot>.sh`와 변형 폴더 바이너리 복사가 필요하다. Windows 빌드는 원래 `NDEBUG`라 바뀌지 않는다.

## T1 설정 변형 (021 "T1 설정 변형", 2026-11-03, AMB-042-15)

원본 봇 폴더를 복사해 오프닝 설정만 바꾼 상대다. 라이선스·원본 BWAPI·포팅·빌드 스크립트·고정은 원본 봇 것을 따른다. 바이너리(`module.dll`·`.so`)는 원본과 해시가 같다. 만들기: `tools/oss/make_variant.py`(`docs/commands.md`). 아래 네 변형은 이 스크립트로 다시 만들어 배포된 폴더와 바이트까지 같음을 확인했다(Windows 빌드, 2026-11-03). 오프닝 실측은 `reports/m6_rush_ladder.md` 3절(9풀·오버풀)과 10절(12풀 저글링·9풀 저글링). 원본 기록은 `runs/rushladder_20261103/variants.md`.

| 변형 (상대 ID, 021-R3) | 원본 상대 ID | 바꾼 파일·줄 | 이전 → 이후 | 바꾼 파일 sha256 |
|---|---|---|---|---|
| `ualbertabot9pool@558899d-9pool` | `ualbertabot@558899d` | `bwapi-data/AI/UAlbertaBot_Config.txt` 85행(Strategy) | `"Zerg" : "Zerg_ZerglingRush",` → `"Zerg" : "Zerg_9Pool",` | `a7f68d12750b5f6b1b37d737b0fcb3ec9b1a148c1252d91004c95eea5b2923f0` |
| `steamhammeroverpool@5.3.6-overpool` | `steamhammer@5.3.6` | `bwapi-data/AI/Steamhammer_5.3.6.json` 230~249행(`ZvT` 오프닝 가중치 20개) | 20개 항목 → `{ "Weight" : 100, "Strategy" : "OverpoolSpeed" }` 한 줄 | `1d776917e8f6439de9cc6293ec409af82836c7b607c737d5d674e14d6d1e191a` |
| `ualbertabot12poollings@558899d-12poollings` | `ualbertabot@558899d` | `bwapi-data/AI/UAlbertaBot_Config.txt` 111행(`Zerg_ZerglingRush` 정의, 85행 Strategy는 그대로) | `"OpeningBuildOrder" : ["Drone", "Spawning Pool", "Zergling", "Zergling", "Zergling", "Zergling"]` → `["Drone", "Drone", "Drone", "Drone", "Drone", "Overlord", "Drone", "Drone", "Drone", "Spawning Pool", "Drone", "Drone", "Zergling", "Zergling", "Zergling"]` | `15277a9eef9580451291282e19c523319dda83551bbb1e9ac5fd5a48b8cdaf94` |
| `ualbertabot9poollings@558899d-9poollings` | `ualbertabot@558899d` | 같은 파일 111행 | 같은 이전 값 → `["Drone", "Drone", "Drone", "Drone", "Drone", "Spawning Pool", "Drone", "Overlord", "Zergling", "Zergling", "Zergling"]` | `9eba66805d94d1511b823ecc325fe503625c86e63c959abe5ec16f654860de36` |

- 명령: `make_variant.py --bot ualbertabot --variant 9pool --file bwapi-data/AI/UAlbertaBot_Config.txt --replace '"Zerg_ZerglingRush",' '"Zerg_9Pool",'`, `make_variant.py --bot steamhammer --variant overpool --file bwapi-data/AI/Steamhammer_5.3.6.json --lines 230-249 --text '            { "Weight" : 100, "Strategy" : "OverpoolSpeed" }'`.
- 12풀·9풀 저글링 변형: `make_variant.py --bot ualbertabot --variant 12poollings`(또는 `9poollings`) `--file bwapi-data/AI/UAlbertaBot_Config.txt --lines 111 --text '<새 111행 전체, 들여쓰기 12칸>'`. 전략 이름(85행)을 그대로 두는 이유: UAlbertaBot은 오프닝 뒤 생산 목표를 전략 이름으로 고른다(`StrategyManager.cpp` `getZergBuildOrderGoal`, `Zerg_ZerglingRush` = 저글링 계속). 그래서 오프닝 정의만 바꾸면 오프닝 뒤 기본 러시처럼 저글링만 나온다(9풀 변형이 저글링 4기에서 멈춘 이유).
- 변형은 설정을 바꾼 종족(저그)으로만 쓴다. 학습·점검에 쓰는 변형은 그 게임을 돌리는 모든 게임 호스트에 같은 내용으로 둔다(위 sha256으로 확인).

### 운영형 저그 변형 (042 `rush-s2` 운영형 그룹, 2026-11-04 사용자 결정, AMB-042-16)

2026-11-03 운영형 저그 측정(`runs/macroladder_20261103/`, 결과 `<shared>/eval/macroladder-20261103/`)용으로 만든 세 변형이다. 셋 모두 `rush-s2` 운영형 그룹에 들어가고, `ualbertabot2hatchhydra`·`steamhammer2hatchmuta`는 100만 step 점검 상대이기도 하다(042, 050). 2026-11-04에 `tools/oss/make_variant.py`로 Windows 원본 폴더에서 다시 만들어 배포된 폴더와 바이트까지 같음을 확인했고, Linux 게임 호스트에는 그 호스트의 원본 폴더(설정 파일은 Windows 원본과 sha256 같음)에서 같은 명령으로 만들었다. 바꾼 설정 파일 sha256은 두 OS에서 같고, 바이너리는 각 OS의 원본과 같다(`module.dll`·`module.so`). Linux hostd 재시작 로그의 봇 목록에 셋 다 나온다.

| 변형 (상대 ID, 021-R3) | 원본 상대 ID | 바꾼 파일·줄 | 이전 → 이후 | 바꾼 파일 sha256 |
|---|---|---|---|---|
| `ualbertabot2hatchhydra@558899d-2hatchhydra` | `ualbertabot@558899d` | `bwapi-data/AI/UAlbertaBot_Config.txt` 85행(Strategy) | `"Zerg" : "Zerg_ZerglingRush",` → `"Zerg" : "Zerg_2HatchHydra",` | `5ea96cb146b081f4bc6db0fe1cc3b9e8dc0699ebff256dffdc88245aa385c3ec` |
| `ualbertabot3hatchmuta@558899d-3hatchmuta` | `ualbertabot@558899d` | 같은 파일 85행(Strategy) | `"Zerg" : "Zerg_ZerglingRush",` → `"Zerg" : "Zerg_3HatchMuta",` | `89851d4cb858ff4b8d6a004f36880ed6ec705c83ba8a8a490550726bf916c43a` |
| `steamhammer2hatchmuta@5.3.6-2hatchmuta` | `steamhammer@5.3.6` | `bwapi-data/AI/Steamhammer_5.3.6.json` 230~249행(`ZvT` 오프닝 가중치 20개, 오버풀 변형과 같은 자리) | 20개 항목 → `{ "Weight" : 100, "Strategy" : "ZvT_2HatchMuta" }` 한 줄 | `bcb0c9a443128c3d4583adf19e718158150be35db1047d66b00a749b22a528d4` |

- 명령: `make_variant.py --bot ualbertabot --variant 2hatchhydra --file bwapi-data/AI/UAlbertaBot_Config.txt --replace '"Zerg_ZerglingRush",' '"Zerg_2HatchHydra",'`, `--variant 3hatchmuta`는 같은 형식에 `'"Zerg_3HatchMuta",'`, `make_variant.py --bot steamhammer --variant 2hatchmuta --file bwapi-data/AI/Steamhammer_5.3.6.json --lines 230-249 --text '            { "Weight" : 100, "Strategy" : "ZvT_2HatchMuta" }'`.
- UAlbertaBot 두 변형은 전략 이름(85행)을 바꾼다. 오프닝은 설정 파일의 그 전략 정의(113·114행: 2해처리 히드라, 3해처리 뮤탈 12기)를 따르고, 오프닝 뒤 생산 목표(`StrategyManager.cpp` `getZergBuildOrderGoal`)도 그 전략의 것이 된다(`Zerg_2HatchHydra`: 히드라 +8·사거리 업그레이드(Grooved Spines)·드론 +4, `Zerg_3HatchMuta`: 히드라 +12·드론 +4. 뮤탈은 오프닝에서만 나온다).
- 원본 폴더 sha256(두 OS 같음): `UAlbertaBot_Config.txt` `8635d29c58c63f25110069cda41ca5b023a68c73d7d1b29e0eb0412dfadc6fcf`, `Steamhammer_5.3.6.json` `d4cfd34a2d311b01b115632c210189e8c6df7c223b6b3eb33c10b547faaeaca4`.

### 러시 단 변형 (042 `rush-s3` 러시 그룹, 2026-11-04 사용자 결정, AMB-042-17)

`rush-s3` 러시 그룹의 50~60% 단(자문 B5)용으로 2026-11-04에 만든 UAlbertaBot 설정 변형이다(원본 기록 `runs/rushladder_20261103/variants.md` "Round d", 스크립트 `runs/rushladder_20261103/d/make_variants.py`, 로그 `d/make_variants.log`). 출발 체크포인트 `rush-s2/ckpt-2151242` 측정(stochastic, 투혼·서킷브레이커 각 15판, 결과 `<shared>/eval/rushrung-20261104/`) 뒤 사용자 결정으로 아래 셋을 러시 그룹에 넣었다(`config/rl_rush3.toml`, 러시 그룹 10개). 같은 라운드의 `ualbertabot9pool8lings`(23-2-5, 77%)는 쉬워서 넣지 않았고, `ualbertabot9pool16lings`·`20lings`(`d/make_variants2.py`)는 측정하지 않았다. 이 세 폴더(8·16·20lings)는 Windows 게임 호스트에만 있고 Linux 게임 호스트에는 배포하지 않았다.

다른 변형과 달리 **두 줄**(85행 전략 이름과 그 전략의 오프닝 정의)을 바꾼다. `tools/oss/make_variant.py`는 한 번에 한 곳만 바꾸므로 위 스크립트로 만들었다(이전 줄을 확인하고 CRLF 유지). Windows 폴더는 원본 `ualbertabot/`과 `VERSION`·설정 파일만 다르고(`diff -rq`), 설정 파일 차이는 아래 두 줄뿐이며 `module.dll`은 원본과 sha256이 같다(`6795a032…c6bc`). Linux 게임 호스트에는 그 호스트의 원본 `ualbertabot/` 폴더(설정 파일 sha256이 Windows 원본과 같음)를 복사하고 설정 파일을 Windows 변형의 것으로 덮어쓰고 `VERSION`을 썼다. `module.so`는 그 호스트 원본과 sha256이 같고, 설정 파일 sha256은 아래 표와 같다(두 OS 같음). Linux hostd 재시작 로그의 봇 목록에 셋 다 나온다(2026-11-04).

| 변형 (상대 ID, 021-R3) | 원본 상대 ID | 측정(승-패-무) | 바꾼 파일·줄 | 이전 → 이후 | 바꾼 파일 sha256 |
|---|---|---|---|---|---|
| `ualbertabot9pool10lings@558899d-9pool10lings` | `ualbertabot@558899d` | 19-7-4 (63%) | `bwapi-data/AI/UAlbertaBot_Config.txt` 85행(Strategy), 112행(`Zerg_9Pool` 정의) | 85행 `"Zerg" : "Zerg_ZerglingRush",` → `"Zerg" : "Zerg_9Pool",`. 112행 `"OpeningBuildOrder" : ["Drone" ×5, "Spawning Pool", "Drone", "Extractor", "Overlord", "Drone", "Zergling" ×3, "Hydralisk Den", "Drone" ×4]` → `["Drone" ×5, "Spawning Pool", "Drone", "Overlord", "Zergling" ×5, "Drone" ×4]` | `98dda3c85dbb4409b72a3636971982c983a838c608f60e049461bb73daeb928e` |
| `ualbertabot9pool12lings@558899d-9pool12lings` | `ualbertabot@558899d` | 17-7-6 (57%) | 같은 파일 85행, 112행 | 85행 같음. 112행 같은 이전 값 → `["Drone" ×5, "Spawning Pool", "Drone", "Overlord", "Zergling" ×6, "Drone" ×4]` | `b8f21eed42bd87300543ad72b4bc424f7aef1c3eb125da6878fa2f5ad86a6b2c` |
| `ualbertabot9poolhydra@558899d-9poolhydra` | `ualbertabot@558899d` | 13-15-2 (43%) | 같은 파일 85행, 113행(`Zerg_2HatchHydra` 정의) | 85행 `"Zerg_ZerglingRush",` → `"Zerg_2HatchHydra",`. 113행 원래 정의(12해처리 2해처리 히드라: `["Drone" ×5, "Overlord", "Drone" ×3, "Hatchery", "Spawning Pool", …, "Hydralisk" ×12, "Hatchery", "Extractor"]`) → `["Drone" ×5, "Spawning Pool", "Drone", "Overlord", "Zergling" ×3, "Hatchery", "Extractor", "Drone" ×4, "Hydralisk Den", "Drone", "Overlord", "Drone" ×2, "Grooved_Spines", "Hydralisk" ×4]` | `a3bf7f63dcca38a5ceb6fd6a761c7671fee9357b83632be9e852368782378beb` |

- `"Zergling"` 한 항목은 저글링 한 쌍(2기)이다. 표의 `×n`은 같은 항목을 n번 쓴 것이고, 바꾼 줄 전체(들여쓰기 12칸, 항목은 `", "`로 구분)는 `d/make_variants.log`에 있다.
- 전략 이름을 바꾸는 이유(`StrategyManager.cpp` `getZergBuildOrderGoal`): `Zerg_ZerglingRush`는 오프닝 뒤 저글링 +6을 계속 만든다. `Zerg_9Pool`은 오프닝 뒤 생산 목표가 없어(5분 뒤 확장 목표 해처리 +1·드론 +10만) 오프닝의 저글링 수가 상한이 된다. `Zerg_2HatchHydra`는 오프닝 뒤 히드라 +8·Grooved Spines·드론 +4를 계속 만든다. 라운드 a의 `ualbertabot9pool`(87%)이 저글링 4기뿐이던 이유는 원래 `Zerg_9Pool` 오프닝이 익스트랙터·히드라 덴에 먼저 자원을 써서다.
- 오프닝 실측(T0 테란 standard 상대 스모크, `runs/rushladder_20261103/d/smoke/`): 10lings 스포닝 풀 0:55·첫 저글링 2:04~2:05·저글링 10기, 12lings 0:56~0:57·2:06~2:07·저글링 12기(한 판은 한 쌍을 건너뛰어 10기), 9poolhydra 0:56~0:57·2:06~2:07·저글링 6기·2번째 해처리 2:27~2:28·히드라 덴 3:32~3:35.

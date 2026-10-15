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

# T1 오픈소스 봇 포팅 조사 (021, AMB-021-1 단계적 접근)

- 일자: 2026-10-16
- 대상 환경: OpenBW BWAPI 4.2 포크, Linux GCC (Windows MSVC는 `docs/windows.md`)
- 방법: 소스를 BWAPI 4.2 헤더로 `-fsyntax-only` 컴파일해 오류를 집계했다. 실행은 하지 않았다.

## 1단계 결과: UAlbertaBot — 연동 완료
| 항목 | 내용 |
|---|---|
| 원본 | `davechurchill/ualbertabot` @ `558899d8` (BWAPI 4.4, 클라이언트 실행 파일 방식, MIT) |
| 포팅 | `third_party/patches/0005-ualbertabot-linux-bwapi42.patch` (약 220줄) |
| 포팅 내용 | MSVC 전용 문법 수정, `#include` 경로 구분자 수정, 호환 헤더(`_s` 함수 등), BOSS의 BWAPI 생성자 활성화, 설정 파일 위치를 `bwapi-data/AI/`로 이동, AI 모듈 어댑터(원래 `main.cpp`가 하던 SparCraft·BOSS 초기화 포함) |
| 빌드 | `tools/openbw/build.sh` → `tools/oss/build_ualbertabot.sh` → `<URANIUM_OSS_BOTS_DIR>/ualbertabot/module.so` |
| 설정 | 맵핵 금지(`CompleteMapInformation: false`, 021-R5) 확인 |
| 실전 | T0 테란 상대 6판(종족별 2판) 모두 완주. UAlbertaBot이 4–7분 러시로 전승 |

### 연동 중 발견해 고친 문제
1. **LAN 동기화 분기(우리 브리지 버그)**
   - 증상: 프레임 34 무렵 상대가 떨어지고 우리 쪽 승리로 기록됐다.
   - 원인: shim이 `onStart`에서 자기 클라이언트만 `setRandomSeed()`로 다시 시드했다. 그래서 다른 모듈과의 대전에서 시뮬레이션 난수가 갈라졌다.
   - 해결: 시드는 OpenBW 패치 0002가 모든 클라이언트에 적용하므로 해당 호출을 제거했다.
   - 브리지끼리의 대전은 양쪽이 같은 값으로 재시드하기 때문에 문제가 드러나지 않았었다.
2. **BOSS 크래시(포팅 누락)**
   - 원인: 클라이언트 실행 파일의 `main()`이 하던 `SparCraft::init()`과 `BOSS::init()`이 AI 모듈 방식에서 빠졌다.
   - 해결: 어댑터에서 1회 호출한다.
3. **BOSS 독립 실행 파일 소스(`BOSS_main.cpp`)**: X11 GUI 의존이 있어 빌드에서 제외했다.

## 2단계 사전 조사: BWAPI 4.4 봇 포팅 비용
| 봇 | 출처 | 종족 | 원 컴파일 오류 | 기계적 처리 후* 남은 오류 위치 | 남은 문제 유형 | 예상 작업 |
|---|---|---|---|---|---|---|
| Steamhammer 5.3.6 | 공식 사이트 소스 zip (MIT, UAlbertaBot 계열) | Z 주력, T/P 일부 | 101 (106 파일 중 82개 실패) | **3곳** + 사용 안 하는 파일 1개 | rapidjson 경로 구분자, `windows.h`(Dll.cpp), 변형 파일 `OpsBoss with immobile.cpp` 제외 | 작음 (UAlbertaBot과 같은 패턴) |
| McRave | `Cmccrave/McRave` (BWEM·BWEB·Horizon 포함) | P/Z 주력 | 1,387 (114 파일 중 105개 실패) | **58곳** | MSVC가 허용하는 삼항 연산자 타입 불일치(헤더 4곳에 집중), 임시 객체의 비 const 참조 바인딩, 람다 capture-default | 중간 |

\* 기계적 처리: `-std=c++20 -fpermissive`, `<climits>`·`<cfloat>`·`<cmath>` 강제 포함, UAlbertaBot 호환 헤더 재사용.

- 두 봇 모두 BWAPI 4.4 전용 API로 인한 컴파일 오류는 없었다. 다만 런타임 동작 차이는 실제 대전으로 확인해야 한다.
- AMB-021-1 결정에 따라 **실제 포팅 진행 여부는 사용자 재확인 대상**이다.
- 권장안(미확정): Steamhammer를 먼저 포팅한다. 비용이 작고, 저그 주력이라 TvZ 상대 다양성(AMB-042-2)에 도움이 된다.

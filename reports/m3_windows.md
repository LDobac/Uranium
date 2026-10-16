# M3 Windows 검증 리포트 — 단일 PC 구성 (011-AC8, 001-R8)

- 일자: 2026-10-16
- 장비: Windows 11, 16 논리 코어, Visual Studio 2026(MSVC 14.51), CMake 4.4, Rust stable-msvc
- 구성: 게임·브리지·hostd·에이전트를 PC 한 대에서 실행했다(`docs/windows.md` 구성 (a)).
- 원본 결과: 세션 임시 폴더에 있다(git 미추적). 이 리포트의 수치는 각 run의 `summary.md`와 `results.jsonl`에서 옮겼다.

## 요약
| 기준 | 결과 | 판정 |
|---|---|---|
| 001-R8: 학습 쪽 crate Windows 빌드 | `cargo test --workspace` 통과. hostd 테스트 1건의 경로 구분자 가정만 고쳤다 | **통과** |
| 011-AC8 / 011-AC1: 두 맵 각 100판 완주 | 200/200 완주, Aborted 0, 세션 오류 0 | **통과** |
| 011-AC8 / 011-AC4: 같은 seed 결정론 | `smoke-tvt` 20판을 두 번 실행. 40세션의 승자·종료 프레임·프레임 다이제스트 일치(브리지 수정 후, 아래 2절) | **통과** |
| 010-AC6: 시야 위반 0 | 모든 run에서 0 | **통과** |
| 011-AC2: 브리지 p99 ≤ 0.3ms | 동시 16게임까지 세션별 p99 중앙값 ≤ 220µs | **통과** |
| 020-AC3: hostd 강제 종료 뒤 고아 프로세스 없음 | 런처 14개 실행 중 `taskkill /F` → 2초 뒤 0개 | **통과** |
| 020-AC5: calibrate 측정 파일 생성·적용 | 권장 16게임 저장, 재시작한 hostd가 16을 사용 | **통과** |
| 021-AC2: T1 봇 6종 대 T0 | 110/110 완주(McRave 수정 후, 아래 3절) | **통과** |
| 구성 (b): Windows 스케줄러 + 원격 Linux hostd | 원격 게임 PC 필요 | **미검증** |

## 1. 빌드 절차의 실제 차이
| 항목 | Linux | Windows |
|---|---|---|
| OpenBW/BWAPI | `build.sh`, Makefile | `build.ps1`, VS 생성기(멀티 config) → 산출물 `bin\Release\` |
| 브리지 | `liburanium_bridge.so`, `visibility` + `--exclude-libs` | `uranium_bridge.dll`, `__declspec(dllexport)`. BWAPILIB이 정적 라이브러리라 Rust staticlib와 함께 `/MD`로 링크 |
| LAN 매칭 | LOCAL 소켓 | TCP 포트 쌍(루프백), LISTEN 테이블(`GetExtendedTcpTable`)로 슬롯 0 대기 |
| 프로세스 정리 | 프로세스 그룹 + `PR_SET_PDEATHSIG` | Job Object(kill-on-close), `CREATE_NO_WINDOW` |
| T1 봇 | `g++` 직접 호출 | 봇마다 CMake 프로젝트를 생성해 MSVC로 빌드, 진입점은 패치의 `UraniumModule.cpp` + `.def` |
| `.env` | 따옴표 없이 경로 | Windows 경로는 작은따옴표 필수(dotenvy가 `\`를 이스케이프로 처리) |

이 과정에서 발견해 고친 것:
- **패치 줄바꿈**: `core.autocrlf=true` 환경에서는 `third_party/patches/*.patch`가 CRLF로 체크아웃되어 적용에 실패했다. `.gitattributes`(`*.patch -text`, `*.sh eol=lf`)를 추가했다.
- **casc 설치**: libclang이 필요하다. 관리자 권한 없이 PyPI 휠의 `libclang.dll`로 해결했다.
- **MSVC 차이(T1 봇)**: `std::random_shuffle` 제거(`_HAS_AUTO_PTR_ETC=1`), `min`/`max` 매크로(`NOMINMAX`), `ssize_t` 없음(Stardust), `<stdexcept>` 암묵 포함 차이(McRave).
- **Stardust 헤더 충돌**: 대소문자를 구분하지 않는 파일 시스템에서 BWEM의 `map.h`가 Stardust의 `Map.h`를 가렸다. include 순서를 바꿔 해결했다.

## 2. 결정론 버그 수정 (011-R15, 011-AC4)
- **증상**: 같은 seed로 두 번 돌리면 승자와 종료 프레임은 같았지만 `frame_digest`는 달랐다.
- **원인**:
  - 첫 프레임에서 시작 유닛의 `getReplayID()`가 0이라 브리지의 `id` 정렬에서 동률이 생겼다.
  - 동률 유닛의 순서는 BWAPI 집합의 포인터 순서를 따른다. Windows에서는 이 순서가 ASLR 때문에 실행마다 다르다.
- **수정**: `crates/uranium-bridge-openbw/src/lib.rs`에서 정렬 키를 `(id, owner, unit_type, pos)`로 바꿨다.
- **결과**: 두 번 실행한 40세션의 다이제스트가 모두 일치했다.
- **영향**:
  - Linux에서도 첫 프레임 관측의 유닛 순서가 바뀔 수 있다. 따라서 이 수정 전 Linux run의 다이제스트와는 비교할 수 없다.
  - Linux와 Windows 사이의 바이트 비교는 참고용이며 하지 않았다.

## 3. McRave 크래시 수정 (021-AC2)
- **증상**: 첫 `m3-t1` 실행은 108/110 완주였다. McRave(저그)가 투혼에서 시작 직후 `0xC0000005`로 종료했고, 8/8 재현됐다.
- **원인**(WinDbg `cdbX64` + `module.pdb`로 확인):
  - BWEB `Blocks::initialize`가 맵 밖에 있는 station 방어 타일을 `addToBlockGrid`로 경계 검사 없이 `blockGrid[256][256]`에 썼다.
  - MSVC 배치에서는 이 쓰기가 바로 앞의 전역 `allBlocks` 벡터를 덮어썼다.
  - Linux(GCC)에서는 덮어쓴 곳이 드러나지 않았을 뿐 같은 미정의 동작이다.
- **수정**: 패치 0009에 경계 검사 hunk를 추가했다.
- **결과**: 같은 조건 8/8 완주, `m3-t1` 전체 110/110 완주. 결과는 4절에 있다.

## 4. 검증 결과
### 4.1 011-AC1 — T0 테란 미러, 맵당 100판
- 조건: 200판, hostd 1개, 동시 16게임, 벽시계 180초
- 결과:
  - 완주 200/200, 시야 위반 0, 세션 오류 0
  - 무승부: 서킷브레이커 7판, 투혼 3판
- 성능:
  - 판당 벽시계: p50 3.8초, p90 9.7초
  - 종료 프레임 p50: 11,814
  - 브리지 p99(세션별): 중앙값 174µs, p90 196µs, 최대 227µs

### 4.2 smoke
- `smoke-tvt` 20/20 완주
- `smoke-cross` 12/12 완주
- `smoke-uab` 6/6 완주
- 모두 시야 위반 0

### 4.3 `uranium-env calibrate` (8000프레임/게임, 단계마다 2N판)
| 동시 게임 | 브리지 p99 중앙값 (µs) | 최대 (µs) | 엔진 fps p50 | 처리량 (프레임/초) | 통과 |
|---|---|---|---|---|---|
| 1 | 110 | 120 | 5,181 | 3,158 | ○ |
| 4 | 116 | 137 | 4,605 | 11,398 | ○ |
| 8 | 167 | 196 | 3,564 | 18,029 | ○ |
| 10 | 191 | 248 | 3,133 | 18,457 | ○ |
| 12 | 200 | 442 | 2,883 | 16,582 | ○ |
| 16 | 220 | 384 | 2,281 | 17,792 | ○ |

- 권장값은 16게임이다(측정 상한 = 코어 수).
- 처리량은 8–10게임에서 약 18k 프레임/초로 포화한다. 이후 늘린 게임 수는 지연만 늘린다.
- 엔진 속도는 BC250(Linux)의 약 2배다. 단일 게임 기준 5,181 대 2,667 fps이다.

### 4.4 021-AC2 — T1 봇 6종 (`config/plans/m3-t1.toml`)
- 조건: 110판, 동시 16게임, 벽시계 235초
- 결과: 110/110 완주, 의심 게임 0, 시야 위반 0, 무승부 2
- T0 전적: UAlbertaBot 상대 3/30승, Steamhammer 상대 5/30승, 나머지 봇 상대 0승이다. Linux 결과와 경향이 같다.

## 5. 남은 일
- 구성 (b)와 020-AC1: 원격 Linux 게임 PC가 필요하다.
- `config/assets.toml`의 `manifest_sha256`은 호스트마다 재현되지 않는다.
  - `manifest.toml`에 추출 원본의 절대 경로(`source`)가 들어가기 때문이다.
  - 이 PC 추출본도 파일 수(956)는 같지만 해시는 다르다.
  - 경로를 뺀 내용 해시로 고정할지 사용자 결정이 필요하다.

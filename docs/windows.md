# Windows 작업 체크리스트 (M3, 001-R8 · 011-R16 · 020-R8)

이 문서는 Windows PC 세션에서 이어서 할 작업 목록이다.
- Linux(BC250) 세션에서는 Windows 빌드·테스트를 하지 않았다(사용자 결정).
- 아래 항목은 **모두 미검증**이다. Windows에서 실제로 해 보면서 고치고, 체크한다.
- 목표 구성은 두 가지다.
  - (a) Windows PC 한 대에서 게임·브리지·hostd·에이전트·학습을 모두 실행한다. lockstep 통신이 VPN을 타지 않게 하기 위해서다.
  - (b) 원격 Linux 게임 PC를 쓰는 기존 구성을 유지한다.
- 개인·보안 정보(주소, 호스트 이름, 계정, 개인 경로)는 문서와 커밋에 쓰지 않는다(001-R10). 실제 값은 `.env`에만 둔다.

## 1. 사전 준비
- [ ] Visual Studio 2022 C++ 도구(데스크톱 C++ 워크로드), CMake 3.16 이상, Git
- [ ] Rust MSVC 툴체인(`rustup default stable-x86_64-pc-windows-msvc`)
- [ ] `git clone` 후 `cargo test --workspace` 통과
  - 학습 쪽 crate는 원래 Windows 빌드 대상이다(001-R8).
  - hostd의 `#[cfg(windows)]` 자리표시자는 컴파일만 되면 된다.
- [ ] SC:R 데이터 추출
  - `cargo install casc --version 0.1.6 --locked`
  - `cargo run -p casc-extract -- <SC:R>\Data <data dir>`
- [ ] 맵 파일 `fighting_spirit_1.3.scx`, `circuit_breaker_1.0.scx`를 `URANIUM_MAPS_DIR`에 둔다.
- [ ] `.env.example`을 `.env`로 복사하고 값을 채운다.
  - 경로는 절대 경로로 쓴다.
  - `URANIUM_HOSTD_PORT_BASE`를 정한다(아래 4절).
- [ ] `tools/env.ps1` 작성: `tools/env.sh`와 같은 규칙의 `.env` 로더
  - 규칙: 프로세스 환경변수가 우선한다. 따옴표는 한 겹만 벗긴다. 전체 줄 `#` 주석만 지원한다. `~`/`$VAR`는 확장하지 않는다.

## 2. OpenBW + BWAPI MSVC 빌드 (`tools/openbw/build.ps1`)
- [ ] `build.sh`와 같은 커밋으로 고정해 clone한다.
  - OpenBW `4b046d5f65302b10cb0a745f0fecd37ec85b20a8`
  - BWAPI `48124ba8ed1b4d52b3dfd52acbaf34afb9a37fe2`
- [ ] 패치를 `git apply`로 적용한다. 이미 적용된 패치는 `--reverse --check`로 건너뛴다.
  - `0001-openbw-loose-data-files`
  - `0002-openbw-fixed-start-seed`
  - `0004-openbw-deterministic-client-uid`
  - `0003-bwapi-launcher-quick-exit`
  - 줄바꿈 때문에 실패하면 `git config core.autocrlf false`로 다시 clone한다.
- [ ] `cmake -S third_party/bwapi -B third_party/bwapi/build -G "Visual Studio 17 2022" -A x64 -DOPENBW_DIR=<abs>/third_party/openbw -DOPENBW_ENABLE_UI=0 -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_CXX_FLAGS="/FIcstdint /EHsc"`
- [ ] `cmake --build third_party/bwapi/build --config Release`
- [ ] 산출물 위치를 확인하고 hostd 설정과 맞춘다.
  - 대상: `BWAPILauncher.exe`, `BWAPILIB.lib`, OpenBW `BWAPI.dll`
  - hostd는 `<URANIUM_BWAPI_DIR>\bin\BWAPILauncher.exe`를 찾는다(`crates/uranium-hostd/src/config.rs`). 위치가 다르면 설정 코드를 고친다.
- [ ] 확인할 점
  - 패치 0003의 `std::_Exit`는 MSVC에서도 쓸 수 있다.
  - 패치 0001의 폴더 로더 경로 처리(`/`·`\`)를 확인한다.

## 3. 브리지 Windows 대응 (`bridge/openbw`, `tools/openbw/build_bridge.ps1`)
- [ ] `shim.cpp`의 `UR_EXPORT`를 플랫폼 매크로로 바꾼다.
  - MSVC: `extern "C" __declspec(dllexport)`
  - GCC: 기존 `visibility("default")`
  - shim 500줄 제한을 유지한다.
- [ ] `CMakeLists.txt`에 MSVC 분기를 추가한다.
  - 컴파일 옵션: `/FIcstdint /W3` (GCC 옵션 `-include cstdint -Wall`은 GCC 분기로 옮긴다)
  - 링크: `RUST_STATICLIB` = `target\release\uranium_bridge_openbw.lib` + `ws2_32 userenv ntdll bcrypt advapi32`
  - `pthread dl m`, `-Wl,--exclude-libs,ALL`은 GCC 분기에만 둔다. MSVC는 dllexport한 심볼만 내보내므로 따로 처리할 필요가 없다.
  - `PREFIX "lib"`는 GCC 분기에만 둔다. Windows 산출물은 `uranium_bridge.dll`이다.
- [ ] Rust staticlib의 CRT(`/MD`)와 BWAPI 빌드의 CRT가 같은지 확인한다. 다르면 LNK2038 오류가 난다.
- [ ] `build_bridge.ps1` 작성: `cargo build --release -p uranium-bridge-openbw` → cmake configure/build(Release)
- [ ] 확인: `dumpbin /exports uranium_bridge.dll`에 `gameInit`과 `newAIModule`만 있어야 한다.

## 4. hostd Windows `Platform` (`crates/uranium-hostd/src/platform.rs`)
현재 `WindowsPlatform::new()`는 `bail!`만 하는 자리표시자다. 다음을 구현한다.

- [ ] **LAN: TCP 포트 쌍 (011-R16)**
  - Windows OpenBW는 `TCP` 모드만 지원한다(`BWData.cpp`의 `server_n == 1`). 각 클라이언트가 `OPENBW_TCP_LISTEN_PORT`를 bind하고 상대 `OPENBW_TCP_CONNECT_PORT`로 connect한다.
  - 게임마다 포트 2개(p, p+1)를 `URANIUM_HOSTD_PORT_BASE`부터 풀에서 할당하고, 게임이 끝나면 반환한다. 할당 범위: `URANIUM_HOSTD_MAX_GAMES × 2`
  - 슬롯 0 env: `OPENBW_LAN_MODE=TCP`, `OPENBW_TCP_LISTEN_HOSTNAME=127.0.0.1`, `OPENBW_TCP_LISTEN_PORT=p`, `OPENBW_TCP_CONNECT_HOSTNAME=127.0.0.1`, `OPENBW_TCP_CONNECT_PORT=p+1`
  - 슬롯 1 env: 포트를 반대로 지정한다.
  - listen 주소는 루프백으로 둔다. 게임은 한 PC 안에서만 짝지어진다.
  - 할당 시 다른 프로그램이 이미 쓰는 포트는 bind 시험으로 건너뛴다.
  - `LanCleanup::Ports(p)`를 쓰고 `release`에서 반환한다.
  - 단위 테스트: 포트 풀 할당·반환·고갈
- [ ] **`wait_first_ready`**: 슬롯 0이 포트 p를 listen할 때까지 대기한다(011-AC4, 슬롯 순서 결정론).
  - 방법 후보: connect 시도. 단, 이 연결이 OpenBW 핸드셰이크를 방해하지 않는지 확인해야 한다. 방해한다면 `GetExtendedTcpTable`로 LISTEN 상태를 조회한다.
  - 패치 0004(`URANIUM_LAN_SLOT`)가 TCP 모드에서도 결정론적 uid를 주는지 확인한다.
- [ ] **프로세스 정리: Job Object (020-AC3)**
  - `windows-sys`로 구현한다: `CreateJobObjectW` + `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, `AssignProcessToJobObject`
  - 게임마다 job 하나. job 핸들을 hostd가 들고 있으므로, hostd가 강제 종료되면 핸들이 닫히면서 런처도 함께 죽는다.
  - `kill_tree`: `TerminateJobObject` 후 `child.wait()`
  - 생성 직후부터 job 할당 전 사이의 경쟁 조건은 `CREATE_SUSPENDED`로 막는다. `std::process`에서는 쉽지 않으므로, 런처가 자식 프로세스를 만들지 않는다는 점을 확인하고 그대로 할당해도 된다.
  - 콘솔 창이 뜨면 `CREATE_NO_WINDOW`를 쓴다.
- [ ] OSS 봇 레지스트리: `<URANIUM_OSS_BOTS_DIR>\<name>\module.dll` (`config.rs`에 반영돼 있음)

## 5. UAlbertaBot Windows 빌드 (`tools/oss/build_ualbertabot.ps1`)
- [ ] 같은 커밋 `558899d8793456f4a6ec4196efbb5235552e24db`에 패치 0005를 적용한다.
  - 원본은 MSVC 프로젝트이므로 패치의 `linux/compat.h`는 필요 없을 수 있다.
  - 단, BWAPI 4.4 → 4.2 차이(`URANIUM_BOSS_BWAPI`, 설정 경로 `bwapi-data/AI`, `UraniumModule.cpp` 어댑터)는 그대로 필요하다.
- [ ] Linux와 같은 파일 목록으로 DLL을 빌드한다.
  - 제외: `main.cpp`, `BOSS_main.cpp`, `StarCraftGUI.cpp`, `TutorialCode.cpp`, `UnitInfoManager.cpp`
  - `newAIModule`/`gameInit`을 export한다.
  - 출력: `<URANIUM_OSS_BOTS_DIR>\ualbertabot\module.dll` + `bwapi-data\AI\UAlbertaBot_Config.txt`

## 6. 검증
hostd + `uranium-env run --plan config/plans/<plan>.toml`로 진행한다.
- [ ] 스모크: `smoke-tvt`, `smoke-cross`, `smoke-uab`
- [ ] 011-AC1: 두 맵 각 100판 완주
- [ ] 011-AC4: 같은 seed 결정론. 결과 JSONL을 Linux 결과와 비교해 본다. 바이트 단위로 같아야 하는 것은 Windows 안에서만이고, Linux와의 비교는 참고용이다.
- [ ] 010-AC6: 시야 위반 0
- [ ] 020-AC3: 작업 관리자나 `taskkill /F`로 hostd를 강제 종료한 뒤 `BWAPILauncher.exe`가 0개
- [ ] 021-AC2: UAlbertaBot 대 T0 각 종족 10판
- [ ] 구성 (b): Windows에서 `uranium-env`를 실행하고, VPN 너머 Linux 게임 PC의 hostd를 쓴다.
  - 호스트 목록은 커밋하지 않는 `URANIUM_HOSTS_FILE`에 둔다.
  - `URANIUM_AGENT_LISTEN`은 게임 PC에서 접근 가능한 주소여야 한다.
  - 판단당 지연(브리지 RTT)을 기록한다.

## 7. 마무리
- [ ] 결과를 `reports/m3_windows.md`에 쓴다. 빌드 절차의 실제 차이, 판정, 성능(벽시계 판당 시간, 브리지 오버헤드 p99)을 포함한다.
- [ ] spec 011-AC8 상태를 갱신하고, `spec/README.md`와 CLAUDE.md 명령어 절에 Windows 명령을 추가한다.
- [ ] 이 체크리스트에서 끝난 항목을 체크하거나, 문서를 정식 Windows 빌드 안내로 바꾼다.

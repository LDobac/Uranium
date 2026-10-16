# Windows 빌드·실행 안내 (001-R8 · 011-R16 · 020-R8)

Windows PC 한 대에서 게임·브리지·hostd·에이전트를 모두 돌리는 절차다. 검증 결과는 `reports/m3_windows.md`에 있다.
- 개인·보안 정보(주소, 호스트 이름, 계정, 개인 경로)는 문서와 커밋에 쓰지 않는다(001-R10). 실제 값은 `.env`에만 둔다.
- 명령은 PowerShell 기준이다. 스크립트는 `powershell -ExecutionPolicy Bypass -File <script>`로 실행한다.

## 1. 사전 준비
- Visual Studio 2026(또는 2022)의 "C++를 사용한 데스크톱 개발" 워크로드, CMake 3.16 이상, Git
  - 빌드 스크립트의 CMake 생성기 기본값은 `Visual Studio 18 2026`이다. VS 2022에서는 `URANIUM_CMAKE_GENERATOR="Visual Studio 17 2022"`로 바꾼다.
- Rust MSVC 툴체인: `rustup default stable-x86_64-pc-windows-msvc`
- 저장소의 `.gitattributes`가 패치(`*.patch`)를 줄바꿈 변환 없이 체크아웃한다. 이 파일이 생기기 전에 clone했다면 패치를 다시 체크아웃한다.
  - 예: `Remove-Item third_party\patches\*.patch; git checkout -- third_party/patches`
- `cargo test --workspace`
- `.env`: `.env.example`을 복사해 값을 채운다.
  - Windows 경로는 **작은따옴표**로 감싼다. 예: `URANIUM_DATA_DIR='D:\sc_data\openbw'`. Rust 로더(dotenvy)가 따옴표 없는 `\`를 이스케이프로 읽기 때문이다.
  - 셸 스크립트용 로더는 `tools\env.ps1`이다(`. tools\env.ps1; Import-UraniumEnv <repo>`).

## 2. 게임 데이터
- `casc` CLI 설치에 libclang이 필요하다.
  - LLVM이 있으면 그대로 설치한다(`cargo install casc --version 0.1.6 --locked`).
  - LLVM이 없고 관리자 권한도 없으면 PyPI `libclang` 휠에서 `libclang.dll`을 꺼내 쓴다. 휠의 sha256은 PyPI 값과 대조한다. 그다음 `LIBCLANG_PATH`를 그 폴더로 지정하고 설치한다.
- 추출: `cargo run -p casc-extract --release -- "<SC:R>\Data" <data dir>` → 956파일
  - 추출이 끝나면 `content_sha256`이 출력된다. 이 값을 `config/assets.toml`의 값과 대조한다(011-R13, 경로에 무관한 내용 해시).
- 맵: SC:R 설치 폴더의 `Maps\ladder`에서 복사하고 `config/assets.toml`의 sha256과 대조한다.
  - `2024Season1\(4)Fighting_Spirit_1.3.scx` → `fighting_spirit_1.3.scx`
  - `Season 10\(4)Circuit Breaker.scx` → `circuit_breaker_1.0.scx`

## 3. 빌드
| 대상 | 명령 | 산출물 |
|---|---|---|
| OpenBW + BWAPI | `tools\openbw\build.ps1` | `third_party\bwapi\build\bin\Release\{BWAPILauncher.exe, BWAPI.dll, OpenBWData.dll}` |
| 브리지 | `tools\openbw\build_bridge.ps1` | `bridge\openbw\build\Release\uranium_bridge.dll` |
| T1 봇 | `tools\oss\build_<bot>.ps1` | `<URANIUM_OSS_BOTS_DIR>\<bot>\{module.dll, module.pdb, VERSION, bwapi-data\}` |
| Rust 바이너리 | `cargo build --release -p uranium-hostd -p uranium-env -p uranium-agent` | `target\release\*.exe` |

- `URANIUM_BWAPI_DIR`는 `third_party\bwapi\build`이다. hostd는 Windows에서 `bin\Release\BWAPILauncher.exe`를 찾는다.
- 브리지는 `gameInit`과 `newAIModule`만 export한다(`dumpbin /exports`로 확인).
- T1 봇은 Linux와 같은 고정 커밋·패치(0005–0010)를 쓴다.
  - `tools\oss\common.ps1`이 봇마다 CMake 프로젝트를 `build\obj\<bot>`에 만들어 MSVC로 빌드한다.
  - 진입점은 패치의 `linux\UraniumModule.cpp`를 Windows에서도 쓰고, `module.def`로 export한다. Stardust만 원본 진입점을 쓴다.
  - 봇별 컴파일 차이는 각 `build_<bot>.ps1`에 있다.

## 4. 실행
- hostd: `target\release\uranium-hostd.exe serve`
  - LAN 매칭은 TCP 포트 쌍이다(011-R16). 게임마다 `URANIUM_HOSTD_PORT_BASE`(기본 17000)부터 포트 2개를 루프백에 할당한다.
  - 다른 프로그램이 LISTEN 중인 포트는 건너뛴다.
  - 런처는 게임마다 Job Object(kill-on-close)에 들어간다. hostd가 강제 종료되면 런처도 함께 죽는다(020-AC3).
- 스케줄러: `target\release\uranium-env.exe run --plan config\plans\<plan>.toml --out runs\<dir>`
- 동시 게임 수 측정: `target\release\uranium-env.exe calibrate` → `<URANIUM_CACHE_DIR>\hostd_calibration.toml`
- 봇 크래시 분석: WinDbg(`winget install Microsoft.WinDbg`)의 `cdbX64`에 `module.pdb` 경로를 `-y`로 준다.

## 5. 구성 (b): 이 PC + 원격 Linux 게임 PC (020 호스트 구성)
학습 PC(Windows)가 `uranium-env`(에이전트 서버 내장)와 로컬 hostd를 돌린다. 원격 Linux 게임 PC는 hostd만 돌린다. 원격 게임의 브리지는 학습 PC의 에이전트에 접속한다. 검증 결과는 `reports/m3_remote.md`에 있다.

**원격 Linux 게임 PC**
1. 저장소를 받아 최신으로 맞춘다: `git pull --ff-only`
2. 빌드한다: `tools/openbw/build.sh` → `tools/openbw/build_bridge.sh` → `cargo build --release -p uranium-hostd -p uranium-env -p uranium-agent` → `tools/oss/build_<bot>.sh`(6종)
   - 패치가 바뀐 봇 소스는 먼저 `git -C third_party/<bot> reset --hard && git -C third_party/<bot> clean -fdx`로 되돌린다. 이전 패치가 적용된 상태에서는 새 패치가 들어가지 않는다.
3. 게임 데이터(`URANIUM_DATA_DIR`, 맵)를 복사하고 `config/assets.toml`의 `content_sha256`과 맵 sha256으로 검증한다(011-R13).
4. `.env`를 설정한다: `URANIUM_DATA_DIR`, `URANIUM_MAPS_DIR`, `URANIUM_CACHE_DIR`, `URANIUM_BWAPI_DIR`, `URANIUM_BRIDGE_LIB`, `URANIUM_OSS_BOTS_DIR`. `URANIUM_HOSTD_LISTEN`에는 그 PC의 LAN 주소를 `<addr>:7700` 형식으로 넣는다.
5. `target/release/uranium-env calibrate`를 한 번 실행한 뒤 hostd를 띄운다: `setsid nohup target/release/uranium-hostd serve > /tmp/hostd.log 2>&1 < /dev/null &`
   - 원격에서 `pkill -f uranium-hostd`를 쓰면 SSH 명령 자신도 매칭되어 함께 죽는다. `pkill -x uranium-hostd`를 쓴다.

**학습 PC (Windows)**
1. `config/hosts.toml`(git 미추적)에 로컬 hostd(`127.0.0.1:7700`)와 원격 hostd 주소를 등록한다. 형식은 `config/hosts.example.toml`을 따른다.
2. `.env`를 설정한다.
   - `URANIUM_HOSTS_FILE`: hosts 파일 경로
   - `URANIUM_AGENT_LISTEN=0.0.0.0:7600`
   - `URANIUM_AGENT_PUBLIC_ADDR`: 원격 PC에서 닿는 이 PC의 주소(VPN 주소 등)와 포트(`:7600`)
   - 이 주소는 로컬 게임도 쓴다. 따라서 VPN이 꺼지면 로컬 게임도 접속하지 못한다.
3. 방화벽에 들어오는 TCP 7600 허용 규칙을 추가한다. 관리자 권한이 필요하다.
   - 예: `New-NetFirewallRule -DisplayName 'Uranium agent' -Direction Inbound -Action Allow -Protocol TCP -LocalPort 7600 -Program <repo>\target\release\uranium-env.exe -Profile Any`
4. 실행: `target\release\uranium-env.exe run --plan config\plans\m3-remote.toml`
   - 응답하지 않는 hostd는 경고 후 빼고 진행한다.
   - 도중에 끊긴 hostd에서 돌던 게임은 Aborted로 기록한다(020-R4).
   - PC를 추가하거나 빼려면 run을 멈추고 hosts 파일을 고친 뒤 다시 시작한다.

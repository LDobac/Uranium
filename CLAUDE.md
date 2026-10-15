# CLAUDE.md — Uranium

스타크래프트1 **테란 밀리 AI 봇**을 구현하고 직접 강화학습하는 프로젝트다.
- 학습: OpenBW(1.16.1)
- 최종 실행: StarCraft: Remastered(SC:R)
- 사용자와의 대화와 문서는 **한국어**로 작성한다. 코드 식별자와 주석은 영어로 쓴다.

## 고정 조건
| 항목 | 값 |
|---|---|
| 플레이 종족 | Terran (고정) |
| 상대 종족 | T / P / Z |
| 맵 | 투혼(Fighting Spirit), 서킷브레이커(Circuit Breaker) |
| 학습 장비 | AMD BC250(Vulkan) 또는 CUDA GPU PC(Linux/Windows). CUDA는 Burn `cuda` feature로만 쓰고(해당 PC에서만 빌드), 직접 작성한 CUDA·ROCm·HIP 코드는 금지 |
| 언어/프레임워크 | Rust + Burn **0.22.0** (런타임 `Device`: `Device::vulkan` / `Device::cuda`). 테스트용 CPU 백엔드는 `flex`. 학습·추론은 f16 혼합 정밀도 (032-R8). 학습 쪽 crate는 Windows에서도 빌드돼야 함 (001-R8) |
| 시뮬레이터 | OpenBW + BWAPI 4.2 포크, Linux 헤드리스. 같은 LAN의 다른 PC에서 실행될 수 있음 |
| SC:R 연동 | samase 플러그인 DLL + samase_scarf (Windows x86_64). **로컬/LAN 커스텀 게임에서만** 실행하며 배틀넷 공개 게임과 래더는 대상이 아님 |
| 체크포인트·데이터 | 네트워크 공용 드라이브(SMB, LAN 또는 VPN 경유, `URANIUM_SHARED_DIR`). 학습 PC는 pull → 학습 → push, run당 한 PC만(heartbeat 잠금) (061) |
| C++ 사용 | OpenBW 브리지의 얇은 C++ shim(BWAPI ↔ C ABI 중계)만 허용. 그 외 영역은 Rust |

## Spec Driven 개발 규칙 (필수)
1. **`spec/`이 단일 진실 공급원이다.** 작업 전에 `spec/README.md`와 관련 spec을 먼저 읽는다.
2. **spec 없이 구현하지 않는다.** 해당 spec이 `Approved`가 아니면 구현하지 말고 사용자에게 알린다.
3. 동작을 바꿀 때 순서는 **spec 수정 → 사용자 확인 → 코드**다. 코드가 spec과 다르면 코드를 고치거나, spec 변경을 사용자에게 제안한다.
4. 코드 주석, 테스트 이름, 커밋 메시지에는 요구사항 ID(`010-R3`, `011-AC1` 등)를 참조한다.
5. 새 spec은 공통 템플릿을 따르고 `spec/README.md` 인덱스에 추가한다. spec 번호는 **구현 순서가 아니라 영역 대역**(00x 개요, 01x 게임 연동, 02x 실행 환경, 03x 모델, 04x 데이터·학습, 05x 평가, 06x 인프라)이고, 한 번 부여한 번호는 바꾸지 않는다. 구현 순서는 마일스톤(M0~M7)을 따른다.
   - 공통 템플릿: `Status / Purpose / Scope & Non-goals / Requirements / Interfaces / Acceptance Criteria / Risks & Open Questions / Related`
6. 큰 작업(새 crate, 아키텍처 변경, 여러 spec에 걸친 변경)은 Plan 모드로 계획을 먼저 세우고 승인을 받는다.

## 모호함 처리 규칙 (필수)
- spec에서 사용자 결정이 필요한 부분은 **`[SPEC AMBIGUOUS] AMB-<spec번호>-<n>`** 으로 표기하고, `spec/README.md`의 Ambiguity Registry에 등록한다.
- **모호하거나 애매한 부분은 추측으로 진행하지 않는다. 항상 사용자에게 되물어 컨펌을 받는다.**
  - 기본안이 있더라도 "기본안(미확정)"으로 제시만 하고 확인을 기다린다.
  - 작업 중 새로 발견한 모호함도 같은 방식으로 처리한다: spec에 태그 추가 → 레지스트리 등록 → 사용자에게 질문.
  - 사용자가 지정하지 않은 수치, 정책, 범위를 임의로 확정하지 않는다. 예: 승률 목표, APM 제한, 데이터 출처, 실행 환경.
- `[SPEC AMBIGUOUS]` 항목에 의존하는 코드는 작성하지 않는다. 의존하지 않는 부분만 진행한다.
- **보류 항목**: 사용자가 결정 시점을 뒤 마일스톤으로 미루기로 확인한 항목은 `[SPEC DEFERRED(Mx)]`로 표기한다.
  - 보류 항목은 spec 승인을 막지 않는다.
  - 의존 코드는 여전히 작성할 수 없다.
  - 해당 마일스톤을 시작하기 전에 반드시 다시 질문한다.
  - 현재 보류 중: 리마스터 관련 AMB-001-1, AMB-012-1, AMB-012-2 (M7), TvP·TvZ 상대 다양성 AMB-042-2 (M6).
- 사용자가 확정하면 처리 순서는 다음과 같다.
  1. 본문에 결정을 반영한다.
  2. 태그를 제거한다.
  3. 레지스트리를 `Resolved (YYYY-MM-DD: 결정 내용)`으로 갱신한다.
- 구분: `TBD(Mx)`는 **측정만 하면 정해지는 값**이고, `[SPEC AMBIGUOUS]`는 **사용자 판단이 필요한 값**이다. 둘을 혼동하지 않는다.

## 아키텍처 원칙
- **에이전트 코어는 게임 버전을 모른다** (001-R1). 버전 의존 코드는 `bridge/*`와 `uranium-adapter`(013 매핑 테이블)에만 둔다.
- 브리지는 **얇게** 유지한다 (001-R3). 브리지가 하는 일은 상태 직렬화, 명령 실행, 시야 필터링뿐이고 의사결정 로직은 넣지 않는다.
- 브리지와 에이전트는 `uranium-proto` 프로토콜(TCP, postcard)로만 통신한다 (010).
- **맵핵 금지**: 관측은 자기 플레이어 시야 기준이다 (010-R5, 011-R5, 012-R3, 030-R2).
- SC:R 명령은 네트워크 커맨드 경로로만 보낸다. 유닛 메모리를 직접 수정하지 않는다 (012-R4).
- 학습은 lockstep, SC:R 실전은 realtime 모드로 동작한다 (001-R5).
- 맵 고정은 RL·평가 범위에만 적용한다. BC 사전학습은 전체 맵 리플레이로 하고 대상 맵으로 파인튜닝한다 (041).

## 저장소 구조 (계획, 001 참조)
```
spec/                 # 스펙 문서 (먼저 읽을 것)
crates/
  uranium-proto       # 공통 스키마·프레이밍
  uranium-adapter     # GameBackend trait, 버전 매핑 테이블
  uranium-env         # 에피소드·feature·보상·executor
  uranium-model       # Burn 모델
  uranium-train       # BC / RL / league
  uranium-replay      # .rep 파서·데이터셋
  uranium-hostd       # OpenBW 인스턴스 데몬
  uranium-agent       # 에이전트 서버(정책 세션)·실전 바이너리
  uranium-bridge-openbw # 브리지 Rust 절반(C ABI, staticlib)
  uranium-scripted    # T0 규칙 기반 상대 봇
bridge/openbw         # BWAPI 4.2 AIModule shim(C++) + C ABI 헤더 → liburanium_bridge.so
bridge/scr            # samase 플러그인 (.dll)
tools/                # 벤치(bench), CASC 추출(casc-extract), OpenBW 빌드(openbw/), 오픈소스 봇 빌드(oss/), .env 로더(env.sh)
config/               # assets.toml, hosts.example.toml, plans/(uranium-env 매치업 계획)
docs/                 # windows.md (Windows 세션 체크리스트)
third_party/          # OpenBW·BWAPI·T1 봇 소스(커밋/아카이브 고정, git 미추적) + patches/(0001 폴더 로더, 0002 고정 시작 seed, 0003 런처 종료 크래시 회피, 0004 결정론적 LAN uid, 0005–0010 T1 봇 BWAPI 4.2 포팅: UAlbertaBot·Steamhammer·ZZZKBot·Stardust·McRave·Iron)
reports/              # 스파이크·벤치·평가 리포트
```
진행 상황(2026-10-16): M0 완료(012·013 외 Approved), M1 완료(벤치·OpenBW·데이터 추출), M2 완료(proto·브리지·T0 테란·셀프매치, `reports/m2_selfmatch.md`), M3 Linux 범위 완료(hostd·env 스케줄러·calibrate, T0 3종족, T1 봇 6종, `reports/m3.md`·`reports/oss_bot_porting.md`). M3 남은 것: Windows(`docs/windows.md`), 원격 2호스트 020-AC1. 다음은 M4(리플레이 파이프라인, spec 013 승인 선행).

## 코딩 규칙
- Burn 0.22 모델 코드는 백엔드에 무관하게 쓰고 실행 시 `Device`로 고른다. 테스트는 `Device::flex()`로 한다 (032-R1).
- 버전별 ID와 오프셋은 데이터 테이블로만 관리한다. 코드에 매직 넘버를 흩뿌리지 않는다 (013-R1).
- **개인·보안 정보 금지** (001-R10): spec, 문서, 리포트, 주석, 커밋에 IP, 호스트 이름, 계정, 이메일, 개인 경로, 자격 증명을 쓰지 않는다. `<server>` 같은 자리표시자를 쓴다.
- **설정은 `.env`로** (001-R9): 호스트별 설정은 저장소 루트 `.env`(git 미추적)에서 읽고, 프로세스 환경변수가 우선한다. 키 템플릿은 `.env.example`에 둔다. `.bashrc`에 두지 않는다. 새 환경변수를 추가하면 `.env.example`도 갱신한다.
- 결정론을 유지한다: executor와 feature 추출은 같은 입력이면 같은 출력을 낸다 (031-R7, 030-R1).
- 각 spec의 Acceptance Criteria를 테스트나 벤치로 구현해 검증한다.

## 명령어
- 전체 테스트: `cargo test --workspace`
- Burn/Vulkan 벤치(060-R2): `cargo run -p bench --release -- --backend vulkan` (`--quick`, `--filter <name>`, `--backend flex`) → `reports/bench_bc250.md`
- OpenBW/BWAPI 빌드(011): `tools/openbw/build.sh` → `third_party/bwapi/build/bin/BWAPILauncher`
- 게임 데이터 추출(011-R13): `cargo install casc --version 0.1.6 --locked` 후 `cargo run -p casc-extract -- ~/sc_data/scr/Data ~/sc_data/openbw` → `.env`의 `URANIUM_DATA_DIR`에 추출 경로 지정
- 브리지 빌드(011): `tools/openbw/build_bridge.sh` → `bridge/openbw/build/liburanium_bridge.so` (shim은 500줄 이하 유지)
- 설정: `cp .env.example .env` 후 값 입력. Rust 바이너리는 시작 시, 셸 스크립트는 `tools/env.sh`로 읽는다
- 셀프매치(M2): `tools/openbw/selfmatch.sh <fighting_spirit|circuit_breaker> <games> [parallel] [seed_base] [agent args...]` → `runs/…/summary.md`. `SAME_SEED=1`이면 모든 게임 동일 seed(결정론 확인), `URANIUM_MAX_FRAMES`로 게임 길이 제한
- hostd(020): `target/release/uranium-hostd serve`. 설정 키는 `.env.example`의 hostd 절(`URANIUM_BWAPI_DIR`, `URANIUM_BRIDGE_LIB`, `URANIUM_HOSTD_LISTEN`, `URANIUM_HOSTD_MAX_GAMES` 등)
- 게임 스케줄러(020): `target/release/uranium-env run --plan config/plans/<plan>.toml [--out runs/<dir>]` → `games.jsonl`, `results.jsonl`, `summary.md`, `sessions.md`. 호스트 목록은 `URANIUM_HOSTS_FILE`(기본: 로컬 hostd 하나, 예시 `config/hosts.example.toml`)
- 동시 게임 수 측정(020-R10): 게임 호스트에서 `target/release/uranium-env calibrate` 한 번 → `<URANIUM_CACHE_DIR>/hostd_calibration.toml`. hostd는 `URANIUM_HOSTD_MAX_GAMES`가 없으면 이 값을 쓴다
- T1 봇 빌드(021): `tools/openbw/build.sh` 후 `tools/oss/build_<bot>.sh` (`ualbertabot`, `steamhammer`, `zzzkbot`, `stardust`, `mcrave`, `iron`) → `<URANIUM_OSS_BOTS_DIR>/<bot>/{module.so, VERSION}`. 상대 ID는 `<bot>@<버전>`(021-R3)
- 에이전트 서버: `target/release/uranium-agent serve --listen 127.0.0.1:7600 --out runs/x.jsonl`, 요약 `uranium-agent summarize runs/x.jsonl --out x.md`
- OpenBW LAN 매칭(011-R16): Linux는 LOCAL 소켓을 쓰고, Unix 소켓 108바이트 경로 제한 때문에 `$XDG_RUNTIME_DIR/uranium/`(보통 `/run/user/<uid>/uranium/`)에 둔다. Windows는 OpenBW가 LOCAL 모드를 지원하지 않아 TCP 포트 쌍(`URANIUM_HOSTD_PORT_BASE`부터)을 쓰므로 소켓 경로·길이 제한이 없다
- 게임 데이터(MPQ, CASC 추출물, 맵, 리플레이)는 저작권 자료라 절대 커밋하지 않는다(.gitignore).

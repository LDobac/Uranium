# CLAUDE.md — Uranium

스타크래프트1 **테란 밀리 AI 봇**을 구현하고 강화학습하는 프로젝트. 학습은 OpenBW(1.16.1), 최종 실행은 SC:R.
- 사용자와의 대화와 문서는 **한국어**, 코드 식별자와 주석은 영어.

## 진행 상황
- 시간순 기록: `reports/timeline.md`
- BC 본학습 전 남은 결정·수정: `reports/m6_pre_bc_review.md` G절
- 마일스톤별 결과: `reports/` (m2~m7)
- M7 SC:R 남은 과제: `reports/m7_scr_bridge.md` 4절

## 고정 조건
| 항목 | 값 |
|---|---|
| 종족 | 테란 고정. 상대 T / P / Z |
| 맵 | 투혼(Fighting Spirit), 서킷브레이커(Circuit Breaker) |
| 학습 호스트 | Linux Vulkan 호스트 또는 CUDA 호스트(Linux/Windows). CUDA는 Burn `cuda` feature로만 쓰고(CUDA 호스트에서만 빌드), 직접 쓴 CUDA·ROCm·HIP 코드는 금지 |
| 프레임워크 | Rust + Burn **0.22.0**, 실행 시 `Device::vulkan`/`Device::cuda`, 테스트는 `flex`. 정밀도는 032-R8(CUDA f32, Vulkan BC 학습 f32). 학습 crate는 Windows에서도 빌드 (001-R8) |
| 시뮬레이터 | OpenBW + BWAPI 4.2 포크, Linux 헤드리스. 다른 LAN 호스트에서 실행될 수 있음 |
| SC:R 연동 | samase 플러그인 DLL + samase_scarf. **로컬/LAN 커스텀 게임만**, 배틀넷 공개 게임·래더 대상 아님 |
| 체크포인트·데이터 | 공용 드라이브(SMB, `URANIUM_SHARED_DIR`). pull → 학습 → push, run당 한 호스트(heartbeat 잠금) (061) |
| C++ | OpenBW 브리지의 얇은 shim만. 나머지는 Rust |

## Spec Driven 규칙 (필수)
1. **`spec/`이 단일 진실 공급원.** 작업 전에 `spec/README.md`와 관련 spec을 읽는다.
2. `Approved`가 아닌 spec은 구현하지 않고 사용자에게 알린다.
3. 동작 변경 순서: **spec 수정 → 사용자 확인 → 코드**. 코드와 spec이 다르면 코드를 고치거나 spec 변경을 제안한다.
4. 주석·테스트 이름·커밋 메시지에 요구사항 ID(`010-R3` 등)를 적는다.
5. 새 spec은 공통 템플릿(`Status / Purpose / Scope & Non-goals / Requirements / Interfaces / Acceptance Criteria / Risks & Open Questions / Related`)을 따르고 `spec/README.md`에 추가한다. 번호는 영역 대역(00x 개요, 01x 게임 연동, 02x 실행 환경, 03x 모델, 04x 데이터·학습, 05x 평가, 06x 인프라)이고 바꾸지 않는다. 구현 순서는 마일스톤 M0~M7.
6. 큰 작업(새 crate, 아키텍처 변경, 여러 spec에 걸친 변경)은 Plan 모드로 승인을 받는다.

## 모호함 처리 (필수)
- 사용자 결정이 필요한 부분은 `[SPEC AMBIGUOUS] AMB-<spec>-<n>`으로 표기하고 `spec/README.md` 레지스트리에 등록한다.
- **추측하지 않고 사용자에게 묻는다.** 기본안은 "기본안(미확정)"으로만 제시한다. 사용자가 정하지 않은 수치·정책·범위(승률 목표, APM 제한, 데이터 출처, 실행 환경 등)를 확정하지 않는다.
- `[SPEC AMBIGUOUS]`에 의존하는 코드는 쓰지 않는다.
- `[SPEC DEFERRED(Mx)]`: 결정을 뒤 마일스톤으로 미룬 항목. spec 승인은 막지 않지만 의존 코드는 쓰지 않고, 그 마일스톤 시작 전에 다시 묻는다. 목록은 레지스트리 참조.
- 확정되면: 본문 반영 → 태그 제거 → 레지스트리 `Resolved (YYYY-MM-DD: 결정)`.
- `TBD(Mx)`는 측정하면 정해지는 값, `[SPEC AMBIGUOUS]`는 사용자 판단이 필요한 값.
- 남은 작업·대규모 배치 전 점검에서는 태그만 보지 말고 spec의 미이행 약속("Mx에서 ~", "비교해 채택")과 미실행 AC도 확인한다.

## 아키텍처 원칙
- 에이전트 코어는 게임 버전을 모른다 (001-R1). 버전 의존 코드는 `bridge/*`와 013 매핑 테이블에만.
- 브리지는 얇게: 상태 직렬화, 명령 실행, 시야 필터링만 (001-R3).
- 브리지 ↔ 에이전트는 `uranium-proto`(TCP, postcard)로만 (010).
- **맵핵 금지**: 관측은 자기 시야 기준 (010-R5, 011-R5, 012-R3, 030-R2).
- SC:R 명령은 네트워크 커맨드 경로로만, 유닛 메모리 직접 수정 금지 (012-R4).
- 학습은 lockstep, SC:R 실전은 realtime (001-R5).
- 맵 고정은 RL·평가에만. BC는 전체 맵으로 사전학습하고 대상 맵으로 파인튜닝 (041).

## 저장소 구조
```
spec/                    스펙 (먼저 읽을 것)
crates/
  uranium-proto          공통 스키마·프레이밍
  uranium-env            feature·마스크·보상·executor·맵 사전계산 (라이브러리)
  uranium-model          Burn 모델
  uranium-train          BC·RL·리그, 체크포인트, 공용 드라이브 잠금(061), worker
  uranium-replay         .rep 파서·데이터셋 build
  uranium-ladder         SC:R 래더 리플레이 수집 (043, Windows)
  uranium-hostd          OpenBW 인스턴스 데몬
  uranium-agent          에이전트 서버(정책 라우팅, 배치 추론)
  uranium-runner         게임 스케줄러 (바이너리 이름 uranium-env)
  uranium-bridge-openbw  브리지 Rust 절반 (C ABI)
  uranium-scripted       T0 규칙 기반 봇
bridge/openbw            BWAPI shim(C++) → liburanium_bridge
bridge/scr               samase 플러그인 (M7, 자체 Cargo workspace, 표는 tables/)
tools/                   bench, casc-extract, openbw/, oss/, scr/, worker/, env 로더
config/                  plans/, ladder/, bc.toml, rl.toml, reward.toml, map_families.toml, map_exclude.toml
docs/                    windows.md, commands.md
third_party/             OpenBW·BWAPI·T1 봇 소스(git 미추적) + patches/
reports/                 리포트, timeline.md
```

## 코딩 규칙
- Burn 모델 코드는 백엔드 무관, 실행 시 `Device`로 고른다. 테스트는 `Device::flex()` (032-R1).
- 버전별 ID·오프셋은 데이터 테이블로만 관리한다 (013-R1).
- **개인·보안 정보 금지** (001-R10): spec·문서·리포트·주석·커밋에 IP, 호스트 이름, 계정, 이메일, 개인 경로, 자격 증명을 쓰지 않는다. `<server>`, "Linux Vulkan 호스트" 같은 표기를 쓴다. 추적 파일(spec·코드·주석·추적 config·docs·CLAUDE.md·tools)에는 장비 종류·모델명·번호·상태를 쓰지 않고 OS·백엔드로만 적는다. 장비 표기는 추적하지 않는 config와 `reports/`에만 둔다.
- **설정은 `.env`** (001-R9): 저장소 루트 `.env`(git 미추적), 프로세스 환경변수 우선. 새 키는 `.env.example`에도 추가.
- 결정론: executor와 feature 추출은 같은 입력이면 같은 출력 (031-R7, 030-R1).
- 각 spec의 AC는 테스트나 벤치로 검증한다.
- **git**: worktree 브랜치는 `fetch` → `rebase origin/master` → 테스트 → `push origin HEAD:master`(fast-forward). merge 커밋 금지. push 뒤 메인 작업 사본(저장소 루트 `master`)도 `pull --ff-only`로 맞춘다. 커밋 안 된 변경이 있거나 `master`가 아니면 건드리지 않고 알린다.
- **기록**: 결정·발견·장시간 실행의 시작과 끝·새 리포트는 `reports/timeline.md`에 한 줄씩 덧붙인다(수정·삭제 금지). 실험 리포트에는 실험 전 배경·예상과 진행 기록을 남긴다. 체크포인트·로그·평가 출력 같은 중간 산출물은 지우지 않고 위치를 적는다.
- **대규모 배치(4시간 이상) 전**: 출력에 영향을 주는 결정이 모두 확정·구현됐는지 확인하고, 표본으로 끝까지(build → 학습) 먼저 돌린다.
- 게임 데이터(MPQ, CASC 추출물, 맵, 리플레이)는 커밋하지 않는다.

## 명령어
상세 옵션은 `docs/commands.md`.
- 테스트: `cargo test --workspace`
- 빌드: `tools/openbw/build.sh` → `tools/openbw/build_bridge.sh` (Windows는 `.ps1`), `cargo build --release [--features cuda]`
- 게임: `uranium-hostd serve`, `uranium-env run --plan config/plans/<plan>.toml`, `uranium-env calibrate`
- 리플레이: `uranium-replay build <dir> --src <tag> --records <jsonl> [--part i --parts N]`
- BC: `uranium-train bc --dataset replays/shards/<src>... --run <id> --backend <cuda|vulkan>`, `bc-eval`, `bc-epoch`, `split-stats`, `manifest`
- 작업자: `uranium-train worker --queue <큐> --backend <b> --priority <n>`, 감시 `uranium-train supervise -- <인자>`
- RL: `uranium-train rl --run <id> --bc <ckpt> --logs <dir>`
- SC:R 실전(`docs/scr.md`): `tools\scr\build.ps1` → `uranium-agent scr --checkpoint <ckpt> --backend cuda` → `tools\scr\run.ps1` → 게임 시작은 사람이

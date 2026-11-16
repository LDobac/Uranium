# AGENTS.md — Uranium

스타크래프트1 **테란 밀리 AI 봇** 구현·강화학습. 학습 OpenBW(1.16.1), 실전 SC:R.

표기: [BC][s041] = 리플레이 모방학습, [RL][s042] = 강화학습, [T0][s021] = 자체 Rust 규칙 기반 봇, [T1][s021] = OpenBW로 빌드한 공개 BWAPI 봇.

## 진행 상황
- [timeline](docs/timeline.md)
- 현재 모델: [model2](docs/report/plan/2026_11_09_22_18_model2.md)
- Checkpoint 1: [리포트](docs/report/release/2026_11_08_12_21_checkpoint1.md), `releases/checkpoint-1/`
- Checkpoint 1 RL: [리포트](docs/report/release/2026_11_08_13_35_checkpoint1_rl.md), `releases/checkpoint-1-rl/`
- SC:R 남은 과제: [m7_scr_bridge](docs/report/milestone/2026_10_23_00_16_m7_scr_bridge.md) 4절
- JEPA 계획 설계: [jepa_planning](docs/report/plan/2026_11_11_14_56_jepa_planning.md)

## 고정 조건
| 항목 | 값 |
|---|---|
| 종족 | 테란 고정. 상대 T / P / Z |
| 맵 | 투혼(Fighting Spirit), 서킷브레이커(Circuit Breaker) |
| 학습 호스트 | Linux ROCm 또는 CUDA(Linux/Windows) ([060][s060]). 직접 쓴 커널은 한 소스로 CUDA·HIP 빌드, 기준 구현·끄는 설정 필수 ([060-R1][s060]) |
| 프레임워크 | 학습·추론 Python **3.14** + PyTorch **2.14.x**, 게임 연동·feature·데이터·저장소 Rust(PyO3 `uranium._core`) ([001][s001], [AMB-000-4][amb]). 정밀도 [032-R8][s032](CUDA bf16 혼합, ROCm f32). 테스트·비트 일치 기준 CPU. Linux·Windows 빌드 ([001-R8][s001]) |
| 시뮬레이터 | OpenBW + BWAPI 4.2 포크, Linux 헤드리스. 다른 LAN 호스트 가능 |
| SC:R 연동 | samase 플러그인 DLL + samase_scarf ([012][s012]). **로컬/LAN 커스텀 게임만**, 배틀넷 공개 게임·래더 금지 |
| 체크포인트·데이터 | 공용 드라이브(SMB, `URANIUM_SHARED_DIR`). pull → 학습 → push, run당 한 호스트(heartbeat 잠금) ([061][s061]) |
| C++ | OpenBW 브리지 shim과 확장 커널만. 나머지는 Rust 또는 Python |

## Spec Driven 규칙 (필수)
1. **`docs/spec/`이 단일 진실 공급원.** 작업 전 [docs/spec/README.md](docs/spec/README.md)와 관련 spec을 읽는다.
2. `Approved`가 아닌 spec은 구현하지 않고 사용자에게 알린다.
3. 동작 변경: **spec 수정 → 사용자 확인 → 코드**. 코드와 spec이 다르면 코드를 고치거나 spec 변경을 제안.
4. 요구사항 ID(`010-R3` 등)는 테스트 이름과 커밋 메시지 body에 적는다.
5. 새 spec: `docs/spec/<YYYY_MM_DD_HH_mm>_<NNN-name>.md`, 템플릿 `Status / Purpose / Scope & Non-goals / Requirements / Interfaces / Acceptance Criteria / Risks & Open Questions / Related`, README에 추가. 번호는 영역 대역(00x 개요, 01x 게임 연동, 02x 실행 환경, 03x 모델, 04x 데이터·학습, 05x 평가, 06x 인프라)에서 빈 번호를 고르고, 한 번 정한 번호는 바꾸지 않는다. 구현 순서 [M0~M7][ms].
6. 큰 작업(새 crate, 아키텍처 변경, 여러 spec에 걸친 변경)은 Plan 모드로 승인받는다.

## 모호함 처리 (필수)
- **추측하지 않는다. 결정이 필요하면 사용자에게 묻는다. 질문은 모아서 한 번에.** 기본안은 "기본안(미확정)"으로만. 사용자가 정하지 않은 수치·정책·범위(승률 목표, APM 제한, 데이터 출처, 실행 환경 등)는 확정하지 않는다.
- `[SPEC AMBIGUOUS] AMB-<spec>-<n>`: 사용자 판단이 필요한 값. README 레지스트리에 등록, 의존 코드 금지.
- `[SPEC DEFERRED(Mx)]`: 뒤 마일스톤으로 미룬 결정. spec 승인은 막지 않음, 의존 코드 금지, 그 마일스톤 전에 다시 묻는다.
- `TBD(Mx)`: 측정하면 정해지는 값.
- 확정: 본문 반영 → 태그 제거 → 레지스트리 `Resolved (YYYY-MM-DD: 결정)`.
- 남은 작업·대규모 배치 전 점검: 태그 외에 미이행 약속("Mx에서 ~", "비교해 채택")·미실행 AC(Acceptance Criteria)도.

## 아키텍처 원칙
- 에이전트 코어는 게임 버전을 모른다 ([001-R1][s001]). 버전 의존 코드는 `bridge/*`와 [013][s013] 매핑 테이블에만.
- 브리지는 얇게: 상태 직렬화, 명령 실행, 시야 필터링만 ([001-R3][s001]).
- 브리지 ↔ 에이전트는 `uranium-proto`(TCP, postcard)로만 ([010][s010]).
- **맵핵 금지**: 관측은 자기 시야 기준 ([010-R5][s010], [011-R5][s011], [012-R3][s012], [030-R2][s030]).
- SC:R 명령은 네트워크 커맨드 경로로만, 유닛 메모리 직접 수정 금지 ([012-R4][s012]).
- 학습은 lockstep, SC:R 실전은 realtime ([001-R5][s001]).
- 맵 고정은 RL·평가만. BC는 전체 맵 사전학습 → 대상 맵 파인튜닝 ([041][s041]).

## 저장소 구조
```
docs/
  spec/                  스펙, README.md = 목록·레지스트리
  report/<목적>/          milestone, investigation, bench, experiment, review, advice, plan, survey, release
  guide/                 commands.md, scr.md, measurement_protocol.md
  timeline.md
crates/                  uranium-* (역할: spec 001, ladder는 043)
python/uranium           model/, bc/, rl/, agent/, distill/, diag/(cue/ 단서 진단, planning/ JEPA 계획 측정), bench/, measure/(측정 프로토콜·표, 실행 파일 tools/eval), kernels/, ckpt.py, clip.py, device.py, env.py, passes.py, provenance.py, tables.py. 진입점 `python -m uranium`
bridge/openbw            BWAPI shim(C++) → liburanium_bridge
bridge/scr               samase 플러그인 (자체 Cargo workspace, 표 tables/)
tools/                   openbw/, oss/, scr/, eval/(측정 명령 실행 파일 → `uranium.measure`, 호스트 목록), archive/(공용 드라이브 압축 보관), ablation/, worker/, casc-extract/, bwapi-tables/, env.sh·env.ps1, build_common.sh·build_common.ps1, pins.txt
tests/                   python/(pytest, CPU 기준, `-m gpu` 장치 비교), fixtures/
config/                  학습·평가 설정(`*.toml`), plans/, protocol/, plan_s0/, ladder/, hosts.example.toml
releases/                checkpoint-1/, checkpoint-1-rl/
third_party/             OpenBW·BWAPI·T1 봇 소스(git 제외) + patches/
runs/                    코드 기본 출력 (git 제외)
local/                   장비별 (git 제외)
  work/<작업>/ scripts/ sc_data/(맵·OpenBW 데이터·캐시) raw/(옛 벤치 로그)
```
- crate 역할은 [001][s001], `uranium-ladder`는 [043][s043].

## 코딩 규칙
- 모델 코드는 장치 무관, 실행 시 `torch.device`로 고른다. 단위 테스트 CPU, 장치 비교 pytest `gpu` 마커 ([032-R1][s032]).
- **최적화는 모델 동작 유지** ([032-R19][s032]): 같은 수식의 빠른 구현만. 구조·헤드·해상도·distance bias·dropout·손실·하이퍼파라미터 불변. Linux ROCm 호스트 메모리 예산([060-R3][s060])에서 최악 모양이 돌아야 채택.
- 버전별 ID·오프셋은 데이터 테이블로만 ([013-R1][s013]).
- **개인·보안 정보 금지** ([001-R10][s001]): 추적 파일 전부(코드·config·문서·커밋)에 IP, 호스트 이름, 계정, 이메일, 개인 경로, 자격 증명 금지(`<server>`, "Linux ROCm 호스트"로). 장비 종류·모델명·번호·상태는 추적하지 않는 config, `docs/report/`, `docs/timeline.md`에만. 나머지 추적 파일은 OS·백엔드로만.
- **설정은 `.env`** ([001-R9][s001]): 루트 `.env`(git 제외), 프로세스 환경변수 우선. 새 키는 `.env.example`에도.
- 결정론: executor와 feature 추출은 같은 입력이면 같은 출력 ([031-R7][s031], [030-R1][s030]).
- 각 spec의 AC는 테스트나 벤치로 검증.
- **주석 금지**: 코드·config에 주석을 쓰지 않는다. 예외는 실행에 쓰이는 것만: 지시문(`# noqa`, shebang), argparse가 `__doc__`로 보여 주는 모듈 설명, clap `///` 도움말, `.env.example`의 키 설명, `.gitignore`. 생성 스크립트도 생성하는 config·plan·표에 주석을 쓰지 않는다.
- **리뷰**: master 반영 전 별개 리뷰어 에이전트로 spec 적합성·코드 품질 리뷰. 지적은 반영하거나 안 한 이유를 남긴다.
- **git**: worktree 브랜치 `fetch` → `rebase origin/master` → 테스트 → `push origin HEAD:master`(fast-forward), merge 커밋 금지. 뒤에 메인 사본(루트 `master`) `pull --ff-only`. 커밋 안 된 변경이 있거나 `master`가 아니면 건드리지 않고 알린다.
- **커밋 메시지**: `<type>: <subject>` + 빈 줄 + `[body]`. type은 feat, fix, docs, test, chore, refactor, style. 제목 50자 이내.
- **대규모 배치(4시간 이상) 전**: 출력에 영향 주는 결정이 모두 확정·구현됐는지 확인, 표본으로 build → 학습 끝까지 먼저.
- 게임 데이터(MPQ, CASC 추출물, 맵, 리플레이)는 커밋하지 않는다.

## 문서 규칙
- 문서는 `docs/` 안에만(폴더는 저장소 구조). 새 문서 이름 `YYYY_MM_DD_HH_mm_<이름>.md`(작성 시각). 고정 이름: `docs/timeline.md`, `docs/guide/*`, `docs/spec/README.md`. 루트에는 `AGENTS.md`, `CLAUDE.md`(`AGENTS.md` 참조만), `README.md`만.
- 짧게: 단답형, 표·목록. 서술·경위·중복 금지. 다른 문서 내용은 링크.
- 코드(`042-AC1`, `AMB-…`, BC 같은 약어)만 쓰지 않는다. 문서에서는 출처 문서를 링크하고, 사용자에게 설명할 때는 그 코드가 담긴 문서를 함께 밝힌다.
- timeline: `## YYYY-MM-DD` 아래 `- HH:MM — 한 일 ([근거](경로))` 한 줄(최대 두 줄). 결정·발견·장시간 실행 시작/끝·새 리포트. 지우지 않고 `정정:`. 상세 [머리말](docs/timeline.md).
- 실험 리포트에는 실험 전 배경·예상과 진행 기록.

## 파일 생성 규칙
- 로그·임시 출력·일회용 스크립트 `local/work/<작업>/`, 장비별 재사용 스크립트 `local/scripts/`, 게임 데이터 `local/sc_data/`. 직접 만드는 로그·스크립트·임시 파일은 `local/` 밖(홈, `/tmp`, 루트, 추적 폴더)에 두지 않는다. 예외: 코드·빌드 기본 출력(`runs/`, `build/`, `target/`), 다른 장비와 나눌 결과(공용 드라이브). 모든 장비 동일.
- 백업 파일(`.bak`, `.orig`, 복사본) 금지. 되돌릴 것은 git으로.
- 임시 문서 금지. 근거로 쓰는 산출물(체크포인트·로그·평가 출력)은 남기고 위치를 문서에 적는다. 나머지 임시 파일은 그 작업에서 지운다.

## 명령어
상세: [commands.md](docs/guide/commands.md), SC:R [scr.md](docs/guide/scr.md).
- 테스트: `cargo test --workspace`, `<venv python> -m pytest tests/python`(GPU 호스트 `-m gpu`도)
- 빌드: `tools/openbw/build.sh` → `build_bridge.sh`(Windows `.ps1`), `cargo build --release`, `uv pip install --python <venv python> -e .`
- 게임: `uranium-hostd serve`(OpenBW 인스턴스 데몬), `python -m uranium run --plan config/plans/<plan>.toml`(스크립트 플레이어만이면 `uranium-env run`), `uranium-env calibrate`
- 리플레이: `uranium-replay build`
- 학습: `python -m uranium bc | rl`, 작업자 `uranium-train worker`, 감시 `uranium-train supervise`
- SC:R 실전: `tools\scr\build.ps1` → `python -m uranium scr --checkpoint <ckpt>` → `tools\scr\run.ps1` → 게임 시작은 사람이
- 평가·측정: [measurement_protocol.md](docs/guide/measurement_protocol.md), `tools/eval/`

[s001]: docs/spec/2026_10_15_22_24_001-system-architecture.md
[s010]: docs/spec/2026_10_15_22_24_010-game-protocol.md
[s011]: docs/spec/2026_10_15_22_24_011-bridge-openbw.md
[s012]: docs/spec/2026_10_15_22_24_012-bridge-remastered.md
[s013]: docs/spec/2026_10_15_22_24_013-version-mapping.md
[s021]: docs/spec/2026_10_15_22_24_021-opponent-pool.md
[s030]: docs/spec/2026_10_15_22_24_030-observation.md
[s031]: docs/spec/2026_10_15_22_24_031-action-space.md
[s032]: docs/spec/2026_10_15_22_24_032-model-architecture.md
[s041]: docs/spec/2026_10_15_22_24_041-imitation-learning.md
[s042]: docs/spec/2026_10_15_22_24_042-rl-training.md
[s043]: docs/spec/2026_10_16_22_48_043-ladder-replay-collector.md
[s060]: docs/spec/2026_10_15_22_24_060-infra.md
[s061]: docs/spec/2026_10_15_22_24_061-shared-storage.md
[amb]: docs/spec/README.md
[ms]: docs/spec/README.md#마일스톤

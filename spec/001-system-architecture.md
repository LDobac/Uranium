# 001 — System Architecture

- **Status**: Approved (2026-10-15, 2026-10-23 개정: 학습·추론을 Python + PyTorch로 전환(AMB-000-4) — 구성도, R6 batch broker, R8 빌드 매트릭스, R9 Python, R10 ROCm 표기, Cargo Workspace·Python 패키지, 데이터 흐름, 의존 방향. 2026-10-17 개정: 스케줄러를 `uranium-runner` crate로 분리(AMB-001-3), 에이전트 서버의 모델 정책·배치 추론 / `uranium-runner` 라이브러리화, `uranium-train rl`(3단계 계획 승인). 2026-10-22 개정: R1·Cargo Workspace에서 `uranium-adapter` 계획 삭제, `bridge/scr` 자체 workspace(AMB-001-5, 사용자 결정). 2026-10-19 개정: 001-R10 장비 표기 규칙(G-D9, 사용자 결정), R1·Cargo Workspace 표기 정정: 현재 매핑 테이블 위치를 적고 `uranium-adapter`를 M7 예정으로 표시)
- **Related**: [010](010-game-protocol.md), [011](011-bridge-openbw.md), [012](012-bridge-remastered.md), [020](020-env-orchestration.md), [060](060-infra.md), [061](061-shared-storage.md)

## Purpose
프로세스 구성, 네트워크 토폴로지, crate 경계, 데이터 흐름을 정의한다.

## 구성도
```
 [게임 호스트 (Linux, 2대 이상)]                 [학습 호스트: Linux(ROCm) 또는 CUDA(Linux/Windows)]       [Windows SC:R PC]
 uranium-hostd ◄──제어 TCP── 러너(Rust 스레드)     Python 프로세스 `python -m uranium` (PyTorch: BC·RL learner, 추론)
   ├ 대전: BWAPILauncher ×2 (1v1 LAN)               └ uranium._core(PyO3): 에이전트 서버·러너·batch broker·데이터 공급(Rust 스레드)  ◄──TCP── bridge_scr.dll (samase)
   │   ├ bridge_openbw.so ──TCP: RawFrame, PrivilegedSummary──►  (listen)                        └ StarCraft.exe
   │   └ 상대: bridge_openbw.so(셀프플레이·T0) | OSS 봇 .so          │ pull / push (061)
   └ 재생: BWAPILauncher(replay) + bridge_openbw.so ──TCP──►         ▼
                                                   [공용 드라이브 (SMB)] checkpoints/ · datasets/ · locks/
```

## Requirements
| ID | 요구사항 |
|---|---|
| 001-R1 | **MUST** 에이전트 코어(env feature, model, executor)는 게임 버전을 모른다. 버전 의존 코드는 `bridge/*`와 013 매핑 테이블에만 둔다. 현재(M4~M6) 매핑 테이블은 `crates/uranium-env/tables`와 `crates/uranium-replay/tables`에 있고 각 crate는 테이블을 읽기만 한다. SC:R 브리지의 표는 `bridge/scr/tables`에 둔다. 별도 crate `uranium-adapter`는 만들지 않는다(2026-10-22 사용자 결정, AMB-001-5: SC:R 원시 ID가 BW와 같고 정규화는 브리지 안에서 끝나며, 에이전트에는 realtime 처리만 더한다 — 012-R18·R19) |
| 001-R2 | **MUST** 브리지와 에이전트는 [010](010-game-protocol.md) 프로토콜로만 통신한다. 같은 호스트든 원격이든 같은 TCP 경로를 쓴다 |
| 001-R3 | **MUST** 브리지는 얇게 유지한다. 상태 직렬화, 명령 실행, 시야 필터링만 하고 의사결정 로직은 넣지 않는다 |
| 001-R4 | **SHOULD** OpenBW 호스트를 추가할 때 코드 수정 없이 설정 파일만 바꾸면 되게 한다 |
| 001-R5 | **MUST** 학습 모드는 lockstep(게임이 에이전트 응답을 기다림), 실전(SC:R) 모드는 realtime(데드라인 초과 시 무명령)으로 동작한다 |
| 001-R6 | **MUST** 에이전트 서버(`uranium-agent` 라이브러리)가 listen하고 브리지가 접속한다. 서버는 원격 PC 또는 같은 PC(`127.0.0.1`)일 수 있다. 셀프플레이에서는 두 브리지가 같은 서버에 접속하고 `game_id`/`slot`(010-R9)으로 구분한다. 연결 처리·feature 추출·executor는 Rust 스레드가 하고, 모델 추론은 batch broker를 거쳐 같은 프로세스의 Python 추론 스레드가 한다: broker는 모델 키(체크포인트 × 샘플링 모드)별로 요청을 모아 collate한 배치를 내주고(GIL을 잡지 않는다), Python이 돌려준 결과를 HostAction으로 해석해 세션에 돌려준다(032-R14, 2026-10-23) |
| 001-R7 | **MUST** 학습 전용 정보(`PrivilegedSummary`)는 learner와 value head로만 흐른다. 추론 서버의 정책 경로에는 들어가지 않는다(030-R9, 032-R11) |
| 001-R8 | **MUST** 학습 쪽 crate(`uranium-proto`, `uranium-env`, `uranium-data`, `uranium-store`, `uranium-league`, `uranium-train`, `uranium-agent`, `uranium-runner`, `uranium-py`)와 Python 패키지 `uranium`은 Linux와 Windows 모두에서 빌드·실행된다(Unix 전용 API 금지). Python 확장 모듈(`uranium._core`)은 호스트마다 maturin으로 빌드하고, PyTorch는 CUDA 호스트에서 CUDA 빌드, Linux ROCm 호스트에서 ROCm 빌드를 쓴다(060). 브리지(`bridge/openbw`)와 `uranium-hostd`도 Linux와 Windows 모두를 지원한다. 브리지 C++ shim은 GCC와 MSVC 모두에서 빌드돼야 한다. (변경 이력: 2026-10-15 추가 → 같은 날 Windows CUDA 호스트에서 게임까지 돌리기 위해 브리지·hostd도 Windows 지원으로 확장. Windows 구현·검증은 `docs/commands.md` "Windows" 절) |
| 001-R9 | **MUST** 호스트별 설정(경로, 주소, 포트, 디버그 옵션 등)은 저장소 루트의 `.env` 파일에서 읽는다. `.env`는 git에 올리지 않는다(`.gitignore`). 같은 이름의 프로세스 환경변수가 있으면 그 값이 `.env`보다 우선한다. 쓸 수 있는 키와 설명은 값 없는 템플릿 `.env.example`로 커밋한다. 셸 프로필(`.bashrc` 등)에 설정을 두는 방식은 쓰지 않는다. Rust 바이너리와 Python 진입점(`python -m uranium`)은 시작 시 `.env`를 읽고(같은 우선순위), 셸 스크립트는 `tools/env.sh`로 읽는다. 작업자가 Python 학습 자식을 띄울 때 쓰는 인터프리터는 `URANIUM_PYTHON`이다(2026-10-23). 비밀번호 같은 자격 증명은 `.env`에도 두지 않고 OS 자격 증명 저장소나 권한 600 파일(예: cifs credentials)에 둔다 (2026-10-15 사용자 결정) |
| 001-R10 | **MUST** spec, 문서, 리포트, 코드 주석, 커밋 메시지에는 IP 주소, 호스트 이름, 계정 이름, 이메일, 개인 경로(`/home/<user>` 등), 자격 증명 같은 개인·보안 정보를 쓰지 않는다. 필요하면 `<server>`, `<user>` 같은 자리표시자를 쓰고 실제 값은 `.env`에만 둔다. `127.0.0.1` 같은 루프백 주소와 예시 사설 대역 표기는 예외다 (2026-10-15 사용자 결정). **장비 표기(2026-10-19 사용자 결정, G-D9, AMB-001-4)**: git이 추적하는 파일(spec, 코드·주석, 추적되는 config, docs, CLAUDE.md, tools 스크립트)에는 장비 종류·상태(보드·GPU 모델명, 장비 번호, 특정 PC 이름·사양, 어느 장비가 어떤 역할을 할 수 없는지 등)를 쓰지 않고 OS와 백엔드(Linux/Windows, CUDA/ROCm/CPU)로만 표기한다(2026-10-23: 학습 백엔드가 Vulkan에서 ROCm으로 바뀌어 표기도 바꿈). 장비 표기는 git이 추적하지 않는 config와 리포트(`reports/`)에만 쓸 수 있다 |

## Cargo Workspace
```
Cargo.toml (workspace)
crates/
  uranium-proto      # 스키마, 프레이밍, 버전 협상 (no_std 호환 지향, postcard)
  uranium-env        # 라이브러리: feature 추출(030), 마스크(034), 보상(033), executor(031), 맵 계열(040-R15)·맵 사전계산(031), 013 매핑 테이블(`tables/`)
  uranium-data       # BC 데이터 공급(041): shard 읽기·게임 순서·Rng·데이터셋 지문·Feeder·FeedState·prefetch·collate(030 §1 배치 레이아웃)·압축 보관
  uranium-store      # 공용 드라이브(061): 잠금·heartbeat·seal·push/pull·Manifest·devices.jsonl, 체크포인트 파일 규칙
  uranium-league     # RL(042) 호스트 쪽: 리그·PFSP·스냅샷, trajectory·SegmentQueue, 설정
  uranium-train      # 바이너리: 호스트 명령(bc-epoch·label-check·split-stats·manifest·devices)과 worker·supervise(060-R6, 061-R13). 학습 자식은 Python
  uranium-replay     # .rep 파서, 데이터셋 생성(040), 리플레이 명령 매핑 테이블(`tables/`, 013)
  uranium-hostd      # OpenBW 인스턴스 데몬(020)
  uranium-agent      # 에이전트 서버(세션별 정책 라우팅, 모델 정책, batch broker) — 추론 자체는 Python
  uranium-runner     # 게임 스케줄러(020). 바이너리 이름은 `uranium-env`(run·calibrate, 스크립트 봇만 쓰는 plan). 모델 plan은 Python 진입점이 이 라이브러리를 부른다
  uranium-py         # PyO3 확장 모듈 `uranium._core`: uranium-data·store·league·agent·runner를 Python에 노출
python/uranium/      # Python 패키지(PyTorch): model(032), bc(041), rl(042), agent 추론, bench(060-R2), ckpt(061-R10)
pyproject.toml       # maturin 빌드(`uranium._core`)
  uranium-scripted   # T0 규칙 기반 상대 봇 (T/P/Z, 010 Command 직접 생성, 021)
bridge/
  openbw/            # C++ BWAPI AIModule shim + Rust staticlib(uranium-proto)
  scr/               # Rust cdylib samase 플러그인 (x86_64-pc-windows-msvc). 루트 workspace 밖 자체 workspace, 표는 scr/tables (012, 013-R1)
spec/
tools/               # 맵 분석, 벤치, 스크립트
```

## 데이터 흐름 (학습 1 스텝)
1. 브리지가 프레임 `f`에서 `RawFrame`을 송신한다. 송신 주기는 이전 액션의 delay가 끝나는 시점이다.
2. `uranium-agent`의 세션(Rust 스레드)이 `uranium-env` 라이브러리로 feature를 만들고([030](030-observation.md)) batch broker에 넣는다.
3. broker가 여러 세션의 요청을 배치로 묶어 collate하고, Python 추론 스레드가 PyTorch로 의미 단위 액션과 다음 delay를 결정해 돌려준다.
4. Executor가 의미 단위 액션을 `Command` 목록으로 변환하고 브리지로 보낸다.
5. 브리지가 명령을 실행하고 `delay` 프레임만큼 진행한 뒤 1로 돌아간다.
6. trajectory(관측, PrivilegedSummary, 액션, 헤드별·joint behaviour log-prob, 보상)는 learner로 전송된다(042).

- **의존 방향 (2026-10-23 개정, AMB-000-4)**: `uranium-py → {uranium-runner → uranium-agent, uranium-data, uranium-store, uranium-league} → uranium-env`. `uranium-model`은 삭제했고 모델은 Python 패키지에 있다. Python 진입점(`python -m uranium run|serve|scr|rl|bc`)이 PyTorch 추론·학습 스레드를 띄운 뒤 `uranium._core`로 Rust 스레드(에이전트 서버, 러너, 데이터 공급)를 GIL 없이 돌린다. RL은 계속 한 프로세스다(042). 아래 2026-10-17 항목은 변경 이력이다.
- **crate 의존 방향 (확정 2026-10-17, AMB-001-3)**: `uranium-runner → uranium-agent → uranium-model → uranium-env`. 스케줄러가 에이전트 서버를 품고 에이전트가 모델을 쓰므로, 스케줄러를 라이브러리 `uranium-env`에서 떼어 `uranium-runner`로 옮겼다. 바이너리 이름(`uranium-env`)과 CLI는 그대로다. 대안이던 "스케줄러가 `uranium-agent serve`를 하위 프로세스로 실행"은 Ctrl+C 연쇄 종료(042-R10)와 RL 가중치 교체가 IPC가 되므로 택하지 않았다.
- **RL 프로세스 (2026-10-17, 3단계 계획 승인)**: `uranium-runner`는 lib + bin이다(020-R12). `uranium-train rl`이 learner, 추론 서버, 에이전트 서버, 게임 배정(runner 라이브러리), 리그 관리를 한 프로세스에서 돌린다. 의존 방향은 `uranium-train → uranium-runner → uranium-agent → uranium-model → uranium-env`로 순환이 없다. 새 crate는 없다.

## Interfaces (요약)
```rust
/// 브리지 1개와의 세션 (에이전트 서버 측)
trait GameSession {
    fn hello(&self) -> &Hello;                              // backend, mode, game_id, slot, replay
    fn game_info(&self) -> &GameInfo;
    fn observe(&mut self) -> Result<(RawFrame, Option<PrivilegedSummary>)>;  // privileged는 lockstep 학습 모드에서만 Some
    fn act(&mut self, cmds: &[Command], advance_frames: u32, valid_until_frame: Option<u32>) -> Result<()>;
    fn result(&self) -> Option<GameEnd>;
}
```

## Acceptance Criteria
- 001-AC1: 같은 에이전트 바이너리가 설정 플래그(`--backend openbw|scr`)만 바꿔서 두 백엔드에 접속한다.
- 001-AC2: OpenBW 호스트가 원격 PC에 있어도 로컬과 같은 결과를 낸다. 같은 seed와 같은 명령 시퀀스로 결정론적 재현이 되어야 한다.

## Risks & Open Questions
> **[SPEC DEFERRED(M7)] AMB-001-1**: SC:R 실전 시 추론을 어디서 할지 정해야 한다. M7에서 지연을 측정한 뒤 사용자 확인을 받는다.
> - 기본안(미확정): SC:R PC에서 로컬 추론(2026-10-23부터 PyTorch, 실전 PC는 CUDA 학습 호스트와 같은 PC — AMB-012-2)
> - 대안: LAN 너머 학습 호스트의 추론 서버에 요청
> - 확인 필요: SC:R PC의 GPU 사양

- **확정 (2026-10-15, AMB-001-2)**: OpenBW 브리지에 **얇은 C++ shim**을 허용한다.
  - shim의 역할은 BWAPI 이벤트와 호출을 C ABI로 중계하는 것뿐이다.
  - 로직은 모두 Rust staticlib에 둔다.
  - 다른 영역의 C++ 사용은 별도 확인 대상이다.

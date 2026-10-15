# 001 — System Architecture

- **Status**: Approved (2026-10-15)
- **Related**: [010](010-game-protocol.md), [011](011-bridge-openbw.md), [012](012-bridge-remastered.md), [020](020-env-orchestration.md), [060](060-infra-bc250.md), [061](061-shared-storage.md)

## Purpose
프로세스 구성, 네트워크 토폴로지, crate 경계, 데이터 흐름을 정의한다.

## 구성도
```
 [게임 PC (Linux, 2대 이상)]                     [학습 호스트: BC250(Vulkan) 또는 CUDA PC(Linux/Windows)]      [Windows SC:R PC]
 uranium-hostd ◄──제어 TCP── uranium-env          uranium-train (BC·RL learner + 리그, Burn)
   ├ 대전: BWAPILauncher ×2 (1v1 LAN)              uranium-agent (에이전트 서버, 배치 추론)  ◄──TCP── bridge_scr.dll (samase)
   │   ├ bridge_openbw.so ──TCP: RawFrame, PrivilegedSummary──►  (listen)                        └ StarCraft.exe
   │   └ 상대: bridge_openbw.so(셀프플레이·T0) | OSS 봇 .so          │ pull / push (061)
   └ 재생: BWAPILauncher(replay) + bridge_openbw.so ──TCP──►         ▼
                                                   [공용 드라이브 (Linux 서버: NFS/Samba)] checkpoints/ · datasets/ · locks/
```

## Requirements
| ID | 요구사항 |
|---|---|
| 001-R1 | **MUST** 에이전트 코어(env feature, model, executor)는 게임 버전을 모른다. 버전 의존 코드는 `bridge/*`와 `uranium-adapter`에만 둔다 |
| 001-R2 | **MUST** 브리지와 에이전트는 [010](010-game-protocol.md) 프로토콜로만 통신한다. 같은 호스트든 원격이든 같은 TCP 경로를 쓴다 |
| 001-R3 | **MUST** 브리지는 얇게 유지한다. 상태 직렬화, 명령 실행, 시야 필터링만 하고 의사결정 로직은 넣지 않는다 |
| 001-R4 | **SHOULD** OpenBW 호스트를 추가할 때 코드 수정 없이 설정 파일만 바꾸면 되게 한다 |
| 001-R5 | **MUST** 학습 모드는 lockstep(게임이 에이전트 응답을 기다림), 실전(SC:R) 모드는 realtime(데드라인 초과 시 무명령)으로 동작한다 |
| 001-R6 | **MUST** 에이전트 서버(`uranium-agent`)가 listen하고 브리지가 접속한다. 서버는 원격 PC 또는 같은 PC(`127.0.0.1`)일 수 있다. 셀프플레이에서는 두 브리지가 같은 서버에 접속하고 `game_id`/`slot`(010-R9)으로 구분한다 |
| 001-R7 | **MUST** 학습 전용 정보(`PrivilegedSummary`)는 learner와 value head로만 흐른다. 추론 서버의 정책 경로에는 들어가지 않는다(030-R9, 032-R11) |
| 001-R8 | **MUST** 학습 쪽 crate(`uranium-proto`, `uranium-model`, `uranium-train`, `uranium-agent`, `uranium-env`)는 Linux와 Windows 모두에서 빌드·실행된다(Unix 전용 API 금지). 브리지(`bridge/openbw`)와 `uranium-hostd`는 Linux 전용이다(2026-10-15 추가) |

## Cargo Workspace
```
Cargo.toml (workspace)
crates/
  uranium-proto      # 스키마, 프레이밍, 버전 협상 (no_std 호환 지향, postcard)
  uranium-adapter    # GameBackend trait, OpenBW/SCR 백엔드 클라이언트, 정규화 테이블(013)
  uranium-env        # 에피소드 루프, feature 추출(030), 보상(033), executor(031)
  uranium-model      # Burn 모델(032)
  uranium-train      # BC(041), RL(042), 리그·PFSP·스냅샷
  uranium-replay     # .rep 파서, 데이터셋 생성(040)
  uranium-hostd      # OpenBW 인스턴스 데몬(020)
  uranium-agent      # 에이전트 서버(배치 추론, 세션 관리) + SC:R 실전 실행 바이너리
  uranium-scripted   # T0 규칙 기반 상대 봇 (T/P/Z, 010 Command 직접 생성, 021)
bridge/
  openbw/            # C++ BWAPI AIModule shim + Rust staticlib(uranium-proto)
  scr/               # Rust cdylib samase 플러그인 (x86_64-pc-windows-msvc)
spec/
tools/               # 맵 분석, 벤치, 스크립트
```

## 데이터 흐름 (학습 1 스텝)
1. 브리지가 프레임 `f`에서 `RawFrame`을 송신한다. 송신 주기는 이전 액션의 delay가 끝나는 시점이다.
2. `uranium-env`가 feature 텐서를 만들고([030](030-observation.md)) 추론 배치 큐에 넣는다.
3. `uranium-agent`가 배치 추론으로 의미 단위 액션과 다음 delay를 결정한다.
4. Executor가 의미 단위 액션을 `Command` 목록으로 변환하고 브리지로 보낸다.
5. 브리지가 명령을 실행하고 `delay` 프레임만큼 진행한 뒤 1로 돌아간다.
6. trajectory(관측, PrivilegedSummary, 액션, logit, 보상)는 learner로 전송된다.

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
> - 기본안(미확정): SC:R PC에서 Burn wgpu로 로컬 추론
> - 대안: LAN 너머 BC250 추론 서버에 요청
> - 확인 필요: SC:R PC의 GPU 사양

- **확정 (2026-10-15, AMB-001-2)**: OpenBW 브리지에 **얇은 C++ shim**을 허용한다.
  - shim의 역할은 BWAPI 이벤트와 호출을 C ABI로 중계하는 것뿐이다.
  - 로직은 모두 Rust staticlib에 둔다.
  - 다른 영역의 C++ 사용은 별도 확인 대상이다.

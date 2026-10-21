# 020 — Env Orchestration

- **Status**: Approved (2026-10-15, 2026-10-16 개정: 020-AC1 기준, Windows 호스트 로컬 hostd, 2026-10-17 개정: 스케줄러 crate `uranium-runner`, plan 플레이어 `model`과 slot별 정책 라우팅(020-R11), 정책 버전 기록 / 020-R12 동적 Job 공급자(RL 리그, 3단계 계획 승인) / 020-R13 승자 불일치 판정(AMB-020-3, 사용자 결정), 2026-10-18 개정: `Finished.replay` 리플레이 바이트 반환(020-R1, 042-R11), 020-R13 상호 퇴장 = Aborted·hostd 드롭 시간(011-R17·R18, 사용자 결정) / summary.md 정책 표시·정책별 합산(사용자 결정, Risks 절 메모). 2026-10-19 개정: R3 에이전트 무응답 30초 중단 삭제 → OpenBW 드롭(011-R18)·상호 퇴장 Aborted(R13)로 정합(C-9, 사용자 결정), 장비 표기 중립화(001-R10). 2026-10-21 개정: 020-R14 plan 항복 옵션(사용자 결정, AMB-031-6))
- **Related**: [011](011-bridge-openbw.md), [021](021-opponent-pool.md), [042](042-rl-training.md), [060](060-infra.md)

## Purpose
여러 OpenBW 게임을 로컬이나 원격 호스트에서 병렬로 실행하고, 에피소드 수명주기를 관리한다.

## 구성
- **uranium-hostd**: 각 OpenBW 호스트에서 실행되는 데몬이다. 학습 호스트의 `uranium-env`가 gRPC가 아닌 단순 TCP 제어 프로토콜(postcard)로 지시한다.
- **게임 쌍**: `BWAPILauncher` 2개로 구성된다.
  - P1: 우리 봇 브리지
  - P2: 상대. 우리 봇 브리지(셀프플레이) 또는 OSS 봇 `.so`
  - 두 런처는 Linux에서 `OPENBW_LAN_MODE=LOCAL`(같은 호스트 소켓), Windows에서 `OPENBW_LAN_MODE=TCP`(포트 2개)로 매칭한다.
- **uranium-env**: 학습 호스트의 게임 스케줄러 바이너리다(crate `uranium-runner`, 2026-10-17 분리, AMB-001-3). 에이전트 서버를 내장하고, 매치업 계획을 여러 hostd에 분배하고, 결과를 모은다.
- hostd와 env는 Linux와 Windows 모두에서 돈다(001-R8).

## Requirements
| ID | 요구사항 |
|---|---|
| 020-R1 | **MUST** hostd API: `StartGame{map, p1, p2, start_locs?, seed, agent_addrs, save_replay}`, `StartReplay{rep_bytes, perspective_slot, agent_addr, privileged: bool}`(040-R12 스트리밍, 011 재생 모드), `StopGame{id}`, `Status`, `ListCapabilities{max_concurrent_games, max_concurrent_replays, oss_bots}`. `save_replay` 게임이 끝나면 hostd는 slot 0이 저장한 리플레이(두 클라이언트가 같은 게임을 기록하므로 하나면 충분) 바이트를 `Finished.replay`에 담아 돌려준다. 원격 게임 호스트의 리플레이를 학습 호스트가 모으기 위한 것이다. `URANIUM_HOSTD_REPLAY_DIR`이 있으면 그 폴더에 파일도 남기고, 없으면 작업 폴더의 임시 파일을 보낸 뒤 지운다. 제어 프로토콜 버전 2 (2026-10-18, 042-R11) |
| 020-R2 | **MUST** 게임별 작업 디렉터리와 포트를 격리한다. 같은 호스트에서 N개 게임이 동시에 돌아도 충돌이 없어야 한다 |
| 020-R3 | **MUST** 타임아웃 처리 |
| 020-R4 | **MUST** 크래시 처리: 런처 비정상 종료 시 에피소드를 `Aborted`로 표시하고 학습 데이터에서 제외한다. 같은 설정이 연속 3회 크래시하면 격리하고 경보를 낸다 |
| 020-R5 | **MUST** 에피소드 메타데이터를 기록한다: id, 맵, 시작 위치, 상대 종류·버전, 정책 버전, 결과, 길이, 리플레이 경로. 모델 정책의 정책 버전은 체크포인트 id(체크포인트 폴더 이름 + `model.bpk` SHA-256 앞 12자리)와 샘플링 모드다 |
| 020-R12 | **MUST** `uranium-runner`는 라이브러리로도 쓸 수 있어야 한다. 게임 배정은 Job 공급자 인터페이스(다음 Job 요청, 시작 못 한 Job 되돌림, 게임 결과 통지)로 받는다. matchup plan은 정적 공급자, RL 리그(042)는 상대와 맵을 그때그때 정하는 동적 공급자다. 정상 종료(042-R10)를 위해 공급자 중지와 진행 중 게임 `StopGame`을 지원한다 (2026-10-17) |
| 020-R13 | **MUST** 같은 게임의 두 세션이 보고한 승자가 다르면 **더 늦게 끝난 세션(GameEnd 프레임이 더 큰 쪽)의 승자**를 게임 결과로 쓴다. 프레임이 같으면 먼저 받은 보고를 쓴다. 불일치 건수는 따로 집계해 보고서에 남긴다. 이 판정은 평가 집계(`uranium-env` summary), RL 리그 승률 표와 r_win(042, 033) 모두에 쓴다 (2026-10-17 사용자 결정, AMB-020-3). **상호 퇴장**: 두 세션이 모두 `Surrender`(상대가 섬멸되지 않고 떠남, 011-R17)로 자기 승리를 보고하면 연결 끊김(OpenBW 드롭)으로 보고 게임을 `Aborted`로 처리한다. 평가 집계에서는 승패에서 빼고 "dropped" 열로 세며, RL에서는 그 게임의 세그먼트를 회수한다(042-R3). 진짜 항복(봇의 GG 퇴장)은 떠난 쪽 세션이 상대를 승자로 보고하므로 기존 규칙대로 판정한다 (2026-10-18 사용자 결정). hostd는 모든 런처에 OpenBW 드롭 시간 `URANIUM_OPENBW_DROP_TIMEOUT_MS`(기본 600,000ms, 011-R18)를 넘긴다 |
| 020-R11 | **MUST** plan의 플레이어 종류에 `model`(체크포인트로 플레이하는 테란 정책)을 둔다. 스케줄러는 `StartGame`을 보내기 전에 에이전트 서버의 라우팅 표에 `(game_id, slot) → 정책`을 등록하고, 세션은 `Hello.game_id`(010-R9)와 `GameSpec.players` 인덱스로 정책을 찾는다. 인덱스는 hostd가 붙이는 플레이어 이름 `uranium-p<i>`에서 읽는다. BWAPI 플레이어 id(`Hello.slot`)는 런처 순서와 다를 수 있기 때문이다. 이름이 없으면 `Hello.slot`을 쓴다. 프로토콜 변경은 없다. 표에 없는 세션은 서버 기본 정책(T0)을 쓴다 (2026-10-17) |
| 020-R6 | **SHOULD** 호스트 목록과 호스트별 동시 게임 수는 설정 파일로 관리한다. 실제 주소가 들어가므로 커밋하지 않는 로컬 파일(`URANIUM_HOSTS_FILE`)에 두고, 형식 예시만 `config/hosts.example.toml`로 커밋한다(001-R10, 2026-10-15 변경) |
| 020-R8 | **MUST** hostd는 OS별 부분(LAN 매칭 설정, 프로세스 정리)을 플랫폼 계층으로 분리한다. Linux는 LOCAL 소켓과 프로세스 그룹 + `PR_SET_PDEATHSIG`를 쓰고, Windows는 TCP 포트 쌍(`URANIUM_HOSTD_PORT_BASE`부터 할당)과 Job Object(kill-on-close)를 쓴다 |
| 020-R9 | **MUST** hostd 설정은 `.env`에서 읽는다(001-R9): `URANIUM_HOSTD_LISTEN`, `URANIUM_BWAPI_DIR`, `URANIUM_BRIDGE_LIB`, `URANIUM_DATA_DIR`, `URANIUM_MAPS_DIR`, `URANIUM_HOSTD_MAX_GAMES`, `URANIUM_HOSTD_PORT_BASE`(Windows), `URANIUM_CACHE_DIR`(측정 파일, 020-R10) |
| 020-R10 | **MUST** 호스트별 동시 게임 수는 측정으로 정한다. 게임 호스트에서 `uranium-env calibrate`를 한 번 실행하면 동시 게임 수 N을 1부터 늘려 가며 T0 대전(단계마다 2N판)을 돌린다. 세션별 브리지 p99(011-AC2 지표)의 중앙값이 300µs 이하인 가장 큰 N을 권장값으로 `<URANIUM_CACHE_DIR>/hostd_calibration.toml`에 저장한다. hostd의 동시 게임 수는 다음 순서로 정한다: `.env` `URANIUM_HOSTD_MAX_GAMES` → 측정 파일 → 측정 전 기본값 `max(1, (코어 수 − 1) / 2)`(경고 출력). 학습 프로세스와 함께 돌릴 때의 CPU 예산(020-R7)은 M5/M6에서 같은 명령을 확장해 정한다 (2026-10-16 사용자 결정) |
| 020-R7 | **SHOULD** Linux Vulkan 학습 호스트의 로컬 hostd도 지원한다. 이 경우 CPU 코어를 추론 서버와 나눠 쓰는 예산을 정한다([060](060-infra.md)) |
| 020-R14 | **MUST** plan 항복 옵션(2026-10-21 사용자 결정, AMB-031-6): `surrender = "off" \| "shadow" \| "on"`(031-R12 모드, OpenBW plan 기본 `"shadow"` — 학습·평가 게임은 나가지 않는다)과 `surrender_chat`(문자열, 기본 `"gg"`, 빈 문자열이면 채팅 없이 퇴장, UTF-8 80바이트 초과는 plan 로드 오류). 모델 정책 슬롯에만 적용된다. summary는 모델 정책별로 열 두 개를 더한다: 판정이 성립한 게임 수(Shadow·On 공통), 그중 그 정책이 이긴 게임 비율(Shadow에서만 의미 있음 — 오항복률). `results.jsonl`의 세션 통계에 첫 성립 프레임을 남긴다 |

### 020-R3 타임아웃 규칙
| 상황 | 기준 | 처리 |
|---|---|---|
| 게임 길이 | 최대 게임 시간 60분(= 86,400 프레임) 초과 | **무승부** (보상 0, `TimeLimitDraw`) |
| 에이전트 무응답 | 별도 중단 기준 없음(2026-10-19 사용자 결정, C-9, AMB-020-4) | lockstep 학습은 에이전트 응답을 기다린다(001-R5). 멈춤이 OpenBW 드롭 시간(011-R18, 기본 10분)을 넘으면 클라이언트가 서로를 끊고, 두 세션이 모두 상대 퇴장으로 승리를 보고하면 `Aborted`로 처리한다(020-R13). 추론 배치가 5초를 넘으면 시각과 함께 로그를 남긴다(011 관측과 대응) |

### 020-R4 크래시 처리
| 상황 | 처리 |
|---|---|
| 런처 비정상 종료 | 에피소드를 `Aborted`로 표시하고 학습 데이터에서 제외 |
| 같은 설정이 연속 3회 크래시 | 해당 설정을 격리하고 경보 |

### 020-R11 plan 플레이어 종류
```toml
p0 = { kind = "t0", race = "terran" }                         # T0 스크립트 봇(021)
p1 = { kind = "oss", name = "iron", race = "terran" }          # T1 OSS 봇(021)
p0 = { kind = "model", checkpoint = "<dir>", sample = "argmax" }  # 모델 정책(테란 고정), sample = "argmax" | "stochastic"(050-R1)
```
- `checkpoint`가 상대 경로면 `URANIUM_SHARED_DIR` 기준으로 찾는다(061). 같은 체크포인트·샘플링 모드를 쓰는 세션은 추론 서버 하나를 공유한다.
- 추론 백엔드는 `uranium-env run --backend <cuda|vulkan|flex>`, 없으면 `.env` `URANIUM_AGENT_BACKEND`, 그것도 없으면 `vulkan`이다(060-R1).
- 추론 배치 파라미터(`max_batch`, `max_wait_ms`)는 CLI로 바꿀 수 있고 기본값은 측정해 정한다(TBD(M6)). lockstep이라 정확성에는 영향이 없고 처리량만 바뀐다. 정확성 제약은 OpenBW 드롭 시간(011-R18)뿐이다.

## 처리량 목표 (M3에서 실측)
| 지표 | 목표 |
|---|---|
| 동시 게임 수 | 호스트별 측정값(020-R10). M3 실측: 런처 수가 코어 수를 넘으면 브리지 p99가 10배 이상 늘어난다(`reports/m3.md`) |
| 평균 게임 길이 | 약 15분 = 21,600 프레임, 평균 delay 8프레임 기준 약 2,700 결정/게임 |
| 전체 결정 처리량 | TBD(M3). 추론 서버 한계가 병목일 것으로 예상([060](060-infra.md)) |

## Acceptance Criteria
- 020-AC1: 원격 게임 호스트 2대와 학습 호스트의 로컬 hostd(3대, 각자 측정한 동시 게임 수)로 T0·T1 혼합 200판(`config/plans/m3-remote.toml`)을 돌렸을 때 Aborted 비율이 1% 미만이다. 결과에는 호스트별·상대별 판당 시간을 기록한다. (2026-10-16 사용자 결정으로 "원격 2대·동시 16게임·1시간"에서 변경)
- 020-AC2: 같은 seed, 같은 상대, 같은 정책(결정론적 샘플링)으로 실행했을 때 결과가 재현된다. 상대는 결정론적인 T0·셀프플레이로 한정한다. OSS 봇은 내부 난수·시간 의존으로 재현되지 않으며 그대로 둔다(AMB-021-3).
- 020-AC3: hostd를 kill -9 한 뒤 재시작하면 고아 프로세스가 남지 않는다. 프로세스 그룹 정리로 보장한다.
- 020-AC4: `StartReplay`로 스트리밍한 RawFrame이 같은 리플레이의 저장 shard와 일치한다(040-AC6).
- 020-AC5: `uranium-env calibrate` 한 번으로 측정 파일이 만들어지고, `URANIUM_HOSTD_MAX_GAMES` 없이 재시작한 hostd가 그 값을 쓴다.

## Risks & Open Questions
- **slot 식별 (2026-10-17 실측)**: Windows 호스트(TCP LAN)에서 BWAPI 플레이어 id(`Hello.slot`)가 런처 순서와 다르게 배정됐다(T0 테란 대 T0 저그 4판 중 2판). 그래서 정책 라우팅(020-R11)과 시나리오 빌드(050)는 hostd가 붙이는 플레이어 이름 `uranium-p<i>`로 `GameSpec.players` 인덱스를 찾는다. 같은 게임의 두 세션이 승자를 다르게 보고한 판도 관측됐다. 원인(2026-10-17 조사): 패자 클라이언트가 마지막 프레임에 상대도 패배로 표시한다. 디싱크가 아니며 OpenBW 이탈 처리 타이밍 문제로 추정한다(재현: seed 30000, 모델 argmax 대 T0 테란, 1번 게임). 판정은 020-R13을 따르고, 보고서는 그런 판을 "winner disagree"로 따로 센다.
- **summary.md 형식 (2026-10-18 사용자 결정, 구현 메모)**: 모델 대 모델 대전(032-R17 맞대결)을 읽을 수 있게 `uranium-env` summary를 바꾼다. 대전 동작은 바꾸지 않는다. ① matchup 표의 p0·p1 모두 모델이면 `model <core> \`<run 폴더>/<체크포인트 폴더>\` (샘플링)`과 정책 버전(020-R5 `policy` 값)을 표시한다. core는 체크포인트 `meta.json`에서 읽는다. ② 서로 다른 두 모델 정책의 게임은 승자 세션의 `policy`로 p0·p1 승을 가른다(이전에는 "mirror decided"로만 셌다). 같은 정책끼리는 그대로 mirror다. ③ "Per policy (slot swaps combined)" 표: (정책, 상대)별로 슬롯 교대 matchup을 합친 승·패·무, Wilson 95%(050-R2), dropped를 맵별과 전체(`all`)로 낸다. ④ "Model sessions" 표에 core·run 열을 더한다. 테스트 `report::tests::model_vs_model_summary_per_policy_032_r17`.


- LAN 모드에서 두 런처의 프레임 동기 오버헤드를 측정해야 한다. 셀프플레이는 같은 에이전트 서버로 두 브리지가 접속하는 구조가 된다.
- 원격 호스트의 MPQ와 맵 배포는 사용자가 직접 준비한다. hostd는 시작 시 파일 해시를 검증한다.

- **호스트 구성 (확정 2026-10-15, AMB-020-1)**
  - 학습 호스트 외에 **게임 전용 Linux PC 2대 이상**을 쓴다.
  - 호스트별 코어 수와 동시 게임 수는 hostd `ListCapabilities`로 자동 감지하고, `config/hosts.toml`로 덮어쓸 수 있다.
  - Linux Vulkan 학습 호스트의 로컬 hostd는 기본 비활성이다. 그 호스트를 learner로 쓰는 동안에는 추론과 학습을 전담한다.
  - **Windows CUDA 호스트**는 학습·추론과 함께 로컬 hostd로 게임도 돌린다(2026-10-16 사용자 결정). 원격 Linux 게임 호스트와 함께 `URANIUM_HOSTS_FILE`에 둘 다 등록하고, 에이전트 서버는 원격 브리지가 접속할 수 있는 주소로 listen한다(`URANIUM_AGENT_LISTEN`, `URANIUM_AGENT_PUBLIC_ADDR`). 학습 중 로컬 동시 게임 수는 hosts 파일의 `max_games`로 줄일 수 있으며, 값은 학습 부하를 측정해 정한다(TBD(M5)).
  - 020-AC1은 원격 2대 + 학습 호스트 구성 200판으로 판정한다(2026-10-16 변경).

- **시간 초과 처리 (확정 2026-10-15, AMB-020-2)**
  - 60분(86,400 프레임)에 게임을 중단하고 **무승부**로 처리한다. 보상은 0이고 점수 판정은 하지 않는다.
  - `EndReason::TimeLimitDraw`로 기록한다.
  - 2026-10-15에 점수 판정안에서 무승부로 변경했다.

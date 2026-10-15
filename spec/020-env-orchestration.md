# 020 — Env Orchestration

- **Status**: Approved (2026-10-15)
- **Related**: [011](011-bridge-openbw.md), [021](021-opponent-pool.md), [042](042-rl-training.md), [060](060-infra-bc250.md)

## Purpose
여러 OpenBW 게임을 로컬이나 원격 호스트에서 병렬로 실행하고, 에피소드 수명주기를 관리한다.

## 구성
- **uranium-hostd**: 각 OpenBW 호스트에서 실행되는 데몬이다. 학습 PC의 `uranium-env`가 gRPC가 아닌 단순 TCP 제어 프로토콜(postcard)로 지시한다.
- **게임 쌍**: `BWAPILauncher` 2개로 구성된다.
  - P1: 우리 봇 브리지
  - P2: 상대. 우리 봇 브리지(셀프플레이) 또는 OSS 봇 `.so`
  - 두 런처는 Linux에서 `OPENBW_LAN_MODE=LOCAL`(같은 호스트 소켓), Windows에서 `OPENBW_LAN_MODE=TCP`(포트 2개)로 매칭한다.
- **uranium-env**: 학습 PC의 게임 스케줄러다. 에이전트 서버를 내장하고, 매치업 계획을 여러 hostd에 분배하고, 결과를 모은다.
- hostd와 env는 Linux와 Windows 모두에서 돈다(001-R8).

## Requirements
| ID | 요구사항 |
|---|---|
| 020-R1 | **MUST** hostd API: `StartGame{map, p1, p2, start_locs?, seed, agent_addrs, save_replay}`, `StartReplay{rep_bytes, perspective_slot, agent_addr, privileged: bool}`(040-R12 스트리밍, 011 재생 모드), `StopGame{id}`, `Status`, `ListCapabilities{max_concurrent_games, max_concurrent_replays, oss_bots}` |
| 020-R2 | **MUST** 게임별 작업 디렉터리와 포트를 격리한다. 같은 호스트에서 N개 게임이 동시에 돌아도 충돌이 없어야 한다 |
| 020-R3 | **MUST** 타임아웃 처리 |
| 020-R4 | **MUST** 크래시 처리: 런처 비정상 종료 시 에피소드를 `Aborted`로 표시하고 학습 데이터에서 제외한다. 같은 설정이 연속 3회 크래시하면 격리하고 경보를 낸다 |
| 020-R5 | **MUST** 에피소드 메타데이터를 기록한다: id, 맵, 시작 위치, 상대 종류·버전, 정책 버전, 결과, 길이, 리플레이 경로 |
| 020-R6 | **SHOULD** 호스트 목록과 호스트별 동시 게임 수는 설정 파일로 관리한다. 실제 주소가 들어가므로 커밋하지 않는 로컬 파일(`URANIUM_HOSTS_FILE`)에 두고, 형식 예시만 `config/hosts.example.toml`로 커밋한다(001-R10, 2026-10-15 변경) |
| 020-R8 | **MUST** hostd는 OS별 부분(LAN 매칭 설정, 프로세스 정리)을 플랫폼 계층으로 분리한다. Linux는 LOCAL 소켓과 프로세스 그룹 + `PR_SET_PDEATHSIG`를 쓰고, Windows는 TCP 포트 쌍(`URANIUM_HOSTD_PORT_BASE`부터 할당)과 Job Object(kill-on-close)를 쓴다 |
| 020-R9 | **MUST** hostd 설정은 `.env`에서 읽는다(001-R9): `URANIUM_HOSTD_LISTEN`, `URANIUM_BWAPI_DIR`, `URANIUM_BRIDGE_LIB`, `URANIUM_DATA_DIR`, `URANIUM_MAPS_DIR`, `URANIUM_HOSTD_MAX_GAMES`, `URANIUM_HOSTD_PORT_BASE`(Windows) |
| 020-R7 | **SHOULD** BC250 로컬 hostd도 지원한다. 이 경우 CPU 코어를 추론 서버와 나눠 쓰는 예산을 정한다([060](060-infra-bc250.md)) |

### 020-R3 타임아웃 규칙
| 상황 | 기준 | 처리 |
|---|---|---|
| 게임 길이 | 최대 게임 시간 60분(= 86,400 프레임) 초과 | **무승부** (보상 0, `TimeLimitDraw`) |
| 에이전트 무응답 | 30초 이상 | 해당 게임 중단 |

### 020-R4 크래시 처리
| 상황 | 처리 |
|---|---|
| 런처 비정상 종료 | 에피소드를 `Aborted`로 표시하고 학습 데이터에서 제외 |
| 같은 설정이 연속 3회 크래시 | 해당 설정을 격리하고 경보 |

## 처리량 목표 (M3에서 실측)
| 지표 | 목표 |
|---|---|
| 동시 게임 수 | 호스트당 `코어 수 - 1` |
| 평균 게임 길이 | 약 15분 = 21,600 프레임, 평균 delay 8프레임 기준 약 2,700 결정/게임 |
| 전체 결정 처리량 | TBD(M3). 추론 서버 한계가 병목일 것으로 예상([060](060-infra-bc250.md)) |

## Acceptance Criteria
- 020-AC1: 원격 호스트 2대에서 동시에 16게임을 1시간 동안 돌렸을 때 Aborted 비율이 1% 미만이다.
- 020-AC2: 같은 seed, 같은 상대, 같은 정책(결정론적 샘플링)으로 실행했을 때 결과가 재현된다.
- 020-AC3: hostd를 kill -9 한 뒤 재시작하면 고아 프로세스가 남지 않는다. 프로세스 그룹 정리로 보장한다.
- 020-AC4: `StartReplay`로 스트리밍한 RawFrame이 같은 리플레이의 저장 shard와 일치한다(040-AC6).

## Risks & Open Questions
- LAN 모드에서 두 런처의 프레임 동기 오버헤드를 측정해야 한다. 셀프플레이는 같은 에이전트 서버로 두 브리지가 접속하는 구조가 된다.
- 원격 호스트의 MPQ와 맵 배포는 사용자가 직접 준비한다. hostd는 시작 시 파일 해시를 검증한다.

- **호스트 구성 (확정 2026-10-15, AMB-020-1)**
  - BC250 외에 **게임 전용 Linux PC 2대 이상**을 쓴다.
  - 호스트별 코어 수와 동시 게임 수는 hostd `ListCapabilities`로 자동 감지하고, `config/hosts.toml`로 덮어쓸 수 있다.
  - BC250 로컬 hostd는 기본 비활성이다. BC250은 추론과 학습을 전담한다.
  - 020-AC1의 "호스트 2대·16게임"은 최소 기준으로 유지한다.

- **시간 초과 처리 (확정 2026-10-15, AMB-020-2)**
  - 60분(86,400 프레임)에 게임을 중단하고 **무승부**로 처리한다. 보상은 0이고 점수 판정은 하지 않는다.
  - `EndReason::TimeLimitDraw`로 기록한다.
  - 2026-10-15에 점수 판정안에서 무승부로 변경했다.

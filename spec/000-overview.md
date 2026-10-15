# 000 — Overview

- **Status**: Approved (2026-10-15)
- **Related**: [001](001-system-architecture.md), [050](050-evaluation.md), [060](060-infra-bc250.md), [061](061-shared-storage.md)

## Purpose
StarCraft: Brood War 1v1 밀리에서 이기는 테란 AI 봇을 만든다. 학습은 OpenBW(1.16.1 규칙)에서 하고, 최종 실행은 StarCraft: Remastered(SC:R)에서 한다.

## Scope
| 항목 | 고정값 |
|---|---|
| 플레이 종족 | Terran |
| 상대 종족 | Terran / Protoss / Zerg (랜덤 포함) |
| 맵 | 투혼(Fighting Spirit), 서킷브레이커(Circuit Breaker). 둘 다 128×128, 4인용 |
| 게임 모드 | 1v1 Melee, 표준 시작 자원·일꾼 |
| 학습 시뮬레이터 | OpenBW + BWAPI 4.2 포크 (1.16.1 게임 로직) |
| 실행 대상 | StarCraft: Remastered (samase 플러그인) |
| 학습 장비 | AMD BC250(Vulkan) **또는** CUDA GPU PC(Linux/Windows). Rust + Burn (2026-10-15 변경) |

## Non-goals
- 테란 이외 종족 플레이, 2v2 이상, UMS 맵.
- 대상 외 맵에서의 성능 보장. 다른 맵 리플레이는 사전학습 데이터로만 쓴다.
- 맵핵이나 메모리 조작으로 얻는 불공정 이득. 정책 관측은 **자기 플레이어 시야 기준**으로 제한한다. 학습 전용 상대 정보(`PrivilegedSummary`)는 OpenBW 학습 모드에서 승률 예측(value head) 학습에만 쓴다. 정책 입력과 실전(SC:R)에는 절대 쓰지 않는다([010](010-game-protocol.md) R10).
- 1.16.1이 아닌 다른 레거시 버전 지원.

## Goals (측정 가능)
| ID | 목표 |
|---|---|
| 000-G1 | 오픈소스 봇 풀 상대로 종족×맵 6개 조합 각각 승률 ≥ 80% (OpenBW) |
| 000-G2 | 셀프플레이 리그에서 BC 초기 정책 대비 승률 ≥ 90% |
| 000-G3 | SC:R 커스텀 게임에서 동일 정책이 크래시·디싱크 없이 완주. 상대는 사람 또는 SC:R 내장 AI |
| ~~000-G4~~ | **삭제(2026-10-15)**: APM 제한 없음 ([031](031-action-space.md)) |

- **목표 수치 (확정 2026-10-15, AMB-000-1)**: G1 80%, G2 90%를 초안 값 그대로 확정한다. M5 기준선 측정 후 조정을 제안할 수 있고, 조정하려면 사용자 확인이 필요하다. APM 제한은 없다.

- **실행 환경 (확정 2026-10-15, AMB-000-2)**: SC:R 실행은 **로컬/LAN 커스텀 게임에 한정**한다. 상대는 사람 또는 SC:R 내장 AI다. 배틀넷 공개 게임과 래더는 비목표다.

- **맵·종족 (확정 2026-10-15, AMB-000-3)**
  - 맵은 현재 리마스터 래더·대회에서 쓰는 **최신 공개 버전**을 기준으로 한다. 1.16.1(OpenBW) 호환성을 검증한 뒤 맵 해시를 고정한다. 해시 값은 TBD(M1).
  - 상대의 **Random 종족을 지원**한다. 정찰 전까지는 `enemy_race = unknown`으로 처리한다([030](030-observation.md)).

## 하드웨어 제약
- BC250 보드에서 가정하는 사항:
  - Zen2 6C/12T CPU, AMD iGPU(gfx1013 "Cyan Skillfish", 24CU로 알려짐. 정확한 아키텍처와 기능은 M1(a)에서 `vulkaninfo`로 확인)
  - 16GB GDDR6를 CPU와 GPU가 공유
  - BC250은 Vulkan 백엔드로 학습·추론한다(ROCm/HIP 불가)
- CUDA GPU PC(Linux 또는 Windows)에서도 Burn `cuda` 백엔드로 학습·추론할 수 있다(2026-10-15 변경, [060](060-infra-bc250.md))
- OpenBW 인스턴스는 기본적으로 **게임 전용 Linux PC 2대 이상**에서 실행한다. BC250 로컬 실행도 가능하다([020](020-env-orchestration.md)).
- 체크포인트와 학습 데이터는 **로컬 네트워크 공용 드라이브(별도 리눅스 서버, Linux 호스트는 NFS·Windows 호스트는 Samba)**에 두고, 학습하는 PC가 받아서 쓴다([061](061-shared-storage.md), 2026-10-15 변경). 저장 예산은 1TB 미만 기준을 공용 드라이브에 적용한다.
- SC:R은 Windows PC에서 실행한다.

## 용어집
| 용어 | 정의 |
|---|---|
| Bridge | 게임 프로세스 안에서 상태를 읽고 명령을 실행하는 얇은 래퍼. OpenBW용 `.so`, SC:R용 `.dll` |
| Agent | 모델 추론과 액션 결정을 맡는 Rust 프로세스. 게임 버전을 모른다 |
| RawFrame | 브리지가 보내는 버전 중립 게임 상태 스냅샷 ([010](010-game-protocol.md)) |
| Command | 에이전트가 브리지로 보내는 버전 중립 명령 |
| Executor | 정책의 의미 단위 액션을 Command로 풀어내는 규칙 계층 ([031](031-action-space.md)) |
| hostd | OpenBW 게임 인스턴스를 띄우고 관리하는 데몬 ([020](020-env-orchestration.md)) |
| BC | Behavior Cloning. 리플레이 기반 모방학습 |
| z | 리플레이에서 추출한 빌드 순서·유닛 구성 조건 벡터 (AlphaStar 방식) |
| 프레임 | 게임 로직 1틱. Fastest 속도에서 42ms (약 23.8 fps) |
| PrivilegedSummary | 학습 모드 전용 상대 실제 상태 요약. value head 학습에만 사용 ([010](010-game-protocol.md)) |
| T0 / T1 / T2 | 상대 계층: 자체 스크립트 봇 / 오픈소스 BWAPI 봇 / 셀프플레이 스냅샷 ([021](021-opponent-pool.md)) |
| 보류 (Deferred) | 사용자 확인을 거쳐 결정 시점을 뒤 마일스톤으로 미룬 항목. `[SPEC DEFERRED(Mx)]`로 표기 ([README](README.md)) |

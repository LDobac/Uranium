# 021 — Opponent Pool

- **Status**: Approved (2026-10-15)
- **Related**: [020](020-env-orchestration.md), [042](042-rl-training.md), [050](050-evaluation.md)

## Purpose
OpenBW에는 내장 컴퓨터 AI가 없다. 그래서 RL과 평가에 쓸 상대 풀을 정의한다. 대상은 테란 미러(TvT)와 TvP, TvZ다.

## 상대 계층
| Tier | 종류 | 용도 |
|---|---|---|
| T0 | **Uranium Scripted Bots**: 우리 proto/브리지 위에서 도는 Rust 규칙 기반 봇. 종족별 2~3개 빌드(러시/표준/테크) | 커리큘럼 초기, 회귀 테스트, BWAPI 버전 비호환 대비 fallback |
| T1 | **OSS BWAPI 봇**: OpenBW BWAPI 4.2로 Linux 빌드 가능한 공개 봇 | 다양한 플레이 스타일, 평가 기준선 |
| T2 | **Self-play 스냅샷**: 과거 정책 체크포인트 | 리그(PFSP), main/exploiter |

## Requirements
| ID | 요구사항 |
|---|---|
| 021-R1 | **MUST** T0 봇은 종족마다 최소 1개 이상, 밀리 1게임을 끝까지 수행할 수 있어야 한다(일꾼 생산, 확장, 공격) |
| 021-R2 | **MUST** T1 후보마다 기록할 항목: 라이선스, 원본 BWAPI 버전, 4.2 포팅 필요 여부, 빌드 스크립트, 커밋 해시 고정 |
| 021-R3 | **MUST** 모든 상대에는 `opponent_id@version`을 부여하고 에피소드 메타데이터에 기록한다 |
| 021-R4 | **SHOULD** 종족마다 T1 봇을 2개 이상 확보한다. 확보하지 못하면 T0 빌드 다양성으로 보완한다 |
| 021-R6 | **MUST** T0 봇은 [034](034-action-table.md)(테란 전용) 대신 [010](010-game-protocol.md)의 `Command`를 직접 생성한다. 저그 라바 생산, 드론 변태 건설, 버로우, 프로토스 워프 등 필요한 명령이 010에 없으면 010 변경을 사용자에게 제안한다(M3에서 확인) |
| 021-R7 | **MUST** T0·T1 봇은 일부 게임에서 Random 종족으로 시작할 수 있어야 한다(AMB-000-3, 042). 비율은 설정 파일로 관리한다 |
| 021-R5 | **MUST** 상대 봇의 맵핵(CompleteMapInformation)은 **기본 금지**한다. 학습 후반에 더 강한 상대가 필요할 때만 별도 상대 ID(`<bot>+maphack@ver`)로 리그에 추가할 수 있다. **평가(050)에서는 사용하지 않는다** (확정 2026-10-15, AMB-021-2) |

## T1 확정 목록 (2026-10-16 사용자 결정, 021-R2 기록 항목)
| 봇 | 종족 | 라이선스 | 원본 BWAPI | 4.2 포팅 | 빌드 스크립트 | 고정 |
|---|---|---|---|---|---|---|
| UAlbertaBot | T/P/Z | MIT | 4.4 계열 | 필요(패치 0005) | `tools/oss/build_ualbertabot.sh` | 커밋 `558899d8` |
| Steamhammer | Z(주력)·P·T | MIT | 4.4 계열 | 필요 | `tools/oss/build_steamhammer.sh` | 5.3.6 소스 zip sha256 |
| McRave | P/Z | MIT | 4.4 계열 | 필요(BWEM·BWEB 포함) | `tools/oss/build_mcrave.sh` | 커밋 고정 |
| Stardust | P | MIT + 대회 출전 금지 조건 | 4.4 계열(자체 OpenBW 구성) | 필요 | `tools/oss/build_stardust.sh` | 커밋 고정 |
| ZZZKBot | Z | LGPL-3 | 확인 필요 | 확인 필요 | `tools/oss/build_zzzkbot.sh` | 커밋 고정 |
| Iron | T | 확인 필요(사용 불가면 대체 후보 제시) | 4.1.2 | 필요(BWEM 포함) | `tools/oss/build_iron.sh` | AIIDE 제출 소스 sha256 |

- 실제 값과 포팅 내역은 `reports/oss_bot_porting.md`에 기록한다. 봇 하나가 막히면 사용자에게 보고하고 다음 봇으로 넘어간다.
- JVM 기반(JBWAPI) 봇은 제외한다.
- 상대 ID(021-R3): OSS 봇은 빌드 스크립트가 쓰는 `<bot>/VERSION`(`<bot>@<커밋 또는 버전>`), T0는 `t0-<race>@<uranium-scripted 버전>`이다.

## Acceptance Criteria
- 021-AC1: T0 봇 3종족이 서로 맞붙어 각 맵에서 50판을 크래시 없이 완주한다.
- 021-AC2: T1 봇 ≥ 3개가 OpenBW에서 빌드되고 우리 T0 봇 상대로 10판을 완주한다.

## Risks & Open Questions
- BWAPI 4.4 봇을 4.2로 포팅하는 비용을 확인해야 한다. 비용이 과하면 OpenBW 포크 자체를 4.4 API로 올리는 방안도 검토한다.

- **상대 봇 확보 방식 (확정 2026-10-15, AMB-021-1): 단계적 접근**
  1. **1단계 (M3)**: T0 스크립트 봇과, 4.2에서 바로 돌거나 포팅이 쉬운 봇(UAlbertaBot 등)으로 시작한다.
  2. **2단계**: BWAPI 4.4 봇(Steamhammer, McRave 등)은 봇별 포팅 비용을 측정해 `reports/oss_bot_porting.md`에 기록한다. 실제로 포팅을 진행할지, OpenBW를 4.4로 업그레이드할지는 **그 시점에 사용자에게 다시 확인**한다.
     - 2026-10-16 재확인: 4.2로 포팅한다. Steamhammer·McRave와 Stardust·ZZZKBot·Iron을 추가한다.
- T1 최종 후보 목록은 1단계 빌드 검증 결과와 함께 사용자 확인을 받는다.
- [SPEC AMBIGUOUS] AMB-021-3: OSS 봇의 결정론. 같은 seed로 2판을 돌리면 ZZZKBot만 같은 결과를 낸다. UAlbertaBot·Steamhammer·Iron은 시각 기반 난수를, Stardust·McRave는 실제 경과 시간 기반 계산량 조절을 쓴다(`reports/oss_bot_porting.md`). 020-AC2(재현)를 OSS 상대에도 요구할지, 난수 시드만 패치로 고정할지, 다양성으로 보고 그대로 둘지 결정이 필요하다.
- T0 봇이 너무 약하면 RL 초기 신호가 왜곡될 수 있다. BC 정책과 섞은 커리큘럼을 [042](042-rl-training.md)에서 다룬다.

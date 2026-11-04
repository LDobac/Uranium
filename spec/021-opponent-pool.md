# 021 — Opponent Pool

- **Status**: Approved (2026-10-15, 2026-10-17 개정: R7 Random 적용 범위를 여러 종족 지원 봇으로 한정, R3 T0 시나리오 빌드 상대 ID 분리, 사용자 결정. 2026-10-19: McRave 종족 표기를 실측(T/P/Z)과 `config/rl.toml`·040에 맞춤. 2026-11-03: run별 T1 학습 풀에서 봇 제외 가능(021-R4 유지), AMB-042-13. 2026-11-03 개정 2: R7 Random은 세 종족 봇에만, run별 봇 종족 제한(T1·셀프 run 2차 Steamhammer 프로토스 제외), AMB-042-14. 2026-11-03 개정 3(사용자 결정, AMB-042-15): 러시 커리큘럼 run에서 T0 `zerg_rush`를 RL 상대로 다시 씀(T0 그대로), 저그가 아닌 T1 제외, T1 설정 변형의 이름·기록, R7 러시 사다리 항목은 종족 고정. 2026-11-04(사용자 결정, AMB-042-17 Open): `rush-s3` 러시 단 UAlbertaBot 설정 변형 추가, 목록은 측정 뒤)
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
| 021-R7 | **MUST** 여러 종족을 지원하는 봇(T0, 여러 종족을 하는 T1 봇)은 일부 게임에서 Random 종족으로 시작할 수 있어야 한다(AMB-000-3, 042). 종족을 하나만 하는 T1 봇에는 적용하지 않는다(2026-10-17 사용자 결정). Random은 그 run 설정(`[[t1]] races`)에 세 종족이 모두 있는 봇에만 준다. 두 종족만 있으면 Random을 주지 않고 종족 샘플링으로 정한 목록 안 종족으로 둔다(Random이면 목록에 없는 종족이 나올 수 있어서, 2026-11-03 메인 임시 결정, AMB-042-14). 비율은 설정 파일로 관리하며 초기값은 20%다([042](042-rl-training.md) 초기 설정값). 042 러시 사다리 항목(`[[ladder]]`)은 종족이 고정이라 Random을 주지 않는다(2026-11-03, 메인 임시 결정, AMB-042-15) |
| 021-R5 | **MUST** 상대 봇의 맵핵(CompleteMapInformation)은 **기본 금지**한다. 학습 후반에 더 강한 상대가 필요할 때만 별도 상대 ID(`<bot>+maphack@ver`)로 리그에 추가할 수 있다. **평가(050)에서는 사용하지 않는다** (확정 2026-10-15, AMB-021-2) |

## T1 확정 목록 (2026-10-16 사용자 결정, 021-R2 기록 항목)
| 봇 | 종족 | 라이선스 | 원본 BWAPI | 4.2 포팅 | 빌드 스크립트 | 고정 |
|---|---|---|---|---|---|---|
| UAlbertaBot | T/P/Z | MIT | 4.4 계열 | 필요(패치 0005) | `tools/oss/build_ualbertabot.sh` | 커밋 `558899d8` |
| Steamhammer | Z(주력)·P·T | MIT | 4.4 계열 | 필요 | `tools/oss/build_steamhammer.sh` | 5.3.6 소스 zip sha256 |
| McRave | T/P/Z | MIT | 4.4 계열 | 필요(BWEM·BWEB 포함) | `tools/oss/build_mcrave.sh` | 커밋 고정 |
| Stardust | P | MIT + 대회 출전 금지 조건 | 4.4 계열(자체 OpenBW 구성) | 필요 | `tools/oss/build_stardust.sh` | 커밋 고정 |
| ZZZKBot | Z | LGPL-3 | 확인 필요 | 확인 필요 | `tools/oss/build_zzzkbot.sh` | 커밋 고정 |
| Iron | T | 확인 필요(사용 불가면 대체 후보 제시) | 4.1.2 | 필요(BWEM 포함) | `tools/oss/build_iron.sh` | AIIDE 제출 소스 sha256 |

- 실제 값과 포팅 내역은 `reports/oss_bot_porting.md`에 기록한다. 봇 하나가 막히면 사용자에게 보고하고 다음 봇으로 넘어간다.
- JVM 기반(JBWAPI) 봇은 제외한다.
- RL run의 T1 학습 풀은 그 run 설정 파일의 `[[t1]]` 목록이다. 처리량 등의 이유로 봇을 뺄 수 있으나 종족마다 2개 이상(021-R4)은 지킨다. 2026-11-03 T1·셀프 run은 McRave를 뺐다(lockstep에서 10~40배 느림, 메인 임시 결정, [042](042-rl-training.md) AMB-042-13). 봇의 종족도 run 설정 `races`로 줄일 수 있다(같은 조건). T1·셀프 run 2차(`t1self-s2`)는 Steamhammer를 테란·저그로만 둔다(2026-11-03 사용자 결정, AMB-042-14: Windows 빌드의 Steamhammer 프로토스가 8~10분에 같은 위치에서 결정적으로 크래시, `reports/m6_t1self_stop.md` 3-4절). 그 run의 프로토스 T1은 UAlbertaBot·Stardust 2개다(021-R4 충족).
- **러시 커리큘럼 run (`rush-s1`, 2026-11-03 사용자 결정, AMB-042-15)**: 봇 상대는 042 러시 사다리(`[[ladder]]`)의 저그 러시 상대뿐이다. T0 `zerg_rush` 빌드(`t0-zerg-rush`)를 RL 상대로 다시 쓰며 T0는 고치지 않는다(기억·탐색 결함 그대로. 무너진 뒤 T0가 마무리하지 못하는 판은 042-R14 붕괴 항복으로 패배 처리). T1은 저그 러시 봇과 그 설정 변형만 쓰고, 저그가 아닌 T1 봇은 이번 단계에서 뺀다(run 풀의 021-R4 조건 예외, 사용자 결정).
- **`rush-s3` 러시 단 추가 (2026-11-04 사용자 결정, AMB-042-17, 자문 B5)**: 지금 정책 기준 승률 50~60%인 UAlbertaBot 설정 변형 1~2개를 042 러시 사다리에 더한다. 만들기·기록·배포는 아래 "T1 설정 변형" 규칙 그대로다. 어떤 오프닝으로 만들지와 목록은 측정 뒤 사용자가 정한다(042 `rush-s3` `[SPEC AMBIGUOUS] AMB-042-17`).
- **T1 설정 변형 (2026-11-03 사용자 결정, AMB-042-15)**: 원본 봇 폴더를 복사해 오프닝 설정만 바꾼 상대다. 바이너리(`module.dll`·`.so`)는 원본과 같다(해시 일치). 폴더(봇) 이름은 `<bot><variant>`, `VERSION`은 `<bot><variant>@<원본 버전>-<variant>`이고 이것이 상대 ID(021-R3)다. 예: `ualbertabot9pool@558899d-9pool`, `steamhammeroverpool@5.3.6-overpool`. 021-R2 기록 항목(라이선스·원본 BWAPI·포팅·빌드 스크립트·고정)은 원본 봇 것을 따르고, 변형마다 원본 상대 ID, 바꾼 파일·줄·이전 → 이후 값, 바꾼 설정 파일의 sha256을 `reports/oss_bot_porting.md`에 더 적는다(첫 두 변형의 내용은 `reports/m6_rush_ladder.md` 3절). 변형은 설정을 바꾼 종족으로만 쓴다. 학습·점검에 쓰는 변형은 그 게임을 돌리는 모든 게임 호스트에 같은 내용으로 둔다(메인 임시 결정: 손으로 복사하지 않고 원본 폴더와 바꿀 줄로 만드는 스크립트를 `tools/oss/`에 둔다).
- 상대 ID(021-R3): OSS 봇은 빌드 스크립트가 쓰는 `<bot>/VERSION`(`<bot>@<커밋 또는 버전>`), T0 표준 빌드는 `t0-<race>@<uranium-scripted 버전>`이다. [050](050-evaluation.md) E-Regression용 시나리오 빌드는 빌드 태그를 붙여 `t0-<race>-<build>@<버전>`으로 따로 구분한다(2026-10-17 사용자 결정): `t0-zerg-rush`, `t0-terran-drop`, `t0-protoss-dt`, `t0-zerg-muta`. 승률 표·PFSP·E-Elo는 이 ID를 상대 단위로 쓴다.

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
- **OSS 봇 결정론 (확정 2026-10-16, AMB-021-3): 그대로 둔다.** 같은 seed로 2판을 돌리면 ZZZKBot만 같은 결과를 낸다. UAlbertaBot·Steamhammer·Iron은 시각 기반 난수를, Stardust·McRave는 실제 경과 시간 기반 계산량 조절을 쓴다(`reports/oss_bot_porting.md`). 판마다 달라지는 것은 상대 다양성으로 보고 패치하지 않는다. 우리 쪽(T0·브리지·엔진)의 결정론은 유지하며, 020-AC2는 결정론적인 상대(T0, 셀프플레이)로 검증한다.
- T0 봇이 너무 약하면 RL 초기 신호가 왜곡될 수 있다. BC 정책과 섞은 커리큘럼을 [042](042-rl-training.md)에서 다룬다.

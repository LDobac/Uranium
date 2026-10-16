# 042 — RL Training & League

- **Status**: Approved (2026-10-15, 2026-10-16 개정: 042-R10 Ctrl+C 정상 종료)
- **Related**: [020](020-env-orchestration.md), [021](021-opponent-pool.md), [032](032-model-architecture.md), [033](033-reward.md), [050](050-evaluation.md), [060](060-infra-bc250.md)

## Purpose
BC 정책을 출발점으로 셀프플레이와 리그를 거쳐 승률을 극대화한다.

## 구조 (actor-learner)
```
[uranium-env actors] ×M 게임 ──obs──► [inference server (학습 호스트 GPU, batched)] ──action──► actors
         └── trajectory(T=64: obs, privileged, action, behaviour logits, rewards, core_state0) ──► replay queue ──► [learner (학습 호스트 GPU)]
learner ──► 새 가중치 (매 K 스텝) ──► inference server
```
학습 호스트는 BC250(Vulkan) 또는 CUDA PC다([060](060-infra-bc250.md), 2026-10-15 변경). 한 호스트의 GPU 하나로 추론과 학습을 모두 하므로 GPU 시간을 분할한다. 학습 중에는 추론 지연이 늘어나지만 lockstep이라 정확성에는 영향이 없다. 게임 PC는 LAN으로 학습 호스트의 에이전트 서버에 접속한다(001-R6). 체크포인트·리그 스냅샷·승격 체크포인트는 공용 드라이브에 저장하며, 학습 시작·재개 절차와 run 잠금은 [061](061-shared-storage.md)을 따른다.

## 알고리즘
- **V-trace**(IMPALA) off-policy 보정: actor 정책 지연을 허용한다.
- **UPGO** policy gradient를 보조로 쓴다.
- **KL(π ‖ π_BC)** 정규화: 계수 초기값 0.1, 점진 감쇠, 하한 0.01.
- 엔트로피 보너스를 헤드별로 둔다.
- 대안: PPO(clip 0.2)를 같은 인터페이스로 구현해 비교한다. 초기 소규모 실험에서 더 안정적인 쪽을 채택한다.

## 리그
| 에이전트 | 상대 샘플링 | 목적 |
|---|---|---|
| Main | PFSP(전체 스냅샷 + T0/T1) 50%, 셀프 35%, T1 15% | 최종 정책 |
| Main Exploiter | 현재 Main만 | Main 약점 발견 |
| League Exploiter | PFSP 전체 | 전략 다양성 |

- 스냅샷 주기: 매 N 업데이트 또는 승률 수렴 시.
- 상대 종족 분포: T/P/Z 균등. 승률이 가장 낮은 종족을 가중 샘플링한다. T0·T1 봇은 일부 게임에서 Random 종족을 고른다(AMB-000-3). 비율은 설정 파일로 관리한다.
- **종족 구성의 한계**: 리그의 학습 에이전트(Main, Exploiter, 스냅샷)는 모두 테란이므로 셀프플레이로는 TvT만 연습된다. TvP·TvZ 상대는 T0·T1 봇뿐이다(아래 AMB-042-2).
- 맵: 투혼과 서킷브레이커 균등, 시작 위치는 랜덤.

## Requirements
| ID | 요구사항 |
|---|---|
| 042-R1 | **MUST** 학습 재개(resume)를 지원한다. 저장 대상은 가중치, 옵티마이저, 리그 상태, 승률 테이블, RNG 상태다. 다른 학습 호스트(Vulkan↔CUDA)에서도 재개할 수 있어야 한다(061-R10, 061-AC5) |
| 042-R2 | **MUST** 에피소드 결과·보상 항목·KL·엔트로피·처리량을 TensorBoard 호환 로그 또는 CSV로 남긴다 |
| 042-R3 | **MUST** Aborted 에피소드는 학습에서 제외한다(020-R4) |
| 042-R4 | **MUST** 정책 지연(behaviour vs target 버전 차이)을 모니터링하고, 임계값을 넘는 trajectory는 버린다 |
| 042-R5 | **SHOULD** 커리큘럼 단계 C0(T0 봇 위주) → C1(T0+T1+셀프) → C2(풀 리그) 전환은 승률 조건으로 자동화한다 |
| 042-R6 | **MUST** 체크포인트 승격은 [050](050-evaluation.md)의 회귀 게이트를 통과해야 한다 |
| 042-R7 | **MUST** 에이전트 3개(Main, Main Exploiter, League Exploiter)는 학습 호스트의 learner를 시분할로 쓴다. Main이 학습 시간의 50% 이상을 차지한다. 비율은 설정 파일로 관리한다 |
| 042-R8 | **MUST** RL 학습 시퀀스 길이는 T=64다(032-R5와 동일) |
| 042-R9 | **MUST** KL 계수, 엔트로피 계수, 커리큘럼 전환 승률 기준, 스냅샷 주기는 튜닝 대상이다. 설정 파일로 관리하고 체크포인트에 기록한다 |
| 042-R10 | **MUST** 정상 종료(Ctrl+C, 2026-10-16 사용자 결정). 첫 Ctrl+C(SIGINT, Windows는 Ctrl+C 이벤트)를 받으면 다음 순서로 멈춘다. ① 새 게임 배정을 멈춘다. ② 진행 중인 게임을 중단하고 `Aborted`로 기록해 학습에서 뺀다(042-R3). ③ 체크포인트를 저장하고 push한다(061-R4 "정상 종료 시"). ④ 잠금을 해제한다(061-R5). 두 번째 Ctrl+C를 받으면 저장 없이 즉시 종료한다. 게임 PC 구성(hosts 파일)을 바꾸려면 정상 종료 후 재시작한다. 재시작하면 hosts 파일을 새로 읽고 마지막 체크포인트에서 재개한다(R1) |

## Acceptance Criteria
- 042-AC1: 작은 설정(1맵, T0 1종족)에서 BC 정책 대비 승률 개선을 재현 가능하게 보인다. 3 seed 평균으로 측정한다.
- 042-AC2: 72시간 연속 학습 중 크래시나 메모리 누수 없이 resume이 가능하다.
- 042-AC3: Main이 이전 승격 체크포인트 대비 승률 ≥ 55%일 때만 승격한다.
- 042-AC4: 학습 중 Ctrl+C 한 번이면 체크포인트 push와 잠금 해제까지 마치고 종료한다. 이어서 hosts 파일의 게임 PC를 하나 빼거나 더해 재시작하면, `--take-over` 없이 같은 step에서 재개하고 바뀐 호스트 구성으로 게임을 배정한다(042-R10).

## Risks & Open Questions
> **[SPEC DEFERRED(M6)] AMB-042-2**: TvP·TvZ 실력이 T0·T1 봇 수준에서 정체될 수 있다(셀프플레이는 TvT만 가능). 지금은 T0·T1 P/Z 봇으로 진행한다(2026-10-15 결정). M6 중간 점검 때 아래 내용을 사용자에게 보고하고 대응을 다시 확인받는다.
> - 보고 내용: TvP·TvZ 승률이 정체되었는지, 또는 T1 봇 상대 승률이 포화되었는지
> - 대응 후보: 프로토스·저그를 플레이하는 상대 전용 정책을 추가할지

- 학습 호스트 GPU 하나에서 추론과 학습이 경쟁한다. 다른 PC GPU를 추론 전용으로 쓰는 옵션은 [060](060-infra-bc250.md) R5를 따른다.

- **학습 예산 (확정 2026-10-15, AMB-042-1)**
  - 기간 제한 없이 성능이 오르는 동안 계속 학습한다.
  - 리그는 Main + Main Exploiter + League Exploiter 전체 구성을 쓴다.
  - 042-R1(resume)과 060-R6(자동 재시작)이 필수 전제다.
  - 학습은 커리큘럼 C2 이후에도 종료 조건 없이 진행한다. 중단은 사용자가 결정한다.

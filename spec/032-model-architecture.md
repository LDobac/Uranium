# 032 — Model Architecture (Burn)

- **Status**: Approved (2026-10-15, 2026-10-17 개정: R8 호스트별 정밀도 — CUDA 호스트 f32(TF32) 허용, 사용자 결정 / 추론 서버 Core 상태 슬롯(032-R14), 샘플링 모드·헤드별 log-prob(032-R15) / 추론 서버 가중치 교체·슬롯 상태 읽기(032-R16, 3단계 계획 승인) / 2026-10-18 개정: 코어 비교 절차(032-R17, AC7, AMB-032-6). 2026-10-19 개정(BC 전 리뷰 결정, 사용자 확정): AC8 backward 허용 오차(H), autotune·fusion 채택(G-D5, 041-R15), R17 ② val 고정 seed 셔플 500게임(G-D8), F_c를 030과 일치, 장비 표기 중립화(001-R10, G-D9))
- **Related**: [030](030-observation.md), [031](031-action-space.md), [033](033-reward.md), [034](034-action-table.md), [041](041-imitation-learning.md), [042](042-rl-training.md), [060](060-infra.md)

## Purpose
Linux Vulkan 호스트의 학습 예산에 맞춘 경량 AlphaStar 계열 정책·가치 네트워크를 Burn으로 정의한다. 입력 계약은 [030](030-observation.md), 액션 구조는 [031](031-action-space.md)과 [034](034-action-table.md)를 따른다.

## 확정 사항 (2026-10-15)
| 항목 | 결정 |
|---|---|
| 정책 구조 | **단일 end-to-end 정책** (AMB-032-1 Resolved) |
| 매치업 | **단일 공유 모델 + 상대 종족 임베딩** (AMB-032-2 Resolved) |
| 전략 조건 z | **포함** (AMB-032-3 Resolved) |
| 시간 축 코어 | **LSTM으로 시작**. M5에서 Transformer 메모리(GTrXL 계열)와 비교해 채택하고, `Core` trait로 교체 가능하게 한다 (AMB-032-4 Resolved) |
| 관계 학습 | 유닛↔건물↔유닛 관계는 EntityEncoder(공간 축 트랜스포머)가 담당한다. 거리 기반 attention bias를 추가한다 |

## 1. 전체 구조
```
entity_* ─► EntityEncoder(Transformer 3L + distance bias) ─► E[B,N,256], e_pool[B,256]
                                   └─ scatter(Linear(E)→32ch @128²) ─┐
spatial[B,26,128,128] ─────────────────────────────────────────────►┴► SpatialEncoder(U-Net enc) ─► S8[B,128,8,8], skips{64²,32²,16²}, s_pool[B,256]
scalar, scalar_cat, z_* ─► ScalarEncoder ─► c[B,256]
concat[e_pool, s_pool, c](768) ─► Core (LSTM 512) ─► h[B,512]
h ─► Heads (자기회귀, ar[B,256]):
      1 ActionType ─► 2 Delay ─► 3 Queued ─► 4 Selected(≤12) ─► 5 TargetUnit | 6 Location(128²)
h, z, enemy_race, privileged(학습 전용) ─► ValueHeads {win, build, units, value}
```

## 2. 모듈 상세

### 2.1 EntityEncoder
- **입력**:
  - `Emb(234,64)[unit_type]` ⊕ `Emb(3,8)[owner]` ⊕ `Emb(190,32)[order]` ⊕ `Emb(A,16)[last_action]` ⊕ `entity_cont(F_c = 76, 030 §1·§2)`
  - → `Linear(→256)`
- **Transformer**:
  - 3층, d=256, heads=8, FFN=1024, pre-LayerNorm, GELU
  - dropout은 BC에서 0.1, RL에서 0
  - padding key는 마스크 처리한다.
- **distance bias**:
  - 각 층의 attention logit에 `bias[l, h, bucket(i,j)]`를 더한다.
  - `bucket` = (같은 소유자 여부: 2) × (픽셀 거리 log bucket: 16) = 32개.
  - 파라미터 수는 3층 × 8 head × 32. 비용은 [B,8,N,N] 덧셈 1회다.
- **출력**:
  - `E[B,N,256]`
  - `e_pool` = `Linear(512→256)`(concat(masked_mean(E | self), masked_mean(E | enemy)))

### 2.2 SpatialEncoder
- **scatter**: `Linear(256→32)(E)`를 `entity_cell`에 scatter-add하고 셀별 개수로 나눈다(평균) → [B,32,128,128]
- **stem**: concat(spatial, scatter) → `Conv3x3(58→32)` @128²
- **다운샘플**:
  - 각 단계는 `Conv3x3 stride2 + ResBlock(2×Conv3x3)`
  - 128²→64²(64ch) → 32²(64ch) → 16²(128ch) → 8²(128ch)
  - 64², 32², 16² 출력은 skip으로 보관한다.
- **bottleneck**: ResBlock×2 @8²×128 → `S8`
- **풀링**: `s_pool` = `Linear(8192→256)`(flatten(S8))

### 2.3 ScalarEncoder
- 연속값: `Linear(F_s→256)` → ReLU → `Linear(256→256)`
- 범주형:
  - enemy_race `Emb(4,32)`
  - map_family `Emb(M,16)`
  - start_loc `Emb(8,8)`
- z:
  - `Emb(A,32)[z_build]`(20개, 위치 임베딩 포함) → masked mean
  - `Linear(U→32)[z_units]`
  - → `Linear(→64)`, `z_present=false`면 0 벡터
- 출력: concat → `Linear(→256)` = `c`. `c`는 ActionTypeHead의 GLU 게이트 문맥으로도 쓴다.

### 2.4 Core
- 인터페이스:
  ```rust
  trait Core: Module {
      type State;
      fn init_state(&self, batch: usize, device: &Device) -> Self::State;
      fn step(&self, x: Tensor<2>, s: Self::State) -> (Tensor<2>, Self::State);          // 추론
      fn forward_seq(&self, x: Tensor<3>, s0: Self::State) -> (Tensor<3>, Self::State);  // 학습 [B,T,·]
      fn select_rows(s: &Self::State, rows: Tensor<1, Int>) -> Self::State;              // 추론 서버: 슬롯 → 배치 (032-R14)
      fn scatter_rows(s: Self::State, rows: Tensor<1, Int>, sub: Self::State) -> Self::State; // 배치 → 슬롯
  }
  ```
- 기본 구현은 `LstmCore`다.
  - `Lstm(768→512)` 1층
  - 학습은 T=64 truncated BPTT로 하고, 시작 상태는 actor가 기록한 값을 쓴다.
  - 에피소드 시작 시 0으로 리셋한다.
- 비교 대상(M5 ablation)은 `GtrxlCore`다.
  - 최근 M=64 step의 core 입력에 causal self-attention 2층(d=512)을 쓴다.
  - GRU-gated residual을 쓰고, 추론 시 KV cache를 유지한다.
  - 채택은 짧은 BC 비교 학습으로 정한다(032-R17, AMB-032-6). 추론 지연·메모리는 M5에서 측정했다(CUDA에서 두 코어가 거의 같다, `reports/m5_model.md`).
  - 비교가 끝나기 전까지 BC 본학습(041)은 시작하지 않는다. 비교 결과는 `reports/m6_core_ablation.md`에 기록한다.
- 장기 정보(적 건물 기억, 관측된 적 최대 수, 정찰 기록)는 관측([030](030-observation.md))에서 명시적으로 유지한다. 코어는 단기·중기 문맥을 담당한다.

### 2.5 Heads
| # | Head | 구조 | 출력 | `ar` 갱신 |
|---|---|---|---|---|
| 1 | ActionType | `Linear(512→256)` → ResMLP×2 → GLU(gate=c) → `Linear(256→A)` | a | `ar = Linear(h→256) + Linear(onehot(a)→256)` |
| 2 | Delay | `MLP(256→128→16)` | 16 bucket {1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32} | `ar += Emb(16,256)[d]` |
| 3 | Queued | `MLP(256→64→2)`, `uses_queued=false`면 0으로 고정 | q | `ar += Emb(2,256)[q]` |
| 4 | Selected | key=`Linear(E→64)`, END key 1개 추가. query는 `GRUCell(64)`로 step마다 갱신(초기값 `Linear(ar→64)`). step마다 이미 고른 유닛을 마스크하고, `max_select` 도달 시 END만 허용. 건물 규칙(034-R7): 첫 선택이 건물이면 다음 step은 END만 허용하고, 첫 선택이 건물이 아니면 이후 step에서 건물을 마스크한다. 최대 12 step | sel ⊆ N | `ar += Linear(64→256)(mean key[sel])` |
| 5 | TargetUnit | `Linear(ar→64)` · key(`Linear(E→64)`, 4와 별도) | u | 종료 |
| 6 | Location | `Linear(ar→16·8·8)` reshape ⊕ S8 → ConvT 업샘플 8²→16²→32²→64²→128²(skip concat) → `Conv1x1→1` | 128² logits | 종료 |

- 헤드 5와 6은 `target` 종류에 따라 하나만 쓴다([034](034-action-table.md)).
- 모든 헤드의 마스크는 [034](034-action-table.md) 테이블로 GPU에서 gather해 만든다(§5).

### 2.6 ValueHeads
- 입력:
  - `h`
  - z 임베딩
  - enemy_race 임베딩
  - **privileged**(학습 전용): [010](010-game-protocol.md)의 `PrivilegedSummary`(상대 실제 자원, 업그레이드, 테크, 유닛 구성)를 쓴다. RL에서는 OpenBW lockstep 학습 모드(셀프플레이와 오픈소스 봇 상대 모두, 011-R5)에서 얻고, BC에서는 리플레이 재생 모드(011-R12)에서 얻는다. 실전(SC:R)에는 존재하지 않는다.
- 구조: 공유 `MLP(→256)`에 출력 4개를 둔다.
  - win: tanh
  - build, units, value: 선형
  - 보상 성분과의 대응은 [033](033-reward.md)을 따른다.
- privileged 입력이 정책 헤드로 흘러가는 경로가 그래프 상 존재하지 않아야 한다(032-R11).

## 3. Requirements
| ID | 요구사항 |
|---|---|
| 032-R1 | **MUST** Burn 0.22의 런타임 `Device` 방식으로 작성한다. 모델 코드는 백엔드에 무관하고, 실행 시 Device로 고른다. 학습·추론은 `Device::vulkan(..)`(Vulkan 호스트) 또는 `Device::cuda(n)`(CUDA 호스트)에 `.autodiff()`를 붙여 쓴다. 테스트는 `Device::flex()`(CPU)를 쓴다. (변경 이력: 0.22에서 백엔드 제네릭 제거·ndarray deprecated, 2026-10-15 CUDA 추가) |
| 032-R2 | **MUST** 기본 크기는 약 12M 파라미터(§4)로 하고, 허용 범위는 10M–40M이다. 확장은 d_model 384, Transformer 4층 순으로 하며 M1(a) 벤치로 확정한다 |
| 032-R3 | **MUST** 헤드는 자기회귀다. 학습에서는 teacher forcing(레이블 액션 주입), 추론에서는 샘플값을 주입한다 |
| 032-R4 | **MUST** 모든 헤드에 마스크를 적용한다. 마스크된 logit은 `-1e9`로 둔다. 샘플링 결과는 항상 마스크를 만족해야 한다 |
| 032-R5 | **MUST** Core 상태는 에피소드 간 리셋한다. 학습 시퀀스는 T=64 truncated BPTT로 한다 |
| 032-R6 | **MUST** 체크포인트에 함께 저장할 항목: 가중치, `UraniumNetConfig`, feature 스키마 버전, action 스키마 버전([034](034-action-table.md)), 정규화 통계 |
| 032-R7 | **MUST** 추론은 batch B개 게임 동시 처리를 지원한다. 엔티티는 배치 내 최대 N으로 동적 패딩한다 |
| 032-R8 | **MUST** 정밀도는 학습 호스트에서 **더 빠른 쪽**을 쓴다(2026-10-17 개정). **Linux Vulkan 호스트**: 학습과 추론은 **f16 혼합 정밀도**로 한다. M1(a)에서 그 호스트가 f16을 지원하고 matmul 오차가 4.5e-4, 속도가 약 1.5배임을 확인했다(2026-10-15 결정). 마스터 가중치·옵티마이저 상태·손실·softmax 누적은 f32로 두고, 손실 스케일링을 적용한다. 수치 문제가 생기면 해당 모듈만 f32로 되돌리고 기록한다. **CUDA 호스트**: f32(CUDA 내부 TF32 matmul)를 기본으로 한다. M5 실측에서 f16이 f32보다 느렸다(추론 중앙값 59 → 78ms, 학습 70 → 54 samples/s, `reports/m5_model.md`). f16 혼합 정밀도 학습 구현은 Vulkan 호스트 학습 전까지 미룬다(2026-10-17 사용자 결정). **예외**: Vulkan 호스트에서 도는 모든 BC 학습(ablation (A)·041-R5, 다른 PC에서 넘겨받아 이어 하는 S1·S2 구간 포함)은 f16 혼합 정밀도 미구현으로 f32로 학습한다(2026-10-18 사용자 결정, 처음 (A)만이던 예외를 넓힘) |
| 032-R13 | **MUST** distance bias는 bias 테이블 gradient를 대규모 scatter-add로 모으지 않는 방식으로 구현한다. 예: bucket one-hot과 matmul, 또는 bucket별 사전 합산. M1(a)에서 기존 방식의 역전파가 정방향 대비 약 40배 느림(310ms)을 확인했다. 정방향 대비 역전파 시간이 5배 이내여야 한다 |
| 032-R9 | **MUST** `act()`는 헤드 6개의 샘플링을 **한 번의 GPU forward 안에서** 수행한다(Gumbel-max). CPU 왕복은 없어야 한다 |
| 032-R10 | **MUST** joint log-prob와 엔트로피는 `action_table`상 사용된 헤드만 합산한다 |
| 032-R11 | **MUST** privileged 입력은 ValueHeads에만 연결한다. `act()` 시그니처는 privileged 텐서를 받지 않는다 |
| 032-R12 | **MUST** Selected 헤드의 BC 레이블은 정규 순서(entity index 오름차순) 시퀀스 + END로 만든다 |
| 032-R14 | **MUST** 추론 서버(`uranium-agent`)는 세션마다 고정 슬롯을 두고 Core 상태를 `[capacity, …]` 텐서 하나에 보관한다. 요청이 모인 세션의 행만 `select_rows`로 꺼내 `act`를 돌리고 `scatter_rows`로 되돌린다. 세션이 시작될 때 그 슬롯을 0으로 리셋한다(032-R5). 배치로 처리한 결과는 세션을 하나씩 처리한 결과와 같아야 하고(허용 오차 이내), 세션 사이에 상태가 섞이지 않는다 (2026-10-17) |
| 032-R15 | **MUST** `act()`의 샘플링 모드는 확률적(temp 1)과 결정론적(temp 0 = 헤드마다 argmax)을 둔다(050-R1). `ActOutput`은 joint log-prob과 함께 헤드별 log-prob [B,6]을 돌려준다. RL actor는 이 값을 behaviour log-prob로 기록한다(042) (2026-10-17) |
| 032-R16 | **MUST** 추론 서버는 실행 중에 가중치를 교체할 수 있다(042 K update마다). 교체는 다음 배치부터 적용되고, 응답마다 정책 버전을 붙인다. 세션은 자기 슬롯의 Core 상태를 host 데이터로 읽을 수 있다(042 core_state0). Core는 상태를 host 데이터로 내보내고 다시 만드는 연산을 제공하며, 왕복은 비트 단위로 같다 (2026-10-17) |
| 032-R17 | **MUST** 시간 축 코어(LSTM vs GTrXL)는 BC 본학습 전에 짧은 비교 학습으로 고른다(2026-10-18 사용자 결정, AMB-032-6). ① 학습: 코어마다 S1을 5,300 update 1회(seed 1개)씩 돌린다. 장비는 혼합 유동 운영이다(2026-10-18 사용자 결정): Linux Vulkan 호스트(f32)와 Windows CUDA 호스트(TF32 matmul)가 run 큐 작업자(061-R13·R14)로 두 run을 나눠 맡고 서로 이어받는다. 장비별 update 구간은 `runs/<id>/devices.jsonl`(061-R15)에 남고, 두 run의 장비 비율을 최종 집계해 리포트에 적는다. 두 run의 장비 비율이 크게 다르면 Vulkan f32와 CUDA TF32의 수치 차이가 비교에 섞일 수 있다(공정성 주의). 데이터 순서(`data.seed`)와 그 밖의 설정(Burn autotune·fusion 포함, 041-R15)은 같다. 학습률 스케줄은 짧은 run 전용으로, warmup 5%(265) 뒤 cosine이 5,300에서 끝난다(`config/bc_core_ablation.toml`). ② 오프라인 지표: `bc-eval --split val --games 500`(두 코어 같은 500게임). 500게임은 val 분할을 **고정 seed로 셔플**해 앞에서 고른다(출처 편중을 줄이기 위해, 2026-10-19 사용자 결정 G-D8, AMB-032-7). 학습 seed는 1개로 유지하고, 그 한계(seed 간 편차를 모름)는 리포트에 적는다. test 분할은 041-AC1용이라 쓰지 않는다. ③ 대전: 두 체크포인트 맞대결(TvT, 맵당 50판, 25판마다 슬롯 교대, `config/plans/core-ablation-h2h.toml`)과 각자 T0 상대 승률(상대 종족 3 × 맵 2 조합당 50판, 코어당 300판·합계 600판, 사용자 확정 2026-10-18, `config/plans/core-ablation-t0.toml`). 샘플링은 확률적(temp 1)이고 z는 없다. 대전 700판의 리플레이를 모두 저장한다(두 plan `save_replay = true`, `<out>/replays/g<id>.rep`, 2026-10-18 사용자 결정). ④ 판정: 규칙을 미리 정하지 않는다. 결과를 사용자에게 보고하고 사용자가 고른다. ⑤ 비교 run의 체크포인트·로그·평가 출력은 지우지 않는다(비교 설정은 061-R6 `keep_checkpoints` 예외로 전부 보존, 2026-10-19 사용자 결정 G-D6) |

## 4. 파라미터·연산 예산 (기본 구성, N=256)
| 모듈 | 파라미터 | FLOPs/step (B=1) |
|---|---|---|
| EntityEncoder (임베딩, 3층, bias) | ~2.6M | ~1.2 GFLOP |
| SpatialEncoder (s_pool 2.1M 포함) | ~3.0M | ~0.8 GFLOP |
| ScalarEncoder + z | ~0.4M | 무시 |
| Core LSTM(768→512) | ~2.6M | 무시 |
| Heads | ~2.5M | ~0.4 GFLOP |
| ValueHeads | ~1.0M | 무시 |
| **합계** | **~12M** | **~2.5 GFLOP** |

배치 16 추론 시 약 40 GFLOP이 필요하다. 실효 처리량 1–2 TFLOPS를 가정하면 20–40ms다. 실측은 M1(a)에서 한다.

## 5. 마스크 파이프라인
- CPU(env)가 계산하는 것: `type_mask[B,A]`, `unit_cap[B,N,K]`, `loc_class_mask[B,L,128,128]` ([030](030-observation.md), [034](034-action-table.md))
- GPU에 상수로 올리는 것: `action_table` 텐서 `select_cap[A]`, `target_cap[A]`, `loc_class[A]`, `max_select[A]`, `uses_*[A]`
- GPU에서 헤드 진행 중에 샘플된 a로 gather한다:
  - `sel_mask = gather(unit_cap, select_cap[a]) ∧ is_self ∧ entity_mask`. Selected step마다 이미 고른 유닛과 건물 규칙(034-R7: 첫 선택이 건물이면 END만, 아니면 건물 제외)을 더 적용한다. 건물 여부는 엔티티별 `is_building`(떠 있는 건물·벙커·터렛 포함)으로 GPU에 올린다
  - `tgt_mask = gather(unit_cap, target_cap[a]) ∧ entity_mask`
  - `loc_mask = gather(loc_class_mask, loc_class[a])`

## 6. 학습·추론 인터페이스
```rust
#[derive(Config)] pub struct UraniumNetConfig { entity, spatial, scalar, core, heads, value }
#[derive(Module)] pub struct UraniumNet<C: Core> { entity, spatial, scalar, core: C, heads, value }

// Burn 0.22: 백엔드는 텐서·모듈이 놓인 Device가 결정한다 (032-R1)
impl<C: Core> UraniumNet<C> {
    /// 추론 1 step. 샘플링은 GPU 내부(Gumbel-max), 온도 temp(0 = argmax, 032-R15). 사용된 헤드의 log-prob 합, 헤드별 log-prob, value 반환.
    pub fn act(&self, obs: ObsBatch, state: C::State, temp: f32) -> ActOutput<C::State>;
    /// 학습 [B,T]. teacher forcing. 헤드별 masked logits + value 반환.
    pub fn forward_seq(&self, obs: ObsSeq, labels: ActionSeq, priv_: Option<PrivSeq>, s0: C::State) -> SeqOutput;
}
```
- 인코더와 헤드는 B×T를 펼쳐 병렬로 계산하고, Core만 T를 따라 순차 계산한다.
- **BC**: 헤드별 CE × 사용 마스크([041](041-imitation-learning.md))
- **RL**: `forward_seq`의 target logits와 actor behaviour log-prob로 V-trace, UPGO, 헤드별 KL(π‖π_BC)을 계산한다([042](042-rl-training.md)).
- 처리량 구현 메모(비규범, 2026-10-18, `reports/m6_bc_preflight.md`): 결과가 같은 계산 순서 변경만 한다.
  - LSTM Core: Burn `Lstm`과 같은 파라미터·수식을 쓰고, T step의 입력 투영과 4 게이트를 한 번의 matmul로 계산한다(step당 [B,H]×[H,4H] 1회).
  - teacher forcing의 Location 헤드: location을 쓰는 행만 계산한다. 다른 행의 `dists.target_loc`은 0이고 R10 사용 마스크로 곱해져 결과에 영향이 없다.
  - Burn kernel autotune·fusion은 CUDA·Vulkan 모두 켠다(2026-10-19 사용자 결정, G-D5, 041-R15, AMB-032-7). 2026-10-18 측정에서 CUDA는 일부 conv gradient가 Flex 대비 2~5% 달라졌고(끈 상태 0.13%), Vulkan은 거의 바뀌지 않았다(`reports/m6_bc_preflight.md` 4차). 판정은 AC1(forward)·AC8(backward)을 따른다.
  - 분석 도구: `uranium-train bc-profile`(모듈별 forward 표시, gradient 절단 비교).

## Acceptance Criteria
- 032-AC1: Flex(CPU) 대비 같은 가중치·입력의 forward 출력(value, 헤드별 log-prob) 차이가 Vulkan에서 ≤ 1e-4(f32), CUDA에서 ≤ 1e-2다.
  - CUDA 허용 오차 (AMB-032-5 Resolved, 2026-10-16 사용자 결정: 허용 오차를 늘린다). Burn 0.22(cubek-matmul)는 CUDA tensor core matmul에서 f32 입력을 TF32(가수 10비트, 누적은 f32)로 계산하고 끌 설정이 없다. 실측(Windows CUDA 호스트, 기본 크기 B16·N256): value ≤ 6.0e-5, action_type ≤ 3.4e-5, delay ≤ 2.5e-4, selected(최대 12 step log-prob 합) ≤ 3.9e-3, 정책 분포 KL ≤ 1.2e-7, greedy 액션 일치 48/48. 1e-2는 최대 실측의 약 2.5배 여유다. 영향 분석은 `reports/m5_model.md`.
- 032-AC2: 추론 지연은 batch=16, N=256 기준으로 측정한다(p95). CUDA 호스트에서는 같은 호스트의 Vulkan보다 빠르면 통과한다(2026-10-16 사용자 결정: 40ms를 넘더라도 Vulkan보다 빠르면 된다). Linux Vulkan 호스트 기준(초안 40ms)은 그 호스트 실측 후 사용자에게 다시 확인한다.
- 032-AC3: 학습 처리량 batch 32×T64 기준 ≥ TBD(M1) samples/s이고, 메모리는 [060](060-infra.md) 예산 이내다. CUDA 호스트에서는 같은 호스트의 Vulkan보다 빠르면 통과한다(2026-10-16 사용자 결정).
- 032-AC4: 무작위 마스크 10만 건에서 샘플링된 액션이 모두 마스크를 만족한다.
- 032-AC5: `act()` 경로의 연산 그래프에 privileged 텐서가 없다(타입 수준 보장 + 테스트).
- 032-AC6: `LstmCore`와 `GtrxlCore`가 같은 `Core` trait으로 교체되고, 둘 다 AC1–AC4를 통과한다.
- 032-AC7: 032-R17 비교의 두 run이 같은 update 수에서 끝나고, val 지표·맞대결·T0 승률(Wilson 95%)이 `reports/m6_core_ablation.md`에 기록되며, 사용자가 채택할 코어를 정한다.
- 032-AC8: backward(학습) 정확도(2026-10-19 사용자 결정, H·D-a, AMB-032-7): Flex(CPU) 대비 같은 가중치·입력·레이블의 BC 손실 차이가 AC1 허용 오차 안이고, 파라미터마다 gradient 차이의 상대 L2(‖g_dev − g_flex‖ ÷ ‖g_flex‖, 분모는 가장 큰 파라미터 gradient norm의 1e-3 이상으로 둔다)가 **Vulkan ≤ 1e-3**이다. **CUDA ≤ 1e-2**(2026-10-19 실측 기준(사용자 결정 D-a): autotune·fusion을 켠 이 테스트를 CUDA 호스트에서 6회 돌린 최댓값 4.9e-3(범위 3.7e-3~4.9e-3)에 약 2배 여유를 두고 올림. `reports/m6_bc_preflight.md` 5차. autotune·fusion 채택(041-R15)은 유지). dropout은 0으로 둔다. 측정은 학습과 같은 Burn 설정(autotune·fusion, 041-R15)으로 한다. 테스트는 `crates/uranium-model/tests/cross_device.rs`의 gradient 비교다. forward(AC1 Vulkan 1e-4)와 기준이 다른 이유: gradient는 반올림 누적이 많아 커널을 바꾸지 않은 Vulkan f32에서도 약 5e-4가 나온다(`reports/m6_bc_preflight.md` 4차).

## Risks & Open Questions
- **M1(a) 실측 (2026-10-15, Linux Vulkan 호스트 f32, `reports/bench_<label>.md`)**:
  - 추론 근사(B16·N256) 87ms로 032-AC2(40ms)에 미달한다.
  - 학습 약 40 프레임/s로, BC 1회독에 약 34일이 걸린다.
  - 사용자는 f16만 채택했다. 공간 64² 축소, 인코더 축소, 프레임 샘플링은 채택하지 않았다.
  - M5에서 f16으로 실측했을 때도 AC2와 학습 속도가 부족하면 사용자에게 결과를 보고하고 추가 대응을 다시 확인받는다.
- Burn의 scatter-add, GRUCell, attention bias 주입, ConvTranspose2d가 Vulkan에서 지원·성능 미확인이다. M1(a)에서 확인한다([060](060-infra.md)).
  - 대체안: scatter → one-hot matmul, attention은 직접 구현
- **AC8 CUDA 기준 (2026-10-19, 사용자 결정 D-a)**: 2026-10-18 측정(`reports/m6_bc_preflight.md` 4차)에서 CUDA autotune+fusion의 일부 conv gradient 차이는 2.2e-2(autotune만 5.3e-2, 끈 상태 1.3e-3)였다. 처음 제안한 CUDA 1e-2는 이 값보다 작으므로 쓰지 않는다. autotune·fusion을 켠 AC8 테스트로 다시 측정하고(학습 코드 수정 단계), 그 값에 여유를 둔 기준을 사용자 확인을 거쳐 이 AC에 적는다. → 2026-10-19 재측정(5차): 같은 테스트·같은 지표(파라미터별 상대 L2)에서 3.7e-3~4.9e-3(6회)이라 CUDA 1e-2로 적었다(실측 기준, D-a). 4차의 2.2e-2와 차이가 나는 원인은 확인하지 못했다(그 뒤 모델 코드 수정, autotune의 측정 시간 기반 커널 선택 등이 후보). 다른 세션·호스트에서 1e-2를 넘으면 이 기준을 다시 사용자에게 확인받는다.
- 128² Location 업샘플이 지연을 지배할 수 있다. 대체안은 64² 출력 + 2×2 subcell 헤드다.
- Selected 12-step 루프 지연이 문제가 될 수 있다. 대체안은 단일 step multi-label(sigmoid)이다.

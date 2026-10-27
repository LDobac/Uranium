# 032 — Model Architecture (PyTorch)

- **Status**: Approved (2026-10-15, 2026-10-23 개정: Burn에서 PyTorch로 전환(사용자 결정, AMB-000-4) — R1 PyTorch·CPU 기준, R6 형식은 061-R10, R8 장비별 정밀도(CUDA bf16 혼합·ROCm f32, AMB-032-9), R13 구현 방식 자유화, R14·R16 Python 추론 스레드·K/V 캐시 조건, R18 Burn 표현 정리, R19 최적화 원칙(AMB-032-10), §2.4·§6 Python 인터페이스, AC1·2·3·6·8·9 기준 장치 변경, AC10 Burn 동등성. 2026-10-17 개정: R8 호스트별 정밀도 — CUDA 호스트 f32(TF32) 허용, 사용자 결정 / 추론 서버 Core 상태 슬롯(032-R14), 샘플링 모드·헤드별 log-prob(032-R15) / 추론 서버 가중치 교체·슬롯 상태 읽기(032-R16, 3단계 계획 승인) / 2026-10-18 개정: 코어 비교 절차(032-R17, AC7, AMB-032-6). 2026-10-19 개정(BC 전 리뷰 결정, 사용자 확정): AC8 backward 허용 오차(H), autotune·fusion 채택(G-D5, 041-R15), R17 ② val 고정 seed 셔플 500게임(G-D8), F_c를 030과 일치, 장비 표기 중립화(001-R10, G-D9). 2026-10-19 T7 개정(사용자 결정): §5 대상 마스크에서 선택 유닛 제외(034-R9), `self_target[A]` 상수). 2026-10-20 개정(사용자 결정, AMB-032-8): AC2 기준은 CUDA 호스트에만 적용하고 Linux Vulkan 호스트는 실측 기록만 하는 대체 장비로 둔다, R17 주 장비 규칙(CUDA 호스트가 유휴면 주, 아니면 Linux Vulkan 호스트가 주, 060-R8). 2026-10-21 개정: 032-R18 win_open 헤드(사용자 결정, AMB-031-6). 2026-10-23 개정: 코어 비교 결과로 GTrXL 채택(032-R17 ④, AC7, 사용자 결정); 2026-10-24 개정: R17 ⑥ PyTorch 2차 코어 비교(사용자 결정); 2026-10-25 개정: §2.7·R20 구조 변형 비교(지형 붙이기·스칼라 토큰·attention 요약·바깥 LSTM과 그 조합, 사용자 결정, AMB-032-11), AC11·AC12)
- **Related**: [030](030-observation.md), [031](031-action-space.md), [033](033-reward.md), [034](034-action-table.md), [041](041-imitation-learning.md), [042](042-rl-training.md), [060](060-infra.md)

## Purpose
학습 호스트(특히 Linux ROCm 호스트의 메모리, 060-R3)의 예산에 맞춘 경량 AlphaStar 계열 정책·가치 네트워크를 PyTorch로 정의한다(2026-10-15~22는 Burn). 입력 계약은 [030](030-observation.md), 액션 구조는 [031](031-action-space.md)과 [034](034-action-table.md)를 따른다.

## 확정 사항 (2026-10-15)
| 항목 | 결정 |
|---|---|
| 정책 구조 | **단일 end-to-end 정책** (AMB-032-1 Resolved) |
| 매치업 | **단일 공유 모델 + 상대 종족 임베딩** (AMB-032-2 Resolved) |
| 전략 조건 z | **포함** (AMB-032-3 Resolved) |
| 시간 축 코어 | **GTrXL**(2026-10-23 코어 비교로 채택, 032-R17). LSTM으로 시작해 짧은 BC 비교 학습으로 골랐고, 두 코어 모두 같은 `Core` 인터페이스 구현으로 남긴다 (AMB-032-4·AMB-032-6 Resolved) |
| 관계 학습 | 유닛↔건물↔유닛 관계는 EntityEncoder(공간 축 트랜스포머)가 담당한다. 거리 기반 attention bias를 추가한다 |

## 1. 전체 구조
```
entity_* ─► EntityEncoder(Transformer 3L + distance bias) ─► E[B,N,256], e_pool[B,256]
                                   └─ scatter(Linear(E)→32ch @128²) ─┐
spatial[B,26,128,128] ─────────────────────────────────────────────►┴► SpatialEncoder(U-Net enc) ─► S8[B,128,8,8], skips{64²,32²,16²}, s_pool[B,256]
scalar, scalar_cat, z_* ─► ScalarEncoder ─► c[B,256]
concat[e_pool, s_pool, c](768) ─► Core (GTrXL 2층 d512 | LSTM 512) ─► h[B,512]
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
  ```python
  class Core(nn.Module):  # 2026-10-23: Python(PyTorch). 이전 Burn trait와 같은 연산
      def init_state(self, batch: int, device) -> State
      def step(self, x: Tensor, s: State) -> tuple[Tensor, State]          # 추론 [B,d_in] → [B,H]
      def forward_seq(self, x: Tensor, s0: State) -> tuple[Tensor, State]  # 학습 [B,T,d_in] → [B,T,H]
      def select_rows(s: State, rows: Tensor) -> State                     # 추론: 슬롯 → 배치 (032-R14)
      def scatter_rows(s: State, rows: Tensor, sub: State) -> State        # 배치 → 슬롯
      def state_to_host(s: State) -> HostState / state_from_host(h) -> State  # 032-R16, 비트 단위 왕복
  ```
- **채택 코어는 `GtrxlCore`다**(2026-10-23 사용자 결정, 032-R17). BC 본학습(041)과 RL(042)은 GTrXL로 한다. `LstmCore`는 처음 구현이자 비교 기준으로 남긴다.
- `LstmCore`:
  - `Lstm(768→512)` 1층
  - 학습은 T=64 truncated BPTT로 하고, 시작 상태는 actor가 기록한 값을 쓴다.
  - 에피소드 시작 시 0으로 리셋한다.
- `GtrxlCore`(M5 ablation 비교 대상, 2026-10-23 채택):
  - 최근 M=64 step의 core 입력에 causal self-attention 2층(d=512)을 쓴다.
  - GRU-gated residual을 쓰고, 추론 시 KV cache를 유지한다. (2026-10-23 메모: Burn 구현은 층 입력 M개를 상태로 두고 K/V를 매 step 다시 계산했다. PyTorch에서는 K/V를 캐시할 수 있고, 조건은 032-R16을 따른다.)
  - 채택은 짧은 BC 비교 학습으로 정한다(032-R17, AMB-032-6). 추론 지연·메모리는 M5에서 측정했다(CUDA에서 두 코어가 거의 같다, `reports/m5_model.md`).
  - 비교가 끝나기 전까지 BC 본학습(041)은 시작하지 않는다. 비교 결과는 `reports/m6_core_ablation.md`에 기록한다.
  - 비교 결과(2026-10-23, 같은 update 수): 맞대결 100판 GTrXL 42승·LSTM 13승·무 45, T0 상대 승률 GTrXL 약 10%(274판)·LSTM 2.7%(300판), val 500게임 모든 지표 GTrXL 우세. GTrXL은 엔티티가 많은 장면에서 추론이 무거워 T0 대전 27판이 실제 시간 상한에 걸렸다(실시간 추론 부담은 060·012에서 다시 본다).
- 장기 정보(적 건물 기억, 관측된 적 최대 수, 정찰 기록)는 관측([030](030-observation.md))에서 명시적으로 유지한다. 코어는 단기·중기 문맥을 담당한다.

### 2.5 Heads
| # | Head | 구조 | 출력 | `ar` 갱신 |
|---|---|---|---|---|
| 1 | ActionType | `Linear(512→256)` → ResMLP×2 → GLU(gate=c) → `Linear(256→A)` | a | `ar = Linear(h→256) + Linear(onehot(a)→256)` |
| 2 | Delay | `MLP(256→128→16)` | 16 bucket {1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32} | `ar += Emb(16,256)[d]` |
| 3 | Queued | `MLP(256→64→2)`, `uses_queued=false`면 0으로 고정 | q | `ar += Emb(2,256)[q]` |
| 4 | Selected | key=`Linear(E→64)`, END key 1개 추가. query는 `GRUCell(64)`로 step마다 갱신(초기값 `Linear(ar→64)`). step마다 이미 고른 유닛을 마스크하고, `max_select` 도달 시 END만 허용. 건물 규칙(034-R7): 첫 선택이 건물이면 다음 step은 END만 허용하고, 첫 선택이 건물이 아니면 이후 step에서 건물을 마스크한다. 최대 12 step | sel ⊆ N | `ar += Linear(64→256)(mean key[sel])` |
| 5 | TargetUnit | `Linear(ar→64)` · key(`Linear(E→64)`, 4와 별도). 4에서 고른 유닛은 대상에서 뺀다(034-R9, `self_target` 액션 제외, §5) | u | 종료 |
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
- **win_open**(2026-10-21, 032-R18): 상대 정보 없이 추정한 승패 기대값(tanh, −1 패 ~ +1 승). 입력은 detach한 `h`, z 임베딩, enemy_race 임베딩뿐이고 공유 MLP와 별도인 `MLP(→256)→1`을 쓴다. 031-R12 항복 판정에만 쓰고 RL advantage에는 쓰지 않는다. win 헤드는 학습 때 항상 privileged를 받는데 실전에는 그 입력이 없어(0 벡터) 실전 승률로 쓸 수 없기 때문에 둔다.

### 2.7 구조 변형 (비교용, 032-R20)
- 목적: 기본 구성의 구조 약점을 메우는 변형을 짧은 BC 비교 학습으로 시험한다(2026-10-25 사용자 결정, AMB-032-11). 약점은 ① e_pool이 유닛 수백 개를 아군 평균·적 평균 두 개로 줄여 개수·소수 유닛·배치가 묻힘 ② 유닛 토큰이 지형(맵 채널에만 있음)과 스칼라 문맥(자원·전략 조건·시간)을 모름 ③ Core가 최근 64 step 밖을 잊고, 세 인코더가 함께 쓰는 학습되는 장기 상태가 없음이다. 채택 결정 전까지 BC 본학습(041)·RL(042)의 모델은 기본 구성(모두 꺼짐)이다.
- 설정은 BC 설정 TOML의 `[model]` 표다(키가 없으면 꺼짐). 실행 중 동작을 바꾸는 옵션이 아니라 **만들 모델 구조를 고르는 스위치**이고, 체크포인트 메타의 모델 설정(061-R10)에 남는다. 네 스위치는 서로 독립이라 어떤 조합도 만들 수 있다.
  - `terrain` (bool): 유닛이 있는 칸(`entity_cell`)의 정적 공간 채널 10개(030 §3의 height ×3, walkable, buildable, start_loc, base_loc, choke, wall_candidate, valid_map)를 읽어 `Linear(10→256)`으로 투영하고 EntityEncoder 입력 투영 결과에 더한다. 관측(030)은 바꾸지 않는다. 대칭 증강 뒤에도 공간 채널과 `entity_cell`이 함께 변환되므로 그대로 맞는다.
  - `ctx_token` (bool): 스칼라 요약 `c`를 `Linear(256→256)`으로 토큰 하나로 만들어 EntityEncoder 토큰 끝에 붙인다. 이 토큰과 다른 토큰 사이의 distance bias는 전용 bias 버킷 하나(층·head별 학습값)를 쓴다. Transformer 뒤에는 E에서 빼므로 e_pool, scatter, Selected·TargetUnit key에 들어가지 않는다.
  - `attn_pool` (bool): e_pool을 attention 요약으로 바꾼다. 학습되는 질문 벡터(입력에서 계산하지 않는 파라미터, 모든 게임·step에서 같음) `Q_self[4,256]`·`Q_enemy[4,256]`(쪽당 4개, 2026-10-25 사용자 확인)가 각자의 집합(`entity_mask` 안의 아군·적 엔티티, 추가 토큰 제외)에 masked attention을 한다. K = `Linear(256→256)(E)`, V = `Linear(256→256)(E)`, 8 head × 32, 점수 [B,8,N]를 유닛 축으로 softmax, 출력 [B,8,256]을 이어 `Linear(2048→256)`으로 줄인다. 집합이 비면(예: 적 미관측) 그 질문의 출력은 0이다. 출력 크기 256은 그대로라 Core 입력은 768이다.
  - `outer` (정수, 0 = 꺼짐. 비교에서는 32): **바깥 LSTM**. 단위는 결정(모델 호출 1회 = 1 step)이다.
    - Core 상태에 더하는 것(게임마다): `h_o[B,256]`, `c_o[B,256]`(`LSTMCell(512→256)`의 상태), 현재 구간의 Core 출력 합 `h_sum[B,512]`, 결정 카운터 `n[B]`.
    - 결정마다 Core 출력 `h[B,512]`를 `h_sum`에 더하고 `n`을 1 늘린다. `n`이 `outer`의 배수가 되면 `(h_o, c_o) = LSTMCell(h_sum / outer, (h_o, c_o))`, `h_sum = 0`. 새 `h_o`는 다음 결정부터 쓴다.
    - `h_o`가 들어가는 곳: ① EntityEncoder 토큰 하나 `Linear(256→256)(h_o)` → [B,1,256](`ctx_token`과 같은 방식, 전용 bias 버킷. 두 토큰을 함께 켜면 `ctx_token`, `outer` 순) ② SpatialEncoder stem 출력(ReLU 뒤) s[B,32,128,128]의 FiLM: `Linear(256→64)(h_o)` → γ[B,32]·β[B,32], s·(1+γ) + β ③ Core 입력[B,768]에 `Linear(256→768)(h_o)`를 더함.
    - 에피소드 시작 때 다른 Core 상태와 함께 0으로 리셋한다.
- 공통 규칙
  - 모두 꺼지면 모델 구조·파라미터 목록·출력이 기본 구성과 같다(비트 동일).
  - 새 모듈은 032-R18 ③처럼 전역 RNG를 쓰지 않는 고정 seed 생성기로 초기화한다. Linear는 U(−k, k), k = 1/√fan_in. 바깥 LSTM 셀은 PyTorch LSTMCell 기본과 같은 U(−k, k), k = 1/√256(모든 가중치·bias). 추가 토큰의 bias 행은 기존 bias 표와 같은 N(0, 0.02). 질문 벡터는 U(−k, k), k = 1/√d(2026-10-25 리뷰 반영, 구현과 일치시킴). 기존 모듈은 기본 구성과 같은 순서로 만들고, 대체되는 모듈(`attn_pool`의 기존 pool Linear)도 같은 자리에서 만든 뒤 쓰지 않는다. 그래서 변형을 켜도 공유 모듈의 초기 가중치는 기본 구성과 같다.
  - 학습: `outer`가 켜지면 BC 학습 묶음([B, T = 64])을 `outer` 길이 구간으로 나눠 차례로 계산한다(구간 안은 지금처럼 병렬). 구간 k는 구간 k−1 끝에서 갱신된 `h_o`를 쓰고, 그 갱신은 구간 k−1의 `h`와 기울기로 이어진다. 묶음 사이에서는 다른 Core 상태처럼 상태를 detach한다. 묶음 시작은 항상 카운터가 `outer`의 배수인 지점이다(게임 시작부터 64 step 단위, 재개는 0부터).
  - 추론: 바깥 갱신은 카운터로 정해지므로 같은 수식으로 step마다 계산할 수 있고, 추론 graph 캡처(032-R19)와 맞는다.
  - 최적화 경로(직접 작성 커널, `torch.compile`, 추론 graph, 엔티티 재계산 041-R17)는 변형에서도 같은 수식이어야 한다(032-R19).
  - RL 학습기(042, 세그먼트 경계가 행마다 다름)의 `outer` 지원은 채택된 뒤에 따로 정한다.

## 3. Requirements
| ID | 요구사항 |
|---|---|
| 032-R1 | **MUST** PyTorch(`python/uranium/model`)로 작성한다. 모델 코드는 장치에 무관하고 실행 시 `torch.device`로 고른다. 학습·추론은 CUDA 호스트에서 CUDA 빌드, Linux ROCm 호스트에서 ROCm 빌드의 `cuda` 장치를 쓴다(ROCm 빌드도 `torch.cuda` API를 쓴다). 기준 구현과 단위 테스트는 CPU에서 돈다. 장치별 차이가 있는 경로(예: ROCm에서 MIOpen LSTM backward 실패로 자체 LSTM 커널 사용)는 같은 수식이어야 하고 AC1·AC8로 확인한다. 직접 작성한 CUDA·HIP 커널은 같은 소스를 두 백엔드에서 빌드하고, 순수 PyTorch 기준 구현과 끄는 설정을 함께 둔다(060-R1). (변경 이력: 2026-10-15 Burn 0.22 `Device`(Vulkan·CUDA·Flex), 2026-10-23 PyTorch로 전환, AMB-000-4) |
| 032-R2 | **MUST** 기본 크기는 약 12M 파라미터(§4)로 하고, 허용 범위는 10M–40M이다. 확장은 d_model 384, Transformer 4층 순으로 하며 M1(a) 벤치로 확정한다 |
| 032-R3 | **MUST** 헤드는 자기회귀다. 학습에서는 teacher forcing(레이블 액션 주입), 추론에서는 샘플값을 주입한다 |
| 032-R4 | **MUST** 모든 헤드에 마스크를 적용한다. 마스크된 logit은 `-1e9`로 둔다. 샘플링 결과는 항상 마스크를 만족해야 한다 |
| 032-R5 | **MUST** Core 상태는 에피소드 간 리셋한다. 학습 시퀀스는 T=64 truncated BPTT로 한다 |
| 032-R6 | **MUST** 체크포인트에 함께 저장할 항목: 가중치, `UraniumNetConfig`, feature 스키마 버전, action 스키마 버전([034](034-action-table.md)), 정규화 통계. 파일 형식은 061-R10을 따른다(2026-10-23: 가중치는 safetensors) |
| 032-R7 | **MUST** 추론은 batch B개 게임 동시 처리를 지원한다. 엔티티는 배치 내 최대 N으로 동적 패딩한다 |
| 032-R8 | **MUST** 정밀도는 장비별로 정한다(2026-10-23 사용자 결정, AMB-032-9). **CUDA 호스트**: 학습·추론 모두 **bf16 혼합 정밀도**다. 행렬곱·conv는 bf16 autocast로, 마스터 가중치·옵티마이저 상태·손실·log-softmax·softmax 누적·value 출력은 f32로 둔다. bf16은 지수 범위가 f32와 같아 손실 스케일링은 쓰지 않는다. 수치 문제가 생기면 해당 모듈만 f32로 되돌리고 기록한다. **Linux ROCm 호스트**: f32다(2026-10-23 실측에서 bf16 autocast가 f32보다 학습 2.9~3.8배·추론 3.3배 느림, `reports/torch_vs_burn.md`). 체크포인트는 정밀도와 무관하게 f32 가중치로 저장하므로 두 호스트가 서로 이어 학습할 수 있다. 정밀도는 `meta.json`과 `devices.jsonl`(061-R15)에 기록한다. GPU 계산은 비결정적일 수 있다(atomic 덧셈 순서 등, 032-R19). 비트 단위 재현은 CPU에서만 판정한다(061-AC1). 아래는 Burn 시기의 변경 이력이다. (이력) 2026-10-17 개정, 정밀도는 학습 호스트에서 **더 빠른 쪽**을 쓴다. **Linux Vulkan 호스트**: 학습과 추론은 **f16 혼합 정밀도**로 한다. M1(a)에서 그 호스트가 f16을 지원하고 matmul 오차가 4.5e-4, 속도가 약 1.5배임을 확인했다(2026-10-15 결정). 마스터 가중치·옵티마이저 상태·손실·softmax 누적은 f32로 두고, 손실 스케일링을 적용한다. 수치 문제가 생기면 해당 모듈만 f32로 되돌리고 기록한다. **CUDA 호스트**: f32(CUDA 내부 TF32 matmul)를 기본으로 한다. M5 실측에서 f16이 f32보다 느렸다(추론 중앙값 59 → 78ms, 학습 70 → 54 samples/s, `reports/m5_model.md`). f16 혼합 정밀도 학습 구현은 Vulkan 호스트 학습 전까지 미룬다(2026-10-17 사용자 결정). **예외**: Vulkan 호스트에서 도는 모든 BC 학습(ablation (A)·041-R5, 다른 PC에서 넘겨받아 이어 하는 S1·S2 구간 포함)은 f16 혼합 정밀도 미구현으로 f32로 학습한다(2026-10-18 사용자 결정, 처음 (A)만이던 예외를 넓힘) |
| 032-R13 | **MUST** distance bias의 EntityEncoder 역전파 시간은 정방향의 5배 이내여야 한다(장비마다). 구현 방식은 자유다: bucket 인덱스로 표를 gather(embedding), bucket one-hot과 matmul, bucket별 사전 합산, 또는 attention과 bias를 합친 직접 작성 커널(커널 안에서 위치로 bucket을 계산해 [B,H,N,N] bias를 만들지 않는 방식 포함). 어느 방식이든 결과는 같은 수식이다(float 반올림과 GPU atomic 순서 차이만 허용, 032-R19). (변경 이력: 2026-10-15 Burn의 scatter-add 역전파가 정방향의 약 40배(310ms)라 "scatter-add 금지, one-hot matmul 등"으로 정했다. 2026-10-23 PyTorch에서는 gather 역전파가 빠르고 ROCm에서는 one-hot matmul 역전파가 오히려 느려서(엔티티 인코더 fwd+bwd 303.7ms 대 gather 79.9ms, `reports/torch_vs_burn.md`) 방식 제한을 풀었다) |
| 032-R9 | **MUST** `act()`는 헤드 6개의 샘플링을 **한 번의 GPU forward 안에서** 수행한다(Gumbel-max). CPU 왕복은 없어야 한다 |
| 032-R10 | **MUST** joint log-prob와 엔트로피는 `action_table`상 사용된 헤드만 합산한다 |
| 032-R11 | **MUST** privileged 입력은 ValueHeads에만 연결한다. `act()` 시그니처는 privileged 텐서를 받지 않는다 |
| 032-R12 | **MUST** Selected 헤드의 BC 레이블은 정규 순서(entity index 오름차순) 시퀀스 + END로 만든다 |
| 032-R14 | **MUST** 추론(batch broker의 슬롯, 001-R6)은 세션마다 고정 슬롯을 두고, Python 추론 스레드가 Core 상태를 GPU의 `[capacity, …]` 텐서에 보관한다(2026-10-23: 이전에는 Rust `uranium-agent` 추론 스레드). broker는 슬롯이 열리고 닫힌 것을 배치와 함께 알린다. 요청이 모인 세션의 행만 `select_rows`로 꺼내 `act`를 돌리고 `scatter_rows`로 되돌린다. 세션이 시작될 때 그 슬롯을 0으로 리셋한다(032-R5). 배치로 처리한 결과는 세션을 하나씩 처리한 결과와 같아야 하고(허용 오차 이내), 세션 사이에 상태가 섞이지 않는다 (2026-10-17) |
| 032-R15 | **MUST** `act()`의 샘플링 모드는 확률적(temp 1)과 결정론적(temp 0 = 헤드마다 argmax)을 둔다(050-R1). `ActOutput`은 joint log-prob과 함께 헤드별 log-prob [B,6]을 돌려준다. RL actor는 이 값을 behaviour log-prob로 기록한다(042) (2026-10-17) |
| 032-R16 | **MUST** 추론 서버는 실행 중에 가중치를 교체할 수 있다(042 K update마다). 교체는 다음 배치부터 적용되고, 응답마다 정책 버전을 붙인다. 세션은 자기 슬롯의 Core 상태를 host 데이터로 읽을 수 있다(042 core_state0). Core는 상태를 host 데이터로 내보내고 다시 만드는 연산을 제공하며, 왕복은 비트 단위로 같다 (2026-10-17). GTrXL이 K/V를 캐시하는 경우(2026-10-23): 상태에는 층 입력도 함께 두고, 가중치를 교체하면 다음 step 전에 저장된 층 입력에서 새 가중치로 K/V를 다시 계산한다. 그래서 캐시를 쓰든 안 쓰든 출력은 같다(허용 오차 이내). host 내보내기(042 core_state0)는 층 입력과 유효 마스크만으로 상태를 다시 만들 수 있어야 한다 |
| 032-R17 | **MUST** 시간 축 코어(LSTM vs GTrXL)는 BC 본학습 전에 짧은 비교 학습으로 고른다(2026-10-18 사용자 결정, AMB-032-6). ① 학습: 코어마다 S1을 5,300 update 1회(seed 1개)씩 돌린다. 장비는 혼합 유동 운영이다(2026-10-18 사용자 결정): Linux Vulkan 호스트(f32)와 Windows CUDA 호스트(TF32 matmul)가 run 큐 작업자(061-R13·R14)로 두 run을 나눠 맡고 서로 이어받는다. 장비별 update 구간은 `runs/<id>/devices.jsonl`(061-R15)에 남고, 두 run의 장비 비율을 최종 집계해 리포트에 적는다. 두 run의 장비 비율이 크게 다르면 Vulkan f32와 CUDA TF32의 수치 차이가 비교에 섞일 수 있다(공정성 주의). 주 장비 규칙(2026-10-20 사용자 결정, AMB-032-8, 060-R8): CUDA 호스트가 유휴일 때는 CUDA 호스트가 주가 되고, 바쁠 때는 Linux Vulkan 호스트가 주가 된다. 이 규칙은 작업자 우선순위(CUDA 호스트에 더 높은 `--priority`)와 양보 요청(061-R14)으로 이미 구현돼 있다. 데이터 순서(`data.seed`)와 그 밖의 설정(Burn autotune·fusion 포함, 041-R15)은 같다. 학습률 스케줄은 짧은 run 전용으로, warmup 5%(265) 뒤 cosine이 5,300에서 끝난다(`config/bc_core_ablation.toml`). ② 오프라인 지표: `bc-eval --split val --games 500`(두 코어 같은 500게임). 500게임은 val 분할을 **고정 seed로 셔플**해 앞에서 고른다(출처 편중을 줄이기 위해, 2026-10-19 사용자 결정 G-D8, AMB-032-7). 학습 seed는 1개로 유지하고, 그 한계(seed 간 편차를 모름)는 리포트에 적는다. test 분할은 041-AC1용이라 쓰지 않는다. ③ 대전: 두 체크포인트 맞대결(TvT, 맵당 50판, 25판마다 슬롯 교대, `config/plans/core-ablation-h2h.toml`)과 각자 T0 상대 승률(상대 종족 3 × 맵 2 조합당 50판, 코어당 300판·합계 600판, 사용자 확정 2026-10-18, `config/plans/core-ablation-t0.toml`). 샘플링은 확률적(temp 1)이고 z는 없다. 대전 700판의 리플레이를 모두 저장한다(두 plan `save_replay = true`, `<out>/replays/g<id>.rep`, 2026-10-18 사용자 결정). ④ 판정: 규칙을 미리 정하지 않는다. 결과를 사용자에게 보고하고 사용자가 고른다. **결과: GTrXL 채택(2026-10-23 사용자 결정, `reports/m6_core_ablation.md` 4절).** ⑤ 비교 run의 체크포인트·로그·평가 출력은 지우지 않는다(비교 설정은 061-R6 `keep_checkpoints` 예외로 전부 보존, 2026-10-19 사용자 결정 G-D6). (2026-10-23 메모: 이 비교는 Burn으로 했고 결과(GTrXL 채택)는 PyTorch 전환 뒤에도 유지한다. Burn 체크포인트는 보관만 하고 새 코드에서 로드하지 않는다) ⑥ **2차 비교(PyTorch, 2026-10-24 사용자 결정)**: BC 본학습 전에 PyTorch 구현으로 비교를 한 번 더 한다. 코어마다 S1을 **10,000 update** 1회(seed 1개)씩 돌린다(run `core-abl-lstm-pt`·`core-abl-gtrxl-pt`). 설정은 `config/bc_core_ablation_pt.toml`이고, 지금 `config/bc.toml`과 S1 길이(10,000)·warmup(5% = 500, cosine이 10,000에서 끝남)·체크포인트 전부 보존만 다르다(win_open 헤드는 입력이 detach돼 나머지 학습에 영향이 없으므로 bc.toml 그대로 둔다). 장비는 1차와 같다: Windows CUDA 호스트(bf16 혼합, 032-R8)는 **유휴일 때만**, Linux ROCm 호스트(f32)는 상시, run 큐 작업자(061-R13·R14, CUDA 호스트 우선순위가 더 높음)로 나눠 맡고 이어받는다. 두 run의 장비 비율(`uranium-train devices`)을 리포트에 적는다(CUDA bf16과 ROCm f32의 수치 차이가 섞일 수 있음). 평가·판정·보존은 ②~⑤와 같다(plan `config/plans/core-ablation-pt-h2h.toml`·`core-ablation-pt-t0.toml`, 1차와 같은 seed·대진). 진행·결과는 `reports/m6_core_ablation_pt.md`. 판정 전까지 채택 코어는 GTrXL(④)이다. |
| 032-R18 | **MUST** `win_open` 헤드(2026-10-21 사용자 결정, AMB-031-6). ① privileged 입력 경로가 없고, 코어 출력 `h`는 detach해서 받으므로 정책·인코더·코어·기존 value 헤드에 gradient가 가지 않는다. ② 설정 `value.win_open`(기본 false)이 true일 때만 만든다. 설정 로더는 이 키가 없는 설정을 false로 읽는다(체크포인트 메타의 설정에 없는 키는 기본값으로 채운다). ③ 파라미터는 전역·장치 RNG를 쓰지 않고 고정 seed 생성기로 초기화한다(U(−k, k), k = 1/√fan_in). 전역 RNG를 쓰면 다른 모듈의 초기 가중치와 이후 dropout 난수가 바뀌기 때문이다. 그래서 헤드가 있어도 기존 모듈의 초기 가중치와 학습 난수는 같다(2026-10-23: Burn 지연 초기화 설명을 장치 무관 표현으로 바꿈). ④ `act()`와 학습 forward는 win_open을 value와 따로 돌려준다(꺼져 있으면 없음) |
| 032-R19 | **MUST** 최적화 원칙(2026-10-23 사용자 결정, AMB-032-10). ① 모델 동작은 바꾸지 않는다. 같은 수식을 더 빠르게 계산하는 구현만 쓴다: 커널 융합·직접 작성 커널, gather, CUDA/HIP graph 캡처, torch.compile, K/V 캐시(032-R16), 배치 안 레인 그룹 재구성(041-R3), 메모리 재계산 등. ② 결과 차이는 float 반올림, bf16 혼합 정밀도(032-R8), GPU atomic 덧셈 순서, dropout 난수열 배치 차이까지만 허용한다. ③ 모델 구조·헤드 방식(Selected 순차 선택 포함)·관측 해상도·distance bias·dropout 비율·손실·하이퍼파라미터·추론 저정밀도(FP8·INT8)는 이 원칙 밖이고, 바꾸려면 별도 사용자 결정이 필요하다. ④ 학습 설정은 Linux ROCm 호스트 메모리 예산(060-R3) 안에서 최악 모양으로 학습돼야 채택한다. ⑤ 채택 전에 AC1·AC8, 짧은 학습 loss 곡선(기준과 잡음 범위 안), 속도·메모리 측정을 리포트에 남긴다 |
| 032-R20 | **MUST** 구조 변형 비교(2026-10-25 사용자 결정, AMB-032-11, §2.7). ① 변형은 §2.7의 `[model]` 스위치 4개(`terrain`, `ctx_token`, `attn_pool`, `outer`)이고 서로 조합할 수 있다. 기본은 모두 꺼짐이며, 꺼지면 기본 구성과 비트 동일, 켜도 공유 모듈 초기 가중치는 기본 구성과 같다. ② 비교 run은 5개다: 하나씩 켠 4개(`arch-abl-terrain-pt`, `arch-abl-ctx-pt`, `arch-abl-pool-pt`, `arch-abl-outer-pt`(`outer` = 32))와 네 가지를 모두 켠 1개(`arch-abl-all-pt`). 코어는 GTrXL이고, 설정은 `config/bc_core_ablation_pt.toml`에 `[model]`만 더한 `config/bc_arch_abl_<변형>_pt.toml`이다. 각 run은 기준 체크포인트에서 이어 학습하지 않고 0 step부터 S1 10,000 update를 1회(seed 1개, `data.seed` 1) 학습한다. 기준은 R17 ⑥의 `core-abl-gtrxl-pt` `ckpt-10000`이다(다시 학습하지 않음). 장비는 R17 ⑥과 같은 혼합 운영이고, R17 ⑥이 쓰는 큐(`<shared>/queue/core-ablation-pt.toml`)에 run을 덧붙여 두 작업자가 R17 ⑥ run을 끝낸 뒤 이어서 잡는다. run별 장비 비율(`uranium-train devices`)을 리포트에 적는다. 잡음 측정용 추가 seed는 두지 않는다. ③ 평가: run마다 `bc-eval --split val --games 500`(R17 ②와 같은 500게임). **리그전**: 7개 모델(기준 GTrXL, R17 ⑥ LSTM, 변형 5개, 모두 `ckpt-10000`)의 21쌍, 쌍당 100판(맵 2 × 50판, 25판마다 슬롯 교대, 확률적 샘플링, z 없음, 리플레이 저장, R17 ③ 맞대결과 같은 방식). LSTM–GTrXL 쌍은 R17 ⑥ 맞대결 결과를 그대로 쓴다. 쌍별 승·패·무와 승률(Wilson 95%), 7×7 승률표, 전체 순위(Bradley–Terry, 무승부 0.5, bootstrap 95% 구간)를 낸다. T0 대전(R17 ③ 방식)은 7개 모델 모두 모델당 300판(종족 3 × 맵 2 × 50판)이다(2026-10-25 사용자 결정, 이전: 리그전 뒤 고른 후보만). LSTM·GTrXL은 `config/plans/core-ablation-pt-t0.toml`, 변형 5개는 `config/plans/arch-abl-pt-t0.toml`이다. 평가(val 500게임 채점·리그전·T0)는 큐의 7개 run 학습이 모두 끝난 뒤 함께 한다(2026-10-25 사용자 결정, R17 ⑥ 평가도 이때 한다). **실행(2026-10-26 사용자 결정, AMB-020-5)**: 대전은 평가 작업(020-R15) 하나로 돌린다. 단계 순서는 R17 ⑥ 맞대결 → 리그전 20쌍 → R17 ⑥ T0 → 변형 T0다. 추론은 CUDA 호스트(bf16 혼합)가 주 장비이고, CUDA 호스트가 빠진 동안에는 Linux ROCm 호스트(f32, 첫째 > 둘째)가 새 판부터 맡는다. 판마다 추론 장비를 기록하고(020-R15 ⑧), 결과 보고에 모델·단계별 추론 장비 비율을 함께 적는다(정밀도 차이가 섞일 수 있음). val 500게임 채점은 대전과 병행해 7개 모두 첫째 Linux ROCm 호스트(f32)에서 한다(같은 장비라 7개끼리 비교 공정). 리플레이 플레이어 이름은 020-R16이다. ④ 판정 규칙은 미리 정하지 않는다. 결과를 보고 사용자가 채택 여부를 고르고, 그 전까지 BC 본학습(041)은 시작하지 않는다. ⑤ 체크포인트·로그·평가 출력은 지우지 않는다(`keep_checkpoints = 0`). 결과는 `reports/m6_arch_ablation_pt.md` |

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
- GPU에 상수로 올리는 것: `action_table` 텐서 `select_cap[A]`, `target_cap[A]`, `loc_class[A]`, `max_select[A]`, `uses_*[A]`, `self_target[A]`(034-R9)
- GPU에서 헤드 진행 중에 샘플된 a로 gather한다:
  - `sel_mask = gather(unit_cap, select_cap[a]) ∧ is_self ∧ entity_mask`. Selected step마다 이미 고른 유닛과 건물 규칙(034-R7: 첫 선택이 건물이면 END만, 아니면 건물 제외)을 더 적용한다. 건물 여부는 엔티티별 `is_building`(떠 있는 건물·벙커·터렛 포함)으로 GPU에 올린다
  - `tgt_mask = gather(unit_cap, target_cap[a]) ∧ entity_mask ∧ ¬(sel ∧ ¬self_target[a])`. `sel`은 4번 헤드가 이번 스텝에 고른 유닛 집합[B,N]이다(샘플이든 teacher forcing이든 같다, 034-R9, 2026-10-19 사용자 결정). 이 조건으로 남는 대상이 없는 행은 원래 마스크(`gather(unit_cap, target_cap[a]) ∧ entity_mask`)를 쓰고, executor가 그 액션을 거부한다(031-R4)
  - `loc_mask = gather(loc_class_mask, loc_class[a])`

## 6. 학습·추론 인터페이스
```python
@dataclass
class UraniumNetConfig:  # entity, spatial, scalar, core, heads, value (설정에 없는 키는 기본값)
    ...

class UraniumNet(nn.Module):  # PyTorch (032-R1). 장치는 텐서·모듈이 놓인 torch.device가 결정한다
    def act(self, obs: ObsBatch, state: State, temp: float) -> ActOutput:
        """추론 1 step. 샘플링은 GPU 내부(Gumbel-max), temp 0 = argmax(032-R9·R15).
        사용된 헤드의 log-prob 합, 헤드별 log-prob [B,6], value [B,4], win_open, 새 상태를 돌려준다."""
    def forward_seq(self, obs: ObsBatch, labels: ActionBatch, priv: PrivBatch | None, s0: State, t: int) -> SeqOutput:
        """학습 [B·T 행, b-major]. teacher forcing. 헤드별 log-prob·entropy·분포, value, win_open."""
```
- `ObsBatch`·`ActionBatch`·`PrivBatch`는 030 §1 레이아웃의 텐서 묶음이다. 호스트 쪽 collate(패딩 포함)는 Rust(`uranium-data`)가 하고 numpy로 넘긴다(041, 001-R6).
- 인코더와 헤드는 B×T를 펼쳐 병렬로 계산하고, Core만 T를 따라 순차 계산한다.
- **BC**: 헤드별 CE × 사용 마스크([041](041-imitation-learning.md))
- **RL**: `forward_seq`의 target logits와 actor behaviour log-prob로 V-trace, UPGO, 헤드별 KL(π‖π_BC)을 계산한다([042](042-rl-training.md)).
- 처리량 구현 메모(비규범, 2026-10-18, `reports/m6_bc_preflight.md`): 결과가 같은 계산 순서 변경만 한다.
  - LSTM Core: 표준 LSTM(PyTorch `nn.LSTM`과 같은 파라미터·수식)이다. CUDA는 cuDNN을 쓰고, ROCm은 MIOpen LSTM backward가 실패해 PyTorch 자체 커널 또는 직접 작성 커널을 쓴다(2026-10-23 확인). Burn 시기에는 T step의 입력 투영과 4 게이트를 한 번의 matmul로 계산했다.
  - teacher forcing의 Location 헤드: location을 쓰는 행만 계산한다. 다른 행의 `dists.target_loc`은 0이고 R10 사용 마스크로 곱해져 결과에 영향이 없다. 호스트 동기화 없이 같은 결과를 내는 방식으로 바꿀 수 있다(032-R19).
  - 커널 선택: CUDA는 `cudnn.benchmark`, ROCm은 MIOpen find-db를 유지한다(041-R15). Burn 시기에는 autotune·fusion을 켰다(2026-10-19, AMB-032-7).
  - 분석 도구: `python -m uranium bc-profile`(모듈별 forward 표시, gradient 절단 비교).

## Acceptance Criteria
- 032-AC1: CPU 대비 같은 가중치·입력의 forward 출력(value, 헤드별 log-prob) 차이가 ROCm(f32)에서 ≤ 1e-4, CUDA f32(TF32)에서 ≤ 1e-2다. CUDA bf16 혼합 정밀도(032-R8)는 forward 출력 최대 차이 ≤ 0.15, 헤드별 정책 분포 KL 평균 ≤ 1e-3, greedy 액션 일치율 ≥ 95%다(2026-10-24 사용자 결정: 실측(실제 크기 모델 log-prob 최대 0.075, KL 최대 1.5e-3, 일치율 98% 이상, `reports/pytorch_opt.md` 2.24)의 약 2배 여유). 테스트는 pytest `gpu` 마커다. (2026-10-23 개정: 기준 장치 Flex → CPU, Vulkan → ROCm. 아래 CUDA 1e-2 근거는 Burn 시기 측정이다.)
  - CUDA 허용 오차 (AMB-032-5 Resolved, 2026-10-16 사용자 결정: 허용 오차를 늘린다). Burn 0.22(cubek-matmul)는 CUDA tensor core matmul에서 f32 입력을 TF32(가수 10비트, 누적은 f32)로 계산하고 끌 설정이 없다. 실측(Windows CUDA 호스트, 기본 크기 B16·N256): value ≤ 6.0e-5, action_type ≤ 3.4e-5, delay ≤ 2.5e-4, selected(최대 12 step log-prob 합) ≤ 3.9e-3, 정책 분포 KL ≤ 1.2e-7, greedy 액션 일치 48/48. 1e-2는 최대 실측의 약 2.5배 여유다. 영향 분석은 `reports/m5_model.md`.
- 032-AC2 (2026-10-23 개정): 추론 지연은 batch=16, N=256 기준으로 측정한다(p95). 판정은 CUDA 호스트에만 적용하고, 같은 호스트의 Burn CUDA 기준선(LSTM 51.4ms·GTrXL 57.4ms, `reports/bench_cuda.md` 2026-10-20)보다 빠르면 통과한다. Linux ROCm 호스트는 실측만 기록한다. 아래는 Burn 시기 기준이다. (이력) **판정은 CUDA 호스트에만 적용한다**(2026-10-20 사용자 결정, AMB-032-8): CUDA 호스트는 추론·학습의 주 장비이고(유휴일 때, 060-R8), 같은 호스트의 Vulkan보다 빠르면 통과한다(2026-10-16 사용자 결정: 40ms를 넘더라도 Vulkan보다 빠르면 된다). **Linux Vulkan 호스트**는 CUDA 호스트가 바쁠 때 주 역할을 맡는 대체 장비로 받아들인다. 통과·실패 기준은 두지 않고 실측 지연만 리포트에 기록한다(초안 40ms는 폐기). lockstep 학습(001-R5)에서는 지연이 결과를 바꾸지 않고 속도만 늦춘다. 실측(2026-10-20, `reports/m6_unverified_ac.md`): CUDA 호스트 LSTM 51.4ms·GTrXL 57.4ms(같은 호스트 Vulkan 77.5·90.3ms보다 빠름, 통과), Linux Vulkan 호스트 f32 179.0~179.5ms·187.4ms(f16 147.3·155.7ms는 참고, f16 경로 미구현).
- 032-AC3 (2026-10-23 개정): 학습 처리량은 batch 32×T64 기준으로 잰다. CUDA 호스트는 같은 호스트의 Burn CUDA 기준선(실데이터 11.1 s/update, `reports/bench_cuda.md`)보다 빠르면 통과한다. Linux ROCm 호스트는 메모리가 060-R3 예산 안이면 통과하고 처리량은 기록한다. (이력: 2026-10-16 CUDA 호스트는 같은 호스트의 Vulkan보다 빠르면 통과.)
- 032-AC4: 무작위 마스크 10만 건에서 샘플링된 액션이 모두 마스크를 만족한다.
- 032-AC5: `act()` 경로의 연산 그래프에 privileged 텐서가 없다(타입 수준 보장 + 테스트).
- 032-AC6: `LstmCore`와 `GtrxlCore`가 같은 `Core` 인터페이스로 교체되고, 둘 다 AC1–AC4를 통과한다.
- 032-AC7: 032-R17 비교의 두 run이 같은 update 수에서 끝나고, val 지표·맞대결·T0 승률(Wilson 95%)이 `reports/m6_core_ablation.md`에 기록되며, 사용자가 채택할 코어를 정한다. — **통과(2026-10-23)**: 두 run 5,300 update, 4절에 기록, GTrXL 채택. T0 GTrXL 26판은 실제 시간 상한으로 결과가 없어 함께 기록했다(사용자 결정). 2차 비교(R17 ⑥)도 같은 기준으로 기록한다(두 run 10,000 update).
- 032-AC8 (2026-10-23 개정): backward 정확도는 CPU 대비로 판정한다. 같은 가중치·입력·레이블의 BC 손실 차이가 AC1 허용 오차 안이고, 파라미터별 gradient 상대 L2(분모 규칙은 아래와 같음)가 ROCm(f32) ≤ 1e-3, CUDA f32(TF32) ≤ 1e-2다. CUDA bf16은 파라미터별 gradient 상대 L2의 중앙값 ≤ 3%, 손실 상대 차이 ≤ 1e-3이다(2026-10-24 사용자 결정, 1단계 실측 중앙값 1.5%의 2배). dropout은 0으로 둔다. 테스트는 pytest `gpu` 마커(`tests/test_cross_device.py`)다. 아래는 Burn 시기 기준과 근거다. (이력) backward(학습) 정확도(2026-10-19 사용자 결정, H·D-a, AMB-032-7): Flex(CPU) 대비 같은 가중치·입력·레이블의 BC 손실 차이가 AC1 허용 오차 안이고, 파라미터마다 gradient 차이의 상대 L2(‖g_dev − g_flex‖ ÷ ‖g_flex‖, 분모는 가장 큰 파라미터 gradient norm의 1e-3 이상으로 둔다)가 **Vulkan ≤ 1e-3**이다. **CUDA ≤ 1e-2**(2026-10-19 실측 기준(사용자 결정 D-a): autotune·fusion을 켠 이 테스트를 CUDA 호스트에서 6회 돌린 최댓값 4.9e-3(범위 3.7e-3~4.9e-3)에 약 2배 여유를 두고 올림. `reports/m6_bc_preflight.md` 5차. autotune·fusion 채택(041-R15)은 유지). dropout은 0으로 둔다. 측정은 학습과 같은 Burn 설정(autotune·fusion, 041-R15)으로 한다. 테스트는 `crates/uranium-model/tests/cross_device.rs`의 gradient 비교다. forward(AC1 Vulkan 1e-4)와 기준이 다른 이유: gradient는 반올림 누적이 많아 커널을 바꾸지 않은 Vulkan f32에서도 약 5e-4가 나온다(`reports/m6_bc_preflight.md` 4차).
- 032-AC9: win_open 경로에 privileged 텐서가 없다(테스트). `value.win_open` 키가 없는 설정이 로드된다(win_open 없음). 같은 seed에서 win_open을 켠 모델과 끈 모델의 공통 파라미터 초기값, 그리고 같은 batch 한 update(041-R19 clip 분리 포함) 뒤의 값이 CPU에서 비트 일치한다(2026-10-23: Flex → CPU).
- 032-AC10 (2026-10-23, 전환 검증): Burn 삭제 전에 PyTorch 모델이 Burn 모델과 같은 가중치·입력(CPU)에서 forward 출력 최대 차이 ≤ 1e-5, 파라미터별 gradient 상대 L2 ≤ 1e-4, AdamW 한 step 뒤 가중치 차이가 같은 기준 안이다. LSTM·GTrXL 모두 확인한다. 작은 설정의 기준 출력(golden fixture)을 저장소에 남겨 Burn 삭제 뒤에도 회귀 테스트로 쓴다.
- 032-AC11 (2026-10-25, 032-R20): §2.7 스위치가 모두 꺼진 모델은 기본 구성과 파라미터 목록·출력이 같다(기존 테스트 그대로 통과, AC10 포함). 각 변형과 네 스위치를 모두 켠 조합에서 ① 공유 모듈 초기 가중치가 기본 구성과 같고 ② step 실행이 시퀀스 실행과 같으며(032-R5, `outer` 구간 경계 포함) ③ 마스크(AC4)·privileged 경로 없음(AC5)·엔티티 재계산 동일(041-R17)·체크포인트 왕복이 성립한다(CPU 테스트). 추론 graph act = eager는 GPU 테스트로 확인한다. 다른 `[model]` 설정으로 재개하면 거부하고, `[model]`에 모르는 키가 있으면 시작을 거부한다.
- 032-AC12 (2026-10-25, 032-R20): 변형 run 5개가 10,000 update에서 끝나고, val 지표, 리그전 쌍별 승률(Wilson 95%)·7×7 표·전체 순위, run별 장비 비율·update 시간이 `reports/m6_arch_ablation_pt.md`에 기록되며, 사용자가 채택 여부를 정한다. — 기록 완료(2026-10-27, 3절), 채택 여부 사용자 결정 대기.

## Risks & Open Questions
- **구조 변형 비교 (2026-10-25, 032-R20)**:
  - 바깥 LSTM(`outer`)은 학습 묶음(T = 64) 안의 구간 전환(`outer` = 32면 1회)에서만 학습 신호를 받는다. 묶음을 넘는 보존은 forward로만 이어지고 "무엇을 오래 들고 갈지"는 배우지 못한다. 더 긴 학습 시퀀스는 Linux ROCm 호스트 메모리(060-R3) 측정 뒤 별도로 정한다.
  - 잡음 측정용 추가 seed가 없고, run마다 장비 비율(CUDA bf16 / ROCm f32)이 다르다. 작은 차이는 판정을 보류한다.
  - 학습 장비 두 대가 R17 ⑥을 학습하는 동안 변형 코드를 준비하므로, GPU에서만 도는 경로(CUDA `torch.compile`, 직접 작성 attention 커널의 추가 bias 버킷)는 변형 run이 시작될 때 처음 실행된다. 각 장비의 첫 update(속도·손실·메모리)를 확인하고, GPU 테스트는 리그전 전에 CUDA 호스트에서 돌린다.
  - 변형은 Python만 바꿔, 학습 장비는 작업 사본을 최신 master로 맞추기만 하면 된다(빌드 없음). 옛 코드는 `[model]`을 모르고 기본 구성을 학습하므로, 두 장비가 새 코드를 받은 뒤에 큐에 run을 넣는다.
- **Linux ROCm 호스트 안정성 (2026-10-23)**: 그 호스트의 ROCm 스택은 공식 지원 밖이라 트리 밖 커널 패치와 직접 빌드한 라이브러리로 돈다. 드문 GPU 멈춤이 있을 수 있고 GPU 리셋이 되지 않아 재부팅이 필요하다. 감시 프로세스(060-R6)가 마지막 체크포인트에서 재개하고, 그동안 CUDA 호스트가 이어받을 수 있다(061-R13·R14). 장비 상세는 리포트에 둔다.
- **CUDA bf16 수치 (2026-10-23)**: 두 호스트의 정밀도가 달라(CUDA bf16, ROCm f32) 이어 학습할 때 수치 차이가 섞인다. AC1·AC8 bf16 기준을 실측으로 정할 때 영향을 함께 기록한다.
- 아래 M1(a)~AC8 CUDA 기준 항목은 Burn 시기 기록이다.
- **M1(a) 실측 (2026-10-15, Linux Vulkan 호스트 f32, `reports/`의 M1(a) Vulkan 벤치 리포트)**:
  - 추론 근사(B16·N256) 87ms로 032-AC2 초안(40ms)에 미달한다. (2026-10-20: 초안 40ms는 폐기했다. Linux Vulkan 호스트는 실측 기록만 한다, AMB-032-8)
  - 학습 약 40 프레임/s로, BC 1회독에 약 34일이 걸린다.
  - 사용자는 f16만 채택했다. 공간 64² 축소, 인코더 축소, 프레임 샘플링은 채택하지 않았다.
  - M5에서 f16으로 실측했을 때도 AC2와 학습 속도가 부족하면 사용자에게 결과를 보고하고 추가 대응을 다시 확인받는다.
- Burn의 scatter-add, GRUCell, attention bias 주입, ConvTranspose2d가 Vulkan에서 지원·성능 미확인이다. M1(a)에서 확인한다([060](060-infra.md)).
  - 대체안: scatter → one-hot matmul, attention은 직접 구현
- **AC8 CUDA 기준 (2026-10-19, 사용자 결정 D-a)**: 2026-10-18 측정(`reports/m6_bc_preflight.md` 4차)에서 CUDA autotune+fusion의 일부 conv gradient 차이는 2.2e-2(autotune만 5.3e-2, 끈 상태 1.3e-3)였다. 처음 제안한 CUDA 1e-2는 이 값보다 작으므로 쓰지 않는다. autotune·fusion을 켠 AC8 테스트로 다시 측정하고(학습 코드 수정 단계), 그 값에 여유를 둔 기준을 사용자 확인을 거쳐 이 AC에 적는다. → 2026-10-19 재측정(5차): 같은 테스트·같은 지표(파라미터별 상대 L2)에서 3.7e-3~4.9e-3(6회)이라 CUDA 1e-2로 적었다(실측 기준, D-a). 4차의 2.2e-2와 차이가 나는 원인은 확인하지 못했다(그 뒤 모델 코드 수정, autotune의 측정 시간 기반 커널 선택 등이 후보). 다른 세션·호스트에서 1e-2를 넘으면 이 기준을 다시 사용자에게 확인받는다.
- 128² Location 업샘플이 지연을 지배할 수 있다. 대체안은 64² 출력 + 2×2 subcell 헤드다.
- Selected 12-step 루프 지연이 문제가 될 수 있다. 대체안은 단일 step multi-label(sigmoid)이다.

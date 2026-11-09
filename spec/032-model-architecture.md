# 032 — Model Architecture (PyTorch)

- **Status**: Approved (2026-10-15, **2026-11-09 개정 초안(Review, AMB-032-12·13 Open, AMB-032-14 Deferred(U2) — 사용자 확인 대기)**: 모델 2 개선 스위치 §2.8 — 헤드 우회로·스칼라 직접 입력(032-R21), 공간 ResBlock pre-activation(032-R22, §2.2에 지금 끝 ReLU 명시), 코어 수정 F1·F2(032-R23), 학습 전용 보조 모듈 privileged 예측·JEPA형 예측(032-R24), 죽은 채널 통계(032-R25), R11 privileged는 학습 정답으로도 허용(사용자 결정 2026-11-09), §1·§4·§6, AC13~AC15, AMB-032-12~14. 이 개정의 요구사항은 확인 전까지 구현하지 않고, 기존 요구사항·AC는 스위치 기본값(꺼짐)에서 그대로 유효하다. 2026-11-03 개정(사용자 결정, AMB-042-16): §2.6 build·units 출력이 033 채널 모드 `"scout"`에서 r_scout·r_survive를 배움, 구조 변경 없음. 2026-10-31 개정: R6 feature 스키마 로드 호환(030-R5, 7은 6을 받음, 사용자 결정). 2026-10-27 개정: §2.7 RL 학습기의 `outer` 지원 방식(사용자 확인), AC11 RL 경로 추가. 2026-10-27 개정: 032-R20 ④ 구조 변형 판정 — 네 스위치 모두 켬 채택(사용자 결정), §2.7 본학습·RL 모델 문장, R17 ⑥ 판정, AC12 통과. 2026-10-23 개정: Burn에서 PyTorch로 전환(사용자 결정, AMB-000-4) — R1 PyTorch·CPU 기준, R6 형식은 061-R10, R8 장비별 정밀도(CUDA bf16 혼합·ROCm f32, AMB-032-9), R13 구현 방식 자유화, R14·R16 Python 추론 스레드·K/V 캐시 조건, R18 Burn 표현 정리, R19 최적화 원칙(AMB-032-10), §2.4·§6 Python 인터페이스, AC1·2·3·6·8·9 기준 장치 변경, AC10 Burn 동등성. 2026-10-17 개정: R8 호스트별 정밀도 — CUDA 호스트 f32(TF32) 허용, 사용자 결정 / 추론 서버 Core 상태 슬롯(032-R14), 샘플링 모드·헤드별 log-prob(032-R15) / 추론 서버 가중치 교체·슬롯 상태 읽기(032-R16, 3단계 계획 승인) / 2026-10-18 개정: 코어 비교 절차(032-R17, AC7, AMB-032-6). 2026-10-19 개정(BC 전 리뷰 결정, 사용자 확정): AC8 backward 허용 오차(H), autotune·fusion 채택(G-D5, 041-R15), R17 ② val 고정 seed 셔플 500게임(G-D8), F_c를 030과 일치, 장비 표기 중립화(001-R10, G-D9). 2026-10-19 T7 개정(사용자 결정): §5 대상 마스크에서 선택 유닛 제외(034-R9), `self_target[A]` 상수). 2026-10-20 개정(사용자 결정, AMB-032-8): AC2 기준은 CUDA 호스트에만 적용하고 Linux Vulkan 호스트는 실측 기록만 하는 대체 장비로 둔다, R17 주 장비 규칙(CUDA 호스트가 유휴면 주, 아니면 Linux Vulkan 호스트가 주, 060-R8). 2026-10-21 개정: 032-R18 win_open 헤드(사용자 결정, AMB-031-6). 2026-10-23 개정: 코어 비교 결과로 GTrXL 채택(032-R17 ④, AC7, 사용자 결정); 2026-10-24 개정: R17 ⑥ PyTorch 2차 코어 비교(사용자 결정); 2026-10-25 개정: §2.7·R20 구조 변형 비교(지형 붙이기·스칼라 토큰·attention 요약·바깥 LSTM과 그 조합, 사용자 결정, AMB-032-11), AC11·AC12. 2026-10-31 개정: R19 RL learner 적용(042 learner 레인 그룹·최적화, 사용자 결정))
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
- 2026-11-09 개정 초안(사용자 확인 대기, §2.8): 아래 스위치를 켜면 위 그림에 다음이 더해진다. 모두 꺼지면 위 그림 그대로다.
  ```
  [h, c, scalar 일부] ─► 질문 ─► E에 교차 어텐션 ─► Linear(0 초기화) ─► ActionType 로짓 직전 + ar   (bypass, 032-R21)
  Core 입력 x_in ─► Linear(0 초기화) ─► h에 더함                                                     (core_skip, 032-R23)
  h ─► AuxPriv ─► privileged 예측 (학습 전용, 정답 = privileged)                                     (aux_priv, 032-R24)
  h_t ─► JEPA 예측기_k ─► [e_pool, s_pool]_{t+k} 예측 (학습 전용, 정답은 기울기 끊음)                 (aux_jepa, 032-R24)
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
- **활성 함수 위치(2026-11-09 명시, 기존 구현 그대로)**: stem conv 뒤 ReLU. 다운샘플 conv 뒤 ReLU. `ResBlock(x) = ReLU(x + Conv3x3(ReLU(Conv3x3(x))))`로 **끝에 ReLU가 있다**(post-activation). 그래서 각 단계 출력·skip·`S8`은 모두 0 이상이다. `spatial_preact`(§2.8, 032-R22)를 켜면 ResBlock만 바뀐다.
- **다운샘플**:
  - 각 단계는 `ReLU(Conv3x3 stride2)` → `ResBlock(2×Conv3x3)`
  - 128²→64²(64ch) → 32²(64ch) → 16²(128ch) → 8²(128ch)
  - 64², 32², 16² 출력은 skip으로 보관한다. 8² 단계 출력은 bottleneck에만 들어간다.
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
- ActionType 행의 GLU는 `x = ReLU(x) ⊙ σ(Linear(c))`이고 그 뒤 `Linear(256→A)`가 로짓이다. 헤드 우회로(`bypass`, §2.8, 032-R21)를 켜면 이 `x`와 `ar`에 우회로 출력이 더해진다(2026-11-09 개정 초안, 사용자 확인 대기).

### 2.6 ValueHeads
- 입력:
  - `h`
  - z 임베딩
  - enemy_race 임베딩
  - **privileged**(학습 전용): [010](010-game-protocol.md)의 `PrivilegedSummary`(상대 실제 자원, 업그레이드, 테크, 유닛 구성)를 쓴다. RL에서는 OpenBW lockstep 학습 모드(셀프플레이와 오픈소스 봇 상대 모두, 011-R5)에서 얻고, BC에서는 리플레이 재생 모드(011-R12)에서 얻는다. 실전(SC:R)에는 존재하지 않는다.
- 구조: 공유 `MLP(→256)`에 출력 4개를 둔다.
  - win: tanh
  - build, units, value: 선형
  - 보상 성분과의 대응은 [033](033-reward.md)을 따른다. RL 보상 채널 모드 `"scout"`(033-R11, 2026-11-03, AMB-042-16)에서는 build 출력이 r_scout, units 출력이 r_survive의 value를 배운다. 출력 이름·개수·구조·초기화는 그대로이고(체크포인트 호환), 바뀌는 것은 RL learner가 그 출력에 주는 목표뿐이다.
- privileged 입력이 정책 헤드로 흘러가는 경로가 그래프 상 존재하지 않아야 한다(032-R11). 2026-11-09 개정 초안(사용자 확인 대기): privileged는 학습 전용 보조 목표 `aux_priv`(§2.8, 032-R24)의 **정답**으로도 쓸 수 있다. 정답은 손실에만 들어가고 어떤 모듈의 입력도 아니다.
- **win_open**(2026-10-21, 032-R18): 상대 정보 없이 추정한 승패 기대값(tanh, −1 패 ~ +1 승). 입력은 detach한 `h`, z 임베딩, enemy_race 임베딩뿐이고 공유 MLP와 별도인 `MLP(→256)→1`을 쓴다. 031-R12 항복 판정에만 쓰고 RL advantage에는 쓰지 않는다. win 헤드는 학습 때 항상 privileged를 받는데 실전에는 그 입력이 없어(0 벡터) 실전 승률로 쓸 수 없기 때문에 둔다.

### 2.7 구조 변형 (비교용, 032-R20)
- 목적: 기본 구성의 구조 약점을 메우는 변형을 짧은 BC 비교 학습으로 시험한다(2026-10-25 사용자 결정, AMB-032-11). 약점은 ① e_pool이 유닛 수백 개를 아군 평균·적 평균 두 개로 줄여 개수·소수 유닛·배치가 묻힘 ② 유닛 토큰이 지형(맵 채널에만 있음)과 스칼라 문맥(자원·전략 조건·시간)을 모름 ③ Core가 최근 64 step 밖을 잊고, 세 인코더가 함께 쓰는 학습되는 장기 상태가 없음이다. **채택(2026-10-27 사용자 결정, 032-R20 ④)**: 네 스위치를 모두 켠 구성(`terrain`, `ctx_token`, `attn_pool`, `outer` = 32, 비교 run `arch-abl-all-pt`와 같음)이 BC 본학습(041)·RL(042)의 모델이다. 본학습 설정은 `config/bc.toml`의 `[model]`이다. 코드의 기본값(키가 없으면 꺼짐)은 그대로 둔다.
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
  - §2.8의 모델 2 스위치도 이 공통 규칙을 따른다(2026-11-09 개정 초안, 사용자 확인 대기).
  - RL 학습기(042, 2026-10-27 사용자 확인): RL 세그먼트는 64 step이고 한 step씩 겹치므로 게임의 아무 결정에서나 시작한다. 세그먼트 시작 상태(042 core_state0)에는 바깥 상태(`h_o`, `c_o`, `h_sum`, `n`)가 들어 있다. 줄마다 첫 구간 길이는 `outer − (n mod outer)`이고, 그 뒤는 `outer` 길이, 마지막은 남은 길이다. 구간 k는 줄마다 k번째 구간의 step을 모아 그 줄의 `h_o`로 인코딩하고, Core를 시퀀스 전체에 돌려(아직 계산하지 않은 step의 입력은 0이고, Core는 인과적이라 앞 step은 정확함) 그 step의 출력으로 줄마다 바깥 상태를 갱신한다. 구간에 step이 없는 줄은 카운터가 `outer`의 배수여도 갱신하지 않는다. 결과는 한 결정씩 계산한 것과 같다(032-R5, 허용 오차). 엔티티·공간 인코더 비용은 BC와 같고 Core는 최대 구간 수(T = 64, `outer` = 32면 3)만큼 돈다. BC 경로는 바꾸지 않는다. RL의 T·겹침(042-R8)은 그대로다.

### 2.8 모델 2 개선 스위치 (2026-11-09 개정 초안, 사용자 확인 대기, 032-R21~R25)
- **배경**: JEPA 진단(`reports/jepa_survey.md` 10절)에서 Checkpoint 1의 구조 문제가 드러났다. ① 적 빌드 단서(건물 HP·수)가 e_pool과 GTrXL 1층 GRU 게이트에서 사라져 h_t에 남지 않고 행동에 쓰이지 않는다(10.1~10.3). ② 스칼라 인코더 출력 `c`가 입력 스칼라(적 최대 관측 수)를 보존하지 못한다(10.2.2·10.2.3). ③ 공간 인코더 깊은 단계가 죽은 ReLU로 0이 되어 `s_pool`이 상수다(10.4). 사용자 결정(2026-11-09): 이 문제를 고친 모델(모델 2)을 BC부터 다시 학습한다(041-R29).
- **설정**: 모두 BC 설정 TOML의 `[model]` 표 키다(키가 없으면 꺼짐). §2.7처럼 만들 모델 구조를 고르는 스위치이고 체크포인트 메타의 모델 설정(061-R10)에 남는다. 서로 독립이고 §2.7 스위치와도 조합할 수 있다. 공통 규칙은 §2.7과 같다: 모두 꺼지면 지금 모델과 파라미터 목록·출력이 비트 동일하고(Checkpoint 1이 그대로 로드·측정된다), 켜도 새 모듈은 전역 RNG를 쓰지 않는 고정 seed 생성기로 만들어 공유 모듈의 초기 가중치가 같다. 새 모듈은 win_open처럼 기존 모듈을 모두 만든 뒤에 만든다. `aux_jepa`는 정수 목록이라 `[model]` 설정 로더(지금은 bool·정수만 받음)가 목록 값을 받도록 넓혀야 한다.
- **구현 시점(P2)**: 확인 뒤에도 `bypass` 코드는 AMB-032-12, aux 코드는 AMB-032-13이 정해지기 전에는 쓰지 않는다(032-R21 ④, R24 ⑤). `recency_init`의 `b(d)`와 경로 2에 켤 코어 수정은 코어 어텐션 조사(U2) 뒤로 보류한 AMB-032-14다.
- **`bypass` (bool) — 헤드 우회로 + 스칼라 직접 입력(032-R21)**: ActionType 헤드가 Core를 거치지 않고 유닛 토큰 E를 직접 읽는다.
  - 질문: `q = Linear([h, c, s_raw] → d_b)`. `s_raw`는 모델 입력 스칼라(030 §4, 정규화 뒤 값) 중 고른 부분집합이다. 스칼라 인코더를 거치지 않으므로 `c`가 보존하지 못한 값(적 최대 관측 수 등)이 헤드에 그대로 닿는다.
  - 교차 어텐션: K = `Linear(256→d_b)(E)`, V = `Linear(256→d_b)(E)`, multi-head, 점수를 유닛 축으로 masked softmax한다. 대상은 `entity_mask` 안의 엔티티 전부(아군·적·중립)이고 추가 토큰(`ctx_token`·`outer`)은 들어가지 않는다(Heads가 받는 E와 같다). 집합이 비면 출력은 0이다(`attn_pool`과 같은 처리, CUDA graph에 안전한 방식).
  - 출력: `u = Linear(… → 256)`. 투영 입력은 기본안(미확정, AMB-032-12) `[어텐션 출력, s_raw]`이다. 이 투영은 **가중치·bias 모두 0으로 초기화**한다. `u`를 ActionType GLU 출력 `x`(로짓 `Linear(256→A)` 직전)와 자기회귀 임베딩 `ar`(`ar = Linear(h) + Emb(a) + u`) 양쪽에 더한다. 그래서 켠 직후의 출력은 끈 모델과 같고, 학습이 진행되며 우회로가 쓰인다.
  - `s_raw`는 헤드와 `ar`에만 닿고 Core·h_t에는 들어가지 않는다. 사용자 항목 ⑥(스칼라 보존)은 행동 헤드 쪽만 해결되고, h_t·`c`가 그 값을 보존하지 못하는 문제(10.2.2)는 남는다(확인 대상).
  - 크기(d_b, head 수)와 `s_raw` 부분집합은 [SPEC AMBIGUOUS] AMB-032-12다.
- **`spatial_preact` (bool) — 공간 ResBlock pre-activation(032-R22)**: 공간 인코더의 ResBlock 6개(내려가는 단계 4개, bottleneck 2개)를 `ResBlock(x) = x + Conv3x3(ReLU(Conv3x3(ReLU(x))))`로 바꾼다. 끝 ReLU가 없어져 블록 출력이 음수일 수 있고, 항등 경로로 기울기가 늘 흐른다(10.4의 붕괴 원인은 끝 ReLU 뒤 모든 채널이 0이 되어 기울기가 정확히 0이 된 것). stem·다운샘플 conv 뒤 ReLU, skip·`S8`·`s_pool`·Location 헤드의 연결, 파라미터 목록·모양은 그대로다(초기 가중치도 같다).
- **`core_skip` (bool) — 코어 입력 skip, F1(032-R23)**: `h ← h + Linear(768→512)(x_in)`. `x_in`은 Core 입력(§2.7 `outer`의 더하기 뒤)이고 이 Linear는 가중치·bias 모두 0으로 초기화한다. 바뀐 `h`를 Core 출력을 쓰는 모든 곳(헤드, value, win_open, `outer`의 `h_sum`, aux)이 쓴다. GTrXL 첫 층 게이트가 지금 step 정보를 지우는 문제(10.3)를 Core 밖으로 돌아가게 한다. LSTM·GTrXL 모두에 쓸 수 있다. 바뀌는 것은 Core 뒤로 나가는 출력뿐이고 Core의 순환 상태(LSTM의 h·c, GTrXL 기억의 층 입력)는 그대로다.
- **`recency_init` (bool) — 최근 step 선호 초기화, F2(032-R23)**: GTrXL 상대 위치 bias 표(`rel_table`, [M+1, 층 × head], 지금 N(0, 0.02) 초기화)를 지금과 같은 난수로 만든 뒤, 거리 d(step 차이)에 따라 정해지는 결정론적 항 `b(d)`를 더해 초기화한다. 1층 어텐션이 지금 step에 0.9%만 주는 문제(10.3)를 겨냥한다. 초기값만 바꾸고 학습은 자유다. GTrXL 전용이다. `b(d)`의 꼴·크기와 적용 층은 [SPEC DEFERRED(U2)] AMB-032-14다.
- **`aux_priv` (bool) — 전체 정보 보조 목표(032-R24)**: 학습 전용 모듈 `AuxPriv`(크기는 AMB-032-13, 기본안(미확정) `MLP(512 → 256 → P)`·은닉 ReLU)가 h_t(detach 안 함)로 그 step의 privileged 벡터(010 `PrivilegedSummary`로 만든 `PrivilegedTensors`, 030-R9, 지금 P = 383)를 예측한다. 손실은 041-R23·042-R21이 정한다. privileged는 정답으로만 쓰이고 어떤 모듈의 입력도 아니다(032-R11). **privileged가 없는 step은 뺀다**: shard 스텝의 `privileged`가 `None`이면 지금 로더는 0 벡터로 채운다(value head용, E-10 통계 `split-stats`). 0 벡터를 정답으로 쓰지 않도록 collate가 step별 privileged 존재 마스크를 함께 넘겨야 한다. 지금 host batch(`HostGroup`)에는 이 열이 없으므로 **필요한 변경**이다(RL trajectory도 같은 마스크).
- **`aux_jepa` (정수 목록, 비면 꺼짐) — JEPA형 자기 예측 손실(032-R24)**: 목록의 k마다 학습 전용 예측기 `g_k`가 h_t로 같은 레인 t+k step의 인코더 요약 `y_{t+k} = [e_pool, s_pool]_{t+k}`(512)를 예측한다. 정답은 엔티티 인코더의 `e_pool`과 공간 인코더의 `s_pool`을 이어 붙인 값으로, **`outer`의 Core 입력 더하기(`core_add`) 전**의 값이다(그래서 인코딩 함수는 Core 입력과 별도로 `e_pool`을 돌려줘야 한다, 지금은 `s_pool`만 따로 돌려줌). 정답은 같은 forward에서 계산한 값에 **기울기를 끊은 것(stop-grad)**이다. 그래서 인코더는 예측 쪽(h_t를 거친 경로)으로만 기울기를 받는다. 유효 쌍은 t와 t+k가 모두 유효 step(아래)이고, 같은 게임이며, t+k가 학습 묶음(BC [B, T = 64], RL 세그먼트) 안인 것뿐이다. 게임 끝·묶음 끝을 넘는 쌍은 빠진다.
- **aux의 유효 step**: 패딩이 아닌 실제 step이다. 레이블 가중치(무효·`unknown` 레이블이면 0, 041-R11·040-R11)와는 따로다: aux 정답은 레이블과 무관하므로 레이블이 무효인 실제 step도 쓴다. 지금 BC 패딩 행은 마지막 실제 step을 복사하고 가중치 0만 붙이므로, collate가 실제 step 마스크를 따로 넘겨야 한다(필요한 변경). `aux_priv`는 여기에 privileged 존재 마스크를 곱한다. RL 자기 모방 레인(042-R16)에는 aux 손실을 주지 않는다. k 목록(기본안 {1, 8, 32}), 예측기 크기, 정답 정규화, 손실 꼴(코사인 또는 정규화 L2)은 AMB-032-13이다.
- **추론**: aux 모듈(`aux_priv`, `aux_jepa`)은 `forward_seq`(학습)에서만, 그것도 aux를 요청했을 때만 계산한다(§6 `aux` 인자: RL BC 참조 재forward·자기 모방 레인·진단은 끔). `act()`와 추론 graph(032-R19)에는 aux 모듈도 privileged도 없고, aux 출력은 행동 선택·value·항복 판정에 쓰지 않는다. `bypass`·`spatial_preact`·`core_skip`·`recency_init`은 정책 구조이므로 추론에도 그대로 쓴다.
- **step = sequence**: `bypass`·`spatial_preact`·`core_skip`은 시간 축이 없는 변경이라 step 실행과 시퀀스 실행이 같다(032-R5, `outer` 구간 경계 포함). `recency_init`은 초기값만 바꾼다.
- **죽은 채널 통계(032-R25)**: 학습 중 공간 인코더 붕괴를 일찍 보기 위한 감시 통계다. 정의는 032-R25, 설정 키·기록 주기·경보 규칙은 041-R25·042-R21이다(키가 없으면 끔).
- **G1(선택 상한 확대)**: 결정 단위 측정(G0) 결과에 따라 Selected 헤드 상한을 바꿀 수도 있다. 측정 전에는 정하지 않고 spec 문안도 쓰지 않는다([031](031-action-space.md) [SPEC DEFERRED(G0)] AMB-031-7).

## 3. Requirements
| ID | 요구사항 |
|---|---|
| 032-R1 | **MUST** PyTorch(`python/uranium/model`)로 작성한다. 모델 코드는 장치에 무관하고 실행 시 `torch.device`로 고른다. 학습·추론은 CUDA 호스트에서 CUDA 빌드, Linux ROCm 호스트에서 ROCm 빌드의 `cuda` 장치를 쓴다(ROCm 빌드도 `torch.cuda` API를 쓴다). 기준 구현과 단위 테스트는 CPU에서 돈다. 장치별 차이가 있는 경로(예: ROCm에서 MIOpen LSTM backward 실패로 자체 LSTM 커널 사용)는 같은 수식이어야 하고 AC1·AC8로 확인한다. 직접 작성한 CUDA·HIP 커널은 같은 소스를 두 백엔드에서 빌드하고, 순수 PyTorch 기준 구현과 끄는 설정을 함께 둔다(060-R1). (변경 이력: 2026-10-15 Burn 0.22 `Device`(Vulkan·CUDA·Flex), 2026-10-23 PyTorch로 전환, AMB-000-4) |
| 032-R2 | **MUST** 기본 크기는 약 12M 파라미터(§4)로 하고, 허용 범위는 10M–40M이다. 확장은 d_model 384, Transformer 4층 순으로 하며 M1(a) 벤치로 확정한다 |
| 032-R3 | **MUST** 헤드는 자기회귀다. 학습에서는 teacher forcing(레이블 액션 주입), 추론에서는 샘플값을 주입한다 |
| 032-R4 | **MUST** 모든 헤드에 마스크를 적용한다. 마스크된 logit은 `-1e9`로 둔다. 샘플링 결과는 항상 마스크를 만족해야 한다 |
| 032-R5 | **MUST** Core 상태는 에피소드 간 리셋한다. 학습 시퀀스는 T=64 truncated BPTT로 한다 |
| 032-R6 | **MUST** 체크포인트에 함께 저장할 항목: 가중치, `UraniumNetConfig`, feature 스키마 버전, action 스키마 버전([034](034-action-table.md)), 정규화 통계. 파일 형식은 061-R10을 따른다(2026-10-23: 가중치는 safetensors). 로드는 두 버전이 현재와 같아야 하고, feature 스키마는 030-R5의 로드 호환 목록에 있는 이전 버전도 받는다(2026-10-31: 7은 6을 받음, BC 체크포인트를 RL 출발점으로 그대로 씀) |
| 032-R7 | **MUST** 추론은 batch B개 게임 동시 처리를 지원한다. 엔티티는 배치 내 최대 N으로 동적 패딩한다 |
| 032-R8 | **MUST** 정밀도는 장비별로 정한다(2026-10-23 사용자 결정, AMB-032-9). **CUDA 호스트**: 학습·추론 모두 **bf16 혼합 정밀도**다. 행렬곱·conv는 bf16 autocast로, 마스터 가중치·옵티마이저 상태·손실·log-softmax·softmax 누적·value 출력은 f32로 둔다. bf16은 지수 범위가 f32와 같아 손실 스케일링은 쓰지 않는다. 수치 문제가 생기면 해당 모듈만 f32로 되돌리고 기록한다. **Linux ROCm 호스트**: f32다(2026-10-23 실측에서 bf16 autocast가 f32보다 학습 2.9~3.8배·추론 3.3배 느림, `reports/torch_vs_burn.md`). 체크포인트는 정밀도와 무관하게 f32 가중치로 저장하므로 두 호스트가 서로 이어 학습할 수 있다. 정밀도는 `meta.json`과 `devices.jsonl`(061-R15)에 기록한다. GPU 계산은 비결정적일 수 있다(atomic 덧셈 순서 등, 032-R19). 비트 단위 재현은 CPU에서만 판정한다(061-AC1). 아래는 Burn 시기의 변경 이력이다. (이력) 2026-10-17 개정, 정밀도는 학습 호스트에서 **더 빠른 쪽**을 쓴다. **Linux Vulkan 호스트**: 학습과 추론은 **f16 혼합 정밀도**로 한다. M1(a)에서 그 호스트가 f16을 지원하고 matmul 오차가 4.5e-4, 속도가 약 1.5배임을 확인했다(2026-10-15 결정). 마스터 가중치·옵티마이저 상태·손실·softmax 누적은 f32로 두고, 손실 스케일링을 적용한다. 수치 문제가 생기면 해당 모듈만 f32로 되돌리고 기록한다. **CUDA 호스트**: f32(CUDA 내부 TF32 matmul)를 기본으로 한다. M5 실측에서 f16이 f32보다 느렸다(추론 중앙값 59 → 78ms, 학습 70 → 54 samples/s, `reports/m5_model.md`). f16 혼합 정밀도 학습 구현은 Vulkan 호스트 학습 전까지 미룬다(2026-10-17 사용자 결정). **예외**: Vulkan 호스트에서 도는 모든 BC 학습(ablation (A)·041-R5, 다른 PC에서 넘겨받아 이어 하는 S1·S2 구간 포함)은 f16 혼합 정밀도 미구현으로 f32로 학습한다(2026-10-18 사용자 결정, 처음 (A)만이던 예외를 넓힘) |
| 032-R13 | **MUST** distance bias의 EntityEncoder 역전파 시간은 정방향의 5배 이내여야 한다(장비마다). 구현 방식은 자유다: bucket 인덱스로 표를 gather(embedding), bucket one-hot과 matmul, bucket별 사전 합산, 또는 attention과 bias를 합친 직접 작성 커널(커널 안에서 위치로 bucket을 계산해 [B,H,N,N] bias를 만들지 않는 방식 포함). 어느 방식이든 결과는 같은 수식이다(float 반올림과 GPU atomic 순서 차이만 허용, 032-R19). (변경 이력: 2026-10-15 Burn의 scatter-add 역전파가 정방향의 약 40배(310ms)라 "scatter-add 금지, one-hot matmul 등"으로 정했다. 2026-10-23 PyTorch에서는 gather 역전파가 빠르고 ROCm에서는 one-hot matmul 역전파가 오히려 느려서(엔티티 인코더 fwd+bwd 303.7ms 대 gather 79.9ms, `reports/torch_vs_burn.md`) 방식 제한을 풀었다) |
| 032-R9 | **MUST** `act()`는 헤드 6개의 샘플링을 **한 번의 GPU forward 안에서** 수행한다(Gumbel-max). CPU 왕복은 없어야 한다 |
| 032-R10 | **MUST** joint log-prob와 엔트로피는 `action_table`상 사용된 헤드만 합산한다 |
| 032-R11 | **MUST** privileged 입력은 ValueHeads에만 연결한다. `act()` 시그니처는 privileged 텐서를 받지 않는다. **2026-11-09 개정 초안(사용자 결정 2026-11-09, 문안은 사용자 확인 대기)**: privileged는 학습 전용 보조 목표(032-R24 `aux_priv`)의 **정답**으로도 쓸 수 있다. 정답은 학습 손실에만 들어가고 인코더·Core·정책 헤드·aux 모듈의 입력이 되지 않는다. 추론 입력은 자기 시야 그대로다(010-R5, 030-R2). privileged가 학습 신호로 가중치에 영향을 주는 것은 허용하고, 추론 때 privileged를 받거나 만드는 경로는 없다 |
| 032-R12 | **MUST** Selected 헤드의 BC 레이블은 정규 순서(entity index 오름차순) 시퀀스 + END로 만든다 |
| 032-R14 | **MUST** 추론(batch broker의 슬롯, 001-R6)은 세션마다 고정 슬롯을 두고, Python 추론 스레드가 Core 상태를 GPU의 `[capacity, …]` 텐서에 보관한다(2026-10-23: 이전에는 Rust `uranium-agent` 추론 스레드). broker는 슬롯이 열리고 닫힌 것을 배치와 함께 알린다. 요청이 모인 세션의 행만 `select_rows`로 꺼내 `act`를 돌리고 `scatter_rows`로 되돌린다. 세션이 시작될 때 그 슬롯을 0으로 리셋한다(032-R5). 배치로 처리한 결과는 세션을 하나씩 처리한 결과와 같아야 하고(허용 오차 이내), 세션 사이에 상태가 섞이지 않는다 (2026-10-17) |
| 032-R15 | **MUST** `act()`의 샘플링 모드는 확률적(temp 1)과 결정론적(temp 0 = 헤드마다 argmax)을 둔다(050-R1). `ActOutput`은 joint log-prob과 함께 헤드별 log-prob [B,6]을 돌려준다. RL actor는 이 값을 behaviour log-prob로 기록한다(042) (2026-10-17) |
| 032-R16 | **MUST** 추론 서버는 실행 중에 가중치를 교체할 수 있다(042 K update마다). 교체는 다음 배치부터 적용되고, 응답마다 정책 버전을 붙인다. 세션은 자기 슬롯의 Core 상태를 host 데이터로 읽을 수 있다(042 core_state0). Core는 상태를 host 데이터로 내보내고 다시 만드는 연산을 제공하며, 왕복은 비트 단위로 같다 (2026-10-17). GTrXL이 K/V를 캐시하는 경우(2026-10-23): 상태에는 층 입력도 함께 두고, 가중치를 교체하면 다음 step 전에 저장된 층 입력에서 새 가중치로 K/V를 다시 계산한다. 그래서 캐시를 쓰든 안 쓰든 출력은 같다(허용 오차 이내). host 내보내기(042 core_state0)는 층 입력과 유효 마스크만으로 상태를 다시 만들 수 있어야 한다 |
| 032-R17 | **MUST** 시간 축 코어(LSTM vs GTrXL)는 BC 본학습 전에 짧은 비교 학습으로 고른다(2026-10-18 사용자 결정, AMB-032-6). ① 학습: 코어마다 S1을 5,300 update 1회(seed 1개)씩 돌린다. 장비는 혼합 유동 운영이다(2026-10-18 사용자 결정): Linux Vulkan 호스트(f32)와 Windows CUDA 호스트(TF32 matmul)가 run 큐 작업자(061-R13·R14)로 두 run을 나눠 맡고 서로 이어받는다. 장비별 update 구간은 `runs/<id>/devices.jsonl`(061-R15)에 남고, 두 run의 장비 비율을 최종 집계해 리포트에 적는다. 두 run의 장비 비율이 크게 다르면 Vulkan f32와 CUDA TF32의 수치 차이가 비교에 섞일 수 있다(공정성 주의). 주 장비 규칙(2026-10-20 사용자 결정, AMB-032-8, 060-R8): CUDA 호스트가 유휴일 때는 CUDA 호스트가 주가 되고, 바쁠 때는 Linux Vulkan 호스트가 주가 된다. 이 규칙은 작업자 우선순위(CUDA 호스트에 더 높은 `--priority`)와 양보 요청(061-R14)으로 이미 구현돼 있다. 데이터 순서(`data.seed`)와 그 밖의 설정(Burn autotune·fusion 포함, 041-R15)은 같다. 학습률 스케줄은 짧은 run 전용으로, warmup 5%(265) 뒤 cosine이 5,300에서 끝난다(`config/bc_core_ablation.toml`). ② 오프라인 지표: `bc-eval --split val --games 500`(두 코어 같은 500게임). 500게임은 val 분할을 **고정 seed로 셔플**해 앞에서 고른다(출처 편중을 줄이기 위해, 2026-10-19 사용자 결정 G-D8, AMB-032-7). 학습 seed는 1개로 유지하고, 그 한계(seed 간 편차를 모름)는 리포트에 적는다. test 분할은 041-AC1용이라 쓰지 않는다. ③ 대전: 두 체크포인트 맞대결(TvT, 맵당 50판, 25판마다 슬롯 교대, `config/plans/core-ablation-h2h.toml`)과 각자 T0 상대 승률(상대 종족 3 × 맵 2 조합당 50판, 코어당 300판·합계 600판, 사용자 확정 2026-10-18, `config/plans/core-ablation-t0.toml`). 샘플링은 확률적(temp 1)이고 z는 없다. 대전 700판의 리플레이를 모두 저장한다(두 plan `save_replay = true`, `<out>/replays/g<id>.rep`, 2026-10-18 사용자 결정). ④ 판정: 규칙을 미리 정하지 않는다. 결과를 사용자에게 보고하고 사용자가 고른다. **결과: GTrXL 채택(2026-10-23 사용자 결정, `reports/m6_core_ablation.md` 4절).** ⑤ 비교 run의 체크포인트·로그·평가 출력은 지우지 않는다(비교 설정은 061-R6 `keep_checkpoints` 예외로 전부 보존, 2026-10-19 사용자 결정 G-D6). (2026-10-23 메모: 이 비교는 Burn으로 했고 결과(GTrXL 채택)는 PyTorch 전환 뒤에도 유지한다. Burn 체크포인트는 보관만 하고 새 코드에서 로드하지 않는다) ⑥ **2차 비교(PyTorch, 2026-10-24 사용자 결정)**: BC 본학습 전에 PyTorch 구현으로 비교를 한 번 더 한다. 코어마다 S1을 **10,000 update** 1회(seed 1개)씩 돌린다(run `core-abl-lstm-pt`·`core-abl-gtrxl-pt`). 설정은 `config/bc_core_ablation_pt.toml`이고, 지금 `config/bc.toml`과 S1 길이(10,000)·warmup(5% = 500, cosine이 10,000에서 끝남)·체크포인트 전부 보존만 다르다(win_open 헤드는 입력이 detach돼 나머지 학습에 영향이 없으므로 bc.toml 그대로 둔다). 장비는 1차와 같다: Windows CUDA 호스트(bf16 혼합, 032-R8)는 **유휴일 때만**, Linux ROCm 호스트(f32)는 상시, run 큐 작업자(061-R13·R14, CUDA 호스트 우선순위가 더 높음)로 나눠 맡고 이어받는다. 두 run의 장비 비율(`uranium-train devices`)을 리포트에 적는다(CUDA bf16과 ROCm f32의 수치 차이가 섞일 수 있음). 평가·판정·보존은 ②~⑤와 같다(plan `config/plans/core-ablation-pt-h2h.toml`·`core-ablation-pt-t0.toml`, 1차와 같은 seed·대진). 진행·결과는 `reports/m6_core_ablation_pt.md`. 판정 전까지 채택 코어는 GTrXL(④)이다. **결과(2026-10-27)**: 맞대결은 LSTM 27승·GTrXL 26승(무 47)으로 거의 동률이었고, 코어는 GTrXL을 유지한다(R20 ④ 모두 켬 채택이 GTrXL 코어, `reports/m6_core_ablation_pt.md` 3절). |
| 032-R18 | **MUST** `win_open` 헤드(2026-10-21 사용자 결정, AMB-031-6). ① privileged 입력 경로가 없고, 코어 출력 `h`는 detach해서 받으므로 정책·인코더·코어·기존 value 헤드에 gradient가 가지 않는다. ② 설정 `value.win_open`(기본 false)이 true일 때만 만든다. 설정 로더는 이 키가 없는 설정을 false로 읽는다(체크포인트 메타의 설정에 없는 키는 기본값으로 채운다). ③ 파라미터는 전역·장치 RNG를 쓰지 않고 고정 seed 생성기로 초기화한다(U(−k, k), k = 1/√fan_in). 전역 RNG를 쓰면 다른 모듈의 초기 가중치와 이후 dropout 난수가 바뀌기 때문이다. 그래서 헤드가 있어도 기존 모듈의 초기 가중치와 학습 난수는 같다(2026-10-23: Burn 지연 초기화 설명을 장치 무관 표현으로 바꿈). ④ `act()`와 학습 forward는 win_open을 value와 따로 돌려준다(꺼져 있으면 없음) |
| 032-R19 | **MUST** 최적화 원칙(2026-10-23 사용자 결정, AMB-032-10). ① 모델 동작은 바꾸지 않는다. 같은 수식을 더 빠르게 계산하는 구현만 쓴다: 커널 융합·직접 작성 커널, gather, CUDA/HIP graph 캡처, torch.compile, K/V 캐시(032-R16), 배치 안 레인 그룹 재구성(041-R3), 메모리 재계산 등. ② 결과 차이는 float 반올림, bf16 혼합 정밀도(032-R8), GPU atomic 덧셈 순서, dropout 난수열 배치 차이까지만 허용한다. ③ 모델 구조·헤드 방식(Selected 순차 선택 포함)·관측 해상도·distance bias·dropout 비율·손실·하이퍼파라미터·추론 저정밀도(FP8·INT8)는 이 원칙 밖이고, 바꾸려면 별도 사용자 결정이 필요하다. ④ 학습 설정은 Linux ROCm 호스트 메모리 예산(060-R3) 안에서 최악 모양으로 학습돼야 채택한다. ⑤ 채택 전에 AC1·AC8, 짧은 학습 loss 곡선(기준과 잡음 범위 안), 속도·메모리 측정을 리포트에 남긴다. ⑥ RL learner(042)에도 같은 원칙을 적용한다(2026-10-31 사용자 결정): 호스트별 레인 그룹(`URANIUM_RL_MICRO_LANES`)과 같은 수식의 learner 최적화만 쓰고, 동등성은 30 update 비교, 속도는 `python -m uranium bench rl`로 잰다(042 구현 구조) |
| 032-R20 | **MUST** 구조 변형 비교(2026-10-25 사용자 결정, AMB-032-11, §2.7). ① 변형은 §2.7의 `[model]` 스위치 4개(`terrain`, `ctx_token`, `attn_pool`, `outer`)이고 서로 조합할 수 있다. 기본은 모두 꺼짐이며, 꺼지면 기본 구성과 비트 동일, 켜도 공유 모듈 초기 가중치는 기본 구성과 같다. ② 비교 run은 5개다: 하나씩 켠 4개(`arch-abl-terrain-pt`, `arch-abl-ctx-pt`, `arch-abl-pool-pt`, `arch-abl-outer-pt`(`outer` = 32))와 네 가지를 모두 켠 1개(`arch-abl-all-pt`). 코어는 GTrXL이고, 설정은 `config/bc_core_ablation_pt.toml`에 `[model]`만 더한 `config/bc_arch_abl_<변형>_pt.toml`이다. 각 run은 기준 체크포인트에서 이어 학습하지 않고 0 step부터 S1 10,000 update를 1회(seed 1개, `data.seed` 1) 학습한다. 기준은 R17 ⑥의 `core-abl-gtrxl-pt` `ckpt-10000`이다(다시 학습하지 않음). 장비는 R17 ⑥과 같은 혼합 운영이고, R17 ⑥이 쓰는 큐(`<shared>/queue/core-ablation-pt.toml`)에 run을 덧붙여 두 작업자가 R17 ⑥ run을 끝낸 뒤 이어서 잡는다. run별 장비 비율(`uranium-train devices`)을 리포트에 적는다. 잡음 측정용 추가 seed는 두지 않는다. ③ 평가: run마다 `bc-eval --split val --games 500`(R17 ②와 같은 500게임). **리그전**: 7개 모델(기준 GTrXL, R17 ⑥ LSTM, 변형 5개, 모두 `ckpt-10000`)의 21쌍, 쌍당 100판(맵 2 × 50판, 25판마다 슬롯 교대, 확률적 샘플링, z 없음, 리플레이 저장, R17 ③ 맞대결과 같은 방식). LSTM–GTrXL 쌍은 R17 ⑥ 맞대결 결과를 그대로 쓴다. 쌍별 승·패·무와 승률(Wilson 95%), 7×7 승률표, 전체 순위(Bradley–Terry, 무승부 0.5, bootstrap 95% 구간)를 낸다. T0 대전(R17 ③ 방식)은 7개 모델 모두 모델당 300판(종족 3 × 맵 2 × 50판)이다(2026-10-25 사용자 결정, 이전: 리그전 뒤 고른 후보만). LSTM·GTrXL은 `config/plans/core-ablation-pt-t0.toml`, 변형 5개는 `config/plans/arch-abl-pt-t0.toml`이다. 평가(val 500게임 채점·리그전·T0)는 큐의 7개 run 학습이 모두 끝난 뒤 함께 한다(2026-10-25 사용자 결정, R17 ⑥ 평가도 이때 한다). **실행(2026-10-26 사용자 결정, AMB-020-5)**: 대전은 평가 작업(020-R15) 하나로 돌린다. 단계 순서는 R17 ⑥ 맞대결 → 리그전 20쌍 → R17 ⑥ T0 → 변형 T0다. 추론은 CUDA 호스트(bf16 혼합)가 주 장비이고, CUDA 호스트가 빠진 동안에는 Linux ROCm 호스트(f32, 첫째 > 둘째)가 새 판부터 맡는다. 판마다 추론 장비를 기록하고(020-R15 ⑧), 결과 보고에 모델·단계별 추론 장비 비율을 함께 적는다(정밀도 차이가 섞일 수 있음). val 500게임 채점은 대전과 병행해 7개 모두 첫째 Linux ROCm 호스트(f32)에서 한다(같은 장비라 7개끼리 비교 공정). 리플레이 플레이어 이름은 020-R16이다. ④ 판정 규칙은 미리 정하지 않는다. 결과를 보고 사용자가 채택 여부를 고르고, 그 전까지 BC 본학습(041)은 시작하지 않는다. **결과: 모두 켬(`arch-abl-all-pt` 구성) 채택(2026-10-27 사용자 결정, `reports/m6_arch_ablation_pt.md` 3·4절)**. 평가 결과는 val 7개 중 1위, 리그전 7위, T0 3위(16.7%)였고, 독립 검토(`reports/m6_arch_ablation_review.md`)는 대전 약세를 잡음 밖 신호로 보고 기준 GTrXL을 권고했다. 사용자는 이 검토를 본 뒤 정했다. ⑤ 체크포인트·로그·평가 출력은 지우지 않는다(`keep_checkpoints = 0`). 결과는 `reports/m6_arch_ablation_pt.md` |
| 032-R21 | **MUST** (2026-11-09 초안, 사용자 확인 대기) 헤드 우회로와 스칼라 직접 입력(사용자 결정 2026-11-09, §2.8 `bypass`). ① ActionType 헤드가 질문 `[h, c, s_raw]`로 E에 교차 어텐션하고, 0으로 초기화한 출력 투영(입력은 기본안(미확정) 어텐션 출력과 `s_raw`, AMB-032-12)의 결과를 GLU 출력과 `ar`에 더한다. ② 켠 직후(학습 전) 출력은 끈 모델과 같다. ③ 빈 엔티티 집합은 0 출력이고 CUDA graph 캡처(032-R19)에서 다시 컴파일하지 않는다. ④ 크기·head 수·`s_raw` 부분집합·출력 투영 입력은 [SPEC AMBIGUOUS] AMB-032-12다. **AMB-032-12가 정해지기 전에는 P2의 우회로 코드(U4b)를 쓰지 않는다** |
| 032-R22 | **MUST** (2026-11-09 초안, 사용자 확인 대기) 공간 ResBlock pre-activation 스위치(사용자 결정 2026-11-09, §2.8 `spatial_preact`). 켜면 공간 인코더 ResBlock 6개가 `x + Conv(ReLU(Conv(ReLU(x))))`이다. 끄면 지금 구조(§2.2, 끝 ReLU)와 비트 동일하다. 파라미터 목록·모양·초기 가중치는 켜도 같다 |
| 032-R23 | **MUST** (2026-11-09 초안, 사용자 확인 대기) 코어 수정 스위치(§2.8). F1 `core_skip`: Core 입력을 0 초기화 Linear로 Core 출력에 더한다(순환 상태는 그대로). F2 `recency_init`: GTrXL `rel_table`을 지금과 같은 난수 뒤에 최근 step 선호 항을 더해 초기화한다. 경로 2(041-R29)에서 어느 것을 켤지, F2 항의 꼴·크기, 그 밖의 후보(F3 게이트 교체 등, 쓰려면 별도 spec 문안)는 코어 어텐션 조사(U2) 뒤로 보류한 [SPEC DEFERRED(U2)] AMB-032-14다. F2 항에 의존하는 코드는 그 전에 쓰지 않는다 |
| 032-R24 | **MUST** (2026-11-09 초안, 사용자 확인 대기) 학습 전용 보조 모듈(사용자 결정 2026-11-09, §2.8 `aux_priv`·`aux_jepa`). ① `aux_priv`: h_t로 privileged 벡터 전체를 예측하고, privileged가 없는 step은 뺀다(존재 마스크, host batch 변경 필요). ② `aux_jepa`: k마다 h_t로 t+k의 `[e_pool, s_pool]`(`outer` 더하기 전, stop-grad)를 예측하고, 게임 경계·학습 묶음 끝을 넘는 쌍은 마스크한다. ③ 두 모듈 모두 h_t를 detach하지 않는다(Core·인코더에 학습 압력을 주는 것이 목적). ④ `forward_seq`가 aux를 요청받았을 때만 계산하고 `act()`·추론 graph에는 없다(AC5 확장). 유효 step은 패딩이 아닌 실제 step이고 레이블 가중치와 따로다(§2.8). ⑤ 구조 세부(예측기 크기, k 목록, 정답 정규화, 손실 꼴, privileged 차원 가중, 정답 계산 방식)는 [SPEC AMBIGUOUS] AMB-032-13이다. **AMB-032-13이 정해지기 전에는 P2의 aux 코드(U4c)를 쓰지 않는다.** 손실 세기는 041-R23·042-R21 |
| 032-R25 | **MUST** (2026-11-09 초안, 사용자 확인 대기) 죽은 채널 통계(사용자 결정 2026-11-09). ① **ReLU 출력**: stem 출력(FiLM 앞 ReLU), 다운샘플 conv 4개의 `ReLU(Conv stride2)` 출력(잔차 밖이라 pre-activation에서도 죽을 수 있다), 각 ResBlock 안 첫 conv 뒤 ReLU 출력 6개. 층마다 `pos`(측정 묶음의 그 층 원소 중 0보다 큰 비율)와 `dead`(측정 묶음 전체에서 한 번도 0보다 크지 않은 채널 수)를 낸다. ② **블록 출력**: 내려가는 단계 출력 4개와 bottleneck 출력 2개(둘째가 `S8`)의 `pos`·`dead`. 지금 구조(끝 ReLU)에서만 뜻이 있고, `spatial_preact`에서는 음수가 정상이라 참고값일 뿐이다. ③ **거의 상수인 채널 수**: `S8`의 채널과 `s_pool`의 차원마다 측정 묶음 안 표준편차를 구해, 문턱(AMB-041-14) 아래인 채널·차원 수 `const`를 낸다. 두 구조 모두에서 뜻이 있고 10.4의 증상(`s_pool`이 상수)을 직접 잡는다. 측정 묶음은 그 update의 학습 묶음 중 실제 step(패딩 아님) 행이다. 계산은 학습 결과를 바꾸지 않는다: 학습 forward의 활성값을 읽거나, 따로 no_grad로 계산하면 dropout을 끄고 학습 RNG를 쓰지 않는다(켜고 끈 학습이 CPU에서 비트 동일). 설정 키·기록 주기·경보는 041-R25·042-R21이다(키가 없으면 계산하지 않음) |

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

- §2.8 스위치의 추가 파라미터(2026-11-09 개정 초안, 기본안 크기 기준 어림): `bypass` 약 0.4M, `core_skip` 약 0.4M, `aux_priv` 약 0.2M, `aux_jepa` k 3개 약 1.6M. aux는 학습 전용이라 추론 비용에 들어가지 않는다. 합쳐도 032-R2 범위(10M–40M) 안이다. 학습 메모리는 060-R3 최악 모양(ROCm BC 레인 그룹 1·T64·최대 N, RL learner는 BC 재forward 포함)으로 확인하고, 넘으면 결과를 바꾸지 않는 방법(재계산 등, 032-R19)만 쓰고, 그래도 넘으면 사용자에게 보고하며 모델은 바꾸지 않는다(060-AC7).

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
- `forward_seq`는 인자 `aux: bool = False`를 더 받는다(§2.8, 2026-11-09 개정 초안). 학습기가 aux 손실을 쓸 때만 참이고, RL BC 참조 재forward·자기 모방 레인·진단·`bc-eval` 행동 지표는 거짓이다. aux 모듈이 있고 `aux`가 참이면 `SeqOutput`에 aux 예측(`aux_priv` [B·T, P], `aux_jepa` k별 [B·T, 512])과 JEPA 정답(기울기 끊은 `[e_pool, s_pool]` [B·T, 512])을 함께 돌려준다. 없으면 이 키가 없다. `act()`의 시그니처·출력은 바뀌지 않는다.
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
- 032-AC11 (2026-10-25, 032-R20): §2.7 스위치가 모두 꺼진 모델은 기본 구성과 파라미터 목록·출력이 같다(기존 테스트 그대로 통과, AC10 포함). 각 변형과 네 스위치를 모두 켠 조합에서 ① 공유 모듈 초기 가중치가 기본 구성과 같고 ② step 실행이 시퀀스 실행과 같으며(032-R5, `outer` 구간 경계 포함) ③ 마스크(AC4)·privileged 경로 없음(AC5)·엔티티 재계산 동일(041-R17)·체크포인트 왕복이 성립한다(CPU 테스트). 추론 graph act = eager는 GPU 테스트로 확인한다. RL 경로(2026-10-27, §2.7): 줄마다 다른 결정 수에서 시작한 시퀀스의 head log-prob·value·마지막 상태·기울기가 한 결정씩 계산한 것과 같고, 시작이 구간 경계면 BC 경로와 같으며, RL 학습기가 이 모델로 update한다(CPU 테스트). 다른 `[model]` 설정으로 재개하면 거부하고, `[model]`에 모르는 키가 있으면 시작을 거부한다.
- 032-AC12 (2026-10-25, 032-R20): 변형 run 5개가 10,000 update에서 끝나고, val 지표, 리그전 쌍별 승률(Wilson 95%)·7×7 표·전체 순위, run별 장비 비율·update 시간이 `reports/m6_arch_ablation_pt.md`에 기록되며, 사용자가 채택 여부를 정한다. — 통과(2026-10-27: 기록 3절, 사용자 결정 모두 켬 채택).

- 032-AC13 (2026-11-09 초안, 사용자 확인 대기, 032-R21~R24): §2.8 스위치가 모두 꺼진 모델은 지금 모델과 파라미터 목록·출력이 같다(AC10·AC11 테스트 그대로 통과, Checkpoint 1 체크포인트 로드와 같은 출력). 실측 확인: 모든 스위치를 끈 코드로 Checkpoint 1(`bc-dagger-r1/ckpt-1137`)의 대상 맵 val 500게임(032-R17 ②, seed 1, Linux ROCm 호스트 f32)이 기존 값 72.19%를 그대로 낸다. 각 스위치와 모두 켠 조합(§2.7 네 스위치 켬 포함)에서 ① 공유 모듈 초기 가중치가 꺼진 모델과 같고(`recency_init`은 `rel_table`에 더한 항만 다름) ② `bypass`·`core_skip`을 켠 직후의 출력이 끈 모델과 같으며(0 초기화) ③ step 실행이 시퀀스 실행과 같고(`outer` 구간 경계, RL 위상별 구간 포함) ④ 마스크(AC4)·엔티티 재계산 동일(041-R17)·체크포인트 왕복이 성립한다(CPU 테스트). `spatial_preact`를 켠 ResBlock은 정의식과 같은 출력을 낸다. 추론 graph act = eager와 재컴파일 없음은 GPU 테스트로 확인한다. `[model]`에 모르는 키가 있거나 다른 `[model]`로 재개하면 거부한다(AC11과 같음).
- 032-AC14 (2026-11-09 초안, 사용자 확인 대기, 032-R11·R24): `act()`와 추론 graph에 aux 모듈과 privileged 텐서가 없다(AC5 확장, 테스트). `forward_seq(aux=False)`는 aux 모듈을 돌리지 않는다. `aux_priv` 손실에서 privileged 없는 step과 패딩 step이 빠지고, 레이블이 무효인 실제 step은 들어간다. JEPA 정답은 `outer` 더하기 전 값이다. `aux_jepa` 정답은 기울기가 없고(정답 쪽 파라미터 gradient 0), 게임 경계와 학습 묶음 끝을 넘는 쌍은 마스크된다(경계 테스트). aux 모듈을 켜고 손실 세기가 0이거나 키가 없으면 aux 손실을 더하지 않고, 공통 파라미터의 기울기가 aux 없는 모델과 CPU에서 비트 일치한다.
- 032-AC15 (2026-11-09 초안, 사용자 확인 대기, 032-R25): 알려진 활성값에서 층별 `pos`·`dead`와 `S8`·`s_pool`의 `const`가 정의와 같고, 설정 키가 없으면 통계를 계산하지 않으며, 통계 계산을 켜고 끈 학습(BC·RL 각 몇 update)이 CPU에서 비트 일치한다.

## Risks & Open Questions
- **모델 2 스위치 (2026-11-09 개정 초안, Review — 사용자 확인 대기, §2.8)**:
  > **[SPEC AMBIGUOUS] AMB-032-12**: 헤드 우회로(`bypass`) 크기, `s_raw` 부분집합, 출력 투영 입력.
  > - 기본안(미확정): d_b = 256, 8 head × 32(`attn_pool`과 같은 꼴), K·V는 `Linear(256→256)`, 질문 입력 `[h(512), c(256), s_raw]`, 출력 투영 입력 `[어텐션 출력, s_raw]`. `s_raw` = 030 §4의 시간 2개, 자원(미네랄·가스·supply 3개), 관측된 적 유닛 타입별 최대 동시 관측 수 전부(진단에서 `c`가 보존하지 못한 값, 10.2.2·10.2.3). 업그레이드·테크·아군 수·직전 액션은 넣지 않음.
  > - 대안: `s_raw` = 스칼라 전체(030 §4), 또는 적 최대 관측 수만. 출력 투영은 어텐션 출력만(`s_raw`는 질문에만).
  > - 함께 확인: `s_raw`는 행동 헤드에만 닿아 h_t·`c`의 스칼라 보존 문제(사용자 항목 ⑥)는 일부만 해결된다. 스칼라 인코더 자체를 고칠지는 이 개정에 없다.

  > **[SPEC AMBIGUOUS] AMB-032-13**: 학습 전용 보조 모듈(`aux_priv`·`aux_jepa`) 구조 세부.
  > - 기본안(미확정): `aux_priv` = `MLP(512→256→P)`, 손실은 차원 평균 MSE, privileged 차원 가중은 모두 1(텐서는 030·010이 이미 정규화한 값). `aux_jepa` k = {1, 8, 32}, 예측기 `g_k` = k마다 따로 `MLP(512→512→512)`(은닉 ReLU), 정답 정규화는 학습 파라미터 없는 LayerNorm(차원 방향), 손실은 1 − 코사인 유사도, k별 유효 쌍 평균을 k 사이 평균. 정답은 같은 forward의 온라인 인코더 값(EMA 사본 없음). 붕괴 감시로 정답 차원별 표준편차 평균을 기록(041-R25·042-R21 로그).
  > - 대안: 정규화 L2 손실, 예측기 공유 + k 임베딩, privileged 차원 묶음별(자원·업그레이드·테크·유닛 구성) 같은 몫 가중, EMA 정답 인코더, **바깥 상태 없는 정답 패스**(정답 `[e_pool, s_pool]`을 `ctx_token`·`outer` 토큰·FiLM 없이 따로 no_grad로 인코딩해, 정답이 모델 자신의 `c`·`h_o`에 기대지 않게 함. 인코더 계산이 늘어남).

  > **[SPEC DEFERRED(U2)] AMB-032-14**: 경로 2에 쓸 코어 수정 세트와 F2 항(코어 어텐션 조사 U2 뒤 결정, 보류 표기도 사용자 확인 대상).
  > - 기본안(미확정): 코어 어텐션 조사(가중치 분석·헤드별 엔트로피·오프라인 개입) 결과를 보고 정한다. 조사 전 기본안은 F1(`core_skip`) 켬, F2(`recency_init`)는 041-R24 λ 비교 run의 "F1 대 F1+F2" 결과로 정함. F2 항은 `b(d) = −α·ln(1 + d)`, α = 1, 모든 층·head.
  > - 대안: F1·F2 둘 다 끔(지금 코어 그대로), F3(GRU 게이트를 잔차 더하기로 교체 등, 쓰려면 별도 spec 문안과 사용자 확인).
  - **JEPA 붕괴**: stop-grad 정답을 같은 인코더로 만들면 인코더가 요약을 상수에 가깝게 만들어 예측 손실을 줄이는 붕괴가 가능하다. 인코더는 정책 손실도 받으므로 막힐 것으로 보지만, 정답 표준편차를 감시하고 줄어들면 보고한다(AMB-032-13 기본안).
  - **JEPA 정답이 모델 자신의 상태에 기댐**: §2.7을 켠 모델(경로 2)에서는 `e_pool`이 `ctx_token`(`c`)·`outer` 토큰(`h_o`)과 함께 인코딩되고 `s_pool`은 `h_o`의 FiLM을 받는다. 그래서 정답 일부는 h_t가 이미 가진 `c`·`h_o`로 맞힐 수 있고, 예측 손실이 관측 내용 대신 그 쉬운 부분을 배울 수 있다. 대안은 AMB-032-13의 바깥 상태 없는 정답 패스다.
  - **privileged 보조 목표의 범위**: 정답은 상대의 실제 상태라 h_t가 "보지 못한 것"을 추정하도록 학습된다. 추론 입력은 자기 시야 그대로이고 실전(SC:R)에는 privileged가 없다(010-R10). 맵핵 금지(010-R5, 030-R2)는 관측 제한이므로 유지된다(사용자 결정 2026-11-09, `reports/jepa_survey.md` 8절).
  - **pre-activation 뒤 재붕괴**: 블록 안 ReLU와 다운샘플 conv 뒤 ReLU(잔차 밖)가 여전히 죽을 수 있다. 죽은 채널 통계(그 ReLU의 `dead`, `S8`·`s_pool`의 `const`)와 경보(041-R25)로 일찍 본다.
  - **비교 가능성**: 스위치는 체크포인트 메타에 남으므로 Checkpoint 1(모두 꺼짐)과 모델 2 경로를 같은 측정 프로토콜(050-R9)로 비교한다.
- **구조 변형 비교 (2026-10-25, 032-R20)**:
  - 바깥 LSTM(`outer`)은 학습 묶음(T = 64) 안의 구간 전환(`outer` = 32면 1회)에서만 학습 신호를 받는다. 묶음을 넘는 보존은 forward로만 이어지고 "무엇을 오래 들고 갈지"는 배우지 못한다. 더 긴 학습 시퀀스는 Linux ROCm 호스트 메모리(060-R3) 측정 뒤 별도로 정한다. → 2026-10-27 사용자 결정: 모두 켬 채택 뒤 BC 본학습도 T = 64(032-R5, `config/bc.toml` `seq_len`) 그대로 한다. 이 한계는 남는다.
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

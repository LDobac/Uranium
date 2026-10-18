# M6 — 시간 축 코어 비교 (LSTM vs GTrXL, 032-R17)

- 상태: **계획 확정, 실행 전** (2026-10-18 18:38)
- spec: [032](../spec/032-model-architecture.md) §2.4, 032-R17, 032-AC7, AMB-032-6
- 구조 설명 페이지(그림): 사용자 claude.ai 아티팩트 "Uranium 정책망 구조"(비공개)

이 문서는 실험 계획, 실험 전에 정리한 배경과 예상, 진행 기록, 결과를 한곳에 둔다. 진행하면서 아래 "진행 기록"에 시각과 함께 덧붙이고, 위 절은 고치지 않는다. 그래야 나중에 무엇을 언제 알았는지 되짚을 수 있다.

## 1. 배경 (실험 전 정리, 2026-10-18)

### 1.1 코어는 모델의 한 부분이다
- 모델은 인코더 3개 → Core → 자기회귀 헤드 6개 + ValueHeads다(032 §1). 파라미터는 실측 11.3M이고, Core는 그중 약 2.6M(spec §4 추정치)이다.
- 공간 축(한 step 안에서 유닛 사이 관계)은 항상 EntityEncoder 트랜스포머가 맡는다(3층, d256, 엔티티 최대 512개).
- 시간 축(step과 step 사이)만 Core가 맡는다. 비교 대상은 이 자리에 들어가는 두 구현이다.
  - `LstmCore`: LSTM 768→512 1층. 기억을 상태 하나에 덮어쓰며 넘긴다. 창 제한이 없다.
  - `GtrxlCore`: 최근 M=64 step의 core 입력에 causal self-attention 2층(d512), GRU 게이트 residual. 각 층 메모리에 그 층의 입력을 저장하므로(`core.rs` `forward_seq`) 이론상 수용 범위는 2 × 64 = 128 step이다.
- 학습은 둘 다 T=64 truncated BPTT다(032-R5). 64 step보다 먼 의존성은 둘 다 배우기 어렵다.

### 1.2 64 step은 게임 시간으로 고정되지 않는다
- step 간격은 직전 액션의 Delay 헤드 값이다. 1–32 프레임 bucket 16개(031-R9) 중 하나이고, Fastest에서 1프레임은 약 42ms다.
- 64 step = 2.7초(모두 1프레임) ~ 86초(모두 32프레임). 실제 평균은 측정하지 않았다. 실효 APM 150–200을 가정한 추정은 약 20–25초다.

### 1.3 장기 기억은 관측이 맡는다
- 관측(030)은 신경망 앞단의 일반 코드(`uranium-env` features)이고, 학습되는 가중치가 없다.
- 이 코드가 장부(`crates/uranium-env/src/features/memory.rs`)를 두고 매 step 입력에 다시 넣는다.
  - 적 건물: 마지막 목격 상태 + 경과 프레임(030-R2, `is_remembered`·`frames_since_seen`).
  - 정찰 기록: 타일별 `last_seen_age`(030 §3, AMB-030-1).
- 그래서 Core에 남는 일은 단기·중기 흐름이다. 예: 이동 방향, 방금 시야에서 사라진 적 유닛, 진행 중인 자기 행동. 사라진 적 유닛의 과거 위치는 관측에 없어서 Core 범위 밖에서는 잊힌다.

### 1.4 실험 전 예상 (Claude, 측정 아님)
- GTrXL이 LSTM보다 뚜렷하게 좋을 가능성은 낮다고 봤다.
  - GTrXL 논문(Parisotto et al., 2019)의 큰 이득은 오래전 정보를 정확히 꺼내 오는 기억 과제에서 나왔다. 우리 구조는 그 일을 관측 장부가 맡는다.
  - 교전처럼 delay가 짧은 구간에서는 64 step 창이 몇 초밖에 안 된다. LSTM은 창이 없다.
  - 학습 길이(T=64)가 같아서 GTrXL의 구조적 이점이 드러날 여지가 작다.
  - AlphaStar·OpenAI Five는 시간 축에 LSTM을 썼다.
  - CUDA 추론·학습 속도가 거의 같다(아래 1.5). 성능 외에 GTrXL을 고를 이유가 없다.
- GTrXL이 이길 수 있는 경우: RL 단계 학습 안정성(논문 주장), 관측에 없는 단기 정보의 정확한 회상.
- 이 예상은 실험 결과와 비교하기 위해 남긴다.

### 1.5 이미 측정한 것 (M5, `reports/m5_model.md`, CUDA f32, B16·N256)
| | LSTM | GTrXL |
|---|---|---|
| 추론 중앙값 / p95 | 58.9 / 61.3ms | 58.7 / 66.6ms |
| 학습 B2·T16 | 69.8 samples/s | 70.4 samples/s |

BC 본 설정(레인 32 × T64)의 GTrXL update 시간은 아직 재지 않았다. 첫 run에서 기록한다.

**추가 측정 (2026-10-18 20시, 실행 전 확인, `reports/m6_bc_preflight.md` 3차 최적화 후 코드):**
- 설정과 조건:
  - `config/bc_core_ablation.toml`: 레인 32 × T64, `micro_lanes 1`, `pad_entities 64`, 대칭 증강 4종
  - 실데이터 `stardata/p00` S1(이 표본의 엔티티 수 N은 64)
  - 랜덤 초기화 망, 장비마다 GPU 단독
- update 시간은 `bc --steps 3~4`의 정상 상태 값이다. 모듈 시간은 `bc-profile`(레인 그룹 = 64행, update = 32그룹)로 쟀다.

| 항목 | 이 PC CUDA (TF32) LSTM | 이 PC CUDA GTrXL | bc250_1 Vulkan f32 LSTM | bc250_1 Vulkan f32 GTrXL |
|---|---|---|---|---|
| update 시간 | 25.0–26.1 s | **22.6–23.4 s** | 128–152 s | 134–156 s |
| 레인 그룹 forward / backward | 0.194 / 0.599 s | 0.159 / 0.537 s | 0.588 / 3.622 s | 0.565 / 3.563 s |
| └ core forward / backward | 0.032 / 0.057 s | 0.004 / 0.005 s | 0.037 / 0.068 s | 0.023 / 0.037 s |
| 최대 GPU 메모리 | 5.1 GB / 16 GB (nvidia-smi, 메모리 풀 포함) | 5.4 GB | VRAM 0.5 + GTT 3.2 = 3.5 GB / 15.9 GB | VRAM 0.5 + GTT 3.2 = 3.5 GB |
| OOM·장치 리셋 | 없음 | 없음 | 없음 | 없음 |

- **bc250_1 Linux에서 GTrXL 학습이 된다.** Windows Vulkan에서 TDR로 측정할 수 없었던 문제는 Linux에서 나지 않았다(update 3회와 프로파일 모두 정상 종료).
- 두 코어의 차이는 core 부분뿐이다. 그 차이가 update 시간에서 차지하는 비중은 작다. 나머지는 spatial encoder(conv)와 location head(ConvTranspose)가 차지하고, 두 코어가 같다.
  - CUDA에서는 GTrXL이 T 방향 병렬이라 약 10% 빠르다.
  - Vulkan에서는 거의 같다. 측정 사이 편차(약 ±10%)가 차이보다 크다.
- 메모리는 N과 엔티티 encoder에 따라 커지고, core는 영향이 작다. N=512 최악 조건은 M5 메모리 측정(`reports/m5_bc_memory.md`, CUDA 16 GB에서 레인 1 동작)을 따른다. 이번 표본은 N=64라 최대치가 아니다.
- 대응이 필요한 OOM이나 속도 저하는 없었다.

### 1.6 왜 지금 나왔나
- AMB-032-4가 "LSTM 시작, M5에서 비교"로 Resolved 처리돼 미결 목록에서 빠졌다.
- 032에는 채택 기준 한 줄만 있었고, 절차와 AC가 없었다. 032-AC6은 교체 가능 여부만 요구했다.
- M5 리포트 "다음 단계"와 진행 메모에서도 빠졌다.
- 2026-10-18 BC 본학습 승인 직전, 사용자 질문("컨텍스트 크기가 몇이지")에서 시작해 발견했다. 재발 방지로, 배치 전 점검 때 spec 본문의 "Mx에서 비교/채택" 같은 약속과 미실행 AC를 대조하기로 했다.

## 2. 결정 (2026-10-18, 사용자)

| 항목 | 결정 |
|---|---|
| 비교 여부 | 둘 다 짧게 학습해 비교한다("일단 실험하는 거니까 둘 다 테스트, 짧게 1–2일") |
| 학습 길이 | 코어마다 S1 5,300 update, 각 1회(seed 1개) |
| 장비 | 혼합 유동 운영(2026-10-18 사용자 결정): bc250_1(Vulkan f32)은 상시, 이 PC(CUDA, TF32 matmul)는 쉬는 동안 run 큐 작업자로 두 run을 나눠 맡고 서로 이어받는다(061-R13·R14). 장비별 update 구간은 `runs/<id>/devices.jsonl`(061-R15). 두 run의 장비 비율을 최종 집계해 4절에 적는다 |
| 학습률 스케줄 | 짧은 run 전용: warmup 5%(265) + cosine이 5,300에서 끝남. 비교 run은 본학습에 다시 쓰지 않는다 |
| 오프라인 지표 | `bc-eval --split val --games 500`(두 코어 같은 500게임) |
| 대전 | 맞대결과 각자 T0 상대 둘 다. 맞대결은 맵당 50판(25판마다 슬롯 교대). 확률적 샘플링(temp 1) |
| T0 판수 | 상대 종족 3 × 맵 2 조합당 50판(코어당 300판, 합계 600판). 처음에는 Claude가 041-AC2 규모를 임시로 넣었고, 같은 날 사용자가 그 값으로 확정했다(다른 세션에서 결정) |
| 판정 | 규칙을 미리 정하지 않는다. 결과를 보고받고 사용자가 고른다 |
| 보존 | 비교 run의 체크포인트·로그·평가 출력은 지우지 않는다(최종 보고서용) |

## 3. 실행 절차

시작 전 확인(선행 조건):
- BC 출력에 영향을 주는 결정이 모두 확정·구현돼 master에 있어야 한다(대규모 배치 전 점검 규칙). 2026-10-18 저녁 기준, 다른 세션이 진행 중인 항목:
  - 030-R4/040-R7 대칭 증강에서 transpose 4종 제외(좌우·상하 뒤집기만) + start_loc 변환 수정 → master에 `030-R4` 증강 커밋이 들어왔는지 확인한다.
  - 041-R6 클래스 가중치(사용자 답 대기).
  - 060-R6 watchdog 자동 재시작(출력 무관, 장시간 실행 안전장치).
- `config/bc_core_ablation.toml`이 그 시점의 `config/bc.toml`과 S1 길이·warmup 외에 같은지 확인한다(`warmup_cosine_schedule_041_r3` 테스트가 검사).
- 실행 시작은 사용자 승인 후에 한다.

시작 전 남은 확인 (2026-10-18 19:10 정리, 아직 하지 않음):
1. **val 500게임의 출처 편중.** `bc-eval`은 val 분할을 셔플 없이 shard 순서대로 앞에서 500게임 읽는다(`bc_eval.rs` `evaluate`). 두 코어가 같은 게임을 보는 것은 보장된다. 다만 데이터셋 순서가 정렬된 고정 순서라서(1076101) 500게임이 소스 하나에 몰릴 수 있다. 소스·맵 분포를 확인하고, 치우치면 "소스별 균등" 또는 "고정 seed 셔플 500게임"을 사용자에게 묻는다.
2. **GTrXL은 BC 본 설정으로 학습해 본 적이 없다.** BC 준비 검증(preflight 1–3차, 이어 하기, 최적화)은 모두 LSTM이었다. GTrXL은 벤치(B2·T16)만 돌렸다. 레인 32 × T64 × N512에서 메모리와 update 시간을 모른다. Vulkan에서는 GTrXL 학습을 측정한 적이 없다(Windows TDR). CUDA와 bc250_1 양쪽에서 짧은 스모크가 필요하다.
3. **장비를 섞으면 "약 2일"이 맞지 않는다.** 2일은 CUDA 단독 32.5 s/update 기준이다(이후 약 18% 단축). bc250_1 Vulkan은 최적화 전 169 s/update로, 단독이면 5,300 update에 약 10일이다. 최적화 후 값은 측정 중이다. 두 run의 장비별 update 구간 비율이 다르면 TF32와 f32의 수치 차이가 섞이므로, 장비별 구간을 기록한다.
4. **맞대결 경로 미시험.** 코어별 모델 정책 테스트는 있지만(`crates/uranium-agent/tests/model_policy.rs`), 한 게임에 LSTM 모델과 GTrXL 모델이 함께 들어간 실행은 없다. `uranium-agent random-checkpoint --core lstm|gtrxl` 두 개로 2–4판 스모크를 한다.
5. **seed 1개.** 작은 차이는 seed 잡음일 수 있고, 판단할 근거가 없다. 결과 보고에 함께 적는다.

설정 파일:
- `config/bc_core_ablation.toml`: `config/bc.toml`에서 `warmup_steps = 265`, `[steps] s1 = 5300`만 다르다. 실행 전에 `bc.toml`이 바뀌었으면 맞춘다.
- `config/plans/core-ablation-h2h.toml`: 맞대결 100판.
- `config/plans/core-ablation-t0.toml`: 코어마다 T0 T/P/Z × 2맵 × 50판 = 300판, 합계 600판.

학습은 run 큐 작업자로 돌린다(2026-10-18 결정, 061-R13·R14). 큐 파일 `<shared>/queue/core-ablation.toml`:
```toml
[defaults]
datasets = ["replays/shards/stardata", "replays/shards/community", "replays/shards/scr_ladder", "replays/shards/personal"]
config = "config/bc_core_ablation.toml"

[[run]]
id = "core-abl-lstm"
core = "lstm"

[[run]]
id = "core-abl-gtrxl"
core = "gtrxl"
```
```
cargo build --release -p uranium-train -p uranium-runner --features cuda     # 이 PC
cargo build --release -p uranium-train                                       # bc250_1
# bc250_1: 상시 실행 (systemd user 서비스 tools/worker/uranium-worker.service, 또는 setsid nohup)
uranium-train worker --queue <shared>/queue/core-ablation.toml --backend vulkan --priority 1 --log runs/worker.log
# 이 PC: 쉬는 동안 실행, 멈출 때 Ctrl+C (현재 run 체크포인트·push·잠금 해제)
uranium-train worker --queue <shared>/queue/core-ablation.toml --backend cuda --priority 10 --log runs/worker.log
```
- 작업자는 큐 순서대로 빈 run을 잡는다. 빈 run이 없으면 이 PC(우선순위 10)가 bc250_1(1)이 돌리는 run에 양보를 요청한다. bc250_1은 다음 update 끝에서 체크포인트·잠금 해제 후 넘기고, 이 PC를 멈추면 다시 이어 간다.
- 장비별 update 구간: `<shared>/runs/core-abl-lstm/devices.jsonl`, `<shared>/runs/core-abl-gtrxl/devices.jsonl`(시작·체크포인트·종료마다 host·백엔드·정밀도·step).
- **공정성 주의:** 두 run의 장비별 update 비율이 크게 다르면 Vulkan f32와 CUDA TF32의 수치 차이가 비교에 섞일 수 있다. 4절에 run별 장비 비율을 집계해 적는다.
```
# (참고) 작업자 없이 한 PC에서 직접 돌릴 때
DATA="--dataset replays/shards/stardata --dataset replays/shards/community --dataset replays/shards/scr_ladder --dataset replays/shards/personal"
uranium-train bc $DATA --run core-abl-lstm  --config config/bc_core_ablation.toml --core lstm  --backend cuda
uranium-train bc $DATA --run core-abl-gtrxl --config config/bc_core_ablation.toml --core gtrxl --backend cuda
# 3) 오프라인 지표 (val 같은 500게임)
uranium-train bc-eval $DATA --checkpoint <shared>/checkpoints/core-abl-lstm  --split val --games 500 --core lstm  --config config/bc_core_ablation.toml --backend cuda
uranium-train bc-eval $DATA --checkpoint <shared>/checkpoints/core-abl-gtrxl --split val --games 500 --core gtrxl --config config/bc_core_ablation.toml --backend cuda
# 4) 대전
uranium-env run --plan config/plans/core-ablation-h2h.toml --backend cuda --out runs/core-ablation-h2h
uranium-env run --plan config/plans/core-ablation-t0.toml  --backend cuda --out runs/core-ablation-t0
```

산출물 위치(지우지 않음):
- 체크포인트: `<shared>/checkpoints/core-abl-lstm/`, `<shared>/checkpoints/core-abl-gtrxl/`(1시간마다, 최종 `ckpt-5300`)
- 학습 로그: 각 run의 stderr JSON을 `runs/core-ablation/<core>.train.log`로 저장한다
- 평가 출력: `runs/core-ablation/<core>.val500.json`
- 대전: `runs/core-ablation-h2h/`, `runs/core-ablation-t0/`(`games.jsonl`, `results.jsonl`, `summary.md`)
- 이 문서의 결과 표는 위 파일에서 옮긴다.

## 4. 결과

(실행 후 기록)

| 지표 (val 500게임) | LSTM | GTrXL |
|---|---|---|
| 총 loss | | |
| action_type top-1 / top-5 | | |
| delay MAE | | |
| target_loc 타일 오차 | | |
| selected F1 | | |
| update 시간(s, GPU 단독) | | |
| 장비 비율 (update 수: 이 PC CUDA / bc250_1 Vulkan) | | |

| 대전 | 결과 (Wilson 95%) |
|---|---|
| 맞대결 LSTM 승률, 투혼 50판 | |
| 맞대결 LSTM 승률, 서킷브레이커 50판 | |
| T0 상대 승률 LSTM (T/P/Z × 2맵) | |
| T0 상대 승률 GTrXL (T/P/Z × 2맵) | |

사용자 결정: (결과 보고 후)

## 5. 진행 기록

시각은 이 PC 기준이다. 항목을 지우거나 고치지 않고 아래에 덧붙인다.

- 2026-10-18 18:38 — 계획 확정. spec 032-R17·AC7, AMB-032-6 등록, 설정·plan 작성. 실행은 아직 하지 않음.
- 2026-10-18 19:01 — T0 대전 판수 사용자 확정: 조합당 50판(코어당 300판, 합계 600판). 맞대결·샘플링은 변경 없음.
- 2026-10-18 19:05 — 시작 전 남은 확인 5가지를 3절에 정리. 장비 혼합 유동 운영(bc250_1 + 이 PC, worker 큐)은 다른 세션이 worker 구현과 함께 032-R17 ①·이 문서 2·3절에 반영하기로 함.
- 2026-10-18 20:22 — GTrXL 실제 BC 설정 확인(1.5절): 이 PC CUDA GTrXL 22.6–23.4 s/update(LSTM 25.0–26.1 s), bc250_1 Linux Vulkan f32 GTrXL 134–156 s(LSTM 128–152 s), OOM·TDR 없음, 최대 메모리 CUDA 5.4 GB·bc250_1 VRAM+GTT 3.5 GB. 3절 확인 2번(GTrXL BC 미검증) 해소. 3번(기간): 단독이면 이 PC 약 1.6일(5,300 × 약 26 s), bc250_1 약 8–9일(5,300 × 약 140 s)이고, 혼합 운영 기간은 이 PC 가동 시간에 따라 달라진다.
- 2026-10-18 21:00 — run 큐 작업자(061-R13·R14·R15)·감시 재시작(060-R6) 구현과 두 PC 실기 시험 완료(`reports/m6_bc_preflight.md` 4차). 2·3절에 작업자 기반 실행 절차와 장비 비율 집계(4절 표)를 반영했다. 실행은 사용자 승인 전이다.

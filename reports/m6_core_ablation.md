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

| 항목 | 이 PC CUDA (TF32) LSTM | 이 PC CUDA GTrXL | BC250 1호기 Vulkan f32 LSTM | BC250 1호기 Vulkan f32 GTrXL |
|---|---|---|---|---|
| update 시간 | 25.0–26.1 s | **22.6–23.4 s** | 128–152 s | 134–156 s |
| 레인 그룹 forward / backward | 0.194 / 0.599 s | 0.159 / 0.537 s | 0.588 / 3.622 s | 0.565 / 3.563 s |
| └ core forward / backward | 0.032 / 0.057 s | 0.004 / 0.005 s | 0.037 / 0.068 s | 0.023 / 0.037 s |
| 최대 GPU 메모리 | 5.1 GB / 16 GB (nvidia-smi, 메모리 풀 포함) | 5.4 GB | VRAM 0.5 + GTT 3.2 = 3.5 GB / 15.9 GB | VRAM 0.5 + GTT 3.2 = 3.5 GB |
| OOM·장치 리셋 | 없음 | 없음 | 없음 | 없음 |

- **BC250 1호기 Linux에서 GTrXL 학습이 된다.** Windows Vulkan에서 TDR로 측정할 수 없었던 문제는 Linux에서 나지 않았다(update 3회와 프로파일 모두 정상 종료).
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
| 장비 | 혼합 유동 운영(2026-10-18 사용자 결정): BC250 1호기(Vulkan f32)은 상시, 이 PC(CUDA, TF32 matmul)는 쉬는 동안 run 큐 작업자로 두 run을 나눠 맡고 서로 이어받는다(061-R13·R14). 장비별 update 구간은 `runs/<id>/devices.jsonl`(061-R15). 두 run의 장비 비율을 최종 집계해 4절에 적는다 |
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
2. **GTrXL은 BC 본 설정으로 학습해 본 적이 없다.** BC 준비 검증(preflight 1–3차, 이어 하기, 최적화)은 모두 LSTM이었다. GTrXL은 벤치(B2·T16)만 돌렸다. 레인 32 × T64 × N512에서 메모리와 update 시간을 모른다. Vulkan에서는 GTrXL 학습을 측정한 적이 없다(Windows TDR). CUDA와 BC250 1호기 양쪽에서 짧은 스모크가 필요하다. → **처리됨(진행 기록 2026-10-18 20:22 참조).**
3. **장비를 섞으면 "약 2일"이 맞지 않는다.** 2일은 CUDA 단독 32.5 s/update 기준이다(이후 약 18% 단축). BC250 1호기 Vulkan은 최적화 전 169 s/update로, 단독이면 5,300 update에 약 10일이다. 최적화 후 값은 측정 중이다. 두 run의 장비별 update 구간 비율이 다르면 TF32와 f32의 수치 차이가 섞이므로, 장비별 구간을 기록한다.
   - 다시 계산(2026-10-18 21:10, 진행 기록 20:22의 실측: GTrXL CUDA 약 23 s·Vulkan 약 145 s, LSTM CUDA 약 26 s·Vulkan 약 140 s; 아래는 두 코어 평균 CUDA 24.5 s·Vulkan 142 s 사용): 두 run 합계 10,600 update. 이 PC 단독으로 차례로 돌리면 약 71시간(약 3.0일). BC250 1호기 단독이면 run당 약 209시간(약 8.7일). 두 PC를 함께 쓰면 처리량은 `a/24.5 + 1/142` update/s다. 여기서 a는 이 PC의 가용률(학습에 쓸 수 있는 시간 비율)이고, BC250 1호기은 상시 가동이며 run이 PC 사이를 옮겨도 손실이 없다고 가정한다. a=100%면 약 62시간(약 2.6일), a=50%면 약 107시간(약 4.5일), a=25%면 약 171시간(약 7.1일)이다. 한 run이 BC250 1호기에만 오래 머물면 그 run이 늦게 끝나므로, 실제 기간은 작업자 큐 배분에 따라 이보다 길어질 수 있다.
4. **맞대결 경로 미시험.** 코어별 모델 정책 테스트는 있지만(`crates/uranium-agent/tests/model_policy.rs`), 한 게임에 LSTM 모델과 GTrXL 모델이 함께 들어간 실행은 없다. `uranium-agent random-checkpoint --core lstm|gtrxl` 두 개로 2–4판 스모크를 한다. → **처리됨(진행 기록 2026-10-18 21:03 참조).**
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
cargo build --release -p uranium-train                                       # BC250 1호기
# BC250 1호기: 상시 실행 (systemd user 서비스 tools/worker/uranium-worker.service, 또는 setsid nohup)
uranium-train worker --queue <shared>/queue/core-ablation.toml --backend vulkan --priority 1 --log runs/worker.log
# 이 PC: 쉬는 동안 실행, 멈출 때 Ctrl+C (현재 run 체크포인트·push·잠금 해제)
uranium-train worker --queue <shared>/queue/core-ablation.toml --backend cuda --priority 10 --log runs/worker.log
```
- 작업자는 큐 순서대로 빈 run을 잡는다. 빈 run이 없으면 이 PC(우선순위 10)가 BC250 1호기(1)이 돌리는 run에 양보를 요청한다. BC250 1호기은 다음 update 끝에서 체크포인트·잠금 해제 후 넘기고, 이 PC를 멈추면 다시 이어 간다.
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
| 장비 비율 (update 수: 이 PC CUDA / BC250 1호기 Vulkan) | 5,300 / 0 (100% CUDA) | 4,474 / 826 (CUDA 84.4%) |

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
- 2026-10-18 19:05 — 시작 전 남은 확인 5가지를 3절에 정리. 장비 혼합 유동 운영(BC250 1호기 + 이 PC, worker 큐)은 다른 세션이 worker 구현과 함께 032-R17 ①·이 문서 2·3절에 반영하기로 함.
- 2026-10-18 20:22 — GTrXL 실제 BC 설정 확인(1.5절): 이 PC CUDA GTrXL 22.6–23.4 s/update(LSTM 25.0–26.1 s), BC250 1호기 Linux Vulkan f32 GTrXL 134–156 s(LSTM 128–152 s), OOM·TDR 없음, 최대 메모리 CUDA 5.4 GB·BC250 1호기 VRAM+GTT 3.5 GB. 3절 확인 2번(GTrXL BC 미검증) 해소. 3번(기간): 단독이면 이 PC 약 1.6일(5,300 × 약 26 s), BC250 1호기 약 8–9일(5,300 × 약 140 s)이고, 혼합 운영 기간은 이 PC 가동 시간에 따라 달라진다.
- 2026-10-18 21:00 — run 큐 작업자(061-R13·R14·R15)·감시 재시작(060-R6) 구현과 두 PC 실기 시험 완료(`reports/m6_bc_preflight.md` 4차). 2·3절에 작업자 기반 실행 절차와 장비 비율 집계(4절 표)를 반영했다. 실행은 사용자 승인 전이다.
- 2026-10-18 21:03 — 맞대결 경로 스모크 통과(3절 4번). 3 step(LSTM)·2 step(GTrXL) 시험 BC 체크포인트로 맞대결 plan의 3판 축소본을 이 PC 로컬 hostd(동시 3판)에서 돌림. 추론 서버가 두 코어를 동시에 띄움. 3판 모두 끝까지 진행해 60분 시간 제한 무승부(TimeLimitDraw, 두 세션 승자 일치). winner disagree·dropped·세션 오류·시야 위반·executor 거부 모두 0. 결정 지연 p50/p95: LSTM 57.4/179.3ms, GTrXL 62.4/194.2ms(평균 배치 약 2). 시간 제한까지 가는 판은 동시 3판에서 판당 실제 약 16분.
- 2026-10-18 21:03 — 스모크 준비 중 GTrXL CUDA BC가 OOM으로 1회 실패(데이터 업로드 시점). 다른 GPU 작업과 겹친 때로 보이며, 바로 재시도하니 정상이었다(샘플링 최대 약 7.5GB 관측). 이 PC가 쉬는 동안만 학습하는 운영이면 다른 GPU 작업과 겹쳐 재발할 수 있고, 그때는 watchdog 재시작(060-R6)에 의존한다.
- 2026-10-18 21:03 — 사용자 결정: ① `uranium-env` summary.md 개선 — p0·p1 모두 정책 표시, 정책 이름에 core와 run id, 슬롯 교대 matchup을 합친 정책별 승·패·무와 Wilson 95% 합산 표(spec 020 Risks 절 메모). 모델 대 모델 게임의 승리를 이전에는 "mirror decided"로만 세던 것을 승자 정책으로 가르게 됨. ② 대전 700판(맞대결 100 + T0 600) 리플레이를 모두 저장(`core-ablation-h2h.toml`·`core-ablation-t0.toml`에 `save_replay = true`, 032-R17). 저장 위치는 `<out>/replays/g<id>.rep`(3,000프레임 제한 3판 시험으로 확인, 새 summary 표도 같은 시험에서 확인).
- 2026-10-18 21:10 — 3절 3번 기간을 실측 update 시간으로 다시 계산(이 PC 가용률 a 가정, 3절 참조).

### 2026-10-20 선행 조건 점검 (master 925b995, 확인만 하고 결정은 하지 않음)
3절 "시작 전 확인"과 메모의 선행 조건을 현재 master에서 다시 확인했다.

1. **030-R4 증강 변경(transpose 제외) — master에 있음.** 2d8dc4e(증강 4종·`start_loc` 변환)와 7ee6442(flip `entity_cell` 재계산, B-6)가 925b995의 조상이다. 학습 로더는 `Sym::augment()`(원본·좌우·상하·둘 다)만 쓰고(`uranium-train/src/bc.rs`), transpose를 포함한 8종 `Sym::all()`은 030-AC4 왕복 테스트에만 쓴다. spec 030-R4 본문도 같다.
2. **041-R6 클래스 가중치 — 결정됨.** 적용하지 않는다(사용자 결정 2026-10-18, spec 041-R6 본문, 2d8dc4e). 재검토 조건은 BC 평가가 너무 낮을 때다. 레지스트리에 열린 항목은 없다.
3. **`config/bc_core_ablation.toml` ↔ `config/bc.toml` — 의도한 차이만 있음.** 다른 값은 `[steps] s1 = 5300`, `warmup_steps = 265`, `keep_checkpoints = 0` 세 개와 주석뿐이다. 모델(lanes 32, seq_len 64, micro_lanes, pad_entities 64, lr 3e-4, min_lr, weight_decay, grad_clip 1.0), 헤드 가중치, value, data(z_drop, augment, enemy_race_mask 0.05, seed 1, checkpoint 주기)는 같다. 방금 갱신한 `s2`·`ablation_a` 11319, `s2_warmup`·`ablation_a_warmup` 566도 두 파일이 같다(S1만 쓰는 비교에는 쓰이지 않음). 테스트 `warmup_cosine_schedule_041_r3`가 세 값을 맞춘 뒤 두 설정 전체가 같은지 검사한다.
4. **plan 파일 — 있음, 맵·상대 유효.** `config/plans/core-ablation-h2h.toml`(맵 2 × 25판 × 슬롯 교대 = 100판)과 `core-ablation-t0.toml`(코어 2 × 종족 3 × 맵 2 × 50판 = 600판). 맵 이름 `fighting_spirit`·`circuit_breaker`는 `config/assets.toml`에 있고, 상대 `t0` terran·protoss·zerg는 `bc-vs-t0.toml`과 같은 형식이다. 둘 다 `save_replay = true`, `zero_last_failed = true`. 체크포인트 `checkpoints/core-abl-*/ckpt-5300`은 학습 뒤에 생긴다. 저장소 plan 파일 전체를 읽어 보는 자동 테스트는 없다.
   - **맞대결 경로 시험 이력**: 2026-10-18 21:03 3판 축소본 스모크가 마지막이다(위 진행 기록). 그 뒤 모델 세션 실게임은 R-2(2026-10-19, proto v2, 무작위 LSTM 대 T0 1판, `m6_rebuild.md` 5절)뿐이고, 모델끼리 대전은 다시 돌린 기록이 없다. 그 사이 proto v3(`Unload_Unit`), 액션 101개, feature 스키마 6, 브리지 그룹 명령(011-R20)이 들어왔다. 옛 스모크 체크포인트는 스키마가 달라 읽히지 않는다(`m6_action_audit.md` 배포·후속). 그래서 지금 바이너리로는 맞대결·T0 경로 모두 미시험이다. 무작위 체크포인트 두 개(`uranium-agent random-checkpoint --core lstm|gtrxl`)로 짧은 스모크를 다시 하는 것이 안전하다(제안).
5. **코어 비교가 기대는 열린 `[SPEC AMBIGUOUS]`·DEFERRED 항목 — 없음.** `spec/README.md` 레지스트리에서 Resolved가 아닌 것은 AMB-001-1·AMB-012-1·AMB-012-2(M7, SC:R)와 AMB-042-2(M6, RL 상대 다양성)뿐이다. 모두 BC 학습·코어 비교와 무관하다. 032-R17·AMB-032-6·AMB-032-7, 041 AMB-041-1~6은 모두 Resolved다. 태그는 아니지만 032-AC2 Linux Vulkan 기준값은 측정 뒤 사용자 결정이 남아 있다. 대전은 lockstep이라 이 값이 비교 결과를 바꾸지는 않는다.

함께 본 것(결정 아님)
- 3절 3번 기간 계산은 autotune·fusion 이전 값(CUDA 약 24.5 s, Vulkan 약 142 s)이다. R-5 스모크는 LSTM 기준 CUDA 약 10 s, Linux Vulkan 약 101 s/update였다(`m6_rebuild.md` R-5 8절). GTrXL은 새 설정으로 다시 재지 않았다.
- 학습 장비는 최신 master(cd2973a·925b995 shard 순서·지문)로 다시 빌드해야 한다. 게임 호스트는 브리지·proto v3를 반영해 재빌드하고 hostd를 재시작해야 하는데, 그 기록을 찾지 못했다.
- `m6_pre_bc_review.md` G절에서 남은 것: G-M2(Linux Vulkan 호스트 메모리·스왑 측정), G-M5(Windows 실제 콘솔 Ctrl+C), C-8의 060-AC4·032-AC2(측정 중). G-M4(060-AC3)는 코어 비교 run의 첫 24시간으로 대신한다.

### 2026-10-20 15시 대전 경로 스모크 (지금 바이너리, 무작위 체크포인트)
위 4번의 제안을 실행했다. 대상은 PROTOCOL 3·액션 101개·feature 스키마 6·브리지 그룹 명령이 들어간 지금 바이너리다. 에이전트·runner는 master 87900c3에서 CUDA로 다시 빌드했다(f513b92 이후 이 두 crate와 proto·env·model은 바뀌지 않음).

준비
- 무작위 체크포인트 두 개: 기존 `<shared>/checkpoints/smoke/random-lstm-fv6`, 새로 만든 `<shared>/checkpoints/smoke/random-gtrxl-fv6`(`uranium-agent random-checkpoint --core gtrxl`, feature 6·action 2, `random_init`).
- 두 plan의 임시 축소본(저장소 밖, 실행 뒤 삭제). 축소한 것은 체크포인트·판수·`max_frames = 6000`(약 4.2분)뿐이고, `seed_base`·확률적 샘플링·`save_replay`·`zero_last_failed`는 원래 plan과 같다.
  - 맞대결: 투혼 1판(p0 LSTM, p1 GTrXL) + 서킷브레이커 1판(p0 GTrXL, p1 LSTM, 슬롯 교대) = 2판.
  - T0: 투혼에서 LSTM 대 T0 테란, GTrXL 대 T0 프로토스, GTrXL 대 T0 저그 = 3판(GTrXL 대 T0 경로도 함께 보려고 코어를 섞음).
- 게임 호스트는 Linux 호스트 A·B만 쓴 임시 호스트 목록, 에이전트는 학습 PC `--backend cuda`.

결과: **두 경로 모두 통과.**
| 항목 | 맞대결 2판 | T0 3판 |
|---|---|---|
| 종료 | 모두 GameEnd(6,001프레임 시간 제한 무승부) | 같음 |
| 세션 오류·빠진 쪽·winner disagree·dropped | 0 | 0 |
| 시야 위반(010-AC6) | 0 | 0 |
| 모델 executor 거부(`rejected`) | 없음 | 없음 |
| 모델 명령 수(판·쪽별) | LSTM 446·450, GTrXL 424·414 | LSTM 450, GTrXL 406·445(T0 쪽 61·70·38) |
| 실패 명령 비율 | 3.3% | 2.2% |
| 결정 지연 p50/p95 | LSTM 77/139ms, GTrXL 86/157ms | LSTM 63/97ms, GTrXL 92/180ms |
| 리플레이 | `replays/g1.rep`·`g2.rep` | `replays/g1~g3.rep` |

- 플레이어 슬롯 라우팅(020-R11): `results.jsonl`의 `slot`은 BWAPI 플레이어 id이고 plan의 p0/p1 순서와 다를 수 있다. 서킷브레이커 판(p0 GTrXL)에서 BWAPI id 0 세션이 LSTM을 받았다. 라우팅이 플레이어 이름(`uranium-p<i>`)의 plan 순서로 정해졌다는 뜻이다. id로 정해졌다면 GTrXL을 받았어야 한다. 두 판 모두 두 세션이 서로 다른 체크포인트를 받았고, 기본 정책(T0)으로 빠진 세션은 없다. T0 판에서는 상대 종족이 plan대로(테란·프로토스·저그) 나왔다. summary.md의 matchup 표 p0/p1도 plan과 일치한다.
- 출력 형식: `summary.md`에 matchup 표와 "Per policy (slot swaps combined)" 표(정책·상대·맵별 승/패/무와 Wilson 95%, 맵 합계 all)가 나온다. "Model sessions" 표에는 core·run·결정 지연이 있다. 4절 대전 표(맞대결 LSTM 승률 맵별, 코어별 T0 승률)는 이 합산 표에서 옮길 수 있다. 무승부만 나온 스모크라 승패 집계 자체는 확인하지 못했다.
- 참고: 추론 서버 첫 배치가 60–80초 걸렸다(`[infer] slow batch`, 시작 직후 1회, CUDA 초기 준비로 보임). 게임은 lockstep이라 결과에 영향은 없고, 본 대전에서는 첫 판 시간만 늘어난다.
- 출력 위치(지우지 않음): 학습 PC 저장소 루트 `runs/eval-smoke-2026-10-20/h2h/`, `runs/eval-smoke-2026-10-20/t0/`(`games.jsonl`, `results.jsonl`, `summary.md`, `sessions.md`, `plan.toml`, `replays/`).

### 2026-10-20 15:21 학습 시작 (사용자 지시)
사용자 지시: 학습은 BC250 1호기(Linux Vulkan, 상시)와 학습 PC(CUDA, 쉬는 동안만, 사용자가 직접 켜고 끔)가 맡는다. 게임 호스트는 네 장비 모두(학습 PC, BC250 1·2호기, asus) 평가 때 쓴다. 네 장비 모두 최신 빌드. BC250 1호기 작업자는 tmux 세션 `train`.

준비
- 빌드: 네 장비 master 2639038(OpenBW·BWAPI, 브리지, release 바이너리). 학습 PC는 `--features cuda`. asus는 OpenBW·BWAPI 소스부터 새로 받아 빌드하고 `.env`에 hostd 항목을 넣었다(옛 고정 build 경로 → 저장소 경로, 원본 `.env.bak-20261020`).
- 큐 `<shared>/queue/core-ablation.toml`: 3절과 같음(`core-abl-lstm`, `core-abl-gtrxl`, `config/bc_core_ablation.toml`, 네 출처).
- 게임 호스트: hostd는 학습에 쓰지 않으므로(BC는 리플레이 shard만 읽음) 사용자 지시로 모두 꺼 두었다. asus는 calibrate만 했다. 결과는 동시 3게임(브리지 p99 중앙값 1/2/3/4게임 78/87/208/420 µs, 4게임에서 기준 300 µs 초과, `~/.cache/uranium/hostd_calibration.toml`). 학습 PC의 미추적 `config/hosts.toml`에 asus를 추가해 평가 때 네 대가 들어가게 했다. 다른 장비의 동시 게임 수는 기존 측정값(학습 PC 24, BC250 1호기 5, 2호기 4)이다.

run 배정 (작업자 큐 순서대로, 061-R13)
- BC250 1호기 작업자(15:21:56)가 먼저 `core-abl-lstm`을 집었다. 그러나 `bc`는 잠금을 데이터 pull(shard 1,913개 로컬 복사, 약 18분)과 맵 검사 뒤에 잡는다. 그사이 학습 PC 작업자(15:24:23)가 같은 run의 잠금을 먼저 잡았다. 학습 PC는 공용 드라이브가 로컬 디스크라 복사 없이 검사만 한다.
- BC250 1호기 `bc`는 맵 검사 실패(아래)로 2회 끝났다. 3번째 시도(15:43:53)는 "locked by <학습 PC>"로 끝나 종료 코드 75를 냈다. 작업자는 이것을 재시작으로 세지 않고 다음 run `core-abl-gtrxl`을 잡았다(15:44:47).
- 결과 배정: **LSTM은 학습 PC CUDA, GTrXL은 BC250 1호기 Vulkan f32**에서 시작했다. 학습 PC가 쉬는 시간에 따라 두 run의 장비 비율이 크게 다를 수 있다(3절 공정성 주의). 4절 표에 run별 장비 비율을 적는다(`uranium-train devices --run <id>`).

발견과 조치
1. **맵 사전계산 누락 (BC250 1호기).** `bc` 시작 검사(B-2)가 "1212 map hash(es) ... have no map precompute v2 in ~/sc_data/maps"로 끝났다(15:39·15:41, 종료 코드 1, 감시 재시작 60·120초). R-5 재build로 만든 build 맵 사전계산은 12:40에 학습 PC에만 복사했고, Linux 학습 호스트에는 복사하지 않았다. `<shared>/datasets/replays/build/maps/*.toml` 1,244개를 `cp -n`으로 복사했다(v2 확인, 폴더 합계 1,248). 다음 재시작부터 `map_precomputes: ok (B-2)`가 나왔고, 학습 PC와 같은 `run_ident`(config `b19dac76b7908acb`, dataset `2044:14919882ffd994af`)가 찍혔다. README 3절에 학습 호스트 준비 단계로 추가했다.
2. **GTrXL Vulkan autotune 후보 panic.** 첫 update 중 `thread 'DSD-4-8192' panicked at cubecl-runtime-0.11.0/src/tune/local.rs:224: Should run when selected by autotune.: matmul_naive ... Cube count too big (9, 131072, 1)`가 났다. autotune이 행렬곱 후보를 시험하다 `matmul_naive`가 GPU 디스패치 한도를 넘은 것이다. 프로세스는 계속 돌았고 update도 정상으로 이어졌다(step 1 161.7 s는 autotune 포함, step 2 78.1 s, loss 유한). 이 사실을 확인하기 전에, 학습이 멈춘 것으로 잘못 보고 작업자를 SIGINT로 멈췄다(15:50:13). 그때 `ckpt-2`가 저장·push되고 잠금이 풀렸다. 작업자를 tmux에서 다시 띄우자 15:50:36에 `ckpt-2`부터 재개했다(같은 호스트·같은 autotune 캐시). 재개 시작 때 같은 panic이 1회 더 났고, step 3·4는 111.7·91.4 s였다. 탈락한 후보는 캐시에 남지 않아 프로세스가 시작할 때마다 1회 날 것으로 보인다. 10-18 GTrXL Vulkan 확인(1.5절)은 autotune 도입 전이었고, R-5 Vulkan 스모크는 LSTM만 했다. 그래서 이번이 GTrXL·Vulkan·autotune 조합의 첫 실행이다. 출력에 영향이 있는지는 확인하지 않았다. 다른 후보로 계산하므로 영향이 없을 것으로 본다.
3. **BC250 1호기 메모리.** GTrXL 학습 중 사용 14.9 GB / 15.2 GB, 스왑 11.9 GB / 16 GB(15:50). R-2 LSTM 스모크 최대 13.3 GB에 가깝다. 남은 확인 항목인 Linux 메모리·스왑 기록(README 6절)으로 계속 본다.

속도(시작 직후)
| run | 장비 | update 시간 |
|---|---|---|
| `core-abl-lstm` | 학습 PC CUDA (TF32) | 약 10.2~14.8 s (step 93까지) |
| `core-abl-gtrxl` | BC250 1호기 Vulkan f32 | 78.1~111.7 s (step 2~4) |

출력 위치(지우지 않음): 체크포인트 `<shared>/checkpoints/core-abl-{lstm,gtrxl}/`, 장비 구간 `<shared>/runs/core-abl-{lstm,gtrxl}/devices.jsonl`, 작업자·학습 로그는 각 장비 저장소 `runs/worker.log`(학습 PC, BC250 1호기). 빌드 로그는 Linux 장비 `~/build-logs/build_20261020.log`, asus calibrate 로그는 asus 저장소 `runs/calibrate_20261020.log`.

### 2026-10-20 16:43 BC250 1호기 OOM과 수정 (041-R16, G-M2)
- **16:43:19 커널 OOM 종료.** GTrXL run이 step 31 뒤에 끝났다. tmux 세션 전체가 OOM으로 종료됐다(systemd: `tmux-spawn-*.scope: Failed with result 'oom-kill'`, 최대 RAM 13.2 GB·스왑 13.6 GB). 1시간 체크포인트 전이라 step 3~31을 잃었고, 마지막 체크포인트는 `ckpt-2`다. 15:50에 스왑 상승(11.9 GB)을 기록했지만 대응하지 않았다. 이 항목은 BC 전 리뷰의 미해결 측정 G-M2(B-5, prefetch 메모리)가 실제로 터진 것이다.
- **사용자 결정(16:5x)**: 원인 측정부터 한다. CUDA 호스트의 LSTM run은 그대로 둔다.
- **측정 1(지금 설정, 17:09~17:23)**: `ckpt-2`부터 로컬 폴더(`--out`, run·잠금 없음)로 같은 데이터를 학습하며 15초마다 기록했다(`~/memdiag.sh`). 학습 프로세스 힙(anon)은 RAM 6.3~8.3 GB와 스왑 8.4~9.6 GB를 합쳐 약 16 GB였다. GTT는 4.5~6.5 GB였다. 17:23:04에 15초 만에 프로세스 스왑이 9.4 GB에서 11.6 GB로 뛰어 남은 스왑이 2.3 GB가 됐고, 측정 스크립트의 안전 정지(남은 스왑 < 2.5 GB면 SIGINT)가 step 9에서 멈췄다. 이때 4줄마다 하나만 출력해 STOP 줄을 놓쳤고, 사용자에게 "외부 중단 신호"라고 잘못 보고했다가 정정했다.
- **원인**: batch 하나(레인 32 × T64 = 2,048 시점, 공간 feature 26채널 f32)는 약 4 GB다. 미리 읽기 깊이 2에서는 학습 중 1개, 채널 대기 2개, 만드는 중 1개로 최대 4개가 동시에 메모리에 있다(`bc_run.rs` `Prefetch::start(…, 2)`). 이 호스트는 RAM 15 GB를 GPU(GTT)와 나눠 쓰므로 늘 스왑 근처였고, 오르내림이 한도를 넘으면 OOM이 났다.
- **수정(c971b1f, 041-R16)**: 깊이를 `.env` `URANIUM_BC_PREFETCH`로 정한다(기본 2). 0이면 데이터 스레드 없이 update마다 그 자리에서 batch를 만든다. 깊이 0·1·2가 같은 batch·같은 순서를 주는 것을 테스트로 확인했다(`prefetch_gives_the_feeder_batches_041_r16`). 설정 지문은 바뀌지 않는다. BC250 1호기 `.env`에 `URANIUM_BC_PREFETCH=0`을 넣고 `uranium-train`만 다시 빌드했다. 학습 PC는 기본값 2이고 다시 빌드하지 않았다(작업자 실행 중이고, 출력은 같다).
- **측정 2(깊이 0, 18:20~19:34, step 3→45 완료)**: 힙은 RAM 1.8~5.9 GB에 스왑 0~5.7 GB였다. GTT는 처음 6.5 GB에서 step 16에 8.6 GB, **step 32에 11.3 GB**로 커진 뒤 줄지 않았다. 남은 스왑 최소는 8.05 GB였다(측정 1은 2.3 GB). 이전에 OOM이 난 step 31~32를 통과했다. update는 82~116 s(배치 생성 제외), 실제 경과는 약 99 s/update로 측정 1과 비슷하다.
- **남은 위험**: GTT는 큰 batch(엔티티 많음)를 만나면 커지고 줄지 않는다. Burn 메모리 풀로 보이며, 최악 N=512에서 얼마까지 커질지는 모른다. GTT는 스왑되지 않아 힙이 대신 스왑으로 밀리고, 그만큼 느려진다. 남은 스왑 여유(약 8 GB)로 운영하면서 계속 본다.
- **재개**: 19:36 tmux `train` 작업자 재시작. `ckpt-2`에서 재개했고, `{"prefetch":0}`, 잠금은 BC250 1호기가 다시 잡았다.
- 측정 출력(지우지 않음): BC250 1호기 `~/diag/g-m2/base/`, `~/diag/g-m2/p0/`(`mem.csv`, `train.log`, 로컬 체크포인트 `out/`).

### 2026-10-20 20:52 BC250 1호기 GPU 메모리 할당 실패와 수정 (041-R17, R18)
- **20:52·21:13 종료(코드 101).** 깊이 0으로 재개한 GTrXL run이 step 47 다음 update에서 `can't allocate buffer of size: 2147483648`으로 panic했다(`bc.rs` loss 읽기). 이번에는 커널 OOM이 아니라 GPU 버퍼 할당 실패다. 감시가 `ckpt-38`부터 재시작했지만 같은 batch(step 48)에서 다시 실패했고, 작업자를 멈췄다(12:14Z). step 39~47 두 번을 잃었다.
- **효과 없던 시도**:
  - update마다 메모리 풀 비우기(Burn `memory_cleanup`): update 사이 GTT는 11.3 → 1.9 GB로 줄었지만 step 48에서 같은 실패.
  - Burn gradient checkpointing(`grad_checkpointing = true`, 실행 전용 설정): 같은 실패. Burn 방식은 행렬곱 결과를 다시 계산하지 않고 들고 있어 attention 메모리가 거의 줄지 않는다.
- **빠른 재현**: `ckpt-38`부터 `--steps 47`로 로컬 체크포인트를 한 번 만들고, 거기서 update 1회(step 48)만 돌려 약 2분에 판정했다.
- **원인**:
  - 레인 1개(64행)의 forward 직후 GPU 사용량이 엔티티 수 n에 따라 늘었다. n=192는 3.3 GB, 256은 4.5 GB, 320은 5.8 GB, 384는 7.2 GB였고, n=512는 약 10.5 GB로 추정된다. step 48 batch의 13번째 레인이 처음으로 n=512(`N_MAX`)였다.
  - 2 GiB 할당은 거리 bias one-hot `[64·512·512, 32]` f32(`entity.rs` `distance_bias`, 032-R13)와 크기가 정확히 같다. 엔티티 인코더의 n² 텐서(one-hot, bias, attention)가 대부분을 차지한다.
  - GTT 상한은 15 GiB(`ttm pages_limit`)로 설정돼 있지만, GTT는 스왑할 수 없는 실제 RAM(전체 14.85 GiB)이다. 실패 순간에는 GTT 9.85 GB, 힙 RSS 4.4 GB, 남은 RAM 0.46 GB였고, 스왑은 11.5 GB가 비어 있었다. 스왑이 부족했던 게 아니라 고정할 RAM이 없었다.
- **수정(사용자 결정)**:
  - 041-R17 엔티티 인코더 재계산(`URANIUM_BC_ENTITY_CHUNK`, 행 수): 1단계에서 인코더를 조각별로 gradient 기록 없이(`Module::no_grad`, dropout 유지) 계산한다. 코어·헤드 backward 뒤에 같은 조각을 같은 seed로 다시 계산해 gradient를 되돌린다.
    - 처음에는 detach만 했는데, Burn이 backward 전까지 1단계 그래프를 놓지 않아 메모리가 줄지 않았다(n=384 forward 7.38 GB).
  - 041-R18 호스트 batch 압축(`URANIUM_BC_PACK_BATCH`): RL `PackedObs`와 같은 zstd 무손실 압축(`pack.rs`)을 쓰고, 레인 그룹 학습 때 그 행만 푼다.
- **검증(BC250 1호기, chunk 16 + 압축)**:
  - step 48 단독 실행: n=512 레인 forward 1.83 GB(이전 약 10.5 GB), 풀 최대 4.5 GB, GTT 최대 4.3 GB로 통과했다. 1단계 출력과 재계산 출력의 최대 차이는 1.2~1.9×10⁻⁶(32개 레인 그룹 모두)이라 dropout 마스크가 같다.
  - `ckpt-38`부터 연속 실행(사용자 지시로 step 40까지): update는 137.9 s(첫 update, autotune 포함), 108.4 s(이전 92~109 s)였다. GTT 최대 4.15 GB, 남은 RAM 최소 3.5 GB, 스왑 0.
  - 테스트: 재계산·압축 켬/끔의 update 동일성(tiny, dropout 0, 조각 균등·불균등, `micro_lanes` 0·1), dropout을 켠 1단계·재계산 출력 동일성, `.env` 파싱.
- **남은 관찰**:
  - 압축을 켜도 힙 RSS가 5.7~6.2 GB다. batch 외의 힙 사용처는 따로 보지 않았다. GTT가 작아져 지금은 여유가 있다.
  - 거리 bias를 one-hot 행렬곱 대신 gather로 계산하면(032-R13 변경) 조각당 메모리가 더 준다. 후보로만 남긴다.
- **운영**: BC250 1호기 `.env`에 `URANIUM_BC_ENTITY_CHUNK=16`, `URANIUM_BC_PACK_BATCH=1`을 넣었다. 학습 PC는 기본값(끔)이다. 설정 지문은 바뀌지 않으므로 `ckpt-38`부터 이어 간다.
- 측정 출력: BC250 1호기 `~/diag/r17/`의 `mem.csv`·`train.log`(dbg48: 레인별 사용량, rc48b: 재계산 차이, rc60: 연속 실행). 로컬 체크포인트와 임시 스크립트는 사용자 지시로 지웠다.

### 2026-10-21 LSTM ckpt-5300 플레이 성향 확인 (T1 대전, 첫 결정 진단, 사용자 요청)
- **배경**: 학습이 끝난 `core-abl-lstm` ckpt-5300을 T1 봇(Iron T·Steamhammer Z·Stardust P)과 각 3판 돌렸다(확률적 샘플링, 학습 PC 로컬 hostd). 9판 모두 이기지 못했고(전멸패 7, 60분 시간 제한 무승부 2), 사용자는 리플레이에서 "시작하자마자 커맨드 띄우기, 커맨드 여러 개 짓기"를 봤다. 무승부 2판은 떠 있는 커맨드를 Iron이 공격하지 않아 시간 제한까지 갔다. 원인 후보로 ① 학습량(5,300 update = S1 한 epoch 38,584의 약 14%, 본 게임 4,502판) ② 확률적 샘플링 ③ 학습·실게임 입력 차이를 세웠다. 예상: ①·② 위주, ③이면 첫 결정 확률부터 달라야 한다.
- **argmax 9판**(같은 상대·seed): 8패, 1판은 결과 미판정(Elimination이지만 won 없음), 4~7분. 분당 명령 수는 확률적 샘플링의 약 3배.
- **진단 방법**: 실게임 에이전트가 결정마다 받은 프레임과 고른 행동을 남기는 디버그 기록(`URANIUM_AGENT_TRACE`)을 추가했다. 이 기록과 BC 리플레이(val split, z 없음, 증강 없음) 200판의 첫 5분을 같은 `extract`·`forward_seq`로 다시 계산해 action_type 확률을 비교했다(예제 `first_decisions`). 실게임은 투혼에서 확률적 샘플링 3판·argmax 3판(종족별 1판), 첫 5분까지 돌렸다. 재구성 확인: 실게임이 기록한 log p와 다시 계산한 값의 차이는 최대 8×10⁻⁴다.
- **결과 1 — 게임 시작 입력은 학습과 같다**: 첫 결정의 확률이 실게임과 리플레이에서 거의 같다(Lift 1.06% / 1.00%, Train_SCV 11.2% / 9.1%, NoOp 35% / 37%, SetRally_Pos 18.9% / 19.4%). 허용 행동 마스크도 같다. ③은 시작 시점에서는 아니다.
  - 부수 발견: 첫 결정의 미네랄 입력이 실게임은 50인데, 대상 맵 리플레이 91판 평균으로 보면 약 18%가 0이다. 첫 결정 확률에는 거의 영향이 없었다. 원인은 조사하지 않았다.
- **결과 2 — 확률적 샘플링의 행동 종류 비율은 사람(유효 레이블)과 비슷하다**: 2~5분 구간 Lift 0.28%(사람 0.19%), Build_CommandCenter 0.09%(0.18%), Train_SCV 2.5%(3.0%), NoOp 23%(20%), SetRally_Pos 21%(11%), Smart_Pos 28%(41%). 리플레이에서 모델 평균 확률과 사람 비율도 맞는다(Train_SCV 0.040 / 0.040). 커맨드 띄우기·짓기는 사람보다 자주 하는 게 아니라, 사람과 비슷한 빈도로 **엉뚱한 때·대상에** 한다. 행동 종류의 평균 비율은 배웠고 상황에 따른 판단은 아직 못 배운 상태다(①과 맞음).
  - 사람 레이블 중 Train_SCV는 무효 레이블을 넣으면 9~10%, 빼면 3~4%다. 생산 큐가 찬 뒤 반복 입력이 대부분 무효로 빠진다.
- **결과 3 — argmax는 랠리 지정 고리에 갇힌다**: 결정의 99.9%가 SetRally_Pos다. SetRally_Pos 뒤 SetRally_Pos 확률이 0.90이라 한 번 고르면 계속 고른다. 사람 데이터에서도 SetRally_Pos 뒤 SetRally_Pos는 64%(반복 클릭, 040-R4 반복 병합 없음)이고, 모델은 이것을 그대로 배웠다(리플레이 0.63). argmax 평가는 이 체크포인트에 쓸 수 없다. 평가 plan은 이미 확률적 샘플링이다.
- **판단(제안, 미확정)**: 지금 플레이는 주로 학습량 부족(①)과 BC의 한계로 보인다. 입력 버그(③)는 시작 시점에서 배제했다. 반복 클릭 학습은 argmax와 APM에 영향을 준다. 다만 "반복 병합 없음"은 사용자 결정(040-R4)이라 바꾸지 않았다.
- 출력(지우지 않음): 학습 PC 저장소 `runs/lstm-vs-t1-2026-10-21/`(확률적 9판), `runs/lstm-vs-t1-argmax-2026-10-21/`(argmax 9판), `runs/lstm-trace-2026-10-21/`(기록 6판 `trace/*.bin`, 리플레이, `first_decisions.md`(무효 레이블 포함), `first_decisions_valid.md`(유효 레이블만)), plan `runs/adhoc-plans/`. 진단 코드는 로컬 브랜치 `diag-first-decisions`(02acb9a, worktree `../Uranium-diag`, master 미반영).

### 2026-10-22~23 비교 단계 (재빌드, 대전 시도 2회 실패, 최종 실행 준비)
- **16:57 시작(사용자 지시 "비교 단계 시작").** GTrXL 학습은 07:53Z에 `ckpt-5300`으로 끝나 있었다(학습 PC 작업자). 장비 비율은 4절 표에 적었다(`uranium-train devices`).
- **17:00 재빌드(사용자 지시 "최신 빌드로 다시 시작").** 네 장비(학습 PC, Linux 호스트 A·B·C)를 master 6fa8fbd로 다시 빌드했다: OpenBW·BWAPI(패치 0014 새로 적용), 브리지, release(학습 PC는 `--features cuda`). 17:01~17:14에 끝났다. Linux 호스트 C의 첫 시도는 비대화형 ssh에서 `cargo`가 PATH에 없어 브리지 단계에서 실패했고, `~/.cargo/env`를 읽게 해 다시 빌드했다. 빌드 로그는 학습 PC 저장소 `runs/build_20261022.log`, Linux `~/build-logs/build_20261022.log`.
- **학습 PC 사용 중단(사용자 지시).** 학습 PC에서 돌던 val 500게임 채점(LSTM, 17:04~17:47)을 끝나기 전에 멈췄다. 대전은 Linux 세 대로만 하기로 했다: 스케줄러·추론은 Linux 호스트 A(Vulkan), 게임은 A·B·C(동시 5·4·3판).
- **1차 시도(17:40) 실패 — 에이전트 주소.** Linux 호스트 A에는 `URANIUM_AGENT_LISTEN`/`PUBLIC_ADDR`가 없어 에이전트가 `127.0.0.1:7600`으로 열렸다. B·C의 브리지는 자기 장비의 127.0.0.1에 접속하려다 30초 재시도 끝에 시작을 포기했고(`launcher.log`: `connect 127.0.0.1:7600: Connection refused`), 게임은 0초에 끝났다. 스케줄러는 이 판들을 실패가 아니라 `Finished (30s)`로 기록했고 `results.jsonl`은 비었다(발견: 브리지 시작 실패를 실패 판으로 세지 않음, 020 수정 제안 예정). 실행 스크립트에서 두 값을 LAN 주소로 지정해 다시 시작했다.
- **2차 시도(17:42~21:21) 중단 — 실제 시간 상한.** plan의 `wall_timeout_s` 기본값 1,800초에 맞대결 85판 중 69판이 `Timeout`으로 끊겼고, 정상 종료는 16판이었다. 추론이 Linux 호스트 A Vulkan 하나에 몰려 게임마다 약 24프레임/초(학습 PC CUDA 동시 9판은 약 100프레임/초)로 진행됐기 때문이다. 게임 시간 약 30분이 넘는 판은 모두 잘렸다. 게임 호스트 세 대의 CPU는 거의 놀았다(부하 0.1 이하). 정상 종료 16판은 짧은 판만 남은 것이라 결과로 쓰지 않는다.
- **결정(2026-10-23, 사용자)**: 학습 PC를 포함해 학습 PC에서 끝낸다. 두 plan에 `wall_timeout_s = 7200`(안전장치, 결과 무관)을 넣었다.
- **빌드 고정(Claude 판단)**: 비교 대전·채점은 6fa8fbd 빌드로 한다. 그 뒤 master(890bf3f·e93e3a0)가 OpenBW 관측에서 상대 생산 큐와 보이지 않는 칸의 탄환을 숨긴다(AMB-030-4·5). 두 모델은 변경 전 관측으로 학습했으므로 학습 때와 같은 입력으로 비교한다. 두 코어 조건은 같다. 학습 PC는 실행 파일·Windows 브리지·BWAPI 빌드를 `%LOCALAPPDATA%\uranium-build\6fa8fbd\`에 고정 복사했다.
- **최종 실행 준비**: `<shared>/artifacts/m6_core_ablation/final/run_final.ps1`(맞대결 100 → T0 600 → val 500 채점, 끝난 단계는 건너뜀, 끝나지 않은 단계 폴더는 보존 후 다시 시작). 고정 plan·호스트 목록·BC 설정은 같은 폴더 `plans/`에 있다. 출력은 모두 공용 드라이브(학습 PC 로컬 디스크)에 바로 쓴다. 학습 PC 재부팅 뒤 시작한다.
- 출력 위치(지우지 않음): 1차 `<shared>/artifacts/m6_core_ablation/core-ablation-eval-20261022-abort1/`, 2차 `.../core-ablation-eval-20261022-abort2-timeout/`(맞대결 16판 리플레이·결과, 스케줄러·동기화 로그). Linux 호스트 A 원본은 `runs/core-ablation-eval-20261022{,-abort1}/`. 이전 대전 리플레이 사본 `.../earlier/`는 만들지 않았다(학습 PC 저장소 `runs/`에만 있음).

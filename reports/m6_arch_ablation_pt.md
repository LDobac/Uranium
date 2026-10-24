# M6 — 구조 변형 비교 (PyTorch, GTrXL 기준 5개 변형, 032-R20)

- 상태: **구현 중** (2026-10-25 결정·spec 확정)
- spec: [032](../spec/032-model-architecture.md) §2.7, 032-R20, 032-AC11·AC12, AMB-032-11
- 기준 run: 2차 코어 비교 [`m6_core_ablation_pt.md`](m6_core_ablation_pt.md)의 `core-abl-gtrxl-pt` `ckpt-10000`

코어 비교 리포트와 같은 규칙으로 쓴다. 배경·예상은 실험 전에 적고 고치지 않는다. 진행 기록은 아래에 시각과 함께 덧붙이기만 한다.

## 1. 배경 (실험 전 정리, 2026-10-25)

### 1.1 왜 하나
- 2026-10-25 모델 구조 검토에서 기본 구성의 약점 세 가지를 정리했다.
  1. **요약 병목**: e_pool이 유닛 수백 개(최대 512 × 256)를 아군 평균·적 평균 두 개로 줄인다. 평균은 개수를 모르고(마린 10기와 50기의 평균이 비슷함), 소수 유닛(적 50기 중 다크템플러 1기)과 배치가 묻힌다. 행동 종류·delay는 이 요약과 Core 출력만 보고 정한다.
  2. **늦은 결합**: 유닛 토큰에는 좌표만 있고 지형(높낮이·길목·걷기 가능)은 맵 채널에만 있다. 맵 결과는 유닛 토큰으로 돌아가지 않는다. 스칼라 문맥(자원·전략 조건·시간)도 유닛 Transformer에 들어가지 않는다. 세 인코더는 768 요약과 헤드에서만 만난다.
  3. **장기 상태 없음**: GTrXL은 최근 64 step(이전 기억을 거치면 최대 약 128 step)만 본다. 세 인코더가 함께 쓰는 학습되는 장기 상태가 없다. 장기 정보는 규칙으로 채우는 AgentMemory(적 건물·정찰 기록·적 최대 관측 수 등)만 들고 간다.
- PyTorch 전환과 최적화로 학습이 CUDA 호스트 약 2.0 s/update, Linux ROCm 호스트 약 10 s/update가 됐다(`pytorch_opt.md` 3절). 그래서 10,000 update 비교 run을 여러 개 돌릴 여유가 생겼다. PyTorch BC 본학습은 아직 시작 전이라 구조를 바꾸기에 가장 싼 시점이다.
- 사용자 결정(2026-10-25)
  - 비교 대상: 지형 붙이기, 스칼라 토큰, attention 요약, 바깥 LSTM을 하나씩 켠 4개 + 네 가지를 모두 켠 1개.
  - 이번에 하지 않는 것: 위치 헤드 보강(Linux ROCm 호스트에서 가장 느린 부분이라 비용이 큼), 크기 확대(d_model 384), 적 병력 기억(관측 spec 030 결정 번복·feature 버전 변경 필요).
  - 평가: val 500게임 + 7개 모델 리그전(21쌍 × 100판). T0 대전은 사용자가 고른 후보만.
  - 잡음 측정용 추가 seed 없음, 장비는 2차 코어 비교처럼 혼합.
  - 빌드·CPU 테스트는 asus에서만 한다. 학습 장비 두 대는 2차 코어 비교를 학습 중이다. 새 run은 그 큐에 덧붙여 두 장비가 이어서 잡는다.

### 1.2 변형 (032 §2.7)
| run | `[model]` | 구조 |
|---|---|---|
| `arch-abl-terrain-pt` | `terrain = true` | 유닛 칸의 정적 맵 채널 10개 → `Linear(10→256)`을 유닛 입력 투영에 더함 |
| `arch-abl-ctx-pt` | `ctx_token = true` | 스칼라 요약 `c` → 토큰 하나를 유닛 Transformer에 추가(전용 bias 버킷), 요약·후보에서는 뺌 |
| `arch-abl-pool-pt` | `attn_pool = true` | e_pool을 질문 8개(아군 4·적 4) attention 요약 → `Linear(2048→256)`으로 대체 |
| `arch-abl-outer-pt` | `outer = 32` | 바깥 LSTM(`LSTMCell(512→256)`): 32 결정마다 직전 구간 Core 출력 평균으로 갱신, 상태를 유닛 토큰·맵 FiLM·Core 입력으로 되돌림 |
| `arch-abl-all-pt` | 네 가지 모두 | 위를 모두 켬. 추가 토큰 2개 |

- 공통 조건
  - 코어 GTrXL, 설정 `config/bc_core_ablation_pt.toml` + `[model]`(`config/bc_arch_abl_<변형>_pt.toml`).
  - 0 step부터 S1 10,000 update(warmup 500 + cosine), `data.seed` 1, 동결 데이터 2,069 파일.
  - 새 모듈은 고정 seed로 초기화하므로 공유 모듈의 초기 가중치와 데이터 순서가 기준과 같다.
- 기준과 다른 점은 `[model]` 하나뿐이다. 장비 비율은 run마다 달라진다(아래 4절에 기록).

### 1.3 실험 전 예상 (Claude, 측정 아님)
- 전체적으로 10,000 update(S1 한 epoch의 26%)는 학습 초반이라, 구조 차이보다 수렴 속도 차이가 먼저 보일 수 있다. 새 모듈이 많을수록 초반에 불리할 수도 있다.
- `attn_pool`: 행동 종류·delay NLL이 가장 개선될 후보로 본다. 유닛이 많은 장면에서 효과가 클 것이다.
- `terrain`: 선택·대상·위치 쪽에 작은 개선을 예상한다. 대상 맵 두 개는 좌표만으로도 지형을 외울 수 있어 차이가 작을 수 있다.
- `ctx_token`: 작은 개선 또는 차이 없음을 예상한다. 스칼라 문맥은 이미 행동 종류 헤드의 게이트로 들어간다.
- `outer`: val 지표 차이는 작을 것으로 본다. 학습 신호가 묶음 안 구간 전환 1회뿐이고, 1·2차 코어 비교에서 긴 기억이 병목이라는 증거가 없었다. 학습은 64 step을 두 구간으로 나눠 차례로 계산하므로 update 시간이 늘 것이다(정도는 첫 update에서 잰다).
- `all`: 효과가 더해지면 가장 좋고, 서로 겹치면 단일 변형 중 가장 좋은 것과 비슷할 것이다.
- 리그전: 1차 맞대결은 무승부(60분 제한)가 45%였다. 변형 사이 차이가 작으면 쌍별 승률 구간이 겹쳐 순위가 갈리지 않을 수 있다.
- 잡음 측정이 없으므로 val 지표의 작은 차이(예: NLL 1~2%)는 판단을 보류한다.

### 1.4 예상 시간
- 학습: 합계 50,000 update.
  - CUDA 호스트는 2차 GTrXL을 끝낸 뒤(10-25 새벽 기준 약 4시간 남음) 변형을 잡는다.
  - Linux ROCm 호스트는 2차 LSTM을 끝낸 뒤 합류한다. 혼자면 약 29시간 남았다.
  - CUDA 가용률 50%면 변형 학습 약 40시간, 25%면 약 62시간이다(변형별 update 시간은 아직 모름).
- 평가
  - val: run당 약 1시간 이하(CUDA 호스트).
  - 리그전 2,000판: 2차 T0 실적(약 77판/시간) 기준 약 26시간.
  - T0: 후보당 300판, 약 4시간.

## 2. 실행 절차
1. spec 확인 → 코드(Python만, Rust·작업자 바이너리·직접 작성 커널 소스는 그대로) → asus에서 빌드·CPU 테스트·CPU 표본 run → 리뷰어 리뷰 → master push.
2. 두 학습 장비의 작업 사본을 최신 master로 맞춘다(빌드 없음). 두 장비 모두 끝난 것을 확인한 뒤에만 다음으로 간다. 옛 코드가 변형 run을 먼저 잡으면 `[model]`을 무시하고 기준 모델을 학습하기 때문이다.
3. 큐 `<shared>/queue/core-ablation-pt.toml`의 LSTM·GTrXL 뒤에 `[[run]]` 5개를 덧붙인다(`core = "gtrxl"`, `config = "config/bc_arch_abl_<변형>_pt.toml"`). 작업자는 반복할 때마다 큐를 다시 읽으므로 재시작하지 않는다.
4. 각 장비가 변형 run을 처음 잡을 때 첫 update의 속도·손실·메모리를 확인한다. Linux ROCm 호스트가 메모리 예산(12 GiB, 060-R3)을 넘으면 그 run을 멈추고 보고한다.
5. 학습이 끝나면 CUDA 호스트에서 GPU 테스트(그래프 act = eager)를 돌린 뒤 평가한다.
   - val: `python -m uranium bc-eval <datasets> --checkpoint <shared>/checkpoints/<run>/ckpt-10000 --split val --games 500 --backend cuda`
   - 리그전: `python -m uranium run --plan config/plans/arch-abl-pt-league.toml --backend cuda`(생성 스크립트로 만든 20쌍 × 4 matchup)
6. 집계(쌍별 Wilson 95%, 7×7 표, Bradley–Terry 순위와 bootstrap 구간)를 보고하면 사용자가 T0 후보를 고른다.

## 3. 결과
(학습·평가 뒤)

## 4. 진행 기록
- 2026-10-25 — 결정과 계획(1.1). spec 032 §2.7·R20·AC11·AC12, 레지스트리 AMB-032-11 초안.
- 2026-10-25 01:5x — 사용자 확인으로 spec 확정. attention 요약 질문은 쪽당 4개(합 8개). 바깥 LSTM·attention 요약의 텐서 모양을 032 §2.7에 보강.
- 2026-10-25 01:5x~02:0x — 구현(Python만, Rust·작업자 바이너리·직접 작성 커널 소스는 그대로). 변형 모듈 `python/uranium/model/variants.py`, `[model]` 읽기 `bc/run.py`, 재개 보호 `bc/trainer.py`, Core 상태 호출을 `net.init_state/mask_state/detach_state`로 모음(학습기·평가·RL·추론 서버), RL은 `outer`를 거부. 설정 5개 `config/bc_arch_abl_*_pt.toml`, 벤치 스위트 `variants`, 테스트 `tests/python/test_variants.py`, 리그전 plan 생성 `tools/ablation/league_plan.py` → `config/plans/arch-abl-pt-league.toml`(20쌍 × 4 matchup, 2,000판, seed 35000~), 집계 `tools/ablation/league_table.py`.
- 2026-10-25 01:5x — asus(Linux 호스트 C) 테스트: 작업 브랜치 `arch-ablation`(ca64fdc)으로 패키지 재설치 14초, `cargo test --workspace` 통과, `pytest tests/python` 100 통과·13 건너뜀(GPU 전용), 그중 `test_variants.py` 36 통과·2 건너뜀(GPU).
- 2026-10-25 02:0x — asus CPU 표본 end-to-end(실제 변형 설정에서 레인만 2, `personal` shard): 다섯 변형 모두 2 update → `last/` 체크포인트 → 재개 1 update, rc 0, 손실 유한, 체크포인트 메타에 `variant` 남음. 다른 `[model]` 설정으로 재개하면 거부(rc 1). asus에 맵 사전계산이 4개뿐이라 처음엔 데이터 로더가 거부했고(B-2 검사), 이 PC의 1,249개를 복사했다. 산출물 asus `~/arch_abl/smoke/`.
- 2026-10-25 02:0x — 최악 모양(B1·T64·N512) 학습 1회의 CPU 최대 상주 메모리(f32, 재계산 없음, asus): 기본 6.16 GiB(파라미터 21,967,973), terrain 6.20(21,970,789), ctx 6.18(22,033,789), pool 6.16(22,626,149), outer 5.75(23,036,093), all 5.95(23,762,901). 변형으로 메모리가 거의 늘지 않는다(outer는 두 구간으로 나눠 계산해 더 낮음). CPU 추정이라 Linux ROCm 호스트 예산(12 GiB)은 첫 update에서 확인한다.
- 2026-10-25 02:1x — 집계 스크립트 검증: 1차 비교 맞대결 출력(`<shared>/_old_burn/artifacts/m6_core_ablation/final/h2h`)에서 GTrXL 42승·LSTM 13승·무 45, 42% [33, 52]·13% [8, 21], 결판 난 판 중 76% [64, 86]으로 1차 리포트 4절과 같다. 처음에는 게임 슬롯을 plan의 p0/p1로 읽어 틀렸다(슬롯이 p0/p1과 항상 같지 않음). 그래서 세션의 정책 ID와 `games.jsonl`의 p1 정책 ID로 이름을 붙이도록 고쳤다.

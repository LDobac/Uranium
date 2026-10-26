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
  - 평가: val 500게임 + 7개 모델 리그전(21쌍 × 100판). T0 대전은 사용자가 고른 후보만. → 2026-10-25 14:3x 사용자 결정으로 변경: 7개 모델 모두 T0 300판씩, 평가는 큐의 7개 run이 모두 끝난 뒤 함께(spec 032-R20 ③, master 20c7389).
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
  - T0: 모델당 300판(7개, LSTM·GTrXL은 2차 코어 비교 plan, 변형 5개는 `arch-abl-pt-t0.toml` 1,500판), 2차 실적 기준 변형 5개 약 20시간(추정).

## 2. 실행 절차
1. spec 확인 → 코드(Python만, Rust·작업자 바이너리·직접 작성 커널 소스는 그대로) → asus에서 빌드·CPU 테스트·CPU 표본 run → 리뷰어 리뷰 → master push.
2. 두 학습 장비의 작업 사본을 최신 master로 맞춘다(빌드 없음). 두 장비 모두 끝난 것을 확인한 뒤에만 다음으로 간다. 옛 코드가 변형 run을 먼저 잡으면 `[model]`을 무시하고 기준 모델을 학습하기 때문이다.
3. 큐 `<shared>/queue/core-ablation-pt.toml`의 LSTM·GTrXL 뒤에 `[[run]]` 5개를 덧붙인다(`core = "gtrxl"`, `config = "config/bc_arch_abl_<변형>_pt.toml"`). 작업자는 반복할 때마다 큐를 다시 읽으므로 재시작하지 않는다.
4. 각 장비가 변형 run을 처음 잡을 때 첫 update의 속도·손실·메모리를 확인한다. Linux ROCm 호스트가 메모리 예산(12 GiB, 060-R3)을 넘으면 그 run을 멈추고 보고한다.
5. 학습이 끝나면 CUDA 호스트에서 GPU 테스트(그래프 act = eager)를 돌린 뒤 평가한다.
   - val: `python -m uranium bc-eval <datasets> --checkpoint <shared>/checkpoints/<run>/ckpt-10000 --split val --games 500 --backend cuda`
   - 리그전: `python -m uranium run --plan config/plans/arch-abl-pt-league.toml --backend cuda`(생성 스크립트로 만든 20쌍 × 4 matchup)
   - T0: `python -m uranium run --plan config/plans/arch-abl-pt-t0.toml --backend cuda`(변형 5개 × 300판, LSTM·GTrXL은 `core-ablation-pt-t0.toml`)
   - 시점: 큐의 7개 run(LSTM·GTrXL·변형 5개)이 모두 10,000 update를 마친 뒤 한꺼번에(2026-10-25 사용자 결정).
   - **실행 방식(2026-10-26 사용자 결정, 020-R15, 위 `run --plan` 명령 대신)**
     - val 7개: 대전과 병행해 Linux ROCm 호스트 첫째에서 `bc-eval ... --backend rocm`. 7개가 같은 장비라 서로 공정하다.
     - 대전: 평가 작업 하나(`<shared>/eval/abl-pt/`). 단계는 맞대결(`core-ablation-pt-h2h`) → 리그전 20쌍 → 코어 T0 → 변형 T0 차례다.
     - CUDA 호스트: `tools\eval\eval_worker.ps1`(우선순위 10). 대기 작업자: Linux ROCm 호스트 첫째(2), 둘째(1, 멈춤 조사·재현 시험 뒤).
     - CUDA 호스트가 빠지면 대기 작업자가 새 판부터 추론한다. 판마다 추론 장비를 기록하고, 결과에 모델·단계별 추론 장비 비율을 함께 적는다.
     - 진행·집계: `python -m uranium eval-status --job eval/abl-pt`. 단계 폴더의 합친 기록으로 `league_table.py`를 돌린다.
6. 집계(쌍별 Wilson 95%, 7×7 표, Bradley–Terry 순위와 bootstrap 구간, 모델별 T0 승률)를 보고한다. 채택은 사용자 결정.

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
- 2026-10-25 02:3x — **리뷰**(별도 리뷰어 에이전트, CLAUDE.md 리뷰 규칙). 지적 5건과 처리:
  1. (중간) 리그전 집계가 시간 초과·끊긴 세션을 무승부로 세고, 한쪽 세션 오류나 상호 이탈이면 멈춤 → **반영**. 러너 `summary.md`(`report.rs`, 020-R13)와 같은 규칙으로 바꿨다: 정상 종료 세션이 있으면 집계, 결정 세션은 늦게 끝난 쪽, 상호 이탈·300프레임 미만·결과 없음은 사유별로 따로 셈. 1차 비교 출력으로 검증: 맞대결 42/13/45, T0 GTrXL 274판 28승 89패 157무(결과 없음 26), LSTM 300판 8승 164패 128무로 1차 리포트 4절과 같다.
  2. (낮음~중간) 바깥 LSTM 학습 경로의 헤드 입력(E·맵·스칼라 재결합)과 학습 묶음 사이 상태 이어받기를 확인하는 테스트 없음 → **반영**. 테스트 2개(`outer`, `all`에 각각)를 추가했다.
  3. (낮음) spec의 초기값 문구(U(−k,k), k = 1/√fan_in)가 일부 모듈(바깥 LSTM 셀, 추가 토큰 bias 행, 질문 벡터)의 구현과 다름, FiLM 위치(ReLU 앞뒤) 모호 → **반영**. 구현은 그대로 두고 spec 032 §2.7 문구를 구현에 맞게 고쳤다(LSTM 셀 k = 1/√256, bias 행 N(0, 0.02), 질문 k = 1/√d, FiLM은 stem ReLU 뒤).
  4. (낮음) `bc-profile`과 `bench bc`가 설정의 `[model]`을 무시하고 기본 모델을 측정 → **반영**.
  5. (낮음) Rust 실행 지문에 `[model]`이 없음 → **반영 안 함**. 학습 장비에 재빌드 없이 배포하려고 Python만 바꾸는 설계다. 다른 `[model]`로 재개·이어 학습하면 Python이 거부한다(032-AC11). 남는 위험(옛 코드가 변형 run을 0 step부터 잡는 것)은 2절 2번의 순서(두 장비 pull 확인 뒤 큐 추가)로 막는다.
- 2026-10-25 02:3x — asus `pytest tests/python` 104 통과·13 건너뜀(GPU 전용).
- 2026-10-25 02:2x — **배포·큐 추가**(2절 2~3번). master 3807031 push(fast-forward) → 두 학습 장비 작업 사본 `git pull --ff-only`(빌드 없음, 학습 venv는 editable 설치) → `<shared>/queue/core-ablation-pt.toml`에 `arch-abl-{terrain,ctx,pool,outer,all}-pt` 5개 run 추가(LSTM·GTrXL 뒤, 각자 `config`). 작업자는 재시작하지 않았다. 직후 두 학습은 그대로 진행(CUDA 호스트 GTrXL 5,189 update 1.7 s/update, Linux ROCm 호스트 LSTM 817 update 12.5 s/update). 예상: CUDA 호스트가 04:3x쯤 GTrXL을 끝내고 첫 변형을 잡는다.

- 2026-10-25 02:5x~05:2x — **GPU 테스트·프로파일링, 두 학습 장비**(사용자 지시: CUDA·Linux ROCm 호스트에서 변형의 학습·추론 확인, 비용이 비정상이면 프로파일 후 수정. 학습 중단 허가 → 다른 세션이 02:47에 2차 코어 비교 두 run을 멈춤). 기준점은 지금 학습 구현이고, 같은 날 같은 측정 방식으로 다시 쟀다(CUDA 1.74 s/update·GPU 최대 4.57 GiB, Linux ROCm 11.04 s·3.50 GiB. 실제 학습 로그 1.76 s·약 10~11 s와 같음).
  - 방법: `bench bc`의 `run_group`(실제 `BcTrainer.train_batch`, 같은 seed·같은 batch)에 update마다 메모리(torch 할당·예약, 프로세스, 시스템 가용, GTT)를 붙인 측정. 기준·모두 켬은 100 update(누수 비교), 나머지는 30. 시간은 같은 step(같은 batch)끼리 짝지은 비율의 중앙값. 추론은 서버 경로(compile + graph, 장비 기본값)로 B 1·4·16 × N 64·128·256. 연산별 장치 시간은 torch profiler. 장비 레인 그룹·데이터 공급 설정은 학습 작업자와 같게.
  - **GPU 테스트 실패 2건 → 테스트 수정(모델 문제 아님)**
    1. 직접 작성 attention 커널 대 torch(추가 토큰): 최대오차/최대값 1e-3 기준에서 1.3e-3(CUDA)·2.1e-3(ROCm). f64 정답과 비교하면 torch f32 경로도 같은 지표로 2~5e-3이고 커널도 같은 수준(상대 L2 최대: 커널 2.2e-4/3.1e-4, torch f32 2.3e-4/3.1e-4) → 바닐라 테스트와 같은 지표(상대 L2, 노름 하한)로 바꿨다. 토큰 bias 행은 gradient 노름(약 7e-4)이 하한(5.2)보다 훨씬 작아 따로 하한 없는 상대 L2 < 1e-2로 본다(측정 1.1e-3·1.9e-3, f32 잡음 2~3e-3).
    2. 그래프 act 대 eager(모두 켬, CUDA 비트 비교): attention 요약의 Linear(2048→256)에서 cuBLAS가 3행과 4행에 다른 커널을 고른다(eager끼리도 3행 대 4행이 다름, 그래프만 떼면 eager와 같음). 그래프는 배치 3을 4로 채워 돌리므로 bf16 1~2칸 차이 → eager도 같은 4행으로 돌려 앞 3행을 비교하도록 바꿨다(패딩된 변형 상태도 함께 확인, 비트 일치).
  - **동등성**(모두 켬, 기본 경로 대 기준 경로 31 update 손실): CUDA compile·직접 작성 커널 대 eager·torch attention 10 update 구간 최대 2.9e-5(bf16), ROCm 직접 작성 커널 대 torch 최대 1.8e-7(f32). 실제 `bc` 루프(Rust run 루프) 모두 켬 2 update → `last/` → 재개: 두 장비 rc 0.
  - **추론**(서버 경로): 모든 변형에서 그래프 끊김 0·재컴파일 없음, 캡처 수 기준과 같음. Linux ROCm B1(그래프) 12.2 ms → 지형 +0%, 스칼라 토큰 +1%, 요약 +3%, 바깥 LSTM +3~4%, 모두 켬 +6~7%(12.9~13.0 ms), B16 N256 +9%. CUDA B1 1.3~1.5 ms → 모두 켬 +4~12%. 메모리 증가분(시스템 가용 기준): ROCm 기준 1.27 GiB, 모두 켬 1.41 GiB(060-R3 추론 상한 2 GiB 안).
  - **학습 시간**(같은 batch끼리 비율 중앙값, 1~30 update)

    | 변형 | CUDA 수정 전 | CUDA 수정 후 | Linux ROCm 수정 전 | Linux ROCm 수정 후 |
    |---|---|---|---|---|
    | 지형 | +0.1% | (수정 대상 아님) | +0.1% | (수정 대상 아님) |
    | attention 요약 | +0.7% | (수정 대상 아님) | +1.2% | (수정 대상 아님) |
    | 스칼라 토큰 | +4.8% | **+2.3%** | +3.2% | **+1.1%** |
    | 바깥 LSTM | +12.3% | **+10.5%** | +11.8% | **+9.7%** |
    | 모두 켬 | +14.3% | +14.6% | +14.4% | +14.8% |

    - 추가 연산량(행당 곱셈-덧셈)은 바깥 LSTM 약 2.7M, 기존 모델 약 1.3G 이상 → 연산량으로 기대되는 증가는 1% 미만. 그래서 비정상으로 보고 프로파일했다.
    - **발견·수정 1(같은 수식, 032-R19)**: BC 묶음은 엔티티 축을 64의 배수로 채우고(`pad_entities = 64`), 추가 토큰 K개(스칼라·바깥 LSTM)는 그 뒤에 붙는다. N+K가 늘 직접 작성 커널의 64칸 타일을 하나 더 쓴다(CUDA B1 N512 학습 1회 115 ms 중 약 6 ms). → 학습기가 모든 행의 끝 K칸이 패딩이면 엔티티 축 7개 배열을 N−K로 자른다(`trim_entity_pad`, collate 스레드에서). 패딩은 어디서나 마스크로 빠지고, 선택 헤드의 END 라벨(패딩된 N)은 `clamp(0, n)`으로 새 N이 된다. 실데이터에서 99~100% 묶음에 적용된다. CPU 테스트: 자른 묶음과 안 자른 묶음의 손실·gradient 1e-5 안에서 같음. dropout이 있으면 난수 배치(텐서 크기)가 달라질 뿐 분포는 같다(변형 run은 아직 시작 전이라 영향 없음).
    - **발견·수정 2**: 바깥 LSTM·모두 켬만 실데이터 update 14에서 76~94초 재컴파일. `TORCH_LOGS=recompiles`: compile이 엔티티 수에 `N ≤ 318` 상한 guard를 걸었고, 출처는 토큰을 떼는 `e[:, :n]`(n을 `entity_mask` 길이에서 가져와 두 입력 크기의 관계가 guard가 됨) → n을 x 자체의 길이에서 가져오게 바꿨다(값 같음). 수정 후 30 update에서 첫 compile 외 재컴파일 없음.
    - **남은 비용(바깥 LSTM, 설계에서 옴)**: 패딩을 자른 뒤 장치 시간 차이는 +2~3.5%뿐이고(실데이터 update 1회 연산별), 나머지는 64 step을 32 step 두 구간으로 나눠 인코더 3종·코어를 두 번 부르는 호출 비용(aten 호출 eager 10,877 → 16,120, compile 2,257 → 3,013)이다. 구간 k+1의 유닛 토큰·맵 FiLM·코어 입력이 구간 k 끝의 바깥 상태에 의존하므로 같은 수식으로는 차례 계산이 필요하다. h_o와 무관한 부분(스칼라 인코더·엔티티 입력 임베딩·거리 bucket)만 한 번에 계산하는 안은 늘어난 호출의 약 5%라 하지 않았다. 더 줄이는 방법은 사용자 결정이 필요하다: (가) 바깥 LSTM run의 CUDA 레인 그룹을 키움(AMB-041-7 값 변경, dropout 난수 배치 달라짐), (나) h_o를 코어 입력에만 넣는 구조로 바꿈(spec 변경, 다른 변형이 됨).
    - 모두 켬의 수정 효과가 표에서 안 보이는 것: 실데이터 update 1회 프로파일에서 모두 켬 1.55 s 대 바깥 LSTM 1.53 s(+1.5%, 지형·요약·토큰 합과 맞음). 30 update 비교의 잡음(약 ±2%) 안이다.
  - **메모리·누수**: GPU 최대 할당 CUDA 4.57 → 4.79 GiB(모두 켬), Linux ROCm 3.50 → 3.85 GiB, ROCm 시스템 가용 최저 4.19 → 4.13 GiB(14 GB 중, 12 GiB 예산 안). 100 update 기울기(5 update 뒤부터): torch 할당은 둘 다 0, 프로세스·예약·GTT 증가는 기준과 모두 켬이 같다(CUDA 프로세스 23 대 18 MiB/update, ROCm 7.1 대 7.4 MiB/update·GTT 9.1 대 9.0) → 변형이 만든 누수 없음. 기존 경로의 초반 증가는 긴 run에서 평평해진다(`m6_core_ablation_pt.md` 4절, glibc arena 제한).
  - bench suite(합성, eager) 표의 최대 GPU 할당은 앞 케이스의 모델을 캐시에 남겨 누적된 값이라 변형 비교에 쓰지 않았다. 합성 `net_train`은 패딩을 자르지 않는다(학습보다 타일 하나 더).
  - **리뷰**(별도 리뷰어 에이전트): 높음 없음. 반영 — 그래프 테스트가 배치 패딩을 다시 확인하게, 자르기를 collate 스레드로, dropout 난수 배치 설명, 자른 묶음의 `n` 키 갱신, Rust 키 목록 위치 주석, 토큰 bias gradient의 하한 없는 비교, bench 설명. 반영 안 함 — 학습기가 실제로 자르는지 보는 테스트(재측정 시간 감소로 확인).
  - 테스트: CPU `pytest tests/python` 105 통과·3 건너뜀, GPU(`-m gpu`) CUDA 12 통과, Linux ROCm 10 통과·2 건너뜀.
  - 원자료: CUDA 호스트 메인 작업 사본 `reports/raw/arch_abl_gpu_20261025/`(git 미추적, 장비별 폴더), Linux ROCm 호스트 홈 `arch_abl_gpu/`(`out/` 1차, `fix1/` 수정 후). 측정 스크립트 `varprobe.py`·`prof_outer.py`·`diag_*.py`도 같은 폴더 `scripts/`.
- 2026-10-25 14:3x — **결정**(사용자, 다른 세션에서 반영 master 20c7389, 032-R20 ③): 평가 범위 변경 — 7개 모델(LSTM, GTrXL, 변형 5개, 모두 ckpt-10000) 평가를 큐의 7개 run 학습이 모두 끝난 뒤 함께 한다. 맞대결 21쌍 × 100판, val 500게임 7개 모두, T0는 후보만이 아니라 7개 모두 모델당 300판(변형 5개 plan `config/plans/arch-abl-pt-t0.toml`, seed_base 36000). 이 리포트 1.1·1.4·2절을 맞춰 고쳤다.

# PyTorch 이식 1단계 진행 기록 (AMB-000-4)

계획: `C:\Users\<user>\.claude\plans\bubbly-mixing-mochi.md`(로컬). 근거 실측: `reports/torch_vs_burn.md`.

## 1. 배경과 예상
- 1단계는 학습·추론을 기능 그대로 Python + PyTorch로 옮긴다. 데이터 공급·저장소·실행 루프(잠금·heartbeat·체크포인트·양보)는 Rust에 두고 PyO3 `uranium._core`로 부른다.
- 예상: 10-23 벤치 기준으로 bc250_1(ROCm)에서 Burn Vulkan보다 학습이 3~4배 빠르다. Rust ↔ Python 경계(브리지)는 update당 호출 몇 번이라 비용이 거의 없을 것이다.
- 사용자 요청(10-24): 브리지 때문에 느려지지 않게 설계하고, 오버헤드를 실측해 크면 최적화 계획에 넣는다.

## 2. 구조 (1-4·1-5)
- `uranium_data::bc_run`: Burn 학습기의 실행 루프를 trait `BcTrainer`로 일반화했다. step·학습률·데이터 위치·`train_state.json`·`run_ident.json`·config 복사·061 잠금/push/양보·종료 코드(75/76)는 Rust가 맡는다.
- Python `uranium.bc.trainer.BcTrainer`: 배치 하나 → update 하나(그룹별 core 상태, 배치 전체 valid로 나눈 손실, 전역 clip·win_open 분리, AdamW). 체크포인트는 `model.safetensors`·`optim-main.safetensors`·`meta.json`(061-R10).
- `python -m uranium bc | bc-eval | bc-profile`. bc-eval의 게임 선택·서렌더 재현(041-AC4)은 Rust `uranium_data::eval::EvalStream`.
- 레인 그룹 collate는 Rust(`Batch.group`, GIL 해제)에서 하고, numpy 배열 소유권만 넘긴다(복사 없음).

## 3. 실측: bc250_1, 실데이터(personal shard), GTrXL, `config/bc.toml`(32레인 × T64, micro_lanes 1)
`URANIUM_BC_PROFILE=1`로 구간마다 GPU를 동기화해 잰 값이다. Burn 비교값은 같은 장비·같은 설정(엔티티 조각 재계산 16·압축 켬)의 코어 비교·표본 실측 약 108초/update(`m6_pre_bc_review.md` I절).

### 3.1 첫 측정 (10-24 02:0x): PyTorch가 Burn보다 3~6배 느림
| 구간 | update 하나 |
|---|---|
| 데이터 대기(prefetch 0) | 3.6s |
| collate(32그룹 합) | 2.07s |
| GPU 전송 | 0.08s |
| Rust → Python 호출 | 약 0.01s |
| forward / backward | 71s / 224s |
| 합계 | 297~630s |

- 브리지 비중은 약 2%였지만, 계산 자체가 Burn(108초)보다 느렸다.
- `bc-profile`(2레인) 연산별 시간: `miopen_convolution_transpose` 8회 20.3초(회당 2.5초, CPU 시간 = 장치 시간), `convolution_backward` 13.4초, 나머지 전부 약 0.7초. 엔티티 조각 재계산 16과 0은 속도가 같고 메모리만 1.3 대 2.1 GiB였다.

### 3.2 원인: ROCm에서 cuDNN benchmark 모드
- 위치 헤드(transposed conv 4개)는 위치를 쓰는 행만 계산한다(032-R10). 그래서 conv의 배치 크기가 호출마다 바뀐다.
- 학습기가 GPU면 `cudnn.benchmark=True`를 켰는데, ROCm에서는 MIOpen이 새 모양마다 탐색(find)을 다시 한다. 같은 모양 transposed conv fwd+bwd 마이크로벤치:

| 설정 | 새 배치 크기 첫 호출 | 반복 호출 |
|---|---|---|
| benchmark 켬 | 4.6~125초 | 17~174ms |
| benchmark 끔(MIOpen immediate) | 88~219ms(첫 커널 컴파일 1회 15.9초) | 38~176ms |
| MIOpen 끔(기본 구현) | 60~314ms | 42~292ms |

- spec 041-R15는 benchmark 모드를 CUDA 호스트에만 쓰고 ROCm은 MIOpen find-db만 쓰도록 정해 두었다. 코드가 ROCm에도 켠 것이 spec과 어긋난 것이었다. `uranium.device.configure`로 CUDA에서만 켜도록 고쳤다(계산 결과는 같다).

### 3.3 고친 뒤 (10-24 02:4x): Burn보다 약 5배 빠름
| | 데이터 대기 | collate | forward / backward | update |
|---|---|---|---|---|
| benchmark 끔, prefetch 2 | 0 | 2.5~3.5s(학습 스레드) | 3.4~4.7 / 15.3~17.6s | 약 22s |

- collate가 update의 약 12%로, 그동안 GPU가 쉰다. → 브리지 최적화로 다음 그룹 collate를 작업 스레드에서 미리 하도록 바꿨다(GIL은 Rust collate가 놓는다. 메모리에는 그룹 두 개, 각 약 120MB).

### 3.4 다음 그룹 collate 미리 하기 (10-24 03:0x)
| prefetch | 데이터 대기 | collate 대기 | update(대기 포함) |
|---|---|---|---|
| 2 | 0 | 0.1~0.3s | **약 20s** |
| 0(bc250_1 `.env` 현재값) | 3.6~3.9s | 0.1s | 약 22.5s |

- 브리지 비중(collate 대기 + 전송 + 호출)은 update 20초 중 0.3~0.5초, 약 2%다. Burn 108초 대비 약 5.4배.
- 가끔 50초대 update가 있다. 새 모양의 MIOpen 커널을 처음 컴파일하는 경우로 보이며, 컴파일된 커널은 디스크 캐시에 남아 갈수록 줄어든다.
- 남은 시간의 대부분은 위치 헤드 transposed conv의 forward·backward(MIOpen)다.

## 4. 2단계 최적화 계획에 추가한 항목 (계획 2단계 기법 12·13)
1. 위치 헤드 transposed conv(ROCm): 연산별로 MIOpen과 기본 구현 중 빠른 쪽 고르기 → 필요하면 직접 작성 커널. 행 수를 몇 가지 크기로 맞춰(남는 행은 계산 후 버림, 같은 수식) 모양 수를 줄이는 방법도 비교한다.
2. bc250_1 prefetch 2 채택 여부: 데이터 대기 3.7초가 사라지지만 RAM 사용량(압축 배치 2개)을 실측한 뒤 정한다.
3. GPU 전송: pinned 버퍼와 별도 stream(지금은 update당 0.1초라 순위 낮음).

## 5. 산출물
- bc250_1 `~/bcprof/`: `host.log`(3.1), `prof_c16.json`·`prof_c0.json`(2레인 연산별), `prof_full.json`·`run2.log`(3.3), `run3_2.log`·`run3_0.log`(3.4)
- 마이크로벤치 스크립트: bc250_1 `~/deconv_bench.py`

## 6. BC 동등성 (1-6, 진행 중)
- 방법: Burn `bc --steps 0`으로 만든 초기 체크포인트의 가중치를 `tools/burn-parity --export` → `bc_parity.py convert`로 PyTorch 체크포인트로 바꾸고, 두 학습기를 `--init`(step 0, 같은 학습률 일정)으로 같은 S1 데이터(personal shard)에서 500 update씩 돌린다. dropout은 둘 다 0으로 두어 난수 잡음을 없앴다. 설정은 `config/bc.toml`에서 lanes 4, warmup_steps 50만 바꿨다(500 update 안에 최대 학습률에 닿도록).
- 예상: 두 손실은 float 반올림 차이만 있고, update가 쌓이며 차이가 천천히 커진다.
- 5 update 시험(10-24 03:0x): 총 손실 상대 차이 update 1 0, 3 3.2e-7, 5 1.9e-6. 같은 5 update에 PyTorch 25초, Burn 110초(시작 시간 포함).
- 500 update: 03:05 시작.

## 7. 추론 이식 (1-7)
- 구조: `uranium-agent`에서 Burn을 뺐다. 세션 쪽 API(`InferenceHandle`·`open`·`act`·`state`)는 그대로이고, 서버 쪽은 `uranium_agent::broker::Broker`다. Broker는 슬롯·배치 규칙(최대 배치, 활성 세션 전원 대기, `max_wait`)을 지키며 요청을 모으고 host collate한다. Python 모델 스레드(`uranium.agent.infer.InferenceServer`)는 `next()`로 배치를 받아 `[capacity, …]` core 상태(장치에 상주)에서 행을 골라 `act`를 돌리고, `reply()`로 돌려준다. 행동 해석(034 열, END = 패딩 n)은 Rust가 한다. Burn RL용 서버는 `uranium_train::burn_infer`로 옮겨 1-8까지만 쓴다.
- 진입점: `python -m uranium serve | scr | run | random-checkpoint`. 모델 플레이어가 있는 plan은 Python이 모델마다 Broker와 스레드를 띄운 뒤 Rust 스케줄러를 GIL 없이 돌린다. `uranium-env run`·`uranium-agent serve`는 스크립트 플레이어만 맡는다.
- 테스트: Rust `uranium-agent` 세션 테스트 7개(mock 모델: 경로 지정, realtime, observer, 종료 보상, 슬롯 상태·재열기, 배치 행 배분, 항복). pytest `test_infer.py` 9개(작은 실모델 CPU: 마스크 안 행동 031-R4, argmax 결정론 031-R7, 배치 = 단독 032-R14, 슬롯 상태 032-R16, random-checkpoint). asus에서 통과(10-24 03:2x).
- end-to-end(10-24 03:22, asus CPU): random-init LSTM 체크포인트 대 T0 테란 1판(투혼, 4,000프레임 시간 제한 무승부). 결정 2,000회, 결정 지연 p50 22ms·p95 37ms, executor 거절 0, 실패 명령 0, results·games·summary 생성. 산출물 asus `~/e2e17/`.
- 남은 것: Windows CUDA PC에서 `uranium._core` 빌드와 SC:R warm-up 확인(학습 PC 부하라 사용자 확인 후). `bc-vs-t0`·코어 비교 plan의 체크포인트 경로는 Burn 체크포인트라 PyTorch BC 체크포인트가 생기면 바꾼다.

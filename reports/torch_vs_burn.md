# Burn 대 PyTorch 속도 실측 (학습 PC·bc250_1, 2026-10-23)

## 배경과 예상 (측정 전)
- 사용자 질문: "Burn이 PyTorch(CUDA native)보다 얼마나 느린가." 처음에는 저장소에 직접 비교한 측정이 없어서, `bench_cuda.md`(10-20)의 수치로 추정했다.
  - 큰 행렬곱 하나만 보면 PyTorch보다 1~1.5배 느린 수준으로 봤다. TF32 4096²가 약 29 TFLOP/s였기 때문이다.
  - 모델 학습 전체로는 PyTorch(cuDNN·FlashAttention·AMP)가 2~4배 빠를 것으로 예상했다. 근거는 다음과 같았다.
    - 실제 BC 학습 중 GPU 사용률이 평균 42%였다.
    - 거리 bias attention의 backward가 forward보다 140배 느렸다.
    - conv의 forward+backward가 forward의 8배였다.
    - Burn 0.22에는 AMP가 없다.
- 그래서 같은 모델 구조를 PyTorch로 옮겨, 같은 GPU에서 Burn `tools/bench`와 같은 케이스로 직접 재기로 했다. 사용자 요청에 따라 Windows와 WSL2에서 각각 쟀다.
- 이 비교는 측정만 한다. 프레임워크(Burn 0.22.0)는 고정 조건이고, 바꾸려면 spec 변경과 사용자 결정이 필요하다.

## 방법
- **PyTorch 이식본**: `crates/uranium-model` 전체를 옮겼다(entity·spatial·scalar 인코더, LSTM·GTrXL 코어, 자기회귀 헤드 6개, value 헤드).
  - 파라미터 수가 Burn과 정확히 같다: LSTM 11,343,764개, GTrXL 21,715,364개.
  - 가중치를 옮겨 출력값까지 같은지는 확인하지 않았다. 속도 비교용이다.
  - 액션 테이블은 저장소의 `actions.toml`과 `actions.rs`를 그대로 읽는다(101개 액션, capability 50개, location 클래스 8개).
- **모드 세 가지**
  | 모드 | 내용 |
  |---|---|
  | faithful | Burn 코드를 연산 단위로 그대로 옮김. 자체 LSTM 스텝 루프, 명시적 attention, one-hot @ table 거리 bias(032-R13), Selected 12스텝 루프, Location에서 쓰는 행만 계산(호스트 동기화 포함), GTrXL 상대위치 표를 호스트에서 생성 |
  | fast | PyTorch에서 보통 쓰는 방식: cuDNN `nn.LSTM`, `scaled_dot_product_attention`, embedding 조회 bias |
  | fast-bf16 | fast + bf16 autocast(AMP). Burn 0.22에는 없는 기능 |
- **같게 맞춘 것**
  - 케이스 이름과 모양
  - 반복 수: warmup 3회, iters 20회, act는 50회. 통계는 중앙값과 p95
  - 합성 입력: `synthetic::random_obs`와 같은 분포와 마스크 불변식
  - 학습 케이스: `act` 샘플로 라벨을 만들고, dropout 0.1을 켠 상태에서 forward + `-logp.sum().mean() + value²` + backward를 잰다. optimizer는 넣지 않았다(Burn 벤치와 같음)
  - TF32 matmul. PyTorch는 `cudnn.benchmark`를 켰다(Burn autotune에 해당)
- **장비**: 학습 PC. RTX 4070 Ti SUPER 16GB, 드라이버 616.92. Windows 11과 WSL2(Ubuntu 26.04)로 각각 쟀다.
  - BC 본학습 작업자는 16:22에 멈춘 뒤라 측정 중에 다른 GPU 작업은 없었다.
- **Burn**: 0.22.0, autotune과 fusion을 켬(041-R15). master 519709a의 `bench`를 썼다.
  - Windows: 기존 `--features cuda` 빌드를 그대로 썼다. 그 뒤 커밋은 `Cargo.toml` 세 줄만 바꿨다.
  - WSL: 같은 커밋을 새로 빌드했다. sudo 없이 CUDA 13.4.3 redist(cudart·nvrtc·nvcc·crt·libnvvm·cccl)를 홈 폴더에 풀어 `CUDA_PATH`로 지정했다.
- **PyTorch**: 2.14.1+cu130, cuDNN 9.24, Python 3.12.

## 진행 기록
- 17:0x: 모델 코드를 읽고 PyTorch 이식본을 작성했다. 파라미터 수가 Burn과 같은지 확인했고, `--quick` 스모크에서 모든 케이스가 통과했다.
- 17:30~17:41: Windows Burn 전체 벤치(`--backend cuda --no-accuracy`, 리포트는 임시 경로로 출력).
- 17:41~17:43: Windows PyTorch 세 모드.
- 17:4x: WSL PyTorch 세 모드.
- 17:45~17:49: WSL Burn 빌드(3분 47초). 첫 시도는 빌드 명령의 셸 인용 오류로 시작하지 못했다.
- 17:5x~18:0x: WSL Burn 전체 벤치.
- 오늘 Windows Burn은 학습 케이스 몇 개가 10-20 리포트보다 크게 느렸다.
  - lstm B4·T16: 1,078ms (10-20에는 298ms)
  - lstm B1·T64·N512: 1,139ms (10-20에는 749ms)
  - gtrxl B2·T16: 278ms (10-20에는 196ms)
  - 같은 날 WSL Burn은 10-20 Windows 값과 비슷했다(165·296·176·770ms).
  - 원인은 확인하지 않았다. Windows에서 autotune 캐시가 남지 않는 문제(`m6_bc_preflight.md` 5차)와 관련이 있을 수 있다.
  - 아래 비교에서 Burn 값은 Burn에 유리하도록 세 측정(오늘 Windows, 오늘 WSL, 10-20 Windows) 중 가장 좋은 값을 쓴다.

## 결과: 전체 모델 (ms, 작을수록 빠름)

PyTorch 칸은 "Windows / WSL" 순서다.

| 케이스 | Burn 최선 | torch faithful | torch fast | torch fast-bf16 | Burn ÷ torch |
|---|---|---|---|---|---|
| 학습 LSTM B2·T16·N256 | 164.6 (WSL) | 90.7 / 79.5 | 60.2 / 89.0 | 63.1 / 78.7 | 1.8~2.7배 |
| 학습 LSTM B4·T16·N256 | 295.6 (WSL) | 119.8 / 124.8 | 96.4 / 104.9 | 85.4 / 100.9 | 2.4~3.5배 |
| 학습 GTrXL B2·T16·N256 | 176.5 (WSL) | 79.2 / 102.8 | 79.6 / 83.0 | 87.4 / 94.4 | 1.7~2.2배 |
| 학습 LSTM B1·T64·N512 | 749.4 (10-20 Windows) | 285.4 / 286.1 | 204.3 / 210.0 | 175.2 / 181.4 | 2.6~4.3배 |
| 학습 GTrXL B1·T64·N512 (본학습 레인 모양) | Burn 벤치에 없음 | 260.6 / 270.5 | 227.5 / 242.8 | 195.4 / 207.9 | – |
| 추론 act LSTM B16·N256 | 39.5 (WSL) | 19.2 / 26.3 | 18.3 / 16.0 | 20.1 / 18.7 | 1.5~2.5배 |
| 추론 act GTrXL B16·N256 | 40.8 (WSL) | 30.5 / 25.7 | 21.2 / 21.6 | 22.2 / 20.5 | 1.3~2.0배 |
| 엔티티 인코더 fwd+bwd B16·N256 | 36.6 (WSL) | 12.6 / 12.6 | 11.1 / 12.1 | 11.2 / 10.9 | 약 3배 |

- 학습 최대 메모리(PyTorch, B1·T64·N512): faithful 7.5 GiB, fast 5.1 GiB, bf16 4.5 GiB.
- 추론 p95: Burn 51~114ms, PyTorch 26~63ms. 두 프레임워크 모두 들쭉날쭉했고 WSL 쪽이 더 컸다.

## 결과: 커널 (Windows, ms)

| 케이스 | Burn fwd / fwd+bwd | torch fwd / fwd+bwd | 비고 |
|---|---|---|---|
| matmul f32(TF32) 4096² | 4.87 | 3.27 | 1.5배 (28 대 42 TFLOP/s) |
| matmul f32 1024² | 0.147 | 0.086 | 1.7배 |
| matmul f16 4096² | 2.27 | 1.63 | 1.4배 |
| matmul f16 1024² | 0.721 | 0.058 | 12배. 10-20 리포트에서도 같은 이상값 |
| conv 3×3 [16,58,128²] | 1.41 / 11.37 | 0.41 / 1.80 | backward 쪽 6배 |
| conv transpose 8→128 | 1.35 / 9.98 | 0.73 / 1.74 | backward 쪽 6배 |
| LSTM B32·T64 (Burn 기본 `Lstm` 대 cuDNN) | 18.4 / 62.4 | 1.38 / 3.82 | 16배. 모델은 자체 LSTM 루프(`core.rs`)를 써서 차이가 이보다 작다 |
| transformer N256 | 3.05 / 12.35 | 1.72 / 5.98 | 2배 |
| transformer N512 | 9.97 / 25.83 | 3.83 / 14.99 | 1.7~2.6배 |
| 거리 bias attention (테이블 `select`) | 0.75 / 105.1 | 0.51 / 2.37 | backward 44배. 모델은 032-R13의 one-hot @ table을 써서 이 경로를 피한다(엔티티 인코더 차이는 약 3배) |
| scatter-add 128² | 0.072 / 0.694 | 0.171 / 0.303 | forward는 Burn이 2.4배 빠름 |
| pointer GRU 12스텝 | 2.76 / 9.12 | 3.26 / 13.41 | Burn이 1.2~1.5배 빠름. 작은 연산이 이어져 Python 호출 비용이 드러남 |

## 해석
- **같은 모델을 같은 GPU에서 학습할 때 PyTorch가 2~2.6배 빠르다.** 코드를 그대로 옮긴 faithful 모드 기준이다. PyTorch에서 보통 쓰는 방식(cuDNN LSTM·SDPA)이면 3~3.7배, bf16 AMP까지 쓰면 최대 4.3배다. 측정 전 예상(2~4배)과 맞는다.
- 차이의 대부분은 **backward**에서 나온다.
  - conv와 conv transpose는 forward 차이가 2~3배인데, backward를 포함하면 6배가 된다.
  - 큰 행렬곱 자체의 차이는 1.4~1.5배로 예상과 같다.
  - 작은 f16 행렬곱(1024²)은 Burn이 12배 느리다. autotune이 고른 커널이 맞지 않는 것으로 보인다.
- Burn이 앞서는 곳은 scatter-add forward와, 작은 연산이 이어지는 pointer GRU다. PyTorch eager는 연산마다 Python 호출 비용이 든다.
- **Windows와 WSL의 차이는 작다.**
  - PyTorch 학습은 ±10% 안이다.
  - Burn은 WSL 값이 10-20 Windows 값과 비슷했다. 오늘 Windows 실행만 느렸다.
  - 작은 커널(1024² matmul, scatter)과 추론 p95는 WSL이 더 나쁘다.
  - 지금 이 PC를 WSL로 옮겨 얻는 속도 이득은 없다.
- Burn LSTM B2·T64·N512는 Windows와 WSL 모두 같은 원인으로 실패했다. autotune 후보 `matmul_naive`의 "Cube count too big (1048576, 3, 1)" 오류이며, `bench_cuda.md`(10-20)와 같은 현상이다. 실제 학습은 레인 1개(micro_lanes 1)라 이 모양을 쓰지 않는다.
- 참고(결정 아님): Burn 안에서 개선 여지가 큰 곳은 conv·conv transpose backward(spatial 인코더, Location 헤드), 작은 f16 행렬곱 커널 선택, AMP 부재다.

## bc250_1: ROCm PyTorch 대 Burn Vulkan (21:49~22:20)

### 배경과 예상 (측정 전)
- 사용자가 bc250_1(Linux Vulkan 호스트, BC-250 gfx1013, 40 CU, 최대 1700 MHz)에 ROCm 7.2.4를 설치하고, gfx1013용 rocBLAS와 PyTorch를 직접 빌드했다. 공개된 BC-250 ROCm 조사 저장소의 절차를 Arch에 맞게 바꿨고, amdgpu 모듈에 flush 패치 2종을 넣고 SDMA 펌웨어도 교체했다. 공식 ROCm은 gfx1013을 지원하지 않는다.
- 사용자 지시: ROCm PyTorch 벤치는 bc250_1에서만 돌린다. 같은 이식본과 같은 케이스를 같은 장비의 Burn Vulkan(f32, autotune·fusion)과 비교한다.
- 예상: 속도 수치는 미리 예상하지 않았다. 셋업 기록상 fp16 autocast가 이 보드에서 7.7배 느렸으므로, bf16 모드도 느릴 것으로 봤다. 비교 기준은 f32다.

### 방법과 진행 기록
- 실행 환경
  - 소스 빌드 torch 2.14.1a0(HIP 7.2, Python 3.14)과 홈 폴더의 gfx1013 rocBLAS를 `LD_LIBRARY_PATH`로 지정하고 `HSA_ENABLE_SDMA=0`을 붙였다.
  - `/opt/rocm`의 rocBLAS에는 gfx1013 Tensile 라이브러리가 없어서, 지정하지 않으면 첫 행렬곱에서 중단된다.
- **MIOpen LSTM backward가 gfx1013에서 실패한다**(`miopenStatusUnknownError`). 연산별로 시험한 결과, MIOpen을 켠 채로 conv·conv transpose·SDPA·bf16 행렬곱은 forward와 backward 모두 정상이었고 LSTM backward만 실패했다.
  - 그래서 ROCm에서는 `nn.LSTM` 호출 때만 MIOpen을 끄고 PyTorch 자체 커널을 쓴다.
  - 영향받는 것은 fast 모드의 LSTM 코어와 `lstm_seq` 커널 케이스뿐이다. 나머지 코드는 학습 PC 측정과 같다.
- 정확성 확인(셋업 기록의 권고)
  - 벤치 전후로 셋업 저장소의 `torch_train.py`(MLP 50스텝을 CPU와 대조)를 돌렸다. 두 번 모두 loss 차이 1.799e-05, 파라미터 차이 9.312e-03으로 셋업 기록의 기준값과 같았다.
  - 이식 모델도 같은 가중치와 입력으로 CPU와 GPU를 비교했다(`check_rocm.py`, B2·T8·N128, dropout 끔). LSTM·GTrXL × faithful·fast 네 조합 모두 logp·value·gradient 최대 차이가 1e-6 수준이었다.
  - GPU 두 번 실행은 비트 단위로는 다르지만 차이가 1e-6 수준이다. atomic 연산의 누적 순서 차이로 보며, 벤치에는 영향이 없다.
  - 벤치 뒤 커널 로그에 fence, page fault, GPU 리셋 메시지는 없었다.
- 측정 경과
  - 21:49~22:10: PyTorch 세 모드
  - 22:10~22:16: Burn Vulkan 전체 벤치(10-22 빌드, 모델 코드는 그 뒤로 바뀌지 않음)
    - `net_train_gtrxl_b2_t16`까지는 정상이었다.
    - `net_train_lstm_b1_t64_n512_full`은 실행 오류로 실패했다.
    - 이어진 `_ckpt` 케이스에서 wgpu가 panic한 뒤 프로세스가 강제 종료(137)됐고, 리포트 파일이 생기지 않았다. 표의 값은 실행 로그에서 옮겼다.
    - 엔티티 512개 장면에서 2 GiB 할당이 실패하는 것은 이미 알려진 Linux Vulkan 호스트의 한계다(`docs/commands.md`, 학습기는 041-R17 조각 재계산으로 피한다). 벤치에는 그 설정이 없다.
  - 22:2x: 빠진 엔티티 인코더 케이스만 `--filter net_entity`로 다시 쟀다.

### 결과: 전체 모델 (ms)

| 케이스 | Burn Vulkan | torch faithful | torch fast | torch fast-bf16 | Burn ÷ torch (faithful / fast) |
|---|---|---|---|---|---|
| 학습 LSTM B2·T16·N256 | 1,075.1 | 746.2 | 293.3 | 1,040.4 | 1.4배 / 3.7배 |
| 학습 LSTM B4·T16·N256 | 2,231.5 | 1,453.8 | 560.8 | 2,125.7 | 1.5배 / 4.0배 |
| 학습 GTrXL B2·T16·N256 | 911.3 | 756.1 | 296.3 | 1,045.3 | 1.2배 / 3.1배 |
| 학습 LSTM B1·T64·N512 | 실패(메모리 할당) | 4,820.0 | 1,225.7 | 3,467.0 | – |
| 학습 GTrXL B1·T64·N512 (본학습 레인 모양) | Burn 벤치에 없음 | 4,810.2 | 1,168.9 | 3,384.8 | – |
| 추론 act LSTM B16 | 179.1 | 40.2 | 37.8 | 123.9 | 4.5배 / 4.7배 |
| 추론 act GTrXL B16 | 187.7 | 48.3 | 46.2 | 133.0 | 3.9배 / 4.1배 |
| 엔티티 인코더 fwd / fwd+bwd B16 | 35.6 / 109.1 | 23.7 / 303.7 | 21.4 / 79.9 | 21.6 / 158.8 | fwd+bwd 0.36배 / 1.4배 |

- 학습 최대 메모리(PyTorch, B1·T64·N512): faithful 7.5 GiB, fast 5.8 GiB, bf16 5.4 GiB. 셋업 기록상 이 장비가 쓸 수 있는 GPU 메모리는 약 13.1 GiB다.
- 추론 p95는 중앙값과 거의 같았다(예: faithful LSTM 40.2 / 40.4ms). 학습 PC보다 오히려 고르다.

### 결과: 커널 (ms)

| 케이스 | Burn Vulkan fwd / fwd+bwd | torch ROCm fwd / fwd+bwd | 비고 |
|---|---|---|---|
| matmul f32 4096² | 43.1 | 27.1 | torch 1.6배 (3.2 대 5.1 TFLOP/s) |
| matmul f32 1024² | 1.91 | 1.65 | 1.2배 |
| matmul f16 4096² | 23.9 | 28.3 | **Burn이 1.2배 빠름** (5.7 대 4.8 TFLOP/s). 이 보드의 rocBLAS f16은 f32보다 빠르지 않다 |
| conv 3×3 [16,58,128²] | 10.84 / 38.14 | 0.87 / 13.38 | forward 12배, 전체 2.9배 |
| conv transpose 8→128 | 5.03 / 66.81 | 1.42 / 23.27 | 2.9~3.5배 |
| LSTM B32·T64 (Burn 기본 `Lstm` 대 torch 자체 커널) | 75.7 / 114.0 | 11.6 / 66.4 | forward 6.5배, 전체 1.7배 |
| transformer N256 | 20.4 / 55.7 | 14.3 / 43.7 | 1.3~1.4배 |
| transformer N512 | 44.6 / 120.3 | 36.3 / 108.2 | 1.1~1.2배 |
| 거리 bias attention (테이블 `select`) | 3.02 / 302.7 | 1.61 / 57.1 | backward 5.3배 |
| scatter-add 128² | 0.471 / 1.012 | 0.288 / 0.410 | 1.6~2.5배 |
| pointer GRU 12스텝 | 4.95 / 16.6 | 3.08 / 11.0 | 1.5배 |

### 해석
- **bc250_1에서 PyTorch(ROCm) fast 모드는 Burn Vulkan보다 학습이 3~4배, 추론이 4~4.7배 빠르다.**
- **반면 코드를 그대로 옮긴 faithful 모드의 학습 이득은 1.2~1.5배로 작다.** 학습 PC(2~2.6배)보다 훨씬 작다.
  - 원인은 faithful 엔티티 인코더의 backward다(303.7ms로 Burn의 109.1ms보다 2.8배 느림).
  - faithful은 거리 bias를 one-hot @ table로 계산한다(032-R13). 이 [B·N·N, 32] 행렬의 backward가 이 보드의 ROCm에서 느리다. embedding 조회로 바꾼 fast 모드는 79.9ms다.
  - 즉 Burn이 scatter-add를 피하려고 쓴 방식이 ROCm PyTorch에는 맞지 않는다.
- **bf16 AMP는 이 보드에서 역효과다.** 학습이 fast 대비 2.9~3.8배, 추론이 3.3배 느려져 faithful보다도 느리다. 셋업 기록의 fp16 결과와 같은 경향이다.
- 장비 간 비교(같은 PyTorch fast, B1·T64·N512 LSTM): bc250_1 1,225.7ms 대 학습 PC 204.3ms로 약 6배 차이다. Burn끼리(LSTM B2·T16)는 bc250_1 Vulkan 1,075.1ms 대 학습 PC CUDA 164.6ms로 약 6.5배다.
- 이 장비의 ROCm 스택은 트리 밖 커널 패치와 직접 빌드한 라이브러리로 돌아가며 안정성이 보장되지 않는다(셋업 기록: 드문 page fault, GPU 리셋 불가, 24시간 이상 검증 없음). 이번 약 30분 동안 GPU 오류는 없었다.

## 산출물
- 메인 작업 사본 `reports/raw/torch_vs_burn_20261023/`(추적하지 않는 폴더)
  - PyTorch 이식본과 벤치: `uranium_torch/`, `bench.py`, `README.md`
  - 측정 원본: `results_full.json`(Windows), `results_wsl.json`(WSL), `burn_bench_cuda_win.md`, `burn_bench_cuda_wsl.md`, 실행 로그 `windows_run_log.txt`, `wsl_burn_run_log.txt`
  - WSL 준비 스크립트(sudo 없이 CUDA redist, rustup, uv 설치): `wsl_setup.sh`, `wsl_burn.sh`
  - bc250_1(`bc250_1/` 하위 폴더)
    - ROCm용으로 고친 코드: `bench.py`, `uranium_torch/model.py`(LSTM에서 MIOpen 끔), `run_rocm.sh`(실행 환경), `check_rocm.py`(CPU·GPU 대조), `README.md`
    - 측정 원본: `results_rocm.json`, `quick.json`, 전체 실행 로그 `bench_rocm.log`(PyTorch·Burn·사후 검사·커널 로그), `burn_bench_vulkan_bc250_log.txt`, `burn_bench_vulkan_bc250_entity.md`
  - bc250_1의 `~/Workspace/uranium-torch-bench`에도 같은 파일이 남아 있다.
- 측정에 쓴 가상환경, WSL 툴체인, 저장소 복제본은 측정 뒤 지웠다. 위 스크립트로 다시 만들 수 있다.

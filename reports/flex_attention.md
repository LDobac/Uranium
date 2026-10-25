# 엔티티 attention: FlexAttention 대 직접 작성 커널, Windows 대 WSL (2026-10-25)

## 0. 배경과 예상
- 사용자 질문: "우리 모델에 FlashAttention을 시도한 흔적이 있나" → "WSL에서는 가능한가" → "WSL에서 FlexAttention 측정해봐".
- 지금까지의 사정(`reports/pytorch_opt.md` 0.1·2.12절)
  - PyTorch SDPA flash 백엔드는 `attn_mask`를 받지 않는다. 학습하는 거리 bias(032 §2.1, 032-R13)와 키 마스크가 있는 엔티티 attention에는 쓸 수 없다. Windows용 PyTorch에는 flash 커널 자체가 없다.
  - FlashAttention 3·4는 Hopper·Blackwell 전용이라 학습 PC(Ada)에서 돌지 않는다. bc250(gfx1013)은 어느 버전도 지원하지 않는다.
  - 그래서 bias를 커널 안에서 읽는 attention 커널(`python/uranium/kernels/attn_bias.cu`, 060-R1)을 직접 만들어 두 장비 기본값으로 쓰고 있다(`URANIUM_ENTITY_ATTN=fused`).
- 이번 후보: PyTorch FlexAttention(`torch.nn.attention.flex_attention`)
  - `score_mod`로 `table[bucket[b,i,j], h]` bias와 마스크를 넣을 수 있고, 표 gradient도 계산한다. Triton이 필요하다.
  - 측정 전에는 Triton이 Windows에서 안 돌 줄 알았다. 실제로는 학습 PC venv에 triton-windows가 있어서 Windows와 WSL을 모두 쟀다.
- 예상: FlexAttention은 범용 Triton 커널이라, 이 모양(head dim 32, bucket bias)에 맞춰 짠 직접 작성 커널보다 빠르기는 어렵다. 다만 compile된 PyTorch 기준 구현보다는 빠를 것이다.
- 사용자 지시
  - 측정 중 학습 PC 작업자 정지는 작업자 관리 세션에 요청한다.
  - 측정 뒤에는 빠른 쪽에서 재개한다. 오차 범위 안이면 Windows에서 재개한다.

## 1. 방법
- 장비: 학습 PC(RTX 4070 Ti SUPER 16GB). Windows 11과 WSL2 Ubuntu를 같은 GPU에서 차례로 쟀다. 측정 중 다른 GPU 작업은 없었다.
- 버전: 두 쪽 모두 torch 2.14.1+cu130, Python 3.14. Windows는 triton-windows, WSL은 triton 3.8.0.
- 케이스: attention 한 층의 fwd+bwd(`out.backward(dout)`, q·k·v·bias 표 gradient)
  - head 8 × dim 32, bucket 32개, 키 마스크는 행마다 N의 40~100%
  - 모양(행 × N): 64×128, 64×256, 64×512, 256×128, 256×256. bf16과 f32
  - TF32 matmul
- 후보
  - `torch_eager`: `kernels.attention.reference`와 같은 수식
  - `torch_compile`: 같은 함수에 `torch.compile`
  - `flex_scoremod`: compile한 FlexAttention, bias와 마스크(`-1e9`)를 `score_mod`에 넣음. 수식이 완전히 같다.
  - `flex_blockmask`: 위와 같고, 패딩 키 블록을 `block_mask`로 건너뜀
  - `fused_kernel`: `uranium.kernels.attention.fused`
- 시간: 워밍업 3회 뒤 20회의 CUDA event 중앙값
- 오차: f32 기준 구현 대비 상대 오차
- 재현성: 같은 입력으로 두 번 돌려 gradient가 비트 단위로 같은지 비교
- 스크립트와 원본: `reports/raw/flex_attention_20261025/`(CUDA 호스트 메인 작업 사본, git 미추적)
  - `flex_probe.py`, `det_probe.py`, `flex_{win,wsl}.{json,log}`

## 2. 진행 기록
- 13:04 작업자 관리 세션이 학습 PC 작업자를 정지했다(`arch-abl-terrain-pt` ckpt-7463).
- 13:0x Windows 1차 실행은 무효였다.
  - 원인 ① fused 커널 로드 실패. 이 세션의 도구 셸에서 vcvars 출력 디코드가 실패했다(`'cp1' codec`). 일반 콘솔에서 도는 작업자에서는 생기지 않는다.
  - 원인 ② dynamo 재컴파일 한도(8)에 걸렸다. 그 뒤 FlexAttention이 융합되지 않은 경로로 돌았다.
  - 조치: vcvars 환경에서 실행하고, 재컴파일 한도를 올려 다시 쟀다.
- 13:0x~13:3x WSL 준비(별도 에이전트)
  - 저장소 복제(`~/Uranium`), venv, sudo 없이 CUDA 13.4 redist 설치, Rust 확장, `uranium-train` 빌드
  - WSL에서 fused 커널이 빌드되고 `pytest -m gpu tests/python/test_kernels.py` 4개를 통과했다.
  - BC 스모크는 데이터 pull(39 GB, 약 6분)과 첫 시작 검사(553초)를 지나 compile 단계에서 중단했다. 사용자가 오래 걸린다고 해서 측정을 먼저 했다. 그래서 update 시간은 없다.
- 13:3x Windows 측정(80초), 이어서 WSL 측정(89초)

## 3. 결과 (ms, Windows / WSL)
- bf16

  | 행×N | torch eager | torch compile | flex score_mod | flex block_mask | **fused 커널** |
  |---|---|---|---|---|---|
  | 64×128 | 7.2 / 7.3 | 9.6 / 3.4 | 1.9 / 3.8 | 3.2 / 4.2 | **0.8 / 3.5** |
  | 64×256 | 33.4 / 33.4 | 12.5 / 7.0 | 6.2 / 7.4 | 10.5 / 7.3 | **2.3 / 2.4** |
  | 64×512 | 148.3 / 148.4 | 49.5 / 27.3 | 46.9 / 24.6 | 40.2 / 21.8 | **8.1 / 8.1** |
  | 256×128 | 33.6 / 33.4 | 7.2 / 7.1 | 11.8 / 7.9 | 11.8 / 14.3 | **2.9 / 3.0** |
  | 256×256 | 149.7 / 149.8 | 49.9 / 27.8 | 24.9 / 46.8 | 22.9 / 23.5 | **8.9 / 8.9** |

- f32

  | 행×N | torch eager | torch compile | flex score_mod | flex block_mask | **fused 커널** |
  |---|---|---|---|---|---|
  | 64×128 | 7.5 / 7.6 | 2.0 / 2.0 | 1.9 / 1.9 | 4.0 / 2.1 | 3.2 / **1.6** |
  | 64×256 | 36.2 / 35.9 | 13.6 / 8.2 | 8.0 / 6.5 | 8.1 / 5.6 | **2.7 / 2.4** |
  | 64×512 | 159.7 / 157.8 | 54.1 / 31.5 | 47.2 / 24.8 | 40.3 / 22.1 | **8.2 / 8.2** |
  | 256×128 | 36.3 / 36.2 | 13.9 / 8.3 | 10.5 / 6.9 | 10.6 / 6.5 | **3.1 / 2.9** |
  | 256×256 | 159.9 / 158.8 | 32.4 / 32.6 | 25.7 / 25.6 | 23.8 / 23.2 | **9.0 / 9.0** |

- 오차(f32 기준 구현 대비 상대 오차)
  - bf16: fused 1.7e-3, flex 2.1~2.5e-3, torch compile 4e-3, torch eager 6~8e-3
  - f32: fused 4~5e-4, flex 1.4~2.3e-3, compile 1e-5 미만
- 메모리: fused와 flex는 [행, H, N, N] 텐서를 만들지 않는다. 64×512 bf16의 최대 증가량은 fused 128 MiB, flex 274 MiB, eager 1.8 GiB다.
- 64×128 칸은 1~4 ms 수준이라 호출·실행 순서의 잡음이 크다. 같은 커널이 장치마다 0.8 / 3.5 ms로 나왔다.

## 4. 해석과 결정
- **직접 작성 커널이 모든 실측 모양에서 가장 빠르다.** N256 이상에서 FlexAttention보다 2.5~5배, compile한 PyTorch보다 3~6배 빠르다. FlexAttention은 교체 후보가 못 된다. 학습 설정은 바꾸지 않는다.
- **Windows와 WSL의 차이는 오차 범위 안이다.** 학습이 쓰는 fused 커널은 8.10 / 8.07, 8.89 / 8.90 ms로 같다.
  - WSL이 빠른 경우는 compile·flex(Triton) 칸뿐이다(약 1.8배). 이 경로는 학습에 쓰지 않는다.
  - 사용자 기준에 따라 학습 PC 작업자는 **Windows에서 재개**한다(작업자 관리 세션에 요청).
- **발견 1: fused 커널의 bias 표 gradient는 실행마다 비트 단위로 다르다.** 차이는 최대 1.5e-5다.
  - 출력과 dq·dk·dv는 비트 단위로 같다.
  - 블록 안에서 shared memory atomic으로 더하는 순서가 매번 다르기 때문으로 보인다. 블록 부분합을 더하는 순서는 고정이다.
  - GPU에서는 비트 일치를 요구하지 않으므로(CLAUDE.md "비트 일치는 CPU에서만") 학습에는 문제가 없다.
  - 다만 `attn_bias.cu` 머리 주석과 `pytorch_opt.md` 2.12절의 "결정론적"이라는 설명은 맞지 않는다. 고칠지는 사용자 확인 사항이다.
- **발견 2: 커널 로더는 vcvars 출력을 콘솔 코드페이지로 디코드한다.** 코드페이지가 비정상인 셸(이 세션의 도구 셸)에서는 빌드가 실패하고 기준 구현으로 넘어간다. 일반 콘솔에서 뜨는 작업자에는 영향이 없다.
- **발견 3: compile·flex 칸의 gradient는 실행마다 다르다.** torch compile, flex 모두 같다. 표 gradient를 atomic으로 합산하기 때문이다.
- WSL 학습 환경(다시 쓸 때)
  - 위치: `~/Uranium`(origin/master 복제), `~/uranium-env.sh`(먼저 source), `~/cuda-13.4.1`, 데이터 캐시 `~/uranium-cache`
  - `.env`
    - 공용 드라이브는 `/mnt/c/...`로 잡고 `URANIUM_SHARED_LOCAL=0`(데이터는 ext4 캐시로 pull)
    - `URANIUM_HOST_NAME`은 Windows와 다른 이름. 대소문자만 다른 호스트 이름이 대소문자를 구분하지 않는 공용 드라이브에서 같은 파일이 되기 때문이다.
  - 아직 확인하지 않은 것: end-to-end update 시간
  - 주의: WSL에 보이는 RAM은 Windows의 절반(15.6 GB)이다. `URANIUM_BC_PREFETCH=2`는 메모리가 빠듯할 수 있다.

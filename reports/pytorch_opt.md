# PyTorch 2단계 최적화 (AMB-000-4)

## 0. 배경과 예상
- 1단계(기능 이식)는 10-24에 끝났다(`reports/pytorch_port.md`). 실데이터 BC update(GTrXL, 32레인 × T64)는 bc250_1 ROCm 약 20초, 학습 PC CUDA 약 11초(장치 시간은 약 2.5초뿐)였다.
- 사용자 지시
  - 속도 최우선, 모델 구조와 결과는 유지(같은 수식만)
  - PyTorch 기법은 플랫폼 전용 코드까지 모두 허용하되 CUDA PC와 bc250 둘 다 돌아야 한다.
  - 후보가 여럿이면 유효한 수치가 나올 때까지 실측해 나은 쪽을 고른다.
- 계획: 10-24 승인된 2단계 계획(2-0 준비 → 기법 1~11, 장비별 후보 실측 비교). 측정 도구는 `python -m uranium bench`(060-R2).
- 예상
  - CUDA: launch 부담이 병목이라 레인 그룹 키우기·그래프 캡처가 큰 이득
  - ROCm: 위치 헤드 transposed conv의 새 모양 비용이 병목이라 행 수 고정이 첫 과녁

### 0.1 계획 단계 실측 (10-24, 워밍업 후 5회 평균)
- 엔티티 attention + 학습하는 거리 bias(gradient) + 마스크, fwd+bwd

  | 장비 | 정밀도 | 지금 구현 | SDPA mem-efficient | SDPA cuDNN·flash | SDPA math |
  |---|---|---|---|---|---|
  | CUDA | f32 | 11.0 ms | 9.0 ms | 커널 없음 | 11.0 ms |
  | CUDA | bf16 | 8.0 ms | 6.4 ms | 커널 없음 | 11.3 ms |
  | bc250 | f32 | 51.0 ms | 커널 없음 | 커널 없음 | 51.2 ms |

  - CUDA PC의 Windows용 PyTorch에는 flash attention이 없고, cuDNN은 gradient가 필요한 bias를 받지 않는다. bc250(gfx1013)에는 fused 백엔드가 없다(AOTriton·CK 미지원, 웹 조사와 일치).

- 위치 헤드 transposed conv 4층 fwd+bwd(행 24 / 40)

  | 장비 | vendor deconv | 기본 구현 | 3×3 sub-pixel conv + pixel_shuffle(같은 수식) |
  |---|---|---|---|
  | CUDA | 5.0 / 8.1 ms | 9.3 / 18.6 | 13.0 / 15.8 |
  | bc250 | 32.6 / 50.1 ms | 58.9 / 98.5 | 35.6 / 58.0 |

  - sub-pixel 형태는 두 장비 모두 져서 후보에서 뺐다.

## 1. 기준선 (master bfa0a02, 최적화 전)
### 1.1 bc250_1 ROCm f32 (`reports/bench_rocm.md`)
- 실데이터 BC update: 그룹 1은 약 19.5초 + 데이터 대기 약 4초(prefetch 0). 그룹 2·4는 새 모양마다 MIOpen 첫 호출 비용으로 update가 21~397초로 들쭉날쭉했다.
  - 모양이 캐시된 update에서도 그룹 2(21초)가 그룹 1(19초)보다 빠르지 않았다. 이 장비는 launch 부담보다 계산이 지배한다.
  - 행 수 고정 없이는 그룹 크기를 키울 수 없다.
- 연산별 장치 시간(update 하나, 합 약 19초)

  | 연산 | 장치 시간 |
  |---|---|
  | `convolution_backward`(704회) | 7.2초 |
  | `mm` + `addmm`(Linear) | 5.3초 |
  | `embedding_dense_backward`(거리 bias·상대위치 표 gradient 등) | 1.6초 |
  | `bmm` | 0.8초 |
  | `miopen_convolution_transpose`(위치 헤드 forward) | 0.1초 |

  - `copy_`는 CPU 시간 7.8초다(SDMA를 꺼서 동기 복사).
  - 1단계에서 지목한 transposed conv는 모양이 캐시되면 작다. **진짜 과녁은 MIOpen conv backward 전반**이다.
- 공간 인코더 + 위치 헤드 conv만(64행 그룹 하나, 실제 모듈): MIOpen 344ms, 기본 구현 905ms. update당 32그룹이면 약 11초로 update의 절반 이상이다.
  - 모양이 고정된 상태에서 MIOpen 탐색 모드(`cudnn.benchmark`)를 켜도 344ms로 같다. gfx1013에서는 해법 선택지가 사실상 하나다.
- 층별 conv backward(64행)

  | 층 | backward | 비고 |
  |---|---|---|
  | stem 3×3, 58→32 @128² | 68.2 ms | forward 2.7ms의 25배 |
  | **위치 헤드 1×1, 32→1 @128²** | **60.6 ms** | 계산량 약 0.07 GFLOP인데 MIOpen 해법이 나쁨 |
  | 위치 헤드 transposed 128→32 @64² | 45.9 ms | |
  | down 3×3 s2, 32→64 @128² | 36.1 ms | |
  | ResBlock 3×3, 64 @64² (2회) | 27.8 ms | |

- 벤치 스위트: CPU 대비 오차는 모두 1e-6 수준이다.
  - 최악 모양 학습(B1·T64·N512, 엔티티 재계산 16): GPU 최대 1.84 GiB, 1.7초
  - `net_act_gtrxl` B16·N256: 41.0ms(032-AC2 40ms 바로 위)
  - 추론은 B1에서도 약 22ms라 고정 부담이 크다.

### 1.2 학습 PC CUDA bf16
- 측정 대기(이 PC를 써도 될 때 잰다). 코드 확인용 짧은 스모크(케이스당 0.5초)에서는 추론 `act`가 B1·N128 약 24ms, B32·N512 약 49ms로 크기에 거의 비례하지 않았다. 추론도 launch 부담이 지배한다는 신호다.

## 2. 후보 비교 기록
### 2.1 위치 헤드 1×1 conv(32→1) — bc250
같은 입력·가중치, 64행 @128², fwd+bwd 중앙값(8회), 오차는 출력 절대 / gradient 상대 최대

| 후보 | bc250 | 오차 |
|---|---|---|
| MIOpen conv(지금) | 62.5 ms | 기준 |
| **곱한 뒤 채널 합(`(x*w).sum(1)+b`)** | **3.6 ms** | 8.8e-6 |
| einsum | 246.6 ms | 3.8e-5 |
| channels-last matmul | 167.3 ms | 3.8e-5 |
| channel-first bmm | 167.3 ms | 3.8e-5 |

- bc250에서는 곱-합 형태가 17배 빠르다(update당 약 1.9초 절감 예상). CUDA 비교 뒤 장치별로 고른다.

### 2.2 3×3 conv를 unfold + 행렬곱으로 — bc250
| 층 | MIOpen | unfold+matmul |
|---|---|---|
| stem 58→32 @128², 16행 | 14.1 ms | 131.9 ms |
| ResBlock 64 @64², 64행 | 16.4 ms | 141.8 ms |
| down 32→64 s2 @128², 32행 | 19.8 ms | 50.3 ms |

- unfold·fold의 메모리 이동이 커서 MIOpen보다 3~9배 느리다. 제외한다.

### 2.3 위치 헤드 사용 행·1×1 출력·전송 방식 — bc250 실데이터
`bench bc --groups 1 --updates 12`, personal shard, 같은 seed·같은 batch. update 시간은 데이터 대기 제외. 모든 후보의 마지막 손실이 17.49563으로 같았다(같은 수식 확인).

| 후보 (`URANIUM_LOC_ROWS` / `URANIUM_LOC_OUT` / `URANIUM_H2D`) | 중앙값 (p95) |
|---|---|
| off / conv / sync (지금) | 18.90 (19.50), 재측정 18.74 (19.37) — 실행 간 잡음 약 0.2초 |
| exact / conv | 18.65 (19.26) |
| mult8 / conv | 19.00 (19.71) |
| pow2 / conv | 19.70 (20.38) |
| pow2 / mulsum | 18.45 (19.25) |
| **exact / mulsum** | **18.09 (18.84)** |
| exact / mulsum / pinned | 17.98 (18.71) |
| 그룹 4, pow2 / mulsum | 22.94 (157.05) |

- bc250 선택: `exact` + `mulsum`(약 0.7초, 3.5%).
  - 행 버킷은 남는 행만큼 계산이 늘어 손해다. MIOpen의 새 모양 비용은 디스크 캐시로 한 번만 든다.
  - pinned는 잡음 수준이라 쓰지 않는다. SDMA를 꺼서 복사가 계산 큐에서 직렬로 돈다.
- 그룹 4는 모양이 캐시된 update도 그룹 1보다 느리다. bc250은 계산이 지배해 그룹 확대 이득이 없다(AMB-041-7).

### 2.4 모듈별 시간 — 64행 그룹 하나, 실제 모듈, fwd+bwd
| 모듈 | bc250 N192 | bc250 N512 | CUDA bf16 N192 |
|---|---|---|---|
| 엔티티 인코더(bc250은 재계산 16) | 292.7 ms | 1419.3 ms | 26.1 ms |
| 공간 인코더 | 206.0 | 209.5 | 20.6 |
| GTrXL 코어 | 11.6 | 11.6 | 10.8 |
| 헤드(teacher forcing) | 49.7 | 62.1 | 32.1 |
| 전체 | 552.3 | 1703.5 | 76.8 |

- bc250은 엔티티 인코더가 최대 과녁이다(N²).
  - 연산별(N192): 재계산 16에서 355ms, 재계산 끔에서 232ms
  - `embedding_dense_backward`(거리 bias 표 gradient)가 약 72ms
  - Linear `mm`이 71ms
- CUDA는 모델 계산이 그룹당 77ms(×32 = 2.5초)뿐이다. 실제 update 약 11초의 나머지는 모델 바깥(동기 복사·collate·동기화)에 있다. 실데이터 phase 측정으로 가린다.

### 2.5 거리 bias 표 gradient — 탈락
작은 시험(무작위 bucket, 64행)

| 방법 | CUDA N192 / N512 | bc250 N192 / N512 |
|---|---|---|
| embedding backward(지금) | 2.3 / 15.7 ms | 66.7 / 490.1 ms |
| index_add | 3.4 / 22.9 | 15.0 / 108.8 |
| one-hot 행렬곱(조각) | 3.9 / 27.6 | 543 / 3858 |

- 작은 시험에서는 bc250 `index_add`가 4.5배 빨랐다. 그런데 실제 update에 넣자 **24.35초로 오히려 6초 느려졌다**. 최악 모양 학습도 1.71초에서 3.54초가 됐다.
- 원인(추정): 실제 거리 bucket은 몇 개 값에 몰려 있어 atomic 누적이 같은 칸에서 경합한다. embedding backward는 정렬로 이를 피한다.
- 교훈: 후보는 실제 분포의 입력으로, 최종적으로는 실데이터 update로 비교한다.
- `URANIUM_BIAS_GRAD=index_add`는 끄는 설정으로만 남긴다(기본 embedding). 다음 후보는 LDS 블록 합산 커널(경합을 블록 안에서 먼저 줄임) 또는 attention 융합 커널이다.

### 2.6 엔티티 재계산(041-R17)과 메모리 — bc250
| 설정 | update 중앙값 | GPU 최대 | 상주 메모리 최대 |
|---|---|---|---|
| 재계산 16 (+index_add) | 24.35초 | 1.78 GiB | 9.99 GiB |
| 재계산 끔 (+index_add) | 21.62초 | 2.87 GiB | 9.57 GiB |
| 최악 모양 B1·T64·N512, 재계산 16 / 끔 | 3.54 / 3.39초 | 1.75 / **5.70 GiB** | – |

- 재계산을 끄면 약 11% 빠르다. 대신 최악 모양 GPU 메모리가 3.3배다.
- bc250에서는 GPU 할당이 상주 메모리 수치에 섞이지 않는다(시험: GPU 4GiB 할당 시 상주 메모리 +0.07GiB). 따라서 **상주 메모리 약 10GiB는 진짜 호스트 메모리**이고, GPU 할당과 합치면 RAM 14GB에 가깝다.
- Feeder만 돌린 시험(학습 없음, prefetch 0)
  - batch 하나 4.35GB, 이전 batch를 쥔 채 다음 batch를 만들면 7.7~8.1GB
  - batch 압축(041-R18)을 켜도 거의 같다(4.35 대 4.44GB). 압축은 레인마다 바로 하므로, 해제한 메모리를 glibc 할당자가 OS에 돌려주지 않는 것이 원인이다(큰 블록 기준이 해제 뒤 스스로 올라감).
  - `GLIBC_TUNABLES=glibc.malloc.mmap_threshold=1048576`(결과 무관)로 **8.2GB → 2.4GB**. 대신 batch 생성이 batch당 약 2초 느려진다(3.7 → 5.9초).
  - 생긴 여유로 prefetch(데이터 대기 숨기기)와 재계산 끄기를 함께 시험한다(2.7절).

### 2.7 호스트 메모리 설정 + 미리 읽기 + 재계산 — bc250 실데이터
`exact` + `mulsum`, 할당자 기준 1 MiB(`URANIUM_MALLOC_MMAP_THRESHOLD`, 시작 시 mallopt. 환경변수 방식과 같은 2.43GB 확인). 그룹 1, 12 update.

| 조합 | 학습 | 데이터 대기 | 합계 | GPU 최대 | 상주 최대 |
|---|---|---|---|---|---|
| 기준선(최적화 전, prefetch 0, 재계산 16) | 19.5초 | 4.0초 | 23.5초 | 1.78 GiB | 10.0 GiB |
| A: prefetch 0, 재계산 16 | 18.36 | 5.73 | 24.1 | 1.78 | 4.52 |
| B: prefetch 2, 재계산 16 | 18.36 | 0.02 | 18.4 | 1.78 | 6.29 |
| C: prefetch 2, 재계산 끔 | 15.83 | 0.02 | 15.85 | 2.87 | 6.21 |

- 할당자 설정은 batch 생성을 늦추지만(대기 4.0 → 5.7초), prefetch 2가 그 대기를 학습 뒤로 완전히 숨긴다.
- 재계산을 끄면 최악 모양 GPU가 5.70 GiB다. 상주 6.3 GiB와 더하면 약 12 GB(RAM 14 GB)다. 060-R3 한도·AMB-041-8 결정 대상이다.
- C의 손실(17.47318)만 다른 것은 재계산 유무에 따른 dropout 난수열 차이다(041-R17).

### 2.8 거리 bias 배치 — 두 장비
엔티티 인코더 fwd+bwd, 64행, 재계산 끔, 무작위 입력

| 배치 | CUDA bf16 N192 / N512 | bc250 N192 / N512 |
|---|---|---|
| 지금(embedding 후 축 바꾼 view) | 25.5 / 151.4 ms | 216.2 / 1129.2 ms |
| 축 바꾼 뒤 연속 복사 | 26.1 / 157.4 | 215.9 / 1138.7 |
| **전치 표에서 꺼내기(`table_t`)** | 93.1 / 702.7 | **171.0 / 871.7** |

실데이터 확인(bc250, B·C 설정에 더함): 재계산 16에서 18.36 → **17.47초**, 재계산 끔에서 15.83 → **14.89초**이고 손실은 같다.
- bc250 기본값으로 채택한다(`device.IMPL_DEFAULTS`).
- CUDA는 지금 방식을 유지한다.

### 2.9 bc250 중간 정리 (10-24 08:5x)
| | update 합계 | 배율 |
|---|---|---|
| 기준선 | 23.5초 | 1.00 |
| 구현 선택(`exact`·`mulsum`·`table_t`) + prefetch 2 + 할당자 설정, 재계산 16 | 17.5초 | 1.35 |
| 같은 조합, 재계산 끔 | 14.9초 | 1.58 |

- prefetch 2, 할당자 설정, 재계산 끄기는 호스트 `.env` 설정이다. 060-R3 한도·AMB-041-8 결정 뒤 bc250 `.env`에 넣는다.
- 손실 곡선은 같은 설정끼리 소수 다섯째 자리까지 같다. 재계산 유무만 dropout 난수열이 다르다.

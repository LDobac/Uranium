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

# BC GPU 메모리 (2026-10-17)

BC 기본 설정(레인 32 × T64)이 16GB GPU에서 메모리 부족을 일으키던 문제를 다룬다. 레인 2에서도 OOM이 난다는 보고가 있었다.

## 원인

- 한 번의 forward/backward에 `레인 × T` 행이 함께 들어간다. 레인 2도 128행이다.
- 엔티티는 배치 최대 N으로 패딩된다(030-R7). 실제 게임은 N_max=512에 자주 닿는다.
- 엔티티 attention(8 heads × N²)과 distance bias one-hot `[N², 32]`이 행마다 수십 MB를 쓴다. 메모리는 `레인 × T × N²`에 비례한다.

## 변경 (041-R3)

| 항목 | 내용 |
|---|---|
| `micro_lanes` (`config/bc.toml`, 기본 1) | 레인을 이 수만큼 묶어 forward/backward를 하고, gradient를 합쳐 AdamW step을 한 번 한다. 손실은 배치 전체 valid 수로 나누므로 레인 32를 한 번에 처리한 것과 같은 업데이트가 된다(`micro_lanes_match_full_batch_041_r3`). 최대 메모리는 `lanes`가 아니라 그룹 크기를 따른다. Core 상태는 그룹별로 보관하고, 저장할 때 detach한다. |
| `pad_entities` (기본 64) | 배치 N을 64의 배수로 올린다. 마스크된 패딩이라 결과는 같다(같은 테스트). |
| `grad_checkpointing` (기본 false) | Burn balanced gradient checkpointing을 켜고 끈다. |
| bench | `net_train_lstm_b{1,2}_t64_n512_{full,ckpt}`: 최악 조건 N=512 학습 스텝 |

## 측정 (RTX 4070 Ti SUPER 16GB, CUDA)

합성 입력(N=512 고정):

| case | 업데이트당 | 샘플/초 | 결과 |
|---|---|---|---|
| 레인 1 × T64, checkpointing 없음 | 1.62 s | 39.5 | 정상 |
| 레인 1 × T64, checkpointing | 1.56 s | 41.1 | 정상 |
| 레인 2 × T64, checkpointing 없음 | 1.96 s | 65.3 | 정상 |

STARDATA 표본 13게임(1/300 분할, schema 3):

| 설정 | 업데이트 | 시간 | 결과 |
|---|---|---|---|
| 기존 동작: 레인 2 한 번에, 패딩 없음 | 150 | 365 s (약 53 샘플/초) | 정상 |
| 레인 8, `micro_lanes=1`, `pad_entities=0` | 35 | 330 s | 정상 |
| 레인 8, `micro_lanes=1`, `pad_entities=64` | 35 | 301 s (약 9% 빠름) | 정상 |

- 호스트 RAM 최대는 약 1.9GB다.
- `nvidia-smi` 사용량은 모든 경우 14.6–15.9GB였다. CubeCL 메모리 풀이 가용 메모리를 미리 잡아 두므로, 이 값으로는 실제 필요량을 구분할 수 없다.

## 한계와 남은 것

- 이 PC·CUDA·표본에서는 레인 2 OOM이 재현되지 않았다. 보고된 OOM의 환경(백엔드 Vulkan/CUDA, PC, 오류 메시지, 설정)을 확인해야 한다.
- distance bias 메모리 절감과 f16 혼합 정밀도(032-R8)는 하지 않았다. 레인 1 × N=512가 동작하므로 메모리 근거가 없다. Burn 0.22에는 AMP(f32 master 가중치·loss scaling)가 없어서 f16은 별도 작업이 필요하다. 주로 처리량 개선 항목이다.
- 레인 32 실학습의 업데이트당 시간은 레인 수에 비례한다(약 50–60 샘플/초 기준, 업데이트당 약 35–40 s).

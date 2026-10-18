# 060 — Infra: Training Hosts (BC250 · CUDA PC)

- **Status**: Approved (2026-10-15, 2026-10-17 개정: AC2·AC4 정밀도를 032-R8 호스트별 기준에 맞춤, 사용자 결정)
- **Related**: [032](032-model-architecture.md), [042](042-rl-training.md), [020](020-env-orchestration.md), [061](061-shared-storage.md)

## Purpose
학습·추론 호스트의 환경, 예산, 벤치 기준을 정의한다. 학습 호스트는 BC250(Burn Vulkan)과 CUDA GPU PC(Burn CUDA, Linux/Windows)다(2026-10-15 CUDA PC 추가).

## 환경
| 항목 | 값 |
|---|---|
| OS | Linux (Mesa RADV Vulkan 드라이버). Mesa 버전은 M1(a) 통과 버전으로 고정하고 `reports/bench_bc250.md`에 기록 |
| GPU | `AMD BC-250 (RADV GFX1013)` 실측(2026-10-15): shaderFloat16 지원, subgroup 64, DEVICE_LOCAL heap 10.33 GiB, Mesa 26.2.4. 상세는 `reports/vulkaninfo_bc250.txt` |
| GPU API | BC250: Vulkan(Burn `wgpu`). CUDA PC: CUDA(Burn `cuda`). ROCm·HIP는 쓰지 않는다 |
| CUDA PC | Linux 또는 Windows, NVIDIA GPU. 사양·메모리 예산은 준비되면 벤치로 측정(TBD(M5)) |
| Burn | **0.22.0** 고정(`Cargo.toml` `=0.22.0` + `Cargo.lock`). 업그레이드는 spec 변경으로 관리 |
| 메모리 | 16GB 공유. GPU 할당 크기는 BIOS/커널 설정에 따르며 M1(a)에서 실측 |
| CPU | 6C/12T. 추론 서버, env feature 추출, (선택) 로컬 OpenBW에 분배 |

## Requirements
| ID | 요구사항 |
|---|---|
| 060-R1 | **MUST** Burn cargo feature는 `vulkan`(BC250), `cuda`(CUDA PC, 해당 PC에서만 켬), `flex`(CPU 테스트)를 쓴다. 실행 시 `--device vulkan\|cuda:<n>\|flex`로 고른다. 직접 작성한 CUDA·ROCm 코드는 두지 않는다. (변경 이력: ndarray deprecated, 2026-10-15 CUDA 추가) |
| 060-R2 | **MUST** M1(a) 벤치 스위트(`tools/bench`)를 만든다. 측정 항목은 아래 표와 같다 |
| 060-R3 | **MUST** 메모리 예산은 학습 호스트별로 지킨다(아래 표). 초과 시 배치 크기를 자동으로 줄인다 |
| 060-R4 | **MUST** 체크포인트, 리그 스냅샷, 학습 데이터 원본은 **공용 드라이브(SMB)**에 둔다([061](061-shared-storage.md)). 학습 호스트의 로컬 디스크에는 로그와 캐시만 둔다. (변경 이력: '로컬만·백업 없음'에서 2026-10-15 변경) |
| 060-R5 | **SHOULD** 다른 PC GPU를 추론 서버로 쓰는 옵션을 둔다. 추론 서버를 별도 바이너리로 만들고 TCP로 요청을 받으며, Vulkan 또는 CUDA로 동작한다 |
| 060-R6 | **MUST** 장시간 학습 시 GPU 행(hang)이나 드라이버 리셋을 감지하면 공용 드라이브의 `LATEST` 체크포인트 기준으로 자동 재시작한다. Linux는 systemd 서비스 + watchdog, Windows는 서비스 또는 작업 스케줄러를 쓴다 |
| 060-R7 | **MUST** 저장 총량은 **1TB 미만**이다(사용자 결정 2026-10-15). 공용 드라이브와 로컬 캐시 모두 아래 예산을 지키고, 90%를 넘으면 경고한다. RL 자가 대전 리플레이는 오래된 것부터 자동 삭제한다(평가 리플레이는 보존) |

### 060-R2 벤치 항목
- matmul (f32/f16, 1024–4096)
- conv2d
- `Lstm` (B=32, T=64, H=512) forward/backward
- `TransformerEncoder` (N=512, d=256, L=3) forward/backward
- softmax / gather / scatter
- 모델 전용 연산 ([032](032-model-architecture.md)):
  - scatter-add ([B,N,32] → [B,32,128,128])
  - gather (마스크 생성)
  - attention logit bias 주입
  - 12-step GRUCell 루프
  - ConvTranspose2d 8²→128²
  - 128² Conv3x3
  - Gumbel-max 샘플링
- 전체 모델 기준 (§4 기본 구성):
  - `act()` batch 16, N=256
  - `forward_seq` batch 32×T64 forward/backward
- 각 항목의 수치 정확도(Flex CPU 대비)

### 060-R3 메모리 예산 (BC250. CUDA PC는 TBD(M5))
| 용도 | 할당 |
|---|---|
| 학습 | ≤ 8GB GPU 가용 메모리 |
| 추론 서버 | ≤ 1.5GB |
| OS·env·리플레이 버퍼 | 나머지 |

### 060-R7 저장 예산 (초기 배분, 합계 < 1TB. 공용 드라이브 기준, 061-R8)
| 용도 | 예산 |
|---|---|
| BC 데이터셋 (원본 .rep + 중간 데이터) | ≤ 600GB (저장 방식은 [040](040-replay-dataset.md) AMB-040-4) |
| 체크포인트·리그 스냅샷 (공용 드라이브) | ≤ 150GB |
| RL·평가 리플레이 | ≤ 100GB (초과 시 RL 리플레이 순환 삭제) |
| 로그·리포트·기타 | 나머지 |

## Acceptance Criteria
- 060-AC1: 벤치 스위트 결과가 `reports/bench_bc250.md`에 기록된다. 이 결과로 032-R2의 모델 크기와 042의 배치 크기를 확정한다.
- 060-AC2: BC250의 f16 경로가 Flex(CPU) f32 대비 허용 오차(1e-2 상대) 이내이거나, 아니면 f32 전용으로 결정하고 기록한다. CUDA PC는 f32(TF32)가 기본이므로(032-R8, 2026-10-17) f16 판정 대상이 아니다. **닫힘(2026-10-18 사용자 결정)**: BC250의 BC 학습은 f32 전용으로 정했다(032-R8 예외). f16 학습 경로는 구현하지 않았다.
- 060-AC3: 24시간 BC 학습 스모크에서 드라이버 오류 없이 완주하거나, 오류 시 자동 재시작 후 이어서 학습한다.
- 060-AC4: CUDA PC에서 같은 벤치 스위트(`cargo run -p bench --release --features cuda -- --backend cuda`)를 돌려 `reports/bench_<host>.md`에 기록한다. Flex 대비 정확도 기준은 032-AC1(CUDA ≤ 1e-2, AMB-032-5)을 따르고, 정밀도는 f32(TF32)다(032-R8).

## Risks & Open Questions
- BC250 GPU(gfx1013) + RADV에서 Burn/CubeCL 커널 성숙도가 불확실하다(특히 f16, subgroup 연산).
- 체크포인트 쓰기는 원자적으로 한다(061-R4).

- 공유 메모리라 CPU 작업(OpenBW, feature 추출)과 메모리 대역폭을 두고 경합한다. 로컬 OpenBW는 최소화하고 원격 호스트를 우선한다.

# 060 — Infra: Training Hosts (Linux Vulkan · CUDA)

- **Status**: Approved (2026-10-15, 2026-10-17 개정: AC2·AC4 정밀도를 032-R8 호스트별 기준에 맞춤, 사용자 결정. 2026-10-18 개정: R6 감시·재시작 구현 서술(supervise·worker), AC2 닫힘. 2026-10-19 개정(BC 전 리뷰 결정, 사용자 확정): R1 `--backend`, R6 자식 프로세스 그룹 분리(H-4)·자동 시작 문구·`TimeoutStopSec`, R7 RL 리플레이 삭제 제거(042-R11)·경고 기준(C-3), 장비 표기·파일 이름 중립화(001-R10, G-D9))
- **Related**: [032](032-model-architecture.md), [042](042-rl-training.md), [020](020-env-orchestration.md), [061](061-shared-storage.md)

## Purpose
학습·추론 호스트의 환경, 예산, 벤치 기준을 정의한다. 학습 호스트는 Linux Vulkan 호스트(Burn Vulkan)와 CUDA 호스트(Burn CUDA, Linux/Windows)다(2026-10-15 CUDA 추가). 호스트의 장비 종류·사양·상태는 이 spec에 쓰지 않고 리포트(`reports/bench_*.md`, `reports/vulkaninfo_*.txt`)와 추적하지 않는 설정에 둔다(001-R10, 2026-10-19).

## 환경
| 항목 | 값 |
|---|---|
| OS | Vulkan 호스트: Linux (Mesa Vulkan 드라이버). Mesa 버전은 M1(a) 통과 버전으로 고정하고 M1(a) 벤치 리포트에 기록 |
| GPU | 장비 사양(f16 지원, subgroup, 메모리 heap, 드라이버 버전)은 M1(a) 실측 리포트(`reports/vulkaninfo_*.txt`, `reports/bench_*.md`)에 둔다 |
| GPU API | Vulkan 호스트: Vulkan(Burn `wgpu`). CUDA 호스트: CUDA(Burn `cuda`). ROCm·HIP는 쓰지 않는다 |
| CUDA 호스트 | Linux 또는 Windows, CUDA를 지원하는 GPU. 사양·메모리 예산은 벤치로 측정해 리포트에 둔다(TBD(M5)) |
| Burn | **0.22.0** 고정(`Cargo.toml` `=0.22.0` + `Cargo.lock`). 업그레이드는 spec 변경으로 관리 |
| 메모리·CPU | 호스트별 실측값은 리포트에 둔다. 메모리를 CPU와 GPU가 공유하는 호스트는 CPU 작업(추론 서버, env feature 추출, 로컬 OpenBW)과 학습이 메모리를 나눠 쓴다(R3) |

## Requirements
| ID | 요구사항 |
|---|---|
| 060-R1 | **MUST** Burn cargo feature는 `vulkan`(Vulkan 호스트), `cuda`(CUDA 호스트, 해당 호스트에서만 켬), `flex`(CPU 테스트)를 쓴다. 실행 시 `--backend vulkan\|cuda\|flex`로 고른다(GPU 0번). 직접 작성한 CUDA·ROCm 코드는 두지 않는다. (변경 이력: ndarray deprecated, 2026-10-15 CUDA 추가, 2026-10-19 옵션 이름을 실제 `--backend`로 정정) |
| 060-R2 | **MUST** M1(a) 벤치 스위트(`tools/bench`)를 만든다. 측정 항목은 아래 표와 같다 |
| 060-R3 | **MUST** 메모리 예산은 학습 호스트별로 지킨다(아래 표). 초과 시 배치 크기를 자동으로 줄인다 |
| 060-R4 | **MUST** 체크포인트, 리그 스냅샷, 학습 데이터 원본은 **공용 드라이브(SMB)**에 둔다([061](061-shared-storage.md)). 학습 호스트의 로컬 디스크에는 로그와 캐시만 둔다. (변경 이력: '로컬만·백업 없음'에서 2026-10-15 변경) |
| 060-R5 | **SHOULD** 다른 호스트의 GPU를 추론 서버로 쓰는 옵션을 둔다. 추론 서버를 별도 바이너리로 만들고 TCP로 요청을 받으며, Vulkan 또는 CUDA로 동작한다 |
| 060-R6 | **MUST** 장시간 학습 시 학습 프로세스의 비정상 종료나 GPU 행(hang)·드라이버 리셋(진행 없음)을 감지하면 같은 run을 공용 드라이브의 `LATEST` 체크포인트 기준으로 자동 재시작한다. 구현(2026-10-18): 감시는 `uranium-train supervise -- <bc|rl …>`(단일 run)와 run 큐 작업자 `uranium-train worker`(061-R13)가 맡는다. 자식 프로세스를 감시하고, stderr 출력(또는 진행 파일 변경)이 `--hang-minutes`(기본 30분) 동안 없으면 종료 후 재시작한다. 재시작은 시간 창(기본 60분)당 횟수 제한(기본 5회)과 늘어나는 간격(기본 60초, 2배씩, 최대 10분)을 지킨다. 재시작하지 않는 경우: 종료 코드 0(완료·`--steps` 도달·Ctrl+C 정상 종료·잠금 상실), 130(두 번째 Ctrl+C), 75(잠금을 얻지 못함, 061-R5), 76(양보, 061-R14), 감시 프로세스가 Ctrl+C를 받은 뒤. 같은 호스트 재시작은 자기 잠금을 넘겨받고 미push 체크포인트를 먼저 push한다(061-R5, R11-5). **Ctrl+C 전달(2026-10-19, H-4, AMB-061-6)**: Linux에서 감시 프로세스(`supervise`·`worker`)는 자식 학습을 **별도 프로세스 그룹**으로 띄우고 SIGINT를 자기가 한 번만 전달한다. 터미널 포그라운드에서 Ctrl+C를 눌러도 자식이 SIGINT를 두 번 받아(프로세스 그룹 + 전달) 저장 없이 130으로 끝나는 일이 없게 한다. **자동 시작**: Linux는 systemd user 서비스(`tools/worker/uranium-worker.service`: `KillSignal=SIGINT`, `KillMode=process`, `TimeoutStopSec=900`)로 감시 프로세스를 띄운다. `TimeoutStopSec`은 900초(15분)로 통일하고 docs도 같은 값을 쓴다. Windows용 자동 시작 구성은 제공하지 않고 사용자가 직접 실행한다(2026-10-19 사용자 결정 D-c, AMB-060-1) |
| 060-R7 | **MUST** 저장 총량은 **1TB 미만**이다(사용자 결정 2026-10-15). 공용 드라이브와 로컬 캐시 모두 아래 예산을 지킨다. 디스크 경고는 061-R8(물리 디스크 사용률 80%, 학습 시작·체크포인트 저장마다, 2026-10-19 사용자 결정 C-3)을 따른다. RL·평가 리플레이는 **자동으로 지우지 않는다**(042-R11). 예산을 넘으면 경고만 한다(2026-10-19 사용자 결정, AMB-061-5, 이전의 "RL 자가 대전 리플레이 자동 삭제"를 제거). 리플레이 총량은 RL 학습(`uranium-train rl`)이 학습 시작과 체크포인트 저장 때 061-R8 디스크 검사와 함께 `runs/*/replays/` 합계로 잰다 |

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

### 060-R3 메모리 예산 (Linux Vulkan 호스트. CUDA 호스트는 TBD(M5))
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
| RL·평가 리플레이 | ≤ 100GB (초과 시 경고만, 삭제하지 않음, 042-R11) |
| 로그·리포트·기타 | 나머지 |

## Acceptance Criteria
- 060-AC1: 벤치 스위트 결과가 Vulkan 호스트 벤치 리포트(`reports/bench_<label>.md`, 장비 라벨은 리포트에만, 001-R10)에 기록된다. 이 결과로 032-R2의 모델 크기와 042의 배치 크기를 확정한다.
- 060-AC2: Vulkan 호스트의 f16 경로가 Flex(CPU) f32 대비 허용 오차(1e-2 상대) 이내이거나, 아니면 f32 전용으로 결정하고 기록한다. CUDA 호스트는 f32(TF32)가 기본이므로(032-R8, 2026-10-17) f16 판정 대상이 아니다. **닫힘(2026-10-18 사용자 결정)**: Vulkan 호스트의 BC 학습은 f32 전용으로 정했다(032-R8 예외). f16 학습 경로는 구현하지 않았다.
- 060-AC3: 24시간 BC 학습 스모크에서 드라이버 오류 없이 완주하거나, 오류 시 자동 재시작 후 이어서 학습한다. 코어 비교 run(032-R17)의 첫 24시간으로 대신한다(2026-10-19 사용자 결정).
- 060-AC4: CUDA 호스트에서 같은 벤치 스위트(`cargo run -p bench --release --features cuda -- --backend cuda`)를 돌려 `reports/bench_<label>.md`(장비 라벨, 001-R10)에 기록한다. autotune·fusion 설정(041-R15)으로 측정한다. Flex 대비 정확도 기준은 032-AC1(CUDA ≤ 1e-2, AMB-032-5)을 따르고, 정밀도는 f32(TF32)다(032-R8).

## Risks & Open Questions
- Linux Vulkan 호스트(Mesa Vulkan 드라이버)에서 Burn/CubeCL 커널 성숙도가 불확실하다(특히 f16, subgroup 연산).
- 060-R6 별도 프로세스 그룹: 감시 프로세스가 강제 종료(SIGKILL 등)되면 자식 학습이 고아로 남아 heartbeat를 계속 갱신할 수 있다(다른 호스트가 인수하지 못함). 대응: Linux는 자식에 `PR_SET_PDEATHSIG`(SIGINT 또는 SIGTERM)를 설정해 감시 프로세스가 강제 종료돼도 자식이 신호를 받아 멈추게 한다.
- 체크포인트 쓰기는 원자적으로 한다(061-R4).

- 메모리를 공유하는 호스트는 CPU 작업(OpenBW, feature 추출)과 메모리 대역폭을 두고 경합한다. 로컬 OpenBW는 최소화하고 원격 호스트를 우선한다.

# 041 — Imitation Learning (BC)

- **Status**: Approved (2026-10-15, 2026-10-17 개정: R11 혼합 선택 레이블 — AMB-041-1, AC2 측정 조건 — AMB-041-2. 2026-10-18 개정: R12 분할 — AMB-041-3, R13 데이터 선택·셔플 — AMB-041-4, R14 학습 길이 1 epoch — 사용자 결정. 2026-10-19 개정(BC 전 리뷰 결정, 사용자 확정): R3 전역 norm clip(G-D4), R7 판정 불가 value 가중치 0(A-3), R9 스트리밍 로더 조건부(E-11), R15 Burn autotune·fusion(G-D5), 장비 표기 중립화(001-R10). 같은 날: R15 CUDA 호스트 비교 기준·Windows autotune 캐시 제약(사용자 결정 2026-10-19). 2026-10-20 개정: R13 셔플 전 파일 순서(`<출처>/<파트>/<파일>` 키)와 같은 출처 중복 지정 거부(사용자 결정))
- **Related**: [032](032-model-architecture.md), [040](040-replay-dataset.md), [042](042-rl-training.md), [050](050-evaluation.md)

## Purpose
리플레이 데이터셋으로 정책을 지도학습한다. 결과물은 RL의 초기 정책이자 KL 정규화의 teacher다.

## 맵 전략
"맵 고정"은 평가·RL 범위에만 적용하고, 데이터 범위에는 적용하지 않는다.

| 단계 | 데이터 | 목적 |
|---|---|---|
| S1 사전학습 | 전체 맵 TvX | 맵 무관 지식(유닛 상성, 마이크로, 빌드, 매크로). 맵 조건은 공간 채널과 map 임베딩으로 처리 |
| S2 파인튜닝 | 투혼·서킷브레이커만 + 대칭 증강 | 맵 특화(초크, 심시티, 확장 경로, 거리 감각) |

## Requirements
| ID | 요구사항 |
|---|---|
| 041-R1 | **MUST** 손실 = 헤드별 cross-entropy 합이다. 헤드 가중치는 action_type 1.0, delay 0.5, queued 0.2, selected 1.0, target_unit 1.0, target_loc 1.0으로 한다. 해당 action_type에서 쓰지 않는 헤드는 마스크로 0 처리한다 |
| 041-R2 | **MUST** 학습은 시퀀스 단위(T=64, truncated BPTT)로 한다. Core 상태는 같은 게임 시점의 연속 구간 사이에서 이어 쓴다 |
| 041-R3 | **MUST** 옵티마이저는 AdamW로 하고, warmup + cosine 스케줄, grad clip 1.0을 쓴다. 하이퍼파라미터는 설정 파일로 관리한다. **grad clip의 의미(2026-10-19 사용자 결정, G-D4, AMB-041-5)**: 모든 파라미터 gradient를 하나의 벡터로 본 **전역 L2 norm**이 1.0을 넘으면 전체를 같은 비율로 줄인다(파라미터 텐서별 클립이 아님). micro-lane(구현 메모)으로 나눠 계산할 때는 모든 레인 그룹의 gradient를 합산한 뒤 한 번 적용한다. Burn 기본 클립이 텐서별이라 직접 구현하고, RL learner(042 옵티마이저 표)도 같은 구현을 쓴다 |
| 041-R4 | **MUST** S1→S2 전환 시 S1 체크포인트를 보존하고, S2는 학습률을 1/10로 낮춘다 |
| 041-R5 | **MUST** ablation: (A) 대상 맵만 학습 vs (B) S1+S2를 비교하고 결과를 `reports/bc_ablation.md`에 기록한다 |
| 041-R6 | **SHOULD** NoOp과 Move 같은 빈도 높은 클래스에 클래스 균형 가중치를 적용할지 검토한다. **결정(2026-10-18 사용자)**: 적용하지 않는다. BC 손실은 클래스 가중치 없이 R1 그대로 쓴다. 사유: 사람 같은 행동 분포를 재현하는 것을 중시한다. 재검토 조건: BC 평가(041-AC1·AC2, `uranium-train bc-eval`)에서 모델 성능이 너무 낮게 나오면 다시 검토한다 |
| 041-R7 | **MUST** value head(win)도 BC 단계에서 학습한다. 정답은 리플레이의 실제 승패(테란 시점 승 +1 / 패 −1)이고, 손실은 MSE(가중치 0.5, 설정 가능)다. 입력에는 재생 모드의 `PrivilegedSummary`(011-R12)를 쓴다. build·units·value 헤드는 BC에서 학습하지 않는다. 승패를 판정할 수 없는 게임(040-R8 `result = 0`, `result_source = none`)은 value 손실 가중치를 0으로 둔다(정책 헤드는 그대로 학습, 2026-10-19 사용자 결정, A-3, AMB-040-9) |
| 041-R8 | **MUST** z 조건은 해당 리플레이 자신의 z를 쓰고, 샘플의 25%는 `z_present=false`로 비워서 학습한다. RL(033-R2)과 같은 비율이다. 같은 방식으로 게임의 5%는 상대 종족을 첫 적 유닛 관측 전까지 `unknown`으로 가린다(030-R11, `data.enemy_race_mask` 기본 0.05, 게임 단위 고정 seed, 사용자 결정 2026-10-19, 맵 대칭 외 증강 없음(030-R4·040-R7)의 명시적 예외) |
| 041-R9 | **MUST** (조건부) 학습 데이터가 저장 예산(040-R12, 060-R7)을 넘어 `.rep`만 보관하는 리플레이가 생기면, 데이터 로더는 저장 shard와 재생 스트리밍(040-R12)을 하나의 인터페이스로 읽는다. 두 경로의 출력은 동일하다(040-AC6). 모든 학습 데이터가 예산 안에서 shard로 저장돼 있는 동안에는 스트리밍 경로를 구현하지 않는다(2026-10-19 사용자 결정, E-11: 예산 초과 시에만, AMB-041-6) |
| 041-R10 | **MUST** 손실 가중치, 학습률 등 하이퍼파라미터는 설정 파일로 관리하고 체크포인트에 함께 기록한다. 튜닝 대상이며 정책 결정 사항이 아니다 |
| 041-R11 | **MUST** 혼합 선택 레이블(2026-10-17, AMB-041-1): 게임은 명령을 선택된 유닛마다 따로 적용하고, 받을 수 없는 유닛은 건너뛴다(OpenBW `action_*`). 그래서 레이블의 `selected`에서 해당 액션의 선택 자격([034](034-action-table.md) `select_cap`)이 없는 유닛을 빼고 학습한다. 남는 유닛이 없으면(게임에서 아무 일도 일어나지 않는 명령) 무효 레이블(weight 0, 031-R4)이다. 에이전트 기억(030 §4 직전 행동)에도 뺀 뒤의 유닛을 기록한다. 이 정책은 학습 시점 변환이며 shard에는 원래 선택을 저장한다. 대상 유닛이 선택에 들어 있는 레이블도 같은 방식이다: 대상 유닛만 선택에서 빼고, 남는 유닛이 없으면 무효(`target_selected`)다(034-R9, `SetRally_Unit` 제외, 2026-10-19) |
| 041-R12 | **MUST** train/val/test 분할(040-R5 세부, 2026-10-18 확정, AMB-041-3): 원본 리플레이 해시(`replay_sha256`)로 게임 단위 결정론 분할한다. TvT 두 시점은 같은 쪽에 들어간다. 대상 맵(투혼·서킷브레이커 패밀리, 040-R15 `config/map_families.toml`로 build 때 정한 `map_family`)은 train 80 / val 10 / test 10, 그 외 맵은 train 95 / val 5(test 없음)다. S1·S2·ablation(041-R5) (A)·(B)가 모두 같은 분할을 쓰고, 대상 맵 test 셋은 어떤 학습에도 들어가지 않는다 |
| 041-R13 | **MUST** 데이터 선택과 순서(2026-10-18 확정, AMB-041-4): S1은 모든 소스(`stardata`·`community`·`scr_ladder`·`personal`)의 train 분할 전체를 소스 가중치 없이 쓴다. S2는 대상 맵 train 분할만(소스 무관) 쓴다. 게임 순서는 고정 seed(설정 `data.seed`) 셔플로 결정론을 유지한다: shard 파일 순서와 파일 안 게임 순서를 섞는다(파일 안 순서의 seed는 `data.seed`와 파일 순번에서 만든다. 재개 시 앞 파일을 읽지 않고 위치를 찾기 위함). 섞기 전 파일 목록은 파일마다 `<출처>/<파트>/<파일>` 키의 순서다. 그래서 같은 shard 파일이면 데이터 지정 방식(`--dataset`/`--shards`)·지정 순서·저장 위치와 상관없이 순서가 같다. 같은 출처를 두 번 지정하면 거부한다(061-R10 다른 호스트 재개. 사용자 결정 2026-10-20) |
| 041-R14 | **MUST** 학습 길이(2026-10-18 사용자 결정): S1, S2, ablation (A)는 각각 **1 epoch**(해당 단계 train 데이터를 한 번)만 학습한다. cosine 스케줄 길이(R3 `total_steps`)는 그 단계의 1 epoch update 수와 같게 해서 학습률이 끝까지 내려가게 한다. update 수는 `uranium-train bc-epoch`가 트레이너와 같은 레인 진행으로 센 값이고 `config/bc.toml` `[steps]`에 둔다. warmup(R3 `warmup_steps`)은 S1은 설정값(2,000) 그대로, S2와 ablation (A)는 그 단계 1 epoch update 수의 5%를 반올림한 값(소수점 첫째 자리에서 반올림, .5는 올림. 7,915 × 0.05 = 395.75 → 396)으로 한다(2026-10-18 사용자 결정). 데이터 위치는 체크포인트에 저장하고 재개 시 이어서 읽는다(본 게임을 다시 학습하지 않는다, 061-R10) |
| 041-R15 | **MUST** Burn kernel autotune과 fusion(2026-10-19 사용자 결정, G-D5, AMB-032-7): CUDA와 Vulkan 학습 모두 autotune과 fusion을 켠다. 코어 비교(032-R17)와 BC 본학습(S1·S2·ablation (A))은 같은 설정으로 돌린다. 이 설정에서 CUDA는 일부 conv gradient가 Flex와 더 달라진다(`reports/m6_bc_preflight.md` 4차. CUDA는 matmul에 이미 TF32를 쓴다, AMB-032-5). 설정은 체크포인트에 함께 기록한다(R10). **autotune 캐시**: autotune이 고른 커널 목록(캐시)은 호스트마다 고정한 로컬 위치(`.env` 키 `URANIUM_AUTOTUNE_CACHE_DIR`, 001-R9. `.env.example` 추가는 구현 때)에 두고, run이 끝날 때까지 지우거나 다시 만들지 않는다. 같은 장비가 같은 캐시로 재개해야 결과가 비트 단위로 같다(061-AC1). 캐시가 다르거나 새로 만들어지면 결과는 허용 오차 기준(032-AC1·AC8)으로만 같다. **CUDA 호스트는 같은 PC·같은 캐시여도 032-AC1 허용 오차로 비교한다**(사용자 결정 2026-10-19). 비트 단위 이어 하기는 Flex와 autotune 캐시를 고정한 Vulkan 호스트에만 요구한다. 근거(`reports/m6_bc_preflight.md` 5차): CUDA 축소 연산 일부가 float atomic add라 실행마다 결과가 달라질 수 있고, **알려진 제약**: Windows에서는 CubeCL 0.11의 영구 autotune 캐시(Turso DB)가 쓰기 모드로 열리지 않아("experimental multiprocess WAL is not supported") 프로세스마다 autotune을 다시 측정한다(첫 update 지연, 실행마다 다른 커널 선택 가능). 구현은 학습 코드 수정 단계에서 한다 |
| 041-R16 | **MUST** 데이터 미리 읽기 깊이(2026-10-20 사용자 결정, G-M2): 학습 루프 앞에서 다음 batch를 미리 만들어 두는 개수를 호스트마다 `.env` 키 `URANIUM_BC_PREFETCH`(001-R9, 없으면 2)로 정한다. 0이면 데이터 스레드 없이 학습 루프가 update마다 그 자리에서 batch를 만들어, 호스트 메모리에 batch가 하나만 있다. batch(레인 32 × T64)는 약 4 GB이고 깊이 2에서는 최대 4개가 동시에 있다. RAM을 GPU와 나눠 쓰는 Linux Vulkan 호스트는 이 때문에 코어 비교 GTrXL run 중 OOM으로 끝났다(`reports/m6_core_ablation.md` 5절). 깊이는 batch 내용·순서·데이터 위치(R13·061-R10)를 바꾸지 않는다. 그래서 설정 지문(R10)에 넣지 않고, 깊이가 다른 호스트끼리 같은 run을 이어 해도 된다 |

## 평가 지표
- 오프라인 지표:
  - 대상 맵 test 셋 기준 action_type top-1/top-5 정확도
  - delay MAE
  - target_loc 타일 오차
  - selected F1
- 온라인 지표: BC 정책 vs T0 스크립트 봇 승률(종족×맵). 측정 방법은 [050](050-evaluation.md)을 따른다.

## Acceptance Criteria
- 041-AC1: 대상 맵 test action_type top-1 ≥ 50%다.
- 041-AC2: BC 정책이 T0 봇 상대로 각 매치업에서 승률 ≥ 50%다.
  - 측정 조건 (확정 2026-10-17, AMB-041-2): 조합(상대 종족 T·P·Z × 맵 2)당 50판(050 E-Matrix와 같음), z 없음(BC에서 25% 확률로 z를 빼고 학습한 조건, 041-R8), argmax와 확률적 샘플링 두 모드를 모두 측정·보고(050-R1), 승률에 Wilson 95% 구간 표기(050-R2). plan은 `config/plans/bc-vs-t0.toml`이다.
- AC1과 AC2의 수치는 초기값이다(2026-10-15 확정). M5 기준선 측정 후 조정을 제안할 수 있고, 조정하려면 사용자 확인이 필요하다(050과 같은 방식).
- 041-AC3: (B)가 (A)보다 대상 맵 test 지표가 같거나 높다. 그렇지 않으면 맵 전략을 재검토한다.

## 구현 메모 (비규범)
- GPU 메모리(2026-10-17, [`reports/m5_bc_memory.md`](../reports/m5_bc_memory.md)): `config/bc.toml`의 `micro_lanes`만큼 레인을 묶어 forward/backward를 하고, gradient를 합쳐 전역 norm clip(R3) 뒤 AdamW step을 한 번 한다. 손실은 배치 전체 유효 스텝 수로 나누므로 `lanes` 전체를 한 번에 처리한 것과 같은 업데이트다(R2·R3 그대로). 최대 메모리는 그룹 크기를 따른다(기본 1 = 레인 1 × T64 × N512).
- 데이터 로더(041-R9, 2026-10-18): `--dataset <rel>`(공용 드라이브 `datasets/` 아래, manifest 검증 후 사용, 061-R11-1·R12)과 `--shards <dir>`(검증 없음, 표본·테스트)를 여러 번 받는다. 폴더는 하위 폴더(build의 `<src>/<part>/`)까지 읽는다. `bc`는 기본 S1, `--s2`면 S2를 고르고, ablation (A)는 `--stage s2`(S2 데이터, 기본 학습률)다. S1→S2(R4)는 `--s2 --init <S1 체크포인트>`로 가중치만 가져오고 옵티마이저·스케줄·step은 새로 시작한다(`--resume`은 같은 단계 이어 하기). 분할 집계는 `uranium-train split-stats`.
- `pad_entities`는 배치 N을 그 배수로 올리는 마스크 패딩이고, `grad_checkpointing`은 Burn balanced checkpointing이다. 둘 다 결과를 바꾸지 않는다.
- 장기 실행(2026-10-18): `bc --run <id>`는 061 run(`checkpoints/<id>/`, 잠금·heartbeat·push, 시작 시 `LATEST`에서 재개)으로 저장해 다른 학습 호스트에서 이어 할 수 있다. 체크포인트에는 데이터 위치(레인별 게임·진행 step·z·대칭, 다음에 읽을 shard 위치, 데이터 RNG)를 넣고, 재개하면 레인은 기록된 step까지 feature만 다시 만들어 이어 간다(Core 상태만 새로 시작). 체크포인트 주기는 `checkpoint_every` update와 `checkpoint_minutes`(시간) 중 먼저 오는 쪽이고, Ctrl+C는 체크포인트 후 종료한다. feature 추출과 shard 읽기는 별도 스레드가 다음 batch를 미리 만든다(같은 batch 순서, 결과 동일).
- 운영(2026-10-18 사용자 결정, 061-R13·R14·R15, 060-R6): BC run은 run 큐 작업자 `uranium-train worker --queue <큐> --backend <cuda|vulkan> --priority <n>`로 돌린다. Linux Vulkan 호스트 작업자는 우선순위를 낮게, Windows CUDA 호스트 작업자는 우선순위를 높게 두고, Windows 작업자는 사용자가 직접 실행하고 Ctrl+C로 멈춘다(빈 run이 없으면 낮은 우선순위 작업자에 양보 요청, 061-R14). 멈춘 run은 다른 장비가 체크포인트·데이터 위치에서 이어 간다. 장비별 구간은 `runs/<id>/devices.jsonl`에 남는다. 어느 장비를 learner로 쓰는지는 추적하지 않는 설정과 리포트에 둔다(001-R10).
- 오프라인 지표(2026-10-18): `uranium-train bc-eval --checkpoint <ckpt|run 폴더> (--dataset …) [--split test|val]`은 대상 맵 test 셋에서 action_type top-1/top-5, delay MAE, target_loc 타일 오차, selected F1(라벨을 앞 헤드에 넣은 teacher forcing 기준), 헤드별 NLL과 AC1 판정을 출력한다. z 없음·대칭 없음. 지표 정의(2026-10-19 기술 결정, 리뷰 E-9): delay MAE는 프레임 단위(bucket 값, bucket 번호 MAE도 함께), 헤드별 NLL은 그 헤드를 쓰는 스텝 수로 나눈 평균(action·delay는 모든 스텝)이다. `--games N`은 분할 전체를 고정 seed(`--seed`, 기본 1)로 섞어 앞 N게임을 고른다(032-R17 ②).
- 재개 검사(2026-10-19, 리뷰 B-3·G-F3, BC 전 리뷰 계획 승인): 체크포인트에 stage, 설정 지문(실효 설정. 갱신 결과를 바꾸지 않는 `micro_lanes`·`grad_checkpointing`·`pad_entities`·체크포인트 주기·보존 개수는 제외), 데이터 지문(파일별 manifest SHA-256)을 남기고, 재개할 때 다르면 거부한다(우회 옵션 없음). 승패 판정 불가 게임 수(R7)는 학습 로그 `no_result_games`에 남긴다.
- value 손실 정규화(2026-10-19, R7 판정 불가 가중치 0의 구현): 정책 헤드 손실은 배치의 유효 스텝 수로 나누고, value 손실은 그중 승패가 판정된 게임의 유효 스텝 수(value 가중치 합)로 나눈다. 판정 불가 게임 비율이 배치마다 달라도 value 손실의 크기가 흔들리지 않게 하기 위해서다.

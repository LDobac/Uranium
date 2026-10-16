# M5 선행 작업 리포트 — 데이터 독립 범위 (2026-10-16)

M4(리플레이 데이터셋)와 병행해 M5 중 대량 데이터가 필요 없는 부분을 먼저 구현했다(사용자 결정 2026-10-16). 데이터 규모가 필요한 041-AC1~3, RL 단계의 033-AC2, SC:R 경로가 필요한 030-AC1은 범위 밖이다.

측정 장비: 개발 PC의 RTX 4070 Ti SUPER(16GB, CUDA Toolkit 13.4.1 / Vulkan). BC250 수치는 아직 없다(아래 "다음 단계").

## 구현 요약

| 영역 | 위치 | 내용 |
|---|---|---|
| 모델 (032) | `crates/uranium-model` | Entity/Spatial/Scalar 인코더, `Core` trait + `LstmCore`·`GtrxlCore`, 자기회귀 6헤드(GPU 마스크 gather, Gumbel-max), ValueHeads(privileged 전용), 체크포인트, host→device collate |
| BWAPI 표 (013-R1) | `tools/bwapi-tables` → `crates/uranium-env/tables/*_types.toml` | BWAPI 4.2 소스에서 유닛 234·테크 47·업그레이드 63·무기 132·order 191 표 생성 |
| feature (030) | `uranium-env::features` | `extract(&RawFrame, &mut AgentMemory, &MapInfo)`, 엔티티 76·공간 26·스칼라, privileged 분리, 대칭 증강 8종 |
| 마스크 (034) | `features::masks` | unit_cap 46, loc_class_mask 8, type_mask(prereq는 BWAPI 표에서 계산) |
| executor (031) | `features::executor` | pick_building·pick_worker 결정론, 마스크 위반 거부 |
| 맵 사전계산 (031, AMB-031-3) | `features::mapgen` | 모든 맵: base 위치·choke·wall 힌트. `uranium-replay build`가 맵마다 `assets/maps/<map_hash>.toml` 생성 |
| 보상 (033) | `uranium-env::reward`, `config/reward.toml` | 항목별 분리, z 조건, 관측된 파괴만 |
| shard (AMB-040-6) | `uranium-replay::shard` | schema 2: 게임마다 `GameInfo` 저장. schema 1 shard는 2026-10-17 삭제, 전체 build 대기 |
| BC 골격 (041) | `crates/uranium-train`, `config/bc.toml` | `StepSource`(shard/스트리밍 공통), 레이블 변환·검증, z 25% drop, 레인별 Core 상태, 손실·AdamW·warmup+cosine·clip, CLI `uranium-train bc` |

## Acceptance Criteria 상태

| AC | 결과 | 근거 |
|---|---|---|
| 030-AC1 (OpenBW vs SC:R 비트 일치) | 범위 밖 | SC:R 경로는 M7 |
| 030-AC2 (시야 밖 누출 없음) | 통과 | `uranium-env/tests/features.rs` |
| 030-AC3 (512 엔티티 ≤ 1ms) | 통과 0.89ms (release, last_seen_age 추가 후. 추가 전 0.72ms) | ignored 테스트 |
| 030-AC4 (증강 역변환 일치) | 통과 (8종 모두 비트 일치, 위치 레이블 포함) | 위치는 좌상단 기준 footprint로 변환 |
| 030-AC5 (privileged 필드 없음) | 통과 (컴파일 타임 전수 패턴) | |
| 031-AC1/AC4 (OpenBW 시나리오) | 일부: AC4의 타일 일치는 단위 테스트로 확인, OpenBW 실행 시나리오는 미실행 | |
| 031-AC2 (마스크된 액션 미실행) | 통과 (proptest 400건) | |
| 031-AC3 / 034-AC3 (역변환 커버리지) | 통과 98.91% (STARDATA 표본 103게임) | `reports/m4_stardata_preflight.md` |
| 032-AC1 (Vulkan ≤ 1e-4, CUDA ≤ 1e-2) | 통과: Vulkan ≤ 1e-6, CUDA 기본 크기 최대 3.9e-3 | `tests/cross_device.rs`, AMB-032-5 Resolved |
| 032-AC2 (CUDA PC: Vulkan보다 빠름) | 통과: CUDA f32 p95 61ms < Vulkan 70~170ms | BC250 기준은 실측 후 재확인 |
| 032-AC3 (CUDA PC: Vulkan보다 빠름) | 통과: CUDA f32 66~70 samples/s, Vulkan 약 29 | |
| 032-AC4 (마스크 10만 샘플) | 통과 (LSTM·GTrXL 각 10만, release) | |
| 032-AC5 (act에 privileged 없음) | 통과 | |
| 032-AC6 (Core 교체) | 통과 (같은 테스트 묶음을 두 Core에 적용) | |
| 033-AC1 (보상 단위 테스트) | 통과 | `uranium-env/tests/reward.rs` |
| 034-AC1 / AC2 | 통과 (액션 100개 모두 합성 상태에서 열림, 인자 마스크 비어 있지 않음) | |
| 041 골격 | 통과: 작은 게임 overfit loss 6.29 → 1.38 (40 업데이트) | `uranium-train/tests/bc.rs` |

## 성능 측정 (기본 config 11.3M 파라미터, B16·N256, 맵 128²)

재측정(2026-10-16 23시, 반복 50회, 측정 간 순차 실행). 처음 측정(76~215ms 등)은 반복 수가 적고 다른 작업과 GPU를 같이 써서 편차가 컸으므로 아래 값으로 대체한다.

| 항목 | CUDA f32 (TF32) | CUDA f16 | Vulkan f32 | Vulkan f16 |
|---|---|---|---|---|
| act LSTM 중앙값 / p95 | **58.9 / 61.3ms** | 78.0 / 85.8ms | 65.3~101.6 / 최대 169.9ms (두 번 측정) | 66.0 / 69.5ms |
| act GTrXL 중앙값 / p95 | **58.7 / 66.6ms** | 88.0 / 93.8ms | 71.5~107.4 / 112.8ms | 72.0 / 77.9ms |
| act 분해: entity / entity+spatial | 9.4 / 34.6ms | 11.3 / 33.8ms | 21.1 / 56.6ms | 12.4 / 30.6ms |
| 학습 LSTM B2·T16 | **69.8 samples/s** | 54.3 | 29.5 | TDR로 측정 불가 |
| 학습 LSTM B4·T16 | **65.5 samples/s** | 58.3 | 28.8 | TDR로 측정 불가 |
| 학습 GTrXL B2·T16 | **70.4 samples/s** | 56.8 | TDR로 측정 불가 | – |
| distance bias fwd+bwd / fwd (032-R13 ≤ 5배) | 5.0배 (경계) | 4.4배 | 10.7배 (첫 측정) | 1.6배 (첫 측정) |

- **판정 (사용자 기준: CUDA PC에서는 Vulkan보다 빠르면 됨)**: CUDA f32가 추론·학습 모두 모든 Vulkan 측정보다 빠르다. 추론 중앙값은 10~40% 빠르고 p95가 훨씬 안정적이며, 학습은 약 2.3배 빠르다. 40ms 초안 기준에는 여전히 미달이다.
- 이 PC에서는 CUDA f16이 CUDA f32(TF32)보다 느리다. 활성값 캐스팅 비용이 tensor core 이득보다 크다. CUDA PC의 권장 설정은 f32(내부 TF32)다.
- Vulkan 학습 측정은 이 PC(Windows)에서 GPU 타임아웃 복구(TDR, 2초 제한)로 두 번 중단됐다. Windows에서 Vulkan으로 학습하는 구성은 운영 대상이 아니므로(사용자 확인 2026-10-16) 대응하지 않는다. Vulkan 학습은 Linux BC250에서만 쓴다. 벤치는 측정마다 시간 제한을 걸고 실행한다.
- f16 측정은 장치 기본 dtype을 F16으로 바꾼 것이다. 헤드의 softmax·log-prob·엔트로피·Gumbel은 f32로 계산한다(032-R8). **마스터 가중치 f32 + f16 연산의 혼합 정밀도 학습은 아직 구현하지 않았다**(Burn 0.22에 AMP가 없어 별도 설계가 필요).

## spec에 수치가 없어 코드 상수로 둔 정규화 (030)

`features::extract::norm`: 속도 /(256·8), spell 쿨다운 /64, 상태 타이머 /256, 개수 log(1+n)/4. 스칼라 레이아웃: `FeatureDims::scalar()` (현재 차원은 표에서 계산). order는 190(Unknown)을 189(None)로 매핑.

## 결정 사항 (2026-10-16)

1. **032-AC2·AC3**: CUDA PC에서는 같은 PC의 Vulkan보다 빠르면 통과(사용자 결정). BC250 기준은 BC250 실측 후 다시 확인한다.
2. **AMB-032-5**: CUDA 허용 오차를 1e-2로 늘림(사용자 결정). Vulkan은 1e-4를 유지한다.
3. **AMB-030-1**: 사용자 위임으로 Claude가 (a) "타일을 마지막으로 본 뒤 경과 시간(정찰 기록)"을 선택했다. 채널명 `last_seen_age`, feature 스키마 버전 2.
   - 근거: 032 §2.4가 "정찰 기록"을 관측에서 명시적으로 유지할 장기 정보로 지정했는데 담을 채널이 없었다(`explored`는 이진값). 적 유닛의 과거 위치(b)는 수십 초 안에 쓸모가 없어지는 단기 정보라 LSTM 코어가 담당할 수 있다. 반면 맵 전역의 확인 시점은 몇 분 단위로 쌓이는 공간 정보라 코어에 담기 어렵다. (a)와 기억된 적 건물을 합치면 "이 확장 위치를 2분 전에 확인했을 때 건물이 없었다"는 판단이 가능하다.

## CUDA 허용 오차 1e-2의 영향 (AMB-032-5)

**측정 (기본 크기 11.3M 모델, B16·N256, 같은 가중치·입력, CPU Flex와 비교)**

| 출력 | LSTM 최대 차이 | GTrXL 최대 차이 |
|---|---|---|
| value | 2.96e-5 | 3.60e-5 |
| log-prob: action_type / delay / queued | 9.1e-6 / 1.4e-4 / 2.6e-4 | 1.8e-5 / 2.5e-4 / 0 |
| log-prob: selected (최대 12 step의 합) | **3.91e-3** | 1.10e-3 |
| log-prob: target_unit / target_loc | 2.4e-3 / 4.8e-7 | 0 / 0 |
| 정책 분포 KL(CPU‖CUDA): action_type / 위치 | 2.9e-8 / ≈0 | 1.2e-7 / 7.5e-8 |
| greedy 액션 일치(전체 6헤드) | 16/16 | 16/16 |

작은 모델 테스트(8행)까지 합치면 greedy 액션은 48/48 행에서 모두 일치했다.

**의미**
- **추론·실전**: 같은 상태에서 고르는 액션이 사실상 같다. log-prob 차이 3.9e-3은 확률로 따지면 최대 약 0.4%의 상대 차이이고, 분포 KL은 1e-7 수준이다. 확률이 거의 같은 두 후보가 경합하는 드문 경우에만 greedy 선택이 갈릴 수 있다.
- **학습(BC)**: TF32는 행렬곱 입력의 가수를 10비트로 줄이고 누적은 f32로 한다. 032-R8이 이미 채택한 f16 혼합 정밀도(가수 10비트, matmul 오차 4.5e-4)와 같은 수준이다. 이 정도 오차로 수렴이 나빠진다는 사례는 일반적으로 보고되지 않으며, PyTorch도 합성곱에서 TF32를 기본으로 쓴다.
- **RL(M6)**: actor와 learner가 다른 장치(CPU·Vulkan vs CUDA)면 behaviour log-prob이 최대 약 4e-3 어긋나 importance ratio가 최대 약 1.004배가 된다. V-trace는 ratio를 1에서 자르므로 영향은 작다. 그래도 actor와 learner는 같은 백엔드를 쓰는 것을 권장한다.
- **재현성**: CUDA와 CPU·Vulkan 사이의 비트 단위 재현은 보장되지 않는다(f16 채택 때 이미 포기한 성질). 체크포인트 가중치 자체는 영향이 없어 다른 장치로 옮겨 써도 된다. CUDA 커널 자동 선택(autotune)이 실행마다 다른 커널을 고를 수 있어 같은 CUDA PC 안에서도 1e-3 수준 차이가 생길 수 있다(실행 간 비교는 미측정).
- **테스트 민감도**: CUDA 테스트는 1e-2보다 작은 구현 오류를 잡지 못한다. 구현 오류는 1e-4 기준을 유지하는 Vulkan 테스트에서 잡히고, CUDA 테스트는 마스크 오류·커널 오류처럼 0.1 이상 차이 나는 큰 문제를 잡는다.
- **여유**: 최대 실측 3.9e-3 대비 약 2.5배다. selected는 step 합이라 선택 수가 많을수록 커지므로 5e-3로는 간헐 실패 위험이 있어 1e-2로 정했다.

## 다음 단계 (미완료)

- f16 혼합 정밀도 학습(마스터 f32, 032-R8).
- BC250 실측: `cargo run -p bench --release -- --backend vulkan --filter net_` (`--f16` 포함).
- 031-AC1/AC4 OpenBW 시나리오 테스트.
- 실전·RL 게임에서 맵 사전계산 조회 키: shard는 CHK 해시, `GameInfo.map_hash`는 맵 파일 해시라 다르다. 라이브 게임에 자원 배치를 어떻게 줄지 M6 전에 정해야 한다.
- crate 의존: `uranium-env`(바이너리)가 `uranium-agent`에 의존한다. 에이전트가 모델·feature를 쓰게 되면(M6) 패키지 순환이 생기므로 스케줄러 바이너리 분리 등 구조 변경이 필요하다.
- BC GPU 메모리(2026-10-17, `reports/m5_bc_memory.md`): `micro_lanes`(기본 1)로 레인을 나눠 forward/backward하고 gradient를 합친다(레인 32와 같은 업데이트). 최대 메모리는 레인 1 × T64 × N512로 고정되고, 16GB CUDA에서 동작을 확인했다. `pad_entities=64`, `grad_checkpointing` 옵션. 보고된 레인 2 OOM은 이 PC에서 재현되지 않아 환경 확인이 필요하다.
- 혼합 선택 레이블 판정(`select_cap`, 남은 무효의 약 절반): 일부 유닛만 자격이 있는 선택(메딕 섞인 스팀 등)을 지금은 전체 무효로 본다. 자격 있는 유닛만 남길지 결정 필요(학습 시점 정책, shard 무관).
- 완료(2026-10-17): BC 데이터 로더를 BPTT 청크 단위 스트리밍으로 변경(게임 전체 feature가 게임당 약 4GB), 애드온 발자국 계산 수정, schema 1 shard 삭제.

## 재현 명령

```
cargo test --workspace
cargo test --release -p uranium-model -- --ignored masks_hold_100k          # 032-AC4
URANIUM_TEST_DEVICE=vulkan cargo test -p uranium-model --test cross_device -- --ignored --nocapture
cargo test --release -p uranium-env --test features -- --ignored --nocapture --test-threads=1   # 030-AC3
cargo test --release -p uranium-train --test bc -- --ignored --nocapture      # BC overfit
cargo run -p bench --release --features cuda -- --backend cuda --filter net_ [--f16]
cargo run -p bwapi-tables -- third_party/bwapi/bwapi crates/uranium-env/tables
```

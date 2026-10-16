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
| shard (AMB-040-6) | `uranium-replay::shard` | schema 2: 게임마다 `GameInfo` 저장. **schema 1 shard는 재빌드 필요** |
| BC 골격 (041) | `crates/uranium-train`, `config/bc.toml` | `StepSource`(shard/스트리밍 공통), 레이블 변환·검증, z 25% drop, 레인별 Core 상태, 손실·AdamW·warmup+cosine·clip, CLI `uranium-train bc` |

## Acceptance Criteria 상태

| AC | 결과 | 근거 |
|---|---|---|
| 030-AC1 (OpenBW vs SC:R 비트 일치) | 범위 밖 | SC:R 경로는 M7 |
| 030-AC2 (시야 밖 누출 없음) | 통과 | `uranium-env/tests/features.rs` |
| 030-AC3 (512 엔티티 ≤ 1ms) | 통과 0.72ms (release) | ignored 테스트 |
| 030-AC4 (증강 역변환 일치) | 통과 (8종 모두 비트 일치, 위치 레이블 포함) | 위치는 좌상단 기준 footprint로 변환 |
| 030-AC5 (privileged 필드 없음) | 통과 (컴파일 타임 전수 패턴) | |
| 031-AC1/AC4 (OpenBW 시나리오) | 일부: AC4의 타일 일치는 단위 테스트로 확인, OpenBW 실행 시나리오는 미실행 | |
| 031-AC2 (마스크된 액션 미실행) | 통과 (proptest 400건) | |
| 031-AC3 / 034-AC3 (역변환 커버리지) | M4 측정 항목 | |
| 032-AC1 (Flex 대비 ≤ 1e-4) | Vulkan 통과(≤ 1e-6), **CUDA 미달(최대 1.3e-3, TF32)** → AMB-032-5 | `tests/cross_device.rs` |
| 032-AC2 (B16·N256 추론 ≤ 40ms) | **미달** (아래 표) | |
| 032-AC3 (학습 처리량) | 측정만 (기준값 TBD) | |
| 032-AC4 (마스크 10만 샘플) | 통과 (LSTM·GTrXL 각 10만, release) | |
| 032-AC5 (act에 privileged 없음) | 통과 | |
| 032-AC6 (Core 교체) | 통과 (같은 테스트 묶음을 두 Core에 적용) | |
| 033-AC1 (보상 단위 테스트) | 통과 | `uranium-env/tests/reward.rs` |
| 034-AC1 / AC2 | 통과 (액션 100개 모두 합성 상태에서 열림, 인자 마스크 비어 있지 않음) | |
| 041 골격 | 통과: 작은 게임 overfit loss 6.29 → 1.38 (40 업데이트) | `uranium-train/tests/bc.rs` |

## 성능 측정 (기본 config 11.3M 파라미터, B16·N256, 맵 128²)

| 항목 | CUDA f32 | CUDA f16 | Vulkan f32 | Vulkan f16 | 기준 |
|---|---|---|---|---|---|
| act LSTM (중앙값) | 76~215ms (측정마다 편차) | 133ms | 237ms | 272ms | ≤ 40ms (p95) |
| act GTrXL | 172ms | 148ms | 291ms | 294ms | ≤ 40ms |
| act 분해 (CUDA f32) | entity 16ms, entity+spatial 42ms, 나머지 Core·헤드 | | | | |
| 학습 LSTM B2·T16 | 56 samples/s | 66 samples/s | 25 samples/s | 60 samples/s | TBD |
| 학습 LSTM B4·T16 | 57 samples/s | 84 samples/s | 27 samples/s | 31 samples/s | |
| distance bias fwd+bwd / fwd (032-R13) | 3.7배 | 2.1배 | **10.7배** | 1.6배 | ≤ 5배 |

- 추론은 entity·spatial 인코더만으로 42ms라, 그 뒤 단계(헤드·Core)까지 합치면 이 GPU에서도 기준을 넘는다. 측정 편차는 데스크톱 GPU를 다른 프로그램과 함께 쓴 영향이 있다.
- f16은 장치 기본 dtype을 F16으로 바꾼 측정이다. 헤드의 softmax·log-prob·엔트로피·Gumbel은 f32로 계산한다(032-R8). **마스터 가중치 f32 + f16 연산의 혼합 정밀도 학습은 아직 구현하지 않았다**(Burn 0.22에 AMP가 없어 별도 설계가 필요).

## spec에 수치가 없어 코드 상수로 둔 정규화 (030)

`features::extract::norm`: 속도 /(256·8), spell 쿨다운 /64, 상태 타이머 /256, 개수 log(1+n)/4. 스칼라 레이아웃: `FeatureDims::scalar()` (현재 차원은 표에서 계산). order는 190(Unknown)을 189(None)로 매핑.

## 사용자 결정이 필요한 항목

1. **032-AC2·학습 속도 미달 대응** (032 Risks: 결과 보고 후 재확인).
2. **AMB-032-5**: CUDA는 f32 matmul을 TF32로 계산(끌 설정 없음). 의존성 패치로 끌지, CUDA 허용 오차를 따로 둘지.
3. **AMB-030-1**: `enemy_last_seen_age` 채널 의미. (a) 타일을 마지막으로 본 뒤 경과 시간, (b) 그 타일에서 적을 마지막으로 본 뒤 경과 시간. 현재 0으로 둠.

## 다음 단계 (미완료)

- f16 혼합 정밀도 학습(마스터 f32, 032-R8).
- BC250 실측: `cargo run -p bench --release -- --backend vulkan --filter net_` (`--f16` 포함).
- 031-AC1/AC4 OpenBW 시나리오 테스트.
- 실전·RL 게임에서 맵 사전계산 조회 키: shard는 CHK 해시, `GameInfo.map_hash`는 맵 파일 해시라 다르다. 라이브 게임에 자원 배치를 어떻게 줄지 M6 전에 정해야 한다.
- crate 의존: `uranium-env`(바이너리)가 `uranium-agent`에 의존한다. 에이전트가 모델·feature를 쓰게 되면(M6) 패키지 순환이 생기므로 스케줄러 바이너리 분리 등 구조 변경이 필요하다.
- 기존 schema 1 shard 재빌드(AMB-040-6).

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

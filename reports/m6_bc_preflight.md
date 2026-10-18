# M6 BC 본학습 전 표본 확인 (2026-10-18)

대규모 배치 전 점검 규칙에 따라 BC(041) 본학습 경로를 표본으로 끝까지 돌렸다. 본학습은 시작하지 않았다.

## 결론
- **경로는 끝까지 동작한다.** 동작한 단계는 다음과 같다.
  - S1 학습(`--dataset` 4개, manifest 검증)
  - 체크포인트 저장과 `--resume`
  - S2 학습(`--s2 --init`, 이번에 추가)
  - 041-AC2 plan 경로(모델 대 T0, 6판)
- **지금 설정 그대로는 본학습을 시작할 수 없다.** 처리량은 약 32.5 s/update이다. 이 속도면 S1 1 epoch에 약 14일이 걸린다. `total_steps = 500000`은 약 14 epoch(약 190일)이고, 학습기는 데이터가 떨어지면 멈추므로 실제로는 1 epoch에서 끝난다. 학습 길이와 처리량 개선은 사용자 결정이 필요하다(아래 "남은 결정").
- **오프라인 평가 경로가 없다.** 041 오프라인 지표(대상 맵 test top-1 등)를 계산하는 명령이 없어서 041-AC1과 AC3(ablation 비교)을 판정할 수 없다.

## 점검 목록 (spec 041 기준)

| 항목 | 상태 |
|---|---|
| R1 손실·헤드 가중치, R3 AdamW·warmup·cosine·clip, R7 value(win), R8 z 25% drop | 구현됨(`config/bc.toml`) |
| R2 T=64 연속 구간, Core 상태 이어 쓰기 | 구현됨 |
| R4 S1→S2, S1 체크포인트 보존, 학습률 1/10 | **고침:** `--init <ckpt>` 추가. 가중치만 가져오고 옵티마이저·스케줄·step은 새로 시작한다. 기존에는 `--resume`밖에 없어서 S1 step 수가 그대로 이어졌다. 그래서 S2가 바로 끝나거나 스케줄이 cosine 끝에서 시작했다. |
| R5 ablation (A) 대상 맵만 | `--stage s2`(기본 학습률)로 가능 |
| R9 로더(하위 폴더, manifest 검증), R12 분할, R13 S1/S2 선택·셔플 | 구현됨(f9ed4e4, eb9b311) |
| R10 하이퍼파라미터 체크포인트 기록 | 구현됨(`meta.json`, `train_state.json`) |
| 오프라인 지표(top-1/top-5, delay MAE, target_loc 오차, selected F1), AC1, AC3 | **미구현:** 평가 명령이 없다 |
| AC2 plan | 동작 확인. **고침:** plan의 체크포인트 경로를 `checkpoints/bc/s2`에서 `checkpoints/bc/s2/last`로 바꿨다. `bc`는 `<out>/last`에 저장하기 때문이다. |
| 학습 길이 | **결정 필요:** `total_steps`는 S1·S2·(A)에 같은 값이다. 학습기는 데이터가 떨어지면 멈추고, 여러 epoch를 도는 기능이 없다. |
| resume 시 데이터 위치 | 저장하지 않는다. 재개하면 shard 순서를 처음부터 다시 읽으므로 앞쪽 게임을 다시 학습한다(이번 실행에서 손실 값으로 확인). |
| 체크포인트 주기 | `checkpoint_every = 5000`이고, 지금 속도로 약 45시간이다. bc에는 Ctrl+C 저장이 없다. |
| PC 간 이어 하기(061) | `bc`는 `--out` 로컬 경로에 저장한다. RL처럼 공용 드라이브 잠금과 push를 쓰지 않는다. |

## 표본 실행 (이 PC, RTX 4070 Ti SUPER 16GB, CUDA f32, `micro_lanes = 1`)

| 실행 | 내용 | 결과 |
|---|---|---|
| S1 | `--dataset` 4개, 12 update | 정상, `last` 저장 |
| resume | `--resume s1/last`, 18 update까지 | step·games 이어짐, 데이터는 처음부터 다시 |
| S2 | `--s2 --init s1/last`, 4 update | step 0부터, `lr_scale 0.1` |
| AC2 경로 | 표본 S2 체크포인트 대 T0 T/P/Z, 6판, 동시 3게임, `max_frames 24000` | 6/6 Finished, 세션 오류·시야 위반·거부 0, 맵 사전계산 일치 6/6, 258초 |

표본 체크포인트는 학습량이 거의 없어 승패는 의미가 없다(0승 4패 2무, 무승부는 frame 상한 때문).

## 측정

| 항목 | 값 |
|---|---|
| update 시간(GPU 단독) | 31.8–33.2 s, 약 32.5 s/update |
| update 시간(다른 세션 RL과 GPU 공유) | 50–60 s |
| update당 샘플 | 레인 32 × T64 = 2,048(유효 약 1,970), 약 61 유효 샘플/s |
| 시작 시간 | 75–90 s(manifest 검증 4개 소스, 첫 update 커널 컴파일 포함) |
| `micro_lanes = 4` | GPU 공유 상태에서 1–6분/update로 오히려 느렸다. 1을 유지한다. |
| GPU 메모리 | `nvidia-smi` 5.6–15.9GB. CubeCL 풀이 미리 잡아 두므로 실제 필요량은 알 수 없다(`reports/m5_bc_memory.md`와 같음). |
| 데이터 로드 | 로컬 디스크 manifest 검증 약 26 s. 학습 중에는 GPU 사용률 100%로, GPU가 병목이다. |
| AC2 경로 결정 지연 | p50 84–91 ms, p95 167–283 ms(RL 세션과 GPU 공유) |

## 본학습 예상 시간 (GPU 단독, 32.5 s/update)

데이터는 records의 `steps` 합(written)으로 계산했다. 전체 81.25M 결정이고, 디싱크 판정 수정으로 추가되는 약 96개 시점을 반영하면 약 81.5M(35,966게임)이다. 분할 비율은 게임 수 기준이다.

| 단계 | 결정 수 | 1 epoch update | 1 epoch 시간 |
|---|---|---|---|
| S1 (전 소스 train, 91.4%) | 약 74.5M | 약 36,400 | **약 13.7일** |
| S2 (대상 맵 train, 20.0%) | 약 16.3M | 약 8,000 | 약 3.0일 |
| ablation (A) (S2 데이터, 처음부터) | 약 16.3M | 약 8,000 | 약 3.0일 |

- 1 epoch씩 하면 S1 + S2 + (A)가 이 PC 하나로 약 20일이다. (A)를 bc250_1(Vulkan)에서 동시에 돌리면 약 17일이다. 단, bc250_1의 Vulkan 속도는 아직 측정하지 않았다.
- 지금 설정 `total_steps = 500000`은 약 13.7 epoch이고 약 190일이다. 그런데 학습기는 1 epoch(약 36,400 update)에서 데이터가 떨어져 멈춘다. 이때 cosine 스케줄은 500,000 기준이라 학습률이 거의 최대값인 채로 끝난다.
- 041-AC2 600판: 표본은 6판에 258초였다(동시 3게임, 상한 24,000 frame). 실제 BC가 강해지면 게임이 길어진다. 대략 7시간 이상이고, 측정 전 추정이다.

## 본학습 실행 명령 (승인 후)

`<shared>` = `URANIUM_SHARED_DIR`. 데이터 인자는 네 단계 모두 같다.

```
DATA="--dataset replays/shards/stardata --dataset replays/shards/community --dataset replays/shards/scr_ladder --dataset replays/shards/personal"
# S1
uranium-train bc $DATA --out <shared>/checkpoints/bc/s1 --backend cuda [--steps <S1 update 수>]
# S2 (041-R4)
uranium-train bc $DATA --s2 --init <shared>/checkpoints/bc/s1/last --out <shared>/checkpoints/bc/s2 --backend cuda --steps <S2 update 수>
# ablation (A) (041-R5)
uranium-train bc $DATA --stage s2 --out <shared>/checkpoints/bc/ablation_a --backend <cuda|vulkan> --steps <(A) update 수>
# 041-AC2 (plan의 checkpoint = checkpoints/bc/s2/last)
uranium-env run --plan config/plans/bc-vs-t0.toml --backend cuda
```

- 지금은 `--steps`가 없으면 `total_steps = 500000`이고, 1 epoch에서 데이터가 떨어져 멈춘다.
- 게임 호스트 브리지는 디싱크 판정 수정(f74bc54) 이후 빌드를 쓴다. 이번 AC2 표본은 그 전 브리지로 돌렸고, feature에는 영향이 없다.

## 남은 결정·작업
1. **학습 길이(사용자 결정).** S1·S2·(A)의 update 수 또는 epoch 수를 정해야 한다.
   - 기본안(미확정): S1 1 epoch(약 36,400 update, 약 14일), S2와 (A) 각 1 epoch(약 8,000 update). `total_steps`를 그 값으로 맞춰 cosine이 끝까지 내려가게 한다.
   - 여러 epoch를 원하면 epoch 반복(epoch마다 seed를 바꿔 셔플)을 구현해야 한다.
2. **처리량(사용자 결정).** 14일을 줄이는 방법은 다음과 같다.
   - f16 학습(032-R8): Burn에 AMP가 없어서 구현이 필요하다.
   - 모델 축소나 레인 조정: 결과가 달라지는 튜닝이다.
   - 구간 표본 추출: 데이터 일부만 쓴다.
   - (A)는 bc250_1로 분리한다.
3. **오프라인 평가 명령 구현(spec 정의, 결정 불필요).** 대상 맵 test 셋에서 top-1/top-5, delay MAE, target_loc 오차, selected F1을 계산한다. AC1·AC3 판정 전에 필요하다.
4. **장기 실행 안전장치(제안).**
   - resume 시 데이터 위치 저장
   - 체크포인트 주기를 약 1시간(`checkpoint_every ≈ 100`)으로
   - bc Ctrl+C 저장
   - 다른 PC에서 이어 하려면 061 공용 저장소(잠금·push) 연동
   - 약 14일 실행이라 중단 가능성이 높다.

# 교사 모드 누출 수정 (교사 라운드 t2 → `m2-rl-fix-3`)

spec [044](../../spec/2026_11_14_08_18_044-self-game-distillation.md) R8 ⑤·R9·R11·R13, [050](../../spec/2026_10_15_22_24_050-evaluation.md) R9·R10. 결정은 사용자 결정 2026-11-17. 메인 임시 결정이던 세부([AMB-044-12](../../spec/README.md))도 2026-11-17 사용자 결정으로 확정(저그 판 600·절단 10:00).

## 용어
| 이름 | 뜻 |
|---|---|
| T0 / T1 | 자체 규칙 기반 봇 / 공개 BWAPI 봇([021](../../spec/2026_10_15_22_24_021-opponent-pool.md)) |
| 022 교사 | 테란·프로토스전 스크립트 교사([022](../../spec/2026_11_14_08_18_022-matchup-teacher.md)) |
| 저그전 교사 | 021-R8 정찰 반응형 테란 교사([021](../../spec/2026_10_15_22_24_021-opponent-pool.md)) |
| gn | `m2-gn-t1e2/ckpt-2844`, 다음 작업 바탕 가중치([gn_resume](2026_11_17_10_05_gn_resume.md) 4절) |
| t1 | 교사 라운드 1 `m2-teacher-t1/ckpt-2844` = Checkpoint 2([model2](../plan/2026_11_09_22_18_model2.md) 10절) |
| d1 | 증류 라운드 1 `m2-distill-d1`([044](../../spec/2026_11_14_08_18_044-self-game-distillation.md) R5) |
| c4 | `m2-rl-fix-2` 마지막 점검 체크포인트 `ckpt-6129424`([model2](../plan/2026_11_09_22_18_model2.md) 11.2절) |
| `held_8m` | 러시 판 8:00 막음([measurement_protocol](../../guide/measurement_protocol.md) "지표 정의") |
| B1·S0b | 분기 게임 러시 파일럿([051](../../spec/2026_11_12_12_39_051-branch-games.md))·그 분석에 쓰는 계획형 JEPA 예측기([jepa_planning](../plan/2026_11_11_14_56_jepa_planning.md) 7절) |
| F | 누출 수정 최종 가중치(044-R13 ②) |

## 1. 배경 (실험 전, 2026-11-17 20:0x 작성)

### 문제
- gn·t1의 테란·프로토스전 일부 판이 저그전 교사 빌드로 둔다([gn_resume](2026_11_17_10_05_gn_resume.md) 4절 교사 모드 판 비율).
- 기존 판별(교사 모드 판, [050-R10](../../spec/2026_10_15_22_24_050-evaluation.md))은 이 현상을 022 교사 방식과 가르지 못한다.

### 발견
근거: `local/work/teacher_leak/`(`builds.json`·`builds_c4.json` = 판별 명령 스트림 요약, `builds.py` = 추출, `rules*.py` = 판별 비교).

| 항목 | 값 |
|---|---|
| 0~6분 분당 명령(테란·프로토스전) | gn·t1 약 70, 022 교사 68, 사람 방식 `m2-s2e2` 약 195, d1 약 215~230 |
| gn·t1 테란·프로토스전 | 대부분 022 교사 방식(분당 명령 < 150: gn T0 60/64·T1 180/196, t1 T0 58/64·T1 177/194) |
| 기존 판별에 걸린 12~28% 판 | 저그전 교사 빌드(가스 없음, 배럭 3) |
| 두 교사 빌드 | 약 100~115초까지 같음 |
| 기존 판별의 분당 명령 < 150 | 리파이너리 없는 판에서 거의 늘 참(gn T0 10/11) → 사실상 리파이너리 조건 하나 |
| `m2-rl-fix-2`(테란·프로토스 KL 참조 `m2-s2e2`) | 20~50 update 안에 `m2-s2e2` 방식으로 끌려감(KL 5.7/5.5 → update 21에 0.25), 022 방식 잃음 |
| 022 방식 판(T0 64판) | c4 1/64, gn 50/64 |

gn 기준 값(새 판별, 테란 + 프로토스):

| 단계 | `zerg_build` | `tp_style` | 승 |
|---|---|---|---|
| T0 표준 64판 | 10 | 50 | 36 |
| T1 196판 | 26 | 150 | 27 |

### 목표
- 테란·프로토스전은 022 교사 방식 유지, 저그전 교사 빌드로 빠지는 것만 없앰.
- 결과는 모든 종족에 한 가중치.

## 2. 실험 전 예상
- t2: 저그전 교사 빌드로 빠진 상태에 022 교사 레이블이 붙어 `zerg_build`가 줄어듦.
- RL(참조 = t2): 022 방식 유지(`m2-rl-fix-2`처럼 `m2-s2e2` 방식으로 끌려가지 않음).
- 위험: TvZ·러시 방어 후퇴, 데이터 종족 비율 흔들림(저그 DAgger 판으로 막음).

## 3. 계획

### 판별 ([050-R10](../../spec/2026_10_15_22_24_050-evaluation.md), [measurement_protocol](../../guide/measurement_protocol.md) "지표 정의")
| 이름 | 정의 |
|---|---|
| `zerg_build` | 0~6분 분당 명령 < 150, 5:00(프레임 7,200) 전 리파이너리·팩토리 지시 없음 |
| `tp_style` | 0~6분 분당 명령 < 120, `zerg_build` 아님 |
| 사람 방식 | 나머지 |
- 기존 교사 모드 판은 기록 필드로 남김. T0·T1 단계, 종족별 수와 승·무·패.

### 순서
1. t2 표본: 20판 → 재레이블 → 50 update.
2. t2 본 실행([044-R8](../../spec/2026_11_14_08_18_044-self-game-distillation.md) ⑤, 두 plan 모두 10분 상한):
   - 테란·프로토스 300판(gn stochastic, t1 구성, seed n = 301..600) → 022 교사 그림자 재레이블 `--relabel terran_tp`(10:00 절단) → `dagger_m2/t2`.
   - 저그 DAgger 판 N판(아래 표) → 저그전 교사 그림자 `--relabel terran_react`(`terran_tp`는 로비 저그 판을 거부) → `dagger_m2/t2z`. 절단 10:00(사용자 결정 2026-11-17, [AMB-044-12](../../spec/README.md)).

     | 항목 | 값 |
     |---|---|
     | 생성 | `python tools/eval/rush_check_plan.py --teacher-round --round t2 --zerg 600 --checkpoint checkpoints/m2-gn-t1e2/ckpt-2844` → `config/plans/dagger_m2_t2_z.toml` |
     | 구성 | 맵당 러시 4상대(T0 `zerg_rush`, UAlbertaBot `12poollings`·`9pool10lings`·`9poolhydra`) × 40 + 운영형 5상대(UAlbertaBot `2hatchhydra`·`3hatchmuta`, Steamhammer `2hatchmuta`, T0 `zerg_muta`, T0 표준) × 28, 두 맵(`dagger_m2_r1.toml` 구성) |
     | seed | `splitmix64(5,000,000 + n) >> 32`, n = 1..3000 예약 |
     | N | 600(사용자 결정 2026-11-17). 어림: 테란·프로토스 t2 추가 뒤 저그 46.3% → 부족 약 31만 step, 저그 판당 학습 step 약 533(10:00 절단) → 약 580판, 600판이면 50.3% |
     | N 출처 | t1 `mix-fractions` 출력 `<shared>/eval/m2-lineage/logs/t1_t1_mixfrac2.json`(종족별 스텝), 저그 판당 step은 경로 2 r1–r3 기록·판 생성 job `results.jsonl`의 `end_frame` |

   - 학습 `m2-teacher-t2`(`--init` gn, `data.seed` 18, 교사 + dagger 65 / 사람 35, 저그 : 프로토스 : 테란 50 : 30 : 20).
3. 지점 `t2` 측정 → 합격 기준 판정. 실패면 멈추고 사용자에게 보고.
4. 통과면 바로 `m2-rl-fix-3`([044-R11](../../spec/2026_11_14_08_18_044-self-game-distillation.md)): `--bc` t2, 참조 네 키 = t2, 600만 agent step, 100 update 표본 먼저.
   - 가벼운 점검: 0.3M 이후 첫 체크포인트. 전체 점검: 150만·300만·450만·600만, 끝 T1 300. 모든 점검에서 `zerg_build`·`tp_style`.
   - 점검·끝 측정은 RL 조건(`zero_last_failed` false) → BC 조건인 gn·t2와 비교할 때 조건이 다름.
   - 중단(t2 기준): T0 테란 또는 프로토스 6판 이상 하락, `tp_style` T0 < 32/64, TvZ < 16/32.
   - 세부(사용자 결정 2026-11-17, AMB-044-12): T0 하락은 두 점검 연속, 가벼운 점검 = T0 테란·프로토스 64판 + 방식 판 수이고 T0 하락·`tp_style` 규칙의 점검으로 셈, TvZ 규칙은 TvZ를 재는 점검에만.
5. 최종 가중치 F 선택([044-R13](../../spec/2026_11_14_08_18_044-self-game-distillation.md) ②): RL 끝 > t2 > gn 유지·보고. 각 run의 끝 체크포인트. RL 끝은 합격 기준 통과 + T1 테란 + 프로토스 승 ≥ t2 − 6(기준 폭 6판, 사용자 결정 2026-11-17).
- 050-R10 게이트·교사 모드 ≤ 3/32는 t2·`m2-rl-fix-3`에서 기록만(사용자 결정 2026-11-17, AMB-044-12).

### 합격 기준 (gn 대비, [044-R13](../../spec/2026_11_14_08_18_044-self-game-distillation.md) ①)
| 지표 | 기준 | gn |
|---|---|---|
| `zerg_build` T0 테란 + 프로토스 | ≤ 2/64 | 10 |
| `zerg_build` T1 테란 + 프로토스 | ≤ 6/196 | 26 |
| `tp_style` T0 | ≥ 44/64 | 50 |
| `tp_style` T1 | ≥ 140/196 | 150 |
| 승 T0 테란 + 프로토스 | ≥ 30/64 | 36 |
| 승 T1 테란 + 프로토스 | ≥ 21/196 | 27 |
| TvZ T0 | ≥ 20/32 | 26 |
| 러시 막음 합(이른 + 늦은, `held_8m`) | ≥ 48 | 54 |
| T1 저그 상대 승 | ≥ 14/104 (gn − 8) | 22/104 |
- T1 저그 gn 값: `<shared>/eval/m2-gn-20261117-t1/`(`m2-gn-t1e2-t1-bc`) UAlbertaBot 11·Steamhammer 9·ZZZKBot 0·McRave 2승 / 104판.
- 검출력: 32판 칸 약 35%p, 64판 약 25%p, T1 196판 약 10%p 차이부터.

### 계획형 JEPA와의 관계
- B1 체크포인트 = gn([051-R1](../../spec/2026_11_12_12_39_051-branch-games.md) ②), S0b는 gn으로 다시 학습. F가 gn과 다르면 S0b·B1 줄기 feature를 F로 다시(게임은 그대로, TvZ 개막 결정이 뚜렷이 다르면 사용자에게 물음).

## 4. 진행 기록
- 22:2x — 결정(사용자): 저그 DAgger 판 600, 저그 판 절단 10:00, 메인 임시 결정 세부 모두 확정([AMB-044-12](../../spec/README.md)). plan `config/plans/dagger_m2_t2_z.toml`.
- 22:54 — t2 표본 끝까지 통과: 테란·프로토스·저그 판 → 재레이블 → BC 50 update(`<shared>/eval/t2-sample-20261117/`, 체크포인트 `m2-teacher-t2-sample/ckpt-50`).
- 23:00~23:31 — t2 판·재레이블(`<shared>/eval/dagger-m2-t2-20261117/`): 테란·프로토스 300판 중 299(1판 Iron 런처 비정상 종료, [050-R9](../../spec/2026_10_15_22_24_050-evaluation.md) 규칙으로 뺌), 저그 600판 중 575(재생 불일치 25판 = 4.2%, 이전 라운드 12/1,500보다 많음, 상대·호스트에 몰리지 않음).
- 23:32 — `mix-fractions`: 교사 + dagger 몫 0.6496, 그 안 저그 51.1%·프로토스 29.5%·테란 19.4%(`logs/t2_mixfrac.json`). 설정 `config/bc_m2_teacher_t2.toml`.
- 23:32~00:56 — BC `m2-teacher-t2` 3,258 update(Windows CUDA 호스트, update당 1.46초), 체크포인트 `ckpt-3258`. 마지막 100 update 손실 total 6.30(처음 100 5.95, t1도 같은 모양). 로그 `<shared>/eval/m2-lineage/train_logs/m2-teacher-t2.log`.
- 11-18 01:00 — t2 측정 시작(지점 `t2`, gn과 같은 표, Windows CUDA 호스트만 — 분기 게임과 장비 분리 [051-R14](../../spec/2026_11_12_12_39_051-branch-games.md)): `<shared>/eval/m2-t2-20261118-{t1,tvz-t0_tp,rush-macro}/`.

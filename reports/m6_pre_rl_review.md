# RL 착수 점검 (M6, 042·033·050)

- 작성: 2026-10-31. BC가 끝난 시점의 상태와, RL 본 실행(042-AC1) 전에 남은 사용자 결정과 작업을 모았다.
- 방법: spec(README 레지스트리, 042·033·050·041·060·020), `m6_action_audit.md` §11, `m6_pre_bc_review.md` F-2-7, `m6_rl.md`, `m6_bc_eval.md`, timeline을 대조했다. 태그만 보지 않고 미이행 약속과 미실행 AC도 봤다.
- 레지스트리 상태:
  - 열린 AMB는 셋이다: AMB-042-2 Deferred(M6), AMB-012-6 Deferred(M7), AMB-001-1 Deferred(M7).
  - spec 본문에 `[SPEC AMBIGUOUS]`는 없다.
  - 아래 결정은 대부분 TBD(M6)이거나, 리포트에 "사용자 결정 필요"로 남은 것이다.

## 1. 지금 상태
- **BC 끝**(`m6_bc_eval.md`):
  - 첫 epoch: `bc-s1-pt` → `bc-s2-pt`
  - 두 번째 epoch(041-R20): `bc-s1e2-pt` → `bc-s2e2-pt`
- **RL 출발점 후보**: `bc-s2e2-pt/ckpt-11387`

| 모델 | test top-1 | NLL action | T0 stochastic 300판 (테란 / 프로토스 / 저그) | 무승부 |
|---|---|---|---|---|
| `bc-s2-pt/ckpt-11387` | 71.58% | 0.901 | 29% (17 / 16 / 55%) | 38% |
| `bc-s2e2-pt/ckpt-11387` | **72.15%** | **0.871** | **39%** (30 / 22 / 66%) | 38% |

- 투혼만 본 `bc-s2e2-pt` stochastic 승률(종족당 50판, 승 / 패 / 무, Wilson 95%):
  - 테란: 12 / 25 / 13, 24% (14–37)
  - 프로토스: 10 / 11 / 29, 20% (11–33)
  - 저그: 33 / 2 / 15, 66% (52–78)
  - 첫 epoch `bc-s2-pt`는 테란 6%, 프로토스 14%, 저그 52%였다.
- argmax는 첫 epoch S2에서 300판 전패였다(랠리 연타 고리, `m6_core_ablation.md` 결과 3). 두 번째 epoch argmax 대전은 하지 않았다.
- 장비 (10-31):
  - 모든 장비가 비어 있다.
  - 학습 PC(Windows CUDA)는 사용자가 다른 용도로 쓸 때가 있다. 쓰기 전에 사용자 허락을 받는다.
  - 패치 0015(011-R22, 승자 불일치 수정)는 Linux 호스트 셋에는 배포했고, **학습 PC에는 아직 안 했다**.
- RL 코드는 10-18에 master에 반영했다(`m6_rl.md`). 이후 PyTorch 전환과 바깥 LSTM 위상별 계산(032 §2.7)도 반영했다.
  - 이미 있는 것: `--max-agent-steps`(042-R12), 리플레이 표본·불일치 저장(042-R11), RL dropout 0, `last_failed` 실제 값(030-R10)

## 2. 사용자 결정 (A)
물을 때는 spec 용어 대신 상황, 예시, 선택지 차이로 설명한다.

| # | 결정할 것 | 정의 위치 | 기본안·참고 | RL 시작을 막나 |
|---|---|---|---|---|
| A1 | **RL 출발 체크포인트**(KL teacher와 같음): `bc-s2e2-pt` 또는 `bc-s2-pt` | 041-R20 ④ | Claude 추천 `bc-s2e2-pt`(모든 지표 우위, 승률 +10%p, z 약 2.6) | **막음** |
| A2 | **042-AC1 상대 T0 종족**(투혼, "BC 승률 20~80%"인 한 종족) | 042-AC1 TBD(M6) | A1이 `bc-s2e2-pt`면 투혼 stochastic 기준 셋 다 범위 안이다(테란 24%, 프로토스 20%는 경계, 저그 66%). `bc-s2-pt`면 저그만 범위 안 | **막음** |
| A3 | **AC1 판정 샘플링 모드**: 050-R1은 두 모드를 보고하라고만 한다. argmax는 0%라 "BC 승률" 기준이 성립하지 않는다 | 042-AC1, 050-R1 | 없음(spec에 없음, AMB 등록 후보). 예: 판정은 stochastic, argmax는 보고만 | **막음**(AC1 해석) |
| A4 | argmax 랠리 연타 고리를 고칠지(반복 명령 병합 등). 10-19에 "040-R4 반복 병합 없음"으로 정했다 | `m6_bc_eval.md` 4절, `m6_core_ablation.md` 결과 3 | "학습량 부족" 판단만 있다(미확정) | 학습은 안 막음(RL actor는 stochastic). A3, SC:R 기본 모드와 이어짐 |
| A5 | 041-AC2 미통과(매치업별 50% 이상은 저그만)를 받아들일지. 041-R6 클래스 가중치 재검토 조건("성능이 너무 낮으면")에 해당하는지 | spec/041 R6·AC2 | 없음 | 안 막음(확인 권장) |
| A6 | 041-AC4 항복 판정 기준값 | spec/041 AC4 TBD(M6) | 측정값: test 오항복 0/17, 대전 성립 98회 중 실제 승 3회 | 안 막음(OpenBW RL은 shadow). SC:R 실전에서 항복을 켜기 전에 필요 |
| A7 | 액션 감사 남은 결정 4개(`m6_action_audit.md` §11) | 같은 곳 | 아래 표 | 마스크는 BC·RL·실전이 같아야 한다(031-R4). 시작 전에 정하는 것을 권장 |
| A8 | 060-R8 주 장비 규칙(장비 전환·재개)을 RL에 어떻게 적용할지 | spec/060 R8("M6 RL 구현에서 정한다"), 042 | 없음. 지금 RL은 `supervise`만 있고, `worker` 큐는 BC 전용 | AC1은 CUDA 호스트 고정이라 안 막음. 긴 리그 전에 필요 |
| A9 | AC1을 z 테이블(pseudo-reward)로 할지, z 없이 할지. AC1 설정에 z 언급이 없다 | 042-AC1, 033-R8·AC2 | 없음(불확실) | B1과 이어짐 |
| A10 | **042-AC1 6 run 실행 승인**: V-trace·PPO 각 3 seed, seed당 1,000만 agent step. 처리량을 잰 뒤 예상 시간을 보고하고 승인받는다 | 042-AC1 규모, "승인 전 시작 금지" | — | **막음** |
| A11 | AMB-042-2 TvP·TvZ 상대 다양성 | spec/042, Deferred(M6) | T0·T1로 시작(10-17 재확인) | 안 막음. M6 중간 점검 때 다시 묻는다 |
| A12 | AMB-012-6 SC:R `CommandFailed` | 012-R14, Deferred(M7) | — | RL과 무관. RL 모델을 SC:R에 올리기 전에 묻는다 |

**A7 세부** (`m6_action_audit.md` §11)

| 항목 | 내용 | 선택지 |
|---|---|---|
| U-1 | Train 서플라이 마스크 | (a) 엔진 규칙대로 / (b) 현행(BWAPI 기준) |
| U-2 | 시야 밖에서 기억하는 적 건물을 공격 대상으로 허용할지 | 정리된 선택지 없음 |
| U-3 | 무기 없는 유닛, 사거리 밖 대상의 Attack_Unit 처리 | 정리된 선택지 없음 |
| U-6 | `CommandFailed`에 유닛별 거절 수를 넣을지(PROTOCOL_VERSION 올림) | 정리된 선택지 없음 |

## 3. 구현·측정 작업 (B)

| # | 할 일 | 위치 | RL 시작을 막나 |
|---|---|---|---|
| B1 | z 테이블 만들기(`uranium-train z-table`). 아직 실행 기록이 없다. 없으면 모든 게임이 z 없이 돌고 로그만 남는다 | 033-R8, `m6_pre_bc_review.md` F-2-7 | 033-AC2에는 필수. AC1은 A9에 따름 |
| B2 | **실제 BC 체크포인트로 RL 표본 end-to-end**(CUDA 호스트). 모두 켬 + 바깥 LSTM 위상별 + win_open 모델이다. 지금까지는 random-init CPU 스모크(10-24)와 Burn 스모크(10-18)만 했다. 잴 것: 정확 KL용 BC 재forward 메모리(부족하면 보고, AMB-042-9), update 시간 | 042("042-AC1 준비 때 잰다"), 대규모 배치 규칙 | **막음** |
| B3 | 처리량 측정과 run 시간 재보고. 추론 `max_batch`·`max_wait_ms` 기본값도 잰다(현재 32 / 5 ms, `python/uranium/rl/run.py`) | 042-AC1 TBD(M6), 020 TBD(M6) | **막음**(A10 전에 필요) |
| B4 | calibrate(020-R10): 학습 PC 로컬 hostd 동시 게임 수. 그다음 원격 Linux 게임 호스트 추가(AMB-042-8) | F-2-7 | 막음(동시 게임 수) |
| B5 | 패치 0015를 학습 PC(Windows)에 배포. `tools\openbw\build.ps1` → 브리지 → hostd 재시작. 무거운 빌드라 PC가 비어 있고 사용자가 허락할 때 한다 | 011-R22, timeline 10-29 | **막음**(AC1은 학습 PC 로컬 hostd) |
| B6 | 승격 게이트: 직접 대전 55% 이상 + E-Regression. RL 루프에는 게이트 코드가 없고 `uranium-store` `promote()`만 있다. `e-regression.toml`의 p0이 아직 T0 대역인지 확인해야 한다 | 042-R6, 042-AC3, 050-R3, AMB-050-2 | AC1에는 불필요. 리그 C2 전에 필요 |
| B7 | 050 평가 파이프라인: E-Matrix·E-Elo, `reports/eval/<ckpt>.md` 자동 생성, G1~G3 요약. 지금은 020-R15 평가 작업(`summary.md`)만 있다 | 050-R4, 050-AC1·AC2 | AC1 전후 200판은 평가 작업으로 된다. 게이트 전에 필요 |
| B8 | RL에서 승자 불일치가 0인지 확인. 원인과 수정(패치 0015)은 끝났다. 불일치 리플레이 저장은 구현돼 있다 | F-2-7, 042-R11 | 안 막음 |
| B9 | RL 장비 전환 구현 | 060-R8(A8 뒤) | 긴 리그 전에 필요 |
| B10 | 042-AC2(72시간 연속), 042-AC4(Ctrl+C 뒤 호스트를 바꿔 재개), 060-AC3(24시간 스모크) | spec/042·060 | 안 막음(리그 중 검증) |
| B11 | 033-AC2 pseudo-reward ablation 3 run | spec/033, AMB-042-7 | B1 뒤. AC1은 안 막음 |
| B12 | 승패 헤드가 약하다(win_open MSE 0.86~0.93, 거의 상수). "RL 전에 다시 볼 것" | timeline 10-27, `m6_win_loss_check.md` §5 | 안 막음(조사) |
| B13 | 후속 제안: 브리지 시작 실패 판이 `Finished`로 기록됨(020). GTrXL 큰 장면 추론 속도 | `m6_core_ablation.md` 10-23 후속 ①④ | 낮음. ④는 B3과 이어짐 |

## 4. 이미 정한 것 (다시 묻지 않는다)
- 설정·규모·채택 규칙:
  - RL 초기 설정값 전부(`config/rl.toml`, AMB-042-3)
  - V-trace vs PPO 채택 규칙(042 채택 기준)
  - AC1 규모: V-trace·PPO 각 3 seed, seed당 1,000만 step(AMB-042-7, 042-R12)
- 장비·실행 구성:
  - 첫 RL은 CUDA 호스트 + 로컬 hostd(042, AMB-042-8)
  - RL 착수 전 검증 정책(AMB-042-5)
  - behaviour log-prob만 기록(AMB-042-4), BC 재forward로 정확한 KL(AMB-042-9)
  - 한 프로세스 구성(AMB-042-10), Aborted 회수(AMB-042-6), Ctrl+C 처리(042-R10)
- 승격:
  - 범위는 직접 대전 + E-Regression(AMB-050-2)
  - 게이트 수치(AMB-050-1)
- 승패·보상:
  - 승자 불일치는 늦게 끝난 세션 기준(AMB-020-3), 패치 0015(011-R22)
  - 60분 무승부는 보상 0(AMB-020-2). BC의 무승부 38%도 이 규칙으로 처리한다
- 기록·모델:
  - RL 리플레이 보존, 지우지 않음(042-R11, AMB-061-5)
  - win_open RL 학습, OpenBW 항복은 shadow(042-R13, AMB-031-6)
  - RL부터 `last_failed` 실제 값(030-R10, AMB-030-2)
  - 모델: GTrXL + 네 스위치 모두 켬(032-R20 ④), RL 바깥 LSTM 위상별(032 §2.7), T=64(10-27), RL dropout 0
- 범위:
  - 학습 기간 제한 없음, 전체 리그(AMB-042-1)
  - AMB-042-2는 M6 시작 재확인(10-17)을 마쳤고, 중간 점검 때만 다시 묻는다
- 이미 끝난 운영 결정: PyTorch 전환(AMB-000-4), 041-R20 운영 결정(자동 멈춤 없음, 최신 체크포인트 채점 등)

## 5. 권장 순서 (기본안, 미확정)
1. 사용자 결정을 받는다: A1·A2·A3·A7(마스크), 필요하면 A9. 결정은 spec 본문에 반영하고, 새 AMB는 레지스트리에 `Resolved`로 남긴다.
2. A7에서 마스크나 프로토콜을 바꾸기로 하면 그 구현을 먼저 한다. spec 수정 → 사용자 확인 → 코드 → 별도 리뷰어 리뷰 순서다. BC 데이터·모델과 맞아야 하는지도 확인한다(031-R4).
3. 학습 PC가 비고 사용자가 허락하면 B5(패치 0015 배포)와 B4(calibrate)를 한다.
4. B1(z 테이블, A9에 따라) → B2(실제 체크포인트로 RL 표본 end-to-end) → B3(처리량, 시간 재보고)
5. A10 승인을 받고 042-AC1 6 run을 시작한다. 학습 전후 200판 평가는 020-R15 평가 작업으로 한다.
6. 리그(C2)로 가기 전에 B6·B7·A8/B9를 한다.

## 6. 참고 위치
- BC 결과: `m6_bc_eval.md`(1~5절), `bc_ablation.md`
- RL 코드·스모크: `m6_rl.md`
- 액션 감사: `m6_action_audit.md` §11
- BC 착수 점검(남은 RL 일 F-2-7): `m6_pre_bc_review.md`
- 평가 산출물:
  - `<shared>/eval/bc-main/`: 첫 epoch S2 600판
  - `<shared>/eval/bc-e2-final/`: 두 번째 epoch S2 300판
  - `<shared>/eval/bc-e2-games/`: 체크포인트별 100판
  - `<shared>/artifacts/bc_main_eval/test_cuda/`: test 채점
  - `<shared>/artifacts/bc_e2_val/val500/`: val 500 채점
- 운영 명령: `docs/commands.md`(평가 작업 020-R15, 작업자 061), 루트 `README.md`

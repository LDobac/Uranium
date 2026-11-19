# 보조 예측기만 AdamW eps 1e-8 비교 (안 B)

- 근거: [aux_jepa_adam_eps](../investigation/2026_11_18_19_12_aux_jepa_adam_eps.md) 5절 안 B·7절 진단, spec [041-R3·R23 ⑦·AC8](../../spec/2026_10_15_22_24_041-imitation-learning.md)(사용자 결정 2026-11-19), RL 거부 [042-R21](../../spec/2026_10_15_22_24_042-rl-training.md) ①
- 설정: 기준 A = `config/bc_m2_teacher_t2.toml` 그대로, B = `config/bc_m2_aux_eps_b.toml`(A + `[aux] adam_eps = 1e-8`)

## 배경·예상 (실행 전, 2026-11-19 11:36)
- 배경: 보조 모듈(`aux_jepa`·`aux_priv`) 원소 거의 전부가 √v̂ < eps 1e-5라 step이 정상 Adam의 약 1/5~1/15([aux_jepa_adam_eps](../investigation/2026_11_18_19_12_aux_jepa_adam_eps.md) 2절). 고정 몸통 위 재맞춤에서 eps 1e-8이 떼어 둔 판 aux_jepa를 0.031 → 0.022로 더 낮춤(같은 리포트 7절).
- 예상(추론):
  - B의 val500 aux_jepa가 A보다 낮음, aux_priv 차이는 작음.
  - val500 `nll.action`·top-1은 A와 잡음 안(몸통이 받는 보조 몫 0.1~1.4%는 eps와 무관, 같은 리포트 3절).
  - optimizer step 비: B 보조 그룹 ≈ 1, 몸통 그룹은 A와 같음. 보조 가중치 이동(gn 대비)은 B가 약 7~9배.
  - 위험: `aux_priv`의 아주 작은 v 원소(√v̂ p10 8e-10)가 lr 크기 step을 받아 B 로그 `aux_priv`가 튈 수 있음. bf16 gradient 잡음 정규화(미측정).

## 계획 (사용자 결정 2026-11-19, 초안 기본안 전부 채택)

| 항목 | 값 |
|---|---|
| 기준 run | A를 새로 돌림(같은 날·같은 코드, A − t2 = GPU 비결정성 폭) |
| 출발·일정 | gn(`m2-gn-t1e2/ckpt-2844`) `--init`, `--s2`, 3,258 update, `data.seed` 18 = t2 레시피 |
| 데이터 | t2와 같은 6개 출처, 2,173 파일. 시작 데이터 지문이 t2 `train_state.json`(`2173:5206f0cfb7d858c6`)과 같은지 확인 |
| 범위 | BC만. RL 설정의 키는 시작 오류 |
| 보조 그룹 lr·β·weight decay | 그대로(eps만 바뀜) |
| 판정 | 미리 정한 규칙 없음, 표로 보고 |
| 게임 평가 | 없음 |
| val500 장비 | Linux ROCm 호스트(고정 조건). 안 되면 Windows CUDA 호스트에서 A·B만 비교(t2 값과는 비교 안 함) |
| 고정 몸통 재맞춤 | 안 함 |

| run | 설정 | update | 장비 |
|---|---|---|---|
| 표본 `m2-aux-eps-b-sample` | B | 50(`--steps 50`) | Windows CUDA 호스트(bf16) |
| A `m2-aux-eps-a` | A | 3,258 | 같음 |
| B `m2-aux-eps-b` | B | 3,258 | 같음 |

- 실행(t2와 같은 형식, `--dataset` 6개 = `replays/shards/{stardata,community,scr_ladder,personal,botgen,dagger_m2}`): `python -m uranium bc --run <run> --backend cuda --core gtrxl --config <설정> --s2 --init <shared>/checkpoints/m2-gn-t1e2/ckpt-2844 --dataset …` (표본만 `--steps 50`)
- val500: `python -m uranium bc-eval --dataset replays/shards/{stardata,community,scr_ladder,personal}(4개) --checkpoint <run>/ckpt-3258 --split val --games 500 --seed 1 --backend rocm --config config/bc.toml --bunker-before 5760 --out <json>` → `python tools/eval/protocol.py check-bc-eval <json> --metric val500`([measurement_protocol](../../guide/measurement_protocol.md), `config/protocol/lineage.toml` `val500`과 같은 인자)
- 비교 지표

| 순위 | 지표 | 출처 |
|---|---|---|
| 1 | val500 `aux_jepa`(slot 1·2·3)·`aux_priv`·`jepa_tgt_std` | bc-eval 출력([041-R23](../../spec/2026_10_15_22_24_041-imitation-learning.md) ⑤) |
| 1 | val500 `nll.action`·top-1·헤드별 NLL·종족별 | 같은 출력 |
| 2 | 학습 로그 끝 200 update 평균(total·헤드·`aux_jepa`·`aux_priv`·`jepa_tgt_std`), `skipped_steps` | run 로그 |
| 2 | optimizer step 비·감쇠(보조·몸통 그룹) | `local/work/aux_eps/optim_state.py` |
| 3 | 보조 가중치 이동(gn 대비) | `local/work/aux_eps/drift.py` |
| 3 | 죽은 채널 경보 | `[monitor]` 로그 |

## 진행 기록
| 시각 | 일 |
|---|---|
| 11-19 11:36 | spec 041-R3·R23 ⑦·AC8, 042-R21 ① 개정과 구현·테스트(학습 시작 전) |
| 11-19 12:08 | 표본 A·B 50 update 끝(`m2-aux-eps-a-sample`·`m2-aux-eps-b-sample`), 이어서 A 시작 |
| 11-19 14:55 | A 끝(3,258 update, 평균 약 3.1초/update — t2 1.69초의 약 2배, A도 같아 param group 탓 아님, 원인 미확인) → B 시작 |
| 11-19 16:27 | A val500 시작(Linux ROCm 호스트 bc250_2, B3-H 게임만 돌고 GPU는 비어 있었음), 17:34 끝 |
| 11-19 17:41 | B 끝 → B val500 시작(같은 장비), 18:44 끝 |

## 결과
- 데이터: A·B 모두 시작 파일 지문 `2173:5206f0cfb7d858c6`(t2와 같음), 7,011판, 건너뜀 1,329. 죽은 채널 경보 없음.
- val500 3개 모두 `check-bc-eval` 통과(같은 1,100,146 step).

### val500(Linux ROCm f32, 학습 안 한 500판)
| 지표 | t2 | A | B | B − A |
|---|---|---|---|---|
| `aux_jepa` | 0.0448 | 0.0410 | **0.0216** | −47% |
| slot 1 · 2 · 3(k 1 · 8 · 32) | 0.036 · 0.045 · 0.053 | 0.032 · 0.041 · 0.050 | 0.015 · 0.022 · 0.028 | −52 · −46 · −45% |
| `aux_priv` | 0.00283 | 0.00283 | 0.00273 | −4% |
| `jepa_tgt_std`(층 정규화 뒤) · raw | 0.429 · 0.565 | 0.439 · 0.521 | **0.306** · 0.397 | −30% · −24% |
| `win_open_mse` | 0.968 | 0.961 | 0.935 | −0.026 |
| action top-1 · top-5 | 71.99% · 95.78% | 71.99% · 95.80% | 71.99% · 95.80% | 0 |
| `nll.action` | 0.8762 | 0.8758 | 0.8758 | 0 |
| `nll` delay · queued · target_unit · target_loc | 1.809 · 0.048 · 1.916 · 5.448 | 1.809 · 0.048 · 1.917 · 5.445 | 1.809 · 0.048 · 1.916 · 5.440 | 잡음 안 |
| `nll.selected`(F1) | 4.43 (0.851) | **1,319** (0.851) | 4.43 (0.851) | — |
| 상대 종족별 `nll.action` T · P · Z | 0.932 · 0.875 · 0.853 | 0.932 · 0.875 · 0.853 | 0.931 · 0.875 · 0.853 | 0 |
- A − t2(같은 레시피, GPU 비결정성): `aux_jepa` −8%, 나머지 주 지표 잡음 안.

### 상수 예측 바닥 대비([aux_jepa_adam_eps](../investigation/2026_11_18_19_12_aux_jepa_adam_eps.md) 비교 지표, 바닥 = 1 − √(1 − s²), s = `jepa_tgt_std`, 상한 근사라 비교용)
| | t2 | A | B |
|---|---|---|---|
| 바닥 | 0.097 | 0.102 | 0.048 |
| `aux_jepa` / 바닥 | 0.46 | **0.40** | 0.45 |
- B는 바닥 자체가 절반(정답 흩어짐이 작아 상수 예측도 잘 맞음). 바닥 대비로는 A보다 나쁨.

### 학습 로그(끝 200 update 3,059~3,258 평균)
| 지표 | A | B |
|---|---|---|
| total | 5.760 | 5.754 |
| action · delay · queued | 0.409 · 0.923 · 0.008 | 0.409 · 0.923 · 0.008 |
| selected · target_unit · target_loc | 3.205 · 0.086 · 1.190 | 3.205 · 0.087 · 1.190 |
| value · win_open | 0.371 · 0.429 | 0.382 · 0.413 |
| `aux_jepa` | 0.052 | 0.027 |
| `aux_priv`(전체 최대) | 0.0019 (0.0050) | 0.0017 (0.0051) |
| `jepa_tgt_std` | 0.463 | 0.368 |
| `skipped_steps` | 0(로그에 없음) | 0 |
- `jepa_tgt_std`(300 update 창): 2,100 update까지 A · B 비슷(0.30~0.43), 그 뒤 B가 낮음(0.31~0.37 대 0.38~0.46).
- 위험(`aux_priv` 튐) 없음: 최대 A 0.0050 · B 0.0051.

### optimizer step 비 · 감쇠 · 가중치 이동(ckpt-3258)
| 모듈 | 지표 | t2 | A | B |
|---|---|---|---|---|
| `aux_jepa` | step RMS 비(eps 있음 / 없음) | 0.15 | 0.14 | **0.96** |
| `aux_jepa` | 감쇠 평균 √v̂ / (√v̂ + eps) | 0.08 | 0.08 | **0.96** |
| `aux_jepa` | gn 대비 이동(상대 L2) | 0.026 | 0.029 | **0.104** |
| `aux_priv` | step RMS 비 | 0.13 | 0.13 | **0.97** |
| `aux_priv` | 감쇠 평균 | 0.035 | 0.035 | **0.40** |
| `aux_priv` | gn 대비 이동 | 0.010 | 0.010 | **0.059** |
| 몸통 core · entity · scalar · CNN | step RMS 비 | - | 0.61 · 0.91 · 0.85 · 0.88 | 0.64 · 0.90 · 0.86 · 0.89 |
| 몸통(core · entity · scalar · spatial) | gn 대비 이동 | 0.0142 | 0.0140 | 0.0140 |
- step 비는 예상대로 ≈ 1, 이동은 예상(7~9배)보다 작은 3.6~6배. 몸통 step 비 · 이동은 A와 같음.
- B `aux_priv`는 RMS 비 0.97이나 감쇠 평균 0.40: √v̂ 중앙값 4e-9 < eps 1e-8이라 원소 다수는 아직 eps에 눌림(큰 원소가 RMS를 정함).
- 계획의 `optim_state.py` · `drift.py`(다른 run 고정) 대신 같은 계산을 A · B용으로 옮긴 `local/work/aux_eps/cmp_ab.py`(보조 그룹 · 이동)와 `optim_state.analyse`(몸통) 사용.

## 결론
- B는 주 손실(행동 · 헤드별 NLL · 종족별, 학습 끝 200 update도)을 바꾸지 않고 보조 예측기 손실만 낮춤: val500 `aux_jepa` −47%(A − t2 비결정성 −8%보다 큼), `aux_priv` −4%, `win_open_mse` −0.026.
- 단, B는 JEPA 정답 표현 자체의 흩어짐이 30% 작음(`jepa_tgt_std` 0.306 대 0.439). 손실은 1 − cos라 정답이 서로 비슷하면 상수 예측도 잘 맞음. 상수 예측 바닥 대비는 A 0.40 · B 0.45로 B가 나쁨 → `aux_jepa` 하락은 "예측이 나아짐"보다 "정답이 쉬워짐" 쪽(추정).
- 판정 규칙은 정하지 않음(사용자 결정). 다음에 무엇을 할지는 사용자 결정.

## 이상한 점
- 표본은 계획 표(B만)와 달리 A · B 둘 다 50 update 돌림.
- A `nll.selected` 1,319: 저그전 판에서만 4,066(t2 · B 5.02), F1은 같음 → 극소수 step에서 정답 유닛 확률이 사실상 0(마스크 또는 수치 문제로 추정, 원인 미확인).
- 학습 속도 약 3.1초/update로 t2(1.69초)의 약 2배. A · B 같음, CPU 85~91% 사용(원인 미확인).

## 출력
- 체크포인트 `<shared>/checkpoints/m2-aux-eps-a/ckpt-3258`, `m2-aux-eps-b/ckpt-3258`, 표본 `m2-aux-eps-{a,b}-sample/ckpt-50`
- val500 `<shared>/artifacts/aux_eps/val500/m2-aux-eps-{a,b}__ckpt-3258.json` · `.log`(t2는 `<shared>/artifacts/bc_e2_val/val500/m2-teacher-t2__ckpt-3258.json`)
- 학습 로그 `local/work/aux_eps/runs/`, 비교 `local/work/aux_eps/out/cmp_ab.json`

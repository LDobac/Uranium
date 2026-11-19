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

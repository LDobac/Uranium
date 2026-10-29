# BC ablation: (A) 대상 맵만 vs (B) S1 + S2 (041-R5)

- 날짜: 2026-10-29
- 모델: BC 본학습 설정(모두 켬, `config/bc.toml`)
  - (A) `bc-ablation-a-pt/ckpt-11387`: 대상 맵 데이터만, 1 epoch
  - (B) `bc-s1-pt/ckpt-39096`(전체 맵 1 epoch) → `bc-s2-pt/ckpt-11387`(대상 맵 1 epoch)
- 채점: 대상 맵 test 1,199게임(2,667,590 step), 첫째·둘째 Linux ROCm 호스트, f32. 출력 `<shared>/artifacts/bc_main_eval/test/`
- 전체 평가는 `reports/m6_bc_eval.md`

| 지표 | (A) | (B) |
|---|---|---|
| action_type top-1 | 68.98% | **71.45%** |
| action_type top-5 | 93.53% | **95.50%** |
| NLL action / delay / selected | 1.034 / 1.919 / 5.098 | **0.901 / 1.853 / 4.835** |
| NLL target_unit / target_loc | 1.982 / 5.454 | **1.765 / 5.192** |
| target_loc 오차 평균(타일) | 10.76 | **9.85** |
| selected F1 | 0.832 | **0.852** |
| win_open MSE | 0.903 | **0.870** |

**041-AC3 통과**: (B)가 모든 지표에서 (A)와 같거나 높다. 전체 맵 사전학습 → 대상 맵 미세조정 전략을 유지한다(041 맵 전략 재검토 불필요).

T0 대전은 (B)만 했다(041-AC2, `m6_bc_eval.md` 2절). (A)의 대전은 하지 않았다(평가 시작 때 제안한 범위대로 spec이 정한 test 지표 비교만 했다. 필요하면 나중에 추가).

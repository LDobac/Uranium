# 045 — Scale-up B with Patch Spatial Path (`m2b` 줄기)

- **Status**: Approved (2026-11-20 작성, 사용자 확인 2026-11-20. **2026-11-20 개정 2(코디네이터 결정(사용자 위임 2026-11-20), 구현 세부)**: R2 ⑤ gn update 수·warmup, R3 ② `m2b` plan 사본 seed 겹침 집계 제외, R3 ④ 교사 판 6폴더 사본, R4 ③ 표본 이름·폴더 분리·진단 경로 생략, R6 ① `s2e2b` 단서 진단 선택, R6 ⑤ 측정 도구의 "측정 대상 없음" 수용. 실측 반영: R1 ⑤ 파라미터 수. Interfaces 측정 이름(구현 확정). 내용은 사용자 결정 2026-11-17·11-19·11-20(Q1~Q13)과 [인수인계][handoff] 기본안, R5 위치 머리 손실 보고(사용자 결정 2026-11-20). AMB-045-1·[032] AMB-032-15·[060] AMB-060-3 Resolved(2026-11-20: 세부 기본안 채택), [041] AMB-041-20·[060] AMB-060-2 Deferred(M2B-4). 함께 바꾸는 spec: 032·041·050·060 2026-11-20 개정)
- **Related**: [021][s021], [022][s022], [032][s032], [040][s040], [041][s041], [044][s044], [050][s050], [060][s060]
- 배경: [인수인계][handoff], [model_scaleup], [spatial_ablation]

## Purpose

모델을 후보 B 크기로 키우고 공간 인코더 CNN을 패치 linear projection([032]-R29)으로 바꾼 모델로 t2 줄기를 같은 단계·데이터·update 규칙으로 다시 학습한다(`m2b-*`). 결과를 Checkpoint 1(`r1`)·Checkpoint 2(`t1`)·t2와 같은 표로 비교해 가설을 본다.

| 가설 | 예상 |
|---|---|
| (가) 용량 부족 | 학습 손실·교사 모드 누출·T1이 나아짐 |
| (나) 데이터 부족 | 학습 손실만 내려가고 게임은 그대로 |

## Scope & Non-goals

- 범위: 모델 구성(R1), 줄기 단계(R2), 판·데이터(R3), 사전 단계(R4), 학습 곡선 보고(R5), 측정 지점(R6), 위치 지표(R7), 기록(R8), 장비(R9).
- 하지 않는 것(사용자 결정 2026-11-20):

  | 항목 | 근거 |
  |---|---|
  | 위치 경로 사전 확인(t2 몸통 고정 + 위치 경로만 학습) | Q3 |
  | 크기·구조 분리 대조 run | Q4. 리포트에 한계로 적음 |
  | linear projection 접기·판정 문턱 | Q7·Q12. 위치 지표는 추적만 |
  | CNN이 빠진 만큼 다른 모듈 확대 | Q13 |
  | 큰 모델용 학습률 run | Q9 |
  | r3 학습·[041]-R29 ④ 재선택 | Q8 |
  | 줄기 뒤 RL | [인수인계][handoff] 2절 |

- 그 밖에 하지 않는 것: 기존 `m2-*` 데이터·체크포인트 변경, RL 학습기 엔티티 재계산, 공간·위치 재계산 시제품 정식화([인수인계][handoff] 8절), 계획 도구([053]·[054]·[055])를 `m2b` 모델에 적용.

## 용어

| 용어 | 뜻 |
|---|---|
| B 구성 | R1 표의 크기 + 패치 경로 |
| 패치 경로 | [032]-R29 `spatial_patch`(CNN 대신 칸 묶음 linear projection) |
| `m2b` 줄기 | R2 단계 전체, run 접두 `m2b-` |
| 원래 줄기 | `m2-s1` → … → `m2-teacher-t2`([인수인계][handoff] 4절) |
| t1 | Checkpoint 2 = `m2-teacher-t1/ckpt-2844`([checkpoint2] 1절) |
| t2 | `m2-teacher-t2/ckpt-3258` |
| `M2B-<n>` | R4 단계 이름(TBD·보류 표기에 씀, README 규칙 8-1) |

## Requirements

| ID | 요구사항 |
|---|---|
| 045-R1 | **MUST** **모델 구성.** ① 크기(`[net]`, [032]-R28): 엔티티 d_model 512·FFN 2048·6층·16 head, Core GTrXL hidden 1024·4층·8 head·기억 64, 스칼라 hidden·헤드 `ar`·가치 hidden 512, 나머지 `[net]` 키는 코드 기본값(사용자 결정 2026-11-17 후보 B, [model_scaleup] 7절). ② 공간: 패치 경로 p 8·d_p 512(Q1, 키 이름 `[model] spatial_patch`·`[net] spatial.patch_dim`, [032] AMB-032-15 ④). 위치 머리·`outer` FiLM·aux_jepa 정답은 [032]-R29(Q2·Q6), 세부는 AMB-032-15. ③ `[model]` 나머지는 t2 설정(`config/bc_m2_teacher_t2.toml`)과 같다. CNN 전용 `spatial_preact`·`spatial_norm`은 뺀다. ④ CNN이 빠진 만큼 다른 모듈을 키우지 않는다(Q13). 고정 폭 모듈([032]-R28 ⑦)은 그대로다. ⑤ 파라미터 수: 실측 138,680,787(추론 135,957,588 + 학습 전용 보조 2,723,199), 어림 약 1억 3,870만([032] §4)과 일치 |
| 045-R2 | **MUST** **줄기 단계.** 원래 줄기를 같은 순서로 밟는다. ① 단계·init·data seed는 아래 표, `model_seed` 1. ② 학습률·스케줄·손실·aux λ·batch는 그 단계 원래 설정 그대로(Q9). ③ update 수는 원래 규칙: 단계마다 데이터 한 epoch, `[steps]`·출처 비율을 다시 셈([044]-R5·R8, Q10). 원래 값과 달라질 수 있고 실제 값을 기록한다. warmup(`[steps]` 안)은 원래 설정의 warmup ÷ 원래 update 비율을 유지해 새 update 수로 다시 센다(AMB-045-1 ④). ④ r3: r2 판 300판을 만들고 재레이블해 d1·t1·t2 데이터에 넣되 r3 학습은 하지 않는다. d1은 r2에서 시작한다(Q8). ⑤ gn 두 단계(t1 데이터로 2 epoch 더)를 유지한다(Q5). GroupNorm은 없다. gn 각 단계 update 수는 t1 데이터 한 epoch(원래 설정 `[steps]` 3,000은 데이터가 2,844에서 끝남), warmup은 5%(원래 단계들의 비율, 코디네이터 결정(사용자 위임 2026-11-20)). ⑥ 중간 단계(mix·r1·r2·d1·gn)는 판정([041]-R22 ④·R29 ④, [044]-R6)으로 갈래를 바꾸지 않고 원래 줄기 경로를 그대로 따른다(측정·판정 없음, AMB-045-1 ③). ⑦ run 이름은 원래 이름에 `m2b-` 접두(`gn` 이름 포함), r3 판 shard는 `dagger_m2b/r3`(AMB-045-1 ①) |
| 045-R3 | **MUST** **판·데이터.** ① 재사용: 사람 shard 전부, 스크립트 교사 판(botgen 6폴더, [022] v2·x1·x2). ② ★ 단계의 판은 그 단계 `m2b` 모델이 둔다. plan은 원래 plan(`config/plans/dagger_m2_*.toml`·`distill_m2_d1*.toml` 등)의 `checkpoint`·`name`만 바꾼 사본이다(같은 seed, `rush_check_plan.py` 재생성 금지: seed 겹침 검사). 증류·교사 라운드 plan 생성의 seed 겹침 집계와 seed 겹침 테스트에서 사본은 뺀다: 이름의 `_m2b_`를 `_m2_`로 바꾼 원본 파일이 있고 seed 목록이 원본과 같은 plan만 사본이고, 원본이 없거나 seed가 다른 plan(보충 plan 등)은 집계한다(코디네이터 결정(사용자 위임 2026-11-20)). ③ 재레이블·레이블러는 원래와 같다: r1·r2·r3 `terran_react` 8:00 절단, d1 레이블러는 `m2b-s2e2` 마지막 체크포인트, t1 `terran_tp` 10:00 + [022] botgen, t2는 [044]-R8 ⑤ 구성. ④ 단계별 botgen 구성은 원래와 같다: r1~d1은 [021]-R8 6폴더만, t1부터 [022] v2·x1·x2. 6폴더만 읽는 단계(mix~d1)는 공용 드라이브 사본 `<shared>/eval/m2b-lineage/data/botgen6/`(30 파일, sha256을 manifest와 대조)에서 읽는다. 이유: 공용 `botgen` 폴더에 [022] v2·x1·x2가 함께 있어 그대로 읽으면 구성이 달라지고, 다른 장비로 학습을 넘겨도 읽을 수 있어야 한다(코디네이터 결정(사용자 위임 2026-11-20)). ⑤ 보충 판 수는 원래 규칙의 목표 몫 ± 3%p([041]-R29 ④, [044]-R5 ③·R8)로 정한다. ⑥ 출력은 새 폴더(`<shared>/datasets/replays/shards/dagger_m2b`·`distill_m2b`, `<shared>/eval/m2b-lineage/`). 기존 `m2` 데이터는 건드리지 않는다 |
| 045-R4 | **MUST** **사전 단계·순서.** 아래 `M2B-0`~`M2B-7`. ① `M2B-0` 결과와 고칠지를 사용자가 정한 뒤에 `M2B-6`을 시작한다(사용자 결정 2026-11-19). 원인은 커널 로드 실패 뒤 조용한 기준 구현 대체였고([bc_slowdown]), 수정은 [060]-R1 ③~⑨·[041]-R15 2026-11-20 개정(사용자 결정 2026-11-20, 방법 ③+④, 045와 별개 승인)이다. 이 수정은 `M2B-2`와 별개 단위이고 `M2B-4` 실측 전에 끝낸다([060] AMB-060-3 ④). ② `M2B-4` 실측 뒤 사용자가 일정·[060]-R3 추론 상한 값을 확인한다. ③ `M2B-5` 표본 끝-끝은 모든 단계 종류(판 → 재레이블 → shard → `mix-fractions` → BC 50 update)를 지난다(AGENTS.md 대규모 배치 전 규칙). 표본은 실제 줄기와 이름·폴더를 나눈다(run `m2bsmp-*`, `<shared>/eval/m2b-lineage-sample/`). 측정 프로토콜의 진단 경로는 표본에서 생략한다(측정 도구가 `m2b-` 접두 체크포인트만 받음). 학습 사슬·`bc-eval`·게임 plan은 표본으로, 진단은 첫 측정 지점 `s2e2b`에서 확인한다(코디네이터 결정(사용자 위임 2026-11-20)) |
| 045-R5 | **MUST** **학습 곡선 보고.** `m2b-s1` update 10,000에서 같은 update 지점의 `m2-s1` 학습 로그와 학습 곡선을 비교해 사용자에게 보고한다. 위치 머리(Location 헤드) 학습 손실도 같은 update 지점의 `m2-s1` 값과 나란히 보고한다(사용자 결정 2026-11-20). 멈추지 않는다(Q12) |
| 045-R6 | **MUST** **측정 지점.** [050]-R9 행 `s2e2b`·`t1b`·`t2b`(2026-11-20 개정). ① `s2e2b` = `m2b-s2e2` 마지막: test, val 500(종족별), R7 위치 지표([050]-R9 `s2e2b` 행). 단서 진단은 선택(코디네이터 결정(사용자 위임 2026-11-20)). ② `t1b` = `m2b-teacher-t1` 마지막, `t2b` = `m2b-teacher-t2` 마지막: t1·t2와 같은 전체 표(T1 300판, T0 표준 테란·프로토스 각 32판, TvZ 32판, 러시 128판 + 운영형 40판, val 500(종족별), test, 교사 모드 판 비율, `zerg_build`·`tp_style`). ③ 비교: `s2e2b`↔`m2-s2e2`, `t1b`↔t1, `t2b`↔t2, Checkpoint 1(`r1`) 기존 수치 재사용. ④ [050]-R10 게이트는 적용하지 않고 기록만, 판정 문턱 없음(gn 선례, Q7·Q12). ⑤ 추가 지표: 단서 진단은 `sp_pool` 단계를 빼고 잰다. 죽은 채널 통계는 없다([032]-R25 적용 범위). 측정 도구(프로토콜 점검·진단 표)는 이 줄기의 죽은 채널 "측정 대상 없음"과 `sp_pool` 단계 없는 단서 진단을 빠진 값이 아닌 정상으로 받는다(코디네이터 결정(사용자 위임 2026-11-20)) |
| 045-R7 | **MUST** **위치 지표(Q7).** 측정 지점마다 기록하고 t2 값과 나란히 둔다. 문턱 없음, 접지 않음. ① 정의([spatial_ablation] 2·3절, teacher-forced `bc-eval` 경로): 위치 NLL(전체·묶음별), 4칸 안 비율(argmax와 레이블의 유클리드 거리 ≤ 4칸, 128² 칸), 정확 칸(argmax = 레이블), 거리 평균. ② 묶음([spatial_ablation] 3절과 같은 분류, 위치 레이블이 있는 step만): 건물 배치(`Build_*` 12개), 이동·공격(`Move`·`Attack`·`Patrol`·`Smart_Pos`), 랠리(`SetRally_Pos`), 기타. ③ 추적 4개: 위치 NLL, 4칸 안 비율, 건물 배치 정확 칸, 랠리 오차(랠리 거리 평균). ④ 데이터: val 전체(2,395판, [spatial_ablation] 1절 조건). t2 값은 그 리포트 A0 기준값을 재사용한다(사용자 결정 2026-11-20, AMB-045-1 ② Resolved) |
| 045-R8 | **MUST** **기록.** ① 리포트 `docs/report/experiment/<ts>_m2b_scaleup.md`: 실험 전 배경·예상(가설 (가)·(나)), 진행 기록, 단계별 실제 update 수·data seed·장비 비율, R5·R6·R7 표. ② 한계를 적는다: 크기·구조 효과 분리 불가(Q4), seed 1개·줄기 하나, 측정 플랫폼·backend 차이. ③ 정의는 `config/protocol/lineage.toml` `[lineage.m2b]`(`run_prefix = "m2b-"`)와 지점 |
| 045-R9 | **MUST** **장비.** ① 주 장비 규칙·스왑은 [060]-R8. ② 게임 추론(★ 단계 판, 측정 게임)은 CUDA 호스트와 Linux ROCm 호스트 모두 쓸 수 있다(Q11). Linux ROCm 호스트는 [060]-R3 추론 상한 개정 뒤에 쓴다. ③ 엔티티 재계산([041]-R17)을 켤 호스트·조각 행 수는 `M2B-4` 실측 뒤 사용자가 정한다([041] AMB-041-20, Deferred(M2B-4)). [060]-R3 B 추론 상한도 같다([060] AMB-060-2) |

### 045-R2 단계 표

| 단계 | init | 데이터(★ = 그 단계 모델이 판을 새로 둠) | update(원래) | data seed |
|---|---|---|---|---|
| s1 | 새로(`model_seed` 1) | 사람 S1 | 39,096 | 1 |
| s1e2 | s1 | S1 | 39,094 | 2 |
| s2e2 | s1e2 | 사람 S2 | 11,387 | 1 |
| mix-e1..e7 | 이전 단계 | S2 + botgen 6폴더 | 8,417(7 epoch 합) | 3..9 |
| dagger-r1 | mix-e7 | ★ mix-e7 600판(`terran_react` 재레이블, 8:00 절단) | 1,285 | 13 |
| dagger-r2 | r1 | ★ r1 300판 | 1,408 | 14 |
| (r3 판만) | – | ★ r2 300판(d1·t1·t2 데이터) | 학습 안 함 | – |
| distill-d1 | r2 | ★ r2 1,100판(테란·프로토스), 레이블러 `m2b-s2e2` | 2,000 | 15 |
| teacher-t1 | d1 | ★ d1 300판(`terran_tp`, 10:00) + [022] botgen v2·x1·x2 | 2,844 | 17 |
| gn-t1, gn-t1e2 | t1, gn-t1 | t1과 같은 데이터 | 2,844 × 2 | 17 |
| teacher-t2 | gn-t1e2 | ★ gn-t1e2 300판(T/P) + 600판(Z) | 3,258 | 18 |

- 원래 무게 줄기 합계 114,477 update(r3 제외). 원래 실행 기록은 [인수인계][handoff] 4절.

### 045-R4 단계

| 이름 | 작업 | 사용자 확인 |
|---|---|---|
| `M2B-0` | BC 2배 느려짐 조사(t2 설정 `bench bc` 100 update, 로그·`.env`·코드·manifest 차이). 고치지 않고 리포트([bc_slowdown]) | 원인·고칠지(2026-11-20: 방법 ③+④) |
| (별개 단위) | 커널 로드 수정: [060]-R1 ③~⑨, [041]-R15 2026-11-20 개정(045와 별개 승인). `M2B-2`와 따로 구현·리뷰·반영, `M2B-4` 전 | 승인(2026-11-20) |
| `M2B-1` | 이 spec과 032·041·050·060 개정 | 승인(2026-11-20) |
| `M2B-2` | 코드(Interfaces) + 테스트 + 별개 리뷰 | – |
| `M2B-4` | 실측: 파라미터, [060]-AC7 최악 모양(Linux ROCm 호스트 12 GiB), CUDA 실데이터 100 update 속도·메모리, 추론 지연·메모리(두 호스트), 엔티티 재계산 조각 | 일정, [060]-R3 추론 상한 |
| `M2B-5` | 표본 끝-끝(R4 ③) | – |
| `M2B-6` | 본 줄기(R2) | 1만 update 보고(R5) |
| `M2B-7` | 측정(R6·R7) + 리포트(R8) | 결과 |

- `M2B-3`(위치 경로 사전 확인)은 없다(Q3).

## Interfaces (명령 이름은 구현 때 확정하고 이 절을 고친다. 모델 설정 키 이름은 [032] AMB-032-15 ④)

- BC 설정: `config/bc_m2b_<단계>.toml` = 원래 단계 설정 사본. 차이는 `[net]` 추가, `[model]` `spatial_patch` 추가·`spatial_preact`·`spatial_norm` 삭제, `[monitor] spatial_every` 삭제뿐(테스트로 확인, `tests/python/test_lambda_configs.py:305-335` 방식).
  ```toml
  [model]
  spatial_patch = 8
  [net.entity]
  d_model = 512
  ffn = 2048
  layers = 6
  heads = 16
  [net.core]
  hidden = 1024
  gtrxl_layers = 4
  [net.scalar]
  hidden = 512
  [net.heads]
  ar = 512
  [net.value]
  hidden = 512
  [net.spatial]
  patch_dim = 512
  ```
- plan: `config/plans/<원래 이름의 m2 → m2b>.toml`(예 `dagger_m2b_r1.toml`), 원래와 `checkpoint`·`name`만 다름.
- 실행 스크립트: 원래 `<shared>/eval/m2-lineage/scripts/`를 `local/work/m2b/`로 복사해 상수만 바꿈(git 밖). t2 단계는 로그(`<shared>/eval/dagger-m2-t2-20261117/logs/`, `local/work/t2_main/`)로 재구성.
- 측정: `config/protocol/lineage.toml` `[lineage.m2b]` + 지점 `s2e2b`·`t1b`·`t2b`, `tools/eval/protocol.py job`·`diag`·`compare`, `tools/eval/tables.py`.
  - lineage 키: `[lineage.m2b]` `run_prefix`·`points`·`cue_drop_stages = ["sp_pool"]`, 지점 `lineages`·`compare`, `[metric.dead_channels]` `no_targets = ["m2b"]`·`not_measured.m2b`.
  - 비교 대상(`compare`): `s2e2b` ↔ `m2:s2e2`·`ck1:r1`, `t1b` ↔ `m2:t1`·`m2:D`(t1에 없는 칸, 같은 체크포인트)·`ck1:r1`, `t2b` ↔ `m2:t2`·`ck1:r1`.
- 위치 지표(R7): `bc-eval` 보고 키 `target_loc_groups`(기존 키·값 불변). 묶음 `all`·`build`·`move`·`rally`·`other`, 값 `steps`·`nll`·`within4`·`exact`·`err_mean`. 추적 4개 = `all.nll`·`all.within4`·`build.exact`·`rally.err_mean`. 지표 `val_loc`: val 전체 2,395판, CUDA f32(`backend = "cuda"`, `bc-eval --precision f32`), 출력 `artifacts/bc_loc/val_all/<run>__<ckpt>.json`, t2 값 재사용은 `[metric.val_loc] reuse`.
- 코드 위치([인수인계][handoff] 8절):

  | 코드 | 변경 |
  |---|---|
  | `python/uranium/model/config.py` | `[net]` 엄격 파서, `new_net_config(..., sizes)`, `spatial_patch`([032]-R28·R29). 지금 `_load`는 모르는 키를 조용히 무시 |
  | `python/uranium/model/` encoders·heads·net·variants | 패치 투영, `s_pool` 제거, 패치 위치 머리, `outer` FiLM, aux_jepa 정답 |
  | `python/uranium/bc/run.py`, `bc/trainer.py` | `[net]` 전달, 새 run 생성, init·resume 크기 불일치 오류(지금은 새 run이 기본 크기로 만들어지고 크기 검사 없음) |
  | `python/uranium/bc/profile.py`, `bench/realbc.py` | 같은 크기 경로 |
  | `python/uranium/model/heads.py`(`location_used_rows`), `model/monitor.py` | 패치 위치 머리 행 고르기, 패치 경로면 공간 통계 없음 |
  | `python/uranium/measure/spatial_logs.py`, `diag/cue/capture.py` | 패치 경로면 죽은 채널 측정 끔, 단서 진단 `sp_pool` 단계 뺌 |
  | `python/uranium/rl/learner.py`(`conv_layout`), `device.py` 공간 실행 설정 | 패치 경로에서 CNN 전용 실행 설정 무시(AMB-032-15 ⑥) |
  | `python/uranium/diag/planning/dump.py` | 범위 밖([053]). 패치 경로 체크포인트면 `s_pool`이 없다는 오류로 멈춤 |
  | `python/uranium/bc/eval.py` | R7 위치 묶음 지표 |
  | `config/bc_m2b_*.toml`, `config/plans/*m2b*.toml`, `config/protocol/lineage.toml` | R2·R3·R6 |
  | `tests/python/` | `[net]` 파서(`test_model2.py:22-47` 방식), 크기 불일치 거부(`test_spatial_norm.py:346-385` 방식), release meta 왕복 유지(`test_model2.py:60-65`), 설정 차이, 위치 지표 |

- 체크포인트 meta는 크기를 이미 저장한다(`python/uranium/ckpt.py`). 추론 서버·RL·평가·SC:R은 meta에서 크기를 읽는다.

## Acceptance Criteria

- 045-AC1 (단위, CPU): `config/bc_m2b_*.toml`이 원래 설정과 Interfaces의 차이만 있다. `[net]`이 R1 표와 같다. 새 run이 R1 크기로 만들어진다.
- 045-AC2 (단위, CPU): R7 위치 지표가 알려진 입력에서 정의와 같다(묶음 분류, 4칸 안, 정확 칸, 거리). 위치 묶음 열이 없던 `bc-eval` 보고 키·값은 그대로다.
- 045-AC3 (측정, `M2B-4`): 파라미터 실측, [060]-AC7 최악 모양 결과, CUDA 100 update 속도·메모리, 두 호스트 추론 지연·메모리를 리포트에 남긴다. 예산을 넘으면 [060]-AC7 처리(절감 설정만, 그래도 넘으면 보고, 모델 불변).
- 045-AC4 (측정, `M2B-5`): 표본 끝-끝이 모든 단계 종류를 끝까지 지난다.
- 045-AC5 (결과): R2 모든 단계가 끝나고 단계별 실제 update 수·data seed·장비 비율이 리포트에 있다. R5 보고(위치 머리 손실 포함)가 있다.
- 045-AC6 (결과): `s2e2b`·`t1b`·`t2b`의 R6 표와 R7 위치 지표가 비교 대상과 나란히 리포트에 있다.

## Risks & Open Questions

> **AMB-045-1 (Resolved 2026-11-20: ② 사용자 결정, ①·③·④ 기본안 채택)**: 줄기 실행 세부.
> ① run 이름: 원래 이름에 `m2b-` 접두(`m2b-s1`, `m2b-s1e2`, `m2b-s2e2`, `m2b-mix-e1`~`e7`, `m2b-dagger-r1`·`r2`, `m2b-distill-d1`, `m2b-teacher-t1`, `m2b-gn-t1`·`m2b-gn-t1e2`, `m2b-teacher-t2`). GroupNorm이 없어도 `gn` 이름을 원래 줄기와 짝 맞추기로 남긴다. r3 판 shard는 `dagger_m2b/r3`.
> ② **Resolved (2026-11-20 사용자 결정: 기본안)**: 위치 지표는 val 전체(2,395판, [spatial_ablation] 1절 조건), t2는 그 리포트 A0 재사용(045-R7 ④). 채택하지 않은 대안: val 500.
> ③ 중간 단계 판정 없음: mix·r1·r2·d1·gn 단계는 측정·판정 없이 원래 경로대로 진행([041]-R22 ④·R29 ④·[044]-R6의 중단·갈래 규칙을 쓰지 않음).
> ④ warmup: 원래 설정의 warmup ÷ 원래 update 비율을 유지해 새 update 수로 다시 셈(예: 원래 t2 163/3,258). 채택하지 않은 대안: 원래 warmup 값 그대로.

| 위험 | 대응 |
|---|---|
| 위치 머리가 패치 경로로 정보를 못 얻음(건물 배치·랠리 붕괴) | 사전 확인 없음(Q3). R7 추적, `t1b` 러시 막음·운영 지표. [032] AMB-032-15 ②·위험 |
| 크기·구조 효과를 못 가름 | 대조 없음(Q4). 리포트에 한계 |
| 줄기 동안 CUDA 호스트 점유(CNN 포함 B 어림 9~10일, CNN 제외 시간은 `M2B-4`) | [060]-R8 스왑, R5 보고 |
| BC 느려짐(커널 조용한 대체) 재발이면 기간 최대 2배 | [060]-R1 ⑤ 시작 오류로 드러남. [model_scaleup] 시간 배수(×4.5)는 커널 사용이 기록으로 확인되지 않아 수정 뒤 `M2B-4`에서 다시 잼([bc_slowdown] 6절) |
| seed 1개, 줄기 하나 | 리포트에 한계 |
| Linux ROCm 호스트 B 학습 50~60초/update | 학습은 사실상 CUDA 호스트. ROCm 호스트는 추론·val 500·진단 |
| 원래 줄기의 판정이 이 줄기에서는 다른 결론을 냈을 수 있음 | AMB-045-1 ③. 리포트에 한계 |
| RL이 초반 빌드 갈림길을 쉽게 뒤집음 | 범위 밖 |

## Related
- 인수인계: [handoff]
- 크기 근거: [model_scaleup] 7절, CNN 사용량: [spatial_ablation]
- 원래 줄기: [041]-R29·R31, [044]-R5·R8, [050]-R9

[handoff]: ../report/plan/2026_11_20_08_41_scaleup_linear_spatial_handoff.md
[model_scaleup]: ../report/bench/2026_11_17_21_45_model_scaleup.md
[spatial_ablation]: ../report/investigation/2026_11_19_23_00_spatial_ablation.md
[checkpoint2]: ../report/release/2026_11_17_09_05_checkpoint2.md
[bc_slowdown]: ../report/investigation/2026_11_20_09_23_bc_slowdown.md
[021]: 2026_10_15_22_24_021-opponent-pool.md
[022]: 2026_11_14_08_18_022-matchup-teacher.md
[032]: 2026_10_15_22_24_032-model-architecture.md
[040]: 2026_10_15_22_24_040-replay-dataset.md
[041]: 2026_10_15_22_24_041-imitation-learning.md
[044]: 2026_11_14_08_18_044-self-game-distillation.md
[050]: 2026_10_15_22_24_050-evaluation.md
[053]: 2026_11_18_08_58_053-planning-predictor-data.md
[054]: 2026_11_18_08_58_054-exploration-games.md
[055]: 2026_11_18_20_30_055-plan-branch-games.md
[060]: 2026_10_15_22_24_060-infra.md
[s021]: 2026_10_15_22_24_021-opponent-pool.md
[s022]: 2026_11_14_08_18_022-matchup-teacher.md
[s032]: 2026_10_15_22_24_032-model-architecture.md
[s040]: 2026_10_15_22_24_040-replay-dataset.md
[s041]: 2026_10_15_22_24_041-imitation-learning.md
[s044]: 2026_11_14_08_18_044-self-game-distillation.md
[s050]: 2026_10_15_22_24_050-evaluation.md
[s060]: 2026_10_15_22_24_060-infra.md

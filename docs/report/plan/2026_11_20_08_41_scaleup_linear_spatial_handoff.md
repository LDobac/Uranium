# 인수인계: 모델 확대 B + 공간 인코더(CNN) → linear projection

- 작성: 2026-11-20 08:41
- 상태: **진행 중(11-20 11시, 아래 마지막 줄).** 작성 때는 착수 전(master `6bfea20`).
- 사용자 결정(2026-11-20 08:3x): 원래 확대 계획(B로 t2 줄기 재학습 → Checkpoint 1·2·t2와 비교)에 **CNN 제거, linear projection으로 변경**을 더한다.
- 다음 세션이 할 일: 6절 질문을 사용자에게 한 번에 묻기 → 7절 0·1단계(병렬 가능)부터.
- 11-20 09:0x: 6절 질문 답을 받음(2절 마지막 행, 6절 "답" 열). 7절 0·1단계 시작.
- 11-20 11시 상태: 7절 0·1·2번 완료, 4번 진행 중, 5번 준비 완료(GPU 대기). 이후 기록은 실험 리포트 [m2b_scaleup](../experiment/2026_11_20_10_33_m2b_scaleup.md)로 이어진다.

## 1. 배경

| 근거 | 내용 | 출처 |
|---|---|---|
| 덜 배움 | 본학습 끝 학습 손실 0.858 = val 0.859, 아직 내려가는 중(과적합 없음) | [model_scaleup] 1절 |
| 크기 효과 미검증 | 크기 비교는 코어만(1,137만 대 2,175만), 1만 update뿐. 엔티티 인코더·끝까지 학습은 시험한 적 없음 | [core_ablation_pt] |
| 교사 누출 | 상대 종족이 입력에 있는데도 테란·프로토스전 12~28%가 저그용 빌드. RL은 204 update 만에 이 갈림길을 뒤집음 | [teacher_leak_fix] |
| 시험할 가설 | (가) 용량 부족 → 키우면 학습 손실·누출·T1이 나아짐 (나) 데이터 부족 → 학습 손실만 내려가고 게임은 그대로 | 2026-11-19 대화 |
| CNN 사용량 | 요약 `s_pool` → Core는 정보 안 쓰임(평균·섞기 차이 ≤ 0.0004 NLL/머리, 상수 바이어스만). 위치 머리는 CNN 특징에 크게 의존(지우면 위치 NLL 5.46 → 8.02, 건물 배치 정확 칸 0.25 → 0.05, 랠리 오차 5 → 27칸) | [spatial_ablation] 2·3·6절 |

## 2. 정해진 것 (사용자)

| 날짜 | 결정 | 출처 |
|---|---|---|
| 11-17 | 확대한다면 후보 B. 크기는 [model_scaleup] 7절 | [timeline] 11-17 21:45 |
| 11-19 22:2x | B로 **t2 줄기만**(Checkpoint 1 줄기는 다시 밟지 않음) 같은 단계·데이터·update 수로 다시 학습, Checkpoint 1(지점 `r1`)·2(`t1`)·t2와 같은 평가표로 비교 | [timeline] 11-19 22:2x, [model_scaleup] 머리말 |
| 11-19 22:2x | BC 학습이 t2 때보다 약 2배 느려진 원인을 B 학습 전에 먼저 조사(3.1초/update 대 t2 1.69([aux_eps_compare]), 1.46([timeline] 11-18 00:56)) | 같음 |
| 11-19 22:2x | GroupNorm 처음부터 켬 + gn 두 단계(t1 데이터 2 epoch 더) 유지 — **CNN을 없애면 GroupNorm 자체가 없어진다. gn 두 단계를 남길지는 6절 Q5** | 같음 |
| 11-19 22:3x·22:4x | 위 작업 멈춤 → 공간 구조를 먼저 정하기로 함 | [timeline] 11-19 22:4x |
| 11-20 08:3x | 확대에 CNN 제거 → linear projection 변경을 더함 | [timeline] 11-20 08:41 |
| 11-20 09:0x | 6절 Q1~Q13 답(6절 "답" 열). Q3 위치 경로 사전 확인 없음(7절 3번 삭제), Q4 대조 없음(Checkpoint 1·2(t2)와만 비교), Q7 linear projection은 접지 않고 위치 지표만 추적, Q11 ROCm 호스트에서도 게임 추론, Q13 CNN이 빠진 만큼 다른 곳을 더 키우지 않음 | 2026-11-20 대화 |
| 11-20 10:0x | spec 승인: [045] Approved, [032]·[041]·[050]·[060] 2026-11-20 개정(커널 로드 수정 포함). 세부(위치 머리 결합·FiLM·키 이름·run 이름·warmup·커널 "요청" 범위 등)는 모두 기본안. 추가 두 가지: [060]-AC8 ⑤ Linux ROCm 호스트 두 대에서 커널 실제 호출 확인, [045]-R5 1만 update 보고에 위치 머리 학습 손실 포함 | [045], [060]-R1·AC8 |
| 11-20 10:0x | 14:30까지 구현 중 생기는 결정은 메인 세션(코디네이터)이 정하고 "코디네이터 결정(사용자 위임 2026-11-20)"으로 기록, 실작업은 서브 에이전트. 내린 결정은 [045]·[032] Status 2026-11-20 개정 2 | 2026-11-20 대화 |

- 정하지 않은 것: 판정 문턱(결과를 보고 사용자가 판단), 줄기 뒤 RL(범위 밖), ROCm 추론 메모리 상한 값(7절 4번 실측 뒤).

## 3. 모델 구성 (기본안(미확정))

### 3.1 크기 (후보 B에서 공간 인코더만 바꿈)

| 모듈 | B 값 |
|---|---|
| 엔티티 인코더 | d_model 512, ffn 2048, 6층, 16 head |
| 코어 GTrXL | hidden 1024, 4층, 8 head, 기억 64 |
| 스칼라 / 헤드 `ar` / 가치 hidden | 512 / 512 / 512 |
| 공간 인코더 | **CNN 삭제 → 패치 linear projection(3.2)** |
| 고정 폭(설정을 안 따름) | aux_priv 은닉 256, aux_jepa 은닉 512, bypass 256, outer LSTM 256, GTrXL FFN 4·d |

- 파라미터 어림(실측 필요):
  - B(d_p 512): 1억 5,628만 − 공간 인코더 1,486만 − 위치 업샘플 282만 − `s_pool` 제거로 Core 입력 1536 → 1024가 되며 줄어드는 몫 약 197만(GTrXL 입력 52만, `core_skip` 52만, `outer` core_add 13만, aux_jepa 출력 79만) + 새 투영·위치 경로 약 200만 ≈ **1억 3,860만**.
  - 지금 크기(d_p 256): 2,643만 − 373만 − 75만 − 약 72만 + 약 100만 ≈ **2,220만**.
- 산식: `local/work/spatial_params/count.py`(B 정의는 `local/work/size_probe/probe.py`의 `cand("e512+el6+s2+c1024+cl4+h512")`).

### 3.2 linear projection 공간 경로

| 항목 | 기본안(미확정) | 근거·메모 |
|---|---|---|
| 입력 | 지금 CNN 입력 그대로: 공간 26채널 + 유닛 scatter 32채널 = 58 × 128² | [030] 관측 변경 없음 |
| 투영 | p × p 칸 묶음(패치)마다 `Linear(58·p² → d_p)`. p 8 → 패치 256개 | ViT식 패치 투영 |
| 요약 `s_pool` | **없앰.** Core 입력은 `[e_pool, c]` | [spatial_ablation]: 정보 안 쓰임. 지금 Core 입력 768 중 256이 `s_pool`(`python/uranium/model/net.py:121,188`) |
| 위치 머리 | 패치 특징과 `ar`을 합쳐 패치마다 `Linear → p²`(칸마다 logit 1개)로 128²를 채움 | **가장 큰 위험.** 지금 위치 머리는 S8·skip(64²·32²·16²)을 업샘플한다([032] §2.5 6행). 지우면 무너짐 |
| `outer` FiLM | stem이 없어지므로 패치 특징에 FiLM(또는 끔) | [032] §2.7 `outer` ② |
| aux_jepa 정답 | `[e_pool, s_pool]` → `e_pool`만(또는 패치 평균) | `net.py:210`, [032] §2.8·R24 ② |
| 없어지는 스위치·감시 | `spatial_preact`·`spatial_norm`·`spatial_end_relu`, 죽은 채널 감시([032]-R25, [041]-R25) | GroupNorm 결정은 의미 없어짐 |
| 영향 없음 | `terrain`(원본 지도 채널을 엔티티에 붙임) | [spatial_ablation] 0절. bypass·행동 공간도 CNN을 안 거침(코드) |

## 4. 학습 줄기 `m2b-*` (원래 t2 줄기를 그대로 밟음)

| 단계 | init | 데이터(★ = 그 단계 모델이 판을 새로 둠) | update(원래) | data seed |
|---|---|---|---|---|
| s1 | 새로(model_seed 1) | 사람 S1 | 39,096 | 1 |
| s1e2 | s1 | S1 | 39,094 | 2 |
| s2e2 | s1e2 | 사람 S2 | 11,387 | 1 |
| mix-e1..e7 | 이전 | S2 + botgen 6폴더 | 8,417 | 3..9 |
| dagger-r1 | mix-e7 | ★ mix-e7 600판(terran_react 재레이블, 8:00 자름) | 1,285 | 13 |
| dagger-r2 | r1 | ★ r1 300판 | 1,408 | 14 |
| dagger-r3 | r2 | ★ r2 300판(d1·t1·t2 데이터에 들어감) | 1,528(원래 학습·측정, [041]-R29 ④ 선택에서 S = r2) | 15 |
| distill-d1 | r2 | ★ r2 1,100판(테란·프로토스만), 레이블러 = 이 줄기의 s2e2 | 2,000 | 15 |
| teacher-t1 | d1 | ★ d1 300판(terran_tp, 10:00) + [022] botgen v2·x1·x2 | 2,844 | 17 |
| gn-t1, gn-t1e2 | t1, gn-t1 | t1과 같은 데이터(Q5) | 2,844 × 2 | 17 |
| teacher-t2 | gn-t1e2 | ★ gn 300판(T/P) + 600판(Z) | 3,258 | 18 |

- 무게 줄기 합계 114,477 update(r3 학습 제외).
- r3: 기본안(미확정)은 판만 만들고 학습은 생략, d1은 r2에서 시작(원래 선택 결과를 따름). 선택을 다시 할지는 Q8.
- update 수: 원래 규칙은 단계마다 데이터 한 epoch, `[steps]`·비율을 다시 셈([044] R5·R8)이라 데이터 양에 따라 원래 값과 조금 달라질 수 있다. 원래 값으로 고정할지 규칙을 따를지는 Q10.
- 재사용: 사람 shard 전부, 스크립트 교사 판(botgen 6폴더, [022] v2·x1·x2). 단계별 botgen 구성은 원래와 같게: r1~d1은 [021]-R8 6폴더만, t1부터 [022] v2·x1·x2. 보충 판 수는 원래 비율 ±3%p에 맞춤.
- 출력은 새 폴더(`<shared>/shards/dagger_m2b`, `distill_m2b`, `<shared>/eval/m2b-lineage/`). 기존 m2 데이터는 건드리지 않는다.
- 게임 추론 장비: B 추론 메모리 3.1 GiB가 [060]-R3 상한 2.5보다 큼. Q11 답: ROCm 호스트에서도 돌게 060-R3 상한을 고친다. 값은 CNN을 뺀 모델을 7절 4번에서 잰 뒤.
- 원래 실행 기록:
  - 단계별 실행 스크립트: `<shared>/eval/m2-lineage/scripts/`(`dagger_r1.sh`·`dagger_r2.sh`·`dagger_r3.sh`·`distill_d1.sh`·`teacher_t1.sh`·`topup_*.sh`, `m2tasks.py`). git 밖이고 호스트 주소가 박혀 있다. 복사해 `local/work/m2b/`에서 상수만 바꿔 쓴다.
  - t2 단계는 스크립트가 없다. 로그(`<shared>/eval/dagger-m2-t2-20261117/logs/`, `local/work/t2_main/`)로 재구성한다.
  - plan은 `config/plans/dagger_m2_*.toml`·`distill_m2_d1*.toml`의 `checkpoint =`·`name`만 바꾼 사본을 쓴다. `rush_check_plan.py`로 다시 만들면 seed 겹침 검사에 걸린다.
  - 레이블러 경로 `checkpoints/m2-s2e2/ckpt-11387`(`distill_d1.sh`)은 m2b-s2e2로 바꾼다.

## 5. 비교·측정

| 항목 | 방법 |
|---|---|
| 학습 곡선 | 같은 update 지점의 `m2-*` 학습 로그와 비교. 기본안(미확정): 1만 update에서 사용자에게 보고하고 멈추지는 않음(Q12) |
| 측정 지점 | `s2e2b`(val500·test), `t1b`·`t2b`(t1·t2와 같은 전체 표: T1 300, T0 테란·프로토스 64, TvZ 32, 러시 128, 운영 40, val500, test, `zerg_build`·`tp_style`), 기존 Checkpoint 1(`r1`) 수치는 재사용 |
| 위치 지표(Q7) | 측정 지점마다 위치 NLL, 4칸 안 비율, 건물 배치 정확 칸, 랠리 오차를 [spatial_ablation] 방식으로 기록하고 t2와 나란히 둔다. 문턱 없음, 접지 않음 |
| 정의 | `config/protocol/lineage.toml`에 `[lineage.m2b]`(`run_prefix = "m2b-"`) + 지점. 명령은 `tools/eval/protocol.py job`·`diag`, `tools/eval/tables.py`, `protocol.py compare` |
| 측정 시간(어림) | 게임은 지점당 gn 2.3·t2 3.1시간이었음([timeline] 11-17·11-18). B는 추론이 느려 더 걸림. val500·진단은 Linux ROCm 호스트에서 병행 |
| 해석 | 1절 가설 (가)·(나). 문턱 없음 |

## 6. 사용자에게 물을 것 (한 번에)

| # | 질문 | 기본안(미확정) | 답(11-20) |
|---|---|---|---|
| Q1 | linear projection 세부: 패치 크기 p, 패치 특징 폭 d_p | p 8, d_p 512(B) | 기본안 |
| Q2 | 위치 머리 설계(CNN 업샘플 대신 무엇으로 128²를 내나) | 패치 특징 + `ar` → 패치당 `Linear → p²` | 기본안 |
| Q3 | 본 줄기 전 짧은 확인을 넣나: 지금 t2 몸통 고정 + 위치 경로만 CNN 대 linear projection 학습해 위치 NLL·건물 배치 비교(CUDA 수 시간) | 넣음. 위치가 크게 나빠지면 줄기 시작 전 보고 | **안 함** |
| Q4 | 크기와 구조를 한 번에 바꾸면 결과가 좋아도 어느 쪽 덕인지 못 가름. 대조를 둘까(예: 지금 크기 + linear projection을 s1 1만 update만) | 사용자 판단 | **안 둠.** Checkpoint 1·2(t2)와만 비교 |
| Q5 | gn 두 단계(GroupNorm 때문에 생긴 "t1 데이터 2 epoch 더")를 CNN이 없는데도 유지할까 | 유지(update 수·데이터 노출을 t2와 맞춤) | 유지 |
| Q6 | `outer` FiLM과 aux_jepa 정답을 어떻게 바꿀까 | FiLM은 패치 특징에, JEPA 정답은 `e_pool`만 | 기본안 |
| Q7 | Q3 확인에서 건물 배치 정확 칸(지금 0.25)이 얼마나 떨어지면 linear projection을 접을까 | 사용자 판단(문턱을 미리 두지 않는 것이 지금까지 원칙) | **안 접음.** 위치 지표 값만 계속 추적 |
| Q8 | r3 학습을 생략하고 d1을 r2에서 시작할까, r3도 학습해 [041]-R29 ④ 선택을 다시 할까 | 생략, r2 시작 | 기본안 |
| Q9 | 학습률: 원래 레시피 그대로인가, 큰 모델용으로 낮춘 run을 더할까([model_scaleup] 9절 미결) | 원래 레시피 | 기본안 |
| Q10 | update 수: 원래 값으로 고정할까, 원래 규칙(한 epoch 재집계)을 따를까 | 원래 규칙 | 기본안 |
| Q11 | 게임 추론: CUDA 호스트만 쓸까, [060]-R3 추론 메모리 상한을 고쳐 ROCm도 쓸까 | CUDA만 | **ROCm에서도 동작하게**(060-R3 상한 개정, 값은 4번 실측 뒤) |
| Q12 | 1만 update 보고 때 멈춤 여부 | 보고만, 멈추지 않음 | 기본안 |
| Q13 | CNN이 빠진 만큼 다른 곳(코어 등)을 더 키울까 | 4번 실측 뒤 결정 | **안 키움** |

## 7. 작업 순서 (시간을 정하는 것)

| # | 작업 | 담당·시간 | 사용자 확인 | 상태(11-20 11시) |
|---|---|---|---|---|
| 0 | BC 2배 느려짐 조사(t2 설정 `bench bc` 100 update, 로그·`.env`·코드 차이·manifest 증가 비교). 고치지 않고 리포트 | 서브 에이전트, 이 PC 단독, 수 시간 | 원인·고칠지 | 완료 |
| 1 | spec 초안 | 서브 에이전트, 1~2시간 | **승인** (2026-11-20 10:0x, [045]) | 완료 |
| 2 | 코드(8절) + 테스트 + 별개 리뷰 | 서브 에이전트(worktree), 4~6시간 | – | 완료 |
| 3 | 안 함(Q3) | – | – | – |
| 4 | 실측: 파라미터, [060]-AC7 최악 모양(ROCm 12 GiB), CUDA 실데이터 100 update 속도·메모리, 추론 지연·메모리 | 서브 에이전트, 2~3시간 | **일정 확정** | 진행 중 |
| 5 | 표본 끝-끝(판 → 재레이블 → shard → mix-fractions → BC 50 update, 모든 단계) | 수 시간 | – | 준비 완료, GPU 대기 |
| 6 | 본 줄기 | GPU. CNN 포함 B는 약 9~10일(×4.5/update) — CNN을 빼면 줄어듦, 4번에서 다시 계산 | 1만 update 곡선 | – |
| 7 | 측정(`s2e2b`·`t1b`·`t2b`) + 리포트 `docs/report/experiment/<ts>_m2b_scaleup.md`(실험 전 배경·예상 포함) | GPU·게임 | 결과 | – |

- 0과 1은 병렬로 해도 된다(1은 GPU를 안 씀).
- 이 PC를 사용자가 쓰는 동안은 [060]-R8 스왑 규칙을 따른다. ROCm에서 B는 50~60초/update라 사실상 거의 멈춘다.

## 8. spec·코드 범위

| spec | 변경 |
|---|---|
| [032] | §2.2 공간 인코더 → 패치 linear projection(구조 스위치 또는 새 기본), §2.5 6행 위치 머리, §2.7 `outer` ② FiLM, §2.8·R24 ② aux_jepa 정답, §1 그림, §4 예산, R2 크기 범위(약 1.4억), 새 R28 `[net]` 크기 표, R25·R26·R27 적용 범위 |
| 045(신규, 04x 빈 번호) | `m2b` 줄기(4절)·비교(5절)·AC |
| [050]-R9 | 측정 지점 `s2e2b`·`t1b`·`t2b`(게이트 미적용·기록만, `gn` 선례) |
| [041] | R25 죽은 채널 감시 적용 범위, R17 엔티티 재계산을 B에서 켬 |
| [060] | R3 추론 서버 메모리 상한 개정(ROCm 추론, Q11). 값은 7절 4번 실측 뒤 |

| 코드 | 변경 | 주의 |
|---|---|---|
| `python/uranium/model/config.py` | `[net]` 엄격 파서(모르는 키·타입 오류), `new_net_config(..., sizes)`, 공간 방식 설정 | `_load`(`config.py:5-14`)는 모르는 키를 조용히 무시한다. 그대로 쓰면 오타가 작은 모듈로 빠짐 |
| `python/uranium/model/` encoders·heads·net·variants | 패치 투영, `s_pool` 제거, 새 위치 머리, `outer` FiLM·aux_jepa 정답 | 끄면 지금 모델과 비트 동일([032] §2.7 공통 규칙) |
| `bc/run.py:51,63`, `bc/trainer.py:85-109` | `[net]` 전달, 새 run이면 그 크기로 생성, init·resume 때 크기 불일치면 오류 | 지금은 새 run이 **조용히 작은 모델**로 만들어지고(`trainer.py:94`), init·resume은 체크포인트 크기를 따르며 크기 검사가 없다(`trainer.py:97-100`만 검사). Rust `BcConfig`는 `[model]`·`[net]`을 지문에 안 넣음 |
| `bc/profile.py:39`, `bench/realbc.py:37` | 같은 크기 경로 | 지금은 작은 모델을 잼 |
| `measure/spatial_logs.py:24-39,129` | CNN이 없으면 해당 측정 끔 | `--checkpoint` 없으면 작은 채널 수로 계산 |
| `config/bc_m2b_*.toml`, `config/plans/*_m2b_*.toml`, `config/protocol/lineage.toml` | 4·5절 | 원래 설정과 `[net]`·공간 스위치만 다르다는 테스트(`tests/python/test_lambda_configs.py:305-335` 방식) |
| `tests/python/` | `[net]` 파서(`test_model2.py:22-47` 방식), 크기 불일치 거부(`test_spatial_norm.py:346-385` 방식), release meta 왕복 유지(`test_model2.py:60-65`) | `test_model.py:21-26`은 기본 파라미터 수·[032]-R2 범위를 검사 |

- 체크포인트 meta는 이미 크기를 저장한다(`python/uranium/ckpt.py:43,79`). 추론 서버·RL·평가·SC:R은 체크포인트에서 크기를 읽으므로 그대로 동작한다.
- 엔티티 재계산([041]-R17, `URANIUM_BC_ENTITY_CHUNK`)은 BC에만 있다. RL 학습기에는 없다(이번 범위 밖).
- 공간·위치 재계산 시제품(`local/work/size_probe/probe.py` 213~254행)은 CNN이 없어지면 필요 없다.

## 9. 위험

| 위험 | 대응 |
|---|---|
| 위치 머리가 linear projection으로 정보를 못 얻음(건물 배치·랠리 붕괴) | 사전 확인 없음(Q3). 위치 지표 추적(5절), `t1b` 러시 막음·운영 지표 |
| 크기·구조 효과를 못 가름 | 대조 없음(Q4). 리포트에 한계로 적음 |
| 줄기 9~10일 동안 이 PC 점유 | 스왑 규칙, 1만 update 보고 |
| BC 느려짐 미해결이면 기간 최대 2배 | 7절 0번 먼저 |
| seed 1개, 줄기 하나 | 리포트에 한계로 적음 |
| RL이 초반 빌드 갈림길을 쉽게 뒤집음 | 이번 범위 밖(RL 안 함) |

## 10. 산출물·자료 위치

| 자료 | 위치 |
|---|---|
| 확대 실측 원자료 | `local/work/size_probe/` |
| 파라미터 산식 | `local/work/spatial_params/count.py` |
| CNN 지워 보기 원자료 | `local/work/spatial_ablation/` |
| BC 느려짐 조사(멈춘 작업이 모은 학습 로그 5개, 분석 안 함) | `local/work/bc_slowdown/logs/` |
| 원래 줄기 실행 스크립트·로그 | `<shared>/eval/m2-lineage/scripts/`, `<shared>/eval/m2-lineage/train_logs/`, `local/work/m2_main/`, `local/work/t2_main/`, `local/work/t2_measure/`, `local/work/gn_measure/` |

[model_scaleup]: ../bench/2026_11_17_21_45_model_scaleup.md
[spatial_ablation]: ../investigation/2026_11_19_23_00_spatial_ablation.md
[core_ablation_pt]: ../experiment/2026_10_24_21_29_m6_core_ablation_pt.md
[teacher_leak_fix]: ../experiment/2026_11_17_20_08_teacher_leak_fix.md
[aux_eps_compare]: ../experiment/2026_11_19_11_36_aux_eps_compare.md
[timeline]: ../../timeline.md
[021]: ../../spec/2026_10_15_22_24_021-opponent-pool.md
[022]: ../../spec/2026_11_14_08_18_022-matchup-teacher.md
[030]: ../../spec/2026_10_15_22_24_030-observation.md
[032]: ../../spec/2026_10_15_22_24_032-model-architecture.md
[041]: ../../spec/2026_10_15_22_24_041-imitation-learning.md
[044]: ../../spec/2026_11_14_08_18_044-self-game-distillation.md
[050]: ../../spec/2026_10_15_22_24_050-evaluation.md
[060]: ../../spec/2026_10_15_22_24_060-infra.md
[045]: ../../spec/2026_11_20_09_12_045-scaleup-b-patch-lineage.md

# RL 4차 자문 — `rush-s4` 판정과 다음 run 설계 (2026-11-05)

- 배경: 원인 분석(`reports/m6_rl_rootcause.md`) 뒤, 사용자 위임(2026-11-05: "차질이면 fable 최고 effort 자문을 받아 계속 진행")에 따라 고성능 모델(fable, effort 최고)에게 다음 run 설계를 물었다(읽기 전용).
- 근거: `reports/m6_rl_rootcause.md`, `runs/rush4_20261104/{summary.md, held/c0, rush-s4/logs/games.csv}`, `runs/rushladder_20261103/features_final.json`, `<shared>/eval/rush4-20261104/c0·c1`, `third_party/openbw/sync.h`(start_game), `crates/uranium-league/src/{run.rs, league.rs, sil.rs}`, `crates/uranium-runner/src/plan.rs`, `python/uranium/rl/learner.py`.

## 0. 원인 분석 검증 (직접 계산)

- **시드 → 시작 위치는 완전히 재현된다.** OpenBW `sync.h start_game`에서 `URANIUM_SEED`가 LCG 상태가 되고, 멜레는 슬롯 0~3을 Fisher–Yates(`lcg_rand(33,0,i)`, LCG `x·22695477+1`, `(x>>16)&0x7fff`)로 섞는다. 파이썬 15줄 시뮬레이션이 c0 리플레이 헤더 g1~g8과 8/8 일치. 예측: c0(43001~43040) "첫 목표 적중" 36/40(실측 33/40, 차이는 오버로드 정찰), 사다리(44001~44030) 0/30(실측 0/30). 무작위 시드 20만 개: 12개 슬롯 쌍 균등, 적중 0.332 → 학습(해시 시드)은 정확히 1/3.
- 조건별 결과(c0, 예측 조건으로 교차): T0 적중 36판 막음 4(11%)·승 9, 빗나감 4판 막음 1·승 2. **UAB12 적중 36판 막음 0**, 빗나감 4판 막음 1. 사다리(전부 빗나감) T0 도착 p50 4:08·막음 10/30, UAB12 10/30. → 원인 1 확인. UAB12는 이른 조건에서 0%.
- 학습 판(games.csv 1,423행): 러시 승 374 중 9풀히드라 198(53%, 최근 300판 56%). T0 300판 중 Concede 162(54%)·승 84, 승 판 judge_scv p50 24(= 러시가 약하게 온 판). UAB12 승 27%는 "빗나감 조건(2/3) × 33%"와 일치 → 학습 판 승은 조건의 운이다. 원인 3·5 확인.
- **원인 2(privileged critic)는 방향을 바로잡는다**: 행동과 무관한 baseline은 어떤 것이든 편향이 없다(정책 gradient 정리). privileged V는 도착 시각이라는 외생 분산을 빼 주므로 오히려 유리하다. advantage가 작은 진짜 이유는 "한 결정이 승패를 바꿀 확률이 작다"(벙커 → 탑승 → 수리 → 일꾼 동원 사슬이 다 맞아야 함)이다. win_open을 baseline으로 쓰면 분산만 는다. **critic은 바꾸지 않는다.**

## 1. `rush-s4` 판정: 3.0M 체크포인트가 써지면 정상 종료

2.39M step 동안 학습 판 T0+UAB12 미니게임 승률 24~29% 평탄, c1 T0 40판 11-10-19 = c0 11-11-18. 3M 기준(막음 55%)은 기준값 자체가 틀려 도달 불가. 3.0M 체크포인트를 다음 run 출발점으로 쓰므로 거기까지만 둔다. 진행 중 c1 평가는 끝까지 두되 구 설계 3M 점검(c2)은 돌리지 않는다(새 설계로 AC1과 함께 측정).

## 2. 측정 수정 (먼저, GPU 0시간)

1. **시드 → 슬롯 생성기**를 Rust(`crates/uranium-runner`, 테스트: 위 8판 헤더 일치)와 드라이버에 둔다. `hit = our_slot == min{i ≠ bot_slot}`(T0 `attack_target`의 인덱스 순 규칙, `crates/uranium-scripted/src/lib.rs:385`; UAB도 같은 방향으로 작동).
2. **plan에 판별 시드 목록** `seeds = [...]`(`plan.rs` Plan 구조체, 289행 `seed_base + id` 대신 `seeds[n-1]`; 020 spec 한 줄).
3. **점검 설계(`rush-s5` 이후 공통)**: 상대 2(T0 러시·UAB12) × 맵 2 × 32판 = **128판/점검**. 맵마다 적중 쌍 4종 × 4판 = 16, 빗나감 쌍 8종 × 2판 = 16(슬롯 쌍 12종 모두). 시드는 생성기로 고르고 **모든 체크포인트에 같은 목록**(조건 짝 비교). 보고: 조건별 막음·승·8:00 SCV ≥ 8 비율 + 자연 분포 가중(1/3·2/3) 합산 + c0 대비 Fisher p. 검정력: 이른 조건 64판에서 4/64 → 12/64(+13%p) p ≈ 0.03.
4. **막음을 리플레이 재생 없이**: `crates/uranium-env/src/scout.rs` `ScoutMetrics`에 `scv_min_1m`(1:00 이후 자기 SCV 최솟값)·`army_s`(적 병력 첫 목격 초, `army_found` 자리) 추가 → 세션 결과(`results.jsonl` model 행)와 `games.csv`. 막음 = `scv_min_1m > 4`. 리허설에서 c0 80판 `held/c0`와 대조(≥ 95% 일치). `held.py`·dumper(80판 1.8시간, RAM 압박) 폐기.
5. `games.csv`에 `slot_self, slot_enemy, start_hit` 추가(run.rs 시드 생성 직후 시뮬레이션). 드라이버는 학습 판을 이른 접촉(`army_s ≤ 225`)/늦은 접촉으로 나눠 상대별 미니게임 승률을 200판 창으로 보고.
6. 첫 적용: AC1과 `rush-s4`@3.0M을 같은 128판 목록으로 측정(= `rush-s5` c0, 기대: 차이 없음, 이른 조건 약 5~10%).

## 3. 다음 run `rush-s5`

순위: ① 측정 수정(필수) ② 이른 접촉 자기 모방 + 고정 상대 비중 + 조건 통제 시드(아래, 약 8 GPU시간) ③ 정찰 반응형 스크립트 교사 시연(실패 대비 1순위, CPU라 병행 착수 권장) ④ advantage 정규화·lr(약함) ⑤ critic 변경(하지 않음) ⑥ 결정 간격·행동 공간(BC 재학습 필요, 보류).

| 항목 | 값 |
|---|---|
| 출발 | `--init rush-s4/ckpt-<3.0M>`(8:00 판정에 맞춰진 value head 유지, AC1과 유의차 없음), `--bc` BC 그대로, KL 0.1 스케줄·λ 0.95·보상·항복·미니게임 그대로 |
| 상대 몫 | `ladder_groups = { rush = 0.70, macro = 0.20 }` + 모델 0.10. warmup(9풀 91%) 삭제 |
| 러시 선택 | `ladder_pick = { rush = "fixed", macro = "f_var" }`, `[[ladder]] weight`: T0 0.35·UAB12 0.30·9풀10링 0.20·9풀히드라 0.15. band는 세 run 동안 한 번도 의도대로 작동하지 않음 |
| 조건 통제 | `[league.start_mix] rush_hit_share = 0.5`: 러시 그룹 판만 시드를 재추첨(≤ 64회, game id 결정론)해 적중 조건 1/3 → 1/2(T0 무수정). 운영형·모델 판은 자연 시드 |
| 자기 모방 | `[sil]` 유지(lanes 4, w_pg 0.05, 버퍼 256) + `select_keys = ["t0-zerg-rush", "ualbertabot12poollings-zerg", "ualbertabot9pool10lings-zerg"]`(히드라 제외), `select_army_before_s = 225`(이른 접촉 승 판만). 버퍼가 "일찍 온 러시를 막은 판"만 담는다 |
| 점검 | 2절 설계, c0 + **3M마다**(128판, 로컬 2슬롯 약 3시간) |
| 판정 | 100 update: 러시 판 적중 비율 0.45~0.55, 상대 비중 ±5%p, 버퍼 증가가 이른 승 판에서만. **3M**: 학습 판 이른 접촉(T0+UAB12+9풀10링) 승률 최근 200판 ≥ 첫 200판 +8%p(p < 0.05) **또는** 점검 이른 조건 막음 ≥ c0 +10%p(p < 0.05) → 6M까지, 둘 다 미달 → **중단, 교사 경로**. **6M**: 이른 조건 막음 ≥ 25% 그리고 늦은 조건 막음 ≥ c0 −5%p → 계속 여부 사용자 결정 |
| 중단 규칙 | ①~③ 그대로, ④ 퇴보 게이트는 "같은 시드 늦은 조건 막음이 c0보다 15%p 넘게 낮음" |

- spec: 042(새 run 절, R16 `select_keys`·`select_army_before_s`, 신규 `[league.start_mix]`, `fixed` 선택, games.csv 5열, AC9 테스트), 050(점검 128판·시드 목록·세션 기반 막음·조건별 보고), 020(plan `seeds`). 033·032 변경 없음.
- 코드: `plan.rs`, `run.rs`, `league.rs`, `sil.rs`, `scout.rs`, `config/rl_rush5.toml`, `runs/rush5_<date>/driver.py`. Python learner 무변경.
- 리허설: 시드 목록 10판 리플레이 헤더 = 시뮬레이션, 새 열 채워짐, 막음(세션) = `held.py`(c0 80판), 고정 비중·적중 비율, 버퍼 로그, 점검 동시 RAM 여유 ≥ 3 GB.

## 4. 실패 시 순서

1. **정찰 반응형 스크립트 테란 교사**(`uranium-scripted` 새 Kit, 상대 T0는 그대로): 1:30 SCV 정찰(가까운 시작 위치부터) → 풀·저글링 목격 시각·수로 "러시" 식별 → 벙커·마린 탑승·SCV 수리·일꾼 동원, 아니면 확장. 새 점검으로 T0·UAB12 이른 조건 막음 ≥ 50% 확인 → 러시 4종 + 운영형 5종 상대 약 1,000판 botgen(CPU 4~8시간) → BC 파인튜닝(사람 S2 + 교사 shard 약 20%, 약 3,000 update) → 사람 val top-1 −0.5%p 이내·이른 조건 막음 ≥ 25% 확인 → `rush-s6`의 `--init`·`--bc`. 1~3일. `rush-s5` 중 병행 시작 권장.
2. 조건 통제 강화(`rush_hit_share` 0.7) + `w_pg` 0.1.
3. 교사 궤적을 버퍼 레인(DAPG식)으로 직접 공급.
4. 결정 간격·선택 상한 변경(BC 재학습).

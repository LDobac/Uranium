# 교사 러시 식별 규칙 자문 (2026-11-05)

- 배경: 교사 `t0-terran-react`(021-R8) 리뷰에서, 러시 식별 규칙이 평범한 운영형 저그에도 거의 항상 성립해 사용자 제약("무조건 방어부터 하는 봇 금지")과 충돌한다는 지적이 나왔다. 사용자 위임(2026-11-05)으로 fable(effort 최고)에 규칙 수치와 운영형 점검 기준을 물었다. 읽기 전용.

## 근거
- 적 유닛의 `remaining_build`는 항상 0이다(BWAPI `UnitUpdate.cpp` 656행은 `canAccessInside()`에서만 채움). HP는 보이면 쓸 수 있다. OpenBW에서 풀(750 HP·1,200프레임)은 착공 HP 75 + 0.5625 × 경과 → **착공 프레임 = 현재 − (hp − 75) × 16/9**(오차 < 1초). 완성 풀은 "착공 ≤ 현재 − 50초"만 안다.
- 풀 착공(저그 시점 중앙값, `reports/m6_rush_ladder.md`·`reports/oss_bot_porting.md`): 러시 — UAB 기본 0:45, 9pool계 0:55~0:58, SH 오버풀 1:09, UAB 12poollings 1:17, SH 11pool 1:21, T0 zerg_rush 1:28(1:25~1:32). 운영 — UAB 2hatchhydra 1:54·1:57, 3hatchmuta 1:58·2:04(`runs/macroladder_20261103/smoke/op_*.json`). SH 2hatchmuta·T0 표준은 미측정(추정 1:50~2:05).
- 저글링: 러시 첫 저글링 1:55~2:37. UAB 운영 2종은 저글링 0(첫 접촉은 히드라 약 5:00+). SH 2hatchmuta는 발업 저글링 3쌍(추정 3:05~3:30), T0 표준은 12기 웨이브(약 4:15), T0 muta는 홈가드 6기. SH 감시 저글링은 4:00 전 없음.
- 정찰 도착(추정): 1:35 출발 → 1번째 시작 위치 2:10~2:20, 2번째 2:50~3:00, 3번째 3:30~3:40.

## 1. 개정 규칙 (관측마다, 8:00 전, 하나라도)
- ①a 미완성 풀의 착공 추정 ≤ **1:40(2,400프레임)**. ①b 완성 풀을 **2:30(3,600프레임) 전**에 봄. 러시 전부 포함(T0 rush p90 1:32), 운영 제외(최소 1:54).
- ② 저글링 ≥ **2기를 2:50(4,080프레임) 전** 어디서든, 또는 ≥ **6기를 3:10(4,560프레임) 전** 어디서든.
- ③ 적 병력(033) **≥ 2기**가 한 관측에 자기 CC 20타일 안(4:00 뒤에도 유효).
- ④′ 정찰 SCV가 적 시작 해처리 중심 **3타일 안**일 때 그 12타일 안 드론 ≤ **6** ∧ 풀 보임 ∧ 기억된 적 해처리가 그것뿐 ∧ 3:30 전. 기존 ④(드론 ≤ 8, 시야 가장자리)는 삭제.
- 분류: 러시 = t0-zerg-rush, UAB 12poollings·9pool10lings·9pool12lings·9poolhydra, UAB 기본, SH 오버풀, SH 11pool. 운영(엄격) = UAB 2hatchhydra·3hatchmuta. 운영(증거 조건부) = SH 2hatchmuta, t0-zerg-muta. **t0-zerg는 운영이 아니다**(1해처리·드론 14·저글링 12기 웨이브 원베이스 올인 — ②·③ 뒤 벙커가 맞음, 단 ①로 걸리면 안 됨).

## 2. 점검 기준 (상대당 10판)
- 기록 추가: `pool_start_s`·`pool_seen_s`·`lings_max`·`contact_s`(병력 CC 20타일 안 첫 시각)·`bunker_s`·`rush_rule`.
- "선제 벙커" = `bunker_s < 240`이고 그 전에 ③ 없음.
- UAB 2hatchhydra·3hatchmuta: 4:00 전 벙커 0/10, ①②④′ 발화 0/10, `expand_s ≤ 240` ≥ 9/10, `scv_8m ≥ 16` ≥ 9/10.
- SH 2hatchmuta·t0-zerg-muta: 선제 벙커 0/10, ①②④′ 발화 0/10(③만 허용), `scv_8m ≥ 16` ≥ 8/10, 확장 ≥ 8/10.
- t0-zerg: 운영 plan에서 빼고 따로 10판. 선제 벙커 0/10, ① 발화 0/10, 막음 ≥ 8/10, `scv_8m ≥ 14` ≥ 8/10.
- 러시 128판은 그대로(이른 ≥ 32/64, 늦은 ≥ 40/64).

## 3. 설계 변경
- `ReactParams`: `pool_start_max` 2400 + `pool_done_before` 3600, `lings_min/before` 2/4080 + `lings_many/before` 6/4560, `army_min` 2, ④′(`drones_max` 6, `main_look` 3타일). 021-AC3 경계값 갱신. 풀 HP는 `UnitState.hp`, 완성 여부는 `flags::COMPLETED`.
- 정찰 look 5타일/10초 → **3타일/15초**.
- 확장 뒤 규칙이 서면 방어 가지의 기준 CC를 **적 병력에 가장 가까운 자기 CC**(보통 앞마당)로.
- 미측정 두 값(T0 표준·SH 2hatchmuta 풀 착공)은 점검 첫 판들의 `pool_start_s`로 확인. 1:46 아래가 나오면 `pool_start_max`를 1:35(2,280프레임)로, `lings_many_before`를 3:00으로 내린다.

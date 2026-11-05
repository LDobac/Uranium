# RL 6차 자문 — `rush-s5` 판정 뒤 해석·교사 설계·CUDA 오류·GPU 계획 (2026-11-05)

- 배경: `rush-s5` 6M 판정 미달(점검 이른 막음 c2 6/64 = c0 6/64, 학습 판 적중 조건 승 평탄)로 spec대로 12:45 정지. 같은 시각 learner가 CUDA illegal memory access로 12:05부터 멈춰 있었다. 사용자 위임(2026-11-05)에 따라 fable(effort 최고) 자문, 읽기 전용.
- 근거: `runs/rush5_20261105/{summary.md, driver.log, rush-s5/logs/games.csv(1~3,335행), train.csv}`, `rush-s5/supervise.log` 15009~15400행, `<shared>/eval/rush5-20261105/{c0, c0-ac1, c1, c2}`, `python/uranium/{rl/learner.py, rl/run.py, agent/infer.py, device.py, kernels/attn_bias.cu}`, `crates/uranium-league/src/{run.rs, sil.rs}`, `crates/uranium-train/src/supervise.rs`, `crates/uranium-scripted/src/*`, spec 030·031·034·040·041·042.

## 1. 해석 — 보상만의 RL은 끝, 교사 경로

- 점검(같은 128판): 이른 막음 c0 6/64 → c1 11/64 → c2 6/64(c2 vs c0 p 0.618, c1의 11은 잡음 p 0.149), 늦은 23 → 21 → 22, AC1 4/64·14/64.
- 학습 판(05:08 재시작 뒤, 3상대·`start_hit = 1`): 680판 87승(12.8%), 100판 블록 13·15·12·9·13·14·11(/80) — 평탄. T0 48/272(18%), UAB12 20/239(8%), 10링 19/169(11%), 빗나감 조건 36%.
- 자기 모방은 **작동했는데 효과 0**: 적중 승 124판 선택, 버퍼 256 가득, `sil_pos_frac` 0.8~0.97, `sil_loss` 0.05~0.19. 정책은 움직였다(KL 0.042 → 0.076, `entropy_action` 0.79 → 1.04 → 0.96) — 더 무작위해졌을 뿐.
- 행동 상관(적중 판): 승은 벙커 64%(p50 3:44)·본진 발견 32%·4분 SCV 14.3, 패는 14%·19%·5.6. 필요한 행동은 있지만 드물다(BC 벙커 결정당 < 1%).
- 결론: 현재 BC 사전분포 + 결정 단위(0.45초·12기·명령 1개)에서 **보상만으로 초반 러시 방어를 올리는 RL은 막다른 길**(5 run, 약 2천만 step 평탄). 값싼 변형(`rush_hit_share` 0.7·`w_pg` 0.1)은 같은 장치의 세기만 올림 → ≥ 14/64 확률 5~10%, 1 GPU일. 하루당 기대값: RL 변형 약 +0.5%p/일, 교사 경로 약 P 0.4(교사 ≥ 50% 0.7 × BC 전이 ≥ 25% 0.6) × +16%p ÷ 3.5일 ≈ +1.8%p/일 + 재사용 자산. **교사 경로. RL 변형은 돌리지 않는다.**

## 2. 교사 설계 (spec 초안)

Kit `T0Build::TerranReact`(id `t0-terran-react`), `crates/uranium-scripted` Brain 확장. 010 Command로 모두 가능: `Build`(벙커), `Train`, `Unit{Load(bunker)·Repair·Attack(Unit/Pos)·Hold·Gather}`. 모델 대응(034): Build_Bunker(빌드타일 = 교사 tile), Load_Unit, Repair_Unit, Attack_Unit/Pos(12기), Hold, Gather_Unit, Train_* — **표현 불가 항목 없음**. 조건: 교사의 기억은 모델에서 기억 적 건물 엔티티(030 §2)뿐이므로 **판단은 목격한 그 결정에서 즉시**(지연 판단 금지), 방어 상태는 보이는 유닛(벙커)에서 읽히게.

- **빌드**: 8 SCV, 9 서플, 11 배럭(1:35), 배럭 착공 SCV가 정찰(가까운 시작 위치 순, 기존 `scouting` 재사용, 발견 뒤 복귀), 15 서플, 배럭 완성부터 마린 연속, 2번째 배럭은 방어 가지에서 즉시·운영 가지에서 확장 뒤. 가스·테크 없음(8:00까지의 반응이 목적).
- **러시 식별 R**(자기 시야만, 하나라도): ① 풀을 2:30 전에 봄 ② 저글링 ≥ 2기를 3:30 전 어디서든 봄 ③ 적 병력(033 `army` 정의)이 CC 20타일 안 ④ 적 본진에서 드론 ≤ 8 + 풀.
- **운영 가지 M**(R 아님): 미네랄 400이면 앞마당 CC(맵 사전계산 `base_locs` 중 CC에 가장 가까운 곳; runner가 `base_locs`·`chokes`를 Kit에 넘김), 이후 `terran()` 운영(배럭 3·10기 웨이브) + 홈가드 + 앞마당 벙커 1. 2:40까지 본진 미발견이면 **선제 벙커 없이** 확장만 3:40까지 보류(사용자 제약).
- **방어 가지 D**(R 즉시): ① 벙커 1: CC에서 초크 방향 4~6타일, `Placer` 나선 첫 자리 ② 마린: 벙커 완성 시 ≤ 4기 Load, 나머지 벙커 옆 Hold ③ SCV: 적 병력이 CC 12타일 안이면 채취 SCV 중 가까운 min(8, n−4)기 Attack(Unit) 최근접 저글링(48프레임마다 재지정), 벙커 HP < 90%면 2기 Repair ④ 적이 16타일 밖 10초 → Gather 복귀 ⑤ 마린 ≥ 8·60초 무접촉 → M.
- **검증**: 050-R6 128판 plan에 `p0 = {kind="t0", race="terran", build="terran_react"}`로 `uranium-env run`(GPU 불필요). T0 세션에도 `scv_min_1m`·`scv_8m` 기록(`crates/uranium-agent/src/lib.rs`). 목표: 이른 막음 ≥ 32/64, 늦은 ≥ 40/64, 운영형 5종 각 10판 8:00 SCV ≥ 16이고 4:00 전 벙커 0건("항상 방어 아님" 확인).
- **botgen**: 러시 4종 × 2맵 × 200 = 1,600 + 운영형 5종 × 2맵 × 40 = 400 = 2,000판(해시 시드, `save_replay`, 로컬 + Linux 3슬롯 약 5시간). `src = botgen`(040 표 문구 개정 필요). build에 `--until-frame 11520`(8:00 뒤 레이블 제외). 약 1,070 step/판 → 1,500판 약 1.6M step.
- **파인튜닝**: `bc-s2e2-pt/ckpt-11387`에서, 데이터 = 대상 맵 train(S2) 1/4 추출(약 5.8M step) + 교사 shard 1,500판(약 22%), 1 epoch 약 3,000~3,500 update, lr 1.5e-5·min 1.5e-7·warmup 5%·cosine. 게이트: val500 top-1 ≥ 기준 −0.5%p, 교사 hold-out 10% top-1 ≥ 60%, 128판 이른 ≥ 16/64·늦은 ≥ 20/64, T0 stochastic 100판 ≥ 29%. 통과 → `rush-s6` `--init`·`--bc` = 파인튜닝 체크포인트.
- **공수**: Kit/Brain 1일, 메트릭·plan·build 옵션 0.5일, 점검 반복 1일, botgen + build + 파인튜닝 + 평가 1~1.5일 → 3.5~4일.

## 3. CUDA illegal memory access

- **사실**: 12:05 update 4,250 스냅샷(`save_checkpoint` ckpt-7362114 → `league_snapshot` → `prune_servers`로 스냅샷 3250 서버 close) 직후 첫 오류가 live Main 서버의 D2H에서 보고. sticky라 프로세스 CUDA 컨텍스트 전체가 죽음. learner는 update 4,251 배치를 계속 기다림(추론 실패 판은 결정 0 → 세그먼트 없음), 12:45 저장도 실패. **LATEST `ckpt-7362114`는 온전**. `games.csv`는 3,335행까지 유효(이후 19,310행 결정 0).
- **원인 순위**: ① 스냅샷 동시성 — 서버 close의 `_free()`(graph 파괴 + `empty_cache()`, `infer.py` 199~202)가 learner update·live 서버 graph replay와 겹침. 이번 run prune 13회·이전 run 수십 회 중 1회 → 경합 ② Windows CUDA 호스트 하드웨어·드라이버 일회성 ③ 융합 attention·Inductor 커널 OOB(낮음) ④ 자기 모방 레인·bf16(아님).
- **싼 진단**: 이벤트 뷰어 nvlddmkm/TDR 12:05 전후 + `nvidia-smi -q`, `compute-sanitizer --tool memcheck`로 커널 테스트 스윕, 스트레스(스냅샷 prune 약 300회 1시간).
- **가드(원인 무관, 필수)**: (a) `infer.py`·`run.py`: `AcceleratorError`·"illegal memory access"·"Sticky error"면 저장 없이 `os._exit(70)` → supervise가 LATEST에서 재시작 (b) `run.rs` 학습 루프: games.csv가 20행 이상 늘었는데 `hang_minutes/2` 동안 update 없음, 또는 연속 10세션 추론 실패·결정 0 → 비정상 종료 (c) `supervise.rs`: `--progress-file`이 있으면 파일 변경만 진행으로 + `--fatal-pattern` (d) 드라이버: train.csv 10분 정체 + games.csv 증가 → STOP·보고 (e) 완화: `_free()`를 `DEVICE_LOCK.exclusive()` + `torch.cuda.synchronize()` 뒤 `empty_cache()`.

## 4. GPU 1~3일

RL run 없음. ① `bc-s2e2-pt/ckpt-11387` 128판 점검(파인튜닝 게이트 기준값) ② 교사 없는 '빈 파인튜닝'(같은 recipe, S2 1/4, 3,000 update) → val500·128판 점검 = 잡음 바닥 ③ 3절 진단과 가드 구현·리허설 ④ 나머지 유휴. 교사 점검·botgen은 CPU라 GPU 작업과 겹쳐도 된다.

사용자 결정 후보: 2:40 미발견 시 확장 보류 규칙, 교사 데이터 8:00 절단, 040 botgen 정의 개정, 파인튜닝 S2 1/4 추출.

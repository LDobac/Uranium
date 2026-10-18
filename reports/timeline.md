# 프로젝트 타임라인

최종 보고서를 쓸 때 중간 산출물을 시간 순서로 되짚기 위한 색인이다(2026-10-18 사용자 지시).

## 기록 규칙
- 결정, 발견, 장시간 실행(학습·build·평가)의 시작과 끝, 새 리포트를 한 줄씩 덧붙인다. 지우거나 고치지 않는다. 틀린 내용은 아래에 정정 항목을 새로 쓴다.
- 형식: `- HH:MM — 내용 (근거: 커밋 / 리포트 / spec ID)`. 시각은 기록하는 PC 기준이다.
- 자세한 내용은 해당 리포트에 쓰고, 여기에는 링크만 둔다.
- 중간 산출물(체크포인트, 학습 로그, 평가 출력, 대전 결과)은 지우지 않고 위치를 적는다. 개인·보안 정보는 쓰지 않는다(001-R10, `<shared>` 같은 자리표시자).
- 2026-10-18 이전 항목은 git 기록과 리포트 작성 시각으로 되살렸다. 세부는 각 리포트와 `git log`가 원본이다.

## 2026-10-15
- 22:24 — M0: spec 000–061, spec driven 규칙 (커밋 "M0: spec-driven project setup")
- 22:24 — M1: Burn/Vulkan 벤치(`bench_bc250.md`), OpenBW 빌드(`openbw_spike.md`), CASC 추출(`mpq_extraction.md`)
- 22:29 — M2: proto·브리지·T0 테란·셀프매치 (`m2_selfmatch.md`)

## 2026-10-16
- 00:18 — T1 오픈소스 봇 6종 BWAPI 4.2 포팅 (`oss_bot_porting.md`)
- 00:32 — M3 Linux: hostd·스케줄러·calibrate (`m3.md`)
- 11:33 — M3 Windows 단일 PC (`m3_windows.md`, 011-AC8)
- 12:28 — M3 원격 구성 (b) (`m3_remote.md`, 020-AC1·011-AC6)
- 13:56 — M4 리플레이 스파이크 (`m4_replay_spike.md`)
- 17:20 — 데이터 출처 정리 (`dataset_sources.md`)
- 22:19 — M5 데이터 독립 부분: 모델·feature·마스크·BC 골격 (`m5_model.md`). LSTM·GTrXL 두 코어 구현, CUDA 속도 거의 같음
- 23:08 — SC:R 래더 수집기 실측 (`m4_ladder_probe.md`, 043)
- 결정: CUDA PC는 Vulkan보다 빠르면 032-AC2/AC3 통과, CUDA 허용 오차 1e-2(AMB-032-5), `last_seen_age`(AMB-030-1)

## 2026-10-17
- 04:29 — STARDATA 표본 검증, 레이블러 버그 수정, 무효 레이블 17.6% → 3.6% (`m4_stardata_preflight.md`)
- 06:03 — BC GPU 메모리 `micro_lanes` (`m5_bc_memory.md`)
- 06:07 — SC:R 디싱크 검증, 맵 제외 040-R16 (`m4_scr_desync.md`)
- 17:25 — 회귀 테스트용 봇 (`m6_regression_bots.md`)
- 17:51 — 모델 정책 세션·`uranium-runner` (`m6_model_policy.md`, 8fe4656)
- 22:01 — RL 3단계 결정 AMB-042-5..9, AMB-050-2 (f6919d9)
- 밤 — 리플레이 전체 shard build 시작(4대, 78파트)

## 2026-10-18
- 새벽 — 전체 build 완료, manifest 생성(9cda147), BC 분할 확정 AMB-041-3·4(eb9b311)
- 04:00 — RL learner·리그·`uranium-train rl` (e25dce8)
- 09:50 — RL 스모크 리포트 (`m6_rl.md`, 9135fd6)
- 11:04 — 디싱크 판정 비결정성 조사 (`m4_desync_nondet.md`), 브리지 수정(f74bc54)
- 11:38 — BC 준비 리포트 1차 (`m6_bc_preflight.md`)
- 12:14 — 오판 96개 시점 추가, 데이터 35,966게임 (f4d048e)
- 14:53 — BC 1 epoch 단계, 1시간 체크포인트, BC 준비 2차 (3181bc1). 본학습은 승인 대기
- 18:38 — **발견**: 시간 축 코어 비교(AMB-032-4의 "M5에서 비교")가 절차·AC 없이 남아 있었다. **결정**: BC 본학습 전에 LSTM·GTrXL을 S1 5,300 update씩 짧게 비교한다(032-R17·AC7, AMB-032-6, `m6_core_ablation.md`)
- 19:01 — 코어 비교 T0 대전 판수 확정: 조합당 50판, 합계 600판 (032-R17, `m6_core_ablation.md`)

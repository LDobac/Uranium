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
- 19:05 — 코어 비교 시작 전 남은 확인 5가지 정리: val 500게임 편중, GTrXL BC 미검증, 장비 혼합 기간, 맞대결 경로 미시험, seed 1개 (`m6_core_ablation.md` 3절)
- 16:00 — BC 준비 3차: 모듈별 프로파일(`bc-profile`), 결과 동일 최적화(LSTM 묶음 계산, teacher forcing location 사용 행만) update 약 18% 단축, Burn autotune·fusion은 conv gradient 2–5% 변화로 제안만, Ctrl+C 시험 (`m6_bc_preflight.md` 3차, 7d395e5)
- 18:49 — 학습 증강 4종(transpose 제외)·start_loc 변환(030-R4·AC6), bc250 BC 학습 f32 예외 확대(032-R8, 060-AC2 닫힘), 041-R6 클래스 가중치 미적용 결정 기록 (2d8dc4e)
- 20:22 — GTrXL 실제 BC 설정 확인: 이 PC·BC250 1호기 모두 OOM·TDR 없음 (`m6_core_ablation.md` 1.5절)
- 20:50 — **발견**: BC 본학습 전 전수 리뷰(e190799, 리뷰어 4명). debug 테스트 11개 실패(heads.rs Bool select, 7d395e5), 기존 데이터 영향 3건(투혼 CP949 제목 → other 약 2,900게임, Train 연타 병합으로 Train 레이블 약 9.4% 손실, value result 표본 70% 0), S2 APM 기준 040↔041 모순, 맵 사전계산 누락 무음 (`m6_pre_bc_review.md`)
- 21:00 — 감시 재시작 `supervise`(060-R6), run 큐 작업자·양보 요청·장비 기록(061-R13·R14·R15) 구현, 두 PC 실기 시험 (`m6_bc_preflight.md` 4차). 시험 run은 지움
- 21:05 — GTrXL 대 LSTM 맞대결 스모크 3판 통과, 요약표에 두 진영 정책·코어 표시와 모델끼리 대전 승리 집계 버그 수정, 코어 비교 대전 리플레이 전부 저장 (695cbfe, `m6_core_ablation.md`)
- 2026-10-19 01시 — BC 본학습 전 통합 체크리스트(`m6_pre_bc_review.md` F·G절), 정규화 상수 spec 기록(030 §6), CLAUDE.md 정밀도·진행 상황 갱신, 문서의 장비 호스트 이름을 중립 표기로 교체(001-R10), 병합된 worktree·브랜치 정리
- 2026-10-19 01시 — **발견**: 5661ff8·695cbfe 추가 리뷰(`m6_pre_bc_review.md` H절). worker가 `init_run` DONE을 확인하지 않아 S2가 미완성 S1 위에서 시작될 수 있음, 공유 I/O 오류 시 `peek_lock` Free 해석·worker 종료, 잠금 자동 인수의 mtime 시계 가정, Linux 포그라운드 Ctrl+C 이중 SIGINT. 테스트 196 통과 / 11 실패(B-1과 같은 원인)

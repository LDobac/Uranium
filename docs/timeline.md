# 프로젝트 타임라인

결정·발견·장시간 실행·새 리포트를 시간 순서로 찾는 색인. 상세는 링크한 문서에 있다.

## 기록 규칙
- 날짜마다 `## YYYY-MM-DD`, 그 아래 `- HH:MM — 한 일 ([근거 문서](상대 경로))` 한 줄(길어도 두 줄). 시각은 기록하는 PC 기준 `HH:MM`, 어림은 `HH:Mx`·`HH시`, 구간은 `~`, 모르면 `시각 없음`.
- 새 항목은 해당 날짜 끝에 덧붙인다. 틀린 내용은 지우지 않고 `정정:` 항목을 새로 쓴다.
- 코드(`042-AC1`, `AMB-…`)만 쓰지 않고 출처 문서를 링크한다.
- 중간 산출물(체크포인트, 로그, 평가 출력)은 지우지 않고 위치를 적는다. 개인·보안 정보는 쓰지 않고, 공용 드라이브는 `<shared>`로 쓴다.
- 2026-10-18 이전 항목의 시각은 git 기록과 리포트 작성 시각으로 되살린 것이다.
- 2026-11-16에 기존 항목을 이 형식으로 줄였다. 줄이기 전 원문은 git 기록(`git log --follow docs/timeline.md`)에 있다.

## 2026-10-15
- 22:24 — M0: [spec 000–061](spec/README.md) 작성, spec driven 규칙 도입 (커밋 "M0: spec-driven project setup")
- 22:24 — M1: Burn/Vulkan 벤치([bench_bc250](report/bench/2026_10_15_22_24_bench_bc250.md)), OpenBW 빌드([openbw_spike](report/milestone/2026_10_15_22_24_openbw_spike.md)), CASC 추출([mpq_extraction](report/milestone/2026_10_15_22_24_mpq_extraction.md))
- 22:29 — M2: proto·브리지·T0 테란 봇·셀프매치 ([m2_selfmatch](report/milestone/2026_10_15_22_29_m2_selfmatch.md))

## 2026-10-16
- 00:18 — T1 오픈소스 봇 6종 BWAPI 4.2 포팅 ([oss_bot_porting](report/survey/2026_10_16_00_18_oss_bot_porting.md))
- 00:32 — M3 Linux: hostd·스케줄러·calibrate ([m3](report/milestone/2026_10_16_00_32_m3.md))
- 11:33 — M3 Windows 단일 PC 구성 ([m3_windows](report/milestone/2026_10_16_11_33_m3_windows.md), [011-AC8](spec/2026_10_15_22_24_011-bridge-openbw.md))
- 12:28 — M3 원격 구성 (b) ([m3_remote](report/milestone/2026_10_16_12_28_m3_remote.md), [020-AC1](spec/2026_10_15_22_24_020-env-orchestration.md), [011-AC6](spec/2026_10_15_22_24_011-bridge-openbw.md))
- 13:56 — M4 리플레이 스파이크 ([m4_replay_spike](report/milestone/2026_10_16_13_56_m4_replay_spike.md))
- 17:20 — 데이터 출처 정리 ([dataset_sources](report/survey/2026_10_16_17_20_dataset_sources.md))
- 22:19 — M5 데이터 독립 부분: 모델·feature·마스크·BC 골격, LSTM·GTrXL 두 코어 구현(CUDA 속도 거의 같음) ([m5_model](report/milestone/2026_10_16_22_19_m5_model.md))
- 23:08 — SC:R 래더 수집기 실측 ([m4_ladder_probe](report/milestone/2026_10_16_23_08_m4_ladder_probe.md), [043](spec/2026_10_16_22_48_043-ladder-replay-collector.md))
- 시각 없음 — 결정: CUDA PC가 Vulkan보다 빠르면 [032-AC2](spec/2026_10_15_22_24_032-model-architecture.md)/AC3 통과, CUDA 허용 오차 1e-2([AMB-032-5](spec/README.md)), `last_seen_age` 도입(AMB-030-1)

## 2026-10-17
- 04:29 — STARDATA 표본 검증·레이블러 버그 수정, 무효 레이블 17.6% → 3.6% ([m4_stardata_preflight](report/investigation/2026_10_17_04_29_m4_stardata_preflight.md))
- 06:03 — BC GPU 메모리 대책 `micro_lanes` ([m5_bc_memory](report/investigation/2026_10_17_06_03_m5_bc_memory.md))
- 06:07 — SC:R 디싱크 검증, 맵 제외 규칙 [040-R16](spec/2026_10_15_22_24_040-replay-dataset.md) ([m4_scr_desync](report/investigation/2026_10_17_06_07_m4_scr_desync.md))
- 17:25 — 회귀 테스트용 봇 ([m6_regression_bots](report/experiment/2026_10_17_17_25_m6_regression_bots.md))
- 17:51 — 모델 정책 세션·`uranium-runner` ([m6_model_policy](report/plan/2026_10_17_17_51_m6_model_policy.md), 8fe4656)
- 22:01 — 결정: RL 3단계 [AMB-042-5..9](spec/README.md), AMB-050-2 확정 (f6919d9)
- 밤 — 장시간 실행 시작: 리플레이 전체 shard build(4대, 78파트)

## 2026-10-18
- 새벽 — 장시간 실행 끝: 전체 build 완료·manifest 생성(9cda147), 결정: BC 분할 [AMB-041-3](spec/README.md)·4 확정(eb9b311)
- 04:00 — RL learner·리그·`uranium-train rl` 구현 (e25dce8)
- 09:50 — 새 리포트: RL 스모크 ([m6_rl](report/experiment/2026_10_18_09_50_m6_rl.md), 9135fd6)
- 11:04 — 디싱크 판정 비결정성 조사, 브리지 수정 f74bc54 ([m4_desync_nondet](report/investigation/2026_10_18_11_04_m4_desync_nondet.md))
- 11:38 — 새 리포트: BC 준비 1차 ([m6_bc_preflight](report/review/2026_10_18_11_38_m6_bc_preflight.md))
- 12:14 — 오판 96개 시점 추가, 데이터 35,966게임 (f4d048e)
- 14:53 — BC 1 epoch 단계·1시간 체크포인트, BC 준비 2차 (3181bc1); 본학습은 승인 대기
- 16:00 — BC 준비 3차: `bc-profile` 프로파일, 결과 동일 최적화로 update 약 18% 단축, Burn autotune·fusion은 제안만, Ctrl+C 시험 ([m6_bc_preflight](report/review/2026_10_18_11_38_m6_bc_preflight.md) 3차, 7d395e5)
- 18:38 — 발견: 시간 축 코어 비교([AMB-032-4](spec/README.md))가 절차·AC 없이 남아 있음 → 결정: BC 본학습 전 LSTM·GTrXL을 S1 5,300 update씩 비교([032-R17](spec/2026_10_15_22_24_032-model-architecture.md)·AC7, AMB-032-6, [m6_core_ablation](report/experiment/2026_10_18_18_43_m6_core_ablation.md))
- 18:49 — 학습 증강 4종·start_loc 변환([030-R4](spec/2026_10_15_22_24_030-observation.md)), bc250 BC f32 예외 확대([032-R8](spec/2026_10_15_22_24_032-model-architecture.md), [060-AC2](spec/2026_10_15_22_24_060-infra.md) 닫힘), [041-R6](spec/2026_10_15_22_24_041-imitation-learning.md) 클래스 가중치 미적용 결정 기록 (2d8dc4e)
- 19:01 — 결정: 코어 비교 T0 대전 조합당 50판, 합계 600판 ([032-R17](spec/2026_10_15_22_24_032-model-architecture.md), [m6_core_ablation](report/experiment/2026_10_18_18_43_m6_core_ablation.md))
- 19:05 — 코어 비교 시작 전 남은 확인 5가지 정리(val 편중, GTrXL BC 미검증, 장비 혼합, 맞대결 경로 미시험, seed 1개) ([m6_core_ablation](report/experiment/2026_10_18_18_43_m6_core_ablation.md) 3절)
- 20:22 — GTrXL 실제 BC 설정 확인: 학습 PC·BC250 1호기 모두 OOM·TDR 없음 ([m6_core_ablation](report/experiment/2026_10_18_18_43_m6_core_ablation.md) 1.5절)
- 20:50 — 발견: BC 본학습 전 전수 리뷰(리뷰어 4명, e190799) — debug 테스트 11개 실패, 기존 데이터 영향 3건(투혼 CP949 제목, Train 연타 병합, value result 0), S2 APM 기준 모순, 맵 사전계산 누락 무음 ([m6_pre_bc_review](report/review/2026_10_18_20_51_m6_pre_bc_review.md))
- 21:00 — 감시 재시작 `supervise`([060-R6](spec/2026_10_15_22_24_060-infra.md)), run 큐 작업자·양보·장비 기록([061-R13](spec/2026_10_15_22_24_061-shared-storage.md)~R15) 구현, 두 PC 실기 시험(시험 run 삭제) ([m6_bc_preflight](report/review/2026_10_18_11_38_m6_bc_preflight.md) 4차)
- 21:05 — GTrXL 대 LSTM 맞대결 스모크 3판 통과, 요약표·대전 승리 집계 버그 수정, 코어 비교 리플레이 전부 저장 (695cbfe, [m6_core_ablation](report/experiment/2026_10_18_18_43_m6_core_ablation.md))

## 2026-10-19
- 01시 — BC 본학습 전 통합 체크리스트([m6_pre_bc_review](report/review/2026_10_18_20_51_m6_pre_bc_review.md) F·G절), 정규화 상수 spec 기록([030 §6](spec/2026_10_15_22_24_030-observation.md)), CLAUDE.md 갱신, 장비 호스트 이름 중립 표기([001-R10](spec/2026_10_15_22_24_001-system-architecture.md)), worktree 정리
- 01시 — 발견: 5661ff8·695cbfe 추가 리뷰 — worker의 `init_run` DONE 미확인, 공유 I/O 오류 시 잠금 오판, mtime 시계 가정, Ctrl+C 이중 SIGINT; 테스트 196 통과 / 11 실패 ([m6_pre_bc_review](report/review/2026_10_18_20_51_m6_pre_bc_review.md) H절)
- 01시 — 결정: git 운영은 worktree → rebase → origin master push 유지, push 뒤 메인 사본도 pull --ff-only; 기존 "사용자 결정" 표기 바로잡음 (CLAUDE.md git 절)
- 시각 없음 — 결정: CLAUDE.md는 짧게(진행 상황·날짜·경위 없이 링크만), 명령 상세는 [commands](guide/commands.md)로 이동
- 시각 없음 — 결정: BC 전 리뷰 결정 G-D1~D9 등 확정(전체 재build, Train 연타 병합 제거, 판정 불가 value 가중치 0, 전역 norm clip 1.0, 장비 표기 OS·백엔드만 등), spec 일괄 개정 (005e6fe, [m6_pre_bc_review](report/review/2026_10_18_20_51_m6_pre_bc_review.md))
- 시각 없음 — 결정: T1 리뷰 후 추가 결정 D-a(032-AC8 허용치는 재측정 후)·D-b([040-R8](spec/2026_10_15_22_24_040-replay-dataset.md) 기록 승자 사용)·D-c(Windows 자동 시작 없음)·D-d(수동 `--take-over` 즉시 인수), 리뷰 지적 spec 보완 ([032-AC8](spec/2026_10_15_22_24_032-model-architecture.md))
- 시각 없음 — 측정 결과: [032-AC8](spec/2026_10_15_22_24_032-model-architecture.md) CUDA backward 재측정(6회) 최대 4.94e-3 → 1e-2 제안, CUDA 결과가 회마다 다름 ([m6_bc_preflight](report/review/2026_10_18_11_38_m6_bc_preflight.md) 5차)
- 시각 없음 — 결정·구현: E-8 `GameEnd.final_frame`([010-R12](spec/2026_10_15_22_24_010-game-protocol.md), PROTOCOL_VERSION 2)으로 마지막 dense 보상 합산, CUDA AC8 1e-2 확정([032-AC8](spec/2026_10_15_22_24_032-model-architecture.md)), final_frame 수신 확인 ([m6_bc_preflight](report/review/2026_10_18_11_38_m6_bc_preflight.md) 5차)
- 13시 — 재빌드(G-M1, R-1): 학습 PC·Linux 호스트 A·B 6523286(proto v2), `.env` 키 2개 추가; 공용 드라이브 82%로 중단 → 사용자 정리 후 64% ([m6_rebuild](report/review/2026_10_19_14_28_m6_rebuild.md))
- 14시 — 표본 end-to-end(R-2): 리플레이 37개 정상, 장비 간 shard 바이트 동일, BC 스모크 CUDA 약 10.6 s/update·Vulkan 약 86 s/update, final_frame 확인, 표본 폴더 삭제 ([m6_rebuild](report/review/2026_10_19_14_28_m6_rebuild.md))
- 14시 — 발견: 표본 scr_ladder 무효 레이블 40.3%(APM 2,376·957 두 시점), 제외 결정 필요; 에이전트 장비에 대상 맵 v2 사전계산 없음 ([m6_rebuild](report/review/2026_10_19_14_28_m6_rebuild.md))
- 16시 — 결정·구현: [040-R19](spec/2026_10_15_22_24_040-replay-dataset.md) 무효 레이블 ≥ 20% 테란 시점은 `filtered`(사용자 결정, [AMB-040-11](spec/README.md)), 레이블 검사 함수 통합, 표본 재build로 두 시점 제외; 출력 학습 PC 로컬 `sc_data/t3c` ([m6_rebuild](report/review/2026_10_19_14_28_m6_rebuild.md))
- 16시 — 재build 준비(R-3): 세 장비 d452c29 재빌드, 옛 shards·records·캐시 삭제(학습 PC 39.6GB·Linux 호스트 A 39.3GB 확보), `jobs.txt` 작업 77개 재생성 ([m6_rebuild](report/review/2026_10_19_14_28_m6_rebuild.md))
- 16시 — 래더 수집 시작([043](spec/2026_10_16_22_48_043-ladder-replay-collector.md), `kr_asia.toml`, 학습 PC): 추적 대상 gw30 3,185명·gw45 462명; dry-run 멈춤은 출력 파이프 버퍼링 탓
- 16:35 — 장시간 실행 시작: 전체 재build(R-4), 세 장비 d452c29(학습 PC는 16:59 합류), 첫 파트 정상·쓰기 검증 실패 0, 완료 예상 10-20 오전 ([m6_rebuild](report/review/2026_10_19_14_28_m6_rebuild.md))
- 18시 — 래더 수집 종료: 일일 상한 도달, ok 1000·failed 37(score 누락 34), 누적 ok 3000; 로그 작업 tmp `ladder_collect_20261019.log`
- 18시 — 결정·구현([043-R9](spec/2026_10_16_22_48_043-ladder-replay-collector.md) 예외, 사용자 결정): `score` 없는 경기도 수집(MMR·랭크 비움), `collect --retry-failed` 추가, score 실패 81경기는 10-20 재시도
- 18:31 — R-4 전체 재build에 Linux 추출 호스트 C 합류(jobs 8), 합류 전 표본 shard·records 바이트 동일 확인
- 시각 없음 — T7 미검증 AC: [010-AC4](spec/2026_10_15_22_24_010-game-protocol.md)·[043-AC4](spec/2026_10_16_22_48_043-ladder-replay-collector.md) 테스트, [031-AC1](spec/2026_10_15_22_24_031-action-space.md) 실게임 100/100, [013-R7](spec/2026_10_15_22_24_013-version-mapping.md) 확인; [060-AC4](spec/2026_10_15_22_24_060-infra.md)·[032-AC2](spec/2026_10_15_22_24_032-model-architecture.md)는 재build 뒤 ([m6_unverified_ac](report/review/2026_10_19_17_37_m6_unverified_ac.md))
- 시각 없음 — 결정·구현: [034-R8](spec/2026_10_15_22_24_034-action-table.md) 애드온 건설 중 건물은 후보 제외, 034-R9 선택 유닛 대상 금지(SetRally_Unit 예외 확인 필요), 재build 없음, 시나리오 재실행 100/100
- 시각 없음 — 결정: [034-R9](spec/2026_10_15_22_24_034-action-table.md) SetRally_Unit 자기 대상 예외 사용자 확정(`self_target = true`)
- 20시 — 새 리포트: 액션 전수 감사 — High 2·Medium 6, 재build 중단 불필요, 커버리지 99.79% ([m6_action_audit](report/investigation/2026_10_19_20_03_m6_action_audit.md))
- 23시 — 구현(T9, 브랜치 `t9-unload-unit`): `Unload_Unit`(`PROTOCOL_VERSION` 3), `unknown` 레이블, shard schema 6 `labeler_version`; 표본 무효 레이블 7.015% → 6.878% ([m6_action_audit](report/investigation/2026_10_19_20_03_m6_action_audit.md) §10)
- 시각 없음 — 결정: `unknown` 레이블은 [040-R19](spec/2026_10_15_22_24_040-replay-dataset.md) 무효 비율·040-AC7 집계에서 제외(사용자 확정) ([m6_action_audit](report/investigation/2026_10_19_20_03_m6_action_audit.md) §10)
- 시각 없음 — T9 리뷰 반영(`is_transport_loaded` 재사용, `target_not_in_transport` 검사), 031-AC1 실게임 보류; 표본 산출물 `<URANIUM_SHARED_DIR>/artifacts/m6_action_audit/t9_sample/` ([031-AC1](spec/2026_10_15_22_24_031-action-space.md))
- 23시 — 결정·구현(사용자 결정, 액션 감사 T8): 대상 자격에서 생산 중·탑승 유닛 제외([034-R10](spec/2026_10_15_22_24_034-action-table.md)), 시야 밖 자원 대상·그룹 명령([011-R20](spec/2026_10_15_22_24_011-bridge-openbw.md)), 마스크 보강, feature 스키마 6; [031-AC1](spec/2026_10_15_22_24_031-action-space.md) 실게임 100/100 ([m6_action_audit](report/investigation/2026_10_19_20_03_m6_action_audit.md) §11)
- 시각 없음 — T8 리뷰 반영: 미완성 알·코쿤·아콘 대상 허용, U-4 사용자 결정(시즈 전환 연타 레이블 BC 유효), [011-AC4](spec/2026_10_15_22_24_011-bridge-openbw.md) 그룹 명령 결정론 통과; 산출물 `<URANIUM_SHARED_DIR>/artifacts/m6_action_audit/t8/`

## 2026-10-20
- 00:10 — 장시간 실행 재개: R-4 재build 보류 50파트를 aa187b2(레이블러 2, schema 6)로 네 장비에서 재개(사용자 결정), 완료 예상 10시경 ([m6_rebuild](report/review/2026_10_19_14_28_m6_rebuild.md))
- 00:5x — 게임 호스트 재배포(f513b92, PROTOCOL 3): Linux 호스트 A·B·학습 PC 재빌드, 실게임 101/101 액션·같은 seed 결정론 통과, 게임 오류 0
- 07:56 — 장시간 실행 끝: R-4 전체 재build 77파트 모두 `.done`(레이블러 1 27파트 / 레이블러 2 50파트)
- 08:35~08:58 — scr_ladder 증분 build(학습 PC, f513b92): 10-19 수집분 1,000개 중 written 378·filtered 694, 새 shard `scr_ladder-p0/shard_00043`~`00062`
- 08:5x — 래더 점수 누락 재수집([043-R9](spec/2026_10_16_22_48_043-ladder-replay-collector.md)): 81경기 중 ok 77, 4경기 failed 유지; 로그 작업 tmp `ladder_retry_20261020.log`
- 09시 — 발견: 검증을 통과한 shard 7개가 공용 드라이브에서 0으로 채워지거나 잘림(Linux CIFS 쓰기, 캐시에서 다시 읽어 검증 통과), 유실 132시점 ([m6_rebuild](report/review/2026_10_19_14_28_m6_rebuild.md) R-5)
- 09:28~09:56 — 결정·구현(사용자 결정, [AMB-040-12](spec/README.md)): 다시 읽기 캐시 우회, manifest 전 shard 디코드 검사, zstd 체크섬, 같은 리플레이 파일 중복 제외([040-R20](spec/2026_10_15_22_24_040-replay-dataset.md)) (67488fc·1ce972c·7b0879e); Linux 마운트는 그대로
- 10시 — 래더 일반 수집 종료: ok 923·failed 4, 일일 상한 1,000 도달, 누적 ok 4,000; 로그 작업 tmp `ladder_collect_20261020.log`
- 10:08~10:35 — scr_ladder 증분 build(Linux 호스트 B, f513b92): written 285·filtered 738, 새 shard `scr_ladder-p0/shard_00063`~`00080`; CIFS 수정 전 바이너리라 디코드 검사는 별도
- 10:2x~10:36 — R-5 수리(7b0879e): 손상 shard 7개 삭제·132시점 재build, personal 전체 재build; 원문·옛 결과 `build/r5-backup/`
- 11시~12시 — 재build 검증 통과: 네 출처 manifest·`--verify`, 위반 0, 36,445시점; 투혼 10,476게임·서킷브레이커 725게임 ([dataset_stats](report/survey/2026_10_20_12_52_dataset_stats.md), [040-AC2](spec/2026_10_15_22_24_040-replay-dataset.md))
- 12시 — 결정 반영([041-R14](spec/2026_10_15_22_24_041-imitation-learning.md)): `bc-epoch`로 `[steps]` 갱신 — s1 38,584, s2·ablation_a 11,319, warmup 566
- 12:2x~12:40 — BC 스모크(새 데이터): CUDA 약 10초/update, Vulkan 약 101초, loss 유한; `--dataset`·`--shards` 첫 배치 순서 차이 확인
- 12:40 — 고정 build 바이너리·worktree 정리, 학습 PC에 build 맵 사전계산 1,244개 복사·대상 맵 mapgen
- 시각 없음 — 결정(U-5, [040-R19](spec/2026_10_15_22_24_040-replay-dataset.md)): 무효 레이블 20% 기준 유지, scr_ladder 보조 필터 없음; 걸러진 시점 216 / 약 36.6k
- 시각 없음 — 발견·수정([041-R13](spec/2026_10_15_22_24_041-imitation-learning.md), [061-R10](spec/2026_10_15_22_24_061-shared-storage.md)): `--shards` 첫 배치 차이는 폴더 순서 연결 탓 → `<출처>/<파트>/<파일>` 키 정렬(사용자 결정), `--dataset` 순서·지문 불변
- 시각 없음 — BC 전 체크리스트 갱신·코어 비교 선행 조건 점검(결정 없음): 남은 것 G-M2·G-M5·[060-AC4](spec/2026_10_15_22_24_060-infra.md)/[032-AC2](spec/2026_10_15_22_24_032-model-architecture.md) ([m6_pre_bc_review](report/review/2026_10_18_20_51_m6_pre_bc_review.md) G절, [m6_core_ablation](report/experiment/2026_10_18_18_43_m6_core_ablation.md) 5절)
- 14시 — 측정 결과([060-AC4](spec/2026_10_15_22_24_060-infra.md), [032-AC2](spec/2026_10_15_22_24_032-model-architecture.md)): 학습 PC CUDA 벤치 통과, BC 중앙값 11.1초/update, 추론 p95 51/57ms 통과; Linux 호스트 A Vulkan 179/187ms는 사용자 결정 대기 ([bench_cuda](report/bench/2026_10_20_14_44_bench_cuda.md), [m6_unverified_ac](report/review/2026_10_19_17_37_m6_unverified_ac.md))
- 시각 없음 — 결정([032-AC2](spec/2026_10_15_22_24_032-model-architecture.md), [AMB-032-8](spec/README.md)): 학습 PC 유휴 시 주 장비, 아니면 Linux 호스트 A; AC2 판정은 CUDA만, 초안 40ms 폐기, 주 장비 규칙 [060-R8](spec/2026_10_15_22_24_060-infra.md) 신설 ([m6_unverified_ac](report/review/2026_10_19_17_37_m6_unverified_ac.md))
- 15시 — 확인([032-R17](spec/2026_10_15_22_24_032-model-architecture.md), [020-R11](spec/2026_10_15_22_24_020-env-orchestration.md)): 코어 비교 대전 경로 스모크 통과(맞대결 2판·T0 3판, 오류 0); 출력 `runs/eval-smoke-2026-10-20/` ([m6_core_ablation](report/experiment/2026_10_18_18_43_m6_core_ablation.md) 5절)
- 15:21~15:51 — 장시간 실행 시작: 코어 비교 학습([032-R17](spec/2026_10_15_22_24_032-model-architecture.md), 사용자 지시, 2639038, 큐 `<shared>/queue/core-ablation.toml`) — `core-abl-lstm` 학습 PC CUDA, `core-abl-gtrxl` Linux 호스트 A Vulkan ([m6_core_ablation](report/experiment/2026_10_18_18_43_m6_core_ablation.md) 5절)
- 16:43 — 장시간 실행 중단: GTrXL run 커널 OOM(step 31, `ckpt-2` 남음), 원인은 BC batch 동시 상주와 GPU 공유 RAM ([m6_core_ablation](report/experiment/2026_10_18_18_43_m6_core_ablation.md) 5절)
- 18시 — 결정·구현([041-R16](spec/2026_10_15_22_24_041-imitation-learning.md), 사용자 결정): 호스트별 미리 읽기 깊이 `URANIUM_BC_PREFETCH`(c971b1f), Linux 호스트 A만 0
- 18:20~19:34 — 측정 결과(G-M2): 깊이 0으로 45 step 완료·OOM 지점 통과, 스왑 최대 5.7 GB; 19:36 GTrXL run 재개; 출력 Linux 호스트 A `~/diag/g-m2/`
- 20:52 — 장시간 실행 중단: GTrXL run GPU 버퍼 할당 실패(2 GiB, step 48 재실패), 원인 엔티티 인코더 forward 약 10.5 GB ([m6_core_ablation](report/experiment/2026_10_18_18_43_m6_core_ablation.md) 5절)
- 23시 — 결정·구현([041-R17](spec/2026_10_15_22_24_041-imitation-learning.md)·R18, 사용자 결정): `URANIUM_BC_ENTITY_CHUNK`·`URANIUM_BC_PACK_BATCH` 추가, step 48 통과·GTT 11.3 → 4.2 GB; 출력 Linux 호스트 A `~/diag/r17/`

## 2026-10-21
- 15:52~16:07 — 성향 확인 대전(사용자 요청): `core-abl-lstm` ckpt-5300(확률적) vs T1 3종 9판 — 7판 전멸패, 2판 시간 제한 무승부; 출력 `runs/lstm-vs-t1-2026-10-21/`
- 16:1x~17:2x — 발견: LSTM ckpt-5300 argmax는 T1 9판 전패, 입력 버그 아님(상황 판단 미학습으로 봄), argmax는 랠리 지정 반복 고리 ([m6_core_ablation](report/experiment/2026_10_18_18_43_m6_core_ablation.md) 5절)
- 시각 없음 — 결정([AMB-031-6](spec/README.md), 사용자): GG 항복 — win_open([032-R18](spec/2026_10_15_22_24_032-model-architecture.md)) P ≤ 5% 30초 연속·10분 이후 + 규칙 조건, SC:R 실전만 On·OpenBW는 Shadow, 채팅 `surrender_chat` ([031-R12](spec/2026_10_15_22_24_031-action-space.md))
- 시각 없음 — 구현·확인([031-R12](spec/2026_10_15_22_24_031-action-space.md), 53a8a01~4fdaf23): proto v4, shim 채팅 → 퇴장, BWAPI 패치 0014, 체크포인트 누락 키 기본값 처리; 테스트 324 통과·통합 3판 정상; 출력 Linux 호스트 C `~/gg-out/`
- 18:26~19:10 — scr_ladder 증분 build(Linux 호스트 B, 2639038): 10-21 수집분 1,069개 중 written 332, 새 shard `00083`~`00107`, manifest 미갱신; 로그 `build/logs/bc250_2/scr_ladder_scr_ladder-p0.inc_20261021.log`

## 2026-10-22
- 02:22~02:44 — 발견·조치([061-R12](spec/2026_10_15_22_24_061-shared-storage.md)): 학습 PC가 `core-abl-gtrxl` 인수 뒤 멈춤 — manifest 밖 증분 shard 25개를 로컬 모드가 읽음; 사용자 결정으로 `<shared>/staging/scr_ladder-inc-20261021/`로 옮기고 `ckpt-1320`부터 재개
- 시각 없음 — M7 착수·결정([AMB-012-2](spec/README.md), 사용자): SC:R 대상 빌드 1.23.10.13515 고정, 실전 PC는 CUDA 학습 호스트와 같은 PC, 스파이크는 이 PC에서
- 시각 없음 — 새 리포트: M7 스파이크 배경·samase 문헌 조사 — 로더 비공개, 게임 종료·나가기 API 없음 ([m7_scr_spike](report/milestone/2026_10_22_18_19_m7_scr_spike.md))
- 시각 없음 — 결정([012](spec/2026_10_15_22_24_012-bridge-remastered.md), 사용자 확인): SC:R 클라이언트 32비트(x86)도 허용, x86 exe 해시 추가, 스파이크는 x64부터
- 17:00~17:14 — 네 장비 6fa8fbd 재빌드(proto v4, BWAPI 0014), 코어 비교 학습 두 run `ckpt-5300` 완료 ([m6_core_ablation](report/experiment/2026_10_18_18_43_m6_core_ablation.md) 4·5절)
- 17:40~21:21 — 코어 비교 대전 2회 실패(루프백 주소 바인딩, 실제 시간 상한 1,800초); 발견: 브리지 시작 실패 판을 `Finished`로 기록; 출력 `<shared>/artifacts/m6_core_ablation/core-ablation-eval-20261022-abort{1,2-timeout}/` ([m6_core_ablation](report/experiment/2026_10_18_18_43_m6_core_ablation.md) 5절)
- 21:23~21:35 — 스파이크 결과([012-AC1](spec/2026_10_15_22_24_012-bridge-remastered.md)): samase 0.10.28-64로 SC:R x64 실행, AC1 목록 모두 확인·SCV 이동 송신 성공, [AMB-012-1](spec/README.md)은 대안 불필요로 닫음; 로그 학습 PC 스파이크 폴더 `logs/` ([m7_scr_spike](report/milestone/2026_10_22_18_19_m7_scr_spike.md))
- 시각 없음 — 정정: 위 스파이크 시각은 21:23~21:50, 로그 이름은 `logs/probe-x64-20261022-run1.log`
- 21:54~22:10 — 스파이크 2·3차(LAN): 명령 지연 2프레임, 0x5C 채팅 가능, `0x57` 나가기로는 클라이언트가 빠지지 않음 → 나가는 방법은 사용자 결정 필요 ([m7_scr_spike](report/milestone/2026_10_22_18_19_m7_scr_spike.md))
- 시각 없음 — 새 리포트: OpenBW 브리지 필드 → SC:R 원시 메모리 대응표 ([m7_scr_field_mapping](report/milestone/2026_10_22_22_11_m7_scr_field_mapping.md))
- 시각 없음 — 발견([010-R5](spec/2026_10_15_22_24_010-game-protocol.md)): OpenBW shim의 CompleteMapInformation으로 시야 밖 탄환이 hazard 채널에 들어감(BC shard·코어 비교 포함), 처리는 사용자 결정 대기
- 시각 없음 — 결정·구현([AMB-030-4](spec/README.md), 사용자): 탄환 등 효과는 중심 타일이 보일 때만 관측, feature 추출·shim에 반영(재build 없음, BC 본학습부터) ([030 §3](spec/2026_10_15_22_24_030-observation.md))
- 시각 없음 — 결정([AMB-012-4](spec/README.md), 사용자): SC:R 항복은 "gg" 채팅 뒤 `GameEnd(Surrender)`·연결 종료, 게임에서 나가지는 않음 ([012-R11](spec/2026_10_15_22_24_012-bridge-remastered.md))
- 시각 없음 — 결정·구현([AMB-030-5](spec/README.md), 사용자): 보이는 적 유닛 내부 정보(생산 큐·진행도·핵 장전)는 0으로 숨김(재build 없음, BC 본학습부터) ([030 §2](spec/2026_10_15_22_24_030-observation.md))
- 시각 없음 — 결정(사용자): SC:R 지연은 보정 없이 기록만·명령 유효 기한 관측 + 24프레임([AMB-012-5](spec/README.md)), `uranium-adapter` 안 만듦(AMB-001-5), SC:R `CommandFailed`는 Deferred(AMB-012-6)
- 시각 없음 — spec 012·013 M7 개정안(Review) 작성·승인 요청: 스파이크 실측·결정·관측 시점·종료 판정·AC 재구성, `uranium-adapter` 삭제 ([012](spec/2026_10_15_22_24_012-bridge-remastered.md), [013](spec/2026_10_15_22_24_013-version-mapping.md))
- 시각 없음 — 승인(사용자): spec 012·[013](spec/2026_10_15_22_24_013-version-mapping.md) M7 개정안 Approved([012-R15](spec/2026_10_15_22_24_012-bridge-remastered.md) 시간 제한 없음 포함), 다음은 구현 계획
- 시각 없음 — 구현(M7 1단계, [012-R18](spec/2026_10_15_22_24_012-bridge-remastered.md)): 에이전트 Realtime 모드(`valid_until_frame`, `last_failed` 0, privileged 미전달), 새 명령 `uranium-agent scr`(루프백 7700)
- 시각 없음 — 구현(M7 2단계, [013-R1](spec/2026_10_15_22_24_013-version-mapping.md)): `bridge/scr` samase 플러그인 — 관측·이벤트·종료·명령·I/O 스레드, 표 `bridge/scr/tables`, 테스트 13개 통과, 실게임 검증 전 ([012](spec/2026_10_15_22_24_012-bridge-remastered.md))
- 시각 없음 — 구현(M7 3단계): `tools/scr/build.ps1`·`run.ps1`, `.env.example` SC:R 절, [scr](guide/scr.md)·[commands](guide/commands.md) 갱신
- 23:37~10-23 00:2x — SC:R 실게임 첫 검증: 싱글 커스텀 3판 — LSTM 1판 14.6분 패배까지 완주(시야 위반·거부 0), GTrXL 2판은 사람이 중단; 발견: 로비 Random 상대 실제 종족 노출([030-R11](spec/2026_10_15_22_24_030-observation.md) 위반); 로그 `runs/scr/` ([m7_scr_bridge](report/milestone/2026_10_23_00_16_m7_scr_bridge.md))

## 2026-10-23
- 시각 없음 — M7 1차 세션 정리: 실게임 로그를 학습 PC 메인 작업 사본 `runs/scr/`로, samase·스파이크 로그는 저장소 밖 스파이크 폴더, `.env`에 `URANIUM_SCR_SAMASE`; 다음 할 일 [m7_scr_bridge](report/milestone/2026_10_23_00_16_m7_scr_bridge.md) 4절
- 시각 없음 — 결정(사용자): 코어 비교 대전·채점은 학습 PC 포함해 학습 PC에서, `wall_timeout_s = 7200`, 빌드 6fa8fbd 고정; 실행 스크립트 `<shared>/artifacts/m6_core_ablation/final/run_final.ps1`
- 00:48~13:26 — 장시간 실행 시작·끝: 코어 비교 최종 — 맞대결 100판 GTrXL 42승·LSTM 13승·무 45, T0 승률 LSTM 2.7%(300판)·GTrXL 약 10%(274판), val top-1 GTrXL 67.9% vs 66.8%; 출력 `<shared>/artifacts/m6_core_ablation/final/` ([m6_core_ablation](report/experiment/2026_10_18_18_43_m6_core_ablation.md) 4절)
- 시각 없음 — 발견: `bc-eval`에 엔티티 조각 재계산([041-R17](spec/2026_10_15_22_24_041-imitation-learning.md))이 없어 Linux Vulkan에서 2 GiB 할당 실패; staging shard 25개 복귀·manifest 재생성(106 shard, `--verify` 통과) ([m6_core_ablation](report/experiment/2026_10_18_18_43_m6_core_ablation.md) 5절)
- 시각 없음 — 결정(사용자, [032-R17](spec/2026_10_15_22_24_032-model-architecture.md) ④·[AMB-032-6](spec/README.md)): 시간 축 코어 GTrXL 채택, BC 본학습·RL은 GTrXL ([m6_core_ablation](report/experiment/2026_10_18_18_43_m6_core_ablation.md) 4절)
- 시각 없음 — BC 본학습 착수 전 문서 정리: [m6_pre_bc_review](report/review/2026_10_18_20_51_m6_pre_bc_review.md) G절 갱신·I절 신설(데이터 동결 시점, `--core` 기본값 결정 대기), 루트 README·CLAUDE.md·[commands](guide/commands.md) 갱신
- 시각 없음 — 결정(사용자, I-D1·I-D2): BC 본학습 데이터 2,069 파일로 동결, `--core` 기본값 `gtrxl`; 발견: 큐에 S1·S2·(A)를 다 넣으면 S1 교대가 안 일어남 → I-D3 신설 ([m6_pre_bc_review](report/review/2026_10_18_20_51_m6_pre_bc_review.md) I절)
- 시각 없음 — 결정(사용자, I-D3): 본학습 큐에 S1·S2·(A) 함께, 학습 PC가 S1 → S2·Linux 호스트가 (A), 작업자는 학습 PC부터 ([m6_pre_bc_review](report/review/2026_10_18_20_51_m6_pre_bc_review.md) I절)
- 시각 없음 — BC 본학습 준비(I-1~I-3): 2d6be1a 재빌드, `[steps]` 재계산(s1 39,096 / s2 11,387), 표본 end-to-end 통과(산출물 `<shared>/checkpoints/sample-bc-*`), CubeCL autotune panic은 무해 ([m6_pre_bc_review](report/review/2026_10_18_20_51_m6_pre_bc_review.md) I절)
- 15:50~ — 장시간 실행 시작: BC 본학습([041](spec/2026_10_15_22_24_041-imitation-learning.md), GTrXL, 2d6be1a, 2,069 파일), 큐 `<shared>/queue/bc.toml` — 학습 PC `bc-s1` → `bc-s2`, Linux 호스트 `bc-ablation-a`; 체크포인트 `<shared>/checkpoints/bc-*`
- 16:22 / 17:39 — 장시간 실행 중지(사용자 지시): BC 본학습 두 작업자 정상 종료, `bc-s1` ckpt-132·`bc-ablation-a` ckpt-44
- 17:30~18:0x — 발견(실측): 같은 모델의 PyTorch 이식이 학습 PC GPU에서 Burn보다 학습 2~2.6배(그대로 이식)~최대 4.3배(bf16 AMP) 빠름; 원본 `local/raw/torch_vs_burn_20261023/` ([torch_vs_burn](report/bench/2026_10_23_18_30_torch_vs_burn.md))
- 21:49~22:20 — 발견(실측): bc250_1 ROCm PyTorch는 Burn Vulkan 대비 학습 fast 3~4배·faithful 1.2~1.5배, bf16 AMP 역효과, MIOpen LSTM backward 실패로 우회; 원본 `local/raw/torch_vs_burn_20261023/bc250_1/` ([torch_vs_burn](report/bench/2026_10_23_18_30_torch_vs_burn.md))
- 시각 없음 — 결정(사용자, [AMB-000-4](spec/README.md)·AMB-032-9·AMB-032-10): 학습·추론을 Burn에서 Python + PyTorch로 전환(나머지는 Rust, PyO3 `uranium._core`), Burn 체크포인트는 보관·새로 학습, CUDA bf16·ROCm f32; spec 개정([032-R19](spec/2026_10_15_22_24_032-model-architecture.md)·AC10 신설)

## 2026-10-24
- 01:3x — 발견([032-AC10](spec/2026_10_15_22_24_032-model-architecture.md), [AMB-000-4](spec/README.md) 1-3): PyTorch UraniumNet이 Burn과 같은 입력에서 출력·손실 1e-5·gradient 1e-4·AdamW 한 step까지 일치; 기준값 `tests/fixtures/burn_parity/`, full은 bc250_1 `~/parity/full-{lstm,gtrxl}`
- 02:0x — 발견([AMB-000-4](spec/README.md) 1-4, [041-R15](spec/2026_10_15_22_24_041-imitation-learning.md)): bc250_1 실데이터 update 297~630초로 Burn보다 느림 — ROCm에 켠 cuDNN benchmark 탓 → CUDA만 켜도록 수정 ([pytorch_port](report/bench/2026_10_24_02_42_pytorch_port.md) 3.1~3.2)
- 03:0x — 발견(사용자 요청 브리지 오버헤드 실측): 수정 뒤 update 약 22초, collate를 작업 스레드로 옮겨 약 20초(Burn 대비 약 5.4배), 브리지 비중 약 2% ([pytorch_port](report/bench/2026_10_24_02_42_pytorch_port.md) 3.3~4)
- 03:05 — 장시간 실행 시작([AMB-000-4](spec/README.md) 1-6 BC 동등성): bc250_1에서 Burn·PyTorch 같은 초기값·배치로 각 500 update, 5 update 손실 차이 1.9e-6; 산출물 bc250_1 `~/parity_bc/` ([pytorch_port](report/bench/2026_10_24_02_42_pytorch_port.md) 6절)
- 03:2x — 발견([AMB-000-4](spec/README.md) 1-7): 추론을 Python으로 이전(Rust Broker + Python 모델), random-init LSTM 대 T0 1판 정상(결정 지연 p50 22ms), Windows SC:R 확인은 남음 ([pytorch_port](report/bench/2026_10_24_02_42_pytorch_port.md) 7절)
- 03:5x — 발견([AMB-000-4](spec/README.md) 1-8): RL을 Python 학습기 + Rust 실행 루프(`uranium-league`)로 이전, Burn RL 경로 삭제, CPU 스모크 정상(누적 312 update) ([pytorch_port](report/bench/2026_10_24_02_42_pytorch_port.md) 8절, [042-R7](spec/2026_10_15_22_24_042-rl-training.md))
- 04:1x — 측정 결과([032-R8](spec/2026_10_15_22_24_032-model-architecture.md), 032-AC1·AC8): 학습 PC CUDA f32 gradient 오차 ≤ 3e-4, TF32 최대 0.23, bf16 중앙 1.5%·최대 약 20% ([pytorch_port](report/bench/2026_10_24_02_42_pytorch_port.md) 9절)
- 04:2x — 장시간 실행 끝·발견([AMB-000-4](spec/README.md) 1-6, 사용자 지시로 100 update에서 끊음): Burn 대 PyTorch ROCm f32 손실 차이 2.3e-4·CUDA bf16 3.4e-3, 이식 정상; update당 ROCm 3.4초·CUDA 1.5초·Burn 11초 ([pytorch_port](report/bench/2026_10_24_02_42_pytorch_port.md) 10절)
- 05:0x — 발견·정리([AMB-000-4](spec/README.md) 1-9): Burn 삭제(태그 `burn-final` = fbb5026), `docs/windows.md` 삭제(내용은 [commands](guide/commands.md)), CUDA BC update 약 11초 — Python·launch 부담이 병목 ([pytorch_port](report/bench/2026_10_24_02_42_pytorch_port.md) 11절)
- 05:2x — 장시간 실행 시작(2단계 2-0 기준선): bc250_1에서 `bench suite`·`bench bc --groups 1,2,4,8`·`pytest -m gpu` 3회; 정정: Burn 정리 항목(05:4x로 적힘)의 실제 시각은 05:0x
- 05:4x — 정리(사용자 지시): Burn 시절 산출물을 `<shared>/_old_burn/`으로 이동(지우지 않음; 공용 `checkpoints/`·`runs/`·`queue/`·`workers/`, 코어 비교 산출물, 학습 PC 기록 → `_old_burn/project/`), Burn 빌드·autotune 캐시 삭제, Linux 호스트 세 대는 미정리
- 06:5x — 장시간 실행 끝·발견(2-0 기준선): bc250_1 그룹 1 update 약 19.5초, 그룹 2·4는 MIOpen 첫 호출로 21~397초(그룹 4에서 중단), ROCm 과녁은 conv backward 전반 ([pytorch_opt](report/bench/2026_10_24_07_03_pytorch_opt.md) 1절, [bench_rocm](report/bench/2026_10_24_07_03_bench_rocm.md))
- 07:0x — 발견: bc250 위치 헤드 1×1 conv backward가 MIOpen에서 60ms, 같은 수식 곱-합은 3.6ms(17배) ([pytorch_opt](report/bench/2026_10_24_07_03_pytorch_opt.md) 2절)
- 08:5x — 발견·결정: bc250 update 23.5 → 14.9~17.5초, ROCm 기본값 `exact`·`mulsum`·`table_t` 채택·나머지 탈락; `URANIUM_MALLOC_MMAP_THRESHOLD`로 Feeder 8.2 → 2.4 GB, `.env`는 사용자 확인 대기 ([pytorch_opt](report/bench/2026_10_24_07_03_pytorch_opt.md) 2.3~2.9)
- 09:0x — 결정(사용자): 엔티티 재계산 끔([AMB-041-8](spec/README.md) Resolved), bc250 `.env`에 미리 읽기 2·할당자 임계 반영(원본 `.env.bak-20261024`), 레인 그룹 크기는 AMB-041-7 Deferred 유지; CUDA 측정 허락
- 11:2x — 장시간 실행 중단·발견(CUDA 기준선): 벤치 스위트 완주, 실데이터 BC는 메모리 부족으로 그룹 1만 — 약 3~4.5초/update(Burn의 약 3배), 추론 B1~B16 약 24ms ([pytorch_opt](report/bench/2026_10_24_07_03_pytorch_opt.md) 1.2, [bench_cuda](report/bench/2026_10_20_14_44_bench_cuda.md))
- 12:1x — 발견·결정: 엔티티 attention 융합 커널 레지스터 스필 수정 뒤 bc250 11.74초·CUDA 3.33초, 기본값 `URANIUM_ENTITY_ATTN=fused`·CUDA `URANIUM_LOC_ROWS=pow2`; bc250 누적 23.5 → 11.7초 ([pytorch_opt](report/bench/2026_10_24_07_03_pytorch_opt.md) 2.12)
- 12:5x — 발견·결정: CUDA 데이터 대기 해소 — 레인별 feature 추출 스레드화·f32 직접 해제로 4.73 → 3.61초, 기본값 `URANIUM_BC_FEED_THREADS=2` ([pytorch_opt](report/bench/2026_10_24_07_03_pytorch_opt.md) 2.14)
- 13:3x — 발견·결정: Linear 가중치 gradient split-K bmm으로 bc250 10.43초(ROCm 기본값), 30 update 손실 동등성 2.2e-7; 동등성 확인은 30 update로 줄임(사용자 질문 후) ([pytorch_opt](report/bench/2026_10_24_07_03_pytorch_opt.md) 2.15~2.17)
- 15:2x — 발견·결정: 64채널 ResBlock도 bmm(bc250 누적 23.5 → 10.0초), CUDA torch.compile 재컴파일 제거로 3.24 → 2.08초 채택, 직접 작성 conv 커널은 탈락 ([pytorch_opt](report/bench/2026_10_24_07_03_pytorch_opt.md) 2.19~2.20)
- 16:1x — 발견·결정: 압축 batch 직접 해제로 collate 1.5 → 0.7초, CUDA 공급 스레드 3 → 업데이트 + 대기 2.27초; 추론 CUDA/HIP Graph로 CUDA 1.3~6.3ms·bc250 B1 12ms ([pytorch_opt](report/bench/2026_10_24_07_03_pytorch_opt.md) 2.21~2.22)
- 16:4x — 결정(사용자, [AMB-041-7](spec/README.md) Resolved): 레인 그룹 CUDA 호스트 2·Linux ROCm 호스트 1(`URANIUM_BC_MICRO_LANES`) ([pytorch_opt](report/bench/2026_10_24_07_03_pytorch_opt.md) 2.23)
- 17:1x — 발견·끝(2단계 마무리): 종단 확인 통과, RL 버그 2건 수정(DeviceLock, warm-up); bc250 23.5 → 10.0초·CUDA 4.7 → 2.0초/update; 남은 결정 [060-R3](spec/2026_10_15_22_24_060-infra.md)·bf16 허용치·[AMB-042-10](spec/README.md) ([pytorch_opt](report/bench/2026_10_24_07_03_pytorch_opt.md) 2.24·3절)
- 17:3x — 결정(사용자): [060-R3](spec/2026_10_15_22_24_060-infra.md) ROCm 학습 프로세스 ≤ 12 GiB·추론 서버 ≤ 1 GiB, [032-AC1](spec/2026_10_15_22_24_032-model-architecture.md)·AC8 CUDA bf16 허용치(forward ≤ 0.15, KL ≤ 1e-3, greedy ≥ 95%, gradient ≤ 3%)
- 19:3x — 발견·결정([AMB-042-10](spec/README.md) Resolved, 사용자): RL 추론 분리 안 함, [060-R3](spec/2026_10_15_22_24_060-infra.md) 추론 상한 2 GiB, ROCm 추론 graph 배치 1만; CUDA RL update 5.70 → 4.60초 ([pytorch_opt](report/bench/2026_10_24_07_03_pytorch_opt.md) 2.25)
- 21:0x — 정리(세션 종료 전): 두 장비 `.env`에 `URANIUM_PYTHON`, CUDA 호스트 `.venv` 설치; 다음은 [m6_pre_bc_review](report/review/2026_10_18_20_51_m6_pre_bc_review.md) J절, 측정 원자료 위치는 [pytorch_opt](report/bench/2026_10_24_07_03_pytorch_opt.md) 4절
- 21:1x — 결정(사용자): BC 본학습 전 코어 비교를 PyTorch로 재실시 — 코어마다 S1 10,000 update, 학습 PC·bc250_1 ([032-R17](spec/2026_10_15_22_24_032-model-architecture.md) ⑥, `config/bc_core_ablation_pt.toml`, [m6_core_ablation_pt](report/experiment/2026_10_24_21_29_m6_core_ablation_pt.md))
- 21:2x — 발견(표본): bc250_1 ROCm 6 update — LSTM 약 11.2 s/update·10.8 GB, GTrXL 약 10 s·11.5 GB 정상; 표본 run `core-abl-smoke-{lstm,gtrxl}-pt`
- 21:3x — 장시간 실행 시작(2차 코어 비교, [032-R17](spec/2026_10_15_22_24_032-model-architecture.md) ⑥): bc250_1 작업자가 `core-abl-lstm-pt`(약 11.8 s/update, 큐 `<shared>/queue/core-ablation-pt.toml`, 1ebb654)
- 21:5x — 발견·장시간 실행 시작: 학습 PC `.env` `URANIUM_PYTHON` 따옴표 문제 수정, 학습 PC CUDA 작업자 `core-abl-gtrxl-pt` 시작(약 1.85 s/update)
- 22:0x~23:0x — 발견·수정([041-R18](spec/2026_10_15_22_24_041-imitation-learning.md)): batch 압축 결과 버퍼가 원본 크기로 남아 bc250_1 OOM → 압축 크기로 줄임(update당 +280 MB → 약 44 MB), [060-R3](spec/2026_10_15_22_24_060-infra.md) 근거 메모리는 재측정 ([m6_core_ablation_pt](report/experiment/2026_10_24_21_29_m6_core_ablation_pt.md) 4절)
- 22:5x — 발견·수정([042-R10](spec/2026_10_15_22_24_042-rl-training.md)): Windows에서 작업자 Ctrl+C가 학습 자식에 안 닿음 → 시작 시 Ctrl+C 무시 속성 해제; 학습 PC 작업자 `ckpt-991` 직후 정지
- 시각 없음 — 결정(사용자): 코드 수정 뒤 master 반영 전 별개 리뷰어 에이전트로 spec 적합성·코드 리뷰 (CLAUDE.md 코딩 규칙)
- 23:5x — 실수·복구([032-R17](spec/2026_10_15_22_24_032-model-architecture.md) ⑥): 측정용 `--steps`로 `core-abl-gtrxl-pt`에 `DONE`이 생김 → `DONE.mistaken-20261024`로 이름 변경

## 2026-10-25
- 00:2x — 조치: bc250_1 BC 메모리 증가(update당 약 25 MB)로 작업자에 `MALLOC_ARENA_MAX=2`를 넣어 재시작
- 00:4x — 발견·결정([032-R19](spec/2026_10_15_22_24_032-model-architecture.md)): CUDA 호스트 데이터 대기 원인은 Windows 힙 페이지 폴트 → 확장 할당자 mimalloc(2.75 → 2.05 s/update, pinned까지 1.94 s) ([pytorch_opt](report/bench/2026_10_24_07_03_pytorch_opt.md) 5절)
- 01:0x — 발견·결정([060-R3](spec/2026_10_15_22_24_060-infra.md)): bc250_1 메모리 증가는 glibc arena 단편화 → `URANIUM_MALLOC_ARENA_MAX` 추가(Linux ROCm 2), 상주 약 6.6 GB에서 평평 ([m6_core_ablation_pt](report/experiment/2026_10_24_21_29_m6_core_ablation_pt.md) 4절)
- 01:2x — 결정([032-R20](spec/2026_10_15_22_24_032-model-architecture.md), [AMB-032-11](spec/README.md)): 구조 변형 비교 — 단일 변형 4개 + 모두 켬, 각 10,000 update, val 500 + 리그전 21쌍 × 100판 ([m6_arch_ablation_pt](report/experiment/2026_10_25_01_38_m6_arch_ablation_pt.md))
- 02:2x — 장시간 실행 예약([032-R20](spec/2026_10_15_22_24_032-model-architecture.md)): 구조 변형 run 5개(`arch-abl-*-pt`)를 `core-ablation-pt.toml` 뒤에 덧붙임, 두 장비 3807031 pull만
- 02:47 — 장시간 실행 중단(사용자 허가, 구조 변형 GPU 테스트): `core-abl-gtrxl-pt` ckpt-6048·`core-abl-lstm-pt` ckpt-935에서 정지
- 04:0x — 발견([032-R20](spec/2026_10_15_22_24_032-model-architecture.md)): 스칼라·바깥 LSTM 토큰이 패딩 뒤에 붙어 attention 타일 하나 추가, 바깥 LSTM 학습 중 재컴파일(76~94초) ([m6_arch_ablation_pt](report/experiment/2026_10_25_01_38_m6_arch_ablation_pt.md) 4절)
- 05:2x — 결정(Claude, [032-R19](spec/2026_10_15_22_24_032-model-architecture.md)): 끝 패딩을 잘라 64 타일에 맞추고 재컴파일 제거 → 스칼라 토큰 +2.3%/+1.1%, 바깥 LSTM +10.5%/+9.7%(CUDA/ROCm)
- 13:04 — 장시간 실행 중단(사용자 허가, WSL FlexAttention 측정): 학습 PC 작업자만 정지(`arch-abl-terrain-pt` ckpt-7463); 09:22 `core-abl-lstm-pt` 10,000 완료
- 13:3x — 발견([032-R19](spec/2026_10_15_22_24_032-model-architecture.md), [060-R1](spec/2026_10_15_22_24_060-infra.md)): 직접 작성 attention 커널이 FlexAttention보다 N256 이상에서 2.5~5배 빠르고 Windows·WSL 같음 → FlexAttention 채택 안 함; 원본 `local/raw/flex_attention_20261025/` ([flex_attention](report/bench/2026_10_25_13_39_flex_attention.md))
- 13:5x — 장시간 실행 재개: 학습 PC 작업자를 Windows에서 `arch-abl-terrain-pt` ckpt-7463부터
- 14:3x — 결정(사용자): 평가를 7개 모델 전체로 — 맞대결 21쌍 × 100판, T0 모델당 300판, val 500 ([032-R20](spec/2026_10_15_22_24_032-model-architecture.md) ③, plan `arch-abl-pt-t0.toml`)
- 17:25 — 장시간 실행 일부 끝: `core-abl-gtrxl-pt` 10,000 완료(CUDA 59.0%·ROCm 41.0%), 14:54 `arch-abl-terrain-pt` 완료, 남은 run ctx·pool·outer·all
- 23:4x — 장시간 실행 재개: 학습 PC 작업자 `arch-abl-ctx-pt` ckpt-7482부터, bc250_1은 `arch-abl-pool-pt` 계속

## 2026-10-26
- 00:38 — 장시간 실행 중단(사용자 지시): bc250_1 작업자 정지(`arch-abl-pool-pt` ckpt-2277), 학습 PC는 계속
- 00:55 — 장시간 실행 재개(사용자 지시): bc250_1 `arch-abl-pool-pt` ckpt-2277부터
- 04:1x — 결정·장시간 실행 시작(사용자 지시): bc250_2를 학습 작업자로 투입(10-18 학습 금지 폐지), `arch-abl-all-pt` 0부터 약 13 s/update
- 04:21 — 결정(사용자): 작업자 우선순위 CUDA 호스트 > bc250_1 > bc250_2(10/2/1), bc250_1 우선순위 2로 재시작
- 04:3x — 결정(사용자): 큐 순서 all을 pool 앞으로, 7개 끝 예상 13:47
- 06:0x — 발견·조치: bc250_2 `arch-abl-all-pt`가 step 389에서 GPU 대기로 멈춤 → 강제 종료(214 update 손실), 학습 PC는 같은 step 정상 → bc250_2/ROCm 쪽 문제; 06:16 outer 완료
- 13:49 — 장시간 실행 끝([032-R17](spec/2026_10_15_22_24_032-model-architecture.md) ⑥·032-R20): 큐 7개 run 모두 10,000 update 완료, 다음은 7개 모델 평가 ([m6_core_ablation_pt](report/experiment/2026_10_24_21_29_m6_core_ablation_pt.md) 4절)
- 13:5x — 결정(사용자, [AMB-020-5](spec/README.md), [032-R20](spec/2026_10_15_22_24_032-model-architecture.md) ③): 7개 모델 평가 방식 — 대전 먼저·T0 다음, val은 bc250_1, CUDA 호스트가 빠지면 ROCm 호스트가 새 판부터 추론 인수, 리플레이에 모델 이름; [020-R15](spec/2026_10_15_22_24_020-env-orchestration.md) 등 spec 개정
- 14:05 — val 표본(bc250_1, 5게임): 122초·상주 최대 2.6 GB, 500게임은 모델당 약 2.5시간 추정
- 14:08 — 장시간 실행 시작([032-R20](spec/2026_10_15_22_24_032-model-architecture.md) ③): val 500 채점 7개(bc250_1 ROCm f32, 약 18시간 예상), 출력 `<shared>/artifacts/m6_arch_ablation_pt/val500/<run>.json`·`.log`, 스크립트 bc250_1 `~/val500/val.sh`
- 시각 없음 — 발견(bc250_2 멈춤 조사): 커널 로그 오류 없음, 소프트웨어는 1호기와 같으나 GPU 클럭 설정이 예전 것·유휴 온도 11~15 °C 높음 → 2호기 하드웨어(발열) 의심 1순위, 재현 시험 미실시
- 14:1x~15:0x — 구현([020-R15](spec/2026_10_15_22_24_020-env-orchestration.md), [061-R16](spec/2026_10_15_22_24_061-shared-storage.md)): 평가 작업 `eval-worker`/`eval-status`, 리플레이 플레이어 이름, 브리지 미접속 감지, `bc-eval --out`; 양보 시험 15판 중복 0, cargo 285·pytest 118 통과; job 폴더 `<shared>/eval/abl-pt/` ([m6_arch_ablation_pt](report/experiment/2026_10_25_01_38_m6_arch_ablation_pt.md) 4절)
- 15:1x — 발견(표본 36판): warm-up 없이 compile·캡처가 판을 46~204초 멈춤 → 평가 작업자가 모든 입력 모양을 미리 돌림(29초 → 2초)
- 15:1x~16:2x — 평가 작업 리뷰 2회와 반영 ([m6_arch_ablation_pt](report/experiment/2026_10_25_01_38_m6_arch_ablation_pt.md) 4절)
- 시각 없음 — 정정: [AMB-020-6](spec/README.md) 결정(16:0x로 적힘)과 평가 작업 리뷰(15:1x~16:2x로 적힘)의 실제 시각은 14:4x~15:0x
- 15:14~15:57 — 평가 작업 장비 이탈·합류 시험([020-AC6](spec/2026_10_15_22_24_020-env-orchestration.md), 56판) 통과; 결정(사용자): val 채점과 추론을 한 장비에서 같이 하지 않음 → 대기 추론은 bc250_2; 연결 대기·중복 판 버그 수정 ([m6_arch_ablation_pt](report/experiment/2026_10_25_01_38_m6_arch_ablation_pt.md) 4절)
- 16:0x — 결정(사용자, [AMB-020-6](spec/README.md)): 평가 작업 재시도 상한 3회, 넘으면 결과 없이 끝내고 포기 판 수 보고 ([020-R15](spec/2026_10_15_22_24_020-env-orchestration.md) ④)
- 16:01~16:14 — 평가 작업 재시험(bd31c87): 끊긴 6판 재실행 중복 0, [020-AC6](spec/2026_10_15_22_24_020-env-orchestration.md)·AC7 통과; 본 평가는 사용자 허락 뒤
- 16:18 — 장시간 실행 시작(사용자 허락): 7개 모델 평가 `<shared>/eval/abl-pt/`(맞대결 100 → 리그전 2,000 → T0 600 → 1,500), 추론 bc250_2, 동시 12판, aed08d8; 결정: CUDA 호스트는 나중에 합류; 로그 bc250_2 `~/eval_abl-pt.out`
- 22:06 — 장시간 실행 끝: val 500 채점 7개 완료(모델당 66~86분), 출력 `<shared>/artifacts/m6_arch_ablation_pt/val500/`; 평가 작업은 계속(리그전 146/2,000)

## 2026-10-27
- 12:2x — 발견·결정(사용자, [020-R15](spec/2026_10_15_22_24_020-env-orchestration.md) ⑨): 세션 끊김으로 "상대 나감" 승리가 된 16판은 재실행 없이 결과 없음(`no_result = "session lost"`)
- 14:18 — 장시간 실행 끝: 7개 모델 평가 4,200판(22시간), 결과 없음 19·포기 0; Bradley–Terry attention 요약 +22 최고·모두 켬 −23, T0 승률 13.0~20.8%, 세 평가 순서 불일치; 채택은 사용자 결정 대기 ([m6_arch_ablation_pt](report/experiment/2026_10_25_01_38_m6_arch_ablation_pt.md), [m6_core_ablation_pt](report/experiment/2026_10_24_21_29_m6_core_ablation_pt.md) 3절)
- 시각 없음 — 새 리포트(사용자 요청, Fable 5.1 high): 7개 모델 평가 독립 검토 — 순위 차이는 잡음 수준, 신호는 모두 켬 약세뿐; 발견: 승패 헤드 학습 손실 0.42 → 0.89 상승 ([m6_arch_ablation_review](report/review/2026_10_27_14_52_m6_arch_ablation_review.md))
- 시각 없음 — 새 리포트(사용자 질문, Fable 5.1 high): 세 약점을 못 메운 이유는 데이터 부족이 아니라 학습량·전제·64 step 상태 단절 ([m6_arch_ablation_review_weaknesses](report/review/2026_10_27_15_08_m6_arch_ablation_review_weaknesses.md))
- 시각 없음 — 결정(사용자, [032-R20](spec/2026_10_15_22_24_032-model-architecture.md) ④): 네 스위치 모두 켬 구성 채택·코어 GTrXL 유지, `config/bc.toml`에 `[model]` 추가; 남은 일 RL 바깥 LSTM 지원·착수 체크리스트
- 시각 없음 — 결정(사용자, [032 §2.7](spec/2026_10_15_22_24_032-model-architecture.md)): BC 본학습 학습 시퀀스 T = 64 유지
- 시각 없음 — 발견: 학습 PC 네트워크 대역 변경으로 두 Linux ROCm 호스트에 닿지 않음, 장비 갱신·큐 파일은 복구 뒤
- 시각 없음 — 결정(사용자 확인, [032 §2.7](spec/2026_10_15_22_24_032-model-architecture.md)·[042](spec/2026_10_15_22_24_042-rl-training.md)): RL 학습기 바깥 LSTM 지원 — 줄마다 첫 구간 길이 `outer − n mod outer`로 구간 차례 계산, 구현·CPU 테스트
- 시각 없음 — 측정 결과: RL 위상별 경로 속도(학습 PC CUDA) — 구간 안 시작 161.3 ms(BC 경로 대비 +32%), 메모리 같음; 스크립트 `<shared>/artifacts/rl_outer_phased_20261027/phased_bench.py`
- 15:5x~16:1x — BC 본학습 준비(J-3·J-5): 세 장비 671268f 빌드, 표본 end-to-end 통과(run `bc-sample-pt`·`bc-sample-s2b-pt`, `bc-sample-s2-pt` 보존) ([m6_pre_bc_review](report/review/2026_10_18_20_51_m6_pre_bc_review.md) J절)
- 시각 없음 — 발견: 승패 헤드 손실 상승은 레이블 버그가 아니라 학습률 감소 탓(상관 −0.96), shard 재build 불필요, val `win_open_mse`로 판단 ([m6_win_loss_check](report/investigation/2026_10_27_16_30_m6_win_loss_check.md))
- 16:31 — 장시간 실행 시작([041](spec/2026_10_15_22_24_041-imitation-learning.md) BC 본학습, 모두 켬, 671268f): 큐 `<shared>/queue/bc.toml` — 학습 PC `bc-s1-pt`(39,096 update, 약 2.0초/update), bc250_1 `bc-ablation-a-pt`(11,387), bc250_2 대기; 로그 `runs/worker-bc.log`
- 16:37~16:47 — 장비 교대: 사용자가 학습 PC를 써서 정지(`bc-s1-pt` `ckpt-102`), 사용자 지시로 bc250_1 → S1(약 13초/update)·bc250_2 → (A)(약 18.8초)

## 2026-10-28
- 00:23 — 장시간 실행 중단: 학습 PC 재부팅으로 ROCm 작업자 정상 종료(S1 `ckpt-2277`, (A) `ckpt-1644`)
- 00:55~01:01 — 장시간 실행 재개: 학습 PC → S1(약 2.0초/update), bc250_1 → (A); 결정(사용자): 학습 PC 학습 중이면 (A)는 bc250_1
- 01:05~ — 사용자 지시: 모두 켬 구조 학습·추론 최적화([032-R19](spec/2026_10_15_22_24_032-model-architecture.md)), 측정 동안 S1·(A)는 두 ROCm 호스트(01:13 CUDA 정지 `ckpt-2730`), OOM 없으면 메모리 증가 허용
- 01:2x~04:xx — 발견·결정([032-R19](spec/2026_10_15_22_24_032-model-architecture.md)): CUDA BC 기본값에 pinned·bf16 변환·channels-last·CUDA graph·attention 역전파 slot 등 채택, 30 update 중앙값 1.87 → 1.31초(−30%), 손실 차이 ≤ 1e-4 ([pytorch_opt](report/bench/2026_10_24_07_03_pytorch_opt.md) 6절)
- 시각 없음 — 발견: 둘째 ROCm 호스트가 35% 느린 원인은 GPU governor 설정(최대 1,500 MHz·열 제한 80 °C), 맞추기는 사용자 결정
- 시각 없음 — 리뷰(별도 리뷰어): 높음 1·중간 3 반영, 재검증 기본값만으로 1.32초; 작업 브랜치 `worktree-train-graph-opt`, master 반영은 ROCm 호스트 확인 뒤
- 10:52~11:00 — 장시간 실행 재개(b1ded78): 학습 PC → S1 `ckpt-5523`부터 1.32 s/update(전 2.25), ROCm 호스트는 pull·테스트 뒤 (A); 학습 PC 9시간 40분 이탈로 CUDA 환산 약 7.9시간 손실(사용자 지적)
- 17:42~17:48 — 장비 교대(사용자 지시): 학습 PC 정지(S1 `ckpt-22906`), 첫째 ROCm 호스트 → S1(약 13.4초/update)·둘째 → (A)(약 17.1초), 남은 S1 약 60시간
- 17:44~18:1x — 발견(M7, PyTorch 전환 뒤 첫 SC:R 실게임, S1 `ckpt-22906`): argmax는 랠리 지정 고리(APM 약 1,410), stochastic은 정상 플레이(APM 195~198, 실패·시야 위반 0), 사용자 확인; 로그 `runs/scr/`

## 2026-10-29
- 00:35~00:40 — 장비 교대(사용자 지시): 학습 PC 투입 → S1 양보(`ckpt-24889`, 약 1.42초/update), 첫째 ROCm 호스트 → (A), 둘째 대기
- 06:09 — S1 끝(`bc-s1-pt` `ckpt-39096` DONE) → 학습 PC가 바로 S2 `bc-s2-pt`(11,387 update) 시작
- 11:11 — S2 끝(`bc-s2-pt` `ckpt-11387` DONE) → 학습 PC가 (A)를 `ckpt-10124`부터 양보받음
- 11:46 — 장시간 실행 끝([041](spec/2026_10_15_22_24_041-imitation-learning.md) BC 본학습, 모두 켬): (A) `bc-ablation-a-pt` `ckpt-11387` DONE; S1·S2·(A) 모두 `<shared>/checkpoints/`
- 11:56~11:58 — 장시간 실행 시작(BC 평가, 사용자 지시): ① 대상 맵 test 채점 S2·S1·(A)(출력 `<shared>/artifacts/bc_main_eval/test/`) ② T0 대전 600판([041-AC2](spec/2026_10_15_22_24_041-imitation-learning.md), `<shared>/eval/bc-main/`)
- 13:10 — 장시간 실행 끝([041-AC2](spec/2026_10_15_22_24_041-imitation-learning.md) T0 대전): S2 600판 — argmax 0/300, stochastic 88/300(29%, 테란 17%·프로토스 16%·저그 55%) → AC2는 저그만 통과; argmax 승자 불일치 90판; 요약 `<shared>/eval/bc-main/bc-vs-t0/summary.md`
- 15:22 — 장시간 실행 끝·새 리포트: 대상 맵 test 1,199게임 — S2 top-1 71.45%([041-AC1](spec/2026_10_15_22_24_041-imitation-learning.md) 통과)·(A) 68.98%(AC3 통과), 오항복 0/17(AC4); 학습 PC S1 채점은 메모리 부족 중단 ([m6_bc_eval](report/experiment/2026_10_29_15_37_m6_bc_eval.md), [bc_ablation](report/experiment/2026_10_29_15_37_bc_ablation.md))
- 시각 없음 — 발견: argmax 승자 불일치 90판 원인 — 진 쪽 남은 예약 명령에 이긴 쪽 OpenBW가 연결을 끊음; 집계 승률 영향 없음, 불일치 판 리플레이 사용 금지 ([m6_bc_eval](report/experiment/2026_10_29_15_37_m6_bc_eval.md) 3절)
- 16:0x~16:30 — 결정·구현·확인(사용자 결정, [011-R22](spec/2026_10_15_22_24_011-bridge-openbw.md), OpenBW 패치 0015): 패배 원격 클라이언트는 예약 명령만 버림, 투혼 24판 불일치 0(패치 전 42/50); 배포는 게임 호스트 재빌드 필요
- 16:46~16:50 — 장시간 실행 시작([041-R20](spec/2026_10_15_22_24_041-imitation-learning.md) 추가 epoch, 사용자 지시): `bc-s1e2-pt`·`bc-s2e2-pt` 큐 추가, 첫째 ROCm 호스트 → `bc-s1e2-pt`(약 11.6초/update), 둘째는 val 500 채점(출력 `<shared>/artifacts/bc_e2_val/val500/`)
- 23:2x — 발견·임시 결정([041-R20](spec/2026_10_15_22_24_041-imitation-learning.md)): `bc-s1e2-pt` 초반 val action NLL 0.894 → 0.901(warmup 구간, 예상대로), win_open MSE 1.00 → 1.27; warmup 체크포인트 2개 채점 건너뜀(`SKIPPED_warmup.txt`)

## 2026-10-30
- 00:38~00:42 — 장시간 실행 이어감(사용자 지시): 학습 PC CUDA가 `bc-s1e2-pt`를 ckpt-2262부터(약 1.38초/update), 첫째 ROCm 호스트는 val 채점으로 전환
- 01:5x — 결정(사용자, [041-R20](spec/2026_10_15_22_24_041-imitation-learning.md) ③ 수정): 두 번째 S1 epoch는 자동 정지 없음, val 기록만 쌓고 계속 여부는 사용자가 정함; ckpt-2084 val NLL 0.915·top-1 71.33%
- 01:5x — 운영 변경(사용자 지시): 둘째 ROCm 호스트를 val 채점에서 뺌(ckpt-4787 중단, 로그 `bc-s1e2-pt__ckpt-4787.aborted-host2.log`); ckpt-2262 val NLL 0.914
- 01:51 — 운영 변경(사용자 지시): val 채점을 "항상 마지막 체크포인트" 방식으로(`~/e2val/val_watch.sh`, 이전 판 `val_watch.sh.v1`), ckpt-4787 채점은 하지 않음
- 10:2x — 발견·조치: ckpt-22578 val 채점이 110분 멈춤 → 중단하고 ckpt-25000으로, 채점당 2시간 제한 추가; val NLL ckpt-17551 0.886(기준 S1 최종 0.894보다 좋음)
- 10:53 — 결정·장시간 실행 시작(사용자, [041-R20](spec/2026_10_15_22_24_041-imitation-learning.md) ③): 체크포인트마다 T0 대전 100판(stochastic, seed_base 42000) 추가, 출력 `<shared>/eval/bc-e2-games/<run>__<ckpt>/`; 둘째 호스트에 패치 0015 배포
- 14:4x — 운영 변경(사용자 지시): 학습 PC는 두 번째 S1까지만, 이후 `bc-s2e2-pt`는 첫째 ROCm 호스트(약 40시간 예상, `~/s2_after_val.sh`); T0 대전 update 32,571 70판 30승 12패 28무(기준 21승 31패 48무)

## 2026-10-31
- 00:1x — 결과·발견([041-R20](spec/2026_10_15_22_24_041-imitation-learning.md)): 두 번째 S1 epoch 끝(10-30 15:06), S2 `bc-s2e2-pt`는 첫째 Linux ROCm 호스트가 학습 중. val action NLL 0.894 → ckpt-37554 0.862(top-1 72.42%), T0 100판 기준 21승 → 최종 39094 36승(`<shared>/eval/bc-e2-games/`). 39094 val 채점은 스크립트 버그로 빠짐(수정함)
- 02:51~02:55 — 장시간 실행 이어감([041-R20](spec/2026_10_15_22_24_041-imitation-learning.md), 사용자 지시): CUDA 작업자가 `bc-s2e2-pt`를 ckpt-2981부터 이어 학습(남은 S2 약 3.5시간), 첫째 ROCm 호스트는 `bc-s1e2-pt/ckpt-39094` val 500 채점 시작
- 03:1x — 준비([041-R20](spec/2026_10_15_22_24_041-imitation-learning.md) ④, 사용자 결정): `bc-s2e2-pt` 끝나면 학습 PC가 T0 stochastic 300판 평가(`<shared>/eval/bc-e2-final/`, seed_base 41300)와 대상 맵 test 채점(`<shared>/artifacts/bc_main_eval/test_cuda/`)을 자동 시작하게 함
- 11:1x — 결과([041-R20](spec/2026_10_15_22_24_041-imitation-learning.md) ④): `bc-s2e2-pt` S2 끝(06:35). `bc-s2-pt` → `bc-s2e2-pt` test top-1 71.58 → 72.15%, T0 stochastic 300판 29 → 39%(무승부 38% 그대로). RL 출발점은 사용자 결정 대기 ([m6_bc_eval](report/experiment/2026_10_29_15_37_m6_bc_eval.md) 5절)
- 11:4x — 새 리포트·문서 정리(사용자 지시): RL 착수 점검 [m6_pre_rl_review](report/review/2026_10_31_11_13_m6_pre_rl_review.md) 작성(사용자 결정 A1~A12, 구현·측정 B1~B13), [m6_bc_eval](report/experiment/2026_10_29_15_37_m6_bc_eval.md) 5절 표 추가, 루트 `README.md`·CLAUDE.md 링크 갱신
- 시각 없음 — 발견(RL 사전준비, 코드 대조): RL 판 벽시계 제한이 3,600초 코드 고정이라 긴 판이 Aborted로 빠질 수 있음, r_value value head는 무작위 초기값, RL 체크포인트 평가·CUDA↔ROCm RL 재개는 실행 기록 없음 ([m6_pre_rl_review](report/review/2026_10_31_11_13_m6_pre_rl_review.md) 5-1절)
- 시각 없음 — 결정(사용자, [041-R20](spec/2026_10_15_22_24_041-imitation-learning.md) ④·[042-AC1](spec/2026_10_15_22_24_042-rl-training.md)): RL 출발·KL 참조 체크포인트 `bc-s2e2-pt/ckpt-11387`, AC1 상대 T0 저그(투혼 stochastic 66%). 판정 샘플링·기준선 `last_failed`·마스크 미결·z는 설명 뒤 결정
- 시각 없음 — 결정(사용자, 액션 감사 미결 마스크): 서플라이 막힘 중 생산 예약 허용(U-1), 시야 밖 기억 적 건물 공격 허용(U-2), 사거리 밖 공격은 현행(U-3). U-1·U-2는 RL 전 spec 수정 → 확인 → 코드 ([m6_action_audit](report/investigation/2026_10_19_20_03_m6_action_audit.md))
- 시각 없음 — 결정(사용자, 액션 감사 U-6 확장): OpenBW 브리지가 부대 명령에서 거절된 유닛별로 이유를 기록(진단용, 모델 입력 불변). 발견: z 테이블에 상대 종족이 없어 RL이 매치업과 무관하게 z를 뽑음 — 결정 필요
- 시각 없음 — 결정(사용자, [AMB-012-6](spec/README.md) 방향·[030-R10](spec/2026_10_15_22_24_030-observation.md)): RL은 `last_failed` 실제 값을 유지하고 SC:R 브리지가 명령 실패를 감지하도록 바꿈(방법은 M7 RL 배포 전에, [012-R14](spec/2026_10_15_22_24_012-bridge-remastered.md) 개정은 spec 수정안에서)
- 시각 없음 — 결정(사용자, [042-AC1](spec/2026_10_15_22_24_042-rl-training.md)): 개선 판정은 확률적 샘플링(argmax도 측정·보고), V-trace·PPO 비교는 계획대로, AC1은 z 없이. spec 반영
- 시각 없음 — 결정(사용자, [042-AC1](spec/2026_10_15_22_24_042-rl-training.md)): 학습 전 기준선은 `last_failed` 0 고정·실제 값 각 200판(같은 seed) 중 높은 쪽 승률, 학습 후는 실제 값. spec 반영
- 시각 없음 — spec 개정(사용자 결정 반영, 액션 감사 U-1·U-2·U-6, [AMB-012-6](spec/README.md)): [010](spec/2026_10_15_22_24_010-game-protocol.md)(`UnitRejected`, PROTOCOL_VERSION 5)·[011](spec/2026_10_15_22_24_011-bridge-openbw.md)(BWAPI 패치 0016)·[012](spec/2026_10_15_22_24_012-bridge-remastered.md)·[013](spec/2026_10_15_22_24_013-version-mapping.md)·[020](spec/2026_10_15_22_24_020-env-orchestration.md)·[030](spec/2026_10_15_22_24_030-observation.md)(feature 스키마 7)·[031](spec/2026_10_15_22_24_031-action-space.md)·[032](spec/2026_10_15_22_24_032-model-architecture.md)·[034](spec/2026_10_15_22_24_034-action-table.md) 개정, [m6_action_audit](report/investigation/2026_10_19_20_03_m6_action_audit.md) §11 결정됨. 코드는 사용자 확인 뒤
- 시각 없음 — spec 개정(리뷰 반영, 메인 결정 — 사용자 보고 예정): [010-R14](spec/2026_10_15_22_24_010-game-protocol.md) `UnitRejected.command` 정의 통일, [011](spec/2026_10_15_22_24_011-bridge-openbw.md)·[012-R17](spec/2026_10_15_22_24_012-bridge-remastered.md)·[020-R15](spec/2026_10_15_22_24_020-env-orchestration.md)·[031](spec/2026_10_15_22_24_031-action-space.md)(유닛별 거절 파일 위치)·[030](spec/2026_10_15_22_24_030-observation.md)·[034](spec/2026_10_15_22_24_034-action-table.md)(`known_building` 제외 규칙) 정합 수정
- 시각 없음 — 발견·spec 정정(통합 시험): 잔상 Attack은 order 9(MoveToAttack)를 보내지만 BWAPI가 10(AttackUnit)으로 보고 → [031-AC1](spec/2026_10_15_22_24_031-action-space.md) ②·[011-R20](spec/2026_10_15_22_24_011-bridge-openbw.md) ①-2 문구 정정, 패킷 order 9는 그대로
- 시각 없음 — 구현·검증 완료(RL 전 액션·브리지 변경): Train 서플라이 엔진 규칙, 기억 적 건물 잔상 명령, `UnitRejected`·PROTOCOL_VERSION 5, BWAPI 패치 0016, feature 스키마 7. 서브 에이전트 병렬 구현·리뷰어 3명, Windows·Linux `cargo test` 302~303·pytest 130·[031-AC1](spec/2026_10_15_22_24_031-action-space.md) 통과. 메인 임시 결정은 사용자 보고 대상
- 시각 없음 — 결정 확인(사용자): RL 전 변경의 메인 임시 결정(떠 있던 건물 잔상 대상 제외 등) 확인. Linux 게임 호스트·Windows CUDA 호스트 재빌드 배포 시작
- 13:4x — 배포 완료(master 29192e3): Windows CUDA 호스트(패치 0015·0016, `uranium._core`, SC:R DLL `runs/scr/mod/samase/`)와 Linux 게임 호스트 셋 재빌드·hostd 재시작, 호스트마다 T0 스모크 정상. Linux 빌드 호스트 `uranium-py` 링크 실패는 `--exclude` + maturin으로 해결
- 시각 없음 — 장시간 실행 시작([042](spec/2026_10_15_22_24_042-rl-training.md) RL 표본, 사용자 지시): 첫째 Linux ROCm 호스트 run `rl-sample-rocm-20261031`(`bc-s2e2-pt/ckpt-11387`, V-trace, 레인 32×T64, 상한 100 update·90분) — 메모리·update 시간·처리량·판 벽시계 측정
- 시각 없음 — 장시간 실행 시작([042](spec/2026_10_15_22_24_042-rl-training.md) RL 표본, 사용자): Windows CUDA 호스트 run `rl-sample-cuda-20261031`(같은 설정, CUDA bf16, 상한 100 update·90분), 로그 `runs/rlsample_cuda_20261031`
- 시각 없음 — 결정(사용자, [061-R6](spec/2026_10_15_22_24_061-shared-storage.md)): RL 체크포인트 전부 보존(`config/rl.toml` `keep_checkpoints = 0`)
- 17:1x — 장시간 실행 끝·발견(`rl-sample-cuda-20261031`, 100 update): learner가 병목(update 중앙값 18.1초), 만든 step 약 74%가 큐 넘침으로 버려짐, 정확 KL GPU 8.8 GiB, 판 최대 728초. `--logs` 상대 경로 버그 발견, step 세는 기준은 사용자 결정 필요. 출력 `runs/rlsample_cuda_20261031*`
- 17:2x — 장시간 실행 끝(`rl-sample-rocm-20261031`, 100 update·77분): 게임이 병목, update 중앙값 23.6초, KL 메모리 5.8 GiB([060-R3](spec/2026_10_15_22_24_060-infra.md) 안), 약 44 step/s(1,000만 step 약 64시간), 엔트로피 0.85 → 0.58. 판 최대 2,845초로 3,600초 제한 우려. 체크포인트 `<shared>/checkpoints/rl-sample-rocm-20261031`
- 17:48 — 장시간 실행 끝·새 리포트(`rl-sample-cuda-remote-20261031`, Linux 게임 12판): 게임을 학습 PC 밖으로 옮기자 update 18.1 → 11.5초, 학습 112 → 176 step/s, 버림 75 → 45%. 6 run 추정 약 59시간~3.9일, step 기준은 사용자 결정 대기 ([m6_rl_sample](report/experiment/2026_10_31_17_51_m6_rl_sample.md))
- 시각 없음 — spec 개정안([042](spec/2026_10_15_22_24_042-rl-training.md)·[020](spec/2026_10_15_22_24_020-env-orchestration.md)·[032](spec/2026_10_15_22_24_032-model-architecture.md)·[060](spec/2026_10_15_22_24_060-infra.md), 사용자 확인 대기): 큐 가득 참을 역압으로(기다림 상한 [AMB-042-11](spec/README.md)), 판 벽시계 제한 설정화 기본안 7,200초(AMB-042-12), `trained_steps` 기록, learner 레인 그룹 `URANIUM_RL_MICRO_LANES` ([m6_rl_sample](report/experiment/2026_10_31_17_51_m6_rl_sample.md))
- 시각 없음 — 결정(사용자, [AMB-042-11](spec/README.md)·AMB-042-12): 큐 기다림 상한 120초, RL 판 벽시계 제한 7,200초. spec [042](spec/2026_10_15_22_24_042-rl-training.md) Approved 복귀, 큐 역압·learner 속도 구현 시작

## 2026-11-01
- 02:2x~03:0x — 측정·결정(큐 역압 + learner 속도 개선): learner bench 7.9 → 2.25초, 실게임 버림 45 → 0%·학습 176 → 266 step/s(게임 병목으로 전환). 메인 임시 결정: CUDA 레인 그룹 2. 6 run 약 3.3~3.7일 추정 ([m6_rl_sample](report/experiment/2026_10_31_17_51_m6_rl_sample.md))
- 03:xx~05:2x — 결과([042-AC1](spec/2026_10_15_22_24_042-rl-training.md) 학습 전 기준선, `bc-s2e2-pt/ckpt-11387`, 투혼 T0 저그): stochastic `last_failed` 0 고정 136승/199판 68.3% [61.6, 74.4], 실제 값 67.0%, argmax 0/200 → 통과 기준 학습 후 Wilson 하한 > 74.4% (`<shared>/eval/ac1-20261101/baseline/`)
- 05:22 — 장시간 실행 시작([042-AC1](spec/2026_10_15_22_24_042-rl-training.md) 6 run, 사용자 지시): V-trace·PPO × seed 3, run당 1,000만 agent step, Windows CUDA learner + Linux 게임 12판, 첫 30분 약 260 step/s. 드라이버 `runs/ac1_20261101/`
- 17:17 — 결정·일시 정지(사용자, [042-AC1](spec/2026_10_15_22_24_042-rl-training.md) run 길이): 학습 곡선으로 1,000만 step 필요성 확인, PPO s1 일시 정지(ckpt-250308). V-trace s1 최종 157승 5패 38무/200 78.5% [72.3, 83.6] — 기준 74.4% 미달
- 19:59 — 결과([042-AC1](spec/2026_10_15_22_24_042-rl-training.md) V-trace s1 학습 곡선, stochastic 200판): 270만 step 75.5%, 490만 76.5%, 730만 76.0%, 1,005만 78.5% — 270만 안에 오른 뒤 평평, 어느 시점도 미통과; 남은 run 길이는 사용자 결정 대기 (`<shared>/eval/ac1-20261101/curve-vtrace-s1/`)
- 시각 없음 — 결정(사용자, [042-AC1](spec/2026_10_15_22_24_042-rl-training.md)·042-R12·[033-AC2](spec/2026_10_15_22_24_033-reward.md)): seed당 agent step 1,000만 → 300만. V-trace s1은 `ckpt-3133569`로 판정, PPO s1부터 300만으로 재개
- 20:44~21:42 — 결과·재개([042-AC1](spec/2026_10_15_22_24_042-rl-training.md), 300만 기준): V-trace s1 `ckpt-3133569` 158-7-35/200 79.0% [72.8, 84.1] 미통과(`<shared>/eval/ac1-20261101/ac1-vtrace-s1-3m/`). 21:42 PPO s1 재개, 캠페인 끝 예상 11-02 21~22시
- 22:10~23:13 — 장애·복구(PPO s1): 인터넷 끊김으로 원격 게임 연결이 끊겨 학습 정지, hostd 격리([020-R4](spec/2026_10_15_22_24_020-env-orchestration.md))로 게임 거부 → hostd 재시작 후 23:13 ckpt-500917에서 재개. 약 1시간 손실, 학습 내용 손실 없음

## 2026-11-02
- 08:15 — 중간 결과([042-AC1](spec/2026_10_15_22_24_042-rl-training.md), 300만 step): PPO s1 77.5% [71.2, 82.7]·V-trace s2 72.5% [65.9, 78.2] 미통과, argmax 모두 0/200
- 13:34 — 중간 결과([042-AC1](spec/2026_10_15_22_24_042-rl-training.md)): PPO s2 157-13-30/200 78.5% [72.3, 83.6] 미통과
- 14:4x — 결정(사용자, [042-AC1](spec/2026_10_15_22_24_042-rl-training.md)·[050-R1](spec/2026_10_15_22_24_050-evaluation.md)): 남은 AC1 평가에서 argmax 제외(BC·네 run 모두 0/200), 드라이버 수정 뒤 학습 중단 없이 재시작
- 14:57 — 장비 변경(사용자, [042-AC1](spec/2026_10_15_22_24_042-rl-training.md)): Linux ROCm 호스트 두 대를 캠페인에서 뺌, 게임은 Windows 로컬 9판 + Linux 3판(약 310 step/s). 평가 호스트 목록에 남은 두 대를 발견해 고침(`<shared>/eval/ac1-20261101/hosts.toml`)
- 18:00 — 중간 결과([042-AC1](spec/2026_10_15_22_24_042-rl-training.md)): V-trace s3 150-8-42/200 75.0% 미통과, V-trace 3 seed 평균 75.5%. PPO s3 학습 시작
- 23:18 — [042-AC1](spec/2026_10_15_22_24_042-rl-training.md) 캠페인 끝: PPO s3 78.5% 미통과, 6 run 모두 미통과. 3 seed 평균 V-trace 75.5%·PPO 78.2%, 차이 2.7%p ≤ 3%p라 규칙대로 V-trace 채택. 요약 `runs/ac1_20261101/summary.md`
- 시각 없음 — 새 리포트: [m6_ac1](report/experiment/2026_11_02_23_37_m6_ac1.md)(AC1 6 run 결과·학습 곡선·처리량·판정·남은 결정), CLAUDE.md 진행 상황 링크

## 2026-11-03
- 00:5x — 새 리포트·발견: [m6_ac1_draws](report/investigation/2026_11_03_00_45_m6_ac1_draws.md) — AC1 무승부 218판 중 81%가 T0 저글링에 무너진 뒤 T0가 못 끝낸 사실상 패배, T0 기억 버그(파괴된 정제소를 적 건물로 기억) 발견. 산출물 `runs/ac1_20261101/draw_analysis/`
- 01:0x~01:46 — 측정(사용자 요청, RL 상대 검토): T1 저그 3종 각 10판, BC 1승 27패 2무, AC1 V-trace s1 3승 24패 3무 — 패배 절반이 4분 전후 러시 전멸 (`<shared>/eval/t1zerg-20261103/`)
- 02:1x — 구현·배포([AMB-042-13](spec/README.md), 사용자 결정): T1·셀프 run spec([042](spec/2026_10_15_22_24_042-rl-training.md)·[033](spec/2026_10_15_22_24_033-reward.md)·[021](spec/2026_10_15_22_24_021-opponent-pool.md)) → 리그·`rl --init`·`config/rl_t1self.toml` 구현, Linux에 T1 봇 5종 빌드(McRave 제외). 리허설 `t1self-rh1` 시작
- 02:33 — 결과(T1·셀프 run 리허설): 셀프·T1 판 정상, 첫 KL 0.019, 약 155 step/s, 점검 평가 경로 정상 ([m6_rl_sample](report/experiment/2026_10_31_17_51_m6_rl_sample.md) 마지막 절)
- 02:43~03:34 — 리허설·발견(사용자 지시 "멈추지 않는 한 계속 학습"): 드라이버를 연속 학습 + 점검 평가 동시 실행으로 바꿈, 평가 포트 충돌(7600)로 평가는 7601·로컬 hostd만. 3회차 정상
- 03:34 — 장시간 실행 시작([042](spec/2026_10_15_22_24_042-rl-training.md) T1·셀프 run, [AMB-042-13](spec/README.md), 사용자 승인): `t1self-s1`(`--init ac1-vtrace-s1/ckpt-3133569`, T1 0.5·셀프 0.5), 상한 없음, 100만 step마다 T1 저그 점검. 드라이버 `runs/t1self_20261103/`
- 08:39 — 정지·발견(사용자 결정): `t1self-s1` 정상 종료(ckpt-4413190, 약 5시간). 셀프 무승부 32 → 72%, T1 학습 판 승률 3%, KL 0.02 → 0.11, 점검도 퇴보. Windows에서 Steamhammer 프로토스 충돌. 원인 조사 시작
- 시각 없음 — 새 리포트·발견: [m6_t1self_stop](report/investigation/2026_11_03_09_09_m6_t1self_stop.md) — 셀프플레이가 현재 Main만 상대해 '경제·터렛, 병력 없음'으로 흘러 무승부 급증(KL 상승보다 먼저), T1 러시 약점 그대로. 산출물 `runs/t1self_20261103/investigation/`, `<shared>/eval/t1self-stop-20261103/`
- 시각 없음 — 구현·배포([AMB-042-14](spec/README.md), 사용자 결정): 스냅샷·anchor 상대, 모델 상대 판 r_value 끔, Steamhammer 테란·저그만 등(8f1ad7d·8727b29). 리허설 `t1self2-rh1` 정상, RAM 때문에 로컬 학습 게임 8판·평가 3판으로 낮춤
- 10:26 — 장시간 실행 시작(T1·셀프 run 2차, [AMB-042-14](spec/README.md)): `t1self-s2`(`config/rl_t1self2.toml`, T1 0.5·현재 0.15·스냅샷 0.25·anchor 0.10), 상한 없음. 드라이버 `runs/t1self2_20261103/`
- 14:xx — 자문·결정(사용자): T1 정체 고성능 모델 자문(4풀 방어 승리 궤적 0, 학습 step 72%가 TvT 모델 상대). 사용자 결정: A 러시 상대 측정 → B 러시 커리큘럼 run, 사람 리플레이 BC 파인튜닝(C)은 제외. A 측정 시작 ([m6_rush_ladder](report/experiment/2026_11_03_17_56_m6_rush_ladder.md))
- 17:03~17:12 — 정지(사용자 결정, `t1self-s2`): ckpt-3525959에서 정상 종료, 스냅샷·anchor 상대 퇴보 신호. A 측정: UAlbertaBot 9풀 변형 26-2-2, T0 `zerg_rush` 14-4-12, Steamhammer 오버풀 변형 1-29-0
- 시각 없음 — 새 리포트: [m6_rush_ladder](report/experiment/2026_11_03_17_56_m6_rush_ladder.md)(러시 상대 기준 측정 A, 150판) — 붕괴는 저글링 도착 뒤 2분 안, 도착 시 마린 1~4기, 무승부 40판 중 34판은 상대가 못 끝낸 사실상 패배
- 시각 없음 — 측정·결정(러시 사다리, 사용자 결정): UAlbertaBot 12풀 37%·9풀 27%, Steamhammer 0~1승. 붕괴 항복 규칙 유지, 사다리 7종. 구현 e317c9d·e8ad109(`config/rl_rush1.toml`, `tools/oss/make_variant.py`) ([m6_rush_ladder](report/experiment/2026_11_03_17_56_m6_rush_ladder.md) 10절)
- 18:55~19:17 — 리허설(`rush1-rh1`, 드라이버 `runs/rush1_20261103/`): 사다리 7종 출전, 항복 처리·점검 정상. 남은 RAM 2.7 GB로 버틴 비율 계산 스레드 4 → 1
- 19:17 — 장시간 실행 시작([042](spec/2026_10_15_22_24_042-rl-training.md) 러시 커리큘럼 run, [AMB-042-15](spec/README.md)): `rush-s1`(`config/rl_rush1.toml`, 사다리 0.70·anchor 0.10·스냅샷 0.15·현재 0.05, 붕괴 항복), `--init ac1-vtrace-s1/ckpt-3133569`, 상한 없음
- 시각 없음 — 새 리포트: [m6_t1_plateau_advice](report/advice/2026_11_03_19_23_m6_t1_plateau_advice.md)(T1 정체 자문 요약, 보상·게임 규칙 개선 후보). 사용자 결정: 자문대로 진행
- 23:4x — 결정(사용자): z로 방어 빌드를 고정하지 않음. 다음 run `rush-s2`(`rush-s1`에서 이어서): 운영형 저그 섞기, 정찰 정보 보상, 초반 생존 보상, 정찰·대응 지표 기록
- 23:42~23:55 — 정정(23:45 RAM 자동 정지 원인): 운영형 저그 측정 에이전트가 띄운 스모크 게임 2판으로 RAM이 떨어짐(ckpt-2798091 정상 종료), 그 에이전트가 23:55에 재개함
- 23:45 / 11-04 00:20 — 정지(RAM 자동 정지, `rush-s1`): 두 번 정상 종료(LATEST `ckpt-3053680`, 305만 step). `rush-s2`가 이어 받으므로 재개 안 함

## 2026-11-04
- 00:3x — 구현([AMB-042-16](spec/README.md)): 정찰 보상 r_scout·생존 보상 r_survive, 사다리 그룹, 승률 표 감쇠, `games.csv` 정찰·대응 지표, `config/rl_rush2.toml`(0862219·f4af960). 리뷰 병합 가능
- 01:0x — 측정·결정(사용자): 운영형 저그 측정(T0 표준 25-2-3 등, Steamhammer 2해처리 뮤탈 2-8-0, `<shared>/eval/macroladder-20261103/`), `rush-s1` 200만은 출발점과 차이 없음. 운영형 그룹 5종 채택([AMB-042-16](spec/README.md) Resolved, 66d35da)
- 01:13~01:33 — 리허설(`rush2-rh1`, 드라이버 `runs/rush2_20261104/`): 러시·운영형 그룹 출전, r_scout·r_survive·지표 기록·점검 정상
- 01:33 — 장시간 실행 시작([042](spec/2026_10_15_22_24_042-rl-training.md) `rush-s2`, [AMB-042-16](spec/README.md)): `--init rush-s1/ckpt-3053680`, `config/rl_rush2.toml`(러시 7·운영형 5, 감쇠 0.97, 정찰 0.15·생존 0.1), 상한 없음, 100만 step마다 점검
- 시각 없음 — 결과: `rush-s1` 200만 재측정은 출발점과 유의한 차이 없음(p > 0.2). 정찰 기준값(출발 ckpt 240판): 정찰 SCV 52%, 3분까지 적 본진 확인 10% — 상대별 차이는 정찰 후 대응이 아님 ([m6_rush_ladder](report/experiment/2026_11_03_17_56_m6_rush_ladder.md) 11·12절)
- 03:49~08:52 — 정지·재개(`rush-s2`): 점검과 학습이 겹쳐 RAM 자동 정지, 감시 누락으로 5시간 멈춤. 로컬 게임 8 → 7·평가 3 → 2판으로 낮춰 재개, 정지 감시 추가
- 시각 없음 — 새 리포트: [m6_rl_stall_advice2](report/advice/2026_11_04_09_13_m6_rl_stall_advice2.md)(RL 정체 2차 자문) — 모델 상대 판 56%·러시 방어 4%, advantage 작음, KL 재시작이 희귀 행동을 막는다는 진단, 코드 버그 없음. 권장: 신호 밀도·KL 완화 run
- 09:31 — 정지·결정(사용자 "권장안대로", 2차 자문): `rush-s2` 정상 종료(ckpt-2151242, 215만 step). 진단 시작, `rush-s3` spec 작성 시작(모델 상대 0.10, 판 시간 상한, 대역 선택, KL 완화, λ 0.95)
- 시각 없음 — 결과(`rush-s2`, 215만 step): 점검 c0 → c1M 러시 4-29-7 → 2-28-10, 4분 전 본진 발견 15 → 7/40, 운영형 벙커 14/20 → 18/19(정찰 없이 방어로 쏠림). 상세 `runs/rush2_20261104/summary.md` ([m6_rl_stall_advice2](report/advice/2026_11_04_09_13_m6_rl_stall_advice2.md))
- 09:4x~ — 진행: 빌드 결정 지점 진단(`runs/diag_20261104/`), 50~60% 러시 단 측정(`<shared>/eval/rushrung-20261104/`), `rush-s3` 구현, 드라이버 `runs/rush3_20261104/` 작성
- 10:0x — 결정(사용자, [AMB-042-17](spec/README.md), bb99674): `rush-s3` 수치 확정 — 모델 상대 판 25분 상한, 러시 대역 선택, KL 0.03 → 0.01 반감기 1,500, λ 0.95, 중단 규칙 3개, 150만 step마다 40판 점검
- 10:54~11:05 — 리허설(`rush3-rh1`): 진단 열(adv_rms_win 약 0.036) 기록, 25분 상한 TimeLimitDraw 동작, 점검 정상
- 11:xx — 진단 결과(rush-s2 ckpt-2151242, 40판 그림자 실행): 창 안 KL 벌점이 결정당 약 0.001이라 KL 가설 기각, advantage 자체가 거의 0. 막은 판·무너진 판을 가르는 것은 벙커가 아니라 저글링을 볼 때 SCV 공격 확률 (`runs/diag_20261104/notes.md`)
- 시각 없음 — 결정(사용자, [AMB-042-17](spec/README.md), fdfab92): 러시 그룹에 50~60% 단 3종 추가(러시 10종), 진단 결과로 KL 완화를 되돌려 기본 스케줄 유지
- 11:05 — 장시간 실행 시작([042](spec/2026_10_15_22_24_042-rl-training.md) `rush-s3`, [AMB-042-17](spec/README.md)): `--init rush-s2/ckpt-2151242`, `config/rl_rush3.toml`(러시 10종 대역·운영형 5종, 모델 상대 0.10·25분), 중단 규칙 자동 판정, 150만 step마다 점검
- 11:19~12:46 — 정지·재개(`rush-s3`): c0 점검과 겹쳐 RAM 자동 정지, 로컬 게임 7 → 5판으로 낮춰 재개. 처음 100 update 러시 판 step 비중 38%(rush-s2 12%)
- 14:05 — 중간 관찰(`rush-s3`, 95만 step): 러시 step 비중 25%로 내려옴, 러시 승률 9~11/40 평탄. 대역 선택이 거의 균등한 것은 승률 표 감쇠 0.97 때문(버그 아님)
- 14:10 — 결정(사용자, `rush-s3`): 상대 선택이 느리게 집중되는 문제는 그대로 두고 150만 step 점검 뒤 자기 모방과 함께 다시 결정
- 14:27 — 점검 끝(`rush-s3` c0 = `rush-s2/ckpt-2151242`): T0 러시 5-16-19, UAlbertaBot 12풀 0-18-22 (`<shared>/eval/rush3-20261104/c0/`)
- 15:41 — 발견(c0 버틴 비율): 출발점 막음 1/80으로 AC1 20/60(33%)보다 크게 나쁨 — rush 계보에서 러시 방어 퇴보로 보임(조건 달라 참고)
- 16:48 — 점검 끝(`rush-s3` c1 1.5M `ckpt-1502897`): T0 러시·UAB12풀 합 11-20-49(14%), c0 6%보다 낫지만 AC1 47% 아래. 학습은 계속 (`<shared>/eval/rush3-20261104/c1/`)
- 17:05 — 결정(사용자): 다음 방향을 fable 3차 자문으로 정함, 그동안 `rush-s3` 계속 학습
- 17:40 — 새 리포트·발견: [m6_rl_stall_advice3](report/advice/2026_11_04_17_04_m6_rl_stall_advice3.md) — AC1 대비 퇴보는 실제(막음 20/60 vs 1/80, p 7e-8)라 진단, 권장 `rush-s4`(AC1 출발, 러시 판 8:00에 SCV ≥ 8이면 승 판정하는 미니게임, 감쇠 0.9, 자기 모방); spec 변경은 사용자 확인 대기
- 17:50 — 결정(사용자, 3차 자문 권장안 전부): `rush-s4` = AC1 `ckpt-3133569` 출발, 러시 미니게임, 러시 4종 + 워밍업, 몫 0.05/0.65/0.20/0.10, 감쇠 0.9, 자기 모방(이긴 판만)
- 17:50 — 정정: 3차 자문 요청·리포트·rush-s4 결정·spec 확정 기록 시각은 추정이었음, 실제(커밋 기준) 16:51·17:04·17:05·17:22
- 17:58 — 장시간 실행 끝(`rush-s3`, STOP 정상 종료): LATEST `ckpt-4092772`(409만 step). 3M 점검 c2는 25/80판에서 중단. rush-s4 코드 master 반영(3ad3aa0~61eaa9d)
- 18:12 — 리허설(`rush4-rh1`, 드라이버 `runs/rush4_20261104/`): `RushJudge` 승 4판 기록, 자기 모방 버퍼·sil_loss 정상, 점검 정상, 오류 없음
- 18:13 — 장시간 실행 시작([042](spec/2026_10_15_22_24_042-rl-training.md) 러시 미니게임 run, [AMB-042-18](spec/README.md)): `rush-s4`(`--init ac1-vtrace-s1/ckpt-3133569`, `config/rl_rush4.toml`), 상한 없음, 1.5M마다 80판 점검, 자기 모방 버퍼 RAM 측정
- 18:30 — spec 확정(사용자): `rush-s4` 러시 미니게임 — [042-R17](spec/2026_10_15_22_24_042-rl-training.md) 판정(`RushJudge`), [020-R19](spec/2026_10_15_22_24_020-env-orchestration.md), [033](spec/2026_10_15_22_24_033-reward.md) r_win, [050](spec/2026_10_15_22_24_050-evaluation.md) 점검, 자기 모방 `wins_only`, [AMB-042-17](spec/README.md)·AMB-042-18 Resolved(f8ba679·9a17a74)
- 18:34 — 정지(`rush-s4`, 18:26 RAM 자동 정지, ckpt-129851): 자기 모방 버퍼가 세그먼트당 약 4~6 MB라 2,048이면 8~12 GB로 RAM에 안 들어감(`runs/rush4_20261104/silmem.csv`)
- 18:45 — 결정·재개(사용자, `rush-s4`): 자기 모방 버퍼 2,048 → 256 세그먼트(ef5394c·fbcb502), ckpt-129851에서 재개
- 18:53 — 판정 시점 100 update(`rush-s4`, 충족): 러시 판 step 52.5%, 승률 표가 상대별로 갈림(T0 러시 0.44 ~ 9풀히드라 0.78)
- 19:10 — 정지·재개(`rush-s4`): 여유 RAM이 자동 정지 기준에 붙어 로컬 학습 게임 5 → 4판으로 낮춰 재개(33만 step)
- 19:49 — 판정 시점 0.75M(`rush-s4`, 미달·계속): 최근 100판 막음 29/100(기준 ≥ 40%). 발견: 감쇠 0.9로 승률 표가 빨리 내려가 UAB12풀·9풀10링이 band에서 제외돼 거의 안 뽑힘
- 19:58 — 결정·재개(사용자, `rush-s4`): 러시 band 폭 0.15 → 0.30, 제외 기준 0.10 → 0(5623957), ckpt-840398에서 재개
- 21:53 — 발견(`rush-s4` c0 = AC1 같은 조건 점검): AC1 막음 6/80(7.5%)으로 이전 측정 20/60(33%)과 크게 다름 → 퇴보 진단의 기준값이 측정 방식 차이였을 가능성. 1.5M 판정 미달·계속
- 21:55 — 요청(사용자): 정체·퇴보 원인을 fable(effort 최고)에 심층 분석 요청(해결안 아닌 원인), `rush-s4`는 계속
- 22:02 — 장시간 실행 정지(사용자: PC 사용): `rush-s4` 정상 종료(ckpt-2391001, 239만 step), 로컬 hostd 종료. 재개는 `runs/rush4_20261104/launch.ps1`
- 22:20 — 새 리포트·발견: [m6_rl_rootcause](report/investigation/2026_11_04_22_20_m6_rl_rootcause.md)(fable) — AC1 두 측정 차이는 시드가 정한 시작 위치(러시 도착 시각) 탓으로 정책 차이 아님, rush 계보 퇴보는 유의하지 않음. 정체 원인은 러시 판 승패가 외생 변수로 정해져 advantage가 거의 없는 것, 신호 경로 버그 없음

## 2026-11-05
- 00:15 — 결정·재개(사용자: "fable 자문 받아가며 멈추라 할 때까지 계속"): `rush-s4` ckpt-2391001에서 재개, 원인 분석을 바탕으로 다음 run 설계를 fable에 자문 ([m6_rl_rootcause](report/investigation/2026_11_04_22_20_m6_rl_rootcause.md))
- 00:34 — 새 리포트·결정(사용자 위임, 4차 자문): [m6_rl_stall_advice4](report/advice/2026_11_05_00_34_m6_rl_stall_advice4.md) — `rush-s4`는 3.0M에서 정지, 측정 수정(시드 → 슬롯, 128판 층화 점검, 세션 기반 막음) + `rush-s5`(러시 고정 비중, 적중 조건 0.5, 자기 모방 필터)
- 00:48 — spec 반영(사용자 위임, 4차 자문, [AMB-042-19](spec/README.md), 2c2f7a4): [020-R20](spec/2026_10_15_22_24_020-env-orchestration.md)·R21 시드 → 시작 슬롯, [050-R6](spec/2026_10_15_22_24_050-evaluation.md) 128판 점검·R7 세션 기반 막음, [042-R18](spec/2026_10_15_22_24_042-rl-training.md) `start_mix`·R19 `fixed` 선택, games.csv 5열, `rush-s5` 절
- 01:05 — 점검 끝(`rush-s4` c1 1.5M `ckpt-1734388`): 합 15/80승으로 c0(AC1) 13/80과 구별 안 됨
- 01:11 — 장시간 실행 끝(`rush-s4`, 4차 자문대로 3.0M 뒤 STOP 정상 종료): `rush-s5` 출발점 `ckpt-3025166`, LATEST ckpt-3054457. 드라이버 `runs/rush5_20261105/` 작성
- 01:27 — 구현 반영(`rush-s5`, 리뷰 통과, 4c7430f·46fc19c·816288e): 시드 → 시작 슬롯 생성기(OpenBW와 80/80 일치), `config/plans/rush5_check.toml`(128판), 세션 막음 지표, `config/rl_rush5.toml`. held.py 대조([050-AC3](spec/2026_10_15_22_24_050-evaluation.md))는 c0 리플레이로(메인 임시 결정)
- 02:13 — 리허설(`rush5-rh1`): 새 열 146/146판 기록, 점검 보고 정상, 여유 RAM 최소 4.2 GB, 러시 적중 39%는 추첨 편차
- 02:13 — 장시간 실행 시작([042](spec/2026_10_15_22_24_042-rl-training.md) `rush-s5`, [AMB-042-19](spec/README.md), 사용자 위임): `--init rush-s4/ckpt-3025166`, `config/rl_rush5.toml`, c0·c0-ac1 128판 + 3M마다 점검. 세션 막음 대조 게이트는 건너뛰고 별도 실행(메인 임시 결정)
- 02:34 — 판정 시점 100 update(`rush-s5`, 미달·계속): 러시 적중 50%(기준 충족), 9풀10링 비중 14%(설정 20%)는 표본 잡음으로 보고 계속
- 04:45 — 점검 끝·발견(`rush-s5` c0 128판): 이른 막음 6/64·늦은 23/64. 발견: `army_s ≤ 225`가 러시 판 거의 전부라 이른·늦은 구분이 안 되고 자기 모방 필터가 사실상 승 판 전부 → fable 자문
- 04:53 — 새 리포트·결정(사용자 위임, 5차 자문): [m6_rl_stall_advice5](report/advice/2026_11_05_04_53_m6_rl_stall_advice5.md) — 자기 모방 `select_start_hit = true` 추가, 판정 재정의(3M은 보고만, 6M에 ⓐ′ 또는 ⓑ′ 이른 막음 ≥ 14/64 둘 다 미달이면 중단·교사)
- 05:09 — 반영·재개(`rush-s5`, 90db800): 05:08 ckpt-2042381에서 재개. 세션 막음 대조 128/128 일치([050-AC3](spec/2026_10_15_22_24_050-evaluation.md) 통과)
- 06:32 — 진행(`rush-s5`, 303만 step): 재시작 뒤 적중 승 24/181(13%)·빗나감 76/201(38%), 자기 모방 버퍼 가득·sil_loss 0.31, 3M 점검 c1 등록
- 07:41 — 점검 끝(c0-ac1 = AC1 `ckpt-3133569`, 128판): 이른 막음 4/64·늦은 14/64 — rush-s4@3.0M과 비슷하거나 낮아 AC1 대비 퇴보 없음 재확인
- 09:50 — 점검 끝·판정 시점 3M(`rush-s5` c1 `ckpt-3033127`): 이른 막음 11/64(c0 대비 p 0.149, ⓑ′ 미달)·늦은 21/64, 퇴보 게이트 통과
- 10:26 — 6M 점검 시작(`rush-s5` c2 `ckpt-6057233`, 128판), 학습은 계속
- 12:46 — 판정 시점 6M·정지(`rush-s5`): c2 이른 막음 6/64(3M 11/64는 잡음), ⓐ′ +1.5%p(p 0.389) → 둘 다 미달 = 중단·교사 경로. 발견: 12:05 CUDA illegal memory access로 learner 정지, supervise가 못 잡아 추론 실패 판이 쌓임(`runs/rush5_20261105/rush-s5/supervise.log`). LATEST ckpt-7362114
- 13:04 — 새 리포트·결정(사용자 위임, 6차 자문): [m6_rl_stall_advice6](report/advice/2026_11_05_13_04_m6_rl_stall_advice6.md) — 보상만의 RL은 막다른 길, 정찰 반응형 교사 `t0-terran-react` → botgen 2,000판 → BC 파인튜닝 → `rush-s6`. CUDA 오류 대비 가드 5종 구현
- 13:07 — 착수(6차 자문): BC 128판 점검(`<shared>/eval/bc-check-20261105/`), 가드 구현, 교사 spec·구현. 발견: 12:05:44 Windows 이벤트 nvlddmkm 153(GPU 오류)이 CUDA 오류와 같은 시각
- 13:30 — 장애: Linux 게임 호스트(asus) 응답 없음, BC 점검은 로컬로 계속(52/128). 가드 구현([042-R20](spec/2026_10_15_22_24_042-rl-training.md), [060-R6](spec/2026_10_15_22_24_060-infra.md), [AMB-042-20](spec/README.md)) 끝, Windows 테스트 통과·리뷰 중
- 13:35 — 전부 정지(사용자: PC 사용): BC 점검, 로컬 hostd, 교사 구현 에이전트(WIP 452c366 보존), 가드 리뷰 에이전트 종료. 학습 run 없음(rush-s5 LATEST ckpt-7362114)
- 22:25 — 재개(사용자: 장비 모두 복귀): Linux 호스트 주소 변경 반영·hostd 재시작, BC 점검·가드 테스트·교사 구현 재개, bc250 두 대를 게임·ROCm 작업자로 준비 시작
- 22:31 — 장비 복귀: bc250_1·bc250_2를 master(815ba83)로 올림, hostd·OSS 봇·ROCm 정상. 할당: 이 PC GPU BC 점검, asus 점검·테스트, bc250_1 빈 파인튜닝, bc250 게임 슬롯 교사 검증·botgen
- 22:4x — 결정·구현(사용자 위임, [041-R21](spec/2026_10_15_22_24_041-imitation-learning.md), [AMB-041-9](spec/README.md)): 교사 BC 파인튜닝 recipe를 spec에 넣고 게임 단위 고르기 `[data.subsample]` 구현. 빈 파인튜닝 `config/bc_ft_empty.toml` S2 1/4 = 2,814 update. 발견: 기준 `bc-s2e2-pt` val 500 채점이 없음
- 22:41 — 반영(사용자 위임, 6차 자문): 실행 안전 가드 master(5b88746·e8d3f8d·86c8db6) — [042-R20](spec/2026_10_15_22_24_042-rl-training.md) 장치 치명 오류 즉시 종료·죽은 learner 감지, [060-R6](spec/2026_10_15_22_24_060-infra.md) supervise 진행 판정·재시작, [AMB-042-20](spec/README.md). 테스트 통과
- 22:44 — 점검 끝(BC `bc-s2e2-pt/ckpt-11387`, [050-R6](spec/2026_10_15_22_24_050-evaluation.md) 128판): 이른 막음 2/64·늦은 19/64 — AC1·rush 계보와 같은 수준, 파인튜닝 게이트 기준값 (`<shared>/eval/bc-check-20261105/`)
- 22:46 — spec·구현(사용자 위임, 6차 자문, [AMB-021-4](spec/README.md)): 교사 `t0-terran-react` [021-R8](spec/2026_10_15_22_24_021-opponent-pool.md)(정찰 → 러시 식별 → 방어, 아니면 확장), [050-R8](spec/2026_10_15_22_24_050-evaluation.md) 세션 지표, `--until-frame` [040-R21](spec/2026_10_15_22_24_040-replay-dataset.md), 점검 plan. 스모크 2판 정상, 리뷰 중
- 22:53 — 반영(사용자 위임, [041-R21](spec/2026_10_15_22_24_041-imitation-learning.md)): 7940eb8 `[data.subsample]`·`config/bc_ft_empty.toml`(lr 1.5e-5, 메인 임시 결정) master. bc250_1 빈 파인튜닝 `bc-ft-empty-s1`, bc250_2 기준 val500 채점 예정
- 22:55 — 리뷰(교사 `t0-terran-react`): 벙커 방향 오류(시작 위치 8곳 중 5곳), 러시 식별 규칙이 평범한 운영형 저그에도 성립('무조건 방어 금지' 제약과 충돌), 벙커 반복 지시 등 지적. 규칙 수치는 fable 자문
- 22:59 — 장시간 실행 시작: bc250_1 빈 파인튜닝 `bc-ft-empty-s1`(2,814 update, `<shared>/checkpoints/bc-ft-empty-s1/`), bc250_2 `bc-s2e2-pt/ckpt-11387` val500 채점(`<shared>/artifacts/bc_e2_val/val500/`)
- 23:08 — 수정(교사, 리뷰 지적): 벙커를 앞마당 쪽 초크로, 대체 자리 제한, SCV 추격 금지 등(c281fa7·064f6bb). 스모크 T0 러시 막음, T0 표준 저그 못 막음. 러시 식별 규칙은 자문 대기
- 23:12 — 새 리포트·결정(사용자 위임): [m6_teacher_rush_rule_advice](report/advice/2026_11_05_23_12_m6_teacher_rush_rule_advice.md) — 풀 착공 HP 추정, 저글링 수·시각 규칙, 운영형 점검 기준 상대별 재정의 등을 교사 구현에 반영 지시
- 23:23 — 구현(교사 러시 식별 규칙, eaeb916): 자문 규칙 반영, 운영형 점검 plan 분리. 스모크 4판 운영형 상대 발화 0, T0 러시 1판 패. 8:00 전 섬멸 승은 통과로 셈(메인 임시 결정)
- 23:52 — 교사 검증 결과(`runs/teacher_check_20261105/`): 운영형 대부분 통과(T0 표준 저그·뮤탈 선제 벙커 남음), 러시 128판 이른 막음 2/62·늦은 18/64로 BC와 같고 목표 32·40에 크게 미달 → 방어 실행 자체가 약함, fable 자문

## 2026-11-06
- 00:06 — 결정·재개(사용자: 토큰 한도 임박, 고정 설정으로 계속 학습): 교사 경로 보류, `rush-s5`를 ckpt-7362114에서 재개(사용자 지시가 6M 중단 판정보다 우선), bc250_2 게임 추가, keepalive `runs/rush5_20261105/keepalive.ps1`
- 00:09 — 채점 끝(bc250_2): `bc-s2e2-pt/ckpt-11387` val500 top-1 72.37% → 파인튜닝 게이트 하한 71.87% (`<shared>/artifacts/bc_e2_val/val500/`). 빈 파인튜닝이 끝나면 bc250_1도 rush-s5 게임에 넣는 스크립트 설치
- 09:16 — 진행(`rush-s5`, 1,500만 step, 게임 16판, 약 315 step/s): c3(9M) 이른 막음 12/64, c4(12M) 5/64로 잡음 범위. 발견: `bc-ft-empty-s1`이 05:37 JSON 읽기 오류로 종료(ckpt-1698, 재개 가능), 원인 조사는 미룸
- 12:30 — 진행(`rush-s5`, 1,768만 step): 11:06 RAM 자동 정지 → keepalive 재기동. c5(15M) 이른 막음 13/64(p 약 0.07), 지금까지 가장 높지만 오르내림
- 13:16 — 정지·이양(사용자: 이 PC 사용): `rush-s5` 정상 종료(ckpt-18365112, 1,837만 step), learner를 bc250_1(ROCm)로 옮기는 작업 시작
- 13:34 — 장시간 실행 재개(`rush-s5`, bc250_1 ROCm learner): 게임 bc250_2 4 + asus 3판, 약 28 step/s(이 PC의 1/5~1/8), 메모리 [060-R3](spec/2026_10_15_22_24_060-infra.md) 예산 안. 128판 점검 중단
- 18:26 — 이양·재개(사용자: 이 PC 다시 사용): bc250_1 learner 정상 종료(ckpt-19183299, RAM 빠듯), 로그를 되돌려 18:25 이 PC에서 게임 16판으로 재개
- 19:12 — 재개(사용자: 주간 토큰 초기화): 교사 방어 실패, `rush-s5` 지속 여부, 빈 파인튜닝 오류를 fable(effort 최고)에 자문. `rush-s5`는 계속
- 19:42 — 새 리포트·결정(사용자 위임, 7차 자문): [m6_rl_stall_advice7](report/advice/2026_11_06_19_42_m6_rl_stall_advice7.md) — 교사 실패 절반은 측정(8:00 뒤 소모전), 나머지는 벙커 자리·SCV·마린·건설 재개 결함과 식별 지연 → 수정 A~F 착수. rush-s5는 18M 점검 뒤 정지, 빈 파인튜닝 오류는 손실 NaN
- 19:54 — 장시간 실행 끝(`rush-s5`, 사용자: 이 PC 사용): 19:47 정상 종료(LATEST ckpt-20174905, 2,017만 step), 18M 점검 c6 미완. 다른 장비로 옮기지 않음
- 20:27 — 교사 수정 A~F 검증(7차 자문, 브랜치 6f6462f, 10분 판): 러시 128판 8:00 기준 이른 47/64·늦은 60/64로 목표 통과(이전 18/62·46/64). T0 뮤탈 선제 벙커 때문에 ①b를 2:25로 조정(메인 임시 결정)
- 20:30 — spec·구현·검증(사용자 위임, 7차 자문, [AMB-021-4](spec/README.md)): 교사 수정 A~F(벙커 자리 지상 경로 BFS, SCV 동원 축소, 마린 공격 전환, 건설 재개, 정찰 조기 출발, 점검 10분 상한·8:00 지표). 러시 이른 47/64·늦은 60/64 통과, T0 뮤탈만 미달. 산출물 asus `~/teacher7-runs/`
- 20:34 — 장시간 실행 재개(`bc-ft-empty-s1`, bc250_1): NaN 가드(8b4cffc, [041-R3](spec/2026_10_15_22_24_041-imitation-learning.md)·[AMB-041-10](spec/README.md)) 반영 뒤 ckpt-1698에서 재개, 같은 batch를 NaN 없이 통과(데이터 원인 아님으로 추정)
- 20:35 — 결정·검증(메인 임시 결정, [AMB-021-4](spec/README.md)): 교사 ①b `pool_done_before` 3,600 → 3,480(2:25). 재점검 러시 이른 46/64·늦은 61/64, T0 뮤탈 선제 벙커 0으로 통과
- 20:42 — 반영(사용자 위임, 7차 자문): 교사 수정 A~F + ①b 2:25(e19ad3b·6f9e23d) 리뷰 통과, ①b 수치는 사용자 확인 요청. 다음: 교사 botgen 2,000판 착수
- 22:15 — 교사 botgen 끝(사용자 위임, 6·7차 자문): 2,000판 모두 Finished(`<shared>/datasets/replays/raw/botgen/teacher-20261106/`) → `replays/shards/botgen/teacher-20261106-{a..d}` 1,773판·102만 step, 디싱크 0. 10분 상한 판 승패 문제로 `--time-limit` 추가(리뷰 중). 파인튜닝 3,242 update
- 22:19 — 반영(사용자 위임): botgen plan·`uranium-replay build --time-limit`·`config/bc_ft_teacher.toml`(60b7fe2·9e41368) 리뷰 통과. 교사 몫 step 12.5%로 자문 예상 22%보다 작음. bc250_2 교사 파인튜닝 착수
- 22:20 — 장시간 실행 끝·데이터(사용자 위임, [040-R14](spec/2026_10_15_22_24_040-replay-dataset.md)·R21, [050-R8](spec/2026_10_15_22_24_050-evaluation.md), [AMB-021-4](spec/README.md)): 교사 botgen 2,000판 build 1,773판·1,022,608 step(`…/botgen/teacher-20261106-{a..d}`), 5분 미만 227판 제외. 발견: 10분 판 승패가 퇴장 순서로 정해져 `--time-limit` 추가(메인 임시 결정)
- 22:27 — 장시간 실행 시작(교사 BC 파인튜닝 `bc-ft-teacher-s1`, bc250_2 ROCm): `--init bc-s2e2-pt/ckpt-11387`, `config/bc_ft_teacher.toml`, 3,242 update, 11-07 09:50쯤 끝 예상. 체크포인트 `<shared>/checkpoints/bc-ft-teacher-s1/`

## 2026-11-07
- 00:11 — 끝·이양(사용자: CUDA PC 유휴): `bc-ft-empty-s1` 23:49 끝(ckpt-2814, NaN 재발 없음). 교사 파인튜닝을 이 PC(CUDA)로 옮기고 bc250 두 대는 빈 파인튜닝 채점·128판 점검
- 00:25 — 이양 끝: `bc-ft-teacher-s1` ckpt-476에서 이 PC 재개(update당 약 1.4초, ROCm의 약 9배, 로그 `runs/bc_ft_teacher/train.log`). 빈 파인튜닝 val500·128판 점검(`<shared>/eval/bc-ftempty-check-20261107/`) 시작
- 01:53 — 장시간 실행 끝(`bc-ft-teacher-s1`, 이 PC CUDA): ckpt-3242. val500·교사 hold-out 채점과 128판 점검(`<shared>/eval/bc-ftteacher-check-20261107/`) 시작
- 02:58 — 게이트 결과(교사 경로): val500 BC 72.37%·빈 FT 72.36%·교사 FT 72.45%(교사 hold-out 65.3 → 92.2%) 통과, 러시 게이트(이른 ≥ 16·늦은 ≥ 20)는 교사 FT 2/64·18/64로 실패 — 교사 행동을 흉내 내도 실전 방어로 안 옮겨감. fable 자문
- 03:20 — 새 리포트·결정(사용자 위임, 8차 자문): [m6_rl_stall_advice8](report/advice/2026_11_07_03_20_m6_rl_stall_advice8.md) — hold-out 92%는 NoOp·어택땅 덕분, 벙커 레이블이 혼합에서 희석(0.04%). 다음: 교사 전용 클론 5 epoch로 희석·이탈 판별
- 04:46 — 진단 결과(교사 전용 클론 `bc-tclone-e1~e5`, `config/bc_teacher_clone.toml`): 벙커 top-1 27 → 63%, e5 128판 이른 막음 18/64·늦은 28/64로 처음 크게 오름, 그러나 SCV 생산·벙커 자리 약함, val500 48.8%. 결과 `<shared>/eval/bc-tclone-check-20261107/`, `runs/tclone/`
- 04:58 — 새 리포트·결정(사용자 위임, 9차 자문): [m6_rl_stall_advice9](report/advice/2026_11_07_04_58_m6_rl_stall_advice9.md) — 희석이 주원인. 생산 recipe `bc-mix-s1`(교사 step 50%, 7 epoch 체인) → 게이트 G1~G4 → 통과 시 `rush-s6`
- 05:05 — 장시간 실행 시작([041-R21](spec/2026_10_15_22_24_041-imitation-learning.md) ⑥, d8cb7c5): 생산 혼합 `bc-mix-e1~e7`(이 PC CUDA, `config/bc_mix_eN.toml`, e1은 `bc-s2e2-pt/ckpt-11387`에서), 교사 몫 7 epoch 합 49.2%, 기록 `runs/mix/`
- 05:11 — 점검 결과(9차 자문 ②, 클론 `bc-tclone-e5` 운영형 40판, `<shared>/eval/bc-tclone-macro-20261107/`): 4:00 전 벙커 28/40(교사 0/40) → 클론은 러시 식별 없이 일찍 방어(G2 미달). 대비로 운영형 botgen +400판 생성
- 05:33 — 데이터 추가(사용자 위임, 9차 자문): 운영형 botgen +400판(`config/plans/teacher_botgen_macro2.toml`) → `replays/shards/botgen/teacher-20261107-macro2`(24만 step, 디싱크 0), `bc-mix`는 다음 epoch부터 포함(메인 임시 결정)
- 06:3x — 결정·발견(메인 임시 결정, [041-R21](spec/2026_10_15_22_24_041-imitation-learning.md) ⑥): `bc-mix-e1`은 macro2 전 데이터로 끝(ckpt-801, 벙커 top-1 30.9%), e2부터 macro2 포함·교사 50%를 맞추는 사람 fraction 0.043으로 체인 재시작
- 11:57 — 결과·결정(`bc-mix-e1~e7` 끝, 메인 임시 결정): 교사 hold-out 벙커 top-1 30.9 → 66.4%(전체 96.8%), 70% 미만·마지막 +3.1%p라 연장 규칙대로 e8~e10 실행. 체크포인트 `<shared>/checkpoints/bc-mix-eN/`
- 16:06 — 게이트 결과(9차 자문, `bc-mix-e10/ckpt-1053` 채택, 벙커 top-1 73.1%): G1 이른 16/64 통과·늦은 19/64 실패, G2 운영형 벙커 19/40 실패, G3·G4(T0 25%) 실패, val500 72.23% → 분기상 DAgger. 결과 `<shared>/eval/bc-mix-{check,macro,t0}-e10-20261107/`, `runs/mix/`
- 16:08 — 게이트 결과(`bc-mix-e10/ckpt-1053`): G1 늦은·G2·G3·G4 실패, 사람 데이터 쪽으로 다시 희석되고 벙커가 식별 뒤 대응이 아니라 3~4분 습관. 자문 분기가 서로 부딪쳐 fable에 자문
- 16:22 — 결정(사용자): 10차 자문의 학습·개선까지만 하고, 이후 T1 봇마다 50판 평가로 마무리
- 16:29 — 새 리포트·결정(사용자 위임, 10차 자문 — 마지막 학습·개선): [m6_rl_stall_advice10](report/advice/2026_11_07_16_29_m6_rl_stall_advice10.md) — 혼합 모델은 명령 이력으로 '사람 모드'를 고르고 벙커는 시계·저글링 목격에 걸림. DAgger 재레이블 3 라운드, G4는 TvZ ≥ 50%로, 끝나면 T1 평가 → Checkpoint 1
- 16:46 — DAgger 1라운드 판 생성(10차 자문): `bc-mix-e10/ckpt-1053` stochastic 600판(`config/plans/dagger_r1.toml`) 모두 Finished (`<shared>/eval/dagger-r1-20261107/`), 재레이블 도구 구현 중
- 18:30 — spec·구현(사용자 위임, 10차 자문, [AMB-041-12](spec/README.md)): DAgger 재레이블 도구 `uranium-replay build --src dagger --relabel terran_react`([040](spec/2026_10_15_22_24_040-replay-dataset.md) 출처 `dagger`, [041-R22](spec/2026_10_15_22_24_041-imitation-learning.md), [021-R8](spec/2026_10_15_22_24_021-opponent-pool.md)). 교사 점검 통과, 스모크에서 교사 레이블이 Train 명령을 관측마다 반복하는 문제 발견. master 미반영
- 18:47 — 새 리포트·결정(사용자 위임, 11차 자문): [m6_rl_stall_advice11](report/advice/2026_11_07_18_47_m6_rl_stall_advice11.md) — 17:09 사용자 PC 사용으로 ROCm 호스트만 남아 DAgger 2라운드 × 1 epoch로 축소, 반복 Train은 32프레임 솎기. 1라운드 동안 BC 기준선 T1 평가 병행
- 19:03 — spec·구현·검증(사용자 위임, 11차 자문, [AMB-041-12](spec/README.md)): [041-R22](spec/2026_10_15_22_24_041-imitation-learning.md) 반복 교사 명령 솎기(32프레임), 재레이블 일치율 97~99% 유지. 벙커 기억 만료 48프레임은 점검 미달로 144로 되돌림. 1라운드 600판 재레이블 약 17분 예상, 측정 중 asus 재부팅
- 20:01 — 장시간 실행 시작(11차 자문): BC 기준선 T1 평가 250판(`<shared>/eval/t1-bc-20261107/`), 1라운드 재레이블 → `<shared>/datasets/replays/shards/dagger/r1`(591판·421,282 step), `bc-dagger-r1` bc250_1 ROCm(`--init bc-mix-e10/ckpt-1053`, 1,137 update). [041-R22](spec/2026_10_15_22_24_041-imitation-learning.md)·[AMB-041-12](spec/README.md)를 축소안으로 맞춤
- 22:15 — 발견·결과(BC 기준선 T1 평가 246판): 상대 봇이 판 도중 죽은 23판(Iron·Steamhammer assert 실패)이 우리 승으로 기록됨 → 무효 처리(메인 임시 결정). 유효 223판 중 BC 11승(모두 UAlbertaBot) (`<shared>/eval/t1-bc-20261107/`)
- 23:0x — 발견·수정(사용자 지시, [021](spec/2026_10_15_22_24_021-opponent-pool.md)): 크래시 원인은 Linux T1 봇 빌드(`tools/oss/common.sh`)에 `-DNDEBUG`가 없어 assert가 살아 있던 것 → 모든 봇에 추가, asus 확인 크래시 Iron 6/12·Steamhammer 1/12 → 0/24. asus만 새 바이너리 ([oss_bot_porting](report/survey/2026_10_16_00_18_oss_bot_porting.md) "NDEBUG 확인")

## 2026-11-08
- 00:55 — 결과·새 리포트·결정(사용자 위임, 12차 자문): `bc-dagger-r1/ckpt-1137` 끝, 이 PC 복귀로 2라운드 300판·재레이블. 첫 시험 정찰·막음 개선으로 셋 중 둘 통과 → 2라운드. [m6_rl_stall_advice12](report/advice/2026_11_08_00_50_m6_rl_stall_advice12.md): 운영형 벙커 계기가 시계에서 정찰로 바뀜, r3까지·선택 규칙. NDEBUG 봇으로 BC T1 평가 300판 재시작(`<shared>/eval/t1-bc2-20261108/`)
- 02:20 — 결과·결정(12차 자문 규칙): `bc-dagger-r2/ckpt-1218` 막음 19·37/64(합 56) → r3 조건 미달·중단 조건 아님, r3 없이 선택(메인 임시 결정: r3는 조건을 모두 채울 때만). r1 TvZ 32판 27승. Checkpoint 1 = `bc-dagger-r1/ckpt-1137`, T1 평가 시작(`config/plans/t1_eval_ck1.toml`)
- 12:20 — 결과(Checkpoint 1 평가, 사용자 결정 2026-11-07·12차 자문, T1 6종 × 50판, NDEBUG 봇, 같은 seed): BC 기준선 13/299승, Checkpoint 1 33/300승(UAlbertaBot 23·McRave 5·Steamhammer 4·ZZZKBot 1). val500 BC 72.37%·r1 72.19% (`<shared>/eval/t1-ck1-20261108/`, `<shared>/eval/t1-bc2-20261108/`)
- 12:5x — 체크포인트(사용자 결정 2026-11-07): Checkpoint 1 = `bc-dagger-r1/ckpt-1137`, 커밋 29b3c1b·태그 `checkpoint-1`, [checkpoint1](report/release/2026_11_08_12_21_checkpoint1.md), `releases/checkpoint-1/`. 평가 콘솔·hostd 종료, 반영된 worktree 30개 정리
- 13:4x — 결과·체크포인트(사용자 요청): Checkpoint 1 RL = `ac1-vtrace-s1/ckpt-3133569`, 태그 `checkpoint-1-rl`([checkpoint1_rl](report/release/2026_11_08_13_35_checkpoint1_rl.md), `releases/checkpoint-1-rl/`). AC1 T1 6봇 20/294승, TvZ 24/32. Windows Steamhammer 프로토스전 5판 상대 Aborted는 무효 (`<shared>/eval/ac1-ck-20261108/`)
- 19:2x — 새 리포트(사용자 요청: JEPA 선행 조사, spec·구현 없음): [jepa_survey](report/survey/2026_11_08_19_26_jepa_survey.md) — 맞는 형태는 시간 예측 보조 손실, BC·RL 정체에는 효과 작을 것으로 예측, 구현 전 Checkpoint 1 표현 선형 진단 제안

## 2026-11-09
- 13:04 — 결정·장시간 실행 시작(사용자 요청): fable 판정 '수정 후 진행 가능'으로 표현 진단 시작(풀 정보 두 경로, 반사실 개입, 헤드 재학습 ≤ 100 update, 체크포인트 4개, 종족별). 설계 [jepa_survey](report/survey/2026_11_08_19_26_jepa_survey.md) 7절, 산출물 `runs/jepa_diag/`
- 13:11 — 발견·정리: 같은 작업이 두 세션으로 열려 7절 개정이 두 번 이뤄짐(미반영 브랜치 e4b476f) → 그쪽에만 있던 항목을 [jepa_survey](report/survey/2026_11_08_19_26_jepa_survey.md) 7절에 합침, 실행은 한 세션만
- 13:2x — 기록: 복제 세션 판본과 전수 대조해 남은 항목(사전 예상 등)을 [jepa_survey](report/survey/2026_11_08_19_26_jepa_survey.md) 7절에 합침. 사전 예상: 여유 폭 작음·학습 신호 문제 → JEPA 가치 낮음
- 13:35 — 장시간 실행 시작(사용자 지시: 순서 없는 일은 병렬): 트랙 A 저그 진단(이 PC), B 사람 리플레이 테란·프로토스(ROCm 호스트 1), C Checkpoint 1 대 T·P 봇 판 생성 + T0 테란·프로토스 각 32판(ROCm 호스트 2). 산출물 `<shared>/eval/jepa-diag-20261109/`·`<shared>/eval/jepa-diag-tp-20261109/`
- 15:14 — 결과·발견(트랙 A 저그 진단): 판정 '코어 병목' — 풀 HP 정보가 코어 입력(R² 0.73)까지 있고 GTrXL 1층 뒤 0.19·h_t 0.10으로 사라짐, '학습 신호 문제' 예상은 틀림. 공간 인코더 `s8` 출력이 모두 0인 것 발견 ([jepa_survey](report/survey/2026_11_08_19_26_jepa_survey.md) 10.1절)
- 16:36 — 결과(코어 안 측정): 풀 HP는 GTrXL 1층 GRU 게이트 두 개에서 사라짐(0.75 → 0.40 → 0.19), 1층 attention은 지금 step에 0.9%만 줌. 손실은 풀 HP에 선택적 ([jepa_survey](report/survey/2026_11_08_19_26_jepa_survey.md) 10.3절, `<shared>/eval/jepa-diag-20261109/core_split/`)
- 16:47 — 결과·발견(트랙 C): T·P 봇 360판 → 진단용 shard `<shared>/eval/jepa-diag-tp-20261109/shards/tp`. Checkpoint 1 T0 표준 테란 9/32·프로토스 2/32(BC 38%·16%) → 프로토스전 퇴보(p 0.033) ([jepa_survey](report/survey/2026_11_08_19_26_jepa_survey.md) 10.2.1절)
- 17:21 — 결과·발견(트랙 D): T·P전 모델 판도 저그와 같이 단서가 표현에서 사라지고 쓰이지 않음(개입 Δ ≤ 0.009), 적 최대 관측 수가 c에서 거의 안 읽힘 ([jepa_survey](report/survey/2026_11_08_19_26_jepa_survey.md) 10.2.2절)
- 17:39 — 결정(사용자): 다음 개선 모델에 헤드 우회로·전체 정보 보조 목표·JEPA형 자기 예측 손실 모두 적용(spec 수정은 나중), 공간 인코더 0·바깥 LSTM은 조사, 프로토스전 퇴보 원인은 조사 안 함 ([jepa_survey](report/survey/2026_11_08_19_26_jepa_survey.md) 8절)
- 18:09 — 결과·발견(사용자 요청 조사): 공간 인코더 `s8` 0은 마지막 ResBlock의 '죽은 ReLU'(warmup 중 update 566~635에 죽음, Checkpoint 1은 정확히 0)라 공간 지도 정보가 코어에 0. 고칠 후보는 [032](spec/2026_10_15_22_24_032-model-architecture.md) 개정 대상 ([jepa_survey](report/survey/2026_11_08_19_26_jepa_survey.md) 10.4절)
- 18:28 — 결정(사용자): 다음 모델은 BC부터 새로 학습, 확정 개선은 범용 대책 3종·공간 ResBlock 끝 ReLU 제거·죽은 채널 감시 ([jepa_survey](report/survey/2026_11_08_19_26_jepa_survey.md) 8절)
- 19:02 — 결과(트랙 B): 사람 리플레이 val·test 약 149만 행에서도 같은 손실(건물 HP는 e_pool에서, 프로토스 빌드 단서는 코어에서 사라짐) → 상태 분포가 아니라 모델 구조·학습의 성질 ([jepa_survey](report/survey/2026_11_08_19_26_jepa_survey.md) 10.2.3절)
- 19:18 — 결과·바로잡음(사용자 요청 조사): 바깥 LSTM은 쓰이고 도움이 됨(리셋 때 사람 test NLL 0.876 → 0.922). 바로잡음: BC는 상태를 묶음마다 리셋하지 않고 기울기만 끊음. T0 저그 짝 대전 중간 집계 차이 없음 ([jepa_survey](report/survey/2026_11_08_19_26_jepa_survey.md) 10.5절)
- 20:01 — 결과(바깥 LSTM 대전 끝): T0 표준 저그 같은 seed 짝 128판 정상 65.6% 대 64결정 리셋 64.1%(McNemar p 1.0), 승패 차이 없음 (`<shared>/eval/jepa-diag-outer-20261109/`, [jepa_survey](report/survey/2026_11_08_19_26_jepa_survey.md) 10.5절)
- 20:47 — 결정·시작(사용자, Plan 모드 승인): 개선 모델(모델 2) 계획 승인 — 확정 5개 + 스칼라 보존·명령 이력 오염 수정·종족 균형 교사 혼합 등, 새 기능은 모두 스위치(기본 = 현재 모델). P0 측정·진단 도구·spec 초안 병렬 시작 ([model2](report/plan/2026_11_09_22_18_model2.md))
- 21:38 — 결정(사용자 확인): 모델 2 spec 개정 확정([001](spec/2026_10_15_22_24_001-system-architecture.md)·[010](spec/2026_10_15_22_24_010-game-protocol.md)·[030](spec/2026_10_15_22_24_030-observation.md)·[031](spec/2026_10_15_22_24_031-action-space.md)·[032](spec/2026_10_15_22_24_032-model-architecture.md)·[040](spec/2026_10_15_22_24_040-replay-dataset.md)·[041](spec/2026_10_15_22_24_041-imitation-learning.md)·[042](spec/2026_10_15_22_24_042-rl-training.md)·[050](spec/2026_10_15_22_24_050-evaluation.md)·[060](spec/2026_10_15_22_24_060-infra.md), 8c6ebf3·92bb47e), [AMB-032-12](spec/README.md) 등 AMB 10개 기본안 채택. Deferred: λ 값·코어 수정 세트·선택 상한. P2 구현 시작
- 22:02 — 결과·발견(결정 단위 측정 G0, `<shared>/eval/granularity-20261109/`): 러시 방어 중 12기 선택 상한 포화율 0~1.4%로 상한은 거의 막지 않음, 막음을 가르는 것은 4:30 전 벙커 완성과 탑승 시간. 결정 간격 약 0.7초(이전 0.45초 서술 정정). 권고: 행동 공간 유지, 사용자 확인 대기([AMB-031-7](spec/README.md))
- 22:14 — 결과·결정(코어 어텐션 조사 + 사용자 결정): 1층이 지금 step을 안 보는 것은 원인 아님, 손실은 게이트마다 분산. 결정: 행동 공간 유지([AMB-031-7](spec/README.md)), 코어 수정은 F1만(AMB-032-14), 벙커 레이블 가중 안 함. 산출물 `<shared>/eval/jepa-diag-20261109/core_attn/`
- 22:15 — 결과(모델 2 U7, [040-R23](spec/2026_10_15_22_24_040-replay-dataset.md)·[AMB-040-13](spec/README.md)): 운영형 반례 교사 판 400판(`config/plans/teacher_botgen_macro3.toml`) → `<shared>/datasets/replays/shards/botgen/teacher-20261109-macro3`(244,649 step, 디싱크 0), 4:00 전 풀을 본 판의 86.6%가 4:00 전 벙커 없음
- 22:18 — 새 리포트·결정 반영: [model2](report/plan/2026_11_09_22_18_model2.md)(모델 2 계획·결정·진행 기록), spec 갱신([AMB-031-7](spec/README.md)·AMB-032-14 Resolved, F2 제거, F1 배선). 남은 Deferred는 λ 값뿐
- 22:26 — 반영(모델 2 U1, [050-R9](spec/2026_10_15_22_24_050-evaluation.md)·[032-R25](spec/2026_10_15_22_24_032-model-architecture.md)·[041-R25](spec/2026_10_15_22_24_041-imitation-learning.md)): 진단 도구 `python -m uranium diag ...`와 죽은 채널 감시 `model/monitor.py` master 반영, 이전 진단과 같은 모양 재현
- 22:36 — 반영(모델 2 U3 기반, [032-R21](spec/2026_10_15_22_24_032-model-architecture.md)~R25·[041-R23](spec/2026_10_15_22_24_041-imitation-learning.md)·R25·[042-R21](spec/2026_10_15_22_24_042-rl-training.md)): `[model]` 스위치(기본 꺼짐), F1 배선, aux·monitor 설정 키, 마스크. 스위치를 모두 끈 모델은 Checkpoint 1과 비트 동일, 테스트 통과
- 23:00 — 반영(모델 2 U4a·U4b·U4d): 공간 ResBlock pre-activation([032-R22](spec/2026_10_15_22_24_032-model-architecture.md), 5dc14e4), F1 코어 입력 skip(032-R23, be9f37c), 헤드 우회로 + s_raw 139열(032-R21) master 반영. Python CPU 366 통과

## 2026-11-10
- 09:28 — 반영(모델 2 U4c+U5): aux 모듈(privileged 383 예측, JEPA k 1·8·32)·BC·RL 보조 손실·죽은 채널 감시·종족별 bc-eval·λ 비교 run 설정 7개, Rust 492·Python CPU 395 통과 ([032-R24·R25](spec/2026_10_15_22_24_032-model-architecture.md), [041-R23~R25](spec/2026_10_15_22_24_041-imitation-learning.md), [042-R21](spec/2026_10_15_22_24_042-rl-training.md))
- 09:42 — 반영(모델 2 U6): shard 스키마 7 `memory_action`, DAgger 모델 결정 기록·relabel `--memory-log`, `mix-fractions`, `config/bc_m2_mix_e1.toml`, botgen manifest 갱신(30 shard·2,573판, Checkpoint 1 시점 백업 보존) ([040-R22](spec/2026_10_15_22_24_040-replay-dataset.md), [041-R26·R28](spec/2026_10_15_22_24_041-imitation-learning.md))
- 10:56 — 장시간 실행 시작(모델 2 P3): λ 비교 run 7개(m2-lam-base·noaux·priv2x·jepa2x·nof1·nof1_noaux·seed2, 각 3,000 update)를 CUDA 호스트에서 차례로, 로그 `runs/m2_lambda/` ([041-R24](spec/2026_10_15_22_24_041-imitation-learning.md), [AMB-041-13a](spec/README.md))
- 16:06 — 결과(모델 2 P3 표본 end-to-end, Linux 호스트만): S1 → S2 → 종족 균형 혼합 → 기억 기록 DAgger 20판 → bc-eval → 측정 프로토콜 → RL 100 update까지 코드 버그 없음, 본학습 준비 완료(λ 결정 대기). 산출물 `<shared>/eval/m2-smoke-20261110/`, `<shared>/checkpoints/m2-smoke-*`
- 19:55 — 장시간 실행 끝(모델 2 P3): λ 비교 run 7개 모두 3,000 update 정상 종료(run당 약 75분), 체크포인트 `<shared>/checkpoints/m2-lam-<이름>/` ([041-R24](spec/2026_10_15_22_24_041-imitation-learning.md))
- 20:55 — 결과·발견(모델 2 P3 채점): 모든 λ·F1 차이가 모델 seed 잡음 폭(행동 NLL 0.014, HP R² 0.080) 안이거나 언저리, h_t HP R² 0.70~0.78, 공간 경보 0. 출력 `<shared>/eval/m2-lambda-20261110/` ([model2](report/plan/2026_11_09_22_18_model2.md) 7절, [041-R24](spec/2026_10_15_22_24_041-imitation-learning.md))
- 21:0x — 결정(사용자): 기본안 유지 — λ_priv 0.2, λ_jepa 0.1, F1 `core_skip` 켬, BC·RL 같음; AMB-041-13b·AMB-042-21 Resolved ([AMB-041-13b·AMB-042-21](spec/README.md), [041-R23·R29](spec/2026_10_15_22_24_041-imitation-learning.md), [042-R21](spec/2026_10_15_22_24_042-rl-training.md))
- 21:2x — 결정 반영: 경로 2 본학습 설정 `config/bc_m2.toml`·`bc_m2_e2.toml`·`rl_m2.toml` 추가, 메인 임시 결정 `model_seed` 1 ([041-R29](spec/2026_10_15_22_24_041-imitation-learning.md) ①②, [042-R21](spec/2026_10_15_22_24_042-rl-training.md))
- 21:32 — 장시간 실행 시작: 경로 2 BC `m2-s1`(S1 1 epoch, `config/bc_m2.toml`, CUDA 호스트, 963f864, 약 15~16시간 예상), 로그 `runs/m2_main/` ([041-R29](spec/2026_10_15_22_24_041-imitation-learning.md) ②)
- 21:4x — 결정(사용자): 경로 2 새 run 초기 가중치 seed `train.model_seed` = 1 고정(메인 임시 결정을 사용자가 확인) ([041-R29](spec/2026_10_15_22_24_041-imitation-learning.md) ①)
- 22:0x — 결정(사용자, AMB-050-4·5·6 Resolved): 경로 측정 진단에 사람 TvT·TvP 보관분 추가, 죽은 채널은 학습 로그 `sp_*`만, `m2-s1e2` 단서 진단은 최종 체크포인트만; `lineage.toml`·`protocol.py diag` 반영 ([050-R9](spec/2026_10_15_22_24_050-evaluation.md), [AMB-050-4](spec/README.md), [measurement_protocol](guide/measurement_protocol.md) 4·5절)
- 22:0x — 반영: 혼합 단계 측정 도구 둘 — `per_action.py` 교사 보관분 러시·운영형 분리 셈, `regression_gate.py` 테란·프로토스 회귀 게이트 ([041-R27](spec/2026_10_15_22_24_041-imitation-learning.md), [050-R10·AC5](spec/2026_10_15_22_24_050-evaluation.md))
- 22:4x — 결정·반영(리뷰 반영): 회귀 게이트 ②의 NLL은 행동 종류 NLL(`nll.action`) — 사용자 확인, 나머지 세부는 메인 임시 결정 ([050-R10](spec/2026_10_15_22_24_050-evaluation.md), [AMB-050-3](spec/README.md), [041-R27](spec/2026_10_15_22_24_041-imitation-learning.md))
- 23:0x — 결과(032-AC13 실측): 모델 2 코드(스위치 모두 끔)로 Checkpoint 1 `bc-dagger-r1/ckpt-1137` val500 재채점 top-1 72.19%, 헤드별 NLL 차이 5e-9 이하로 통과. `<shared>/eval/m2-ac13-20261110/` ([032-AC13](spec/2026_10_15_22_24_032-model-architecture.md), [model2](report/plan/2026_11_09_22_18_model2.md) 8.6절)

## 2026-11-11
- 00:0x — 발견·결정(사용자): `m2-s1` update 3,200부터 공간 인코더 경보(c3·r4·r5 dead ≥ 50%), 상수 신호 0; 학습 계속·감시 강화 ([041-R25](spec/2026_10_15_22_24_041-imitation-learning.md))
- 07:0x — 발견(r1 cue_diag 실패 원인): 사람 덤프가 느린 것은 덤프가 모든 헤드를 돌려 ROCm MIOpen이 새 행 수마다 커널을 찾기 때문(약 70초씩), ROCm forward는 실행마다 fp16 1 ULP 다름. 기록 bc250_1·asus `~/diagfix/` ([050-R9](spec/2026_10_15_22_24_050-evaluation.md))
- 07:3x — 반영(커밋 7f14f6a, master 미반영): `diag dump`는 trunk·행동 종류 헤드만, `--read-until`, 판 미리 디코드, `protocol.py diag --part gpu|cpu`; 임시 결정 probes는 Linux CPU 호스트 ([050-R9](spec/2026_10_15_22_24_050-evaluation.md))
- 07:40 — 장시간 실행 시작: Checkpoint 1 r1 `cue_diag` 재측정(`--part gpu` bc250_1 → `--part cpu` asus), 실패한 시도는 `<shared>/artifacts/diag/bc-dagger-r1__ckpt-1137.failed-1`로 보존 ([050-R9](spec/2026_10_15_22_24_050-evaluation.md))
- 09:07 — 장시간 실행 끝: r1 `cue_diag` 끝·`check-diag` 통과(gpu 64분·cpu 23분), 교사 미완성 풀 HP h_t R² 0.195·착공 0.617, 사람 P 0.035. `<shared>/artifacts/diag/bc-dagger-r1__ckpt-1137/` ([050-R9](spec/2026_10_15_22_24_050-evaluation.md))
- 09:1x — 발견·반영(커밋 7dc79b8): r1 교사 값이 U1·λ 덤프와 다른 원인은 shard 판 읽기 순서; 연결 폴더를 `by_name/<name>`으로, `check-diag`에 판 순서 지문 추가, 틀린 출력은 `<shared>/artifacts/diag/bc-dagger-r1__ckpt-1137.failed-2`로 보존 ([050-R9](spec/2026_10_15_22_24_050-evaluation.md))
- 09:1x — 발견(사용자 판단 필요): 사람 TvT 적 최대 관측 배럭 수가 거의 상수(6,557행 중 6,497이 1)라 모든 R² 음수, 추출 오류 아님; 프로토스는 값 있음 ([050-R9](spec/2026_10_15_22_24_050-evaluation.md), [jepa_survey](report/survey/2026_11_08_19_26_jepa_survey.md) 10.2.3절)
- 09:15 — 장시간 실행 시작: r1 `cue_diag` 재측정(7dc79b8, 교사 판 순서 = λ, gpu bc250_1 → cpu asus) ([050-R9](spec/2026_10_15_22_24_050-evaluation.md))
- 10:43 — 장시간 실행 끝: r1 `cue_diag` 재측정 끝·`check-diag` 통과, 교사 값 U1 대조와 같음(풀 HP h_t R² 0.227, 착공 0.642, 반사실 dP(벙커) +0.0102), 사람 T −0.026·P 0.035. `<shared>/artifacts/diag/bc-dagger-r1__ckpt-1137/` ([050-R9](spec/2026_10_15_22_24_050-evaluation.md))
- 10:5x — 결정(사용자 확인, AMB-050-4 후속): 사람 TvT 배럭 수 단서는 "이 데이터로 측정 불가"로 기록, 지점 비교는 TvP만; spec·`lineage.toml`·측정 프로토콜 문서 반영 ([050-R9](spec/2026_10_15_22_24_050-evaluation.md), [AMB-050-4](spec/README.md), [measurement_protocol](guide/measurement_protocol.md))
- 10:5x — 반영(리뷰 반영): `ChosenGames` 스레드 panic 처리, `action_seq` 동등성 테스트 확장, cpu 단계는 Linux CPU 호스트(메인 임시 결정), 실패하던 `test_bc_eval_out_writes_the_report_032_r20` 수정 ([050-R9](spec/2026_10_15_22_24_050-evaluation.md))
- 11:03 — 장시간 실행 시작(AMB-050-5): Checkpoint 1 경로 `cue_diag` 14지점(GPU bc250_1·CPU asus 연쇄, ee92359, 지점당 약 65분), s1e1 HP R² 0.290. 스크립트·로그 `<shared>/eval/ck1-lineage-diag-20261110/` ([050-R9](spec/2026_10_15_22_24_050-evaluation.md), [AMB-050-5](spec/README.md))
- 13:24 — 장시간 실행 끝: `m2-s1` 39,096 update 정상 종료(약 1.45초/update), 공간 경보 318회·완전 붕괴 없음. 로그 `runs/m2_main/s1.log` ([041-R29·R25](spec/2026_10_15_22_24_041-imitation-learning.md))
- 13:3x — 장시간 실행 시작: `m2-s1e2`(`config/bc_m2_e2.toml`, `--init m2-s1`, CUDA 호스트, ee92359, 약 16시간 예상), 측정·채점은 bc250_2(사용자 허락)·asus ([041-R29](spec/2026_10_15_22_24_041-imitation-learning.md) ②)
- 14:5x — 새 리포트(사용자 결정: 모델 2 경로 그대로, 계획 가능한 JEPA를 나란히 설계, spec 변경·구현 없음): 행동 조건 건너뛰기 세계 모델 + 결정 순간 재정렬 제안, 오프라인 가능성 시험 설계, 사용자 결정 8개 ([jepa_planning](report/plan/2026_11_11_14_56_jepa_planning.md))
- 15:2x — 중단(사용자 요청: bc250_1 장비 성능 측정): Checkpoint 1 경로 cue_diag GPU 연쇄를 mix-e1 GPU 부분 뒤 멈춤, 끝난 지점 5개, 남은 10지점은 bc250_1 반환 뒤
- 15:3x — 정리(사용자 요청, CLAUDE.md "작업 산출물 위치"): 세 Linux 호스트 홈의 작업 로그·생성 파일·임시 폴더를 프로젝트 밖 작업 폴더(현재 `local/work/`)로 옮김(worktree는 `git worktree move`), 이후 모든 호스트 산출물은 이 폴더에
- 16:30 — 결과: 경로 2 s1e1(`m2-s1/ckpt-39096`) val500 top-1 71.75%·NLL 0.889(Checkpoint 1 71.82%·0.894), 풀 HP R² 0.508(Checkpoint 1 0.290), 착공 0.809, 사람 P 0.159. `<shared>/eval/m2-lineage/` ([050-R9](spec/2026_10_15_22_24_050-evaluation.md), [041-R29](spec/2026_10_15_22_24_041-imitation-learning.md) ⑦)
- 16:4x — 결정·설계(사용자): JEPA 목적은 계획·탐색·조종, 경로 2는 계속하고 계획형 JEPA 설계 병행; S0 가능성 측정 도구 `diag plan-dump`·`plan-s0`(bbad17c) ([jepa_planning](report/plan/2026_11_11_14_56_jepa_planning.md))

## 2026-11-12
- 00:3x — 재투입(사용자 재튜닝 뒤): bc250_1 hostd calibrate 최대 6판, S0 추출 단독 실행으로 안정성 확인; 11-11 21:45 bc250_1 하드행 뒤 사용자가 bc250_1·bc250_2 재부팅
- 01:59 — 결과·발견(jepa_planning S0): 117만 행·1,343판, 행동 조건 이득 0·방향 확인 M2b 0.33~0.38 → "파이프라인 확인", 레이블 문제 세 가지 발견. `<shared>/eval/jepa-plan-feas-20261111/s0/`, 점검 `local/work/s0_check/` ([jepa_planning](report/plan/2026_11_11_14_56_jepa_planning.md))
- 05:26 — 장시간 실행 끝/시작: `m2-s1e2` 39,094 update 정상 종료 → `m2-s2e2`(`--s2 --init m2-s1e2`, CUDA 호스트) 시작, 로그 `local/work/m2_main/` ([041-R29](spec/2026_10_15_22_24_041-imitation-learning.md) ②)
- 06:0x — 결정(사용자): S0는 세 문제를 고쳐 재측정, 같은 프레임 명령 체인 레이블 밀림은 교사 혼합 전에 영향을 재고 결정(그때까지 혼합 보류)
- 09:3x — 발견·결정(사용자, AMB-041-19): 교사 botgen 같은 프레임 명령 체인 2번째부터 레이블이 효과가 보이는 관측에 놓임(Checkpoint 1이 이를 읽음); 로더 규칙(a′, 체인 시작 관측) 채택. 분석 `local/work/label_chain/`, 구현 526cf89 ([041-R30](spec/2026_10_15_22_24_041-imitation-learning.md), [040-R4](spec/2026_10_15_22_24_040-replay-dataset.md), [AMB-041-19](spec/README.md))
- 10:36 — 장시간 실행 끝/시작: `m2-s2e2`(경로 2 BC 기준선) 11,387 update 정상 종료 → 10:37 교사 혼합 `m2-mix-e1`~e7 연쇄 시작(CUDA 호스트), 드라이버 `local/work/m2_main/mix_chain.sh` ([041-R29](spec/2026_10_15_22_24_041-imitation-learning.md) ②③)
- 12:1x — 결정(사용자): 경로 2 교사 혼합 연장 판정은 체인 시작 관측 점수로(문턱 70%·50% 그대로), 현재 관측 점수는 기록만; 근거 m2-mix-e1 러시 벙커 top-1 체인 시작 35.7%·현재 0.7% ([041-R30·R29](spec/2026_10_15_22_24_041-imitation-learning.md) ⑤·③)
- 12:2x — 설계·시험(사용자 결정: 행동의 인과 효과를 분기 게임으로): 강제 재생 방식 권장, T0 같은 seed 재현·`branch_replay` 테스트로 분기 확인, 러시 파일럿 기본안 1,280판·사용자 결정 7개. 출력 `local/work/branch_test/` ([jepa_planning](report/plan/2026_11_11_14_56_jepa_planning.md) 7절)
- 12:3x — 결정·spec 초안(사용자 결정): 분기 게임 B1(T0 `zerg_rush`, 8:00)·B2, `m2-s1/ckpt-39096` stochastic, 새 spec 051 Draft, AMB-051-1~5 Open·AMB-051-6 Deferred ([051](spec/2026_11_12_12_39_051-branch-games.md), [AMB-051-1~6](spec/README.md), [jepa_planning](report/plan/2026_11_11_14_56_jepa_planning.md) 7.13·7.14절)
- 17:33 — 장시간 실행 끝·판정: `m2-mix-e1`~e7 정상 종료, 체인 시작 러시 벙커 top-1 35.7 → 66.1%, e6→e7 +1.4%p < 2%p라 연장 없이 e7에서 멈춤, DAgger 출발점 `m2-mix-e7/ckpt-1212`. 표 `<shared>/eval/m2-lineage/mix_table.md` ([041-R29·R30](spec/2026_10_15_22_24_041-imitation-learning.md) ③·⑤)
- 23:54 — 장시간 실행 끝: 경로 2 s2e2 지점 측정(게임 1,392판, val500, 교사 hold-out, cue_diag) 끝, 혼합 e1~e7 val500은 채점기가 이어 감 ([050-R9](spec/2026_10_15_22_24_050-evaluation.md), [041-R29](spec/2026_10_15_22_24_041-imitation-learning.md) ⑦)

## 2026-11-13
- 02:13 — 장애·재실행: 11-12 23:07경 bc250_2가 mix-e5 val500 중 멈춤 → 드라이버 재시작 뒤 재채점 끝(top-1 72.21%), 첫 시도는 `<shared>/artifacts/bc_e2_val/val500/m2-mix-e5__ckpt-1187.failed-1.*`로 보존
- 02:37 — 결과: 경로 2 s2e2 test(대상 맵 1,199게임, CUDA bf16) top-1 72.14%·NLL 0.868(Checkpoint 1 72.15%·0.871). `<shared>/artifacts/bc_main_eval/test_cuda/m2-s2e2.json` ([050-R9](spec/2026_10_15_22_24_050-evaluation.md), [041-AC1](spec/2026_10_15_22_24_041-imitation-learning.md))
- 09:42 — 결과(s2e2 게임, 모델 2 대 Checkpoint 1): T0 300 46.3% 대 39.3%(p 0.099), T1 300 8.0% 대 4.3%, AC1 stochastic lf0 80.0% 대 68.3%(p 0.009); p < 0.01은 AC1 lf0뿐, 러시 막음은 그대로. 표 `local/work/m2_main/s2e2_tables.md` ([050-R9](spec/2026_10_15_22_24_050-evaluation.md))
- 16:54 — 결정 반영(사용자 2026-11-12 승인): spec 051 Approved, AMB-051-1~5 기본안 채택(B1 1,280판, 8:00까지), [020](spec/2026_10_15_22_24_020-env-orchestration.md)·[031](spec/2026_10_15_22_24_031-action-space.md)·[032](spec/2026_10_15_22_24_032-model-architecture.md)·[041](spec/2026_10_15_22_24_041-imitation-learning.md)·[050](spec/2026_10_15_22_24_050-evaluation.md) 문구 반영 ([051](spec/2026_11_12_12_39_051-branch-games.md), [AMB-051-1~5](spec/README.md))
- 17:00 — 새 리포트 절·발견: model2 9절 "모델 2 경로 지점별 비교" — val500 차이 작음, 풀 HP R² 0.51~0.56(Checkpoint 1 0.29~0.31)이나 반사실 개입 효과는 작음; 회귀 게이트 mix-e7 ③a 실패(사용자 보고 대상). 표 `local/work/m2_report/` ([model2](report/plan/2026_11_09_22_18_model2.md) 9절, [050-R9·R10](spec/2026_10_15_22_24_050-evaluation.md))
- 시각 없음 — 결정·발견·준비(사용자 결정: 경로 2 DAgger r1은 `m2-mix-e7/ckpt-1212`에서, 교사 + dagger 몫은 Checkpoint 1 r1 실현 몫): 재계산 59.54%, plan `config/plans/dagger_m2_r1.toml`·설정 `config/bc_m2_dagger_r1.toml` 준비, `mix-fractions --teacher-share` 추가 ([041-R29·R21·R22](spec/2026_10_15_22_24_041-imitation-learning.md) ④)
- 17:1x — 결정·spec 승인(사용자): 리플레이 약점 네 가지(채취 효율, 서플라이 막힘, 자원 묵힘, 초반 러시 전투)를 세션 지표로 재는 새 spec 052 Approved, AMB-052-2·6·10은 사용자 결정·나머지 기본안, 보고만 ([052](spec/2026_11_13_17_19_052-ops-metrics.md), [AMB-052-1~10](spec/README.md))
- 17:27 — 장시간 실행 시작: 경로 2 DAgger r1 판 600판(`m2-mix-e7/ckpt-1212` stochastic, `memory_log`) `<shared>/eval/dagger-m2-r1-20261113/`, 실행 순서 `<shared>/eval/m2-lineage/scripts/dagger_r1.sh` ([041-R29·R22·R28](spec/2026_10_15_22_24_041-imitation-learning.md) ④)
- 17:29 — spec 반영(사용자 결정 2026-11-12): 경로 2 DAgger 교사 + dagger 몫 = Checkpoint 1 r1 실현 몫 0.5954 ± 3%p, R29 ④ dagger shard 문구 수정 ([041-R29·R22·R26](spec/2026_10_15_22_24_041-imitation-learning.md))
- 17:31 — 발견·결정(사용자): 재레이블 없는 build는 분기 시점 행이 우연히만 있음 → "결정마다 행 추가"(관측·기억 전용 스텝, BC 학습 제외) ([051-R13](spec/2026_11_12_12_39_051-branch-games.md) ④, [041-R28](spec/2026_10_15_22_24_041-imitation-learning.md))
- 시각 없음 — 구현·발견: 052 지표 구현(`OpsTracker`, `uranium-replay opsdump`, `tables.py` 두 표), opsdump 판당 약 3~5초, 메인 임시 결정 다수; 표본 운영형 판에서 SCV가 맵 중앙 미네랄을 캐 `stranded_scv_s` 264초 ([052-R1~R9·AC1·AC3](spec/2026_11_13_17_19_052-ops-metrics.md))
- 시각 없음 — 결정 정정·리뷰 반영(별도 리뷰어): `bunker_load_s`를 'E 관측에 마린이 타 있으면 0'으로 좁힘(앞 정의 철회, 사용자 확인 대상), opsdump 실패 처리·캐시 버전 키; Checkpoint 1 168판 1분 53초·실패 0, 캐시 `local/work/ops052/cache` ([052-R5·R7](spec/2026_11_13_17_19_052-ops-metrics.md))
- 18:10 — 결정·spec 개정(사용자): 분기 판 결정 표본 d8·d32는 명령을 실제로 보낸 결정만 셈(시각 표본은 그대로) ([051-R9·R11·R13](spec/2026_11_12_12_39_051-branch-games.md))
- 18:30 — 확인(051-AC6): Linux CPU 호스트 작은 end-to-end에서 분기 8판 모두 유효·다이제스트 일치, 같은 판 batch 1이면 비트 일치. `<shared>/eval/branch-ac6-20261113/`, 작업 사본 `local/work/branch-ac6/` ([051-AC6](spec/2026_11_12_12_39_051-branch-games.md))
- 18:38 — 결정·spec 개정(사용자, AMB-051-1 보완): B1 두 시점이 같은 결정에 걸리면 둘째 시점을 첫 시점 + 30초 이상 결정으로(`after_rule`·`after_s`) ([051-R3·R11](spec/2026_11_12_12_39_051-branch-games.md), [AMB-051-1](spec/README.md))
- 19:08 — 장시간 실행 끝·준비: DAgger r1 판 600판 끝 → 재레이블(쓴 판 592) → 교사 + dagger 59.53%, s2 1,285로 `config/bc_m2_dagger_r1.toml` 채움. 산출물 `<shared>/eval/dagger-m2-r1-20261113/`, `<shared>/datasets/replays/shards/dagger_m2/r1` ([041-R29·R22·R28](spec/2026_10_15_22_24_041-imitation-learning.md) ④)
- 19:14 — 장시간 실행 시작: `m2-dagger-r1`(1,285 update, data seed 13, CUDA 호스트, f2ce0d9), 로그 `local/work/m2_main/dagger_r1.{log,out}` ([041-R29·R22](spec/2026_10_15_22_24_041-imitation-learning.md) ④)
- 20:23 — 장시간 실행 끝: `m2-dagger-r1` 정상 종료, `<shared>/checkpoints/m2-dagger-r1/ckpt-1285` ([041-R29·R22](spec/2026_10_15_22_24_041-imitation-learning.md) ④)
- 21:46 — 장시간 실행 시작: 경로 2 r1 지점 측정 `m2-dagger-r1/ckpt-1285`(val500·진단·per_action, 게임 job 3개) ([050-R9·R10](spec/2026_10_15_22_24_050-evaluation.md))
- 21:49 — 준비: DAgger r2 plan `config/plans/dagger_m2_r2.toml`(300판)·설정 `config/bc_m2_dagger_r2.toml`(data seed 14), 실행 순서 `<shared>/eval/m2-lineage/scripts/dagger_r2.sh` ([041-R29·R22·R28](spec/2026_10_15_22_24_041-imitation-learning.md) ④)
- 21:57 — 장시간 실행 시작: DAgger r2 판 300판(`m2-dagger-r1/ckpt-1285`) `<shared>/eval/dagger-m2-r2-20261113/`; 학습은 r1 첫 시험 판정 파일에 `proceed`가 있어야 시작 ([041-R29·R22·R28](spec/2026_10_15_22_24_041-imitation-learning.md) ④)
- 22:15 — 결과·리포트: 052 리허설과 기존 판 다시 셈, 052-AC2 통과 판정(라이브·재생 값 128/128 일치), 러시 방어에서 모델이 교사보다 크게 뒤짐. 산출물 `<shared>/eval/ops052/`, `local/work/ops052/report/` ([ops_metrics](report/experiment/2026_11_13_22_14_ops_metrics.md), [052-AC2·R7·R8](spec/2026_11_13_17_19_052-ops-metrics.md))
- 22:40 — 결정·spec 개정(사용자 "기준 보완 후 통과", AMB-052-9 보완): 052-AC2 셋째 조건을 ≤ 8 × 창 안 최대 SCV 수로 바꿔 다시 돌려 128/128 → 통과 ([052-AC2](spec/2026_11_13_17_19_052-ops-metrics.md), [AMB-052-9](spec/README.md), [ops_metrics](report/experiment/2026_11_13_22_14_ops_metrics.md) 3.1절)
- 23:12 — 결과: r1 첫 시험 셋 중 둘(정찰 64·62/64, 벙커 지시 28/40, 이른 막음 19/64) → 2라운드, 판정 파일 `<shared>/eval/m2-lineage/r1_first_test.verdict` = proceed; T1 21/300 ([041-R22·R29](spec/2026_10_15_22_24_041-imitation-learning.md) ④)
- 23:20 — 장시간 실행 끝·준비: DAgger r2 판 300판 끝 → 재레이블(쓴 판 299) → 교사 + dagger 59.57%, s2 1,408로 `config/bc_m2_dagger_r2.toml` 채움 ([041-R29·R22·R28](spec/2026_10_15_22_24_041-imitation-learning.md) ④)
- 23:24 — 장시간 실행 시작: `m2-dagger-r2`(s2 1,408, data seed 14, CUDA 호스트, 0aa11b9) ([041-R29·R22](spec/2026_10_15_22_24_041-imitation-learning.md) ④)
- 23:30 — 구현 반영: 분기 게임 U1~U6 master 반영(단위별 리뷰 + 통합 리뷰), B1 본실험은 경로 측정 틈에 ([051-R1~R15·AC1~AC6](spec/2026_11_12_12_39_051-branch-games.md), [020-R22](spec/2026_10_15_22_24_020-env-orchestration.md), [031-R4](spec/2026_10_15_22_24_031-action-space.md), [032-R16](spec/2026_10_15_22_24_032-model-architecture.md), [041-R28](spec/2026_10_15_22_24_041-imitation-learning.md))
- 23:30 — 장시간 실행 기록(보충): 혼합 e7 퇴보 판정 게임 `<shared>/eval/m2-mix-e7-t0tp-20261113/` 17:27~19:14, DAgger r1 지점 게임 job 3개 21:47~23:12 ([050-R9·R10](spec/2026_10_15_22_24_050-evaluation.md))
- 23:30 — 결정(사용자): 052에서 가장 큰 약점은 초반 러시 방어, 반영 방법은 DAgger r1·r2 지점에서 다시 잰 뒤 정하고 052 지표는 계속 보고만 ([052-R9](spec/2026_11_13_17_19_052-ops-metrics.md))
- 23:3x — 결과: 회귀 게이트 e7 FAIL(프로토스 T0 승 10 → 2/32)·r1 FAIL(10 → 1/32), 사용자 보고 대상. 원본 `local/work/m2_r1/gate_r1.json`, `local/work/m2_report/gate_e7.json` ([050-R10](spec/2026_10_15_22_24_050-evaluation.md))

## 2026-11-14
- 00:06 — 결과: 단서 진단 h_t 풀 HP R² e1 0.625·e7 0.602·r1 0.553(Checkpoint 1 0.303·0.199·0.227), 반사실 개입 Δ는 e1 → r1로 줄어듦 ([050-R9](spec/2026_10_15_22_24_050-evaluation.md))
- 00:1x — 결과: r1 러시 128판 벙커 탑승 56%(s2e2 6%), SCV 손실 중앙값 13(s2e2 18), 일꾼당 수입 교사 수준. 표 `local/work/m2_r1/ops_r1_tables.md` ([052-R8](spec/2026_11_13_17_19_052-ops-metrics.md))
- 00:35 — 장시간 실행 끝: `m2-dagger-r2` 정상 종료(`ckpt-1408`, 0aa11b9), r2 지점 측정 01:1x 시작 ([041-R29·R22](spec/2026_10_15_22_24_041-imitation-learning.md) ④)
- 01:07 — 장시간 실행 시작: 경로 2 r2 지점 측정 `m2-dagger-r2/ckpt-1408`(게임 job 3개, val·per_action·진단은 Linux ROCm 호스트) ([050-R9·R10](spec/2026_10_15_22_24_050-evaluation.md))
- 01:10 — 발견(사용자 결정 "r2 계속 + 원인 분석 병행"): 프로토스전 퇴보 주원인은 저그 전용 교사 빌드가 종족 무관 '교사 모드'로 옮은 것(TvP 교사 모드 판 s2e2 0 → r1 30/32, 그 판 1/49승); 정찰 판정 바이트 어긋남 부수 발견. 분석 `local/work/protoss_regress/` ([050-R10](spec/2026_10_15_22_24_050-evaluation.md), [041-R26·R29](spec/2026_10_15_22_24_041-imitation-learning.md), [021-R8](spec/2026_10_15_22_24_021-opponent-pool.md))
- 01:3x — 결과: r2 첫 시험 막음 합 76(r1 57), 운영형 CC ≤ 5:00 15/40 미달 → r3 없이 선택(Checkpoint 1 r2와 같은 갈래) ([041-R29·R22](spec/2026_10_15_22_24_041-imitation-learning.md) ④)
- 02:05 — 결과: r2 val500 top-1 72.03%, 종족별 NLL(테란 +0.0231·프로토스 +0.0177)·4:00 전 p(벙커) 모두 게이트 실패 ([050-R10](spec/2026_10_15_22_24_050-evaluation.md) ②·③a)
- 02:1x — 결정(사용자, 프로토스전 퇴보 수정): 테란·프로토스전 자기 판 증류, 테란·프로토스전 교사 새로 만들기, RL에 맡기기를 모두 진행(spec 초안 → Plan 승인 → 구현), 정찰 지표는 고치고 기존 값도 함께 기록
- 08:12 — 구현·재집계(사용자 결정 "고치고 양쪽으로 기록"): 기존 `scout_3m`은 두고 `scout_3m_fixed` 추가(문턱 변경 없음), 수정 값 경로 2 64·64, Checkpoint 1 mix-e7 58·58·mix-e10 50·49, T0 31·34; BC 계열 정찰은 거의 모두 SCV 땅 우클릭. 표 Linux CPU 호스트 C `local/work/scout_fix/` ([050-R9](spec/2026_10_15_22_24_050-evaluation.md), [041-R22·R29](spec/2026_10_15_22_24_041-imitation-learning.md), [measurement_protocol](guide/measurement_protocol.md))
- 08:15 — 발견(리뷰 지적): 같은 바이트 어긋남이 git 미추적 옛 스크립트(`runs/dagger/test_table.py`, `runs/adv8/cmds.py`)에도 있음, runs/는 고치지 않고 그 값을 인용한 리포트 목록을 기록 ([050-R9](spec/2026_10_15_22_24_050-evaluation.md), [m6_rl_stall_advice8](report/advice/2026_11_07_03_20_m6_rl_stall_advice8.md) 1-2절, [checkpoint1](report/release/2026_11_08_12_21_checkpoint1.md), [m6_rl_stall_advice10](report/advice/2026_11_07_16_29_m6_rl_stall_advice10.md) 1절)
- 08:20 — 새 리포트·spec 초안(사용자 결정): protoss_fix_plan, spec 044(자기 판 증류)·022(테란·프로토스전 교사 `terran_tp`) Draft, AMB-044-1~11·AMB-022-1~7 Open ([protoss_fix_plan](report/plan/2026_11_14_08_18_protoss_fix_plan.md), [044](spec/2026_11_14_08_18_044-self-game-distillation.md), [022](spec/2026_11_14_08_18_022-matchup-teacher.md), [AMB-044-1~11·AMB-022-1~7](spec/README.md))
- 08:30 — 결정·spec 확정(사용자): AMB-044-1~11·AMB-022-1~7 모두 기본안으로 Resolved, spec 022·044 Approved, 관련 spec 개정 반영, 구현 단위 U1~U11 ([044](spec/2026_11_14_08_18_044-self-game-distillation.md), [022](spec/2026_11_14_08_18_022-matchup-teacher.md), [AMB-044-1~11·AMB-022-1~7](spec/README.md), [protoss_fix_plan](report/plan/2026_11_14_08_18_protoss_fix_plan.md) 8절)
- 09:03 — 발견·수정·재판정: 명령 스트림 분석이 첫 테란 플레이어를 읽어 TvT에서 상대 명령을 셈 → 수정, 테란전 `bunker_order_4m` 칸이 e7·r1·r2 모두 PASS → FAIL(전체 FAIL 그대로). `local/work/gate_recheck/` ([050-R10](spec/2026_10_15_22_24_050-evaluation.md) ③b)
- 09:05 — U9 구현: RL 종족별 참조 KL `[kl.reference]`, `train.csv` `kl_ref_<종족>` 4열, `bench rl --references K`; 표가 없으면 비트 동일 ([044-R9·AC4](spec/2026_11_14_08_18_044-self-game-distillation.md), [042-R22](spec/2026_10_15_22_24_042-rl-training.md) ①)
- 09:05 — spec 반영·준비(사용자 결정): R29 ④ 경로 2 예외(r2 뒤에도 DAgger r3), plan `config/plans/dagger_m2_r3.toml`·설정 `config/bc_m2_dagger_r3.toml`(data seed 15), 실행 순서 `<shared>/eval/m2-lineage/scripts/dagger_r3.sh` ([041-R29·R22·R28](spec/2026_10_15_22_24_041-imitation-learning.md) ④)
- 09:12 — 정정: 10:2x "결정(남은 질문 일괄)" 줄의 실제 기록 시각은 09:02(커밋 3188ba0), 같은 날 x 표기 시각도 대략값
- 09:12 — 결정(사용자): 경로 2에도 r4 규칙 적용, [022](spec/2026_11_14_08_18_022-matchup-teacher.md) 교사가 합격선 미달이면 방향 안 수치 조정 반복, 증류 d1·d2 모두 실패하면 T0 프로토스가 높은 쪽에서 RL, B1은 장비가 빌 때마다 ([041-R29](spec/2026_10_15_22_24_041-imitation-learning.md) ④, [051-R14](spec/2026_11_12_12_39_051-branch-games.md))
- 09:14 — 장시간 실행 시작: DAgger r3 판 300판(`m2-dagger-r2/ckpt-1408`) `<shared>/eval/dagger-m2-r3-20261114/`, 학습은 `r2_first_test.verdict`의 `proceed` 필요 ([041-R29·R22·R28](spec/2026_10_15_22_24_041-imitation-learning.md) ④)
- 09:15 — U10 구현: `config/rl_m2_race.toml`(로비 종족 테란 24·프로토스 28·저그 28·Random 20, `[kl.reference]`), 사다리 항목 `race = "Random"` 허용, 메인 임시 결정 3가지 ([044-R10·R11·AC7](spec/2026_11_14_08_18_044-self-game-distillation.md), [042-R22](spec/2026_10_15_22_24_042-rl-training.md), [021-R7](spec/2026_10_15_22_24_021-opponent-pool.md))
- 09:46 — 결정(사용자, 단계 끝 일괄 질문): 증류 레이블 마스크·직전 실패 명령 규칙, `r_value_stages = ["C0", "C1"]`, T1 몫, r4 방식 확정; 021·042·044 표기를 "사용자 확인 2026-11-14"로 ([044-R3](spec/2026_11_14_08_18_044-self-game-distillation.md), [021](spec/2026_10_15_22_24_021-opponent-pool.md), [042](spec/2026_10_15_22_24_042-rl-training.md))
- 10:03 — U6 구현: 테란·프로토스전 교사 `terran_tp`(`crates/uranium-scripted/src/tp.rs`), 스모크 64판 T0 테란 32/32·프로토스 22/32승. 산출물 `local/work/u6tp/` ([022-R1~R10·AC1](spec/2026_11_14_08_18_022-matchup-teacher.md), [021-R9](spec/2026_10_15_22_24_021-opponent-pool.md))
- 10:17 — 장시간 실행 끝·준비·시작: DAgger r3 판 300판 끝 → 재레이블(쓴 판 297) → 교사 + dagger 59.59%, s2 1,528로 설정 채움 → `m2-dagger-r3`(data seed 15, CUDA 호스트) 시작 ([041-R29·R22·R28](spec/2026_10_15_22_24_041-imitation-learning.md) ④)
- 10:2x — 결정(사용자, 남은 질문 일괄; 실제 기록 시각 09:02, 09:12 정정 참고): 경로 2는 r2 판정에도 DAgger r3를 돌고 S는 r3 뒤 선택, 증류·RL·052 세부 확정(수정 RL run 이름 `m2-rl-fix-<번호>` 등) ([041-R29](spec/2026_10_15_22_24_041-imitation-learning.md) ④, [044-R1·R2·R5·R9·R10](spec/2026_11_14_08_18_044-self-game-distillation.md), [052](spec/2026_11_13_17_19_052-ops-metrics.md))
- 10:29 — 실행 시작(022-AC2 공식 점검 1회차): `config/plans/teacher_tp_check.toml` 208판, 교사 5f78c49. 산출물 `local/work/teacher_tp_check/check1/` ([022-AC2](spec/2026_11_14_08_18_022-matchup-teacher.md))
- 10:31 — U7·U8 리뷰 후속: 022 교사 botgen build·`--relabel terran_tp`에 `--games <games.jsonl>`와 거부 규칙 추가, 메인 임시 결정(사용자 확인 대기)으로 spec 반영 ([022-R12·R2·AC3](spec/2026_11_14_08_18_022-matchup-teacher.md), [040](spec/2026_10_15_22_24_040-replay-dataset.md), [041-R22](spec/2026_10_15_22_24_041-imitation-learning.md), [044-R8](spec/2026_11_14_08_18_044-self-game-distillation.md))
- 10:51 — 실행 끝·결과(022-AC2 1회차): T0 프로토스 39승 3무 22패(합격선 44/64 미달), T0 테란 59/2/3 합격, 크래시 0 ([022-AC2](spec/2026_11_14_08_18_022-matchup-teacher.md))
- 11:28 — 조정(사용자 결정: 미달이면 정해진 방향 안 수치 조정): 개발 seed 비교로 프로토스전 앞마당 탱크 3기 뒤·D 마린 어택땅 없음을 채택 후보로 ([022-R5·R6](spec/2026_11_14_08_18_022-matchup-teacher.md))
- 11:31 — 장시간 실행 끝: `m2-dagger-r3` 정상 종료(`ckpt-1528`, 7da1281); 사용량 한도로 에이전트 작업 11:3x~13:00 멈춤 ([041-R29·R22](spec/2026_10_15_22_24_041-imitation-learning.md) ④)
- 13:06 — 실행 시작(022-AC2 2회차): 같은 plan 208판, 교사 3007e23 + 조정 후보(557db49). 산출물 `local/work/teacher_tp_check/check2/` ([022-AC2](spec/2026_11_14_08_18_022-matchup-teacher.md))
- 13:09 — 장시간 실행 시작: 경로 2 r3 지점 측정 `m2-dagger-r3/ckpt-1528`(게임 job 3개, e59df01) ([050-R9·R10](spec/2026_10_15_22_24_050-evaluation.md))
- 13:35 — 실행 끝·결과(022-AC2 2회차): T0 프로토스 51승 4무 9패·테란 59승 2무 3패 → 합격선 통과, 크래시 0 ([022-AC2](spec/2026_11_14_08_18_022-matchup-teacher.md))
- 13:53 — 결과·결정(사용자): 022 교사 점검 1회차 미달 → 조정 → 2회차 합격(결과 `local/work/teacher_tp_check/`); 교사 임시 동작 4가지 유지, `--games` 확정, 교사 라운드는 증류 뒤 한 번·300판·교사 65 / 사람 35 ([022-AC2](spec/2026_11_14_08_18_022-matchup-teacher.md), [044-R6](spec/2026_11_14_08_18_044-self-game-distillation.md))
- 13:55 — 발견: 첫째 Linux ROCm 호스트가 네트워크에서 빠짐(14:47 사용자 전원 재시작으로 복구), r3 남은 판은 CUDA 호스트 hostd에서, val500 첫 시도는 `.failed-1`로 보존
- 13:58 — 실행 시작(022-R12 교사 botgen): `config/plans/teacher_botgen_tp.toml` 800판(10분 상한), 교사 97798a0, Linux CPU 호스트 3슬롯 ([022-R12](spec/2026_11_14_08_18_022-matchup-teacher.md))
- 14:0x — 결과: r3 첫 시험 막음 합 61(r2 76, −15) → r4 없음, r3 뒤 S 선택 ([041-R29](spec/2026_10_15_22_24_041-imitation-learning.md) ④)
- 15:13 — 실행 끝(022-R12 botgen): 800판 완료·실패 0, 대부분 10분 상한 무승부. 원본 `<shared>/datasets/replays/raw/botgen/teacher-tp-20261114/teacher_botgen_tp/` ([022-R12](spec/2026_11_14_08_18_022-matchup-teacher.md))
- 15:59 — build 끝: `<shared>/datasets/replays/shards/botgen/teacher-tp-20261114`(800 시점·616,348 스텝, 학습 설정 미반영); 022-AC3 재레이블 일치는 3판 중 1판만 ≥ 95% ([022-R12·AC3](spec/2026_11_14_08_18_022-matchup-teacher.md))
- 16:10 — 장시간 실행 끝·결과: r3 측정 끝 — TvZ 30/32, T0 테란·프로토스 7·1/32, T1 33/300, val500 72.01%, 회귀 게이트 FAIL. 표 `local/work/m2_r3/` ([050-R9·R10](spec/2026_10_15_22_24_050-evaluation.md))
- 16:1x — 결과: S 선택 — r1·r2·r3 모두 자격 통과, 막음 합 57·76·61 → S = r2(`m2-dagger-r2/ckpt-1408`) ([041-R29](spec/2026_10_15_22_24_041-imitation-learning.md) ④)
- 16:15 — 결정(사용자, 일괄 질문): 증류 d1은 Linux ROCm 호스트에서, 022-AC3 미달은 원인을 고쳐 점검·botgen 다시(v1 보존), t1 통과 시 D = t1, t1의 이전 증류 데이터 사용은 Fable 자문 뒤(사용자 위임), run 이름 `m2-teacher-t1` 등 10가지 ([022-AC3](spec/2026_11_14_08_18_022-matchup-teacher.md), [044-R6](spec/2026_11_14_08_18_044-self-game-distillation.md))
- 16:16 — 결정(사용자 위임, Fable 자문): t1 학습에 이전 증류 데이터(`distill_m2`)는 넣지 않음(교사 65 / 사람 35) ([044-R8](spec/2026_11_14_08_18_044-self-game-distillation.md) ③)
- 16:22 — 준비(사용자 결정: S = r2): 증류 d1 plan `config/plans/distill_m2_d1.toml`(600판, `m2-dagger-r2/ckpt-1408`), 모두 Linux 호스트에서, 600판 전에 표본 end-to-end ([044-R1](spec/2026_11_14_08_18_044-self-game-distillation.md), [041-R31](spec/2026_10_15_22_24_041-imitation-learning.md))
- 16:56 — 수정(사용자 결정 ②): 022 교사만 바꿔 라이브 교사와 그림자 재레이블이 같은 결정을 내리게 함(021-R8 교사는 바이트 동일), 개발 seed 6판 일치 95.0~97.7%, 커밋 dc1f8d9 ([022-R9·AC3](spec/2026_11_14_08_18_022-matchup-teacher.md), [021-R8](spec/2026_10_15_22_24_021-opponent-pool.md))
- 16:56 — 실행 시작(022-AC2 공식 점검 3회차): 208판, 교사 7b24d7d, Linux CPU 호스트 3슬롯만. 산출물 `local/work/teacher_tp_check/check3/` ([022-AC2](spec/2026_11_14_08_18_022-matchup-teacher.md))
- 17:33 — 결과(044-R12 표본 end-to-end): 증류 판 20판 → build → 레이블 → `mix-fractions` → BC 50 update → 측정 job → RL 100 update 정상(CUDA), 600판이면 약 490판 보충 필요 추정, ROCm BC·RL은 사용자 지시로 멈춤. `<shared>/eval/distill-d1-sample-20261114/` ([044-R12·AC5](spec/2026_11_14_08_18_044-self-game-distillation.md))
- 17:46 — 결정(사용자, 표본 뒤 일괄 질문): 증류 d1은 600판 + 보충 500판을 한 번에, 레이블 seed 1, T1 크래시 판은 build 전에 뺌, 판 추론은 CUDA 호스트; [022](spec/2026_11_14_08_18_022-matchup-teacher.md) botgen 보충 규칙 확정 ([044](spec/2026_11_14_08_18_044-self-game-distillation.md))
- 17:46 — 결정·준비(사용자): 보충 plan `config/plans/distill_m2_d1_x1.toml`(500판), ROCm RL 최악 모양 메모리는 Linux ROCm 호스트를 다시 쓸 수 있을 때 재측정, spec 044 반영 ([044-R1·R2·R3](spec/2026_11_14_08_18_044-self-game-distillation.md), [050-R9](spec/2026_10_15_22_24_050-evaluation.md))
- 17:49 — 실행 끝·결과(022-AC2 3회차): T0 프로토스 53승 3무 8패·테란 60승 3무 1패 → 합격선 통과, 크래시 0 ([022-AC2](spec/2026_11_14_08_18_022-matchup-teacher.md))
- 18:08 — 실행 시작(022-R12 교사 botgen v2): 800판, 교사 7b24d7d, Linux CPU 호스트 3슬롯만 ([022-R12](spec/2026_11_14_08_18_022-matchup-teacher.md))
- 18:30 — 장시간 실행 끝: 증류 d1 판 1,100판(Finished 1,097, Aborted 3 = T1 Iron 크래시). job `<shared>/eval/distill-m2-d1-20261114` ([044-R1](spec/2026_11_14_08_18_044-self-game-distillation.md))
- 18:46 — 결정(사용자): 증류 d1 val500·종족별 val·단서 진단은 CUDA bf16으로 재고 ROCm f32 고정과의 차이를 기록, 022 교사 구현 메모 6가지 유지 ([022](spec/2026_11_14_08_18_022-matchup-teacher.md))
- 19:42 — 실행 끝(022-R12 botgen v2): 800판 완료·실패 0. 원본 `<shared>/datasets/replays/raw/botgen/teacher-tp-20261114-v2/teacher_botgen_tp/` ([022-R12](spec/2026_11_14_08_18_022-matchup-teacher.md))
- 20:10 — 장시간 실행 끝·준비: 증류 d1 크래시 3판 제외 → build(written 1,097) → 레이블(CUDA bf16, seed 1) `shards/distill_m2/d1` → 교사 + dagger 45.5%·증류 19.2%·사람 35.3%, s2 2,000으로 `config/bc_m2_distill_d1.toml` 채움 ([044-R1~R5](spec/2026_11_14_08_18_044-self-game-distillation.md))
- 20:19 — build 끝·결과: `<shared>/datasets/replays/shards/botgen/teacher-tp-20261114-v2`(800 시점·612,366 스텝, v1 보존), 022-AC3 일치 6판 중 5판 ≥ 95%(g601 botgen 기준 94.8%) ([022-R12·AC3](spec/2026_11_14_08_18_022-matchup-teacher.md))
- 20:23 — 결정(사용자): 022-AC3 botgen 일치율 6판 중 5판을 합격으로 받아 교사 라운드에 botgen v2를 씀, R9 그림자 일치 세부 7가지 확인, 모델 판 재레이블이 실패하면 교사 라운드 중단 ([022-R9·AC3](spec/2026_11_14_08_18_022-matchup-teacher.md))
- 20:28 — 결과: 022-AC3 경로 2 모델 판 20판 재레이블 20/20 written·무효 0 → 통과, 022-AC3 전체 통과. `local/work/teacher_tp_check/ac3_relabel20/` ([022-AC3](spec/2026_11_14_08_18_022-matchup-teacher.md))
- 20:29 — 리뷰(별개 리뷰어, 361645a): 반영 가능·코드 버그 없음, 021-R8 골든 다이제스트 같음, 주석 2건 반영·나머지는 반영 안 한 이유 기록 ([022-R9·AC3](spec/2026_11_14_08_18_022-matchup-teacher.md), [021-R8](spec/2026_10_15_22_24_021-opponent-pool.md))
- 22:34 — 결과·준비: 증류 d1 지점 측정 → 판정 d2(`<shared>/eval/m2-lineage/d1_first_test.verdict`, T0 프로토스 9/32, 막음 합 66 < 70, 게이트 FAIL); d2 plan `config/plans/distill_m2_d2.toml`(300판)·설정 `config/bc_m2_distill_d2.toml` 준비 ([044-R6·R1·R5](spec/2026_11_14_08_18_044-self-game-distillation.md), [050-R9·R10](spec/2026_10_15_22_24_050-evaluation.md))
- 22:35 — 판정·결정(사용자): 증류 d1 판정 d2로 감; d2부터 종족별 val 기준은 같은 백엔드로 잰 `m2-s2e2`, 4:00 전 벙커는 엄격 그대로, d2·t1 모두 실패하면 멈추고 RL 출발점을 물음 ([044-R6](spec/2026_11_14_08_18_044-self-game-distillation.md), [050-R10](spec/2026_10_15_22_24_050-evaluation.md))
- 23:51 — 장시간 실행 끝·준비: 증류 d2 판 300판 → build → 레이블(seed 2) `shards/distill_m2/d2` → 증류 20.0%, s2 2,014로 `config/bc_m2_distill_d2.toml` 채움; ROCm RL 메모리 측정 첫 시도는 호스트 전원 꺼짐으로 끊겨 23:50 재시작 ([044-R1~R6](spec/2026_11_14_08_18_044-self-game-distillation.md))

## 2026-11-15
- 00:21 — 결과(044-AC5 ROCm 부분): RL learner 최악 모양 벤치(참조 모델 2개) 6.82 GiB로 12 GiB 안 → 통과. 산출물 `<shared>/eval/m2-lineage/logs/rocm_rl_mem_bench_20261115-0019.*` ([044-AC5](spec/2026_11_14_08_18_044-self-game-distillation.md), [060-AC7](spec/2026_10_15_22_24_060-infra.md) ②)
- 01:22 — 수정(별개 리뷰어 2회): `per_action.py`가 botgen manifest 파트에서 [022](spec/2026_11_14_08_18_022-matchup-teacher.md) 교사 `teacher-tp-*`를 빼고 씀, 디스크·manifest가 다르면 멈춤; d2 행동별 지표는 이전 코드로 잼 ([041-R27](spec/2026_10_15_22_24_041-imitation-learning.md), [044-R8](spec/2026_11_14_08_18_044-self-game-distillation.md), [050-R9](spec/2026_10_15_22_24_050-evaluation.md))
- 02:07 — 장시간 실행 끝·결과: `m2-distill-d2` 학습(2,014 update) → 지점 측정 → 판정 stop(`<shared>/eval/m2-lineage/d2_first_test.verdict`): T0 프로토스 12/32·테란 15/32, 막음 합 45 < 70, 게이트 FAIL ([044-R5·R6](spec/2026_11_14_08_18_044-self-game-distillation.md), [050-R9·R10](spec/2026_10_15_22_24_050-evaluation.md))
- 02:10 — 결정(사용자 위임·Fable 자문): 교사 라운드 t1은 d1(`m2-distill-d1/ckpt-2000`)에서 시작(data.seed 17), t1부터 종족별 NLL은 기록만·교사 모드 판 ≤ 3/32를 통과 조건에, RL 출발점 규칙(D = t1 또는 d1) ([044-R6·R7·R8](spec/2026_11_14_08_18_044-self-game-distillation.md), [050-R10](spec/2026_10_15_22_24_050-evaluation.md))
- 02:20 — 데이터 변경(사용자 위임·자문 권고): 022 botgen v2를 botgen manifest에 넣고 v1은 `<shared>/datasets/replays/shards_unused/botgen/teacher-tp-20261114`로 옮김(되돌리지 말 것), 이전 manifest 백업 `local/work/botgen_manifest_pre_t1_20261115.json` ([044-R8](spec/2026_11_14_08_18_044-self-game-distillation.md), [022-R12](spec/2026_11_14_08_18_022-matchup-teacher.md), [061](spec/2026_10_15_22_24_061-shared-storage.md))
- 02:22 — 구현·결정(사용자 위임·Fable 자문): `distill_verdict.py`에 t1 판정·RL 출발점 `d_choice` 반영(리뷰 반영), `m2-rl-fix-1`은 `--bc <D>`·`--init` 없음; d2 하락은 개막이 아니라 수비 동작. `<shared>/eval/m2-lineage/logs/rush_opening_diag.md` ([044-R6·R7·R11](spec/2026_11_14_08_18_044-self-game-distillation.md), [050-R10](spec/2026_10_15_22_24_050-evaluation.md), [042-R22](spec/2026_10_15_22_24_042-rl-training.md))
- 02:36 — 표본 끝: 교사 라운드 t1 표본 end-to-end(d1 20판 → [022](spec/2026_11_14_08_18_022-matchup-teacher.md) 교사 그림자 재레이블 → `mix-fractions` → BC 50 update) 정상. `<shared>/eval/teacher-t1-sample-20261115` ([044-R12·R8](spec/2026_11_14_08_18_044-self-game-distillation.md))
- 02:36 — 장시간 실행 시작: t1 자기 판 300판(`config/plans/dagger_m2_t1.toml`, d1 stochastic), job `<shared>/eval/teacher-m2-t1-20261115` ([044-R8](spec/2026_11_14_08_18_044-self-game-distillation.md) ②)
- 03:03 — 실행 끝·결과: t1 300판 → 재레이블 `shards/dagger_m2/t1`(308,707 스텝) → 교사 안 종족 비율 기준 미달 → 022 교사 botgen 보충 프로토스 1,296·테란 608판 plan 둘(`teacher_botgen_tp_x1.toml`·`_x2.toml`) ([044-R8](spec/2026_11_14_08_18_044-self-game-distillation.md) ②③, [022-R12](spec/2026_11_14_08_18_022-matchup-teacher.md))
- 03:29 — 표본 끝: RL 100 update 표본 `t1-rl-sample-cuda`(`--bc` d1, `--init` 없음) 정상, 종족별 참조 KL 확인. `local/work/m2_rl_fix/t1-rl-sample-cuda` ([044-R12·R11](spec/2026_11_14_08_18_044-self-game-distillation.md))
- 04:11 — 실행 끝·세기: 022 botgen 보충 x1 992·x2 912판 → build `shards/botgen/teacher-tp-20261114-v2-x1`·`-x2` → manifest 69 shard·5,277판 → 교사 + dagger 65.0%(Z:P:T 49.2:30.8:20.0), s2 2,844 update로 학습 설정 채움; 사고 2건(데이터 영향 없음, `.partial-killed` 보존) ([044-R8](spec/2026_11_14_08_18_044-self-game-distillation.md) ③, [022-R12](spec/2026_11_14_08_18_022-matchup-teacher.md))
- 04:12 — 장시간 실행 시작: `m2-teacher-t1` 학습(CUDA bf16, `--init m2-distill-d1/ckpt-2000`, 2,844 update, data.seed 17) ([044-R8](spec/2026_11_14_08_18_044-self-game-distillation.md) ③)
- 05:58 — 장시간 실행 끝: `m2-teacher-t1` 정상 종료(ckpt-2844, 약 2.2초/update). 로그 사본 `<shared>/eval/m2-lineage/train_logs/m2-teacher-t1.log` ([044-R8](spec/2026_11_14_08_18_044-self-game-distillation.md) ③)
- 06:55 — 판정·결정: t1 판정 stop(`<shared>/eval/m2-lineage/t1_first_test.verdict`): T0 테란 18/32·프로토스 16/32, 막음 합 61 < 70, 교사 모드 TvT 4·TvP 9 > 3, 게이트 FAIL → 규칙상 RL 출발점 D = d1 ([044-R6·R7](spec/2026_11_14_08_18_044-self-game-distillation.md), [050-R9·R10](spec/2026_10_15_22_24_050-evaluation.md))
- 06:55 — 장시간 실행 시작: 경로 2 수정 RL `m2-rl-fix-1`(CUDA 호스트, `config/rl_m2_race.toml`, `--bc` d1, 600만 step 상한), 점검은 150만 step마다 첫째 Linux ROCm 호스트. 드라이버 `local/work/m2_rl_fix/` ([044-R9~R11](spec/2026_11_14_08_18_044-self-game-distillation.md), [042-R22](spec/2026_10_15_22_24_042-rl-training.md))
- 07:02 — 결정(사용자 위임·Fable 자문, 사후 변경): 규칙상 D = d1이었으나 D = t1(`m2-teacher-t1/ckpt-2844`)로 바꿈(T0 테란 + 프로토스 34/64 대 20/64, 선택 편향 주의), `m2-rl-fix-1`은 정상 종료·보존, 본 run `m2-rl-fix-2`. 근거 `<shared>/eval/m2-lineage/logs/t1_verdict.json` ([044-R7·R11](spec/2026_11_14_08_18_044-self-game-distillation.md))
- 07:02 — 장시간 실행 끝: `m2-rl-fix-1`(D = d1) 정상 종료, 33,477 agent step·점검 없음. `<shared>/checkpoints/m2-rl-fix-1/ckpt-33477`, 로그 `local/work/m2_rl_fix/m2-rl-fix-1/` ([042-R10](spec/2026_10_15_22_24_042-rl-training.md), [044-R7](spec/2026_11_14_08_18_044-self-game-distillation.md))
- 07:03 — 장시간 실행 시작: `m2-rl-fix-2`(CUDA 호스트, `--bc <shared>/checkpoints/m2-teacher-t1/ckpt-2844`, 600만 step 상한), 중단 기준은 두 점검 연속 테란 ≤ 12 또는 프로토스 ≤ 10, 또는 TvZ < 16. 드라이버 `local/work/m2_rl_fix/m2-rl-fix-2/` ([044-R9~R11](spec/2026_11_14_08_18_044-self-game-distillation.md), [042-R22](spec/2026_10_15_22_24_042-rl-training.md))
- 10:28 — 점검 결과(c1 = `ckpt-1566130`): T0 테란 12/32·프로토스 16/32, TvZ 31/32, 막음 합 50, 교사 모드 0·0; 테란 ≤ 12 1회. 점검이 밀리면 가장 새 점검만(코디네이터 결정). 표 `<shared>/eval/m2-lineage/logs/m2-rl-fix-2_c1_tables.json` ([044-R11](spec/2026_11_14_08_18_044-self-game-distillation.md), [050-R9](spec/2026_10_15_22_24_050-evaluation.md))
- 12:32 — 점검 결과(c2 = `ckpt-3251723`): T0 테란 13/32·프로토스 11/32, TvZ 31/32, 막음 합 57, 교사 모드 0·0 → 연속 조건 끊김, 계속 ([044-R11](spec/2026_11_14_08_18_044-self-game-distillation.md), [050-R9](spec/2026_10_15_22_24_050-evaluation.md))
- 12:33 — 장시간 실행 끝: `m2-rl-fix-2` 정상 종료(600만 step 도달, 마지막 `ckpt-6129424`, 중단 규칙 안 걸림). 로그 사본 `<shared>/eval/m2-lineage/train_logs/m2-rl-fix-2.train.csv`, run 로그 `local/work/m2_rl_fix/m2-rl-fix-2/` ([044-R11](spec/2026_11_14_08_18_044-self-game-distillation.md), [042-R12](spec/2026_10_15_22_24_042-rl-training.md))
- 14:07 — 측정 결과(D = t1 추가 측정): `m2-teacher-t1/ckpt-2844` test 1,199판 top-1 71.63%·NLL 0.8928(`m2-s2e2` 72.14%·0.8683). `artifacts/bc_main_eval/test_cuda/m2-teacher-t1.json` ([044-R7](spec/2026_11_14_08_18_044-self-game-distillation.md))
- 14:50 — 측정 결과: `m2-rl-fix-2/ckpt-6129424` T1 300 유효 298판 중 30승(Checkpoint 1 33/300, BC 13/299). job `<shared>/eval/m2-rl-fix-2-final-20261115` ([044-R11](spec/2026_11_14_08_18_044-self-game-distillation.md), [050-R9](spec/2026_10_15_22_24_050-evaluation.md))
- 15:43 — 점검 결과(c3·c4): c3 T0 테란 9·프로토스 13/32, c4(마지막 `ckpt-6129424`) 테란 19·프로토스 11/32·TvZ 32/32; 연속 중단 조건 없음, 교사 모드 판은 c1부터 모두 0 ([044-R11](spec/2026_11_14_08_18_044-self-game-distillation.md), [050-R9](spec/2026_10_15_22_24_050-evaluation.md))
- 17:00 — 측정 결과(D = t1 추가 측정): D T1 300 중 299판 44승(마지막 한 판은 코디네이터 임시 결정으로 무효), `m2-rl-fix-2` 30/298·Checkpoint 1 33/300·`m2-s2e2` 24/300. 표 `<shared>/eval/m2-lineage/logs/D_t1_t1_tables.json` ([044-R7](spec/2026_11_14_08_18_044-self-game-distillation.md))
- 17:01 — 장시간 실행 시작: 경로 2 AC1 비교 RL `m2-ac1-vtrace-s1`(이어 s2·s3; `--bc m2-s2e2/ckpt-11387`, `--ac1-race zerg`, 300만 step), seed 판정은 Wilson 하한 > BC 상한 85.0%. 드라이버 `local/work/m2_ac1/` ([042-R21·AC1](spec/2026_10_15_22_24_042-rl-training.md) ④, [041-R29](spec/2026_10_15_22_24_041-imitation-learning.md) ⑥)
- 17:06 — 결정(사용자 위임·Fable 자문): 경로 2 수정 모델은 D = t1(`m2-teacher-t1/ckpt-2844`) 그대로, `m2-rl-fix-2/ckpt-6129424`는 기록·보존만; 다음 RL 전 테란·프로토스 KL 참조 재검토 ([044-R7·R11·R9](spec/2026_11_14_08_18_044-self-game-distillation.md))
- 17:17 — 장시간 실행 일시 정지(사용자가 CUDA 호스트 사용): `m2-ac1-vtrace-s1` 정상 종료(`ckpt-180368`, DONE 없음); 사용자 지시 "CP1이랑 동일 학습량이 되면 정지해" — seed마다 300만 step에서 멈춤 ([042-R10·R21](spec/2026_10_15_22_24_042-rl-training.md))
- 17:20 — 장시간 실행 재개·호스트 바꿈(사용자 지시 "다른 pc가 먼저 유휴상태가 되면 거기서 학습 이어가"): `m2-ac1-vtrace-s1`을 둘째 Linux ROCm 호스트에서 `ckpt-180368`부터 ROCm f32로 이어 감(약 45초/update). 드라이버 `local/work/m2_ac1/ac1_linux.sh` ([042-R21](spec/2026_10_15_22_24_042-rl-training.md) ④, [061](spec/2026_10_15_22_24_061-shared-storage.md), [060-R8](spec/2026_10_15_22_24_060-infra.md))
- 20:43 — 측정 결과: `m2-rl-fix-2/ckpt-6129424` 진단 — 풀 HP h_t R² 0.642(t1 0.554), 반사실 Δ 줄어듦, 죽은 채널 최악 r4 80.5%; 기록만, D = t1 그대로. `<shared>/artifacts/diag/m2-rl-fix-2__ckpt-6129424-agents-main` ([050-R9](spec/2026_10_15_22_24_050-evaluation.md))
- 20:55 — 장시간 실행 시작: `m2-ac1-vtrace-s2`를 첫째 Linux ROCm 호스트에서 s1과 병렬로(ROCm f32, seed 2), seed 배분은 공용 드라이브 claim 폴더 + 061 잠금. 드라이버 `local/work/m2_ac1/ac1_linux2.sh` ([042-R21·AC1](spec/2026_10_15_22_24_042-rl-training.md) ④, [061](spec/2026_10_15_22_24_061-shared-storage.md), [060-R8](spec/2026_10_15_22_24_060-infra.md))

## 2026-11-16
- 00:28 — 장시간 실행 호스트 바꿈(사용자 "이 pc 써"): `m2-ac1-vtrace-s2`를 정상 종료(`ckpt-505726`) 뒤 CUDA 호스트로 옮겨 이어 감(약 205 step/s), s1은 둘째 Linux ROCm 호스트에 그대로 ([060-R8](spec/2026_10_15_22_24_060-infra.md), [042-R10·R21](spec/2026_10_15_22_24_042-rl-training.md), [061](spec/2026_10_15_22_24_061-shared-storage.md))
- 08:05 — 장시간 실행 끝·결과: `m2-ac1-vtrace-s2` 학습 끝(`ckpt-3040542`) → AC1 167/200 = 83.5% [77.7, 88.0], seed 판정 통과 못 함. 표 `<shared>/eval/m2-lineage/logs/m2-ac1-vtrace-s2_ac1_tables.json` ([042-AC1·R21](spec/2026_10_15_22_24_042-rl-training.md))
- 08:05 — 장시간 실행 끝·결과: `m2-ac1-vtrace-s3`(전부 CUDA bf16, `ckpt-3082043`) → AC1 170/200 = 85.0% [79.4, 89.3], 통과 못 함. 표 `<shared>/eval/m2-lineage/logs/m2-ac1-vtrace-s3_ac1_tables.json` ([042-AC1·R21](spec/2026_10_15_22_24_042-rl-training.md))
- 08:15 — 장시간 실행 호스트 바꿈: CUDA 호스트가 비어 `m2-ac1-vtrace-s1`을 둘째 Linux ROCm 호스트에서 정상 종료(`ckpt-2644955`) 뒤 CUDA 호스트로 옮겨 이어 감 ([060-R8](spec/2026_10_15_22_24_060-infra.md), [042-R10·R21](spec/2026_10_15_22_24_042-rl-training.md), [061](spec/2026_10_15_22_24_061-shared-storage.md))
- 08:56 — 장시간 실행 끝·결과: `m2-ac1-vtrace-s1`(`ckpt-3024918`) → AC1 172/200 = 86.0% [80.5, 90.1], 통과 못 함; 3 seed 평균 84.8%(509/600), 셋 다 BC 상한 85.0% 못 넘음(Checkpoint 1 V-trace 평균 75.5%), 추가 학습 없음(사용자 지시) ([042-AC1·R21](spec/2026_10_15_22_24_042-rl-training.md))
- 10:13 — 새 리포트 절·정정: model2 12절 "경로 2 AC1 비교", 11.3절 `m2-rl-fix-2` 진단·D = t1, 9.9절 정정(5b2d3fd로 테란전 `bunker_order_4m` PASS → FAIL, 전체 FAIL 그대로); protoss_fix_plan에 최종 결과 한 줄 ([model2](report/plan/2026_11_09_22_18_model2.md) 9.9·11.3·12절, [protoss_fix_plan](report/plan/2026_11_14_08_18_protoss_fix_plan.md), [042-AC1·R21](spec/2026_10_15_22_24_042-rl-training.md), [050-R9·R10](spec/2026_10_15_22_24_050-evaluation.md))
- 12:25 — 결정·구조 변경(사용자 결정): 문서를 `docs/`로 통합(spec·리포트 날짜 접두사), 코드·config 주석 제거, 장비별 작업 파일·게임 데이터를 `local/`로 이동, timeline 축약, 커밋 메시지 형식 도입 ([AGENTS.md](../AGENTS.md))
- 12:41 — 결정(사용자): 에이전트 규칙을 `AGENTS.md`로 옮겨 다른 에이전트도 읽게 함, `CLAUDE.md`는 `@AGENTS.md` 참조만 ([AGENTS.md](../AGENTS.md), [001-R10](spec/2026_10_15_22_24_001-system-architecture.md))
- 14:30 — 결정(사용자): 공용 드라이브 끝난 자료 작업 단위별 압축 보관, 검증 후 원본 삭제 ([061](spec/2026_10_15_22_24_061-shared-storage.md) R17)
- 13:10~15:45 — 구조 리팩토링(사용자 지시, 동작 변경 없음): crate 의존 정리(store→replay·train→league·scripted↔replay 제거), 미사용 Rust 학습 코드 제거, 큰 파일 6개·T0 봇 crate 모듈 분리, Python 패키지 import·중복 통합, 테스트 helper 통합, 참조 없는 plan·도구 삭제 (커밋 `a994edb`~`1a81f5e`, [001](spec/2026_10_15_22_24_001-system-architecture.md))
- 15:45 — 수정: `uranium-replay`·`uranium-train`·`python -m uranium`이 `URANIUM_ENV_FILE`을 따르고 Python `.env` 큰따옴표 값 해석을 Rust 로더와 맞춤, 순서에 따라 실패하던 테스트 6건 해결 ([001-R9](spec/2026_10_15_22_24_001-system-architecture.md), [commands.md](guide/commands.md#1-설치설정))
- 16:10~21:00 — 구조 변경(사용자 승인): shard 형식·분할 규칙을 `uranium-data`로(replay → data), 호스트 배치 레이아웃을 `uranium-env`로, 분기 게임·운영 지표·양보·항복 판정을 새 crate `uranium-match`로 이동; `tools/eval` 코드를 `uranium.measure`로(명령 그대로), `diag`를 `cue`·`planning`으로 분리 (커밋 `b398301`~`028b485`, [001](spec/2026_10_15_22_24_001-system-architecture.md))
- 21:00 — 수정·검증: Python `.env` 로더를 Rust 로더와 같은 규칙으로(`$NAME` 치환 포함, 네 장비 `.env` 값 차이 0), 재레이블 셔틀 ID 72 → 69; Linux ROCm 호스트에서 Rust 649·Python 834 통과, 실제 shard 3개 데이터셋 split 통계가 이동 전 실행 파일과 일치 ([001-R9](spec/2026_10_15_22_24_001-system-architecture.md), [commands.md](guide/commands.md#1-설치설정))
- 21:03 — 결정(사용자)·구조 변경: 릴리스 가중치를 Git LFS로 전환, 2026-11-08 이후 커밋 370개 해시가 바뀜(문서 속 인용은 새 해시로 고침, 옛 해시 대응표 [lfs_commit_map](report/release/2026_11_16_21_03_lfs_commit_map.md))
- 21:48 — 새 리포트·측정 결과: 학습 최적화 여지 검토 — BC 섞기 run은 shard 파일의 66~79%가 0판인데 전부 디코드(데이터 대기 34~52%), CUDA RL 학습기 벤치 3.66초/update(레인 그룹 2, eager), ROCm RL 병목은 추론 대기, ROCm 모델 2 BC 10.70초/update ([opt_review](report/review/2026_11_16_21_25_opt_review.md))
- 21:48 — 결정(사용자): 빈 shard 파일 건너뛰기·시작 스캔 파일 단위 캐시·shard 파일 미리 읽기를 spec에 적음, RL 레인 그룹은 2 그대로 ([041-R13·R16](spec/2026_10_15_22_24_041-imitation-learning.md), [opt_review](report/review/2026_11_16_21_25_opt_review.md) 3절)

## 2026-11-17
- 01:27 — 측정 결과·판정: CUDA RL 실게임 100 update — 벤치 3.66초 대 실게임 6.82초의 차이는 추론 동시 실행 약 +2.2초; 레인 그룹 4는 최악 모양 추정 18.5 GiB로 탈락, 2 유지 ([opt_review](report/review/2026_11_16_21_25_opt_review.md) 7.2·7.3절, [042](spec/2026_10_15_22_24_042-rl-training.md))
- 02:55 — 반영·검증: 빈 shard 파일 건너뛰기·시작 스캔 파일 단위 캐시(커밋 `9ac0e75`·`050eb4c`·`ac83c02`), 실데이터 300 batch digest 일치 ([041-R13](spec/2026_10_15_22_24_041-imitation-learning.md), [opt_review](report/review/2026_11_16_21_25_opt_review.md) 5절·7.8절)
- 05:25 — 결정(사용자)·채택: Linux ROCm 호스트 추론 graph 배치 4까지(생성 속도 +25.0%, 잡음 0.2%), 추론 서버 메모리 상한 2 → 2.5 GiB ([060-R3](spec/2026_10_15_22_24_060-infra.md), [opt_review](report/review/2026_11_16_21_25_opt_review.md) 7.6·7.7절)
- 05:30 — 측정 결과·판정: 빈 파일 건너뛰기 뒤 `m2-teacher-t1` 조합 100 update 데이터 대기 31.5% → 13.0%, 시작 스캔 671 → 0초(캐시 적중) → 10% 초과라 shard 파일 미리 읽기 착수 ([opt_review](report/review/2026_11_16_21_25_opt_review.md) 7.10절, [041-R13·R16](spec/2026_10_15_22_24_041-imitation-learning.md))
- 05:40 — 판정: Linux ROCm 엔티티 패딩 단위 32·16은 −2.8%·−2.6%(잡음 1.3%)로 기준 3% 미달, `pad_entities` 64 그대로 ([opt_review](report/review/2026_11_16_21_25_opt_review.md) 7.9절)
- 07:05 — 측정 결과·판정: RL 학습기 compile은 update 6.8 → 3.5초지만 85 update 동안 10회 재컴파일(추론 최장 307초 멈춤)·남은 RAM 0.98 GB → 기본값 켜지 않음, channels-last도 실게임 이득 없음 ([opt_review](report/review/2026_11_16_21_25_opt_review.md) 7.5·7.12절, [032-R19](spec/2026_10_15_22_24_032-model-architecture.md))
- 07:20 — 반영: shard 파일 미리 읽기(커밋 `b217df6`, 키 `URANIUM_BC_FILE_AHEAD`, 기본 끔·CUDA 호스트 2·Linux ROCm 호스트 끔) ([041-R16](spec/2026_10_15_22_24_041-imitation-learning.md), [opt_review](report/review/2026_11_16_21_25_opt_review.md) 7.11절)
- 08:25 — 측정 결과: shard 파일 미리 읽기 깊이 1·2·4에서 `m2-teacher-t1` 조합 100 update 데이터 대기 10.3%·9.6%·9.1%(건너뛰기만 13.0%, 원래 31.5%), CUDA 호스트 깊이 2로 둠 ([opt_review](report/review/2026_11_16_21_25_opt_review.md) 7.13절, [041-R16](spec/2026_10_15_22_24_041-imitation-learning.md))
- 08:40 — 결정(사용자)·정정: 05:40의 "`pad_entities` 64 그대로" 판정을 뒤집어 Linux ROCm 호스트 BC 학습 엔티티 패딩 단위 32 채택(−2.8% ± 0.4%p), 키 `URANIUM_BC_PAD_ENTITIES` ([041](spec/2026_10_15_22_24_041-imitation-learning.md) 구현 메모, [opt_review](report/review/2026_11_16_21_25_opt_review.md) 7.14절)
- 09:05 — 체크포인트(사용자 결정 2026-11-17): Checkpoint 2 = `m2-teacher-t1/ckpt-2844`(태그 `checkpoint-2`), Checkpoint 2 RL = `m2-ac1-vtrace-s1/ckpt-3024918` Main(태그 `checkpoint-2-rl`), `releases/checkpoint-2/`·`releases/checkpoint-2-rl/` ([checkpoint2](report/release/2026_11_17_09_05_checkpoint2.md))
- 09:10 — 반영: RL 학습기 compile·channels-last 스위치(기본 끔, 커밋 `2e5b92e`·`06358d8`·`b760226`) ([032-R19](spec/2026_10_15_22_24_032-model-architecture.md), [opt_review](report/review/2026_11_16_21_25_opt_review.md) 7.12·7.15절)
- 09:30 — 배포·확인: Linux ROCm 호스트 두 대에 `bab6cbc` 배포·재빌드, 추론 graph 배치 4·BC 패딩 32 기본 적용 확인(BC 벤치 10.68 → 10.40초), 시작 스캔 캐시 미리 채움 ([opt_review](report/review/2026_11_16_21_25_opt_review.md) 7.16절·10절)
- 10:0x — 결정(사용자)·반영: 공간 인코더에 GroupNorm 스위치 `spatial_norm`(pre-activation ResNet 위치, 그룹당 16채널), Checkpoint 2에서 BC 3,000 update로 죽은 채널이 살아나는지 시험 ([032-R26](spec/2026_10_15_22_24_032-model-architecture.md), [gn_resume](report/experiment/2026_11_17_10_05_gn_resume.md))
- 10:03 — 시작: `m2-gn-t1`(Windows CUDA 호스트, Checkpoint 2 가중치 + GroupNorm, BC 3,000 update); update 100에서 c3·r4 죽은 채널 0 ([gn_resume](report/experiment/2026_11_17_10_05_gn_resume.md))
- 10:4x — 결정(사용자)·반영: 같은 구조에 블록 끝 ReLU만 다시 넣는 스위치 `spatial_end_relu`로 두 번째 시험 `m2-gn-relu-t1` ([032-R27](spec/2026_10_15_22_24_032-model-architecture.md), [gn_resume](report/experiment/2026_11_17_10_05_gn_resume.md))
- 11:18 — 끝·결과: `m2-gn-t1` 2,844 update(한 epoch 데이터 소진, 3,000 미달) — c3·r4 죽은 채널 80.9·84.9 → 0, r5 90.7 → 1, 재붕괴·경보 없음; 이어서 `m2-gn-relu-t1` 시작 ([gn_resume](report/experiment/2026_11_17_10_05_gn_resume.md) 3절)
- 12:37 — 끝·결과: `m2-gn-relu-t1` 2,844 update — 끝 ReLU를 넣으면 bottleneck 출력 죽은 채널 0 → 약 10, `S8` 상수 채널 0 → 약 10(128 중), 블록 안 ReLU는 유지 → 끝 ReLU 없는 쪽이 나음 ([gn_resume](report/experiment/2026_11_17_10_05_gn_resume.md) 3절)
- 12:5x — 결정(사용자): GroupNorm(끝 ReLU 없음) 본 계열 채택, BC는 처음부터가 아니라 Checkpoint 2에서 이어 학습, `m2-gn-t1e2`(한 epoch 더) 뒤 Checkpoint 2 표 전체로 측정, 측정 지점 `gn` 추가 ([032-R26](spec/2026_10_15_22_24_032-model-architecture.md), [050-R9](spec/2026_10_15_22_24_050-evaluation.md), [gn_resume](report/experiment/2026_11_17_10_05_gn_resume.md) 3절, [checkpoint2](report/release/2026_11_17_09_05_checkpoint2.md) 3절)
- 13:17 — 시작: `m2-gn-t1e2`(Windows CUDA 호스트, `m2-gn-t1/ckpt-2844`에서 같은 설정으로 한 epoch 더) ([gn_resume](report/experiment/2026_11_17_10_05_gn_resume.md) 4절)
- 14:29 — 끝: `m2-gn-t1e2` 2,844 update(update당 1.44초, 처음 100 update 제외), 경보 0건, c2·c3·r4 죽은 채널 0·r5 2, `S8`·`s_pool` 상수 채널 0 ([gn_resume](report/experiment/2026_11_17_10_05_gn_resume.md) 4절)
- 14:39 — 측정 시작: `m2-gn-t1e2/ckpt-2844`를 지점 `gn`으로 Checkpoint 2 표 전체; 첫째 Linux ROCm 호스트는 12:16 ROCm 패키지 업그레이드로 torch import 실패 → ROCm 채점은 둘째 호스트에서 ([050-R9](spec/2026_10_15_22_24_050-evaluation.md), [gn_resume](report/experiment/2026_11_17_10_05_gn_resume.md) 4절)
- 16:4x — 발견·복구: 첫째 Linux ROCm 호스트 ROCm 7.2.4 → 10.1(12:16 전체 업그레이드)로 torch import 실패 → rocBLAS 재빌드 + MIOpen 버전 검사 건너뛰기, 시험한 범위(장치 테스트·벤치 스위트·BC 벤치 12 update)에서 7.2.4와 같음 ([rocm_10_1](report/investigation/2026_11_17_16_27_rocm_10_1.md))
- 16:57 — 측정 끝·결과: T1 49/300(Checkpoint 2 44/299), T0 테란 23·프로토스 13/32(18·16), TvZ 26/32(29), 러시 막음 18/36(21/40), test 71.73%(71.63%), val500 71.94%(71.83%) — 게임 차이는 잡음 범위, Checkpoint 2보다 같은 데이터를 두 epoch 더 본 모델이라 GroupNorm 효과로는 못 읽음 ([gn_resume](report/experiment/2026_11_17_10_05_gn_resume.md) 4절, [050-R9](spec/2026_10_15_22_24_050-evaluation.md))
- 17:5x — 결정(사용자): 다음 작업(계획형 JEPA, 교사 모드 누출 수정, T1 패배 분석)의 바탕 가중치는 `m2-gn-t1e2/ckpt-2844`, 정식 Checkpoint 지정은 아님 ([gn_resume](report/experiment/2026_11_17_10_05_gn_resume.md) 4절)
- 17:1x — 정정: 바로 위 줄의 시각 17:5x는 17:1x
- 17:29 — 결정(사용자): `gn` 지점은 050-R10 게이트 미적용·지표 기록만, T1 플랫폼 고정 안 함; 죽은 채널 경보 대상 `S8` 상수 채널 확인(블록 출력 제외, 문턱 50% 유지 — 끝 ReLU 시험의 13/128 같은 부분 붕괴는 경보 없이 로그 열로 봄, [gn_resume](report/experiment/2026_11_17_10_05_gn_resume.md) 3절); RL 장치 비교 테스트는 첫 update만 비교 ([050-R9](spec/2026_10_15_22_24_050-evaluation.md), [041-R25](spec/2026_10_15_22_24_041-imitation-learning.md), [opt_review](report/review/2026_11_16_21_25_opt_review.md) 7.15절)
- 19:3x — 결정(사용자)·끝: torch를 ROCm 10.1에 맞춰 재빌드(같은 커밋·옵션, 약 1시간 38분), 장치 테스트·벤치 스위트 통과, BC 벤치 손실·속도·메모리 같은 범위, 두 Linux ROCm 호스트 전환·MIOpen 검사 건너뛰기 제거(둘째 호스트도 같은 날 ROCm 10.1로 올림) ([rocm_10_1](report/investigation/2026_11_17_16_27_rocm_10_1.md) 9절)
- 20:0x — 발견: gn·t1의 테란·프로토스전은 대부분 022 교사 방식(0~6분 분당 명령 약 70, gn T0 60/64가 < 150), 누출은 저그전 교사 빌드(가스 없음, 배럭 3)로 빠지는 12~28% 판; `m2-rl-fix-2`는 `m2-s2e2` 참조로 022 방식을 잃음 ([teacher_leak_fix](report/experiment/2026_11_17_20_08_teacher_leak_fix.md) 1절)
- 20:0x — 결정(사용자): B1 체크포인트 `m2-gn-t1e2/ckpt-2844`, S0b를 gn으로 다시 학습; 누출 수정 최종 가중치가 다르면 계획형 JEPA 작업은 그 가중치로 옮김 ([051-R1](spec/2026_11_12_12_39_051-branch-games.md) ②)
- 20:1x — 결정(사용자): 테란·프로토스전은 022 방식 유지·저그전 빌드 누출만 제거, 한 가중치; 교사 라운드 t2(gn 출발, 저그 DAgger 판 추가) → 통과하면 `m2-rl-fix-3`(참조 네 키 = 출발 모델) ([044-R8·R9·R11](spec/2026_11_14_08_18_044-self-game-distillation.md), [teacher_leak_fix](report/experiment/2026_11_17_20_08_teacher_leak_fix.md) 3절)
- 20:1x — 결정(사용자): 새 판별 `zerg_build`·`tp_style`(기존 교사 모드는 기록만), gn 대비 합격 기준·최종 가중치 선택 순서, 지점 `t2`·`rl-fix-3`; 게이트 기록만·중단 세부는 메인 임시 결정(사용자 확인 전, [AMB-044-12](spec/README.md)) ([050-R9·R10](spec/2026_10_15_22_24_050-evaluation.md), [044-R13](spec/2026_11_14_08_18_044-self-game-distillation.md))
- 21:45 — 새 리포트·결정(사용자): Linux ROCm 호스트에서 학습·추론 가능한 모델 확대 수단 실측 — 코어 확대는 싸고 엔티티 인코더 확대는 비쌈, 3억 이상은 옵티마이저 상태로 메모리 초과; 확대한다면 후보 B(1억 5,630만), 진행은 모델 구조 변경 테스트 뒤 ([model_scaleup](report/bench/2026_11_17_21_45_model_scaleup.md), [032-R2](spec/2026_10_15_22_24_032-model-architecture.md), [060-R3](spec/2026_10_15_22_24_060-infra.md))
- 22:2x — 결정(사용자): 교사 모드 누출 수정 세부 확정 — t2 저그 DAgger 판 600·저그 판 재레이블 절단 10:00(8:00 버림), 게이트·교사 모드 ≤ 3/32 기록만, T0 하락 두 점검 연속(기준 t2), 가벼운 점검도 점검으로 셈, RL 끝 최종은 T1 테란 + 프로토스 ≥ t2 − 6 ([044-R8·R11·R13](spec/2026_11_14_08_18_044-self-game-distillation.md), [teacher_leak_fix](report/experiment/2026_11_17_20_08_teacher_leak_fix.md) 3절)
- 22:54 — 끝: t2 표본 끝까지(판 → 재레이블 → BC 50 update) 통과 ([teacher_leak_fix](report/experiment/2026_11_17_20_08_teacher_leak_fix.md) 4절)
- 23:00 — 시작: t2 판(테란·프로토스 300 + 저그 600) → 재레이블 → BC `m2-teacher-t2`(Windows CUDA 호스트) ([044-R8](spec/2026_11_14_08_18_044-self-game-distillation.md), [teacher_leak_fix](report/experiment/2026_11_17_20_08_teacher_leak_fix.md) 4절)
- 23:52 — 끝: S0b 예측기를 gn으로 다시 학습(1,340판, 행동을 주면 내 상태 오차 −21%, 이전과 같음) ([051-R1](spec/2026_11_12_12_39_051-branch-games.md) ②, [jepa_planning](report/plan/2026_11_11_14_56_jepa_planning.md))
- 23:58 — 시작: 분기 게임 B1(Linux 호스트 3대 전용, 작은 끝-끝 먼저) ([051-AC7](spec/2026_11_12_12_39_051-branch-games.md))

## 2026-11-18
- 00:56 — 끝: BC `m2-teacher-t2` 3,258 update(update당 1.46초); 저그 판 재생 불일치 25/600(이전 라운드보다 많음) ([teacher_leak_fix](report/experiment/2026_11_17_20_08_teacher_leak_fix.md) 4절)
- 01:00 — 측정 시작: `m2-teacher-t2/ckpt-3258`를 지점 `t2`로 gn과 같은 표(Windows CUDA 호스트만) ([050-R9](spec/2026_10_15_22_24_050-evaluation.md), [044-R13](spec/2026_11_14_08_18_044-self-game-distillation.md))
- 01:10 — 끝·새 리포트: B1 분기 1,272판 무효 0 — 벙커 90초 금지 `held_8m` −16%p, 벙커 한 결정 강제 `held_s` +19%p, 마린 효과 없음; S0b 채점은 한 결정 효과를 못 맞힘(부호 일치 약 0.5), 내 상태 변화 방향은 맞힘(k8 0.91) ([branch_b1](report/experiment/2026_11_18_01_10_branch_b1.md), [051-R13](spec/2026_11_12_12_39_051-branch-games.md))
- 04:06 — 측정 끝·결과: t2 — `zerg_build` T0 4/64(gn 10)·T1 16/195(26), `tp_style` T0 55/64·T1 142/195, 승 T0 테란 + 프로토스 38/64(36)·T1 19/195(27)·T1 전체 49/299(49/300), TvZ 27/32, 러시 막음 56(54), val500 71.99%(71.94%) → 합격 기준 불합격(`zerg_build`·T1 승) ([044-R13](spec/2026_11_14_08_18_044-self-game-distillation.md), [teacher_leak_fix](report/experiment/2026_11_17_20_08_teacher_leak_fix.md) 4절)
- 04:1x — 결정(사용자): t2 불합격이나 t2에서 `m2-rl-fix-3` 시작, RL 끝 판정은 t2 대비(`zerg_build` t2 이하, T1 테란 + 프로토스 승 ≥ 13), 못 넘으면 최종 가중치 t2 ([044-R13](spec/2026_11_14_08_18_044-self-game-distillation.md) ⑤)
- 08:35 — 시작: `m2-rl-fix-3`(t2 출발, KL 참조 네 키 t2, Windows CUDA 호스트) ([044-R11](spec/2026_11_14_08_18_044-self-game-distillation.md), [teacher_leak_fix](report/experiment/2026_11_17_20_08_teacher_leak_fix.md) 4절)
- 08:5x — 결정(사용자): 계획 예측기 단계 1~4(결과 정답·섞기·검증, 모델 판 출처, B1 재확인, 탐색 판·B2 학습), B2 = 줄기 100 × 3 × 4 × 반복 2 = 2,400판(12:00, F) ([053](spec/2026_11_18_08_58_053-planning-predictor-data.md), [054](spec/2026_11_18_08_58_054-exploration-games.md), [051](spec/2026_11_12_12_39_051-branch-games.md) R16)
- 09:05 — 새 리포트·spec: 예측기 데이터 점검(사람 판 60%·봇 판 `scv8` 늘 0·검증 없는 학습 → 결과 외우기·행동 무시, 교사·모델 판만이면 벙커 효과 0.004 → 0.78), spec 053·054(Review), 051 B2 개정 ([jepa_predictor_data](report/investigation/2026_11_18_08_58_jepa_predictor_data.md), [spec README](spec/README.md))
- 09:1x — 결정(사용자): 예측기 학습 세부 기본안 확정(500 step·참을성 4·최대 20,000 등), 탐색 판 상대 세부 기본안 확정, B2 상대·맵·시점은 B2 판 직전에 정함 → spec 053·054 Approved ([053](spec/2026_11_18_08_58_053-planning-predictor-data.md), [054](spec/2026_11_18_08_58_054-exploration-games.md), [051](spec/2026_11_12_12_39_051-branch-games.md) R16)
- 09:58 — 끝(중단 규칙): `m2-rl-fix-3` 0.36M step 첫 점검 — 테란·프로토스전 `tp_style` 13/64(< 32), `zerg_build` 51/64, 가스 중앙값 0 → 최종 가중치 F = t2(`m2-teacher-t2/ckpt-3258`) ([044-R11·R13](spec/2026_11_14_08_18_044-self-game-distillation.md) ⑤)
- 10:59 — 측정 끝·결과: t2를 RL 점검 조건(직전 실패 명령 보임, Linux ROCm)으로 64판 — `zerg_build` 3/64·`tp_style` 52/64, 직전 실패 명령 가림 5/64·53/64 → 붕괴는 측정 조건이 아니라 RL 204 update 탓 ([teacher_leak_fix](report/experiment/2026_11_17_20_08_teacher_leak_fix.md) 4절)
- 11:0x — 결정(사용자): 탐색 판 plan 설정은 t2 승·판별 측정 plan과 같게(`zero_last_failed = true`, 항복 기록만), F = t2 ([054-R1](spec/2026_11_18_08_58_054-exploration-games.md))
- 11:44 — 끝·결과: 계획 예측기 단계 1 비교(gn 덤프, 사람 몫 0·20·35%) — 20%가 가장 고름, 벙커 행동 민감도 0.005 → 0.85(실제 0.90), B1 부호 일치 k8 0.47 → 0.66, 결과 망은 첫 검사(500 step)가 늘 최저 ([s0c_stage1_compare](report/experiment/2026_11_18_13_42_s0c_stage1_compare.md))
- 13:31 — 끝: t2로 단계 2 덤프 전체(S0b 출처·B1 줄기·모델 판 9출처·탐색 판) ([053-R1·R5](spec/2026_11_18_08_58_053-planning-predictor-data.md))
- 13:3x — 결정(사용자): `Aborted` 판(상대 봇 크래시)은 결과 정답 없음(상태 행은 학습에 남김), 결과 망 검증 100 step마다(세계 망 500, 참을성 각 4회) ([053](spec/2026_11_18_08_58_053-planning-predictor-data.md) R2 ⑧·R4 ④, [s0c_stage1_compare](report/experiment/2026_11_18_13_42_s0c_stage1_compare.md))
- 14:34 — 끝·새 리포트: 계획 예측기 단계 1~3 본 실행(t2 덤프, 모델 판 9출처) — 모델 행 결과 머리가 상수보다 나음(held_8m BCE 0.383 대 0.619), 벙커 민감도 0.86(실제 0.90), B1 M3u는 판단 불가(Spearman −0.05) ([s0c_final_t2](report/experiment/2026_11_18_17_18_s0c_final_t2.md))
- 15:27 — 결정(사용자): B2 상대·맵·시점 기본안 확정 — T0만(저그 50 : 프로토스 30 : 테란 20), 맵 반반, 3:00~8:00 열린 운영 행동 3개 이상 결정·앞 시점 + 60초(구현: seed 블록 키 9,000,000) ([051](spec/2026_11_12_12_39_051-branch-games.md) R16, [AMB-051-7](spec/README.md))
- 15:56 — 시작: 분기 게임 B2(F, 줄기 100 + 교체 9, Linux 호스트 3대) ([051-R16](spec/2026_11_12_12_39_051-branch-games.md))
- 17:12 — 끝·새 리포트: 단계 4 첫 실행(탐색 시범 1,000판만, 짝 손실 끔) — 떼어 둔 탐색 판 결과 BCE 소폭 나아짐(k32 held_8m 0.417 → 0.392), 강제 효과는 못 배움(예측 +0.007 대 측정 −0.044 [−0.120, 0.024]) ([s0c_stage4_explore](report/experiment/2026_11_18_17_18_s0c_stage4_explore.md), [054-R6](spec/2026_11_18_08_58_054-exploration-games.md))
- 17:1x — 결정(사용자): 짝 손실의 `none` 기준 = Σπ(c)σ(q̂(c))(M3u와 같음), 학습 줄기 10%를 검증 줄기로 떼고 결과 망 검증 손실에 짝 손실 포함 ([053-R8](spec/2026_11_18_08_58_053-planning-predictor-data.md) ①)
- 18:22 — 끝·새 리포트: B2 2,400판 무효 0 — 정책 순위 1~3위 한 결정 강제의 `held_8m`·`held_s`·`alive_10m` 효과는 0과 구별 안 됨(±0.02), `win` +0.03~+0.04는 판정 보류(승패 난 판만, 다중 비교), 시점별 split-half r 0~0.2, 교체 9판 모두 `zerg_rush` ([branch_b2](report/experiment/2026_11_18_18_40_branch_b2.md))
- 19:12 — 새 리포트: 모델 안 보조 예측기(JEPA·priv)는 AdamW eps 1e-5에 눌려 step이 정상의 약 1/5~1/15(aux_jepa 1/7~1/10), Core도 0.35~0.66배; 몸통이 받는 JEPA 몫은 0.1~1.4%라 eps로는 안 바뀜, 고치는 안은 사용자 결정 ([aux_jepa_adam_eps](report/investigation/2026_11_18_19_12_aux_jepa_adam_eps.md))
- 19:44 — 끝·새 리포트: 단계 4 r4(탐색 + B2 짝 손실) — 떼어 둔 B2 줄기에서 r2·r3보다 낫지 않음(`held_8m` Spearman −0.04·+0.01, 짝 손실 0 예측보다 큼), 학습 줄기만 외움; B1 M3u Spearman 0.27·0.22로 오름; 짝 손실이 `win`·`alive_10m`을 못 읽던 문제 고침 ([s0c_stage4_b2](report/experiment/2026_11_18_19_50_s0c_stage4_b2.md), [053-R8](spec/2026_11_18_08_58_053-planning-predictor-data.md))
- 20:18 — 진단(사용자 승인): t2 몸통 고정, 보조 예측기만 다시 맞춤 — 떼어 둔 판 aux_jepa 0.053 → 0.031(eps 1e-5, lr 3e-5 일정·3,258 update) → 0.022(eps 1e-8), 상수 예측 대비 0.33 → 0.14; 예측기는 최적에서 멀고 eps가 거기로 가는 속도를 늦춤(추론), priv는 여지 작음 ([aux_jepa_adam_eps](report/investigation/2026_11_18_19_12_aux_jepa_adam_eps.md) 7절)
- 20:30 — 결정(사용자)·spec: 계획 단위 분기 B3 기본안 확정 — 계획 6개(CC 이르게·늦게, 팩토리, 배럭 추가, SCV 끊기, 벙커) + `none`, B2 줄기 재사용 2,100판(12:00), 주 결과 score, T0 상대 4칸, 줄기 추가는 결과 뒤 → spec 055 Approved ([055](spec/2026_11_18_20_30_055-plan-branch-games.md))
- 23:40 — 결정(사용자): B3 표본에서 `factory_now`의 팩토리 강제 6/6 `missed`(F는 그 시점까지 가스 0) → 리파이너리 강제 [0, 30초) 뒤 팩토리 강제 [60, 180초)로 바꿈 ([055-R2](spec/2026_11_18_20_30_055-plan-branch-games.md), [branch_b3](report/experiment/2026_11_18_20_36_branch_b3.md))

## 2026-11-19
- 08:53 — 시작: B3 2,100판(재표본 12판에서 `factory_now` `missed` 0) ([branch_b3](report/experiment/2026_11_18_20_36_branch_b3.md))
- 10:50 — 끝: B3 2,100판 무효 0 — 계획 6개 모두 걸렸으나(물림 0.36~0.96) 12:00 score 효과 |D| < 0.05(`cc_now` +0.037만 구간 0 밖), `bunker_now` 상대별 방향 차이는 약한 신호(러시 +0.09 · 저그 −0.06, 보정 전 p 0.013), 프로토스 · 테란전은 12:00 무승부 7할 ([branch_b3](report/experiment/2026_11_18_20_36_branch_b3.md))
- 11:15 — 정정: 11-18 23:40 결정 줄의 시각은 23:27(재표본 23:27~23:29, 결과 확인 11-19 08:52) ([branch_b3](report/experiment/2026_11_18_20_36_branch_b3.md))
- 11:35 — 결정(사용자)·spec: B3 다음은 30:00 재측정 B3-H(같은 시점 300 · 계획 7, 2,100판, 12:00 score도 다시 계산해 비교) → 055-R13 ([055](spec/2026_11_18_20_30_055-plan-branch-games.md), [branch_b3h](report/experiment/2026_11_19_11_35_branch_b3h.md))
- 11:36 — 결정(사용자): 보조 예측기만 AdamW eps 1e-8(안 B) — [spec 041](spec/2026_10_15_22_24_041-imitation-learning.md) R3·R23 ⑦·AC8(BC 키 `[aux] adam_eps`, RL은 시작 오류 [042-R21](spec/2026_10_15_22_24_042-rl-training.md)), gn 출발 t2 레시피로 기준 A·B 3,258 update씩, 판정 규칙·게임 평가 없음 ([aux_eps_compare](report/experiment/2026_11_19_11_36_aux_eps_compare.md))
- 17:55 — 끝: B3-H 2,099판(30:00) 무효 0 — 30:00 score 효과 모두 |D| < 0.04(구간 0 포함), 12:00 score 효과는 B3와 재현 안 됨(계획 6개 상관 −0.59), `factory_now` 8분 버티기 상대별 방향(프로토스 · 테란 +, 저그 −)만 두 실행 모두 재현 ([branch_b3h](report/experiment/2026_11_19_11_35_branch_b3h.md))
- 18:44 — 끝: 보조 예측기 eps 비교 — B는 주 손실 그대로, val500 `aux_jepa` −47%(A 0.0410 → B 0.0216)이나 JEPA 정답 흩어짐도 −30%(상수 예측 바닥 대비 A 0.40 · B 0.45), `aux_priv` −4% ([aux_eps_compare](report/experiment/2026_11_19_11_36_aux_eps_compare.md))
- 19:07 — 결정(사용자): 계획 갈래 판 측정 멈춤(줄기 확장 AMB-055-1 · 예측기 055-R8 안 함, seed 블록 AMB-055-2는 정하지 않음) · 보조 eps 안 B 보류(기본 eps 1e-5 유지) ([055](spec/2026_11_18_20_30_055-plan-branch-games.md), [branch_b3h](report/experiment/2026_11_19_11_35_branch_b3h.md), [aux_eps_compare](report/experiment/2026_11_19_11_36_aux_eps_compare.md))
- 19:22 — 마무리(사용자 요청): 계획형 JEPA 결과 정리(단계별 결과 · 예상 대조 · 결론 · 산출물) · 모델 2 리포트 11-17~11-19 요약 ([jepa_planning](report/plan/2026_11_11_14_56_jepa_planning.md) 11절, [model2](report/plan/2026_11_09_22_18_model2.md) 13절)
- 22:2x — 결정(사용자): 모델 확대 후보 B로 t2 줄기만 같은 단계 · 데이터 · update 수로 다시 학습해 Checkpoint 1 · 2 · t2와 비교(GroupNorm 처음부터 · gn 두 단계 유지, BC 2배 느려짐 조사 먼저) → 22:3x 사용자 지시로 멈춤, 코드 · spec 변경 없음 ([model_scaleup](report/bench/2026_11_17_21_45_model_scaleup.md))
- 22:4x — 결정(사용자): 공간 인코더(CNN)를 linear projection으로 바꾸는 안 검토 — 학습 없는 지워 보기 측정 먼저, B 줄기는 공간 구조를 정한 뒤로 ([spatial_ablation](report/investigation/2026_11_19_23_00_spatial_ablation.md))
- 23:01 — 시작: 지워 보기 측정(t2, val 전체 2,395판, Windows CUDA 호스트 f32) ([spatial_ablation](report/investigation/2026_11_19_23_00_spatial_ablation.md))

## 2026-11-20
- 07:36 — 끝 · 새 리포트: 요약 `s_pool` → Core는 정보가 안 쓰임(평균 · 섞기 차이 ≤ 0.0004 NLL/머리, 상수 바이어스만), 위치 머리는 CNN 특징에 크게 의존(지우면 위치 NLL 5.46 → 8.02, 4칸 안 0.49 → 0.07, 건물 배치 정확 칸 0.25 → 0.05) ([spatial_ablation](report/investigation/2026_11_19_23_00_spatial_ablation.md))
- 08:41 — 결정(사용자, 08:3x) · 새 리포트: 모델 확대 B(t2 줄기 재학습)에 공간 인코더 CNN 제거 → linear projection 변경을 더함, 인수인계 문서(구성 기본안 · 줄기 · 질문 12개 · 작업 순서) ([scaleup_linear_spatial_handoff](report/plan/2026_11_20_08_41_scaleup_linear_spatial_handoff.md))
- 09:0x — 결정(사용자): 확대 B + linear projection 질문 답 — 위치 경로 사전 확인 · 크기 대조 없음(Checkpoint 1 · 2(t2)와만 비교), linear projection은 접지 않고 위치 지표만 추적, ROCm 호스트에서도 게임 추론([060](spec/2026_10_15_22_24_060-infra.md) R3 상한 개정, 값은 실측 뒤), CNN이 빠진 만큼 더 키우지 않음, 나머지 기본안 ([scaleup_linear_spatial_handoff](report/plan/2026_11_20_08_41_scaleup_linear_spatial_handoff.md) 2 · 6절)
- 09:23 — 새 리포트: BC 2배 느려짐 원인 — UTF-8(65001) 콘솔에서 띄운 run은 엔티티 attention 직접 작성 커널을 못 불러 기준 구현으로 조용히 넘어감(t2 1.44 → 3.0초/update, 949 콘솔 재현 1.53 · 65001 3.30초), 체크포인트 메타로는 구분 안 됨, 고치는 방법은 사용자 결정 ([bc_slowdown](report/investigation/2026_11_20_09_23_bc_slowdown.md))
- 10:0x — 결정(사용자) · spec: 커널 로드 수정 승인 — 콘솔 코드 페이지와 무관하게 로드, 요청한 커널(백엔드 기본값 포함)을 못 부르면 기준 구현 대신 시작 오류, 체크포인트 메타에 실제 쓴 구현, 확인에 Linux ROCm 호스트 두 대 포함 ([060](spec/2026_10_15_22_24_060-infra.md) R1 · AC8, [041](spec/2026_10_15_22_24_041-imitation-learning.md) R15)
- 10:0x — 결정(사용자) · spec: 확대 B + 패치 경로 spec 045 Approved — 패치 위치 임베딩을 더함, 위치 지표는 val 전체 · t2 값 재사용, 나머지 세부 기본안, 1만 update 보고에 위치 머리 손실 포함 ([045](spec/2026_11_20_09_12_045-scaleup-b-patch-lineage.md), [032](spec/2026_10_15_22_24_032-model-architecture.md))
- 10:0x — 결정(사용자): 14:30까지 구현 중 생기는 결정은 메인 세션(코디네이터)이 정하고 "코디네이터 결정"으로 기록, 실작업은 서브 에이전트 ([scaleup_linear_spatial_handoff](report/plan/2026_11_20_08_41_scaleup_linear_spatial_handoff.md) 2절)
- 10:34 — 모델 코드 master 반영: `[net]` 크기 표·패치 공간 경로(1f55631b·a025ec2a·d31ea734) ([032](spec/2026_10_15_22_24_032-model-architecture.md) R28·R29)
- 10:41 — 확대 줄기 설정·r1 plan·실험 리포트 틀 master 반영(dcc51db6). update 수 다시 셈이 원래 줄기와 같음(s1 39,096 / s1e2 39,094 / s2e2 11,387 / 혼합 7 epoch 8,417) ([m2b_scaleup](report/experiment/2026_11_20_10_33_m2b_scaleup.md) 3절)
- 10:47 — 커널 로드 수정 master 반영(049b19ea·e717ccde·11a033dc): 같은 `bench bc`가 949 콘솔 1.75 · UTF-8 콘솔 1.71초/update(수정 전 UTF-8 3.3초), Linux ROCm 호스트 두 대 커널 로드 확인 ([bc_slowdown](report/investigation/2026_11_20_09_23_bc_slowdown.md) 8절)
- 10:48 — 패치 경로 주변 코드 master 반영(1ca88ada·e8af7404): BC·프로파일·벤치·RL·측정 경로의 `[net]`·패치 경로 처리 ([045](spec/2026_11_20_09_12_045-scaleup-b-patch-lineage.md) Interfaces)
- 10:51 — 측정 지점 `s2e2b`·`t1b`·`t2b`와 위치 지표 `val_loc` master 반영(d59e2626·91ac7ce2). CNN 제거 측정의 A0 계산과 val 6판에서 5묶음 × 5값 일치 ([m2b_scaleup](report/experiment/2026_11_20_10_33_m2b_scaleup.md) 7절)
- 10:53 — 확대 줄기 plan 사본을 seed 겹침 집계에서 빼는 규칙 master 반영(012fd830) ([045](spec/2026_11_20_09_12_045-scaleup-b-patch-lineage.md) R3 ②)
- 10:55 — 측정 도구 보완 master 반영(d0b62861·7abb36ac): 패치 모델의 죽은 채널 "측정 대상 없음"과 공간 요약 단계 없는 단서 진단을 정상으로 받음 ([045](spec/2026_11_20_09_12_045-scaleup-b-patch-lineage.md) R6 ⑤)
- 10:5x — 시작: B 구성 실측(최악 모양 메모리·학습 속도·추론 지연·메모리, Windows CUDA 호스트 + Linux ROCm 호스트 두 대). 파라미터 138,680,787(추론 135,957,588 + 학습 전용 보조 2,723,199), 어림 약 1억 3,870만과 일치 ([m2b_scaleup](report/experiment/2026_11_20_10_33_m2b_scaleup.md) 2절)
- 11:01 — `s2e2b` 단서 진단을 선택 측정으로 바꾼 정의 master 반영(304cb9bc) ([045](spec/2026_11_20_09_12_045-scaleup-b-patch-lineage.md) R6 ①)
- 10:2x~11:0x — 결정(코디네이터, 사용자 위임): 크기 설정 표 엄격 규칙 — CNN 설정에 패치 폭 키 오류, `entity.dropout`이 체크포인트와 다르면 오류(안 주면 체크포인트 값), 정수 전부 0 이하 오류·dropout [0, 1) ([032](spec/2026_10_15_22_24_032-model-architecture.md) R28·R29)
- 10:2x — spec 본문 보완(새 결정 아님): RL 설정에 크기 표가 있으면 시작 오류 — 이미 승인된 기본안이 요구사항 본문에 빠져 있던 것 ([032](spec/2026_10_15_22_24_032-model-architecture.md) R28 ⑤)
- 10:2x~11:0x — 결정(코디네이터, 사용자 위임): 줄기 실행 세부 — plan 사본 seed 집계 제외, 교사 판 6폴더 공용 사본, gn update = t1 한 epoch·warmup 5%, 표본 끝-끝 이름·폴더 분리·진단 생략, `s2e2b` 단서 진단은 선택, 측정 도구가 패치 모델의 "측정 대상 없음"을 정상으로 받음 ([045](spec/2026_11_20_09_12_045-scaleup-b-patch-lineage.md) R2~R6)
- 11:23 — 새 리포트: B 구성 실측(Linux ROCm 호스트) — 최악 모양 학습은 엔티티 재계산 끄면 13.4 GiB 이상(예산 12 초과), 조각 16 8.9 GiB·63.4초/update, 실데이터 23.8초. 추론 서버 3.18 GiB(엔티티 64~256)·4.60(64~512), 배치 순서에 따라 할당자 조각화로 6.0~8.2 GiB까지 불어남(`expandable_segments` 켜면 2.8~3.7) ([m2b_measure](report/bench/2026_11_20_11_23_m2b_measure.md))
- 12:2x — 끝: B 구성 실측(Windows CUDA 호스트) — 재계산 끔 + compile `default` + 레인 그룹 2가 2.81초/update·GPU 11.63 GiB(최악 5.67초), 재계산 + `reduce-overhead`는 손실 non-finite·컴파일 미완, 학습 시간 어림 약 3.7~7.5일 + 게임. 발견: 학습기의 non-finite update 건너뛰기에 연속 제한이 없어 이 경우 멈추지 않음, `bench bc` None 손실 결함 (상세는 [m2b_measure](report/bench/2026_11_20_11_23_m2b_measure.md) CUDA 절)
- 12:2x — 결정(코디네이터, 사용자 위임): 엔티티 재계산 Linux ROCm 호스트 조각 16·Windows CUDA 호스트 끔, B 추론 서버 상한 3.5 GiB·한 호스트 학습과 게임 추론 동시 실행 안 함, 줄기 ROCm 추론만 할당자 설정, 줄기 스크립트 감시, 표본 게임에 Linux ROCm 호스트 포함. 열린 항목 4개는 사용자 결정 ([045](spec/2026_11_20_09_12_045-scaleup-b-patch-lineage.md) R4·R9, [m2b_scaleup](report/experiment/2026_11_20_10_33_m2b_scaleup.md) 6절)
- 12:2x — 시작: 표본 끝-끝(판 → 재레이블 → shard → 혼합 몫 → BC 50 update, 게임은 Linux ROCm 호스트 포함) ([m2b_scaleup](report/experiment/2026_11_20_10_33_m2b_scaleup.md) 6절)
- 13:1x — 발견: Windows CUDA 호스트 학습에 할당자 설정 `expandable_segments`를 켜면 속도 그대로 GPU 전용 11.63 → 9.44 GiB(최악 모양 11.43). 표본 학습에서는 설정 전 15.64 GiB로 공유 메모리에 넘치기 시작 → 설정 뒤 10.44 GiB. 학습 시간 정상 상태 어림 약 4.6일(벤치 3.7, 상한 7.5) ([m2b_measure](report/bench/2026_11_20_11_23_m2b_measure.md) CUDA 4·5·7절, 표본 수치는 [m2b_scaleup](report/experiment/2026_11_20_10_33_m2b_scaleup.md) 7절)
- 13:0x~13:3x — 결정(코디네이터, 사용자 위임): 표본·실제 줄기는 같은 실행 스크립트, 재레이블·집계는 Windows CUDA 호스트, 그 호스트 학습에 할당자 설정, 긴 단계는 분리 실행 + 스크립트 안 감시(non-finite 손실·여유 메모리·GPU 넘침 자동 재시작), Linux ROCm 조각 16은 스크립트에서, 레인 그룹 1 재측정 안 함, 표본 남은 단계 연속 실행 ([045](spec/2026_11_20_09_12_045-scaleup-b-patch-lineage.md) R4·R9)
- 13:3x — 시작: 표본 끝-끝 남은 단계 연속 실행(실패하면 그 단계에서 멈춤) ([m2b_scaleup](report/experiment/2026_11_20_10_33_m2b_scaleup.md) 7절)
- 14:01 — 표본 끝-끝 진행: 사람 데이터 BC·손실 곡선·r1·r2·r3 판·d1·t1·gn 두 번 통과, t2·측정 진행 중 ([m2b_scaleup](report/experiment/2026_11_20_10_33_m2b_scaleup.md) 7절)
- 14:34 — 끝: 표본 끝-끝(M2B-5) 통과 — 모든 단계 종류 통과, 할당자 설정 뒤 학습 GPU 전용 9.4~10.4 GiB, 스크립트 결함 6건 고침, 본 줄기 시작은 사용자 확인 대기 ([m2b_scaleup](report/experiment/2026_11_20_10_33_m2b_scaleup.md) 7절)
- 15:10 — 시작(사용자 확인): m2b 본 줄기 사람 데이터 단계(s1 → s1e2 → s2e2 → 혼합 7 epoch, 약 98,000 update, Windows CUDA 호스트, 분리 실행) — 첫 update부터 손실 유한, 초반 약 2.2초/update, GPU 전용 6.6 GiB, 학습 시간 어림 약 3.4일. 첫 DAgger부터는 보충 판 경로를 넣은 뒤 따로 시작 ([m2b_scaleup](report/experiment/2026_11_20_10_33_m2b_scaleup.md) 7절)

## 2026-11-21
- 02:1x — 보고: m2b 1만 update 학습 곡선 — 정책 머리(행동·지연·선택·대상 유닛)는 원래 m2-s1과 같은 수준, 위치 머리 +0.32(3,500 무렵 +0.42에서 줄어드는 중), 가치 머리 +0.22로 느리게 내려감(원인 미확인), 학습 계속 ([m2b_scaleup](report/experiment/2026_11_20_10_33_m2b_scaleup.md) 8절)
- 09:22 — 새 리포트: m2b 가치 머리 학습 손실 상승 — val 30판에서는 B와 원래 모델이 같은 수준(1.7만 update 가치 MSE 0.99 대 0.90, 1만 1.50 대 1.48), 출력 포화·발산 아님, 학습 손실만의 효과로 봄(추정). 두 모델 모두 val에서 상수 예측 수준을 못 넘는 것은 기존 문제. 평가 중 Linux ROCm 1호기 응답 없음(재부팅 필요) ([m2b_value_head](report/investigation/2026_11_21_09_22_m2b_value_head.md))

## 2026-11-23
- 03:56 — 중단: 정전으로 m2b-s1e2 학습 멈춤(update 24,534) → 19:50 ckpt-24209부터 재개 ([m2b_scaleup](report/experiment/2026_11_20_10_33_m2b_scaleup.md) 8절)

## 2026-11-24
- 18:1x — 중단(사용자 지시): m2b-s2e2 update 8,136에서 정상 종료 → 22:4x 재개, 공용 드라이브 NAS 이전 반영 ([m2b_scaleup](report/experiment/2026_11_20_10_33_m2b_scaleup.md) 8절)

## 2026-11-25
- 09:38 — 끝: m2b 사람 데이터 단계(s1e2·s2e2·혼합 7 epoch) — 원래 줄기 대비 학습 손실 정책·가치 머리 같거나 낮음, 위치 머리 차이 +0.32(1만) → +0.01(mix-e7 끝) ([m2b_scaleup](report/experiment/2026_11_20_10_33_m2b_scaleup.md) 8절)
- 11:0x — 시작: m2b 첫 측정 지점 s2e2b와 DAgger r1 판 ([m2b_scaleup](report/experiment/2026_11_20_10_33_m2b_scaleup.md) 4·6절)
- 14:06 — 결과: m2b 첫 측정 지점 s2e2b — val 500 행동 top-1 72.58%(원래 72.42%), 위치 NLL +0.17 나쁨; val 전체 위치 지표는 t2와 비슷(NLL +0.02, 4칸 안·건물 정확 칸·랠리 오차는 더 좋음) ([m2b_scaleup](report/experiment/2026_11_20_10_33_m2b_scaleup.md) 8절)
- 14:06 — 시작: m2b DAgger r1 학습(1,267 update, r1 판 582판 기록) ([m2b_scaleup](report/experiment/2026_11_20_10_33_m2b_scaleup.md) 3절)
- 16:42 — 끝: m2b DAgger r1(15:12, 1,267 update)·r2(1,390 update, 판 294판 기록) 학습 → r3 판 시작 ([m2b_scaleup](report/experiment/2026_11_20_10_33_m2b_scaleup.md) 3절)
- 18:2x — 시작: m2b 증류 d1 학습(1,948 update; r3 294판·d1 1,100판 기록, s2e2b 레이블 905,013 step) — 레이블 첫 실행은 test 측정과 겹쳐 메모리 부족 중 CUDA 오류, 단독 재실행 성공 ([m2b_scaleup](report/experiment/2026_11_20_10_33_m2b_scaleup.md) 3절)
- 19:1x — 정리(사용자 지시): Linux ROCm 호스트 두 대·노트북 Linux 호스트를 비움(hostd 중지, 저장소 사본 삭제). 각 장비 `local/work`·`local/scripts`·`runs`·`.env`는 공용 드라이브 `host_backup/<장비>-20261125.tar.zst`(파일 수·바이트 일치 확인) — 문서의 그 장비 `local/work` 경로는 이 파일 안 같은 경로. m2b 줄기 판은 이 PC hostd만, ROCm 고정 측정은 장비 없음
- 19:48 — 중단(사용자 지시): m2b 증류 d1 학습 update 1,727/1,948에서 정상 종료(체크포인트 `m2b-distill-d1/ckpt-1727`, 이어서 재개 가능). 모든 작업 대기 ([m2b_scaleup](report/experiment/2026_11_20_10_33_m2b_scaleup.md) 3절)

## 2026-11-26
- 00:21 — 끝: m2b 증류 d1 학습(ckpt-1727에서 재개, 1,948) → t1 판 300판(이 PC만, 무 248·패 52, 원래 214·86)
- 01:49 — 결과: m2b s2e2b test — 행동 top-1 72.33%(원래 72.14%), 위치 NLL +0.15 나쁨, val 500과 같은 방향. 이어서 t1 학습 시작(2,845 update) ([m2b_scaleup](report/experiment/2026_11_20_10_33_m2b_scaleup.md) 8절)
- 04:15 — 끝: m2b 교사 t1 학습(2,845 update, 2시간 26분, 재시작 0, GPU 전용 최대 13.02 GiB) → gn-t1 학습 시작(t1과 같은 데이터) ([m2b_scaleup](report/experiment/2026_11_20_10_33_m2b_scaleup.md) 3절)
- 09:08 — 끝: m2b gn-t1(06:42)·gn-t1e2 학습(각 2,845 update, 재시작 0) → t2 판 plan 커밋(원래 plan과 이름·체크포인트만 다름), t2 판 900판 시작(이 PC만) ([m2b_scaleup](report/experiment/2026_11_20_10_33_m2b_scaleup.md) 3절)
- 10:06 — 시작: m2b 교사 t2 학습(3,246 update, 원래 3,258). t2 판 900판 09:50 끝(상대 Iron 비정상 종료 1판 중단), 재레이블 기록 298 + 578판(원래 299 + 575) ([m2b_scaleup](report/experiment/2026_11_20_10_33_m2b_scaleup.md) 3절)
- 12:50 — 끝: m2b 교사 t2 학습(3,246 update, 2시간 45분, 재시작 0). m2b 줄기 학습 모두 끝 → t1b·t2b 측정 시작(이 PC만; ROCm 고정 val 500은 CUDA f32, 단서 진단은 CUDA bf16·Windows CPU probes로 따로 저장) ([m2b_scaleup](report/experiment/2026_11_20_10_33_m2b_scaleup.md) 8절)

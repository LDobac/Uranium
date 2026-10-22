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
- 2026-10-19 01시 — **결정**: git 운영은 worktree → rebase → origin master push를 유지하고, push 뒤 메인 작업 사본도 pull --ff-only로 맞춘다(로컬 master가 계속 뒤처지던 문제). 기존 규칙의 "사용자 결정" 표기는 사용자 지적(merge 커밋)을 세션이 절차로 옮긴 것이었음을 바로잡음 (CLAUDE.md git 통합)
- 2026-10-19 — **결정**: CLAUDE.md는 짧게 유지하고 진행 상황·날짜·경위는 쓰지 않는다(링크만). 명령 상세는 `docs/commands.md`로 옮김
- 2026-10-19 — **결정**: BC 전 리뷰 결정 G-D1~D9·A-2·A-3·C-3·C-9·060-R7·H·E 확정(전체 재build, Train 연타 병합 제거, 첫 LEAVE 이탈자 패배·판정 불가 value 가중치 0, shard 로컬 우선 쓰기·재읽기 검증, S2 품질 기준 미적용(041-R13), 전역 norm clip 1.0, CUDA·Vulkan autotune+fusion, `keep_checkpoints`·코어 비교 전부 보존, 061-AC1·AC5 기준, val 고정 seed 셔플 500게임, 장비 표기는 OS·백엔드만, 020-R3 30초 중단 삭제, RL 리플레이 삭제 안 함, 디스크 사용률 80% 경고, 작업자 H-1~H-4 코드 수정, E절 관측·마스크 항목), spec 일괄 개정(005e6fe)
- 2026-10-19 — **결정**: T1 리뷰 후 추가 결정 D-a(032-AC8 CUDA backward 허용치는 autotune 켠 재측정 후 확정), D-b(040-R8 기록 승자를 판정 불가 전 단계로 사용, `result_source = recorded`), D-c(Windows 자동 시작 없음), D-d(수동 `--take-over` 즉시 인수 + 경고, 단조 시계 관측은 작업자 자동 인수만). 리뷰 지적(건물 단일 선택 전 액션, 자원 출처 = 맵 사전계산 AMB-030-3, 계통 복구 조건, autotune 캐시, 060-AC3 대체)과 함께 spec 보완
- 2026-10-19 — **측정**: 032-AC8 CUDA backward 재측정(autotune+fusion, 6회) 최대 4.94e-3 → CUDA 기준 1e-2 제안(D-a, 사용자 확인 필요). 같은 장비·같은 캐시여도 CUDA 결과가 회마다 다름(Windows에서 CubeCL 영구 캐시 미저장, atomic add) — `reports/m6_bc_preflight.md` 5차
- 2026-10-19 — **결정·구현**: E-8 `GameEnd.final_frame`(010-R12, 011-R19, 012-R10, PROTOCOL_VERSION 2)으로 마지막 결정 뒤 dense 보상을 마지막 transition에 합산. CUDA 호스트는 같은 PC여도 032-AC1 허용 오차로 비교(041-R15·061-AC1), Windows autotune 캐시 미저장은 알려진 제약. CUDA AC8 1e-2 확정. Linux 브리지 빌드·OpenBW 1판에서 final_frame 수신 확인 — `reports/m6_bc_preflight.md` 5차
- 2026-10-19 13시 — 재빌드(G-M1, R-1): 학습 PC·Linux 호스트 A·B를 6523286으로 재빌드(브리지 4인자 ABI·proto v2, hostd 재시작). `.env`에 `URANIUM_SHARD_STAGING_DIR`·`URANIUM_AUTOTUNE_CACHE_DIR` 추가. 시작 전 공용 드라이브 사용률 82%로 중단 → 사용자 worktree 정리 후 64% (`m6_rebuild.md`)
- 2026-10-19 14시 — 표본 end-to-end(R-2): 리플레이 37개, 투혼 CP949 8개 → fighting_spirit, result none 10.8%, 스텝 +2.4%(병합 제거), 첫 스텝·is_enemy·자원 사전계산 v2·staging 정상, `--jobs 1` 장비 간 shard 바이트 동일. BC 스모크 CUDA 약 10.6 s/update, Vulkan 약 86 s/update(Linux 호스트 A 스왑 최대 13.3GB, G-M2), Vulkan 032-AC8 5.81e-4, proto v2 실게임 final_frame 확인. 표본 폴더 삭제 (`m6_rebuild.md`)
- 2026-10-19 14시 — **발견**: 표본 scr_ladder 무효 레이블 40.3%(APM 2,376·957 두 시점이 대부분, 나머지 출처 2.3–3.9%). APM 상한·제외 결정 필요. 에이전트 장비 `URANIUM_MAPS_DIR`에 대상 맵 v2 사전계산이 없어 mapgen 필요 (`m6_rebuild.md`)
- 2026-10-19 16시 — **결정·구현**: 040-R19 무효 레이블 비율 ≥ 20% 테란 시점은 build에서 `filtered`(사용자 결정, AMB-040-11, 데이터 품질 기준이며 APM 상한 아님). 레이블 검사를 `uranium_env::features::label`로 옮겨 build·BC 로더·`label-check`가 같은 함수 사용. 표본 scr_ladder 8리플레이 재build(학습 PC, `--jobs 1`): 비정상 두 시점 54.6%·24.6%로 제외, 나머지 written. 검사 비용 약 0.54 ms/스텝, build 벽시계 189초 중 29.5초(15.6%). 출력은 학습 PC 로컬 `sc_data/t3c`. 재개를 (리플레이, 시점) 단위로 바꿔 flush 전 중단 시 시점이 빠지던 구멍을 고침 (`m6_rebuild.md`)
- 2026-10-19 16시 — 재build 준비(R-3): 세 장비 d452c29 재빌드(hostd는 Linux 호스트 A만 바이너리가 바뀌어 재시작). 옛 shards·records(fix-desync 포함)·build claims/logs/jobs/maps·quarantine, 학습 PC v1 맵 사전계산·스캔 캐시·옛 build 바이너리 사본, Linux 호스트 A BC shard 캐시 삭제(학습 PC 39.6GB, Linux 호스트 A 39.3GB 확보). 같은 내용의 `jobs.txt`(작업 77개)와 빈 폴더 재생성, 큐 미시작 (`m6_rebuild.md`)
- 2026-10-19 16:35 — **장시간 실행 시작**: 전체 재build(R-4), 세 장비 d452c29(Linux 호스트 A·B jobs 8, 학습 PC jobs 12는 권한 승인 뒤 16:59). 첫 파트 정상(stardata written/filtered 0.97·1.16, scr_ladder 570/1,494, 쓰기 검증 실패 0). 040-R19 걸러짐 stardata 0.5% 이하, scr_ladder 5.9%(36). Linux 호스트 A·B 대상 맵 v2 mapgen 완료. 완료 예상 10-20 오전 (`m6_rebuild.md`)
- 2026-10-19 16시 — 래더 수집 시작(043): `uranium-ladder collect --plan config/ladder/kr_asia.toml`(학습 PC, d452c29 빌드). 추적 대상 gw30 3,185명·gw45 462명, raw index 82,630개. 앞선 dry-run이 멈춘 것처럼 보인 것은 출력 파이프(`| tail`) 버퍼링 때문이었고 API 감지는 정상
- 2026-10-19 18시 — 래더 수집 종료: 일일 상한 도달, ok 1000·skipped 0·failed 37(`schema: missing field score` 34, `game_info` 3). 오늘 받은 1000개 중 길이 5분 이상 78.3%(783개). 누적 ok 3000·failed(schema) 105. 로그는 작업 tmp `ladder_collect_20261019.log`
- 2026-10-19 18시 — **결정·구현**: 043-R9 예외(사용자 결정) — match info 플레이어에 `score`가 없으면 경기를 수집하고 MMR·랭크 칸만 비움, 다른 스키마 불일치는 그대로 failed. `collect --retry-failed [사유 접두어]` 추가(최신 기록이 해당 사유의 failed인 경기만 재조회, 043-R4 한도 동일, `status`는 경기별 최신 기록). score 실패 81경기(10-16 7·10-17 40·10-19 34)는 10-20 일반 수집 전에 재시도. 표본 match info 재조회에서 replays URL은 서명·만료 파라미터 없음
- 2026-10-19 18:31 — R-4 전체 재build에 Linux 추출 호스트 C 합류(d452c29 바이너리를 그 호스트에서 빌드, jobs 8). 합류 전 표본(scr_ladder 12, stardata 20)을 `--jobs 1`로 Linux 호스트 A와 비교해 shard·records 바이트 동일
- 2026-10-19 — T7 미검증 AC: 010-AC4·043-AC4 테스트, 031-AC1 OpenBW 시나리오 실게임 100/100(비무장 Attack_Pos → Move 확인), 013-R7 build 경로 확인. 060-AC4·032-AC2는 재build 뒤 측정(`m6_unverified_ac.md`)
- 2026-10-19 — **결정·구현**: 034-R8 애드온 건설 중인 건물은 생산·연구·업그레이드·이륙 후보에서 제외, 034-R9 유닛 대상 액션은 선택 유닛을 대상으로 못 함(SetRally_Unit 예외, 확인 필요). 재build 없음(build의 040-R19 판정은 옛 마스크, 무시할 수준). 표본 무효 레이블 5.384% → 5.427%, 시나리오 재실행 100/100
- 2026-10-19 — **결정**: 034-R9 SetRally_Unit 자기 대상 예외 사용자 확정(2026-10-19). 건물 자신을 랠리 대상으로 하는 랠리 해제를 허용(`self_target = true`)
- 2026-10-19 20시 — 액션 전수 감사(034·031·030·032·040·041-R11·013·010·011, 네 경로 + OpenBW·BWAPI 소스 대조): `reports/m6_action_audit.md`. High 2(대상 자격이 생산 중·탑승 아군 포함 = T7 오류 6 원인 / shim이 시야 밖 한 번 본 자원 대상을 못 찾음), Medium 6(유닛별 BWAPI 호출·Stop 시즈 탱크·Unknown→NoOp·max_select=1 첫 유닛·target_unit_missing·Build 위치 거절). 재build 중단·재실행 불필요. 커버리지 99.79%(records 6,768 시점), 표본 2 shard 무효 레이블 5.853%
- 2026-10-19 23시 — **구현(T9, 브랜치 `t9-unload-unit`, 리뷰 대기)**: 액션 감사 F7·F12 사용자 결정 반영. `Unload_Unit`(id 100, action 스키마 2, feature 5, `PROTOCOL_VERSION` 3), 미분류 명령 `unknown` 레이블(가중치 0), 선택 전체 저장 + 검사 자격 필터 후 자르기, shard schema 6 `labeler_version`(schema 5 = 레이블러 1로 계속 읽음, 재build 혼합). 표본 80리플레이: SpiderMine `select_cap` 무효 9.9% → 0.35%, unknown 70 → 26(0x29 44건이 `Unload_Unit`), 무효 레이블 7.015% → 6.878%. Linux 브리지 빌드 확인 (`m6_action_audit.md` §10)
- 2026-10-19 — **결정**: `unknown` 레이블(레이블러 2)은 040-R19 무효 비율과 040-AC7 무효 집계에서 제외(사용자 확정). spec 040-R19·`m6_action_audit.md` §10 반영
- 2026-10-19 — T9 리뷰 반영: `Unload_Unit` 선택 자격을 `is_transport_loaded` 재사용(K 47 → 48), BC 검사에 `target_not_in_transport`, 034 Risks(수송선↔탑승 유닛 연결 feature 없음). 표본 산출물 `<URANIUM_SHARED_DIR>/artifacts/m6_action_audit/t9_sample/`. 101개 031-AC1 실게임은 T8 반영·브리지 재빌드 뒤로 보류
- 2026-10-20 00:10 — **장시간 실행 재개**: R-4 전체 재build, 보류한 50파트(`hold-f7f12`)를 master aa187b2 바이너리(레이블러 2, shard schema 6)로 네 장비에서 재개(사용자 결정 2026-10-20, 학습 PC jobs 12, Linux 8). 옛 레이블러 27파트(scr_ladder p0, stardata p00–p25) / 새 레이블러 50파트. 표본 3리플레이 records·shard 레이블러 2 확인. Linux 호스트 A·B hostd와 그 브리지는 그대로. 완료 예상 10-20 오전 10시경 (`m6_rebuild.md`)
- 2026-10-19 23시 — **결정·구현**(사용자 결정 2026-10-19, 액션 감사 T8): 대상 자격에서 생산 중·탑승 유닛 제외(034-R10, F1), 브리지가 시야 밖 한 번 본 자원 대상을 맵 자원에서 찾고 대상·위치 없음은 즉시 CommandFailed, 다수 유닛 명령은 그룹 명령·1기 이상 수락이면 성공(011-R20, F2·F3·F17), Stop `can_stop`·UnloadAll_Pos `can_unload_pos`·벙커 비공격·건설 중 SCV·시즈 전환 중 탱크 제외·`pick_worker` 중단 불가 SCV 제외(F5·F6·F11·F15·F16), feature 스키마 버전 6(T9 위). 031-AC1 판정 강화(정당화된 거절만) 실게임 100/100 통과. F23·F24는 scr_ladder 일부 리플레이(배치 실패 명령 기록·재생 어긋남)로 마스크 문제 아님. F7·F12는 T9, 나머지 레이블러 항목은 "다음 재build 목록", 결정 대기 U-1~U-6 (`m6_action_audit.md` §11)
- 2026-10-19 — T8 리뷰 반영: 알·코쿤·합체 중 아콘은 미완성이어도 대상(034-R10), U-4 사용자 결정(시즈 전환 중 연타 레이블은 BC 유효, 라이브만 차단, `AgentMemory::replay_labels`), 사라진 시야 밖 자원 대상은 실패 대신 마지막 위치로 이동(메인 기술 결정, 011-R20), 메딕 적 우클릭은 HealMove. 011-AC4 그룹 명령 결정론 실측 통과(같은 seed·Act 10분 게임 2회, 관측 1,800개 체크섬 동일). label-check 재측정 scr_ladder 5.670%·stardata 3.850%. 산출물 `<URANIUM_SHARED_DIR>/artifacts/m6_action_audit/t8/`
- 2026-10-20 08:5x — 래더 점수 누락 재수집(043-R9, `collect --retry-failed "schema: missing field \`score\`"`, 학습 PC, master f513b92 빌드): 대상 81경기 중 ok 77, 4경기는 이번엔 `game_info` 누락으로 failed 유지. 요청 160회 0.5 req/s 이내(043-AC4). 로그는 작업 tmp `ladder_retry_20261020.log`
- 2026-10-20 10시 — 래더 일반 수집 종료: 추적 대상 gw30 3,203명·gw45 471명, ok 923·failed 4(`game_info` 누락), 재수집분과 합쳐 일일 상한 1,000 도달. 누적 ok 4,000·failed(schema) 32. 로그는 작업 tmp `ladder_collect_20261020.log`
- 2026-10-20 10:08~10:35 — scr_ladder 증분 build(Linux 호스트 B, f513b92 바이너리, jobs 8, 학습 PC의 08:35~08:58 증분 뒤 순차 실행): 새 처리분 written 285·filtered 738·desync 19·error 0(records 3,219 → 4,261줄), 새 shard `scr_ladder-p0/shard_00063`~`00080`. 이 바이너리는 CIFS 쓰기 검증 수정(7b0879e) 이전이라 scr_ladder 전체 shard 디코드 검사·손상분 재build는 BC 전 리뷰 수정 세션이 맡음. 공용 드라이브 mtime이 Linux 호스트 B 시계보다 약 6~8분 늦음
- 2026-10-20 07:56 — **장시간 실행 끝**: R-4 전체 재build 77파트 모두 `.done`, 네 장비 큐 `queue empty`(레이블러 1 27파트 / 레이블러 2 50파트, 040-R19 판정은 액션 감사 수정 전 마스크 — 받아들임)
- 2026-10-20 08:35~08:58 — scr_ladder 증분 build(학습 PC, f513b92, jobs 12): 10-19 수집분 1,000개, written 378·filtered 694(040-R19 39)·desync 38·error 1, 새 shard `scr_ladder-p0/shard_00043`~`00062`
- 2026-10-20 09시 — **발견**: 040-R17 검증을 통과한 shard 7개(stardata p00 2·p03·p08·p27, scr_ladder 2)가 공용 드라이브에서 4MiB 단위로 0으로 채워지거나 잘림. 모두 Linux 호스트 A·B(CIFS, `cache=strict`, `wsize` 4MiB)가 쓴 것, 커널 로그에 CIFS 메시지 없음. 클라이언트 캐시에서 다시 읽어 검증이 통과한 것으로 판단. 유실 132시점 (`m6_rebuild.md` R-5)
- 2026-10-20 09:28~09:56 — **결정·구현**(사용자 결정, AMB-040-12): 040-R17·061-R4 다시 읽기는 fsync 뒤 캐시 우회(Linux `O_DIRECT`, Windows `FILE_FLAG_NO_BUFFERING`), manifest 전 shard 전체 디코드 검사, 새 shard zstd 체크섬, 040-R20 같은 리플레이 파일은 출처 전체 목록 기준 첫 파일만 한 게임, `shard-stats` 하위 폴더·중복 검사, `dataset-stats` 키에 출처. master 67488fc·1ce972c·7b0879e. Linux 마운트 설정은 그대로(사용자 결정)
- 2026-10-20 10:2x~10:36 — R-5 수리(학습 PC, 7b0879e): 손상 shard 7개 삭제·written records 132줄 제거(원문·목록 `build/r5-backup/`), 해당 게임 재개 build로 132시점 모두 다시 written, personal 전체 재build(같은 리플레이 사본 1개가 두 게임이던 것 해소, 옛 결과는 `build/r5-backup/personal/`)
- 2026-10-20 11시~12시 — 재build 검증 통과: 네 출처 manifest 생성(전 shard 디코드)·`--verify`, `shard-stats` is_enemy·game_info·첫 스텝·중복 위반 0, shard 게임 수 = records written(36,445시점). 투혼 10,476게임/11,211시점, 서킷브레이커 725/781, result 0 20.36%, privileged 없음 0%. `reports/dataset_stats.md`(040-AC2·AC4)
- 2026-10-20 12시 — **결정 반영**(041-R14): `bc-epoch`로 `[steps]` 갱신 s1 36,985 → 38,584, s2·ablation_a 7,915 → 11,319, warmup 396 → 566(`config/bc.toml`, `bc_core_ablation.toml`은 s1 5,300 유지)
- 2026-10-20 12:2x~12:40 — BC 스모크(새 데이터 네 출처, autotune·fusion·전역 clip): 학습 PC CUDA 6 update(첫 237초, 이후 약 10초/update), Linux 호스트 A Vulkan 3 update(첫 124초, 이후 약 101초), loss 유한. `--dataset`과 `--shards`는 첫 배치 순서가 다름(장비 차이 아님)
- 2026-10-20 12:40 — 고정 build 바이너리 정리: 학습 PC `uranium-build/{aa187b2,d452c29}`, Linux 호스트 A·B 같은 사본과 `bridge/openbw/build-aa187b2`, Linux 호스트 C `src-*` worktree 2개(약 1.2GB)와 사본. 학습 PC `URANIUM_MAPS_DIR`에 build 맵 사전계산 1,244개 복사·대상 맵 mapgen(v2 확인)
- 2026-10-20 — **결정**(U-5, 040-R19): 무효 레이블 20% 기준 유지, scr_ladder 보조 필터 없음. 걸러진 시점 약 36.6k 중 216(stardata 132 0.4%, scr_ladder 83 6.3%, personal 1, 대부분 20~25%). scr_ladder 비율이 높은 이유는 나중에 조사
- 2026-10-20 — **발견·수정**(041-R13, 061-R10): `--dataset`/`--shards` 첫 배치 차이의 원인은 폴더를 적힌 순서대로 이어 붙인 뒤 섞은 것(`--dataset`만 이름 정렬). 실제 데이터로 재현(정렬 `1491…`, stardata·community·scr_ladder·personal 순서 `46ae…`). 파일별 `<출처>/<파트>/<파일>` 키 정렬로 바꿈(041-R13 문구 추가, 사용자 결정 2026-10-20). `--dataset` 순서·지문은 그대로, 정렬 안 된 `--shards`만 바뀜. 같은 출처·같은 파일 두 번 지정과 build 배치(`<출처>/<파트>/shard_*`)가 아닌 폴더는 실패
- 2026-10-20 — BC 전 체크리스트 상태 갱신과 코어 비교 선행 조건 점검(결정 없음): G절 수정 G-F1~F9·재빌드·문서 완료 표시, 남은 것 G-M2·G-M5·060-AC4/032-AC2(측정 중), G-M4는 코어 비교 첫 24시간. 증강 4종·041-R6 가중치 없음·설정 동기화 확인, 열린 AMB 없음. 맞대결·T0 경로는 proto v3·스키마 6 이후 미시험, 게임 호스트 브리지 재빌드 기록 없음 (`m6_pre_bc_review.md` G절 결과, `m6_core_ablation.md` 5절)
- 2026-10-20 00:5x — 게임 호스트 재배포(master f513b92, PROTOCOL 3): Linux 호스트 A·B 브리지·hostd 재빌드·재시작, 학습 PC hostd·agent·runner·train(cuda)·Windows 브리지 재빌드. Linux 호스트 A 실게임: 101/101 액션 종류, 같은 seed 결정론 통과(묶음 명령 2=2), 학습 PC 에이전트 → 호스트 A 연결 게임 오류 0.
- 2026-10-20 14시 — **측정**(060-AC4, 032-AC2): 학습 PC CUDA 벤치 통과(정확도 모두 ≤ 1e-2, 하네스가 CUDA에도 1e-4를 쓰던 것 수정), 실제 데이터 BC 중앙값 11.1초/update·데이터 대기 0·GPU 최대 약 9.7GB(`bench_cuda.md`). 추론 p95(B16·N256) 학습 PC CUDA 51/57ms < 같은 PC Vulkan 78/90ms로 통과, Linux 호스트 A Vulkan f32 179/187ms(f16 147/156ms)로 초안 40ms의 약 4.5배 → Linux 기준은 사용자 결정 대기(`m6_unverified_ac.md`)
- 2026-10-20 — **결정**(032-AC2, AMB-032-8): 학습 PC(CUDA)는 항상 유휴가 아니므로, 유휴일 때는 학습 PC가 주 장비, 아닐 때는 Linux 호스트 A가 주 장비. AC2 판정은 CUDA 호스트만(51/57ms < 같은 PC Vulkan, 통과), Linux Vulkan 호스트는 실측(f32 179~187ms)만 기록하고 초안 40ms 폐기. 주 장비 규칙 060-R8 신설(BC는 작업자 우선순위·양보로 이미 구현, RL 적용은 M6). C-8 032-AC2 완료 (`m6_unverified_ac.md`)
- 2026-10-20 15시 — **확인**(032-R17, 020-R11): 코어 비교 대전 경로 스모크 통과(지금 바이너리, 무작위 LSTM·GTrXL 체크포인트, 6,000프레임). 맞대결 2판(맵당 1판, 슬롯 교대)·T0 3판(T/P/Z) 모두 GameEnd, 세션 오류·시야 위반·executor 거부 0, 두 코어 모두 판마다 명령 400여 개, 플레이어 이름 기준 체크포인트 라우팅 확인, summary.md 정책별 합산 표 생성. 출력 `runs/eval-smoke-2026-10-20/` (`m6_core_ablation.md` 5절)
- 2026-10-20 15:21~15:51 — **장시간 실행 시작**: 코어 비교 학습(032-R17, 사용자 지시). 네 장비 master 2639038 빌드, 큐 `<shared>/queue/core-ablation.toml`. 학습 PC(CUDA, 사용자 직접 실행) `core-abl-lstm` 약 10~14 s/update, Linux 호스트 A(Vulkan f32, tmux `train`) `core-abl-gtrxl` 약 78~112 s/update. Linux 호스트 A는 맵 사전계산 v2가 없어 시작 검사(B-2)에서 2회 실패 → build 맵 사전계산 복사 후 통과. GTrXL Vulkan에서 autotune 후보 `matmul_naive`가 프로세스 시작마다 1회 panic(Cube count too big), 학습은 계속됨. 게임 호스트 hostd는 학습에 쓰지 않아 모두 꺼 둠(평가 때 켬), Linux 추출 호스트 C calibrate 동시 3게임 (`m6_core_ablation.md` 5절)
- 2026-10-20 16:43 — **장시간 실행 중단**: Linux 호스트 A GTrXL run이 커널 OOM으로 종료(step 31, 체크포인트 전이라 step 3~31 유실, `ckpt-2` 남음). 원인은 BC batch(약 4 GB) 최대 4개 동시 상주(미리 읽기 깊이 2, G-M2)와 GPU와 나눠 쓰는 RAM (`m6_core_ablation.md` 5절)
- 2026-10-20 18시 — **결정·구현**(041-R16, 사용자 결정): 호스트별 미리 읽기 깊이 `URANIUM_BC_PREFETCH`(기본 2, 0 = 데이터 스레드 없음, batch 동일·설정 지문 불변). master c971b1f. Linux 호스트 A만 0으로 두고 재빌드
- 2026-10-20 18:20~19:34 — 측정(G-M2): 깊이 0으로 `ckpt-2`부터 45 step 완료, OOM 지점 통과. 힙 스왑 최대 5.7 GB(깊이 2는 11.6 GB에서 안전 정지), GTT는 큰 batch에서 11.3 GB까지 커지고 유지, 남은 스왑 최소 8.05 GB. 19:36 GTrXL run 재개(`ckpt-2`부터). 출력 Linux 호스트 A `~/diag/g-m2/`
- 2026-10-20 20:52 — **장시간 실행 중단**: Linux 호스트 A GTrXL run이 GPU 버퍼 할당 실패(2 GiB)로 종료, `ckpt-38`부터 재시작했으나 같은 batch(step 48)에서 재실패, 작업자 정지. 원인은 엔티티 512개 레인(64행)의 엔티티 인코더 forward 약 10.5 GB(거리 bias one-hot 2 GiB 포함)와 GPU와 나눠 쓰는 RAM. 풀 정리·Burn gradient checkpointing은 효과 없음 (`m6_core_ablation.md` 5절)
- 2026-10-20 23시 — **결정·구현**(041-R17·R18, 사용자 결정): 엔티티 인코더 조각 재계산 `URANIUM_BC_ENTITY_CHUNK`, 호스트 batch zstd 압축 `URANIUM_BC_PACK_BATCH`(둘 다 기본 끔, 설정 지문 불변). Linux 호스트 A(16, 1)에서 step 48 통과, GTT 최대 11.3 → 4.2 GB, update 108 s. 측정 출력 Linux 호스트 A `~/diag/r17/`
- 2026-10-21 15:52~16:07 — 플레이 성향 확인용 대전(사용자 요청): `core-abl-lstm` ckpt-5300(확률적) vs T1 Iron(T)·Steamhammer(Z)·Stardust(P) 각 3판, 학습 PC 로컬 hostd. Steamhammer·Stardust 6판과 Iron 1판 전멸패(4~16분), Iron 2판은 60분 시간 제한 무승부. 세션 오류·시야 위반 0. 출력 학습 PC 저장소 `runs/lstm-vs-t1-2026-10-21/`(`replays/g1~g9.rep`, `results.jsonl`, `summary.md`)
- 2026-10-21 16:1x~17:2x — **발견**(032-R17 코어 비교 부수): LSTM ckpt-5300 argmax로 T1 9판 전패(4~7분). 디버그 결정 기록(`URANIUM_AGENT_TRACE`, 로컬 브랜치)과 리플레이 200판 비교에서 게임 시작 입력·첫 결정 확률은 실게임과 학습이 같음(입력 버그 배제). 확률적 샘플링의 행동 종류 비율은 사람과 비슷하고 커맨드 띄우기·짓기도 사람 빈도 수준이라, 이상한 플레이는 상황 판단 미학습(학습량 14% epoch)으로 봄. argmax는 랠리 지정 반복 고리(결정의 99.9%, 사람 반복 클릭 학습). 부수: 대상 맵 리플레이 약 18%가 첫 결정 미네랄 0 (`m6_core_ablation.md` 5절)
- 2026-10-21 — **결정**(AMB-031-6, 사용자 요청·결정): GG 항복(승률이 현저히 낮으면 채팅 후 퇴장, 패배). 판단은 상대 정보 없이 학습한 승률 추정 win_open(032-R18, 기존 win 헤드는 학습 때 privileged를 받아 실전 추정으로 못 씀) + 규칙 조건(내 병력 ≤ 본 적 최대 적 병력의 1/3 또는 배럭·팩토리·스타포트 없음/SCV ≤ 5), P ≤ 5%가 30초 연속·10분 이후. SC:R 실전만 On, OpenBW 학습·평가는 기본 Shadow(판정만 기록). 채팅은 plan 옵션 `surrender_chat`(기본 "gg"). 본학습 설정부터 win_open 학습, 기존 학습과 shard는 불변(031-R12, 010-R13, 011-R21, 012-R11, 020-R14, 041-R19, 042-R13)
- 2026-10-21 — **구현·확인**(031-R12 외): master 53a8a01~4fdaf23. proto v4(`Act.surrender`), shim 채팅 → 24프레임 뒤 퇴장(떠난 쪽 상대 승·Surrender), BWAPI 패치 0014(OpenBW `NetMode`가 0이라 LAN 채팅이 로컬 출력만 되던 것을 채팅 액션 0x5C로, frame 0 안내문 제외), 채팅 상한 79바이트(BWAPI 버퍼 80 + NUL). **발견**: Burn `Config` derive는 누락 필드를 거부해 기존 체크포인트(코어 비교)가 못 읽힐 뻔함 → `net_config`가 빠진 키를 기본값으로 채움. Burn 파라미터는 처음 쓸 때 장치 RNG로 초기화되므로 win_open은 고정 seed로 초기화(032-AC9 비트 일치). a09f36e 이후 `rl/trajectory.rs` 테스트 import 누락으로 uranium-train lib 테스트가 컴파일되지 않던 것 수정. 검증(Linux 호스트 C, 학습 중 아닌 장비): workspace 테스트 324 통과·0 실패(무시 11), 통합 3판 — 강제 항복(On, `URANIUM_TEST_SURRENDER_FRAME`) 2판 모두 Finished·상대 승·드롭 0·승자 불일치 0, 리플레이에 채팅 1개("gg"/"good game")와 퇴장, Shadow 1판은 시간 제한까지 진행. 출력 Linux 호스트 C `~/gg-out/`, 빌드 `~/wt-gg/`. 남은 것: 게임 호스트·학습 PC 브리지·BWAPI(0014)·proto v4 재빌드, 041-AC4 측정(win_open 있는 BC 체크포인트 이후), SC:R 경로(M7)
- 2026-10-21 18:26~19:10 — scr_ladder 증분 build(Linux 호스트 B, 2639038 바이너리 = CIFS 쓰기 검증 수정 이후, 브리지는 f513b92 사본과 바이트 동일, jobs 8): 10-21 수집분 1,069개(누적 raw 5,000), written 332·filtered 700(제외 맵 341, APM 기준 미달 199, 040-R19 1)·desync 37·error 0(records 4,261 → 5,330줄), 새 shard `scr_ladder-p0/shard_00083`~`00107`. manifest는 미갱신. 로그 `build/logs/bc250_2/scr_ladder_scr_ladder-p0.inc_20261021.log`
- 2026-10-22 02:22~02:44 — **발견·조치**(061-R12, 032-R17): 학습 PC가 `core-abl-gtrxl`을 Linux 호스트 A(step 1319, 양보)에게서 이어받은 뒤 verify 다음에서 멈춘 것처럼 보임. 원인: 10-21 scr_ladder 증분 build가 manifest 갱신 없이 공용 드라이브에 둔 `scr_ladder-p0/shard_00083`~`00107` 25개를, 로컬 모드(공유 제공 PC)는 manifest가 아니라 폴더 목록으로 읽어 2069개가 됨 → scan 캐시 불일치로 전체 재디코딩(진행 출력 없음), 끝나도 데이터 지문(`2044:…` ≠ 2069)으로 재개 거부될 상태. 원격 PC는 manifest 기준 pull이라 영향 없음. 조치(사용자 결정): 코어 비교가 끝날 때까지 25개를 `<shared>/staging/scr_ladder-inc-20261021/scr_ladder-p0/`로 옮기고 manifest 유지(네 출처 폴더 = manifest 확인), 비교 후 되돌리고 manifest 갱신. 02:42 학습 PC 작업자 재시작 → scan 캐시 적중·지문 일치, `ckpt-1320`부터 재개(첫 update 314초, 이후 약 17초). 로컬 모드가 manifest 밖 파일을 읽는 코드 문제는 비교 후 수정 제안
- 2026-10-22 — **M7 착수·결정**(AMB-012-2, 사용자 결정): SC:R 대상 빌드는 1.23.10.13515 하나로 고정(x86_64 exe SHA-256으로 판별, 다른 빌드는 012-R6 비활성), 기존 samase 환경 없음 → 스파이크에서 로더 확보·설치 절차 조사, 실전 PC는 CUDA 학습 호스트(Windows)와 같은 PC. AMB-001-1(추론 위치)은 지연 측정 뒤, AMB-012-1(분석 실패 시 대안)은 스파이크 결과 뒤 다시 묻는다. 학습 PC는 현재 유휴(사용자 확인)라 스파이크를 이 PC에서 진행
- 2026-10-22 — 새 리포트 `reports/m7_scr_spike.md`(M7 스파이크 배경·예상, samase 문헌 조사): samase 로더는 비공개 바이너리이고 최신 64비트 빌드는 제3자 저장소에만 있음. 플러그인 API(MIT)에 게임 종료·채팅·나가기 전용 API 없음. samase_scarf 테스트 데이터는 1.23.10g까지
- 2026-10-22 — **결정**(012, 사용자 확인): SC:R 클라이언트는 64비트가 아니라 32비트(x86)여도 된다. 같은 빌드 1.23.10.13515의 x86 exe 해시도 고정 목록에 추가했다. 스파이크는 x64부터 해 보고, 막히면 x86으로 바꾼다. 둘 중 무엇을 쓸지는 결과를 보고 정한다
- 2026-10-22 21:23~21:35 — **스파이크 결과**(012-AC1, AMB-012-1): 학습 PC에서 samase 0.10.28-64(API 44)로 SC:R 1.23.10.13515 x64를 실행했다. 탐침 플러그인이 012-AC1 목록(units·players·game·selections·send_command·step 훅·visibility)을 모두 찾았다. 싱글 커스텀 게임(투혼, vs 컴퓨터)에서 상태 읽기, 시야 비트(하위 8비트 = 안 보임), Select121·RightClick121 송신으로 SCV 이동, 프레임 훅 p99 0.020ms를 확인했다. 유닛 배열은 3400(확장 한도). `--map` 자동 시작은 Code 8000000d로 실패해 쓰지 않는다. AMB-012-1은 대안 불필요로 닫음. 로그는 학습 PC 스파이크 폴더 `logs/` (`reports/m7_scr_spike.md`)
- 2026-10-22 — 정정: 위 스파이크 줄의 "21:23~21:35"는 21:23~21:50이 맞다. 탐침 게임은 21:41에 띄웠다(samase 로그 기준). 로그 이름도 `logs/probe-x64-20261022-run1.log`로 바꿨다
- 2026-10-22 21:54~22:10 — 스파이크 2·3차(LAN, `reports/m7_scr_spike.md`): LAN 명령 지연 2프레임(싱글 1). `send_command` 0x5C 채팅은 게임이 처리하고 화면에 뜸(항복 채팅 가능). `0x57` 나가기는 명령으로 처리되지만 로컬 클라이언트가 게임에서 빠지지 않음(유닛·`victory_state` 변화 없음). samase API에 클라이언트 게임 상태 변수가 없어 나가는 방법은 사용자 결정 필요. 프레임 훅 p99 0.018~0.034ms
- 2026-10-22 — 새 리포트 `reports/m7_scr_field_mapping.md`(OpenBW 브리지 필드 → SC:R 원시 메모리 대응표, BWAPI 소스 근거)
- 2026-10-22 — **발견**(010-R5, 030 §3): OpenBW shim이 CompleteMapInformation을 켜서 BWAPI가 모든 탄환을 보이는 것으로 처리한다(`BulletImpl.cpp:45`). 그래서 shim 탄환 필터(`shim.cpp:522`)가 아무것도 거르지 않고, feature 추출(`extract.rs:302`)도 시야를 보지 않아 시야 밖 탄환(스톰·EMP·투사체)이 hazard 채널에 들어간다. BC shard와 코어 비교 학습에 포함됐다. 처리 방법은 사용자 결정 대기
- 2026-10-22 — **결정·구현**(AMB-030-4, 사용자 결정): 탄환 등 효과는 사람 화면처럼 중심 타일이 지금 보이는 것만 관측한다. 에이전트 feature 추출에서 `RawFrame.visibility`로 거르고(기존 shard 재build 없음, BC 본학습부터 적용, 끝난 코어 비교는 그대로), OpenBW shim 탄환 루프에도 같은 검사를 넣었다(게임 호스트·학습 PC 브리지 재빌드 때 반영, C++ 컴파일은 그때 확인). 테스트 `hazard_ignores_effects_with_unseen_centre_amb_030_4`, uranium-env 테스트 전체 통과(030 §3, 010-R5)
- 2026-10-22 — **결정**(AMB-012-4, 사용자 결정): SC:R 항복은 "gg" 채팅(`send_command` 0x5C) 뒤 브리지가 명령을 멈추고 `GameEnd(Surrender)`를 보낸 다음 연결을 닫는다. 게임에서 나가지는 않는다(게임 창은 사람이 닫음). 메뉴 키 입력·내부 함수 호출은 쓰지 않는다(012-R11)
- 2026-10-22 — **결정·구현**(AMB-030-5, 사용자 결정): 보이는 적 유닛의 내부 정보(생산 큐 길이, 생산·건설·연구 진행도, 핵 장전)는 BWAPI 공정 모드 `canAccessInside` 기준으로 숨긴다. 에이전트 feature 추출에서 다른 플레이어 유닛의 해당 값 3개를 0으로 둔다(재build 없음, BC 본학습부터, 코어 비교는 그대로). 체력·에너지·쿨다운·order·마인 수는 BWAPI가 보이는 적에게도 공개하므로 유지. 테스트 `enemy_inside_info_is_hidden_amb_030_5` (030 §2)
- 2026-10-22 — **결정**(사용자): SC:R 실전 지연은 보정하지 않고 기록만, 명령 유효 기한은 관측 + 24프레임(AMB-012-5). `uranium-adapter` crate는 만들지 않고 SC:R 표는 `bridge/scr/tables`(AMB-001-5). SC:R `CommandFailed`는 보내지 않고 RL 모델 SC:R 배포 전에 다시 정함(AMB-012-6 Deferred)
- 2026-10-22 — spec 012·013 M7 개정안(Review): 스파이크 실측(빌드·로더 고정, API 목록, 시야 비트, 지연 1/2프레임, 유닛 배열 3400, 채팅 0x5C), 위 결정, 관측 시점 규칙(012-R8), 게임 종료 판정(012-R15), 에이전트 realtime 처리(012-R18·R19), AC 재구성(AC2~AC7). 001 R1·workspace에서 `uranium-adapter` 삭제. 승인 요청
- 2026-10-22 — **승인**(사용자): spec 012·013 M7 개정안 Approved(012-R15 시간 제한 없음, 012-R16 접속 실패 시 비활성 포함). 다음은 `bridge/scr`·에이전트 realtime 구현 계획(Plan 모드)
- 2026-10-22 — **구현**(012-R7·R14·R18·R19, M7 1단계): 에이전트 실전 모드. `Hello.mode = Realtime` 세션은 Act에 `valid_until_frame` = 관측 프레임 + 24를 붙이고, 모델의 `last_failed`를 0으로 고정하며, PrivilegedSummary는 세기만 하고 정책에 넘기지 않는다. 새 명령 `uranium-agent scr`(기본 listen 127.0.0.1:7700, 항복 on, 결과 `runs/scr/sessions.jsonl`). 테스트 `realtime_session_012_r18`, uranium-agent·runner 테스트 통과, workspace 컴파일 확인. lockstep(OpenBW) 경로는 바뀌지 않아 게임 호스트 재빌드 불필요

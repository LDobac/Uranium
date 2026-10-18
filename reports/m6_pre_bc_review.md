# BC 본학습 전 요구사항·버그 리뷰 (2026-10-18)

## 배경
- 목적: 약 15일짜리 BC 본학습(041, S1 → S2, ablation (A))과 그 전의 코어 비교(032-R17)를 시작하기 전에, M0~M5 범위 spec이 제대로 구현됐는지와 구현에 버그가 없는지 점검한다.
- 대상: master e190799. 읽기 전용으로 리뷰했다(코드 변경 없음). 공용 드라이브는 records·표본 shard 읽기만 했다.
- 방법: Fable 5.1(high) 리뷰어 4명이 영역을 나눠 맡았다. (1) 030·031·033·034 ↔ `uranium-env`, (2) 040·013·043·011 ↔ `uranium-replay`·브리지, (3) 032·041·061·060 ↔ `uranium-model`·`uranium-train`, (4) 교차 점검(미이행 약속, spec 간 모순, config, `.env.example`, 001-R10, CLAUDE.md). 영향이 큰 주장은 코드에서 다시 확인했다(아래 "재확인" 표시).
- 실험 전 예상: 개별 기능은 테스트돼 있으니 큰 문제는 없고, 운영 쪽 빈틈 정도가 나올 것으로 봤다. 실제로는 이미 build된 데이터의 레이블·메타에 영향을 주는 문제가 3건 나왔다.

## 진행 기록
- 2026-10-18 저녁: 리뷰 워크트리 생성(e190799), 리뷰어 4명 실행, `cargo test --workspace` 실행.
- 같은 날: `cargo test --workspace`(debug) 실패 확인(11개). 원인은 아래 B-1.
- 같은 날: 네 리뷰 결과를 받아 핵심 주장(B-1, A-1, A-2, A-4, C-1)을 코드에서 재확인한 뒤 이 리포트를 썼다.
- 2026-10-19 master 반영 전 재확인(695cbfe 기준): 5661ff8에서 watchdog(B-4)이 구현됐고, 032-R17 ①이 장비 혼합 운영으로 개정됐다(C-1 둘째 항목 해소). B-1(`heads.rs:322`), B-7(`bc_run.rs` `r.prune()?`), A-4(`split.rs:63`)는 그대로 남아 있다. 나머지 항목은 이후 커밋이 해당 파일을 건드리지 않아 그대로다.

## 테스트
`cargo test --workspace --no-fail-fast`(debug, flex): **193 통과 / 11 실패**(3개 타깃). 실패 11개는 원인이 모두 같다(B-1).
- uranium-model `tests/model.rs`: 032-R3 teacher(LSTM·GTrXL), 032-AC5 privileged, 032-R13 backward, 032-R10 location_on_used_rows. 어느 테스트가 실패하는지는 실행마다 다르다(무작위 배치에 일부 행만 location을 쓰는 경우에만 걸림).
- uranium-train lib: RL learner 3개(vtrace, ppo, reset_to_bc).
- uranium-train `tests/bc.rs`: `bc_smoke_041`, `micro_lanes_match_full_batch_041_r3`, `resume_continues_bit_identically_061_r10`.
- 리뷰어 (3)에 따르면 `--release`에서는 통과한다(burn-flex의 dtype 검사가 `debug_assert`).

## A. 이미 build된 데이터에 영향 (사용자 결정 필요)

| # | 심각도 | 내용 | 위치 | 복구 |
|---|---|---|---|---|
| A-1 | High (재확인) | **'투혼' 약 2,900게임이 `map_family = other`**: 1.16.1 리플레이 헤더 맵 제목(CP949)을 `from_utf8_lossy`로 읽어 "투혼"이 U+FFFD로 깨진다. `config/map_families.toml`의 `chk_sha256 = []`가 비어 있어 해시 판정도 없다. 그 결과 이 게임들이 S2·(A)·대상 맵 test에서 빠지고, 그 게임들의 `map_family` 스칼라가 other로 학습된다. 추정 규모: fighting_spirit 8,204 → 약 11,100, S2 train 7,201 → 약 9,500, test 895 → 약 1,180 | `crates/uranium-replay/src/header.rs:56`, `pipeline.rs:313`, `uranium-env/src/maps.rs:35` | 로더에서 shard `game_info.map_name`(정상 UTF-8)으로 재분류하면 재build 없이 가능. 어느 방법이든 `[steps]` s2·warmup 재계산(041-R14). 서킷브레이커 CP949 제목도 같이 확인 필요 |
| A-2 | High (재확인) | **Train 연타가 스팸으로 병합됨**: 2프레임 안의 같은 명령을 합치는데, Train은 선택 인자가 없어 다음 턴 연타(M 연타, 다중 건물 순환)가 하나가 된다. 게임에서는 전부 큐에 들어간다. 표본(STARDATA 1/150, 테란 시점 220개)에서 Train 레이블 9.42% 손실(SCV 10.3%, 마린 9.7%) | `crates/uranium-replay/src/label.rs:21,247`, 040-R4 | 재build로만 복구. 040-R4의 "같은 명령 반복"에 Train이 들어가는지가 spec 빈틈 |
| A-3 | High | **value 정답 `result`가 표본 shard 67게임 중 70%에서 0**: 패자가 LEAVE로 나가면 OpenBW `isDefeated`가 서지 않는다. value head가 0으로 회귀 학습된다(정책 헤드는 무관) | `pipeline.rs:192-196`, `shim.cpp:329`, `uranium-train/src/bc.rs:526-530`, 041-R7 | 브리지 승자가 없으면 "첫 이탈자의 상대 = 승자"로 정하는 규칙을 로더에 넣으면 재build 없이 가능하다. 또는 result 0이면 value 가중치 0. 전체 비율 집계 필요 |
| A-4 | Medium (재확인) | **S2 선택이 040 품질 기준(테란 APM ≥ 150, 프로·상위 래더)을 무시**: `selects(S2)`는 대상 맵 여부와 train 분할만 본다. 040 품질 표·043-R8과 041-R13("대상 맵 train 분할만, 소스 무관")이 모순이다. 대상 맵 written 8,983 중 stage S1 743게임이 S2에 들어 있다 | `crates/uranium-train/src/split.rs:62` | spec 결정만 필요(재build 불필요) |

정리하면 A-2만 재build가 필요하다. A-1·A-3은 로더 패치로도 고칠 수 있다. 재build(uranium-replay만 재빌드, 브리지 변경 없음, 약 24시간·4대)를 하면 셋을 한 번에 반영할 수 있다.

## B. 학습 코드 (본학습 전 수정 권고)

| # | 심각도 | 내용 | 위치 |
|---|---|---|---|
| B-1 | High (재확인) | **debug 테스트 회귀**: `location_used_rows`의 `loc_ok.select(0, idx)`가 burn-flex에서 Bool 대신 U8 dtype이 되어 `bool_not`이 panic한다. 7d395e5(BC 18% 가속)에서 들어왔다. 그래서 지금은 041-R3 micro_lanes 동등성과 061-R10 재개 동일성을 문서의 명령(`cargo test --workspace`)으로 검증할 수 없다. CUDA는 preflight에서 동작했지만 **Vulkan에서 이 경로를 실행한 기록이 없다**. 수정안: `loc_ok.int().select(0, idx).equal_elem(1)` | `crates/uranium-model/src/heads.rs:322` |
| B-2 | High | **맵 사전계산 파일이 없으면 경고 없이 base_loc·choke·wall 채널이 0**: 사전계산 1,251개는 이 PC에만 수집돼 있다. BC250 PC에서 (A)를 돌리거나 S1을 이어 받으면 그 구간만 다른 입력 분포로 학습된다. `URANIUM_MAPS_DIR` 기본값은 상대 경로다. 리뷰어 3명 중 3명이 지적했다 | `uranium-env/src/maps.rs:64`, `uranium-train/src/data.rs:263,524` |
| B-3 | High | **재개 시 단계·설정을 검증하지 않음**: 체크포인트 `extra`에 stage가 없고, 데이터 지문은 파일 목록만 해시한다. `--stage s2`나 `--config`를 잘못 주고 재개하면 경고 없이 다른 데이터·cosine 길이로 이어 간다 | `uranium-train/src/bc.rs:290-299,558`, `data.rs:92`, `bc_run.rs:96-139` |
| B-4 | ~~High~~ 해결 | ~~060-R6 watchdog 미구현~~ → 리뷰 직후 5661ff8(21:00)에서 `uranium-train supervise`(이상 종료·멈춤 시 재시작)와 run 큐 worker로 구현됐다. 060-AC3(24시간 스모크) 통과 기록은 아직 없다 | 5661ff8 |
| B-5 | High(측정 필요) | **prefetch 메모리**: 최대 4배치(배치당 약 4GB, spatial f32) + shard 1GB. BC250(공유 16GB)에서 스왑·OOM이 날 수 있다. BC250의 "169 s/update" 실측이 스왑 상태였을 수 있다 | `uranium-train/src/bc.rs:343-355` |
| B-6 | Medium (재확인) | **증강 flip에서 짝수 폭·높이 건물의 `entity_cell`이 1타일 어긋남**: 픽셀은 `W*32-px`, 셀은 `W-1-x`로 변환하는데, 짝수 폭 건물 중심이 32의 배수라 실제 거울 게임의 셀과 1 차이가 난다(예: W=96, 배럭 x0=20 → 73, 실제 74). flip 샘플마다 건물 scatter가 `entity_pos`와 어긋난다. 수정은 한 줄(변환한 픽셀로 `cell_of` 재계산) | `uranium-env/src/features/augment.rs:128-130` |
| B-7 | Medium | `prune()` 실패(SMB 삭제 오류)가 run 전체를 종료시킨다. push 오류는 흡수하지만 prune은 아니다. `supervise`가 재시작하므로 영향은 줄었다 | `bc_run.rs:328`(695cbfe), `storage.rs:587` |
| B-8 | Medium(가능성) | 시간 체크포인트 직후 Ctrl+C가 오면 같은 step을 두 번 저장한다. 두 번째 내용이 다르면 manifest 검사로 push가 영구히 막힌다 | `bc_run.rs:198-211`, `storage.rs:552` |
| B-9 | Medium | grad clip 1.0이 Burn에서는 파라미터 텐서별 L2 클립이다(전역 norm이 아님). spec 041-R3에서 의미를 확정해야 한다(코어 비교와 본학습이 같은 방식이면 됨) | `bc.rs:411` |
| B-10 | Medium | 데이터셋이 바뀌어도 `[steps]`와 실제 1 epoch 길이를 대조하지 않는다. 본학습 직전에 `bc-epoch`를 다시 돌려 대조해야 한다(A-1을 고치면 S2 값이 바뀜) | `bc_run.rs:131,172` |
| B-11 | Medium | 034 "무력화 유닛은 명령 불가" 규칙이 이동·공격·캐스터 클래스에만 있다. 락다운된 SCV·탱크·드랍십 등은 선택 자격이 남고, `pick_worker`도 락다운 SCV를 고른다 | `uranium-env/src/features/masks.rs:109-161`, `executor.rs:163` |
| B-12 | Medium(미확인) | shard `GameInfo`에서 `is_enemy` 정확히 1명인지 검사하지 않는다. 표본 67게임은 정상이다. 전수 검사는 `shard-stats`에 추가 권고 | `uranium-replay/src/main.rs:475` |
| B-13 | Low | 재개 시 레인마다 shard 파일을 다시 디코드한다(32회, 1~2분, 일시 1GB) | `data.rs:144` |

## C. 미이행 약속·미검증 AC

| # | 내용 | 비고 |
|---|---|---|
| C-1 | 032-AC7 코어 비교 미실행. 032 §2.4가 BC 본학습을 비교 뒤로 묶어 둠 | 알려짐(승인 대기). 비교 run 전 정리 사항: 061-R6 최신 5개 보존 ↔ 032-R17 ⑤ "지우지 않는다", 장비 혼합 운영이면 032-R17 ① 개정 선행 |
| C-2 | 060-AC3 24시간 스모크 | 060-R6은 5661ff8에서 구현(B-4) |
| C-3 | 061-R8·060-R7 여유 공간 검사·90% 경고(MUST) 미구현 | 위험 낮음 |
| C-4 | 040-R6·AC4 `reports/dataset_stats.md` 없음, 040-AC2 전체 build 폐기율 표 없음 | records 집계: written 35,966 / desync 2,326 / filtered 40,665 / unplayable 320 / error 4, 커버리지 99.78% |
| C-5 | 031-AC1(모든 action_type OpenBW 실행), 비무장 유닛 move 대체 실게임 확인 | 041-AC2·코어 비교 T0 대전 해석 전에 필요 |
| C-6 | 061-AC1(RNG 비트 일치: dropout RNG는 저장 불가), 061-AC5(1e-4가 CUDA 1e-2와 모순), 061-AC4 | spec 문구 조정 |
| C-7 | 013-R7 왕복 검증이 build 경로에서 호출되지 않음(CLI만) | AC4는 통과 |
| C-8 | 010-AC4 executor Command 왕복, 060-AC4 CUDA 벤치 리포트, 032-AC2 BC250 기준 재확인, 043-AC4 | 정리 |
| C-9 | 020-R3 "에이전트 무응답 30초 → 중단" 미구현, 011-R18(드롭 10분)과 모순 | RL 영향 |

## D. spec 간 모순·문서 정리 (낮음)
- 061 구조 절 `datasets/rep/`·`datasets/shards/`와 실제 `datasets/replays/{raw,shards}/`가 다르다. 060-R7 "RL 리플레이 자동 삭제"와 042-R11 "지우지 않는다"가 모순이다. 060-R1 `--device`는 실제로는 `--backend`다. 030 F_c≈76과 032 ≈64가 다르다. 031 `tools/mapgen`은 실제로 `uranium-replay mapgen`이다. 021 McRave 종족 표기와 rl.toml이 다르다. 000 맵 해시 `TBD(M1)`가 남아 있다. 061-R4 push 30분과 bc.toml 60분이 다르다.
- CLAUDE.md: 고정 조건표 정밀도(BC 전부 f32 예외 미반영), 존재하지 않는 `uranium-adapter`, 진행 상황이 10-17에서 멈춤.
- `.env.example` 누락 키: `URANIUM_CMAKE_GENERATOR`, `URANIUM_BC_PROFILE`, `URANIUM_REPLAY_DEBUG`, `URANIUM_TEST_DEVICE`.
- 001-R10: 장비 라벨(BC250 PC 이름)이 032-R8과 BC 리포트 두 개에 있다. 호스트 이름으로 볼지는 사용자 판단.
- 확인 결과 정상: README 인덱스 Status와 본문 일치(20개), AMB 레지스트리와 본문 양방향 누락 없음, `bc.toml`과 `bc_core_ablation.toml` 동기화, `reward.toml`·`rl.toml` 값이 spec과 일치, crate 의존 방향, CLAUDE.md 명령·플래그와 clap 정의 일치, 코드의 요구사항 ID 207개 모두 존재.

## E. Low (RL·후속)
- hazard 반지름 16px 고정(030 §3 규칙 없음). BC 시퀀스가 첫 명령에서 시작한다. 안 보이는 자원은 엔티티가 아니다(우클릭 채취 레이블 일부 무효). SetRally 다중 건물 중 첫 건물만 실행된다. `last_failed` 스칼라는 BC에서 항상 0이다. CMI 상대 종족이 시작부터 보인다(M7 Random). `DELAY_BUCKETS` 상수가 중복돼 있다. 마지막 결정 이후의 dense 보상이 버려진다. bc-eval 헤드별 NLL 분모와 delay MAE 단위. `privileged` None 비율 미확인. 041-R9 스트리밍 로더 미구현(예산 안이라 지금은 불필요). 040-R8 ④(SC:R 기록 승자) 미구현. `uranium-replay`가 `uranium-scripted` 유닛 상수에 의존한다.

## 권장 순서
1. 사용자 결정: A-1~A-4(재build 여부 포함), B-9 클립 의미, C-1 보존 범위, 061-AC5 허용 오차.
2. 코드 수정(spec 개정 → 확인 → 코드): B-1, B-2, B-3, B-6, B-7, B-8. 결정에 따라 A-1·A-2·A-3.
3. B-5 BC250 메모리 실측, Vulkan에서 최신 커밋으로 1~2 update 스모크, `bc-epoch` 재대조(B-10).
4. 그 뒤 코어 비교(032-R17) → BC 본학습.

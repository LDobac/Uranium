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

## F. 다른 세션 발견·결정 통합 (2026-10-19, rl-remaining-tasks-decisions 세션)
A~E에 없는 내용과, 리뷰 이후 이미 반영된 결정을 모은다. 근거 커밋과 리포트를 함께 적는다.

### F-1. 이미 반영된 BC 관련 결정 (사용자 확정)
| 결정 | 날짜 | 근거 |
|---|---|---|
| 분할: `replay_sha256` 게임 단위, 대상 맵 80/10/10·그 외 95/5, S1 네 소스 균등·S2 대상 맵 (AMB-041-3·4) | 10-18 | eb9b311, 041-R12·R13 |
| 학습 길이 각 1 epoch, S2·(A) warmup = 1 epoch의 5%(396) | 10-18 | 3181bc1, c2bba6e, 041-R14 |
| BC250의 모든 BC 학습은 f32 (f16 미구현, 032-R8 예외, 060-AC2 닫힘) | 10-18 | 2d8dc4e |
| 증강 4종(원본·좌우·상하·둘 다, transpose 제외)과 `start_loc` 변환 (030-R4·AC6). 맵 대칭 외 증강(드롭·노이즈 등)은 넣지 않음 | 10-18 | 2d8dc4e. 남은 버그는 B-6 |
| 클래스 균형 가중치 미적용 (041-R6). 사유: 사람 같은 행동 분포를 중시. BC 평가가 너무 낮으면 재검토 | 10-18 | 2d8dc4e |
| watchdog(060-R6)과 run 큐 worker·양보 요청·장비 기록(061-R13~R15). 운영: BC250 1호기는 상시, 학습 PC는 쉬는 동안만, 먼저 빈 장비가 남은 run을 이어받음 | 10-18 | 5661ff8 |
| 장비 역할: 게임 실행은 BC250 두 대 모두 가능, learner는 학습 PC와 BC250 1호기만(2호기는 장기 고부하 불가) | 10-18 | 사용자 지시 |
| 코어 비교: S1 5,300 update씩, val 500게임, 맞대결 맵당 50판 + T0 조합당 50판(600판), 리플레이 전부 저장, 판정은 결과를 보고 사용자가 정함 | 10-18 | d1efaf4, c787baa, 695cbfe, 032-R17 |
| 요약표: 두 진영 정책·코어·run 표시, 모델끼리 대전 승리를 승자 정책으로 집계(기존 버그 수정), 슬롯 교대 합산 승률 | 10-18 | 695cbfe |
| 정규화 상수 spec 기록(동작 불변) | 10-19 | 030 §6 |

### F-2. A~E에 없는 발견·제안
| # | 내용 | 영향 | 근거 |
|---|---|---|---|
| F-2-1 | **Burn autotune 제안**: CUDA 약 2.5배(fusion 함께 약 3.5배, S1 약 11일 → 약 5일 또는 그 이하). 일부 conv gradient가 CPU 기준 대비 5.3e-2(fusion 함께 2.2e-2) 달라짐(끈 상태 1.3e-3). Vulkan은 약 1.35배, 차이 4.3e-4 | 결정 필요. 코어 비교와 본학습이 같은 설정이어야 함 | `m6_bc_preflight.md` 4차 |
| F-2-2 | **GTrXL CUDA OOM 1회**: 다른 GPU 작업과 겹쳤을 때 데이터 업로드 중 발생, 재시도 정상, 관측 최대 약 7.5GB(단독 5.4GB). 학습 PC를 쉬는 동안만 쓰는 운영이면 재발 가능, watchdog 재시작에 의존 | 운영 위험 | `m6_core_ablation.md` 진행 기록 |
| F-2-3 | **shard 쓰기 후 재읽기 검증(040)**: 10-18 사용자 보류("다음 build 전에 다시 질문"). A-2로 재build를 하면 그 전에 정해야 함. 10-17 p16 shard 하나가 BC250 1호기의 SMB 쓰기에서 앞 8MiB가 0으로 깨진 전례가 있음(로컬에 먼저 쓰고 복사 후 검증 권장) | 재build 시 결정 필요 | `m4_desync_nondet.md`, p16 복구 기록 |
| F-2-4 | Windows Ctrl+C: 시험 환경에서 콘솔 Ctrl+C 이벤트가 무시돼 같은 처리 경로의 BREAK 이벤트로만 확인. 실제 콘솔 Ctrl+C는 미확인 | 운영 확인 | `m6_bc_preflight.md` 4차 |
| F-2-5 | **빌드 상태**: 학습 PC 메인 체크아웃 바이너리는 7954346 기준(worker·supervise·BC 최적화 없음), BC250 1호기 저장소도 7954346. 본학습·코어 비교 전 master로 재빌드 필요. 추출 전용 Windows PC는 재build를 하면 원격 데스크톱에서 재빌드 필요 | 실행 전 필수 | 이 절 작성 시 확인 |
| F-2-6 | debug 테스트 실패는 B-1과 같음(맞대결 스모크 세션은 12개로 셈, 실행마다 다름) | B-1 | 695cbfe 보고 |
| F-2-7 | RL 본 실행 전 항목(BC와 무관): z 테이블 생성(033-R8), 추론 `max_batch`·동시 게임 수 측정(020 TBD), BC250 1호기 calibrate, 승격 게이트(042-R6) 미구현(C2 전), 050 평가 파이프라인, 승자 불일치 리플레이 대조(권장) | RL 전 | spec 전수 점검(10-18) |

## G. BC 본학습 전 통합 체크리스트 (2026-10-19 기준)
상태: 미결 = 사용자 결정 대기, 대기 = 결정 뒤 작업, 완료 = 반영됨. 코어 비교(032-R17)도 같은 데이터·코드로 돌려야 하므로 이 표가 끝난 뒤에 시작한다.

### G-1. 사용자 결정
| # | 항목 | 출처 | 상태 |
|---|---|---|---|
| G-D1 | 데이터 수정 방식: 재build(A-1·A-2·A-3 한 번에, 약 24시간·4대) 또는 로더 수정(A-1·A-3만, A-2 손실 감수) | A-1~A-3 | 미결 |
| G-D2 | 재build 시 shard 쓰기 후 검증·로컬 우선 쓰기 적용 여부 | F-2-3 | 미결 |
| G-D3 | S2 품질 기준(테란 APM ≥ 150 등) 적용 여부 — 040과 041-R13 모순 | A-4 | 미결 |
| G-D4 | grad clip 의미(텐서별 유지 / 전역 norm) | B-9 | 미결 |
| G-D5 | Burn autotune(·fusion) 채택 — CUDA·Vulkan 각각 | F-2-1 | 미결 |
| G-D6 | 코어 비교 체크포인트 보존 범위(061-R6 최신 5개 ↔ 032-R17 "지우지 않음") | C-1 | 미결 |
| G-D7 | 061-AC5 허용 오차(1e-4 ↔ CUDA 1e-2) | C-6 | 미결 |
| G-D8 | 코어 비교 사전 확인 남은 것: val 500게임 출처 편중, seed 1개 | `m6_core_ablation.md` 3절 | 미결 |
| G-D9 | 장비 라벨(BC250 PC 이름)이 문서에 있는 것을 001-R10 위반으로 볼지 | D | 미결 |

### G-2. 코드 수정 (spec 개정 → 확인 → 코드)
| # | 항목 | 출처 | 상태 |
|---|---|---|---|
| G-F1 | debug 테스트 회귀 수정(`heads.rs:322`) 후 `cargo test --workspace`(debug) 전부 통과, Vulkan에서 같은 경로 1~2 update 확인 | B-1, F-2-6 | 대기 |
| G-F2 | 맵 사전계산: BC250에도 배포하고, 없으면 경고 없이 0이 되지 않게(시작 시 검사·실패) | B-2 | 대기 |
| G-F3 | 재개 시 단계·설정 검증(체크포인트에 stage·config 지문) | B-3 | 대기 |
| G-F4 | 증강 flip의 짝수 폭 건물 `entity_cell` 1타일 어긋남 | B-6 | 대기 |
| G-F5 | `prune()` 실패가 run을 끝내지 않게 | B-7 | 대기 |
| G-F6 | Ctrl+C 직후 같은 step 이중 저장 | B-8 | 대기 |
| G-F7 | 무력화 유닛 명령 불가 규칙을 모든 클래스·`pick_worker`에 적용(034) | B-11 | 대기 |
| G-F8 | shard `GameInfo` `is_enemy` 정확히 1명 검사(`shard-stats`) | B-12 | 대기 |
| G-F9 | G-D1·G-D2·G-D3·G-D4 결정에 따른 수정(맵 분류, Train 병합, result 0, S2 선택, grad clip, build 검증) | A-1~A-4, B-9, F-2-3 | 대기 |

### G-3. 측정·실행 준비
| # | 항목 | 출처 | 상태 |
|---|---|---|---|
| G-M1 | 학습 PC 메인 체크아웃과 BC250 1호기를 master로 재빌드(재build를 하면 추출 전용 Windows PC도) | F-2-5 | 대기 |
| G-M2 | BC250 prefetch 메모리 실측(스왑 여부) | B-5 | 대기 |
| G-M3 | `bc-epoch`로 `[steps]` 재대조(A-1을 고치면 S2 값이 바뀜), `bc_core_ablation.toml` 동기화 | B-10 | 대기 |
| G-M4 | 060-AC3 24시간 스모크 — 코어 비교 run의 첫 24시간으로 대신할지 포함 | C-2 | 대기 |
| G-M5 | Windows 실제 콘솔 Ctrl+C 확인 | F-2-4 | 대기 |

### G-4. 문서 (동작 불변)
| # | 항목 | 상태 |
|---|---|---|
| G-W1 | 정규화 상수 spec 기록 | 완료(030 §6, 10-19) |
| G-W2 | CLAUDE.md 정밀도(BC250 BC f32 예외), `uranium-adapter` 미생성 표시, 진행 상황 갱신 | 완료(10-19) |
| G-W3 | D절 나머지(061 경로 표기, 060-R1 `--device`, 030·032 F_c, 031 mapgen, 021 McRave, 000 맵 해시, 061-R4 주기, `.env.example` 누락 키) | 대기 |

### G-5. 진행 순서
1. G-1 결정.
2. G-2 수정(G-F1 먼저), G-W3 문서.
3. G-M1 재빌드 → (재build를 하면) 표본 build → 학습 end-to-end 확인 → 전체 재build → manifest·분할 재집계.
4. G-M2·G-M3·G-M5 측정.
5. 코어 비교(032-R17, worker 운영) → 코어 선택 → BC 본학습(S1 → S2, (A)).

H 절(5661ff8·695cbfe 추가 리뷰)의 본학습 전 필수 항목 H-1~H-4도 이 체크리스트에 포함한다(G-2 수정과 함께 진행).

## H. 추가 리뷰: 5661ff8(supervise·worker·양보)·695cbfe(summary.md 집계) (2026-10-19)

### 배경·방법
- 첫 리뷰(e190799)를 마친 뒤 master에 들어온 두 커밋을 같은 방식으로 점검했다(Fable 5.1(high) 리뷰어 1명, 읽기 전용, 2af92d2 기준). 핵심 주장 H-1·H-2는 코드에서 다시 확인했다(아래 "재확인").
- 테스트(2af92d2, debug, flex): `cargo test --workspace --no-fail-fast` **196 통과 / 11 실패**. 실패 11개는 모두 B-1(`bool_not ... got U8`)과 같은 원인이고, 두 커밋이 만든 새 실패는 없다(`tests/worker.rs`·`tests/supervise.rs` 통과). 처음 돌린 실행은 리뷰어 빌드와 같은 target 디렉터리를 써서 `cubek_matmul` rlib 오류로 깨졌다. 별도 target 디렉터리에서 다시 돌린 결과가 위 숫자다.

### 본학습·코어 비교 전 필수
| # | 심각도 | 내용 | 위치 |
|---|---|---|---|
| H-1 | High (재확인) | **worker가 `init_run`의 DONE을 확인하지 않는다.** 큐가 `[bc-s1, bc-s2(init_run=bc-s1)]`이고 S1을 다른 PC가 잡고 있으면 S2가 Free로 보여 바로 시작된다. 그러면 S1 중간 `LATEST` 가중치로 S2를 끝까지 학습하고 DONE까지 기록한다. 한 PC에서도 S1이 GiveUp(1시간 skip)되면 같은 일이 생긴다. S1에 `LATEST`가 없으면 S2 잠금이 해제되지 않고 남는다. 코어 비교 큐(S1 두 개)는 해당 없음 | `crates/uranium-train/src/worker.rs:101-117`, `bc_run.rs` `resolve_checkpoint` |
| H-2 | High (재확인) | **공유 I/O 오류 처리.** `peek_lock`은 읽기·파싱 실패를 `Free`로 해석한다. worker 루프는 `Queue::load`·`pick`(probe 쓰기)·`request_yield` 오류에서 바로 종료한다. VPN이 끊기면 `setsid nohup`으로 띄운 worker는 그대로 죽는다. 끊김이 계속되는 동안에는 매 poll마다 자식을 띄웠다 실패하는 일을 반복하다 GiveUp한다 | `storage.rs:683-688`, `worker.rs:159-165` |
| H-3 | Medium→High(가능성) | **잠금 만료 자동 인수가 "공유 서버가 mtime을 부여한다"는 가정에 의존한다.** Linux cifs가 클라이언트 시계로 mtime을 쓰면 문제가 된다. BC250 1호기 시계가 10분 넘게 느려지면 이 PC worker가 살아 있는 run을 계속 빼앗고, 그때마다 최대 60분 작업이 손실된다. 4차 리포트에 "몇 분 빠름" 기록이 있다. 본학습 전에 BC250이 쓴 잠금의 `share_age`를 한 번 측정할 것. 자동 인수를 "같은 heartbeat를 단조 시계로 stale_after 동안 연속 관측"으로 바꾸면 시계와 무관해진다 | `storage.rs:674-688`, `worker.rs:112,181` |
| H-4 | Medium | **Linux 포그라운드 터미널에서 Ctrl+C를 누르면 자식이 SIGINT를 두 번 받는다**(프로세스 그룹 + supervise 전달). 그러면 "두 번째 Ctrl+C"로 저장 없이 130으로 종료된다. 문서에 있는 systemd·`setsid nohup` 경로에서는 생기지 않고, Windows도 안전하다. 운영 경로를 고정하거나 자식을 별도 프로세스 그룹으로 띄울 것 | `supervise.rs:89-97`, `bc_run.rs:206-212` |

### Medium·Low
- [Medium] 요청한 PC가 멈추거나 다른 run을 잡아도 그 PC의 양보 요청(`.yield`)이 30분간 남아, 그동안 BC250이 논다. 종료하거나 다른 run을 잡을 때 자기 요청을 지우게 하는 것을 권장(061-R14 보완).
- [Medium(가능성)] 양보 중 최종 push가 Deferred면 잠금을 쥔 채 76으로 끝난다. 그러면 요청 PC는 10분 stale까지 기다리고, 캐시에 남은 옛 계통 미push 체크포인트를 나중에 `recoverable`이 골라 계통이 섞일 수 있다.
- [Medium] `cross_device.rs`의 gradient 허용치 1e-3은 어느 spec에도 없다. 처음부터 1e-4를 통과한 적이 없으니 문제를 숨긴 것은 아니고, 처음 정한 기준이다. 032에 backward AC를 추가하거나 테스트명에서 `032_ac1`을 뗄 것(사용자 확인).
- [Medium] 첫 데이터셋 pull(약 40GB)·manifest 해시 구간에는 진행 출력이 없다. 지금은 30분 멈춤 기준 안에 들지만 여유가 문서화돼 있지 않다. `pull_tree`가 파일마다 출력하면 해결된다. kill해도 체크포인트는 깨지지 않는다(스테이징·rename 확인).
- [Low] stale인데 `take_over=false`인 경쟁은 exit 1 → 재시작 반복 → GiveUp(드묾). worker 상태 파일이 spec의 "1분마다"가 아니라 루프 시점에만 갱신된다. 수동 `bc --run`은 우선순위 0으로 취급된다. 061-AC8 자동 테스트가 run 1개만 쓴다.
- [Low] 문서 불일치: 서비스 `TimeoutStopSec=900` vs CLAUDE.md 600, 061-R13 `init` vs 코드 `init_run`, 060-R6 "Windows 작업 스케줄러"(구성 없음), 061-R15 "매 줄 백엔드" vs start 줄만, user 매니저에서 `After=remote-fs.target` 무시, 실행 중 바이너리 교체 시 다음 자식부터 새 바이너리.
- [Low] 695cbfe: `model_info` 실패 시 모델 대 모델 게임이 경고 없이 "mirror"로 들어가 정책별 표가 빈다. 경고 출력 권장.

### 확인 결과 정상
- 동시 획득은 `create_new`로 한쪽만 성공하고 진 쪽은 75로 끝난다. 양보 흐름(요청 → 체크포인트·release·76 → 요청한 host만 획득)이 맞다.
- DONE은 데이터 소진이나 `stop_at` 도달일 때만 기록된다(Ctrl+C·LockLost·Yielded 제외). DONE 기록 전 크래시는 다음 획득자가 복구한다. 같은 step 이중 저장(B-8)은 `saved_at`으로 해결됐다.
- 종료 코드 75·76을 다른 오류가 우연히 내는 경로는 없다. devices.jsonl은 step 구간 기준으로 일관된다.
- 695cbfe 집계: 같은 게임을 두 번 세지 않는다. 무승부는 n에 포함되고, Aborted는 승패에서 제외되며, 승자 판정은 020-R13을 따른다. Wilson 공식이 맞다. 001-R10 위반이 새로 들어간 곳은 없다.

### spec 개정 제안 (사용자 확인 필요)
1. 061-R13: `init_run`이 DONE이 아닌 run은 잡지 않는다. 공유 I/O 오류 시 종료하지 않고 재시도한다. 자동 인수 조건을 명시한다.
2. 061-R14: 요청 host가 종료하거나 다른 run을 잡으면 자기 요청을 지운다.
3. 060-R6·061-AC9: Linux 감시 프로세스를 터미널 포그라운드로 실행하는 것을 금지한다(또는 코드로 해결).
4. 032: backward 허용치 AC(Vulkan 1e-3, CUDA 1e-2) 추가 또는 테스트명 수정.
5. 사소: R13 `init`→`init_run`, R15 줄 형식, 060-R6 Windows 자동 시작 문구, `TimeoutStopSec` 통일.

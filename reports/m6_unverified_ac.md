# M6 미검증 AC 정리 (T7, C-5·C-7·C-8) (2026-10-19)

## 배경
- `reports/m6_pre_bc_review.md` C절에서 확인되지 않은 AC가 남았다.
  - C-5: 031-AC1(모든 action_type이 OpenBW에서 실행되는지)과 비무장 유닛 Attack_Pos → move 대체(AMB-041-1)
  - C-7: 013-R7 왕복 검증이 build 경로에서 불리는지
  - C-8: 010-AC4(executor Command 왕복), 043-AC4(래더 수집 속도 제한), 060-AC4(CUDA 벤치 리포트), 032-AC2(추론 지연 기준 재확인)
- 이 작업은 전체 재build와 병렬로 진행했다. 그래서 GPU 벤치와 긴 OpenBW 배치는 돌리지 않고, 재build 뒤 명령만 아래 체크리스트로 남긴다(060-AC4, 032-AC2).

## 예상
- 010-AC4: 034의 100개 action_type 모두 합성 상태에서 Command가 나오고, postcard 왕복 뒤에도 같다.
- 031-AC1: 실제 게임에서 대부분 받아들여진다. 마스크가 게임보다 느슨한 경우가 일부 드러날 수 있다.
- 043-AC4: 수집기는 `Limiter`로 0.5 req/s를 지키지만, 지금 로그에는 요청 단위 시각이 없어 직접 증명은 어렵다.

## AC별 상태
| AC | 상태 | 근거 |
|---|---|---|
| 010-AC4 | **통과(테스트 추가)** | `crates/uranium-env/tests/features.rs` `every_action_type_command_round_trips_010_ac4` |
| 031-AC1 | **통과(실게임 1회)** | `crates/uranium-env/tests/openbw_actions.rs` `every_action_type_runs_in_openbw_031_ac1`(`#[ignore]`, hostd 필요) |
| AMB-041-1 비무장 Attack_Pos | **통과** | 같은 테스트. 드랍십·사이언스 베슬 모두 다음 관측에서 order = Move(6) |
| 043-AC4 | **통과(테스트 추가) + 기존 로그 위반 흔적 없음** | `r043_ac4_collect_requests_stay_under_the_rate`, `r043_ac4_rate_stats_from_send_times`. 실행마다 속도 요약 한 줄을 남긴다 |
| 013-R7 / 013-AC4 | **build 경로 호출 확인** | T3a(C-7) 이후 `uranium-replay build`가 리플레이마다 `roundtrip_check`를 부른다(`crates/uranium-replay/src/pipeline.rs`, 실패 시 `013-R7 round trip` 오류 또는 review로 기록). AC4 자체는 이전에 통과 |
| 060-AC4 | **대기(재build 뒤)** | 아래 체크리스트 1 |
| 032-AC2 | **대기(재build 뒤) + 사용자 확인 필요** | 아래 체크리스트 2 |

## 010-AC4
- 합성 상태 두 개(테크 있음·없음)에서 action_type마다 마스크를 만족하는 인자를 고른다. 선택은 1기, 타깃은 선택하지 않은 유닛이다.
- 그다음 executor로 Command를 만든다.
- 확인 항목
  - Command 종류가 034 표의 `command`와 같다.
  - Command는 정확히 1개다(031-R3). NoOp은 Command가 없다(delay만 쓴다).
  - `AgentMsg::Act`로 `write_msg`/`read_msg`(postcard) 왕복한 뒤에도 같다.
- 100개 모두 통과했다.
- 인자 선택 함수(`tests/common/mod.rs` `valid_action`)는 031-AC1 시나리오와 같이 쓴다.

## 031-AC1 시나리오
### 설계
- hostd에 `StartGame`을 보내 투혼(seed 31)에서 테란 대 테란 게임을 연다. 두 브리지는 테스트 안의 에이전트에 붙는다.
- p0(스크립트)
  - 한 기지로 테크를 다 올린다.
  - 매 스텝(8프레임) 아직 받아들여지지 않은 action_type 중 type_mask가 열린 것 하나를 실제 마스크·executor로 실행한다.
  - 경제 행동(일꾼, 보급, 가스, 필요 건물 재건설)도 같은 executor 경로로 보낸다.
- p1: 아무것도 하지 않는 테란이다. 락다운·플레어·야마토·공격 대상 역할만 한다.
- 판정
  - 다음 관측에 그 명령의 `CommandFailed`(031-R10)가 없으면 "받아들여짐"으로 센다.
  - 100개 모두 1회 이상 받아들여지면 통과다.
  - 비무장 유닛(드랍십·베슬)에게 Attack_Pos를 준 다음 관측에서 order가 Move인지도 본다.
- 준비 동작
  - 가스 절약: 업그레이드는 시작 직후 취소한다(Cancel_Upgrade). 파이어뱃·골리앗·레이스·발키리는 생산 직후 취소한다(Cancel_Train). 클로킹 필드 연구도 취소한다(Cancel_Research). 머신샵은 한 번 취소 후 다시 짓는다(Cancel_Addon).
  - Repair_Unit 대상: 아군 핵으로 엔지니어링 베이를 손상시킨다.
  - Heal 대상: 스팀팩으로 손상된 마린.
  - Restoration 대상: 아군 기계 유닛에 이레디에이트.
  - 수송: 드랍십·벙커에 태운다.
  - 커맨드센터 이착륙: 두 번째 CC로 한다.
- 다 덮으면 `StopGame`으로 끝낸다.
- 실행: `docs/commands.md` "액션 시나리오(031-AC1)".

### 진행 기록 (학습 PC, debug 빌드, 테스트 전용 hostd 1게임 동시, Idle 우선순위)
| 회차 | 결과 | 원인·수정 |
|---|---|---|
| 1 | 16/100, 12,403프레임에 Elimination | 같은 건물 주문을 매 스텝 반복해 일꾼이 건설 장소로 가다가 다른 주문으로 끌려감. Move_Unit·Smart_Unit이 자기 자신을 대상으로 골라 실패(오류 6·25). 정찰 SCV가 건설에 뽑혀 Move가 Unit_Busy(3). Elimination 원인은 확인하지 못했다(이후 회차에서는 재현되지 않음) |
| 2 | 82/100, 시간 제한 무승부 | 테스트 쪽 수정: 건물 종류별 주문 간격 30초, 필요한 건물 재건설, 대상에서 선택 유닛 제외, 건설에 뽑힌 정찰 교체. 이륙 직후 내린 착륙 명령은 받아들여졌지만 배럭이 계속 떠 있어 바이오닉이 막힘 |
| 3 | 97/100 | 떠 있는 건물 재착륙을 추가. 배틀크루저(보급 6) 자리가 없었고, 메딕·베슬이 아군 핵에 죽어 Restoration을 못 함 |
| 4 | 덮기 완료(59초) 뒤 테스트가 멈춤 | hostd는 `StopGame`에 답하지 않는데 테스트가 답을 기다림. 테스트를 수정 |
| 5 | **100/100 통과**, 비무장 대체 통과 | 마지막 첫 수락 프레임 27,065(게임 약 19분). 벽시계 46초(재build와 동시 실행) |

- 산출물(git 미추적): 메인 작업 사본 `reports/raw/t7_031_ac1/`
  - `run1.log`, `run2.log`, `run3.log`, `run5.log`
  - 5회차 표 `openbw_actions_031_ac1.md`, 리플레이 `openbw_actions_031_ac1.rep`

### 실게임에서 본 거절 (5회차와 3회차)
게임이 거절했지만 재시도에서 받아들여진 경우다. AC는 통과지만 마스크와 게임 판정의 차이를 보여 준다.

| 액션 | BWAPI 오류 | 횟수 | 해석 |
|---|---|---|---|
| Train_SCV | 3 Unit_Busy | 3회차 48 | 추정: 애드온을 짓는 CC에 생산 명령을 줬다. `World::producers`·`pick_building`이 애드온 건설 중인 건물을 빼지 않는다 |
| Train_Vulture, Upgrade_TitanReactor, Research_EMP·Irradiate | 3 | 수 회 | 같은 원인으로 추정(애드온 건설 중인 팩토리·사이언스 퍼실리티) |
| Move_Unit, Smart_Unit | 6 Incompatible_State | 각 4 | 게임 초반 SCV가 다른 SCV를 따라가거나 우클릭할 때. 미확인 |
| Heal_Unit | 6 | 4 | 메딕이 자동으로 먼저 치료했거나 대상 상태가 바뀐 것으로 추정 |
| BuildAddon_* | 19 Unbuildable_Location | 각 1 | 애드온 자리에 유닛이 서 있었다. 034는 건물만 막힌 것으로 본다(현재 관측 기준, 실행 실패로 처리, 034 위치 마스크 비고와 일치) |
| Build_* 일부 | 3 / 11 | 수 회 | 같은 Act의 두 명령이 같은 자원을 썼거나 일꾼이 바쁜 경우(테스트 스크립트 영향 포함) |

- 1회차에서 본 것: 034 마스크는 Move_Unit·Smart_Unit·SetRally_Unit에서 **선택 유닛 자신을 대상으로 허용**한다. 게임은 이를 거절한다(오류 6·25). executor(031-R4)도 거절하지 않는다.
  - 정정(2026-10-19): 실측 거절은 Move_Unit·Smart_Unit뿐이다. SetRally_Unit은 게임·BWAPI(`canSetRallyUnit`)가 건물 자신을 받는다(랠리 해제). 그래서 034-R9에서 예외로 두었다(아래 진행 기록, 사용자 확정).

## 043-AC4
- 코드(`crates/uranium-ladder/src/limit.rs`, `api.rs`)
  - `Limiter`가 실제로 내보낸 요청 시각을 기록한다.
  - 실행이 끝나면(오류로 끝나도) 요약 한 줄을 출력한다.
    `requests: N paced, min gap X s, max K in any 1 s, limit R req/s: within|OVER (043-R4, 043-AC4)`
  - URL·식별자는 쓰지 않는다(043-R11).
  - 판정: 어떤 1초 구간에도 `ceil(rps)`개 이하이고, 최소 간격이 `1/rps`의 95% 이상이다. 5%는 sleep 깨어남 지연이다. 슬롯 자체는 정확히 `1/rps` 간격이다.
- 테스트
  - 가짜 시계로 판정을 확인한다.
  - 가짜 HTTP 서버로 `collect` 전체 흐름(연결 탐색 2, 리더보드 목록·페이지, 프로필, 경기 정보, 다운로드 = 요청 7개)을 10 req/s로 돌린다.
  - 서버가 받은 시각 간격이 80ms 이상(100ms − loopback 지터 20ms)이고, 수집기 자체 기록도 기준 안이다.
- 기존 수집 기록(공용 드라이브 `datasets/replays/ladder/collect.jsonl`, 읽기만 함)
  - 2,068줄(ok 2,000, failed 68 `schema: missing field`), 3회 실행, 10-16 23:23 ~ 10-17 03:18
  - 기록에는 경기마다 시작 시각만 있다. ok 경기는 요청 2개(경기 정보 + 다운로드), 실패는 1개 이상으로 보고 하한을 셌다.

    | 구간 | 하한 요청 수 최대 | 0.5 req/s 허용(구간 ÷ 2 + 1) |
    |---|---|---|
    | 300초 | 149 | 151 |
    | 60초 | 32 | 31 |
    | 10초 | 7 | 6 |

  - 짧은 구간에서 1개 넘는 것은 경기 단위 시각의 경계 효과다. 경기 시작 시각에 그 경기의 요청을 모두 넣어 세기 때문이다.
  - 연속 ok 경기 간격은 최소 2.49초, 1퍼센타일 3.13초, 중앙값 4.00초다. 요청 2개 × 2초와 맞는다.
  - 결론: 위반 흔적은 없다. 다만 요청 단위 증명은 다음 수집부터 새 요약 줄로 한다.

## 빌드 뒤 측정 체크리스트 (재build 종료 후, 다른 GPU·CPU 부하 없을 때)
### 1. 060-AC4 CUDA 벤치 리포트 (학습 PC)
```
cargo run -p bench --release --features cuda -- --backend cuda --out reports/bench_cuda_host.md
```
- 전제: `.env`에 `URANIUM_AUTOTUNE_CACHE_DIR`가 있다. bench가 `.env`를 읽고 autotune·fusion을 켠다(041-R15).
- Windows에서는 프로세스마다 autotune을 다시 측정한다. warmup 3회가 이를 흡수한다.
- 예상 출력: `reports/bench_cuda_host.md`
  - 모든 케이스 행, Flex 대비 최대 오차 열이 있다.
  - f32 케이스는 032-AC1 CUDA 기준 1e-2 이하다(TF32, 032-R8).
  - `net_act_*_b16_n256`에 p95가 있다(m5 기준 CUDA f32 p95 약 61ms).
- 통과 조건: 리포트가 생기고, 정확도 열이 기준 안이다.

### 2. 032-AC2 추론 지연 (B16·N256, p95)
- 학습 PC: CUDA와 같은 PC Vulkan을 비교한다.
  ```
  cargo run -p bench --release --features cuda -- --backend cuda   --filter net_act --out target/ac2_cuda.md
  cargo run -p bench --release --features cuda -- --backend vulkan --filter net_act --out target/ac2_vulkan.md
  ```
  - 통과: `net_act_lstm_b16_n256`·`net_act_gtrxl_b16_n256` p95가 CUDA < Vulkan이다(2026-10-16 결정).
- Linux 호스트 A(Vulkan)
  ```
  cargo run -p bench --release -- --backend vulkan --filter net_act --out reports/bench_linux_vulkan_ac2.md
  ```
  - 예상: 이전 측정은 f32 근사 87ms(M1), Vulkan 70~170ms(M5)였다. 초안 40ms는 넘을 것이다.
  - 기준은 측정값을 보고 사용자에게 다시 묻는다(032-AC2 문구).

### 3. 031-AC1 다른 호스트 재확인 (선택)
- 재build가 끝난 Linux 호스트에서 같은 시나리오를 1회 돌려 Linux OpenBW에서도 100/100인지 본다.
  ```
  URANIUM_TEST_HOSTD=<호스트 hostd> URANIUM_TEST_AGENT_LISTEN=0.0.0.0:0 URANIUM_AGENT_PUBLIC_ADDR=<이 PC 주소> \
    cargo test -p uranium-env --test openbw_actions -- --ignored --nocapture
  ```

## 질문 (사용자 결정 필요)
1. **애드온 건설 중인 건물**(2026-10-19 결정·반영, 034-R8, 아래 진행 기록): `pick_building`(031-R1)과 Train·Research·Upgrade 마스크가 애드온을 짓는 건물을 후보에서 빼지 않는다. 그래서 게임이 Unit_Busy로 거절하는 것으로 추정된다(실게임 3회차 Train_SCV 48회).
   - 빼도록 고치면 마스크가 바뀐다. 재build 중인 shard 생성 경로(레이블 무효 판정)에도 영향이 있을 수 있어 이번에는 고치지 않았다.
   - 고칠지, 고친다면 언제(재build 뒤 RL 전) 할지 결정이 필요하다.
2. **자기 자신 대상**(2026-10-19 결정·반영, 034-R9, SetRally_Unit 예외는 사용자 확정 2026-10-19, 아래 진행 기록): Move_Unit·Smart_Unit·SetRally_Unit에서 선택 유닛 자신을 대상으로 고를 수 있고, 게임은 거절한다. 마스크·executor에서 막을지 결정이 필요하다(현재는 031-R10 실패로 기록되고 RL이 피하도록 학습).
3. **032-AC2 Linux Vulkan 기준**: 체크리스트 2 측정 뒤 기준값을 정해 주기 바란다.

## 진행 기록: 질문 1·2 반영 (사용자 결정 2026-10-19)
- 결정: 두 마스크 문제를 지금 고친다. 데이터는 다시 build하지 않는다.
  - 진행 중인 전체 재build는 d452c29 바이너리를 그대로 쓴다.
  - 마스크는 학습 시점에 shard 관측으로 계산하므로 학습에는 바로 반영된다.
  - 다만 build의 040-R19 시점 걸러냄(무효 레이블 20% 이상)은 옛 마스크로 판정했다. 알려진 불일치이고 영향은 아래 측정처럼 무시할 수준이다.
- 034-R8(애드온 건설 중인 건물)
  - 완성된 건물이 `CONSTRUCTING`이거나 애드온 자리에 미완성 애드온이 있으면, `World::producers`(Train·Research·Upgrade의 `pick_building`)와 `can_lift`에서 뺀다.
  - 근거: BWAPI `canTrain`(`isConstructing`)·`canResearch`/`canUpgrade`/`canLift`(`isIdle`)가 Unit_Busy를 낸다.
- 034-R9(자기 대상 금지)
  - `target = "unit"` action_type의 대상에서 이번 선택을 뺀다. 모델 대상 마스크(샘플·teacher forcing, Bool 연산), executor, BC 레이블 모두 같다.
  - 레이블은 041-R11처럼 대상 유닛만 선택에서 빼고, 남는 유닛이 없으면 `target_selected`로 무효다.
  - 모든 대상이 선택에 들어 남는 대상이 없으면, 모델은 원래 마스크를 쓰고 executor가 거부한다.
- **지시와 다르게 한 점(사용자 확정 2026-10-19: 예외 유지)**: `SetRally_Unit`은 예외(`self_target = true`)로 두었다.
  - BWAPI `canSetRallyUnit`에는 자기 대상 검사가 없다. 게임도 건물 자신을 랠리 대상으로 받는다(랠리 해제).
  - 처음 리포트에서 SetRally_Unit도 거절된다고 쓴 것은 실측이 아니었다(실측 거절은 Move_Unit·Smart_Unit뿐).
  - 처음 구현처럼 SetRally_Unit까지 막으면 표본 레이블 564개(사람의 랠리 해제)가 무효가 됐다.
- 레이블 영향(`uranium-train label-check`, 재build 출력 중 완성 shard 3개를 로컬로 복사해 읽기만 함: stardata p00 2개, scr_ladder p0 1개)

  | 기준 | 게임 | 레이블 | 무효 | 비율 |
  |---|---|---|---|---|
  | 변경 전(d452c29 release) | 65 | 135,411 | 7,290 | 5.384% |
  | 변경 후, SetRally_Unit도 막음(첫 구현) | 65 | 135,411 | 7,912 | 5.843% |
  | 변경 후(최종, SetRally_Unit 예외) | 65 | 135,411 | 7,348 | 5.427% |

  - 최종 증가 58개는 모두 `Smart_Unit target_selected`이다.
  - Lift 2개는 무효 사유만 `select_cap`에서 `type_mask`로 바뀌었다.
  - Train·Research·Upgrade 레이블의 무효 수는 바뀌지 않았다.
- OpenBW 시나리오 재실행(6회차, Idle): **100/100**, 비무장 대체 통과, 벽시계 65초.
  - Train Unit_Busy 거절은 없다(3회차 Train_SCV 48회가 원인).
  - 자기 대상 오류(1회차 오류 25)도 없다. 남은 거절은 5회차와 같다.
    - Move_Unit·Smart_Unit 오류 6 각 4회: 게임 초반 SCV가 다른 SCV를 따라갈 때로, 자기 대상이 아니다. 원인 미확인.
    - Heal 오류 6 4회.
    - 애드온 자리에 유닛(19), 자원 부족(11), Unit_Busy(3)가 몇 회 있다(건설·TankMode).
  - 로그: 메인 작업 사본 `reports/raw/t7_031_ac1/run6.log`

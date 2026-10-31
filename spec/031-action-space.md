# 031 — Action Space & Executor

- **Status**: Approved (2026-10-15, 2026-10-17 개정: 라이브 게임 맵 사전계산 키(AMB-031-4), executor 거부 시 동작(031-R4, AMB-031-5). 2026-10-19: 맵 사전계산 생성 경로 표기를 실제(`uranium-replay`)로 정정. 2026-10-19: AC3 미분류 명령 처리 구체화(사용자 결정, 액션 감사 F7). 2026-10-19 액션 감사: 031-R10 다수 유닛 명령의 실패 판정 명시(011-R20, 사용자 결정 2026-10-19), 031-AC1 판정 강화. 2026-10-21 개정: 031-R12 GG 항복 판정(사용자 결정, AMB-031-6). 2026-10-31 개정(사용자 결정, `reports/m6_action_audit.md` §11 U-1·U-2·U-6): R1 서플라이 부족 시 Train 후보, R10 유닛별 거절 기록과 "유닛별 거절 기록" 절, AC1 시나리오 추가. 2026-10-31 리뷰 반영(메인 결정): 유닛별 거절 파일 위치(plan 실행은 시작 때 비움, 평가 작업은 `results/unit_rejections/<host>.jsonl`), AC1 ①·③ 판정 구체화. 2026-10-31 통합 시험 정정: AC1 ② 잔상 Attack의 관측 order는 AttackUnit(10))
- **Related**: [010](010-game-protocol.md), [030](030-observation.md), [032](032-model-architecture.md), [034](034-action-table.md), [040](040-replay-dataset.md)

## Purpose
정책이 내는 **의미 단위 액션**의 구조와, 이를 프로토콜 `Command`로 바꾸는 최소 규칙 계층(executor)을 정의한다. action_type 전체 목록과 인자 테이블은 [034](034-action-table.md)에 있다.

## 확정 사항 (2026-10-15)
| 항목 | 결정 |
|---|---|
| 정책 구조 | 단일 end-to-end 정책. 매크로와 마이크로를 같은 헤드 체인으로 처리 |
| executor 범위 | **최소** (AMB-031-1 Resolved) |
| APM 제한 | **없음** (AMB-031-2 Resolved) |
| delay | 1–32프레임 비균등 bucket 유지 |
| 선택 상한 | **12기**. 게임 동시 선택 제한과 같다 |

## 액션 구조 (자기회귀 순서)
```
1 action_type  ∈ A (~100, 034)
2 delay        ∈ 16 buckets {1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32} 프레임 (다음 관측까지)
3 queued       ∈ {0,1}                         (action_table.uses_queued)
4 selected     ⊆ 아군 entities, 1..max_select(≤12), END 토큰 (action_table.uses_select)
5 target_unit  ∈ entities                     (target = "unit")
6 target_loc   ∈ 128×128 빌드타일              (target = "loc")
```
- 타깃 종류가 다른 액션은 **서로 다른 action_type**으로 분리한다(예: `Attack_Unit` / `Attack_Pos`). 그래서 5번과 6번 헤드는 상호 배타다.
- 각 action_type이 쓰는 헤드, 선택 자격, 타깃 자격, 위치 클래스는 [034](034-action-table.md) 테이블이 결정한다.

## Executor 규칙 (최소)
| ID | 규칙 |
|---|---|
| 031-R1 | **MUST** `executor = pick_building` 액션(Train, Research, Upgrade, ScannerSweep): 가능한 건물 중 큐가 가장 짧은 건물을 고른다(Scan은 energy 최대 Comsat). tie-break는 unit id 오름차순으로 결정론적으로 한다. **Train 서플라이(2026-10-31 사용자 결정, 액션 감사 U-1)**: 서플라이가 부족하면 생산 대기열이 비지 않은(1~4개) 건물만 후보이고, 순서는 같다(대기열 길이, id). 엔진은 대기열이 빈 건물에만 서플라이를 검사하기 때문이다(034-R3). 서플라이가 충분하면 모든 후보에서 고른다. type_mask(034-R3)와 executor는 같은 후보 함수를 쓴다 |
| 031-R2 | **MUST** `executor = pick_worker` 액션(Build_*): 대상 타일까지 가장 가까운 비운반·비건설 SCV를 고른다. 운반 중인 SCV만 있으면 그중 가장 가까운 SCV를 고른다. **건설 위치는 정책이 고른 타일(좌상단)을 그대로 쓰고, 위치 보정은 하지 않는다.** 실행 시점에 위치가 무효면 명령 실패로 기록하고 재시도하지 않는다 |
| 031-R3 | **MUST** 선택 상한은 12기다. selected 헤드는 최대 12 step 뒤에 END로 끝난다. 액션 하나는 Command 하나로 변환하며 분할하지 않는다. 12기를 넘는 군대에 명령하려면 여러 액션을 낸다 |
| 031-R4 | **MUST** 무효 액션 마스크는 [034](034-action-table.md)의 정의를 따른다. 마스크는 BC, RL, 실전에서 동일하다. 모델 정책(RL·평가·실전)에서 executor가 액션을 거부하면 명령 없이 샘플된 delay만큼 진행하고, 거부 건수를 이유별로 세션 기록에 남긴다(2026-10-17, AMB-031-5). 에이전트 기억(030 §4 직전 행동)에는 거부 여부와 관계없이 샘플된 액션과 선택을 기록한다. BC 레이블 처리(041-R11)와 같은 규칙이다 |
| 031-R5 | ~~APM 제한~~ — **삭제(2026-10-15)**. APM 제한을 두지 않는다. SC:R 네트워크 턴당 커맨드 버퍼 한도는 기술적 제약이며 브리지가 큐잉으로 처리한다([012](012-bridge-remastered.md)) |
| 031-R6 | ~~자동 채취·자동화~~ — **삭제(2026-10-15)**. 새 SCV의 채취를 포함해 모든 유닛 지시는 정책이 직접 내린다 |
| 031-R7 | **MUST** executor는 결정론적이어야 한다. 같은 입력이면 같은 Command를 낸다. BC 레이블 역변환이 가능해야 하기 때문이다([040](040-replay-dataset.md)) |
| 031-R8 | **MUST** 타깃 스킬(`max_select = 1`)은 시전 유닛을 1기만 선택한다. BW에서는 여러 시전 유닛을 선택하면 전원이 시전해 에너지를 낭비한다 |
| 031-R9 | **MUST** `delay`는 이 액션 실행 후 다음 관측까지 진행할 프레임 수다. lockstep 모드에서는 정확히 그 프레임만큼 진행한다. realtime 모드에서는 그 프레임을 목표로 다음 관측을 요청한다 |
| 031-R10 | **MUST** 명령 실패(위치 무효, 대상 소멸 등)는 이벤트(`CommandFailed`, 010)로 기록한다. 다음 관측의 스칼라 feature(직전 액션 실패 여부, 030 §4)로 제공한다. 다수 유닛 명령은 선택 중 1기 이상이 받으면 성공이고, 아무도 받지 못했을 때만 실패다(브리지 그룹 명령, 011-R20, 사용자 결정 2026-10-19, 액션 감사 F3). **유닛별 거절(2026-10-31 사용자 결정, 액션 감사 U-6)**: 받지 못한 유닛은 유닛마다 `UnitRejected`(010-R14)로 온다. 진단 기록일 뿐이고 `last_failed`와 성공·실패 판정은 바꾸지 않는다. 에이전트는 아래 "유닛별 거절 기록"대로 남긴다 |
| 031-R11 | **MUST** 32프레임을 넘는 무행동 구간은 `NoOp(delay=32)`을 연속으로 이어 표현한다. 마지막 남은 구간은 가장 가까운 하위 bucket의 NoOp으로 채운다. 리플레이 레이블 생성([040](040-replay-dataset.md))과 RL 실행 모두 같은 규칙을 쓴다 |
| 031-R12 | **MUST** GG 항복(2026-10-21 사용자 결정, AMB-031-6). 모델 액션이 아니라 모델 정책(에이전트)의 규칙이고, 판정 코드는 `uranium-env`에 두어 에이전트와 `bc-eval`(041-AC4)이 같은 함수를 쓴다. 결정마다 P = (win_open + 1) / 2(032-R18)로 검사해 다음을 모두 만족하면 항복한다. ① 게임 프레임 ≥ 14,400(10분, 24fps 환산, 020과 같다). ② 최근 720프레임(30초) 동안의 모든 결정에서 P ≤ 0.05. 연속 구간은 P ≤ 0.05인 첫 결정의 프레임부터 세고, P > 0.05인 결정이 하나라도 나오면 끊긴다. 10분 전 결정도 구간에 들어간다(①은 항복 시점만 제한). ③ 규칙 조건 (a) 또는 (b), 이번 결정 시점 기준: (a) **병력 격차** — 내 현재 병력 supply ≤ 적 최대 관측 병력 supply ÷ 3. 병력은 일꾼(SCV·Probe·Drone)과 건물을 뺀 유닛이고 supply는 유닛 타입 테이블의 `supply_required`를 쓴다(오버로드·라바 등 0인 유닛은 자연히 빠진다). 적 최대 관측 병력은 `AgentMemory::enemy_max_seen`(타입별 동시 관측 최대 수, 030 §4)에 supply를 곱해 합한 값이며, 0이면(적 병력을 본 적 없음) (a)는 거짓이다. (b) **붕괴** — 완성된 배럭·팩토리·스타포트(떠 있는 것 포함)가 하나도 없거나, 완성된 SCV가 5기 이하. 입력은 자기 시야(010-R5)와 자기 기억뿐이고 privileged는 쓰지 않는다. 모드: `Off`(판정 안 함), `Shadow`(판정만 하고 첫 성립 프레임을 세션 통계에 남긴다, 나가지 않음), `On`(첫 성립 결정의 `Act`에 명령 없이 채팅 문자열을 담은 `surrender`를 실어 보내고(010-R13) 그 세션은 더 결정하지 않는다, 첫 성립 프레임을 통계에 남긴다). win_open이 없는 체크포인트에서는 성립하지 않고 경고를 1회 남긴다. 모드와 채팅 문자열은 plan 옵션(020-R14)으로 정하고, SC:R 실전 설정의 기본은 `On`·`"gg"`(012-R11) |

## 유닛별 거절 기록 (031-R10, 2026-10-31 사용자 결정, 액션 감사 U-6)
- 입력: 브리지가 보낸 `UnitRejected`(010-R14). 지금은 OpenBW 브리지만 보낸다(SC:R은 012-R14).
- **세션 기록**(`SessionRecord`, 세션마다 한 줄인 결과 파일 — plan 실행 `results.jsonl`, RL `sessions.jsonl`, 평가 작업 `results/<host>.jsonl`)에 두 필드를 더한다.
  - `unit_rejections`: 그 세션의 `UnitRejected` 합계.
  - `unit_rejections_by`: 키 `"<command>/<unit_type>/<error>"` → 수. 세 값은 10진 BWAPI id다(`UnitCommandType`, `UnitType`, `Error`). 유닛을 찾지 못한 거절의 `unit_type`은 `none`이다.
  - executor 거부 수(031-R4, 모델 통계의 거부 항목)와 이름이 겹치지 않게 둔다. SC:R 세션은 0과 빈 표다.
- **상세 파일**: 거절마다 JSON 한 줄을 덧붙인다. 필드는 `game_id`, `slot`(`Hello` 값), `frame`(이벤트가 실린 `RawFrame.frame`), `cmd_index`, `order`(그 `Command::Unit`에 에이전트가 넣은 `CmdOrder`, serde JSON), `unit`, `unit_type`(없으면 null), `command`, `error`다.
- **파일 위치**(2026-10-31 리뷰 반영, 메인 결정):
  - plan 실행(`run_plan`): `results.jsonl` 옆 `unit_rejections.jsonl`. plan 실행은 시작할 때 이전 `results.jsonl`·`games.jsonl`을 지우고 새로 쓰므로, 이 파일도 그때 함께 지운다(한 폴더의 세 파일이 같은 실행의 기록이 되도록). 실행 중에는 덧붙인다.
  - RL: `sessions.jsonl` 옆 `unit_rejections.jsonl`. 덧붙이기만 하고 비우지 않는다.
  - 평가 작업(020-R15): 세션 기록이 장비별이므로 job 폴더의 `results/unit_rejections/<host>.jsonl`로 장비마다 따로 쓴다(같은 파일에 두 장비가 덧붙이지 않는다, 020-R15 ①). 하위 폴더에 두는 것은 `results/*.jsonl`을 장비별 세션 기록으로 읽는 쪽이 이 파일을 세션 기록으로 잘못 읽지 않게 하기 위해서다. 덧붙이기만 한다. 평가 단계의 합친 기록(020-R15 ⑨)에는 넣지 않는다.
- 진단 전용이다. 관측 feature, `last_failed`(030 §4), 보상, 학습 데이터에 쓰지 않는다.

## 맵 사전계산 (모든 맵)
- 시작 위치별 본진·앞마당 베이스 위치, 초크 포인트, 확장 순서, 입구 벽 후보 타일.
- BC 데이터는 `uranium-replay build`가 리플레이를 재생할 때 맵마다 `<URANIUM_MAPS_DIR>/<CHK 해시>.toml`을 만든다(알고리즘은 `uranium-env` `mapgen::precompute`). 라이브 대상 맵의 자원 파일은 `uranium-replay mapgen`이 만든다(아래). 사전계산에는 CHK 유닛 배치에서 뽑은 중립 자원(종류·위치·초기량)도 들어가며, 시야 밖 자원 엔티티의 초기값으로 쓴다(030 §2, 010-R5 예외, AMB-030-3). (2026-10-19 정정: 이전 표기 `tools/mapgen`·`assets/maps/`는 실제와 달랐다)
- 대상 맵(투혼, 서킷브레이커)만이 아니라 BC 데이터에 나오는 **모든 맵**에 같은 맵 무관 알고리즘으로 적용한다. S1(전체 맵)과 S2(대상 맵)의 입력 분포를 같게 하기 위해서다 (AMB-031-3 Resolved, 2026-10-16).
- **관측 채널(`base_loc`, `choke`, `wall_candidate`)로만 제공한다. executor가 위치를 강제하거나 보정하지 않는다.**
- **BC 학습·평가에서 사전계산이 없을 때 (2026-10-19, 리뷰 B-2·G-F2, BC 전 리뷰 계획 승인)**: `URANIUM_MAPS_DIR`는 필수다(상대 경로 기본값 없음). `bc`·`bc-eval`·`label-check`는 시작할 때 고른 게임의 맵 해시마다 현재 버전 사전계산이 있는지 확인하고, 없으면 빠진 해시를 나열하고 실패한다. 로더도 사전계산이 없는 게임을 오류로 처리한다(채널 0으로 조용히 학습하지 않는다). 라이브 게임은 아래 규칙을 따른다(대상 맵은 세션 거부, 그 외 맵은 경고 후 진행).
- **라이브 게임의 사전계산 (확정 2026-10-17, AMB-031-4)**
  - BC shard의 사전계산 키는 리플레이 CHK의 SHA-256이고, 라이브 게임의 `GameInfo.map_hash`는 맵 파일(.scx)의 SHA-256이다(010). 키가 달라 그대로는 라이브 게임에서 사전계산을 찾지 못한다.
  - RL·평가 대상 맵 2개(`config/assets.toml`)만 지원한다. `uranium-replay mapgen <scx>...`가 맵 파일에서 CHK를 꺼내 자원 배치를 `<URANIUM_MAPS_DIR>/<맵 파일 SHA-256>.resources.toml`로 저장한다(오프라인, 맵마다 한 번).
  - 에이전트 세션은 게임 시작 시 `<맵 파일 SHA-256>.toml`(사전계산)이 있으면 그것을 쓰고, 없으면 `.resources.toml`과 `GameInfo.static_map`으로 BC와 **같은 알고리즘**(`mapgen::precompute`)을 돌려 만든다. 리플레이 경로와 입력(정적 맵 + CHK 자원)이 같으므로 결과도 같다.
  - 그 외 맵은 사전계산 없이(`base_loc`·`choke`·`wall_candidate` 채널 0) 진행하고 경고를 남긴다. 브리지·프로토콜은 바꾸지 않는다(001-R3).
  - **대상 맵 예외(메인 결정 2026-10-19)**: 맵 계열(`config/map_families.toml`)이 투혼(`fighting_spirit`)·서킷브레이커(`circuit_breaker`)인 게임에서 위 두 방법으로도 현재 버전 사전계산을 얻지 못하면(파일 없음, 이전 버전, `URANIUM_MAPS_DIR` 미설정) 에이전트는 그 세션을 거부하고 이유를 오류로 남긴다. BC 평가 대전과 RL이 채널 0·맵 자원 엔티티 없이 진행되는 것을 막기 위해서다. 그 외 맵은 바로 위 규칙(경고 후 진행)을 유지한다.
  - 맵 계열(`map_family`, 040-R15)은 BC와 같은 `config/map_families.toml` 규칙(제목 패턴)으로 `GameInfo.map_name`에서 정한다. 판정 코드는 `uranium-env` 라이브러리에 둔다.

## Acceptance Criteria
- 031-AC1: [034](034-action-table.md)의 모든 action_type이 OpenBW에서 실제로 실행되는지 시나리오 테스트로 확인한다. 판정(2026-10-19 액션 감사 강화): 모든 action_type이 1회 이상 받아들여지고, 거절은 정당화된 것만 있다: 현재 관측 기준 위치 마스크라 건물 자리에 서 있는 유닛 때문에 생기는 Build_*·BuildAddon_*·Land_*의 Unbuildable_Location, 시나리오가 한 Act에 경제·커버리지 명령을 함께 보내 생기는 자원 부족(에이전트는 결정당 명령 1개, 031-R3). 그 밖의 BWAPI 오류는 마스크·executor·브리지 불일치로 보고 실패다. 시야 밖 한 번 본 미네랄에 Smart_Unit·Gather_Unit이 받아들여지고 SCV가 채취 order로 바뀌는지도 본다(011-R20). 2026-10-31 추가(사용자 결정, 액션 감사 U-1·U-2·U-6): ① 서플라이가 막힌 상태에서 대기열이 있는 생산 건물(CC·배럭 등)에 Train이 받아들여진다. `CommandFailed`가 없는 것만으로는 부족하고, 다음 관측에서 그 건물의 생산 대기열 길이가 늘어야 한다(034-R3, 011-R23). ② 한 번 본 뒤 시야 밖이 된 적 건물에 Attack_Unit·Smart_Unit이 받아들여지고 `CommandFailed`가 없으며, 그 자리가 시야 밖인 동안 Attack_Unit을 받은 유닛의 관측 order는 AttackUnit(10)이고 `order_target`은 없으며 `order_pos`는 기억한 건물 위치다(게임에 보낸 order는 MoveToAttack(9)이지만 BWAPI가 9를 AttackUnit으로 보고한다, 011-R20 ①-2, 2026-10-31 통합 시험으로 정정). ③ `UnitRejected`를 명령·유닛 종류·오류별로 보고한다. 선택이 공격도 이동도 할 수 있는 유닛만인 잔상 명령(②)에는 `UnitRejected`가 없다(2026-10-31 리뷰 반영: 공격은 하되 이동하지 못하는 유닛의 잔상 Attack_Unit은 거절될 수 있다, 034 Risks). 그 밖의 유닛별 거절은 판정 기준에 넣지 않고 보고만 한다.
- 031-AC2: 마스크된 액션이 실행되지 않는다(property test).
- 031-AC3: 리플레이 사람 명령을 의미 단위 액션으로 역변환할 때 커버리지 ≥ 95%다. 미분류는 `Unknown`으로 기록하고 학습에서 제외한다. 구체적으로(2026-10-19 사용자 결정, 액션 감사 F7, 레이블러 버전 2) 미분류 명령은 다음 명령까지의 delay를 가진 자기 스텝(`unknown` 레이블)이 되고 BC 가중치는 0이다. 그 시간을 `NoOp` 레이블로 채우지 않는다(레이블러 버전 1은 채웠다, 040 "Shard 포맷"). 에이전트 기억에는 `NoOp`으로 기록한다.
- 031-AC4: Build 액션 실행 시 건설 위치가 정책이 고른 타일과 정확히 같다.
- 031-AC5: 031-R12 판정 단위 테스트 — 10분 경계(14,399 불가·14,400 가능), 30초 연속(719프레임 불가·720 가능)과 P > 0.05 한 번에 끊김, 규칙 (a)만·(b)만·둘 다 거짓·적 병력 0, win_open 없음. 통합: OpenBW 1판에서 `On` 항복이 우리 패배로 집계되고(Aborted 아님) 리플레이에 채팅 문자열이 남는다.

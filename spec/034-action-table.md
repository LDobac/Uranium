# 034 — Action Table

- **Status**: Approved (2026-10-15, 2026-10-17 개정: Attack_Pos 선택 자격 can_move, Smart_Unit 선택 자격 can_right_click_unit, can_rally 완성 조건 제거, 무력화에 메일스트롬 포함 — AMB-041-1. 2026-10-19 개정(BC 전 리뷰 결정, 사용자 확정): R6 무력화 규칙을 모든 선택 자격·executor 선택에 명시(기존 규칙과 정합, B-11), R7 선택 집합에 건물이 들어가면 그 건물 1개만(E-4, SetRally max_select 12 → 1, Selected 헤드 마스크), AMB-034-1. 2026-10-19 T7 개정(사용자 결정): R8 애드온 건설 중인 건물 제외, R9 자기 대상 금지. 2026-10-19 액션 감사 개정(사용자 결정, F7): `Unload_Unit`(id 100) 추가 → 101개, 선택 자격 `is_transport_loaded`·타깃 자격 `loaded_unit`, action 스키마 2. 2026-10-19 액션 감사 개정(사용자 결정 2026-10-19, `reports/m6_action_audit.md` F1·F5·F6·F11·F15·F16): R10 대상 자격에서 생산 중·탑승 유닛 제외, Stop 선택 자격 `can_stop`, UnloadAll_Pos 선택 자격 `can_unload_pos`, `can_attack`에서 벙커 제외, 건설 중 SCV·시즈 전환 중 탱크는 Stop 외 명령 불가, Build_*는 `pick_worker` 후보가 있을 때만). 2026-10-31 개정(사용자 결정, `reports/m6_action_audit.md` §11 U-1·U-2, F4·F18): R3 Train 서플라이는 엔진 규칙(생산 대기열이 빈 건물에만 검사), 시야 밖 기억 적 건물을 `attackable_enemy`·`any_unit` 대상으로(브리지 잔상 명령, 011-R20 ①-2), R10, R2 feature 스키마 7(6과 로드 호환, 030-R5), AC4. 2026-10-31 리뷰 반영(메인 결정): `known_building`에서 마지막 목격 때 떠 있던 건물 제외, 변태 중 목격한 저그 건물 한계, Risks 이동 못 하는 유닛의 잔상 Attack_Unit)
- **Related**: [030](030-observation.md), [031](031-action-space.md), [032](032-model-architecture.md), [040](040-replay-dataset.md)

## Purpose
테란 action_type 전체 목록과 각 액션의 인자 규칙(헤드 사용 여부, 선택 자격, 타깃 자격, 위치 클래스, executor)을 정의한다. 아래 세 텐서의 의미도 정의한다.
- `type_mask`
- `unit_cap`(capability 클래스)
- `loc_class_mask`(위치 클래스)

## Requirements
| ID | 요구사항 |
|---|---|
| 034-R1 | **MUST** 테이블은 데이터 파일 `crates/uranium-env/tables/actions.toml`로 관리한다. env(마스크 계산), model(GPU 마스크 gather), replay(역변환)가 같은 파일을 쓴다 |
| 034-R2 | **MUST** action_type ID는 0부터 연속이고 0은 `NoOp`이다. ID를 변경하거나 재사용하려면 action 스키마 버전을 올려야 한다(032-R6). 액션 추가도 버전을 올린다: 스키마 2(2026-10-19)는 `Unload_Unit`(id 100)과 capability 클래스 `loaded_unit`을 끝에 더했다(K 47 → 48, feature 스키마 5). 같은 날 액션 감사 T8이 선택 자격 `can_stop`·`can_unload_pos`를 그 뒤에 더해 K = 50, feature 스키마 6이다(action id는 그대로라 action 스키마는 2). 기존 id의 의미는 같으므로 스키마 1로 만든 shard 레이블은 그대로 읽는다(040 "Shard 포맷"). 2026-10-31 feature 스키마 7은 action id·K를 바꾸지 않고 `type_mask`(R3 Train 서플라이)와 타깃 자격(`attackable_enemy`·`any_unit`, 기억 적 건물)의 의미만 바꿨다. 체크포인트는 버전이 다르면 읽지 않는다. 예외는 030-R5의 로드 호환 목록이다(feature 스키마 7은 6을 받는다) |
| 034-R3 | **MUST** `type_mask[a]`가 true인 조건은 아래 셋을 모두 만족하는 것이다. 아니면 false다. 해당 조건이 없는 항목은 자동으로 만족한 것으로 본다(vacuous true). |
| 034-R4 | **MUST** `sel_mask = unit_cap[:, select_cap[a]] ∧ owner==self ∧ entity_mask`, `tgt_mask = unit_cap[:, target_cap[a]] ∧ entity_mask`, `loc_mask = loc_class_mask[loc_class[a]]`. 이 gather는 GPU에서 수행한다([032](032-model-architecture.md) §5) |
| 034-R5 | **MUST** 유효한 타일이 하나도 없는 위치 클래스를 쓰는 action_type은 type_mask에서 닫는다 |
| 034-R6 | **MUST** 무력화 유닛(락다운·스테이시스·메일스트롬, OpenBW `unit_is_disabled`)은 명령을 받지 못한다(아래 "선택 자격 공통" 규칙의 적용 범위 명시, 정합. 2026-10-19 사용자 결정, B-11, AMB-034-1). 이 규칙은 이동·공격·캐스터 클래스만이 아니라 **모든 선택 자격 클래스**(일꾼, 수송, 스킬, 건물(이륙·착륙·랠리·애드온·취소) 등)에 적용하고, executor가 유닛을 고르는 경우(`pick_worker`, `pick_building`)에도 무력화 유닛·건물은 후보에서 뺀다. 고를 수 있는 후보가 없으면 해당 action_type은 type_mask에서 닫는다(R3 선택 자격과 같은 처리) |
| 034-R7 | **MUST** 건물 단일 선택(2026-10-19 사용자 결정 E-4, AMB-034-1): 스타1은 건물을 다중 선택할 수 없다(040-R3 ① 게임 규칙). 그래서 **모든 action_type**에서 선택 집합에 건물(떠 있는 건물·벙커·미사일 터렛 포함)이 들어가면 그 건물 1개만 선택한다. Selected 헤드 마스크(032 §2.5·§5): 첫 선택이 건물이면 다음 step은 END만 허용하고, 첫 선택이 건물이 아니면 이후 step에서 건물을 제외한다. 선택 자격이 건물뿐인 action_type(`SetRally_*`, `Lift`, `Land_*`, `BuildAddon_*`, `Cancel_*`)은 `max_select = 1`이다(이번 개정으로 `SetRally_Pos`·`SetRally_Unit`을 12에서 1로 바꿈). 생산·연구·업그레이드·스캔은 executor `pick_building`이 건물 1개를 고른다 |
| 034-R8 | **MUST** 애드온 건설 중인 건물(사용자 결정 2026-10-19, T7 실게임 확인): 완성된 건물이 애드온을 짓는 동안 게임(BWAPI `isConstructing`, `isIdle` 거짓)은 생산·연구·업그레이드·이륙을 Unit_Busy로 거절한다. 그래서 이 건물은 `pick_building` 후보(생산·연구·업그레이드, 031-R1)와 `can_lift`에서 빠진다. 판정은 관측의 `CONSTRUCTING` 플래그(완성 건물) 또는 애드온 자리의 미완성 애드온이다. 고를 건물이 없으면 해당 action_type은 type_mask에서 닫는다(R3) |
| 034-R10 | **MUST** 대상 자격의 존재 조건(사용자 결정 2026-10-19, 액션 감사 F1): 모든 타깃 자격 클래스는 **생산 중인 유닛**(생산 건물 안에서 미완성으로 만들어진 비건물 유닛, 관측의 `COMPLETED` 없음)과 **탑승한 유닛**(`LOADED` 또는 `loaded_in`)을 대상으로 삼지 않는다(탑승 유닛을 내리는 `Unload_Unit`의 `loaded_unit`만 예외). 사람은 이런 유닛을 클릭할 수 없고, BWAPI `canTargetUnit`은 미완성 비건물 대상을 Incompatible_State로 거절한다(T7 Move_Unit·Smart_Unit·Heal_Unit 오류 6의 원인). 건설 중인 건물은 계속 대상이다(게임·BWAPI 모두 허용). 알(`Zerg_Egg`·`Zerg_Lurker_Egg`)·코쿤·합체 중인 아콘·다크 아콘도 미완성이지만 대상이다(BWAPI `canTargetUnit`은 변태 중 유닛과 아콘을 예외로 받고 게임도 받는다, 리뷰 2026-10-19). 시야 밖 한 번 본 자원과 시야 밖 기억 적 건물(아래)은 이 조건과 무관하게 대상이다(2026-10-31 기억 적 건물 추가) |
| 034-R9 | **MUST** 자기 대상 금지(사용자 결정 2026-10-19, T7 실게임 확인): `target = "unit"`인 action_type에서 대상은 이번 선택에 든 유닛일 수 없다(게임·BWAPI는 따라가기·이동 유닛의 우클릭·공격·수리·스킬의 자기 대상을 거절한다). 예외는 `SetRally_Unit`이다: BWAPI `canSetRallyUnit`과 게임은 건물 자신을 랠리 대상으로 받는다(랠리 해제, 사람 리플레이에 흔함). 예외는 표의 `self_target = true`로 둔다. 대상 마스크는 샘플·teacher forcing 모두 선택 결과로 다시 막는다(032 §5, Bool 연산). 선택이 대상 자격 유닛을 모두 포함해 남는 대상이 없으면 그 행은 원래 마스크를 쓰고 executor가 거부한다(031-R4). executor도 선택 유닛을 대상으로 받지 않는다. BC 레이블(041-R11과 같은 규칙): 게임은 대상 유닛만 건너뛰고 나머지 선택에 명령을 실행하므로 레이블 선택에서 대상 유닛을 빼고, 남는 유닛이 없으면 무효(`target_selected`)다 |

### 034-R3 `type_mask[a]` 조건
1. **선행 조건**: `prereq`(자원, 가스, 서플라이, 테크, 필요 건물)를 만족한다.
   - **Train 서플라이(2026-10-31 사용자 결정, 액션 감사 F4·U-1 (a))**: 엔진 규칙을 따른다. 엔진(`build_queue_push`)은 생산 대기열이 5개면 거절하고, 서플라이는 그 건물의 대기열이 비어 있을 때만 검사하며, 자원은 항상 검사한다. 그래서 서플라이가 부족해도 `pick_building` 후보(031-R1) 중 생산 대기열이 1~4개인 건물이 있으면 서플라이 조건을 만족한 것으로 본다. 이때 executor는 대기열이 비지 않은 건물 중에서만 고른다(031-R1). 서플라이가 충분하면 이전과 같다. 사람이 서플라이가 막힌 동안(서플라이 디팟 건설 중) 배럭에 생산을 더 예약하는 행동이다. 자원·가스·테크·필요 건물 조건은 그대로다. 라이브: OpenBW 브리지는 BWAPI의 서플라이 검사를 이 규칙에 맞춘다(011-R23). SC:R 브리지는 명령 바이트만 보내고 게임이 같은 규칙으로 처리한다. BC 레이블 검사(`label-check`, 040-R19의 무효 레이블)도 같은 규칙이다. 마스크는 로드 때 계산하므로 shard는 다시 build하지 않는다. build 때 이미 정한 040-R19 필터 결과는 그대로 둔다.
2. **선택 자격**: `uses_select`이면 선택 자격 유닛이 1기 이상 있다.
3. **타깃 자격**: `target="unit"`이면 타깃 자격 엔티티가 1개 이상 있다.

## 테이블 스키마 (TOML)
```toml
[[action]]
id = 4
name = "Attack_Unit"
uses_select = true
max_select = 12          # 1..12
select_cap = "can_attack"
uses_queued = true
target = "unit"          # "none" | "unit" | "loc"
target_cap = "attackable_enemy"
loc_class = ""           # target = "loc"일 때만
executor = "none"        # "none" | "pick_worker" | "pick_building"
prereq = { minerals = 0, gas = 0, supply = 0, tech = "", requires = [] }
command = "UnitAttackUnit"   # 010 Command 매핑
```

## action_type 목록 (~100)
| 그룹 | action_type | select (max) | target | 비고 |
|---|---|---|---|---|
| 기본 | NoOp | – | – | delay만 사용 |
| 이동·전투 | Move_Pos, Move_Unit(follow), Attack_Pos, Attack_Unit, Patrol_Pos, Hold, Stop, Smart_Pos, Smart_Unit | can_move (Attack_Unit은 can_attack, Smart_Unit은 can_right_click_unit, Stop은 can_stop) (12) | loc / unit / none | Smart = 우클릭. 게임이 의미를 결정하며, 사람 리플레이 역변환에 필요. 유닛 우클릭은 시즈모드 탱크도 받는다(적이면 제자리 공격, OpenBW `right_click_action` 3), 땅 우클릭은 받지 않는다. Attack_Pos(어택땅)는 게임이 이동 가능한 모든 유닛에 적용하므로(OpenBW `unit_order_allowed`: AttackMove = 이동 가능) can_move다. 무기 없는 유닛은 이동만 하고 메딕은 HealMove로 바뀐다. BWAPI는 무장 유닛·메딕만 어택땅을 허용하므로 OpenBW 브리지는 나머지 이동 가능 유닛에 move를 보낸다(2026-10-17, AMB-041-1) |
| 일꾼 | Gather_Unit, ReturnCargo, Repair_Unit | is_worker (12) | target_resource / – / repairable | |
| 건설 ×12 | Build_{CommandCenter, SupplyDepot, Refinery, Barracks, EngineeringBay, MissileTurret, Academy, Bunker, Factory, Starport, Armory, ScienceFacility} | executor `pick_worker` | loc (클래스별) | 정책이 정확한 좌상단 타일 선택 |
| 애드온 ×6 | BuildAddon_{ComsatStation, NuclearSilo, MachineShop, ControlTower, PhysicsLab, CovertOps} | addon_capable_{cc, factory, starport, scifac} (1) | – | 어느 건물에 붙일지는 정책이 선택 |
| 생산 ×14 | Train_{SCV, Marine, Firebat, Medic, Ghost, Vulture, SiegeTank, Goliath, Wraith, Dropship, ScienceVessel, Battlecruiser, Valkyrie, NuclearMissile} | executor `pick_building` | – | |
| 연구 ×11 | Research_{Stim, Restoration, OpticalFlare, SpiderMines, TankSiege, EMP, Irradiate, Yamato, CloakingField, Lockdown, PersonnelCloaking} | executor `pick_building` | – | |
| 업그레이드 ×15 | Upgrade_{InfantryWeapons, InfantryArmor, VehicleWeapons, VehiclePlating, ShipWeapons, ShipPlating, U238Shells, IonThrusters, CharonBoosters, OcularImplants, MoebiusReactor, ApolloReactor, ColossusReactor, CaduceusReactor, TitanReactor} | executor `pick_building` | – | |
| 스킬(무타깃) | Stim, SiegeMode, TankMode, Cloak, Decloak | can_stim / can_siege / can_unsiege / can_cloak / can_decloak (12) | – | |
| 스킬(타깃) | SpiderMine_Pos, Lockdown_Unit, Nuke_Pos, Restoration_Unit, OpticalFlare_Unit, DefensiveMatrix_Unit, EMP_Pos, Irradiate_Unit, Yamato_Unit | can_cast_* (**1**) | loc / unit (클래스별) | 031-R8 |
| 스킬(다수) | Heal_Unit | can_heal (12) | healable | |
| 스캔 | ScannerSweep_Pos | executor `pick_building` | loc(any) | |
| 수송 | Load_Unit, UnloadAll, UnloadAll_Pos, Unload_Unit | can_be_loaded (12) / is_transport_loaded (12) / can_unload_pos (12) / is_transport_loaded (1) | loadable_transport / – / walkable / loaded_unit | UnloadAll_Pos(MoveUnload)는 드랍십만 받는다(BWAPI `canUnloadAllPosition`은 벙커를 Incompatible_UnitType으로 거절, 액션 감사 F6). `Unload_Unit`(2026-10-19 사용자 결정, 액션 감사 F7): 선택한 수송선(드랍십·벙커)이 탑승 유닛 1기를 내린다(리플레이 opcode 0x29, BWAPI `unload(unit)`, `uses_queued = false`). 대상은 선택한 수송선 안의 유닛이어야 하는데 마스크(독립 gather)는 "탑승 중"까지만 표현하므로, 다른 수송선의 탑승 유닛이면 executor가 거부하고(031-R4), BC 레이블 검사도 무효(`target_not_in_transport`)로 둔다. 테이블 끝(id 100)에 추가해 기존 id 0–99의 의미는 그대로다 |
| 건물 | Lift, Land_Pos, Land_CommandCenter_Pos, SetRally_Pos, SetRally_Unit | can_lift (1) / can_land_noncc (1) / can_land_cc (1) / can_rally (1) | – / landable_4x3 / cc_slot / any / any_unit | CC 착륙은 건설과 같은 자원 거리 규칙 적용. 건물은 1개만 선택(R7) |
| 취소 | Cancel_Construction, Cancel_Train, Cancel_Research, Cancel_Upgrade, Cancel_Addon | can_cancel_* (1) | – | |

## capability 클래스 (`unit_cap`, K = 50)
| 분류 | 클래스 |
|---|---|
| 선택 자격 | can_move, can_right_click_unit(can_move ∨ 시즈모드 탱크), can_stop(완성 비건물 ∨ 떠 있는 건물 ∨ 미사일 터렛. 시즈모드 탱크·건설 중 SCV·시즈 전환 중 탱크 포함 = BWAPI `canStop`, 액션 감사 F5), can_unload_pos(탑승 유닛이 있는 비건물 수송선 = 드랍십, F6), can_attack(무기 보유. 벙커는 무기·공격 order가 없어 게임이 건너뛰고 BWAPI가 Unable_To_Hit로 거절하므로 제외, F11), is_worker, can_be_loaded, is_transport_loaded, can_stim, can_siege, can_unsiege, can_cloak, can_decloak, can_lay_mine, can_cast_lockdown, can_cast_nuke, can_heal, can_cast_restoration, can_cast_flare, can_cast_matrix, can_cast_emp, can_cast_irradiate, can_cast_yamato, can_lift, can_land_noncc, can_land_cc, can_rally, addon_capable_{cc,factory,starport,scifac}(완성 ∧ 착지 상태 ∧ 애드온 미보유 ∧ 오른쪽 애드온 자리 2×2 비어 있음 ∧ 생산·연구 중 아님), can_cancel_{construction,train,research,upgrade,addon} |
| 타깃 자격 | (공통: 생산 중·탑승 유닛 제외, R10. 탑승 유닛이 대상인 loaded_unit만 예외) attackable_enemy(보이는 적 + 시야 밖이지만 본 적 있는 적 건물, 2026-10-31), any_unit(보이는 유닛 + 시야 밖이지만 한 번 본 중립 자원과 적 건물), own_unit, target_resource(미네랄·아군 정제소. 미네랄은 시야 밖이라도 한 번 본 것 포함), repairable(아군 기계·건물, 피해), lockdownable(적 기계, 미락다운), healable(아군 생체 지상, 피해), restorable(아군, 상태이상), flarable(적, 비건물), irradiatable(비건물), matrixable(아군), yamato_target(적), loadable_transport(아군 드랍십·벙커, 여유 공간), loaded_unit(수송선에 탑승한 아군 유닛. 자기 유닛뿐이라 시야 정보가 늘지 않는다(010-R5), 2026-10-19) |

선택 자격 공통: 무력화(락다운·스테이시스·메일스트롬, OpenBW `unit_is_disabled`)된 유닛은 명령을 받지 못한다. 모든 선택 자격 클래스와 executor 선택에 적용한다(R6). 건물을 짓고 있는 SCV(order `ConstructingBuilding`, OpenBW `unit_can_receive_order`는 Stop만 허용)와 시즈·언시즈 전환 중인 탱크(order `Sieging`·`Unsieging`, orders.dat 중단 불가 → BWAPI Unit_Busy)는 `can_stop` 외 모든 선택 자격에서 빠진다(사용자 결정 2026-10-19, 액션 감사 F15·F16). **레이블과 라이브의 의도된 차이(사용자 결정 2026-10-19, U-4)**: 전환 중 탱크 제외는 라이브 마스크·executor에만 적용한다. 엔진은 전환 중에 받은 명령을 전환 뒤로 큐에 넣어 실행하므로, 리플레이 레이블(BC 학습 마스크, 040-R19 판정, `label-check`)은 이 제외 없이 계산해 사람의 전환 중 연타 레이블을 유효로 둔다(`AgentMemory::replay_labels`). 031-R4 "마스크는 BC·RL·실전에서 동일"의 예외는 이것 하나다. Build_*의 type_mask는 `is_worker` 유닛이 아니라 `pick_worker` 후보(완성·비탑승·비무력화·건설 중 아님)가 있을 때만 연다(R6 "후보 없으면 닫음", F15). 클로킹 진행 중(secondary order) 제외는 관측에 secondary order가 없어 반영하지 않는다(F16 잔여, 기록만). `can_rally`는 완성 여부를 보지 않는다(게임과 BWAPI `canSetRallyPosition` 모두 건설 중인 생산 건물의 랠리를 받는다, 2026-10-17).

시야 밖 자원·적 건물의 타깃 자격(메인 결정 2026-10-19(게임 규칙), 010-R5 예외·030-R2·030 §2. 적 건물은 2026-10-31 사용자 결정): 스타1에서 탐색했지만 안개 속인 자원은 보이고 클릭할 수 있고, 탐색하지 않은 곳의 자원은 그렇지 않다. 그래서 시야 밖이라도 한 번 본 자원(실제 unit id를 아는 기억 엔티티)은 `any_unit`·`target_resource` 대상이고, 한 번도 못 본 자원(맵 사전계산 초기값, 합성 id)은 어떤 타깃 자격에도 들지 않는다. 시야 밖 간헐천의 `refinery_slot`은 탐색된 타일일 때만 연다. OpenBW 브리지는 이런 대상 id를 관측 유닛 외에 맵의 중립 미네랄·간헐천 전체에서도 찾아 명령을 실행한다(011-R20, 액션 감사 F2). 적 유닛은 관측 유닛에서만 찾는다(맵핵 아님). **기억 적 건물(액션 감사 F18·U-2)**: 사람은 안개 속에 남은 적 건물의 잔상을 클릭해 공격하거나 우클릭한다. 그래서 시야 밖이지만 본 적 있는 적 건물(`known_building` = 기억 엔티티 ∧ 소유자 적 ∧ 건물 ∧ 마지막 목격 때 떠 있지 않음(`LIFTED` 아님))은 `attackable_enemy`(Attack_Unit)와 `any_unit`(Smart_Unit, Move_Unit, SetRally_Unit) 대상이다. `attackable_enemy`의 무적(`INVINCIBLE`) 제외는 그대로다. 다른 타깃 자격(lockdownable, yamato_target 등)은 지금처럼 보이는 유닛만 대상이다. K는 50 그대로다. `LIFTED` 조건(2026-10-31 리뷰 반영, 메인 결정)을 두는 이유: 떠 있는 건물은 날아다니는 유닛처럼 옮겨 다니므로, 그 자리에 클릭할 잔상이 남는 것으로 보지 않는다. 정보는 마지막 목격 상태뿐이다(030-R2). 건물이 그 뒤 부서졌는지, 떠서 옮겨졌는지는 마스크에 드러나지 않는다. 브리지는 실제 유닛을 찾지 않고 마지막 위치와 건물 종류로 잔상 명령을 보낸다(011-R20 ①-2, 012-R17). 중립 유닛은 해당하지 않는다. 부서진 정제소는 같은 id의 중립 간헐천이 되므로(엔진 동작), 그 간헐천을 관측한 뒤에는 기억 적 건물이 아니다. 관측 전까지는 기억 속 정제소로 남고, 잔상 명령은 그 자리에서 정제소를 찾지 못한다. 알려진 한계(받아들임): ① 저그 건물이 안개 속에서 변태하면(해처리 → 레어 등) 기억의 종류가 옛 종류라, 잔상 명령이 그 자리에서 대상을 찾지 못하고 이동만 한다(사람의 잔상 클릭과 같다). ② 반대로 변태 중에 본 저그 건물은 BWAPI가 알려 주는 종류(변태 목표 종류, 예: 레어로 변태 중인 해처리는 레어)로 기억된다. 도착했을 때 아직 변태 중이면 그 자리에서 그 종류의 유닛을 찾지 못해 이동만 할 수 있다(2026-10-31 리뷰 반영).

스킬 선택 자격은 연구 완료, 에너지 충분, 상태 조건을 모두 반영한다.
- 예: `can_cast_lockdown` = Ghost ∧ Lockdown 연구 ∧ energy ≥ 100
- 예: `can_stim` = (Marine ∨ Firebat) ∧ Stim 연구 ∧ hp > 10

## 위치 클래스 (`loc_class_mask`, L=8)
| 클래스 | 의미 | 사용 |
|---|---|---|
| any | 맵 내 유효 타일 | Attack_Pos, Patrol, Smart_Pos, Nuke, EMP, Scan, SetRally |
| walkable | 지상 이동 가능 | Move_Pos, SpiderMine, UnloadAll_Pos |
| buildable_4x3 | 4×3 footprint 건설 가능 (좌상단 기준) | Barracks, Factory, Starport, ScienceFacility, EngineeringBay |
| buildable_3x2 | 3×2 | SupplyDepot, Academy, Armory, Bunker |
| buildable_2x2 | 2×2 | MissileTurret |
| cc_slot | 4×3 + 자원 최소 거리 규칙 | Build_CommandCenter, Land_CommandCenter_Pos |
| refinery_slot | 빈 가스 간헐천 좌상단 | Refinery |
| landable_4x3 | 착륙 가능 4×3 | Land_Pos (CC 제외) |

위치 마스크는 **현재 관측 기준**으로 계산한다. 보이지 않는 곳에 적 건물이 있어서 실제로는 무효인 경우가 있을 수 있는데, 실행 시 실패로 처리한다(031-R2).

## Acceptance Criteria
- 034-AC1: 테이블 로더 테스트를 통과한다. ID 연속성, 참조된 capability와 위치 클래스의 존재, Command 매핑 존재를 확인한다.
- 034-AC2: 모든 action_type에 대해 type_mask가 true인 합성 상태에서 sel/tgt/loc 마스크가 비어 있지 않다.
- 034-AC3: 리플레이 역변환 커버리지 ≥ 95% (031-AC3).
- 034-AC4 (2026-10-31, R3·기억 적 건물): 마스크 단위 테스트. ① 서플라이 부족 + 대기열이 있는 배럭 → Train 열림, `pick_building` 후보는 대기열이 있는 건물만(대기열 길이, id 순). ② 서플라이 부족 + 모든 후보의 대기열이 빔 → 닫힘. ③ 서플라이 충분 → 이전과 같음. ④ 시야 밖 기억 적 건물은 `attackable_enemy`·`any_unit`에 들고 다른 타깃 자격에는 들지 않음. 마지막 목격 때 `LIFTED`였던 기억 적 건물은 어느 쪽에도 들지 않음(2026-10-31 리뷰 반영). ⑤ 정제소와 같은 id의 중립 간헐천이 관측되면 그 엔티티는 기억 적 건물이 아님. ⑥ `label-check`의 서플라이 판정이 마스크와 같음.

## Risks & Open Questions
- **잔상 Attack_Unit과 이동하지 못하는 유닛**(2026-10-31 리뷰 반영, 받아들인 차이): 공격은 할 수 있지만 이동하지 못하는 유닛(시즈 모드 탱크, 미사일 터렛 등)이 선택에 있으면, 기억 적 건물 잔상 Attack_Unit(011-R20 ①-2)은 이동이 필요한 order(MoveToAttack)라 그 유닛에서 실패한다. OpenBW는 BWAPI 유닛 쪽 검사(011-R23 ②, 어택땅과 같은 검사)에서 그 유닛을 거절한다(`UnitRejected`, 선택 전원이면 `CommandFailed`). SC:R에서는 그 유닛에 아무 명령도 가지 않는다(게임이 명령을 받을 수 없는 선택 유닛을 건너뛴다. 실패 신호는 012-R14대로 아직 없다). 마스크로 막으려면 선택과 대상을 함께 보는 조건이 필요한데, 대상 마스크는 선택과 독립으로 모은다(`Unload_Unit`과 같은 구조). 그래서 마스크는 열어 두고, RL은 `last_failed`(030 §4)로 배운다. 거절이 많으면 034-R9처럼 선택 결과로 대상 마스크를 다시 막는 방식을 검토한다.
- `Unload_Unit`(2026-10-19): 관측에는 수송선과 탑승 유닛을 잇는 feature가 없고 대상 마스크(`loaded_unit`)는 독립 gather다. 수송선이 여럿이면 정책이 다른 수송선의 탑승 유닛을 고를 수 있고, 그때 executor가 `BadTarget`으로 거부한다(RL에서 거부 건수로 관찰). 많으면 대상 마스크를 선택 결과로 다시 막는 방식(034-R9와 같은 구조) 또는 관측 feature 추가를 검토한다.

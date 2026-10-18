# 034 — Action Table

- **Status**: Approved (2026-10-15, 2026-10-17 개정: Attack_Pos 선택 자격 can_move, Smart_Unit 선택 자격 can_right_click_unit, can_rally 완성 조건 제거, 무력화에 메일스트롬 포함 — AMB-041-1. 2026-10-19 개정(BC 전 리뷰 결정, 사용자 확정): R6 무력화 규칙을 모든 선택 자격·executor 선택에 적용(B-11), R7 건물 대상 명령은 건물 1개만 선택(E-4, SetRally max_select 12 → 1))
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
| 034-R2 | **MUST** action_type ID는 0부터 연속이고 0은 `NoOp`이다. ID를 변경하거나 재사용하려면 action 스키마 버전을 올려야 한다(032-R6) |
| 034-R3 | **MUST** `type_mask[a]`가 true인 조건은 아래 셋을 모두 만족하는 것이다. 아니면 false다. 해당 조건이 없는 항목은 자동으로 만족한 것으로 본다(vacuous true). |
| 034-R4 | **MUST** `sel_mask = unit_cap[:, select_cap[a]] ∧ owner==self ∧ entity_mask`, `tgt_mask = unit_cap[:, target_cap[a]] ∧ entity_mask`, `loc_mask = loc_class_mask[loc_class[a]]`. 이 gather는 GPU에서 수행한다([032](032-model-architecture.md) §5) |
| 034-R5 | **MUST** 유효한 타일이 하나도 없는 위치 클래스를 쓰는 action_type은 type_mask에서 닫는다 |
| 034-R6 | **MUST** 무력화 유닛(락다운·스테이시스·메일스트롬, OpenBW `unit_is_disabled`)은 명령을 받지 못한다(2026-10-19 사용자 결정, B-11). 이 규칙은 이동·공격·캐스터 클래스만이 아니라 **모든 선택 자격 클래스**(일꾼, 수송, 스킬, 건물(이륙·착륙·랠리·애드온·취소) 등)에 적용하고, executor가 유닛을 고르는 경우(`pick_worker`, `pick_building`)에도 무력화 유닛·건물은 후보에서 뺀다. 고를 수 있는 후보가 없으면 해당 action_type은 type_mask에서 닫는다(R3 선택 자격과 같은 처리) |
| 034-R7 | **MUST** 건물 대상 명령은 건물 1개만 선택한다(2026-10-19 사용자 결정, E-4). 스타1은 건물을 다중 선택할 수 없다(040-R3 ①). 선택 자격이 건물인 action_type(`SetRally_*`, `Lift`, `Land_*`, `BuildAddon_*`, `Cancel_*`)은 `max_select = 1`이다(이번 개정으로 `SetRally_Pos`·`SetRally_Unit`을 12에서 1로 바꿈). 생산·연구·업그레이드·스캔은 executor `pick_building`이 건물 1개를 고른다 |

### 034-R3 `type_mask[a]` 조건
1. **선행 조건**: `prereq`(자원, 가스, 서플라이, 테크, 필요 건물)를 만족한다.
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
| 이동·전투 | Move_Pos, Move_Unit(follow), Attack_Pos, Attack_Unit, Patrol_Pos, Hold, Stop, Smart_Pos, Smart_Unit | can_move (Attack_Unit은 can_attack, Smart_Unit은 can_right_click_unit) (12) | loc / unit / none | Smart = 우클릭. 게임이 의미를 결정하며, 사람 리플레이 역변환에 필요. 유닛 우클릭은 시즈모드 탱크도 받는다(적이면 제자리 공격, OpenBW `right_click_action` 3), 땅 우클릭은 받지 않는다. Attack_Pos(어택땅)는 게임이 이동 가능한 모든 유닛에 적용하므로(OpenBW `unit_order_allowed`: AttackMove = 이동 가능) can_move다. 무기 없는 유닛은 이동만 하고 메딕은 HealMove로 바뀐다. BWAPI는 무장 유닛·메딕만 어택땅을 허용하므로 OpenBW 브리지는 나머지 이동 가능 유닛에 move를 보낸다(2026-10-17, AMB-041-1) |
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
| 수송 | Load_Unit, UnloadAll, UnloadAll_Pos | can_be_loaded (12) / is_transport_loaded (12) | loadable_transport / – / walkable | |
| 건물 | Lift, Land_Pos, Land_CommandCenter_Pos, SetRally_Pos, SetRally_Unit | can_lift (1) / can_land_noncc (1) / can_land_cc (1) / can_rally (1) | – / landable_4x3 / cc_slot / any / any_unit | CC 착륙은 건설과 같은 자원 거리 규칙 적용. 건물은 1개만 선택(R7) |
| 취소 | Cancel_Construction, Cancel_Train, Cancel_Research, Cancel_Upgrade, Cancel_Addon | can_cancel_* (1) | – | |

## capability 클래스 (`unit_cap`, K≈48)
| 분류 | 클래스 |
|---|---|
| 선택 자격 | can_move, can_right_click_unit(can_move ∨ 시즈모드 탱크), can_attack, is_worker, can_be_loaded, is_transport_loaded, can_stim, can_siege, can_unsiege, can_cloak, can_decloak, can_lay_mine, can_cast_lockdown, can_cast_nuke, can_heal, can_cast_restoration, can_cast_flare, can_cast_matrix, can_cast_emp, can_cast_irradiate, can_cast_yamato, can_lift, can_land_noncc, can_land_cc, can_rally, addon_capable_{cc,factory,starport,scifac}(완성 ∧ 착지 상태 ∧ 애드온 미보유 ∧ 오른쪽 애드온 자리 2×2 비어 있음 ∧ 생산·연구 중 아님), can_cancel_{construction,train,research,upgrade,addon} |
| 타깃 자격 | attackable_enemy, any_unit, own_unit, target_resource(미네랄·아군 정제소), repairable(아군 기계·건물, 피해), lockdownable(적 기계, 미락다운), healable(아군 생체 지상, 피해), restorable(아군, 상태이상), flarable(적, 비건물), irradiatable(비건물), matrixable(아군), yamato_target(적), loadable_transport(아군 드랍십·벙커, 여유 공간) |

선택 자격 공통: 무력화(락다운·스테이시스·메일스트롬, OpenBW `unit_is_disabled`)된 유닛은 명령을 받지 못한다. 모든 선택 자격 클래스와 executor 선택에 적용한다(R6). `can_rally`는 완성 여부를 보지 않는다(게임과 BWAPI `canSetRallyPosition` 모두 건설 중인 생산 건물의 랠리를 받는다, 2026-10-17).

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

# M6 액션 전수 감사: 034 액션 테이블과 액션 관련 코드 (2026-10-19)

기준 커밋 master 72cf539. 읽기 전용 감사(코드·spec 수정 없음). 전체 재build가 이 PC에서 도는 동안 진행했으므로 OpenBW 실게임은 돌리지 않았고, 게임 규칙 판정은 OpenBW·BWAPI 4.2 소스(`third_party/openbw`, `third_party/bwapi`)와 T7 실게임 기록(`reports/m6_unverified_ac.md`, `reports/raw/t7_031_ac1/run6.log`)으로 했다.

## 1. 배경·방법

### 배경
- 034 액션 테이블(100개)과 네 경로 — ① 리플레이 레이블러(`uranium-replay` `label.rs`), ② BC 레이블 검사(`host_label_check`), ③ 라이브 마스크(`masks.rs` + 모델 헤드 마스크 `heads.rs`), ④ executor(`executor.rs`) → proto `Command` → 브리지 shim(`shim.cpp`)의 BWAPI 호출 — 가 모두 같은 게임 규칙을 말하는지 확인한다.
- T7(031-AC1 실게임)에서 남은 거절: Move_Unit·Smart_Unit 오류 6(게임 초반 SCV가 다른 SCV를 따라갈 때), Heal_Unit 오류 6, 애드온 자리에 유닛(19)의 원인을 찾는다.

### 방법
1. spec 034·031·030·032 §2.5/§5·040(R3·R4·R11·R18·R19)·041-R11·013·010·011을 읽고 테이블(`crates/uranium-env/tables/actions.toml`, `crates/uranium-replay/tables/replay_orders.toml`)과 대조했다.
2. action_type마다 선택 자격·대상 자격·위치 클래스·executor·shim 호출을 OpenBW 엔진 규칙(`actions.h` `action_order`/`action_default_order`/`action_train`, `bwgame.h` `unit_can_receive_order`/`unit_order_allowed`/`unit_can_use_tech`/`build_queue_push`/`unit_can_load_target`/`get_default_order`)과 BWAPI 4.2 `can*`(`Shared/Templates.h`)에 대조했다. 게임 데이터는 `<URANIUM 추출물>/openbw/arr/units.dat`(right_click_action), `orders.dat`(can_be_interrupted·can_be_queued)을 직접 읽었다.
3. 재build가 쓰는 중인 records(`<URANIUM_SHARED_DIR>/datasets/replays/records/*.jsonl`, 읽기만)로 레이블 커버리지와 미분류 opcode·order를 집계했다(§7).
4. 완성된 shard 2개(stardata 1, scr_ladder 1)를 로컬로 복사해 현재 마스크로 `label-check`를 돌렸다(§8, Idle 우선순위, 재build 비간섭).

### 표기
- ✅ 네 경로가 게임 규칙과 맞음. ⚠️ 게임과 차이가 있으나 영향이 작거나 사용자 결정으로 의도된 것. ❌ 고쳐야 할 불일치.
- "엔진"은 OpenBW 게임 로직, "BWAPI"는 OpenBW 브리지가 거치는 BWAPI 4.2 `can*` 검사다. 둘이 다를 때는 둘 다 적었다. 라이브 경로는 BWAPI가 거절하면 실패한다.

## 2. 액션별 표

열: 선택 자격(select_cap, max) / 대상 / 레이블 출처(리플레이 opcode·order) / 판정 / 근거·비고. 모든 선택 클래스는 034-R6(무력화 제외)을 `commandable`(`masks.rs:110`) 또는 `own_any`(`masks.rs:158`)·`can_rally`(`masks.rs:152`)로 적용한다(✅, `disabled_units_take_no_command_034_r6`).

| id | action | 선택 (max) / 대상 | 레이블 출처 | 판정 | 근거·비고 |
|---|---|---|---|---|---|
| 0 | NoOp | – | 간격 채움(031-R11, `label.rs:417`) | ⚠️ | 미분류 명령(Unknown)의 시간도 NoOp으로 채워진다(§4 F7) |
| 1 | Move_Pos | can_move (12) / walkable | 0x15 Move(6) 위치 | ✅ | 엔진 `unit_order_allowed(Move)`=`unit_can_move`, BWAPI `canMove`(`Templates.h:1213`). 건설 중 SCV(ConstructingBuilding)는 엔진이 Stop 외 거절(`bwgame.h:17761`)인데 can_move에 포함(F15, Low) |
| 2 | Move_Unit | can_move (12) / any_unit | 0x15 Move(6)·Follow(49) 유닛 | ❌ | 대상 any_unit이 **생산 중(미완성) 아군 유닛·탑승 유닛**을 포함(F1). shim은 `follow`(`shim.cpp:141`) → BWAPI `canTargetUnit` 미완성 거절(`Templates.h:469`) = T7 오류 6. 시야 밖 자원 대상은 shim이 못 찾음(F2) |
| 3 | Attack_Pos | can_move (12) / any | 0x15 Attack*(8,9,10,12,14)·HealMove(177) 위치 | ✅ | 엔진 AttackMove=이동 가능 전부(`bwgame.h:17862`), shim 비무장 → move(`shim.cpp:146`), T7 확인 |
| 4 | Attack_Unit | can_attack (12) / attackable_enemy | 0x15 Attack*(8,9,10,12,14) 유닛 | ⚠️ | can_attack에 벙커 포함(`masks.rs:121`): units.dat 벙커 right_click_action=0·attack_unit=Nothing → 엔진 `action_order`가 건너뜀(`actions.h:484` `default_action(u) == 0`), BWAPI Unable_To_Hit(`Templates.h:622`) (F11). 시즈 탱크·터렛은 사거리 밖이면 BWAPI Out_Of_Range(`Templates.h:689`), 공중/지상 무기 불일치는 Unable_To_Hit — 마스크 미반영(F11). 시야 밖 기억 적 건물은 대상 불가(attackable_enemy=visible)인데 엔진은 허용(F18). 아군·미네랄 공격 금지는 사용자 결정(2026-10-17) |
| 5 | Patrol_Pos | can_move (12) / any | 0x15 Patrol(152) | ✅ | BWAPI canPatrol=canMove |
| 6 | Hold | can_move (12) / – | 0x2b, 0x15 HoldPosition(107) | ✅ | BWAPI canHoldPosition=`getType().canMove()`(`Templates.h:1397`), 시즈 탱크 제외 맞음 |
| 7 | Stop | can_move (12) / – | 0x1a, 0x15 Stop(1) | ❌ | 엔진 `unit_order_allowed(Stop)`=항상 true(`bwgame.h:17773`), BWAPI canStop은 시즈 탱크·터렛·떠 있는 건물 허용(`Templates.h:1429`). can_move는 시즈 탱크·터렛을 빼므로 사람의 시즈 탱크 Stop 레이블이 `select_cap`으로 무효, 라이브는 시즈 탱크를 멈출 수 없음(F5) |
| 8 | Smart_Pos | can_move (12) / any | 0x14 위치, 선택이 착지 건물이 아닐 때 | ✅ | 엔진 `get_default_order` 1·2·4·5=Move, 3(시즈 탱크)=idle(제외 맞음), 0(벙커·디팟)=없음 |
| 9 | Smart_Unit | can_right_click_unit (12) / any_unit | 0x14 유닛, 선택이 착지 건물이 아닐 때 | ❌ | F1(미완성·탑승 대상), F2(시야 밖 자원). 시즈 탱크가 비적 대상: 엔진 idle, BWAPI Incompatible_State(`Templates.h:1975-1978`) — 마스크 허용(F14, Low). 메딕·드랍십·베슬이 적 대상: 엔진 HealMove/Follow, BWAPI 거절(무기 없음) — 라이브 유닛별 실패(F17, Low). 터렛(right_click_action 3) 적 우클릭은 레이블러가 SetRally_Unit으로 바꿔 무효(F8) |
| 10 | Gather_Unit | is_worker (12) / target_resource | 0x15 Harvest*(79,80,81,85) | ⚠️ | 시야 밖 한 번 본 미네랄은 마스크 허용(034)인데 shim `find_unit`이 못 찾아 항상 실패(F2). 사람 레이블은 거의 Smart_Unit |
| 11 | ReturnCargo | is_worker (12) / – | 0x1e, 0x15 ReturnGas/Minerals(84,90) | ✅ | |
| 12 | Repair_Unit | is_worker (12) / repairable | 0x15 Repair(34)·MoveToRepair(35) | ✅ | repairable=아군·기계·완성·피해(`masks.rs:177`), 엔진 Repair=SCV만(`bwgame.h:17948`) |
| 13 | Build_CommandCenter | pick_worker / cc_slot | 0x0C PlaceBuilding(30) type 106 | ✅ | 자원 거리 3타일(`masks.rs:248`). 유닛이 서 있는 타일은 실행 시 실패(034 비고) |
| 14 | Build_SupplyDepot | pick_worker / buildable_3x2 | 0x0C type 109 | ✅ | |
| 15 | Build_Refinery | pick_worker / refinery_slot | 0x0C type 110 | ✅ | 탐색한 간헐천만(`masks.rs:289`), 시야 밖 한 번 본 간헐천 포함(E-3) |
| 16 | Build_Barracks | pick_worker / buildable_4x3 | 0x0C type 111 | ✅ | |
| 17 | Build_EngineeringBay | pick_worker / buildable_4x3 | 0x0C type 122 | ✅ | |
| 18 | Build_MissileTurret | pick_worker / buildable_2x2 | 0x0C type 124 | ✅ | |
| 19 | Build_Academy | pick_worker / buildable_3x2 | 0x0C type 112 | ✅ | |
| 20 | Build_Bunker | pick_worker / buildable_3x2 | 0x0C type 125 | ✅ | |
| 21 | Build_Factory | pick_worker / buildable_4x3 | 0x0C type 113 | ✅ | |
| 22 | Build_Starport | pick_worker / buildable_4x3 | 0x0C type 114 | ✅ | |
| 23 | Build_Armory | pick_worker / buildable_3x2 | 0x0C type 123 | ✅ | |
| 24 | Build_ScienceFacility | pick_worker / buildable_4x3 | 0x0C type 116 | ✅ | Build_* 공통: type_mask는 `is_worker` 존재(`masks.rs:388`)인데 `pick_worker`는 건설 중 SCV를 빼므로(`executor.rs:172`) 후보가 없으면 거부(NoExecutorUnit) — 034-R6 "후보 없으면 type_mask 닫음"과 어긋남(F15, Low) |
| 25 | BuildAddon_ComsatStation | addon_capable_cc (1) / – | 0x0C PlaceAddon(36)/BuildAddon(37) type 107 | ✅ | 자리 검사는 건물·자원만(`masks.rs:193`), 유닛은 실행 실패(T7 오류 19, 034 비고와 일치) |
| 26 | BuildAddon_NuclearSilo | addon_capable_cc (1) / – | 0x0C type 108 | ✅ | |
| 27 | BuildAddon_MachineShop | addon_capable_factory (1) / – | 0x0C type 120 | ✅ | |
| 28 | BuildAddon_ControlTower | addon_capable_starport (1) / – | 0x0C type 115 | ✅ | |
| 29 | BuildAddon_PhysicsLab | addon_capable_scifac (1) / – | 0x0C type 118 | ✅ | |
| 30 | BuildAddon_CovertOps | addon_capable_scifac (1) / – | 0x0C type 117 | ✅ | |
| 31 | Train_SCV | pick_building(CC) / – | 0x1f type 7 | ⚠️ | Train 공통: 서플라이 선행 조건(`masks.rs:346`)은 BWAPI `canMake`(`Templates.h:371`)와 같지만 엔진은 **큐가 비어 있을 때만** 서플라이를 본다(`bwgame.h:2297`). 사람의 서플라이 막힘 중 추가 큐 레이블이 `type_mask`로 무효(F4) |
| 32 | Train_Marine | pick_building(Barracks) / – | 0x1f type 0 | ⚠️ | F4 |
| 33 | Train_Firebat | pick_building(Barracks) / – | 0x1f type 32 | ⚠️ | F4. 요구 건물(아카데미)은 `required_ok` |
| 34 | Train_Medic | pick_building(Barracks) / – | 0x1f type 34 | ⚠️ | F4 |
| 35 | Train_Ghost | pick_building(Barracks) / – | 0x1f type 1 | ⚠️ | F4 |
| 36 | Train_Vulture | pick_building(Factory) / – | 0x1f type 2 | ⚠️ | F4. 034-R8(애드온 건설 중 제외) 반영 확인(`world.rs:359`) |
| 37 | Train_SiegeTank | pick_building(Factory+MachineShop) / – | 0x1f type 5 | ⚠️ | F4. 애드온 요구는 `producer_for`(`masks.rs:302`) |
| 38 | Train_Goliath | pick_building(Factory) / – | 0x1f type 3 | ⚠️ | F4 |
| 39 | Train_Wraith | pick_building(Starport) / – | 0x1f type 8 | ⚠️ | F4 |
| 40 | Train_Dropship | pick_building(Starport+ControlTower) / – | 0x1f type 11 | ⚠️ | F4 |
| 41 | Train_ScienceVessel | pick_building(Starport+ControlTower) / – | 0x1f type 9 | ⚠️ | F4 |
| 42 | Train_Battlecruiser | pick_building(Starport+ControlTower) / – | 0x1f type 12 | ⚠️ | F4 |
| 43 | Train_Valkyrie | pick_building(Starport+ControlTower) / – | 0x1f type 58 | ⚠️ | F4 |
| 44 | Train_NuclearMissile | pick_building(NuclearSilo) / – | 0x1f type 14 | ✅ | 서플라이 8이지만 `has_available_supply_for`는 큐 빈 경우 검사 → 사일로 큐는 항상 비므로 F4 해당 없음. `nuke_armed`·큐 빈 사일로만(`masks.rs:324`) |
| 45 | Research_Stim | pick_building(Academy) / – | 0x30 tech 0 | ✅ | 연구 중·완료·자원 검사(`masks.rs:352`) |
| 46 | Research_Restoration | pick_building(Academy) / – | 0x30 tech 24 | ✅ | |
| 47 | Research_OpticalFlare | pick_building(Academy) / – | 0x30 tech 30 | ✅ | |
| 48 | Research_SpiderMines | pick_building(MachineShop) / – | 0x30 tech 3 | ✅ | |
| 49 | Research_TankSiege | pick_building(MachineShop) / – | 0x30 tech 5 | ✅ | |
| 50 | Research_EMP | pick_building(ScienceFacility) / – | 0x30 tech 2 | ✅ | |
| 51 | Research_Irradiate | pick_building(ScienceFacility) / – | 0x30 tech 7 | ✅ | |
| 52 | Research_Yamato | pick_building(PhysicsLab) / – | 0x30 tech 8 | ✅ | |
| 53 | Research_CloakingField | pick_building(ControlTower) / – | 0x30 tech 9 | ✅ | |
| 54 | Research_Lockdown | pick_building(CovertOps) / – | 0x30 tech 1 | ✅ | |
| 55 | Research_PersonnelCloaking | pick_building(CovertOps) / – | 0x30 tech 10 | ✅ | |
| 56 | Upgrade_InfantryWeapons | pick_building(EngineeringBay) / – | 0x32 up 7 | ✅ | 레벨별 비용·요구 건물(`masks.rs:354`) |
| 57 | Upgrade_InfantryArmor | pick_building(EngineeringBay) / – | 0x32 up 0 | ✅ | |
| 58 | Upgrade_VehicleWeapons | pick_building(Armory) / – | 0x32 up 8 | ✅ | |
| 59 | Upgrade_VehiclePlating | pick_building(Armory) / – | 0x32 up 1 | ✅ | |
| 60 | Upgrade_ShipWeapons | pick_building(Armory) / – | 0x32 up 9 | ✅ | |
| 61 | Upgrade_ShipPlating | pick_building(Armory) / – | 0x32 up 2 | ✅ | |
| 62 | Upgrade_U238Shells | pick_building(Academy) / – | 0x32 up 16 | ✅ | |
| 63 | Upgrade_IonThrusters | pick_building(MachineShop) / – | 0x32 up 17 | ✅ | |
| 64 | Upgrade_CharonBoosters | pick_building(MachineShop) / – | 0x32 up 54 | ✅ | |
| 65 | Upgrade_OcularImplants | pick_building(CovertOps) / – | 0x32 up 20 | ✅ | |
| 66 | Upgrade_MoebiusReactor | pick_building(CovertOps) / – | 0x32 up 21 | ✅ | |
| 67 | Upgrade_ApolloReactor | pick_building(ControlTower) / – | 0x32 up 22 | ✅ | |
| 68 | Upgrade_ColossusReactor | pick_building(PhysicsLab) / – | 0x32 up 23 | ✅ | |
| 69 | Upgrade_CaduceusReactor | pick_building(Academy) / – | 0x32 up 51 | ✅ | |
| 70 | Upgrade_TitanReactor | pick_building(ScienceFacility) / – | 0x32 up 19 | ✅ | |
| 71 | Stim | can_stim (12) / – | 0x36 | ✅ | hp>10, 연구(`masks.rs:126`) = 엔진 `unit_can_use_tech` |
| 72 | SiegeMode | can_siege (12) / – | 0x26 | ⚠️ | 시즈/언시즈 진행 중(order Sieging·Unsieging)을 빼지 않아 BWAPI Unit_Busy(`Templates.h:2191`, T7 TankMode 오류 3) (F16, Low) |
| 73 | TankMode | can_unsiege (12) / – | 0x25 | ⚠️ | F16 |
| 74 | Cloak | can_cloak (12) / – | 0x21 | ⚠️ | 클로킹 진행 중(secondary order Cloak) 미반영(F16, Low) |
| 75 | Decloak | can_decloak (12) / – | 0x22 | ✅ | |
| 76 | SpiderMine_Pos | can_lay_mine (1) / walkable | 0x15 VultureMine(20)·PlaceMine(132) | ⚠️ | 엔진은 선택된 벌처 전원이 시전(`actions.h` `action_order` 루프, 1기 제한 없음) — 031-R8 결정과 일치. 레이블은 선택의 첫 유닛 1기(`label.rs:232`): 그 유닛에 마인이 없으면 다른 유닛이 있어도 무효(F12, Low) |
| 77 | Lockdown_Unit | can_cast_lockdown (1) / lockdownable | 0x15 CastLockdown(115) | ✅ | F12 해당. 대상: 적·기계·비건물(`masks.rs:178`) = BWAPI(`Templates.h:2338,2428`) |
| 78 | Nuke_Pos | can_cast_nuke (1) / any | 0x15 NukePaint(126)·CastNuclearStrike(128) | ✅ | 사일로 nuke_armed(`world.rs:312`) = BWAPI `completedUnitCount(Nuclear_Missile)`(`Templates.h:2213`) |
| 79 | Restoration_Unit | can_cast_restoration (1) / restorable | 0x15 CastRestoration(180) | ⚠️ | restorable이 미완성·탑승 유닛을 빼지 않음(F1) |
| 80 | OpticalFlare_Unit | can_cast_flare (1) / flarable | 0x15 CastOpticalFlare(185) | ✅ | 적만(설계). 엔진은 아군도 허용 |
| 81 | DefensiveMatrix_Unit | can_cast_matrix (1) / matrixable | 0x15 CastDefensiveMatrix(141) | ⚠️ | matrixable=own·visible·비건물 — 미완성·탑승 포함(F1). 아군만(설계) |
| 82 | EMP_Pos | can_cast_emp (1) / any | 0x15 CastEMPShockwave(122) | ✅ | |
| 83 | Irradiate_Unit | can_cast_irradiate (1) / irradiatable | 0x15 CastIrradiate(143) | ⚠️ | irradiatable=targetable·비건물 — 미완성·탑승 아군 포함(F1) |
| 84 | Yamato_Unit | can_cast_yamato (1) / yamato_target | 0x15 FireYamatoGun(113) | ⚠️ | `MoveToFireYamatoGun`(114) 미매핑 → Unknown(§7, 35건)(F22, Low) |
| 85 | Heal_Unit | can_heal (12) / healable | 0x15 MedicHeal(176)·HealMove(177) 유닛 | ❌ | healable=아군·생체·지상·피해인데 **생산 중 마린(hp 상승 중=피해)·탑승 유닛**을 포함(F1) → BWAPI `canTargetUnit` 미완성 거절 = T7 Heal 오류 6 |
| 86 | ScannerSweep_Pos | pick_building(Comsat, energy 최대) / any | 0x15 CastScannerSweep(139) | ✅ | |
| 87 | Load_Unit | can_be_loaded (12) / loadable_transport | 0x15 EnterTransport(92)·PickupTransport(94) | ⚠️ | 대상 벙커는 선택 유닛 종류를 보지 않음: 엔진 `unit_can_load_target`은 건물이면 테란·space≤1만(`bwgame.h:1795`), BWAPI는 테란 생체만(`Templates.h:1691`) → 벌처·탱크·골리앗이 벙커 대상이면 거절(F10, Low). PickupTransport(94)는 선택이 수송선이라 `select_cap` 무효(F21, Low). 사람은 거의 우클릭(Smart_Unit) |
| 88 | UnloadAll | is_transport_loaded (12) / – | 0x28 | ✅ | |
| 89 | UnloadAll_Pos | is_transport_loaded (12) / walkable | 0x15 MoveUnload(112) | ⚠️ | is_transport_loaded에 벙커 포함(`masks.rs:124`): BWAPI `canUnloadAllPosition` 벙커 Incompatible_UnitType(`Templates.h:1801`) (F6, Low) |
| 90 | Lift | can_lift (1) / – | 0x2f | ✅ | busy=큐·연구·애드온 건설(034-R8) ≈ BWAPI `isIdle` |
| 91 | Land_Pos | can_land_noncc (1) / landable_4x3 | 0x0C BuildingLand(71) 비CC | ✅ | |
| 92 | Land_CommandCenter_Pos | can_land_cc (1) / cc_slot | 0x0C BuildingLand(71) CC | ✅ | |
| 93 | SetRally_Pos | can_rally (1) / any | 0x14 위치, 선택이 착지 건물 / 0x15 RallyPointTile(40) | ⚠️ | can_rally=ProducesUnits 건물(CC·배럭·팩토리·스타포트)·착지·미완성 허용 = 엔진 `unit_is_factory`. 디팟·벙커·터렛 우클릭도 레이블러가 SetRally로 바꿔 무효 처리 — 디팟·벙커는 엔진도 무효(맞음), 터렛 적 우클릭은 엔진이 공격(F8) |
| 94 | SetRally_Unit | can_rally (1) / any_unit, self_target | 0x14 유닛, 선택이 착지 건물 / 0x15 RallyPointUnit(39) | ✅ | 자기 대상 예외 = 엔진 `unit_order_can_target_self`(`bwgame.h:1480`) |
| 95 | Cancel_Construction | can_cancel_construction (1) / – | 0x18 | ✅ | |
| 96 | Cancel_Train | can_cancel_train (1) / – | 0x20 | ⚠️ | 0x20 페이로드는 **슬롯**(254=마지막, `actions.h:1097`, `actions.h:575`)인데 `cmd.rs:185`는 유닛 태그로 보고 `map_tag`를 적용 — 값 범위(0..4, 254) 때문에 결과는 바뀌지 않음. 레이블·executor는 슬롯을 잃고 "마지막 취소"(`cancelTrain()`)로 통일(F13, Low) |
| 97 | Cancel_Research | can_cancel_research (1) / – | 0x31 | ✅ | |
| 98 | Cancel_Upgrade | can_cancel_upgrade (1) / – | 0x33 | ✅ | |
| 99 | Cancel_Addon | can_cancel_addon (1) / – | 0x34 | ✅ | |

요약: ✅ 76, ⚠️ 20, ❌ 4 (Move_Unit, Smart_Unit, Stop, Heal_Unit). ❌는 모두 두 원인(F1 대상 자격, F5 Stop 선택 자격)에서 나온다.

## 3. 경로 간 불일치

같은 액션을 네 경로가 다르게 보는 곳만 적는다. 번호는 §4 발견 사항이다.

| 액션 | 레이블러 | BC 레이블 검사 | 라이브 마스크·모델 | executor·shim | 엔진 / BWAPI | 발견 |
|---|---|---|---|---|---|---|
| Move_Unit·Smart_Unit·Heal_Unit·Matrix·Irradiate·Restoration | 사람은 미완성·탑승 유닛을 클릭할 수 없어 레이블에 없음 | 통과(대상 자격 있음) | 미완성·탑승 아군을 대상으로 **샘플 가능** | 명령 전송 | 엔진·BWAPI 거절(오류 6) | F1 |
| Move_Unit·Smart_Unit·Gather_Unit → 시야 밖 한 번 본 자원 | 사람 레이블에 흔함 | 유효(`known_resource`) | 유효 | shim `find_unit`이 **보이는 유닛만** 찾음 → Gather·Repair·Load는 `CommandFailed`, Move·Smart는 (-1,-1) 이동으로 떨어져 **조용히 무시** | 엔진·BWAPI(CMI 켜짐)는 허용 | F2 |
| 12기 그룹 명령 전부 | 사람 명령은 선택 12기 1회 | – | – | shim이 유닛마다 BWAPI 호출(`shim.cpp:176`) → 선택·명령 12회, 대형 이동(`calc_group_move`) 없음, 한 유닛 거절이 명령 전체 실패 | 엔진은 1회 `action_order`로 12기 처리 | F3 |
| Train_* (서플라이 막힘, 큐 비어 있지 않음) | 레이블 생성 | `type_mask` 무효 | 닫힘 | – | 엔진 허용(`bwgame.h:2297`), BWAPI 거절 | F4 |
| Stop (시즈 탱크·터렛) | 레이블 생성 | 시즈 탱크만이면 `select_cap` 무효 | 선택 불가 | – | 엔진·BWAPI 허용 | F5 |
| 0x29 단일 Unload, Guard(2), PickupBunker(95), MoveToFireYamatoGun(114), TowerAttack(19) | **Unknown → 시간이 NoOp 레이블로 채워짐** | NoOp 유효 | – | – | 게임 상태를 바꾸는 명령 | F7 |
| 터렛·벙커 적 우클릭 | SetRally_Unit | `select_cap` 무효 | – | – | 터렛: 공격(right_click_action 3) / 벙커: 없음(0) | F8 |
| Attack_Unit (시야 밖 기억 적 건물) | 레이블 생성 | `target_unit_cap` 무효 | 대상 불가 | – | 엔진 허용(유닛 태그로 공격) | F18 |
| 타깃 스킬 max_select=1 (SpiderMine·Lockdown·…) | 선택의 **첫** 유닛 1기 | 그 유닛이 자격 없으면 무효 | 1기 선택 | – | 엔진은 선택 전원 시전 | F12 |
| Cancel_Train 슬롯 | 슬롯 버림 | – | – | 마지막 취소 | 엔진 슬롯별 | F13 |

일치 확인(불일치 없음): 034-R7 건물 단일 선택(레이블러 `admit`, `host_label_check` `select_building`, 모델 `heads.rs:355-372`, executor `executor.rs:75`), 034-R9 자기 대상(`label.rs:82`, `heads.rs:247-251`, `executor.rs:84`, 엔진 `unit_order_can_target_self`), 034-R8, 041-R11 retain, queued(`uses_queued` 레이블·모델·executor 세 곳 같은 규칙; UseTech·Siege·Lift는 BWAPI에 큐 인자가 없음 = 테이블 `uses_queued=false`), delay bucket(`bucket_index`/`floor_bucket`/`DELAY_BUCKETS` 한 정의), flip 증강(`transform_target_loc`가 footprint 사각형 기준, `transform`이 위치 마스크를 같은 기준으로 옮김, 검사 뒤 변환 — `data.rs:905`), 시야 밖 자원 결정론(`unseen_resources` BTreeMap, 합성 id는 어떤 대상 클래스에도 없음 `masks.rs:172`), executor 결정론(`(queue_len, id)`·`(dist, id)` 정렬), 맵핵 없음(대상 자격에 보이지 않는 적은 없음, 기억 적 건물은 `attackable_enemy`·`any_unit`에서 제외).

## 4. 발견 사항

심각도: High = 마스크·실행 경로가 게임 규칙과 어긋나 반복 실패하거나 학습 데이터 의미가 틀어짐. Medium = 특정 상황에서 레이블 손실·행동 차이. Low = 드물거나 영향이 작음.

### F1. [High] 대상 자격이 생산 중(미완성)·탑승 아군 유닛을 포함한다 — T7 Move_Unit·Smart_Unit·Heal_Unit 오류 6의 원인
- 근거
  - 엔진은 생산을 시작할 때 유닛을 바로 만든다(`third_party/openbw/bwgame.h:8208` `secondary_order_Train` → `create_unit`, 미완성·숨김). OpenBW BWAPI는 숨은 유닛도 열거한다(`third_party/bwapi/bwapi/OpenBWData/BW/BWData.cpp:1320` `hidden_units`). shim은 아군이면 모두 관측에 넣는다(`bridge/openbw/shim/shim.cpp:119` `visible_to_self`). 그래서 관측에 `COMPLETED=0`인 아군 SCV·마린 엔티티가 생긴다(생산 중 내내).
  - `masks.rs:166` `targetable = visible && !(cloaked && !detected && !own)` — 완성·탑승 여부를 보지 않는다. `any_unit`(174), `own_unit`(175), `healable`(179, 생산 중 유닛은 hp가 올라가는 중이라 `hurt`), `restorable`(182), `irradiatable`(184), `matrixable`(185)이 모두 이 유닛을 대상으로 허용한다.
  - BWAPI `canTargetUnit`(`Templates.h:469`)은 미완성 비건물 대상을 Incompatible_State(6)로 거절한다. Move_Unit(`follow`)·Smart_Unit(`rightClick`)·Heal(`useTech`)·Matrix·Irradiate·Restoration 모두 이 검사를 거친다.
  - T7 run6: Move_Unit·Smart_Unit 오류 6이 프레임 17~353(첫 SCV 생산 중)에 48프레임마다, Heal 오류 6이 7369~7657(마린 생산 중)에 나왔고 그 뒤에는 성공했다(`reports/raw/t7_031_ac1/run6.log:121-136`). 시나리오의 대상 선택(`openbw_actions.rs:758` "아군 비건물, 마린 우선, 그다음 인덱스 순")이 미완성 유닛을 고를 수 있다. 탑승 유닛도 같은 경로다(사람은 클릭할 수 없고 엔진 `Follow`는 숨은 유닛을 따라가지 못함).
- 영향: BC 레이블 없음(사람은 이런 유닛을 클릭할 수 없음). 평가 대전·RL에서 생산 중 유닛이 있는 동안(거의 항상) Move_Unit·Smart_Unit·Heal·Matrix·Irradiate·Restoration이 실패 가능한 대상을 포함하고, 그때마다 `CommandFailed`·`last_failed`=1(RL)과 낭비된 스텝이 생긴다. 관측 쪽 부수 효과: 생산 중 유닛이 건물 위치의 아군 밀도 채널(`extract.rs:270` 이후 `counts`)에 더해진다(이번 감사 범위 밖, 기록만).
- 수정 제안: `targetable`에 `(done || t.is_building()) && !loaded` 를 더한다(정제소 안 SCV·건설 중 SCV(order ConstructingBuilding)도 사람이 클릭할 수 없으므로 함께 빼는 것을 권장). 레이블 검사·모델·executor가 같은 `unit_cap`을 쓰므로 한 곳 수정으로 네 경로가 같이 바뀐다. 034 "타깃 자격" 표에 "완성(건물 제외)·비탑승" 문구를 추가해야 한다(spec 수정 → 사용자 확인 → 코드). 테스트: 합성 상태에 미완성 아군 SCV·탑승 마린을 두고 어떤 대상 클래스에도 들지 않음을 확인.

### F2. [High] shim이 명령 대상을 **지금 보이는 유닛**에서만 찾아, 034·010-R5가 허용한 "시야 밖 한 번 본 자원" 대상이 라이브에서 항상 실패한다
- 근거: `shim.cpp:25-35` `g_units`는 관측마다 `visible_to_self` 유닛으로만 다시 만든다. executor는 기억 자원의 실제 id를 `Target::Unit(id)`로 보낸다(`masks.rs:172` `known_resource`, `executor.rs:84`). `exec_unit_order`의 `t = find_unit(...)`이 null → Gather/Repair/Load는 `t && ...`로 false(`CommandFailed`, 오류 코드는 직전 호출의 잔값), Move/RightClick은 **`u->move(pos(c))`로 떨어지는데 `pos`는 `UR_NONE`(-1,-1)** 이다. BWAPI는 Move 위치를 검사하지 않아(`Templates.h:2833`) 명령을 받아들이고 엔진이 `is_in_map_bounds` 실패로 버리므로(`actions.h:369`) **실패 이벤트조차 없이 아무 일도 일어나지 않는다**. 엔진·BWAPI(CompleteMapInformation 켜짐, 011-R5)는 그 자원 유닛을 안다.
- 영향: BC 데이터에는 안개 속 미네랄·간헐천 우클릭(일꾼 이동, 확장 자원 지정)이 흔하고 레이블은 유효하다. 평가 대전·RL에서는 같은 행동이 100% 실패한다(F1과 달리 사람 데이터가 가르치는 행동이 막힘). 테스트 공백: 031-AC1 시나리오는 보이는 자원만 대상으로 한다.
- 수정 제안: shim이 대상 id를 풀 때 관측 집합 외에 **중립 자원(ResourceContainer) 유닛 전부**를 담은 보조 맵(`g_resources`, 매 관측 갱신)을 함께 본다. 에이전트가 보내는 id는 이미 본 유닛의 id뿐이므로 시야 정보가 새지 않는다(010-R5). 적 유닛은 계속 관측 집합에서만 찾는다. 추가로 `UR_ORD_MOVE`/`RIGHT_CLICK`에서 `t`가 null이고 `c.x == UR_NONE`이면 바로 실패로 처리해 (-1,-1) 이동 시도를 없앤다. 테스트: 031-AC1 시나리오에 "한 번 본 뒤 시야 밖이 된 미네랄에 Smart_Unit/Gather_Unit"을 추가.

### F3. [Medium] shim이 12기 명령을 유닛별 BWAPI 호출로 쪼갠다
- 근거: `shim.cpp:173-181` `exec`는 `Command::Unit`의 유닛마다 `u->move(...)` 등을 부른다. BWAPI `UnitInterface::*`는 유닛 1기 선택+명령 1회이고, `Game::issueCommand(Unitset)`(`third_party/bwapi/bwapi/BWAPI/Source/BWAPI/GameImpl.cpp:673`)만 12기를 한 번의 Select+명령으로 보낸다. 엔진은 그룹 위치 명령에 대형 보정(`actions.h` `calc_group_move`/`get_group_move_pos`)을 적용하므로 유닛별 명령은 사람 리플레이(BC 데이터)와 이동 결과가 다르다. 또한 한 유닛이 거절되면(예: 그룹 안의 파이어뱃에 공중 대상, 메딕 우클릭 적) `ok`가 false가 되어 명령 전체가 `CommandFailed`로 기록되고(`shim.cpp:178`, 031-R10) 나머지 유닛은 이미 명령을 받은 상태다.
- 영향: 평가 대전·RL의 이동·공격 행동이 BC 데이터와 어긋남(대형 없음), `last_failed`(RL) 잡음, 명령 수 12배.
- 수정 제안: shim에서 `Unitset`에 모아 `set.move/attack/rightClick/useTech/load/unloadAll/…`(또는 `Broodwar->issueCommand(set, cmd)`)을 쓴다. 유닛별 성공 여부는 `canIssueCommandGrouped`로 미리 세어 실패 유닛 수를 이벤트에 넣는 것을 권장(010 `CommandFailed`에 필드 추가는 프로토콜 변경이므로 사용자 확인).

### F4. [Low, 측정 뒤 하향] Train 서플라이 선행 조건이 엔진 규칙과 다르다(큐가 비어 있지 않으면 엔진은 서플라이를 보지 않음)
- 근거: 엔진 `build_queue_push`(`bwgame.h:2297`)는 `build_queue.empty()`일 때만 `has_available_supply_for`를 검사하고 자원은 항상 검사한다. 사람은 디팟이 지어지는 동안 배럭에 2~3기를 더 큐에 넣는다(서플라이 막힘). `prereq_ok`(`masks.rs:346`)는 항상 서플라이를 요구한다. BWAPI `canMake`(`Templates.h:371`)도 항상 요구하므로 **라이브 브리지도 이 명령을 못 보낸다**(BWAPI 제약).
- 영향: BC 레이블 `type_mask`(supply) 무효. 규모는 §8 `label-check` `Train_* <- supply` 행으로 추정한다(m4 표본에서는 `type_mask` 7,044건 중 서플라이 몫을 따로 세지 않았다).
- 수정 제안(사용자 결정 필요): (a) 마스크를 엔진 규칙으로 바꾸고(어느 producer 큐가 비어 있지 않으면 서플라이 조건 면제) 라이브는 shim에서 BWAPI `canMake` 우회가 불가능하므로 실패로 두거나 OpenBW BWAPI 포크의 `canTrain`을 패치한다, (b) 현재(BWAPI 기준) 유지하고 레이블 손실을 받아들인다. 034-R3 선행 조건 문구가 달라지므로 spec 결정이 먼저다.

### F5. [Medium] Stop의 선택 자격 `can_move`가 시즈 탱크·터렛을 뺀다
- 근거: 엔진 `unit_order_allowed(Stop)`은 항상 true(`bwgame.h:17773`), BWAPI `canStop`(`Templates.h:1429`)은 비건물 완성 유닛 전부와 터렛·떠 있는 건물을 허용. 시즈 탱크에 S를 눌러 대상 지정을 푸는 것은 흔한 조작이다. `Stop`은 `select_cap = "can_move"`(`actions.toml:112`)라 시즈 탱크·터렛이 빠지고, 041-R11 retain 뒤 남는 유닛이 없으면 `select_cap` 무효다.
- 영향: BC 레이블 손실(§8 `Stop <- Terran_Siege_Tank_Siege_Mode (void)`), 라이브에서 시즈 탱크를 멈출 수 없음.
- 수정 제안: 새 선택 클래스 `can_stop`(= commandable ∧ (can_move ∨ 시즈 탱크 ∨ 터렛))을 `CAPABILITIES`에 추가하고 Stop에 쓴다. `unit_cap` K가 바뀌므로 feature 스키마 버전(030-R5·032-R6)을 올려야 하지만 shard에는 관측만 있어 재build는 필요 없다(§5). Hold는 BWAPI `canHoldPosition`이 `canMove()`를 요구하므로 현재가 맞다.

### F6. [Low] UnloadAll_Pos 선택 자격에 벙커가 포함된다
- `is_transport_loaded`(`masks.rs:124`)는 space_provided>0이면 모두 포함. BWAPI `canUnloadAllPosition`은 벙커를 Incompatible_UnitType으로 거절(`Templates.h:1801`). 사람은 벙커 MoveUnload를 낼 수 없어 레이블 영향 없음. 수정: UnloadAll_Pos는 드랍십만(클래스 분리 또는 `is_transport_loaded ∧ can_move`를 executor에서 거부).

### F7. [Medium] 미분류 사람 명령이 "학습에서 제외"가 아니라 **NoOp 시간**으로 들어간다
- 근거: `label.rs:224-231` `push(None)`은 Raw를 만들지 않고, `with_delays`가 그 구간을 NoOp 레이블(가중치 1)로 채운다. 031-AC3은 "`Unknown`으로 기록하고 학습에서 제외"다.
- 규모(§7, 재build records 6,768 시점): 명령 13,076,259 중 unknown 27,158(0.21%). 내역: 0x29 단일 Unload 17,164 / 0x15 order Guard(2) 9,486 / PickupBunker(95) 167 / TowerAttack(19) 41 / MoveToFireYamatoGun(114) 35 / TowerGuard(18) 24 / 0x2d 192 / 0x1f(테이블에 없는 유닛 타입) 32.
- 영향: 그 스텝에서 모델이 "아무것도 하지 않음"을 학습한다(단일 유닛 드랍 micro 등). 0.21%라 작지만 spec과 다르다.
- 수정 제안: (a) 레이블러가 Unknown 명령에도 Raw를 남기고(`action = NoOp`, `reason = unknown`) 로더가 weight 0으로 두게 하거나, (b) 단일 Unload를 034 액션(`Unload_Unit`: 수송선 선택, 탑승 유닛 대상)으로 추가. (b)는 action 스키마 버전 상승(034-R2)과 재build가 필요하므로 다음 재build 때 묶어서 결정. Guard(2)는 어떤 UI 조작인지 확인 필요(엔진은 `unit_order_allowed(Guard)` 비건물에 허용 → 사실상 제자리 대기). MoveToFireYamatoGun(114)은 `replay_orders.toml` Yamato 행에 추가하면 된다(데이터 변경, 재build 전까지는 미적용).

### F8. [Low] 착지 건물 우클릭을 모두 SetRally로 바꾸는 규칙이 터렛 적 우클릭(공격)을 잃는다
- `label.rs:288` 선택이 모두 착지 건물이면 SetRally_*. units.dat: 터렛 right_click_action=3(적이면 공격), 벙커·디팟 0(없음). 터렛 적 우클릭 레이블은 `can_rally` 없음으로 무효. 040-R11 문구("착지한 건물이 선택된 상태의 우클릭은 SetRally")도 터렛 예외가 없다. 수정: 선택이 터렛이고 대상이 적이면 Smart_Unit(또는 Attack_Unit)으로, `can_right_click_unit`에 터렛 포함. 빈도는 §8 `SetRally_Unit <- Terran_Missile_Turret` 행.

### F10. [Low] Load_Unit 대상 벙커가 선택 유닛의 종류를 보지 않는다
- 엔진 `unit_can_load_target`(`bwgame.h:1791`): 건물(벙커)은 테란·space_required ≤ 1만. BWAPI는 테란 생체만(`Templates.h:1691`; SCV는 생체 플래그가 있어 허용). 마스크는 벌처·탱크·골리앗이 벙커를 대상으로 고를 수 있다. 사람 레이블은 우클릭(Smart_Unit)이라 영향 없음.

### F11. [Low] Attack_Unit: 벙커 포함, 공중/지상 무기·사거리 미반영
- `can_attack`(`masks.rs:121`)이 벙커를 포함하지만 엔진(`default_action==0` → 건너뜀)·BWAPI(Unable_To_Hit) 모두 거절한다. 터렛·시즈 탱크는 사거리 밖 대상이면 BWAPI Out_Of_Range, 파이어뱃·벌처·탱크가 공중 대상이면 Unable_To_Hit — 엔진은 명령을 받아 추적만 한다(사람 레이블은 유효). 라이브는 유닛별 실패(F3과 결합해 명령 전체 실패 기록). 수정: 벙커 제거는 바로, 무기·사거리는 F3 뒤 유닛별 실패 집계로 관찰 후 결정.

### F12. [Medium, 측정 뒤 상향] max_select=1 액션의 레이블이 선택의 첫 유닛을 고른다 (SpiderMine_Pos 레이블의 8.6%가 이 이유로 무효, §8)
- `label.rs:232` `take(max_select)` 뒤에 `host_label_check`가 자격 필터를 하므로, 첫 유닛이 자격이 없으면(마인 없는 벌처, 에너지 없는 고스트) 다른 유닛이 시전했더라도 `select_cap` 무효. 엔진은 선택 전원이 시전한다(`actions.h` `action_order` 루프). 수정: 레이블은 12기를 저장하고 검사에서 자격 필터 뒤 `max_select`로 자른다(shard `selected` 내용이 바뀌므로 다음 재build에서).

### F13. [Low] Cancel_Train 슬롯
- 0x20 페이로드는 슬롯(254=마지막, `actions.h:1097`, `:575`)이다. `cmd.rs:185`가 유닛 태그로 보고 `map_tag`를 거치지만 값 범위상 결과가 같다(1.16.1·확장 한도 모두). 레이블·executor는 "마지막 취소"로 통일되어 사람이 특정 슬롯을 취소한 경우 의미가 다르다. 주석 정정 권장.

### F14. [Low] Smart_Unit: 시즈 탱크가 비적 대상
- 엔진 `get_default_order(3)` 비적 → idle(아무 일 없음), BWAPI Incompatible_State. 마스크는 허용. RL이 피하도록 학습되므로 두되, `can_right_click_unit`의 시즈 탱크 항목을 "적 대상일 때만"으로 좁히려면 선택·대상 결합 조건이 필요해 지금 구조(독립 gather)로는 표현 불가. 기록만.

### F15. [Low] 건설 중 SCV의 선택 자격
- 엔진은 order ConstructingBuilding인 SCV에 Stop 외 모든 명령을 거절(`bwgame.h:17761`), BWAPI는 Unit_Busy. `is_worker`·`can_move`·`can_be_loaded` 등이 포함한다. Build_* type_mask는 `is_worker` 존재만 보는데 `pick_worker`는 건설 중 SCV를 빼므로(`executor.rs:172`) 후보가 없으면 NoExecutorUnit으로 거부된다 — 034-R6 "후보 없으면 type_mask 닫음"과 어긋난다. 수정: `commandable`에서 ConstructingBuilding SCV 제외(Stop만 예외), Build_* type_mask에 `pick_worker` 후보 존재 검사.

### F16. [Low] 전환 중 상태
- 시즈/언시즈 진행 중(order Sieging·Unsieging)·클로킹 진행 중에 다시 명령하면 BWAPI Unit_Busy·Incompatible_State(`Templates.h:2185-2200`). T7 TankMode 오류 3 1회. 마스크에 order 98·99 제외를 넣으면 된다.

### F17. [Low] 무기 없는 유닛의 적 우클릭
- 메딕·드랍십·베슬 right_click_action=2: 엔진은 HealMove/Follow, BWAPI `canRightClickUnit`은 적 대상에 `canAttackUnit`을 요구해 거절(`Templates.h:1971`). 사람 레이블(마린+메딕 그룹 적 우클릭)은 유효하고 엔진도 처리하지만 라이브는 메딕 몫이 실패(F3 결합 시 명령 전체 실패 기록). shim에서 무기 없는 유닛의 적 우클릭을 `follow`로 바꾸는 것이 AMB-041-1(Attack_Pos → move)과 같은 결의 보정이다.

### F18. [Low] 시야 밖 기억 적 건물은 대상이 될 수 없다
- `attackable_enemy`·`any_unit`은 `visible`(지금 보임)을 요구(`masks.rs:102,166`). 엔진은 유닛 태그로 안개 속 건물 공격을 받는다(사람은 잔상을 클릭). 레이블은 `target_unit_cap` 무효(§8 `Attack_Unit [target_unit_cap] -> not_in_frame`). 라이브: 모델이 안개 속 건물을 공격 대상으로 못 고르고 Attack_Pos로 대신해야 한다. 시야 밖 자원과 같은 논리(한 번 본 실제 id)로 기억 적 건물을 `attackable_enemy`에 넣을 수 있다(맵핵 아님, 030-R2 "마지막 목격 상태"). 다만 shim 대상 해석(F2)도 함께 넓혀야 한다. 사용자 결정 사항.

### F19. [정보] 근접 프레임 delay 정규화의 부수 효과
- 같은 프레임 두 명령 중 두 번째 레이블은 다음 프레임 관측(첫 명령 적용 뒤)에 붙는다(`label.rs:424`). 그래서 두 번째 Train이 자원 부족으로 `type_mask` 무효가 될 수 있다(게임은 받았음). 040-R4(A-2) 결정의 알려진 비용이며 m4 분석의 `old`/`new` 구분에서 `new`가 이 경우다(연구별 20~30건 수준).

### F20. [Low] 프레임 0 명령
- 브리지 첫 관측은 프레임 1(011-R15)인데 레이블러는 명령 프레임마다 `units_at`을 요구하고 없으면 빈 집합을 쓴다(`pipeline.rs:214`). 프레임 0의 SELECT는 아무것도 뽑지 못해 그 뒤 첫 명령이 `select_empty`가 된다. 재생 파이프라인에서 프레임 0 명령은 프레임 1 관측을 쓰도록 하면 된다. m4 `select_empty` 316건의 일부일 수 있다(미측정).

### F21·F22. [Low] 매핑 보완
- PickupTransport(94)·PickupBunker(95): 선택이 수송선/벙커이고 대상이 탑승 유닛. `Load_Unit`은 방향이 반대라 `select_cap` 무효(94) 또는 Unknown(95). 엔진·BWAPI 모두 양방향을 받는다. 작은 수(§7 167건)라 기록만.
- MoveToFireYamatoGun(114) → Yamato_Unit, TowerAttack(19) → Attack_Unit(터렛)로 `replay_orders.toml`에 추가 가능(재build 때).

### F23·F24. [Medium, 요조사] `label-check`에서 드러난 무효 레이블 덩어리
- §8 참고: 대상 유닛이 관측에 없음(708, 0.78%)과 사람 건물 배치가 위치 마스크에서 거절됨(561, 0.62%; 엔지니어링 베이 59%). 둘 다 BC 레이블 손실이며 원인은 미조사다.

### 확인했고 문제 없음
- 034-R7(건물 단일 선택) 네 경로 일치. 034-R9(자기 대상) 네 경로 + 엔진 일치(`unit_order_can_target_self`: 착지 건물 랠리, 수송선 MoveUnload만 자기 대상 허용 → SetRally_Unit 예외가 맞고, UnloadAll_Pos는 대상이 위치라 해당 없음). 034-R8 반영(`world.rs:338-365`, `masks.rs:145`).
- 031-R8 "여러 시전 유닛을 선택하면 전원이 시전": 엔진 `action_order` 루프가 선택 유닛마다 `unit_can_use_tech`를 통과하면 모두 명령한다 — 결정 근거 맞음.
- 랠리: `can_rally`의 ProducesUnits 건물 = CC·배럭·팩토리·스타포트 = 엔진 `unit_is_factory`. 미완성 허용 = 엔진 `unit_can_receive_order`(RallyPoint*만 완성 불요, `bwgame.h:17754`).
- 큐 가능 order: orders.dat `can_be_queued`가 Stop·Move·Attack·Repair·Follow·Harvest·EnterTransport·HoldPosition·MoveUnload·RightClick·Patrol·MedicHeal에만 있고 Siege·Cloak·Lift·UseTech 계열에는 없다 — 테이블 `uses_queued`와 일치.
- `isInterruptible`: BWAPI는 `canMove`·`canRightClickUnit`·`canUseTech`에서 비중단 상태(Unit_Busy 3)를 거절하지만 orders.dat에서 비중단 order는 Sieging·Unsieging·HarvestGas·Harvest3/4·LiftingOff·NukeTrack 등 전환·채취 동작뿐이라 T7 오류 6의 원인이 아니다(오류 코드도 3이어야 함).
- delay 버킷·NoOp 채움·첫 스텝(040-R18) 테스트(`label.rs:456-490`) 맞음. flip 증강은 검사 뒤 변환, footprint 기준(030-R4) 맞음.
- 010-AC4·031-AC1·034-AC1·034-AC2·031-AC2·031-AC4 테스트는 주장하는 것을 실제로 검사한다(§6 참고 제한 있음).

## 5. 데이터 재build 필요 여부

**지금 도는 전체 재build를 멈추거나 다시 할 필요는 없다.**

- shard에는 RawFrame·레이블(유닛 id·타일)만 있고 마스크는 학습 때 계산하므로 F1·F2·F3·F4·F5·F6·F10·F11·F14·F15·F16·F17·F18은 코드 수정만으로 BC·RL·평가에 반영된다. F2·F3·F17은 브리지(shim) 수정이라 모든 호스트의 브리지 재빌드가 필요하다(011).
- 레이블러 수정이 필요한 것(F7 Unknown 처리, F8 터렛 우클릭, F12 12기 저장, F20 프레임 0, F21·F22 매핑)은 shard 내용(`selected`·NoOp 스텝)을 바꾸므로 재build가 있어야 반영된다. 합쳐도 명령의 0.3% 안팎(§7)이라 이번 재build를 되돌릴 이유는 못 되고, 다음 재build 전 목록(034 테이블·040-R11 개정)으로 남긴다.
- 040-R19(무효 레이블 ≥20% 시점 걸러냄)는 build 시점 마스크(d452c29)로 판정한다. 이번 발견은 무효 비율을 **줄이는** 쪽(F4·F5·F18)이거나 레이블에 영향이 없다(F1·F2·F3). 그래서 걸러진 60개 시점(§7, 전체의 0.9%)이 새 마스크로는 통과했을 수 있지만, 반대로 새로 걸러질 시점은 없다. 영향은 사용자 결정(2026-10-19 "재build 없음, 알려진 불일치")의 범위 안이다.
- 따라서 BC 본학습 전에는 코드 수정(F1·F5 필수, F2·F3 평가 대전 전 필수)만 하면 되고, 재build 없이 진행할 수 있다.

## 6. 테스트 공백

| 공백 | 설명 | 제안 |
|---|---|---|
| 대상 자격의 완성·탑승 조건 | `tests/common/mod.rs` `rich_units`의 미완성 유닛은 디팟·컨트롤타워(건물)뿐이고 탑승 마린은 선택 쪽만 검사(`disabled_units_take_no_command_034_r6`). F1을 잡을 테스트가 없다 | 미완성 아군 SCV·탑승 마린이 any_unit·own_unit·healable·matrixable·irradiatable·restorable에 없음을 검사 |
| 031-AC1 판정 기준 | "1회 이상 수락"만 보므로 Move_Unit 5회 중 4회 실패(`openbw_actions_031_ac1.md`)가 통과였다 | 액션별 실패 비율 상한 또는 실패 사유(BWAPI 오류) 집계를 리포트에 남기고 0이 아닐 때 경고 |
| 시야 밖 자원 대상 실행 | 시나리오는 보이는 자원만 대상 | 한 번 본 뒤 시야 밖이 된 미네랄에 Smart_Unit·Gather_Unit 1회 이상 수락 확인(F2) |
| 그룹 명령 | shim 유닛별 호출을 검사하는 테스트 없음 | 12기 Move_Pos 뒤 유닛들의 `order_pos`가 대형 보정을 받는지(엔진 `calc_group_move`) 또는 최소한 명령 1회 기록 |
| Stop·Hold 선택 자격 | `every_action_has_arguments_when_allowed_034_ac2`는 비어 있지 않음만 검사 | 시즈 탱크 Stop 허용(F5), 벙커 Attack_Unit 불허(F11), 벙커 UnloadAll_Pos 불허(F6) |
| 레이블러 | 터렛·벙커 우클릭, Unknown 명령의 시간 처리, max_select=1 다수 선택, 프레임 0 명령 테스트 없음 | 각 경우 1개 |
| 모델 헤드 034-R9 | `uranium-model/tests/model.rs`에 `self_target`·대상 마스크 테스트가 없다(034-R7은 있음) | teacher forcing에서 선택 유닛을 대상으로 한 레이블의 log-prob가 마스크됨, `none_free` 폴백 확인 |
| `label-check` 사유 세분 | `type_mask` 무효가 `afford`/`supply`로 나뉘지만(`data.rs:668`) 리포트 집계에 서플라이 몫이 없다 | §8처럼 주기적으로 기록 |
| Train 서플라이 규칙 | 엔진 규칙(큐 비어 있을 때만) 테스트 없음 | F4 결정 뒤 |

## 7. 커버리지 측정 (재build records, 읽기만, 2026-10-19 19:35 기준)

`<URANIUM_SHARED_DIR>/datasets/replays/records/*.jsonl` 13개 파일(진행 중).

| 항목 | 값 |
|---|---|
| 리플레이 결과 | written 6,768 / filtered 7,251 / desync 143 / unplayable 61 / error 3 |
| 스텝(레이블) | 16,179,181 |
| 명령(게임 영향) | 13,076,259 |
| 레이블된 명령 | 13,049,101 (**99.79%**, 034-AC3 ≥ 95% 통과) |
| Unknown | 27,158 (0.21%) |
| Unknown opcode | 0x29 Unload(단일) 17,164 · 0x15(미매핑 order) 9,763 · 0x2d 192 · 0x1f 32 · 0x12 4 · 0x2e 3 |
| 미매핑 order(0x15) | Guard(2) 9,486 · PickupBunker(95) 167 · TowerAttack(19) 41 · MoveToFireYamatoGun(114) 35 · TowerGuard(18) 24 · Harvest2(80) 5 · PlayerGuard(3) 3 · Nothing(23) 2 |
| filtered 사유 | APM 미달 5,642 · 제외 맵(040-R16) 722 · 5분 미만 643 · 활동 플레이어 수 1/0/3/4: 124/14/43/2 · **invalid_labels ≥ 20% (040-R19) 60** |

## 8. 현재 마스크로 본 레이블 무효 (`label-check`, 표본 shard 2개)

재build 출력 중 완성 shard 2개(stardata p00 shard_00000, scr_ladder p0 shard_00000)를 로컬로 복사해 master 72cf539 release 빌드(이 PC, `-j 2` Idle)로 `label-check`를 돌렸다. 맵 사전계산은 재build가 쓰는 공용 드라이브 `datasets/replays/build/maps`를 읽기 전용으로 썼다(로컬 `URANIUM_MAPS_DIR`에는 대상 맵 2개의 .scx만 있고 사전계산이 없다 — BC 전 동기화 필요, G-F2 참고). 출력 JSON: 작업 사본 `target/tmp/label_check.json`.

| 항목 | 값 |
|---|---|
| 게임 / 레이블 / 무효 | 45 / 90,650 / 5,306 (**5.853%**, T7 65게임 5.427%와 같은 수준) |
| 사유별 | type_mask 2,951 · select_cap 825 · target_unit_missing 708 · target_loc 561 · select_empty 151 · target_selected 55 · target_unit_cap 53 · select_building 2 |

발견 사항별 측정값:

| 발견 | 측정 | 해석 |
|---|---|---|
| F4 Train 서플라이 | `Train_* <- supply` 합계 149 (SCV 75, Vulture 37, Marine 22, SiegeTank 15) = 레이블의 0.16% | 큐가 빈 경우(게임도 거절)를 포함한 상한. **Low로 낮춤** |
| F5 Stop 시즈 탱크 | `Stop <- Terran_Siege_Tank_Siege_Mode (void)` 37 / Stop 672 (5.5%), `(kept)` 9 | 사람은 시즈 탱크에 Stop을 쓴다. Hold(void 45)는 BWAPI도 거절하므로 제외 |
| F12 max_select=1 첫 유닛 | `SpiderMine_Pos <- Terran_Vulture (void)` **120 / 1,390 (8.6%)** | 선택의 첫 벌처에 마인이 없어 무효. 엔진은 마인이 있는 벌처가 깐다. **Medium으로 올림** |
| F8 터렛 우클릭 | `SetRally_Unit <- Terran_Missile_Turret (void)` 1 | 무시 가능. 디팟 207·정제소 38·컴샛 27·벙커 23은 엔진도 아무 일 없음(무효 판정 맞음) |
| F18 안개 속 적 건물 공격 | `Attack_Unit [target_unit_missing] -> not_in_frame` 11, `[target_unit_cap]` 53(아군 SCV 25·마인 9 = 설계) | 작음 |
| F19 연타·선행 조건 | `Train_SCV <- afford (old)` 1,236 등 자원 부족 `old`가 type_mask의 대부분 | 게임도 거절, 무효 맞음 |

새로 드러난 것(원인 미조사, 다음 작업):

- **F23 [Medium, 요조사] `target_unit_missing` 708 (0.78%)**: `Smart_Unit` 531(그중 `not_in_frame` 482), `SetRally_Unit` 152(`not_in_frame` 139). 사람이 클릭한 유닛이 그 프레임의 RawFrame에 없다. 후보 원인: ① 브리지 시야 필터(`visible_to_self`: 발자국 타일 가시 + `isVisible`)가 사람 클라이언트의 가시 판정보다 좁다(시야 가장자리 유닛) ② SC:R 태그 변환(013, 재사용 카운터 5비트)에서 id가 어긋남 — scr_ladder shard에서 비율이 높은지 shard별로 다시 세면 구분된다 ③ 같은 프레임에 죽은 대상. `neutral Protoss_Dragoon` 15·`neutral Terran_SCV` 6은 GameInfo 플레이어 목록에 없는 소유자(관전자·이탈자 칸)의 유닛이다.
- **F24 [Medium, 요조사] `Build_* target_loc` 561 (0.62%)**: Supply Depot 151/780(19%), Barracks 75/214(35%), Engineering Bay 75/128(**59%**), Bunker 41/76(54%), Academy 36/80(45%). 사람의 건물 배치가 위치 마스크(`buildable_*`)에서 거절되는 비율이 너무 높다. 후보: ① 같은 건물 재클릭(이미 배치된 뒤) ② `blocked`(건물·자원·기억 적 건물) 또는 `explored`·creep 판정 ③ 레이블 타일과 footprint 기준 불일치. 함께 `Build_Refinery <- producer_or_required` 62/214(29%)도 설명이 필요하다(`label-check`의 Build 선행 조건 세분은 afford·요구 건물·`refinery_slot` 없음을 구분하지 않는다). 다음 작업: 실패 레이블의 타일에서 어느 조건(blocked/explored/creep/footprint/afford)이 막는지 `label-check`에 세분 출력을 추가해 확인(코드 변경이므로 이 감사 범위 밖).
- `select_empty` 151 (0.17%, Smart_Pos 66·Smart_Unit 60): m4 표본과 같은 수준. F20(프레임 0) 포함 여부 미확인.
- `select_building` 2: 레이블러 admit 규칙상 나올 수 없는 조합이 2건 있다(핫키 그룹에 건물+유닛이 남는 경로가 있을 수 있음). 무시 가능하나 기록.
- `(kept)` 상위: `Smart_Pos <- Siege_Mode (kept)` 465, `Attack_Pos <- Siege_Mode (kept)` 317 — 시즈 탱크가 섞인 이동 명령에서 탱크만 빠지고 나머지가 유효. 041-R11대로 동작.

## 9. 재현 명령
- 커버리지 집계: records 폴더에서 `coverage` 필드를 합산(§7 스크립트는 리포트에 넣지 않음, 간단한 JSON 합산).
- 레이블 검사: `URANIUM_MAPS_DIR=<maps> uranium-train label-check --shards <로컬 복사 폴더>` (release, Idle 우선순위).
- 게임 규칙 근거: `third_party/openbw/actions.h` `action_order`(438), `action_default_order`(368), `action_train`(333), `read_action_cancel_build_queue`(1097); `bwgame.h` `unit_can_receive_order`(17752), `unit_order_allowed`(17769), `build_queue_push`(2295), `unit_can_load_target`(1791), `get_default_order`(3603), `unit_order_can_target_self`(1480), `secondary_order_Train`(8196); `third_party/bwapi/bwapi/Shared/Templates.h` `canMake`(287), `canCommand`(396), `canTargetUnit`(465), `canAttackUnit`(613/651), `canMove`(1213), `canFollow`(1292), `canStop`(1429), `canLoad`(1653), `canUnloadAllPosition`(1792), `canRightClickUnit`(1941/1958), `canUseTechWithOrWithoutTarget`(2144), Healing(2343); `OpenBWData/BW/BWData.cpp`(1320 hidden_units); units.dat right_click_action(오프셋 9084), orders.dat can_be_interrupted/can_be_queued.

## 10. 후속 구현 T9: F7·F12, `Unload_Unit`, 레이블러 버전 혼합 (2026-10-19)

### 배경·예상
- 사용자 결정(2026-10-19): F7 미분류 사람 명령은 NoOp 시간이 아니라 `unknown` 레이블(가중치 0, 031-AC3)로 둔다. 단일 유닛 하차(0x29)는 새 액션 `Unload_Unit`(id 100, 101개)으로 배운다. 기존 액션에 맞는 order는 데이터 표로 매핑한다. F12 max_select = 1 레이블은 선택 전체를 저장하고 검사에서 자격 필터 → `max_select` 자르기.
- 진행 중인 전체 재build는 part 경계에서 멈췄고, 끝났거나 진행 중이던 part는 옛 레이블러(버전 1, shard schema 5), 남은 part는 새 바이너리(버전 2, schema 6)로 만든다. 그래서 로더·검사가 두 버전을 함께 읽어야 한다.
- 예상: NoOp 비율은 거의 그대로(미분류는 명령의 0.2%), SpiderMine `select_cap` 무효(§8 8.6%)는 거의 사라지고, 0x29는 `Unload_Unit` 레이블이 된다.

### 구현(브랜치 `t9-unload-unit`, master 미반영)
| 항목 | 내용 |
|---|---|
| 034 | `Unload_Unit`: 선택 `is_transport_loaded`(탑승 유닛이 있는 수송선, max 1. 리뷰 뒤 같은 정의의 별도 클래스 대신 재사용), 대상 `loaded_unit`(탑승한 아군 유닛), `UnitUnload`. 다른 수송선의 탑승 유닛이면 executor가 `BadTarget`(BWAPI `canUnload` Invalid_Parameter)이고 BC 레이블 검사(`LabelWalker::step`)는 `target_not_in_transport`로 무효. action 스키마 2, feature 스키마 5(K 47 → 48, A 100 → 101) |
| 010·011 | `CmdOrder::Unload(u32)`(마지막 variant), `PROTOCOL_VERSION` 3, shim `UR_ORD_UNLOAD` → `unload(unit)` |
| 040 레이블러 버전 2 | 0x29 → `Unload_Unit`, order 114 → `Yamato_Unit`, 유닛 대상 order 19(TowerAttack) → `Attack_Unit`. Guard(2)·TowerGuard(18)·PickupBunker(95)·위치 대상 TowerAttack·0x2d 등은 `unknown`(action id 65535). 선택은 전체 저장 |
| 검사·로더 | `host_label_check`: `unknown` → 사유 `unknown`, host 형태 NoOp(기억에 NoOp 기록). 자격 필터 → 자기 대상 제외 → 선택 순서 앞 `max_select`기. `select_too_many` 사유 없어짐. 040-R19 비율·040-AC7 무효 수·`label-check` `invalid`에서 `unknown` 제외(따로 `unknown`) |
| shard | schema 6 header `labeler_version`. schema 5는 레이블러 1로 읽음. records `labeler_version`(없으면 1). `dataset-stats` 레이블러 버전 표, `shard-stats` `labeler_versions`(shard·게임·스텝·unknown·unload_unit·selected_over_max) |

레이블러 1 shard에서 달라지는 것은 없다: 검사의 자르기는 이미 잘린 선택에 영향이 없고, `unknown`·`Unload_Unit` 레이블이 없다(테스트 `labeler_v1_v2_train_the_same_except_unknown_steps_040`, `schema5_shards_read_as_labeler_v1_040`, `resume_reads_labeler_v1_records_040_r17`).

### 표본 비교(같은 리플레이, 옛 바이너리 99eee6b vs 새 바이너리)
리플레이 80개(stardata 40, scr_ladder 40, 정렬 앞쪽), 학습 PC, `--jobs 4` Idle, 출력·맵 사전계산·staging은 작업 사본 임시 폴더(공용 shards·records 미사용, 보존 사본은 아래 "산출물"). 표의 검사 수치는 `target_not_in_transport` 검사를 넣기 전(커밋 5e96a20)의 것이다. 결과 동일: stardata written 30 / filtered 16, scr_ladder written 12 / filtered 30 / desync 1. 검사는 새 `uranium-train label-check`.

| 항목 | 옛(레이블러 1) | 새(레이블러 2) |
|---|---|---|
| 스텝 | 94,237 | 94,300 |
| NoOp 레이블 (비율) | 16,426 (17.43%) | 16,419 (17.41%) |
| 게임 영향 명령 / 레이블 / unknown | 77,931 / 77,861 / 70 | 77,931 / 77,905 / 26 |
| unknown 내역 | 0x29 44, 0x15 25(전부 Guard), 0x2d 1 | 0x15 25(Guard), 0x2d 1 |
| `unknown` 레이블 | 0 | 26 |
| `Unload_Unit` 레이블 | 0 | 44 (무효 11: 대상 자격 7, 선택 자격 2, type_mask 2) |
| 무효 레이블(`label-check`, unknown 제외) | 6,611 / 94,237 (7.015%) | 6,484 / 94,274 (6.878%) |
| SpiderMine_Pos 레이블 / `select_cap` 무효 | 1,431 / 142 (9.9%) | 1,431 / 5 (0.35%), 첫 벌처 제외 후 유효 224 |
| 선택이 `max_select`보다 긴 레이블 | 0 | 609 |

- `Unload_Unit` 대상 자격 무효 7건은 대상이 그 관측에서 탑승 상태가 아니었다(탱크 4·골리앗 2·마린 1). 내린 직후 같은 아이콘을 다시 누른 경우로 보이며 원인은 확인하지 않았다(작음).
- 이 표본에는 MoveToFireYamatoGun·TowerAttack이 없어 그 매핑의 실데이터 확인은 못 했다(단위 테스트만).

### 검증
- `cargo test --workspace --no-fail-fast`(debug, Idle `-j 2`) 통과.
- 031-AC1 OpenBW 실게임 시나리오(`Unload_Unit` 포함 101개로 확장)는 **보류**: T8(마스크·shim 수정)이 반영되고 브리지·hostd를 재빌드한 뒤 메인이 실행한다.
- Linux 호스트에서 임시 clone으로 `tools/openbw/build_bridge.sh`(BWAPI·OpenBW는 기존 checkout을 읽기 전용 링크, 임시 CARGO_HOME·target) 빌드 성공, 경고 0. 임시 폴더 삭제.

### 남은 일
- master 반영 뒤 모든 호스트의 브리지·hostd·에이전트 재빌드(`PROTOCOL_VERSION` 3, 옛 브리지는 010-R2로 거부). 재build를 이어 갈 장비는 새 `uranium-replay`로 바꾼다(재생 모드는 브리지 프로토콜을 검사하지 않아 옛 브리지로도 build는 된다).
- action·feature 스키마가 바뀌어 기존 체크포인트(스모크용 무작위 체크포인트 `<URANIUM_SHARED_DIR>/checkpoints/smoke/random-lstm` 포함)는 읽히지 않는다. master 반영 뒤 `uranium-agent random-checkpoint --out <URANIUM_SHARED_DIR>/checkpoints/smoke/random-lstm`로 다시 만든다.
- 결정(2026-10-19 사용자 확정): `unknown` 레이블은 040-R19 비율과 040-AC7 무효 집계에서 뺀다(레이블러 1 시점과 판정 기준을 같게). spec 040-R19에 반영.
- `unknown` 스텝은 가중치 0이므로 value 손실(스텝 가중치 × 게임 value 가중치)에서도 빠진다(spec 040-R11에 명시).

### 산출물
- `<URANIUM_SHARED_DIR>/artifacts/m6_action_audit/t9_sample/`: `build-old`·`build-new`(shards·records·maps·로그), `stats-{old,new}-{stardata,scr_ladder}.json`(`shard-stats`), `lc-{old,new}.json`(`label-check`), `sample_build.ps1`, `summarize.py`(약 80MB).

## 11. T8 조치: 마스크·executor·브리지 수정 (2026-10-19, 사용자 결정 2026-10-19)

### 배경·예상
- 사용자 결정(2026-10-19): 마스크·executor·브리지 쪽 발견은 지금 고치고, shard 내용을 바꾸는 레이블러 쪽(F7·F12, 레이블러 쪽 F8·F20~F22)은 아래 "다음 재build 목록"에만 남긴다. 돌고 있는 전체 재build는 건드리지 않는다(별도 복사 바이너리).
- 예상: F1·F5가 T7 오류 6과 시즈 탱크 Stop 무효를 없애고, F2·F3은 실게임에서만 드러난다. F23·F24는 마스크 문제가 아니라 특정 리플레이의 문제일 가능성이 크다.

### 발견별 상태
| 발견 | 상태 | 내용 |
|---|---|---|
| F1 | **수정** | 모든 타깃 자격에서 생산 중(미완성 비건물)·탑승 유닛 제외, 건설 중 건물은 유지(034-R10). 리뷰 반영(2026-10-19): 알·럴커 알·코쿤·합체 중 아콘·다크 아콘은 미완성이어도 대상(BWAPI `canTargetUnit`이 변태 중 유닛·아콘을 받음), 테스트 `eggs_cocoons_and_merging_archons_are_targets_034_r10`. 정제소 안 SCV(order HarvestGas, 숨김)도 대상·선택에서 제외. 테스트 `units_in_production_or_loaded_are_no_targets_034_audit_f1` |
| F2 | **수정** | shim이 대상 id를 관측 유닛 외에 맵 중립 미네랄·간헐천(시야 밖 포함)에서 찾음. 적 유닛은 관측에서만. 대상 없음·위치 없음은 즉시 `CommandFailed`(Unit_Does_Not_Exist·Invalid_Parameter), (-1,-1) 이동 제거(011-R20). 실게임 확인: 시야 밖 미네랄 Smart_Unit·Gather_Unit 수락, SCV order MoveToMinerals(85). 리뷰 반영(메인 기술 결정 2026-10-19, 맵핵 금지): 한 번 본 자원이 없어졌으면(채굴 완료·정제소 아래 간헐천) `CommandFailed` 대신 마지막 위치로 이동(SetRally는 그 위치로 랠리) — 실패 이벤트가 "자원이 없어졌다"는 시야 밖 정보를 알리지 않도록 |
| F3 | **수정** | `Command::Unit`을 `Game::issueCommand` 그룹 명령으로(같은 명령끼리 묶음, 그룹 불가 유닛은 개별). 1기 이상 수락이면 성공, 전원 거절일 때만 첫 거절 오류로 `CommandFailed`(011-R20, 031-R10). 유닛별 거절 수 필드는 프로토콜 변경이라 넣지 않음(U-6) |
| F4 | 사용자 결정 필요 | U-1 |
| F5 | **수정** | 선택 자격 `can_stop`(BWAPI `canStop`: 완성 비건물·떠 있는 건물·미사일 터렛, 시즈 탱크·건설 중 SCV 포함). feature 스키마 버전 5(T9) → 6, `unit_cap` K 48 → 50. 테스트 `stop_unload_attack_and_busy_selection_classes_034_audit` |
| F6 | **수정** | 선택 자격 `can_unload_pos`(탑승 유닛이 있는 비건물 수송선 = 드랍십) → UnloadAll_Pos. 벙커 제외 |
| F7 | T9에서 구현 | §10(레이블러 버전 2, `Unload_Unit`) |
| F8 | 다음 재build | 아래 목록 3 |
| F10 | 기록만 | 선택·대상 결합 조건(벙커는 테란 생체만)이라 독립 마스크로 표현 불가. 사람 레이블 영향 없음 |
| F11 | **일부 수정** | `can_attack`에서 벙커 제외. 공중/지상 무기·사거리 불일치는 U-3 |
| F12 | T9에서 구현 | §10 |
| F13 | **주석 정정** | `cmd.rs` 0x20 페이로드가 슬롯임을 명시(출력 변화 없음). 슬롯 지정 취소는 기록만 |
| F14 | 기록만 | 선택·대상 결합 조건 |
| F15 | **수정** | 건설 중 SCV(order ConstructingBuilding)는 `can_stop` 외 모든 선택 자격에서 제외. Build_* type_mask는 `pick_worker` 후보가 있을 때만 |
| F16 | **수정(클로킹 제외)** | 시즈·언시즈 전환 중(order Sieging·Unsieging) 탱크는 **라이브에서만** `can_stop` 외 선택 자격 제외. 리플레이 레이블(BC·040-R19·label-check)은 엔진 규칙(전환 뒤로 큐)대로 유효(U-4 사용자 결정 2026-10-19, `AgentMemory::replay_labels`, 테스트 `siege_transition_live_blocked_labels_valid_u4`). 클로킹 진행 중은 관측에 secondary order가 없어 미반영. 추가: `pick_worker`가 중단 불가 order(orders.dat `can_be_interrupted` = 0: HarvestGas·Harvest3/4·ResetHarvestCollision 등)인 SCV를 고르지 않음 — 031-AC1 Build_* Unit_Busy(오류 3)의 원인이었다 |
| F17 | **수정** | shim: 적을 공격할 수 없는 유닛의 적 유닛 우클릭은 follow로, 메딕은 대상 위치로 어택땅(BWAPI가 메딕 어택땅을 허용하고 게임이 HealMove로 바꿈 = 게임의 메딕 우클릭과 같은 order)으로 보냄(AMB-041-1과 같은 결) |
| F18 | 사용자 결정 필요 | U-2 |
| F20·F21·F22 | 다음 재build | 아래 목록 4·5 |
| F23·F24 | **원인 조사 완료, 마스크 수정 없음** | 아래 "F23·F24 원인", U-5 |

### 레이블 무효 비교 (`label-check`, 표본 shard 4개)
재build 출력 중 완성 shard 4개(scr_ladder p0 shard_00003·00006, stardata p05 shard_00002·p10 shard_00001, §8과 다른 shard)를 작업 사본 `target/lc/`로 복사해, 같은 shard를 master 72cf539 release(감사 때 빌드)와 이 작업 release로 돌렸다. 맵 사전계산은 공용 `datasets/replays/build/maps` 읽기 전용, Idle 우선순위.

| 출처 | 게임 | 레이블 | 무효 (72cf539) | 무효 (T8) | 차이의 주된 원인 |
|---|---|---|---|---|---|
| scr_ladder | 33 | 86,581 | 4,934 (5.699%) | 5,035 (5.815%) | Stop select_cap −56(F5), Smart_Unit select_cap +115(시즈 전환 중 탱크·건설 중 SCV만 고른 우클릭), TankMode·SiegeMode +28 |
| stardata | 44 | 92,259 | 3,515 (3.810%) | 3,686 (3.995%) | Smart_Unit select_cap +65, TankMode +51, SiegeMode +33(전환 중 재입력), Stop −4 |

- 무효가 약간(0.1~0.2%p) 늘었다. 늘어난 몫은 F15·F16이다: 사람은 시즈 전환 중에 같은 명령을 연타하고, 건설 중 SCV만 고른 채 우클릭한다. 엔진은 전자를 전환 뒤로 미뤄 받고(중단 불가 order 뒤에 큐), 후자는 거절한다. BWAPI(라이브)는 둘 다 Unit_Busy로 거절한다. 마스크가 BC·RL·실전에서 같다는 031-R4에 따라 라이브 기준을 택했다. 전환 중 연타 레이블을 살리려면 F16을 되돌리고 라이브는 실행 실패로 두는 선택지가 있다(U-4).
- F5는 시즈 탱크 Stop 무효를 없앴다(`Stop <- Terran_Siege_Tank_Siege_Mode (void)` 54 → 0).
- **리뷰 반영 뒤(F1 알·아콘 예외, U-4 전환 중 레이블 유효) 다시 측정**: scr_ladder 4,909 (5.670%), stardata 3,552 (3.850%). 72cf539 대비 scr_ladder −0.03%p, stardata +0.04%p(건설 중 SCV만 고른 명령 = 엔진도 거절, F15). 시즈 탱크 Stop 무효 0 유지. `unknown` 0(레이블러 1 shard).
- 출력: `<URANIUM_SHARED_DIR>/artifacts/m6_action_audit/t8/lc/` (`old_*`=72cf539, `lc_*`=리뷰 전, `lc2_*`=리뷰 반영 뒤 `.json`, 복사한 shard 4개, 최악 게임 추적 `t8_debug_d4778.txt`).

### F23·F24 원인
- **두 덩어리 모두 scr_ladder 일부 리플레이에 몰려 있다.** stardata 44게임은 `target_unit_missing` 53, `target_loc` 4뿐이고, scr_ladder 33게임은 1,224와 469다. scr_ladder에서도 상위 2게임이 F23+F24의 60%, 상위 8게임이 95%다(`per_game`). 같은 맵의 다른 게임은 깨끗하므로 맵 문제는 아니다.
- **F24 (`target_loc`)**: 거절된 사람 배치는 리플레이 재생에서도 지어지지 않았다(469건 중 같은 자리 재배치 16건 외에는 200스텝 안에 그 자리에 그 건물이 없음). 내용은 같은 타일에 10~20프레임 간격으로 반복된 배치, `y = 65535`(맵 밖) 배치, 탐색 안 된·건설 불가 타일 배치다. SC:R은 1.16.1과 달리 배치 실패 명령도 리플레이에 기록하는 것으로 보인다(1.16.1 클라이언트는 빨간 자리에서 명령을 보내지 않음). 게임도 거절한 명령이므로 **무효 판정이 맞고 마스크 문제가 아니다**. 무효 레이블은 이미 weight 0이다. `Build_Refinery <- producer_or_required` 54도 scr_ladder에만 있고 같은 현상(간헐천이 아닌 타일 14건 등)으로 본다.
- **F23 (`target_unit_missing`)**: 대부분 최근(≤ 96프레임) 관측된 적 유닛이 그 프레임 관측에 없다(830/1,224). 최악 게임(TvP, 무효 19.2%)은 같은 질럿을 1프레임마다 20여 회 우클릭·랠리 지정하는 구간이 반복되고, TvP인데 `Zerg_Egg`·`Special_Mature_Chrysalis`(맵 장식 중립 유닛)를 대상으로 하는 레이블이 있다. 사람 클라이언트에서는 보이던 유닛이 우리 재생에서는 안 보이거나(재생 경로 차이), 대상 id가 다른 유닛을 가리킨다(SC:R 태그 변환, 013). 어느 쪽이든 그 게임의 재생이 원 게임과 어긋난 것이고 마스크 문제가 아니다.
- 제안은 U-5.

### 031-AC1 강화와 실게임 (이 PC, 임시 hostd, debug, Idle)
- 판정 강화: "1회 이상 수락"에 더해 **모든 거절이 정당화된 것**이어야 한다(`justified`: Build_*·BuildAddon_*·Land_*의 Unbuildable_Location = 유닛이 선 자리, 지출 명령의 자원 부족 = 시나리오가 한 Act에 여러 지출 명령을 보냄). 시야 밖 한 번 본 미네랄에 Smart_Unit·Gather_Unit이 받아들여지고 SCV가 채취 order가 되는지도 본다(011-R20). 시나리오는 라이브 에이전트처럼 맵 사전계산을 읽는다(없으면 CC 자리 규칙이 BWAPI `canBuildHere`와 달라 오류 None(26) 1건이 났다).
- 실행: 작업 사본 hostd(`127.0.0.1:7791`, 포트 27000~, 최대 1게임)와 작업 사본 브리지 DLL. 재build와 동시.

| 회차 | 결과 | 원인·수정 |
|---|---|---|
| 1 | 98/100 (Lockdown_Unit·UnloadAll_Pos 없음), 시간 제한 | Build_* Unit_Busy(3) 7건 → `pick_worker` 중단 불가 order 제외. 고스트가 자기 핵에 죽어 Lockdown 불가 → 시나리오가 고스트 1기 유지. 벙커가 탑승 대상이 되면 Load 반복이 멈춤 → 반복 조건을 `can_unload_pos`로 |
| 2 | 99/100 (UnloadAll_Pos 없음) | Build_* Unit_Busy 0. Build_CommandCenter 오류 26 → 시나리오에 맵 사전계산 |
| 3·4 | 99/100 | 드랍십이 생산 중일 때 Load 반복 8회를 다 써 버림(생산 중 드랍십은 F1로 대상 아님 — 마스크가 맞음) → 완성 드랍십이 있을 때만 반복 |
| 5 | 100/100, 비무장 확인 실패 | 탑승 중인 드랍십에 Attack_Pos를 줘 다음 order가 PickupTransport → 탑승 직후·탑승 중 수송선은 확인에서 제외 |
| 6 | **통과**: 100/100, 거절 1건(BuildAddon_ControlTower Unbuildable_Location, 정당화), 시야 밖 미네랄 2/2, 베슬 Attack_Pos → Move | 송신 230·수락 229, 마지막 첫 수락 프레임 25,257(Yamato_Unit), 모두 덮어 StopGame(벽시계 28초) |
| 7·8 | **통과**(T9 반영 뒤, master aa187b2 위로 rebase): 101/101(`Unload_Unit` 포함), 거절 2건(BuildAddon_ControlTower·CovertOps Unbuildable_Location, 정당화), 시야 밖 미네랄 2/2 | 8회차 송신 231·수락 229, 마지막 첫 수락 프레임 25,209, 벽시계 28초 |

- T7 5회차와 비교: Move_Unit·Smart_Unit·Heal_Unit 오류 6(F1)과 Build_* Unit_Busy가 사라졌다.
- 리뷰 반영 뒤 9회차(`ac_run9.log`): 031-AC1 101/101, 거절 2건(Unbuildable_Location, 정당화), 시야 밖 미네랄 2/2 통과.
- 한계(U-6): 시나리오는 명령 단위 `CommandFailed`만 보므로, 그룹 명령에서 일부 유닛만 거절된 경우(1기 이상 수락 = 성공, 011-R20)는 잡지 못한다. 유닛별 거절 수는 프로토콜 필드가 생겨야 검사할 수 있다.
- 산출물: `<URANIUM_SHARED_DIR>/artifacts/m6_action_audit/t8/t8_031_ac1/` (메인 작업 사본 `reports/raw/t8_031_ac1/`에도 사본, `ac1_run1~8.log`·`ac_run9.log`, 8회차 표 `openbw_actions_031_ac1.md`, 리플레이 `.rep`).

### 다음 재build 목록 (shard 내용이 바뀌는 항목, 사용자 결정 2026-10-19)
다음 전체 재build 전에 한꺼번에 결정·구현한다. 지금 shard에는 반영되지 않는다.
1. ~~**F7 미분류 명령**~~ — T9(§10)에서 레이블러 버전 2로 구현됨. 원래 내용: Unknown 명령(0x29 단일 Unload, Guard(2), PickupBunker(95), TowerAttack(19), MoveToFireYamatoGun(114), TowerGuard(18) 등, 명령의 0.21%)의 시간이 NoOp 레이블(weight 1)로 채워진다. 레이블러가 Raw(`reason = unknown`)를 남기고 로더가 weight 0으로 두거나, 단일 Unload를 034 액션 `Unload_Unit`(수송선 선택, 탑승 유닛 대상)으로 추가(action 스키마 버전 상승).
2. ~~**F12 max_select=1 레이블 자르기**~~ — T9(§10)에서 구현됨. 원래 내용: 레이블러가 선택 12기를 저장하고 검사가 자격 필터 뒤 `max_select`로 자른다(SpiderMine_Pos 무효의 원인, 이번 표본 scr_ladder 84·stardata 208건).
3. **F8 터렛 적 우클릭**: 선택이 터렛이고 대상이 적이면 SetRally가 아니라 Smart_Unit(또는 Attack_Unit), `can_right_click_unit`에 터렛 포함(빈도 매우 작음).
4. **F20 프레임 0 명령**: 재생 파이프라인이 프레임 0 명령에 프레임 1 관측을 쓴다(`select_empty` 일부).
5. **F21·F22 매핑**: PickupTransport(94)·PickupBunker(95) 방향 반대 레이블, `replay_orders.toml`에 MoveToFireYamatoGun(114) → Yamato_Unit, TowerAttack(19) → Attack_Unit.
6. ~~(U-5 결정 시) scr_ladder 어긋난 재생 보조 필터.~~ — U-5 결정(2026-10-20)으로 하지 않음(040-R19 20% 유지).

### 사용자 결정 필요 (구현하지 않음)
- U-1 **F4 Train 서플라이**: 엔진은 큐가 비어 있지 않으면 서플라이를 보지 않지만 BWAPI `canMake`(라이브)는 항상 본다. (a) 마스크를 엔진 규칙으로(라이브는 실패 또는 BWAPI 포크 패치), (b) 현재 유지. 이번 표본 `Train_* <- supply`는 scr_ladder 193·stardata 258건(레이블의 0.2~0.3%, 큐가 빈 경우 포함 상한).
- U-2 **F18 시야 밖 기억 적 건물 공격 대상**: 맵핵은 아니지만(마지막 목격) shim 대상 해석을 적 건물로 넓혀야 한다.
- U-3 **F11 무기·사거리**: 공중/지상 무기 없는 유닛·사거리 밖 대상의 Attack_Unit 처리(F3 뒤로는 그룹 일부 거절이 명령 실패가 되지 않음).
- U-4 **F16 전환 중 연타 레이블**: **결정됨(사용자 결정 2026-10-19)** — BC 레이블은 유효(엔진이 받음), 라이브 마스크·executor는 막음(BWAPI 거절). 구현 완료.
- U-5 **F23·F24 scr_ladder 어긋난 재생**: **결정됨(사용자 결정 2026-10-20): 040-R19 20% 기준 유지, 보조 필터 없음((a)).** 재build(R-4·R-5) 뒤 040-R19로 걸러진 시점은 약 36.6k 중 216개다. stardata 132(0.4%), scr_ladder 83(6.3%), personal 1. 대부분 무효 20~25% 구간이다. scr_ladder 비율이 높은 이유는 나중에 조사하고 지금은 하지 않는다(`m6_rebuild.md` R-5). 이전 상태: 지금은 변경 없음, build 뒤 재평가(사용자 결정 2026-10-19). (a) 지금처럼 무효(weight 0)로 두고 040-R19(시점 무효 ≥ 20%)에 맡김 — 최악 게임 19.2%는 통과했다, (b) scr_ladder 보조 기준(`target_unit_missing`+`target_loc` 비율, 재생에서 실패한 배치 비율)을 040-R19에 더해 어긋난 재생을 걸러냄(재build 필요), (c) 상위 게임의 SC:R 태그 변환을 원본 리플레이로 대조(013-R7 왕복 검증은 통과했으므로 재생 차이일 가능성이 큼).
- U-6 **010 `CommandFailed` 유닛별 거절 수**: F3 뒤 부분 거절을 기록하려면 프로토콜 필드가 필요하다(PROTOCOL_VERSION 상승).

### 배포·후속
- feature 스키마 버전 6(T9의 5 위에 한 번 더): 버전 4·5 체크포인트는 로드가 거부된다. 스모크 랜덤 체크포인트를 `<URANIUM_SHARED_DIR>/checkpoints/smoke/random-lstm-fv6`로 새로 만들고 `config/plans/smoke-model.toml`이 그것을 가리킨다(옛 `random-lstm`은 그대로 둠). BC 본학습 전이라 다른 체크포인트 영향은 없다.
- 브리지(shim) 변경(011-R20): 이 브랜치가 master에 들어가면 **모든 게임 호스트(이 PC, Linux 호스트 A·B)의 브리지 재빌드와 hostd 재시작**이 필요하다. 재build(리플레이 재생)는 명령을 보내지 않으므로(011-R11) shim 변경의 영향을 받지 않지만, 재build가 끝난 뒤 배포한다. Linux 컴파일 확인: Linux 호스트 A 임시 폴더(nice 19, `-j 2`)에서 `liburanium_bridge.so` 빌드 성공, 경고 없음, 내보내기 심볼 `gameInit`·`newAIModule`만. 임시 폴더 삭제.
- **011-AC4 재확인(그룹 명령 뒤)**: 테스트 `same_seed_same_acts_same_game_011_ac4`(`#[ignore]`, hostd 필요)가 같은 seed(41)·같은 스크립트로 10분(14,400프레임) 게임을 두 번 돌려 관측마다 유닛 체크섬(id·종류·위치·hp·실드·에너지·order·order 대상·플래그, 자원)을 비교한다. 이 PC 임시 hostd에서 **통과**: 관측 1,800개 체크섬 전부 같음, 두 게임 모두 TimeLimitDraw 14,401프레임·승자 없음, 명령 191개씩. BWAPI `Unitset` 순회 순서가 프로세스마다 달라도 결과가 같다는 판단이 실측으로 맞았다.
- 진행 중인 전체 재build(R-4)의 040-R19 필터는 수정 전 마스크(d452c29)로 판정했다(F1·F15·U-4 미반영). BC 로더의 weight 0 판정과 아주 작게 다를 수 있으며(위 비교 기준 ±0.1%p 안팎), 다음 재build에서 맞춰진다.
- 작업 사본 `target/lc/`·`target/maps_tmp/`(투혼 자원 파일)는 `<URANIUM_SHARED_DIR>/artifacts/m6_action_audit/t8/`로 옮겼다.

# M7 — OpenBW 브리지 필드 → SC:R 원시 메모리 대응표

- 작성: 2026-10-22, M7 스파이크 중 서브 에이전트 조사(읽기 전용)
- 메인 확인: §4-2 탄환 시야 항목은 `BulletImpl.cpp:45`, `shim.cpp:522`, `extract.rs:302`로 직접 확인했다. 나머지 항목은 조사 결과를 그대로 옮긴 것이고, 구현할 때 하나씩 검증한다.
- 근거: `third_party/bwapi/bwapi`(OpenBW 포크 BWAPI 4.2 소스), `bridge/openbw/shim/shim.cpp`, `crates/uranium-bridge-openbw`, aise `bw_dat` 구조체(`structs.rs`)
- 약어:
  - UU=`BWAPI/Source/BWAPI/UnitUpdate.cpp`, GU=`GameUnits.cpp`, GE=`GameEvents.cpp`, GUP=`BWAPI/Source/GameUpdate.cpp`, PI=`PlayerImpl.cpp`, UI=`UnitImpl.cpp`, BI=`BulletImpl.cpp`, GC=`GameCommands.cpp`, MP=`Map.cpp`
  - OT=`BWAPI/Source/BW/OrderTypes.h`, US=`Shared/UnitShared.cpp`, UL=`BWAPILIB/Source/Unit.cpp`, BD=`OpenBWData/BW/BWData.cpp`, TP=`Shared/Templates.h`
- 전제: shim은 onStart에서 `CompleteMapInformation`(CMI)을 항상 켠다(shim.cpp:345). CMI를 켜면 BWAPI의 시야·접근 판정이 사실상 꺼지고, 실제 필터는 shim의 `visible_to_self`(141–155) 하나다. SC:R 브리지는 원시 메모리를 직접 읽으므로 CMI를 켠 것과 같은 조건이 된다.

## 1. 필드별 대응
| 필드 | shim.cpp (BWAPI 호출) | BWAPI 내부 계산 | SC:R 원시 소스 | 난이도·위험 |
|---|---|---|---|---|
| map_name | 383 `mapName()` | MP:58 `mapTitle` = CHK 시나리오 이름 | `Game.map_title[0x20]` | 쉬움. 인코딩(CP949/UTF-8)은 확인 필요 |
| map_hash | 383 `mapPathName()` → lib.rs:92 맵 파일 전체 SHA-256 | – | `Game.map_path[0x104]`가 가리키는 파일을 같은 방식으로 해시한다. `read_map_file`은 MPQ 안의 파일을 읽으므로 정의가 다르다 | 보통. 맵 사전계산 키라서 해시 대상 바이트가 같아야 한다. LAN 참가자는 받은 파일의 경로를 쓴다 |
| map_size | 350 | – | `Game.map_width_tiles/height_tiles` | 쉬움 |
| walkable (8px) | 354 `isWalkable` | MP:105: 맨 아래 4행, 아래 8행의 좌우 20칸은 false. 나머지는 크립 → true, partially_walkable이면 vf4 minitile 비트 0, 아니면 타일 walkable(bwgame.h:769) | `map_tile_flags`(0x400000 크립, 0x20000000 부분, 0x10000 walkable) + VarId `TilesetIndexedMapTiles`·`TilesetCv5`·`MinitileData` | 보통. 저그 시작 크립까지 포함해 프레임 0에 계산해야 같다 |
| buildable | 357 `isBuildable(x,y,false)` | BD:1205 `!(flags & unbuildable)` | `!(map_tile_flags & 0x800000)` | 쉬움 |
| ground_height | 358 | BD:1236 very_high +1, middle +2, high +4 | `(map_tile_flags >> 24) & 7` | 쉬움 |
| start_locations | 361 | GE:117–133 startPosition − (64,48) → 타일 | `Game.start_position[8]` | 쉬움 |
| PlayerInfo.start_tile | 376 | PI: CMI라 적 시작 위치도 나온다 | 위와 같다. 모델은 자기 시작 위치만 쓴다(map.rs:130) | 쉬움. 적은 None 권장 |
| race | 342, 373 | PI:41 로비 Random이고 아직 못 본 적은 Unknown | 로비 종족 위치는 미확인. `Player.race`는 무작위로 정한 뒤의 값이다 | 보통. 미확인 |
| is_enemy / name / id | 375, 380, 371 | BWAPI id = 슬롯 index | `Game.alliances[me][p]==0`, `Player.name[25]`, self = `local_player_id` | 쉬움 |
| minerals / gas | 61–62 | PI updateData | `Game.minerals/gas[p]` | 쉬움 |
| supply used/total | 63–64 | PI: total = min(provided, max) | `Game.supplies[race]` | 쉬움 |
| upgrades[63] | 65 | `currentUpgradeLevel` | aise `upgrade_level()` | 쉬움 |
| tech | 67 | PI: 연구 건물이 없는 테크는 true, Nuclear_Strike는 핵 가용성 | aise `tech_researched` + BWAPI `whatResearches` 표 | 보통 |
| researching / upgrading | 68, 71 | in-progress 비트 | aise `tech_in_progress`/`upgrade_in_progress` | 쉬움 |
| unit id | 22 `getReplayID` | UU:53 (index+1) \| gen<<11 | aise `to_unique_id` (index+1) \| minor<<13. 플레이어 유닛 index는 +1700만큼 밀려 있다(cmd.rs:146) | 쉬움. id는 불투명 값으로 다룬다 |
| owner / type | 99–100 | UU:136–155 미네랄 176–178 → 176, 변태 중 건물은 build_queue | `Unit.player`, `unit_id`, `build_queue`, `current_build_slot` | 쉬움 |
| pos | 101 | UU:173 탑승 중이면 수송선 위치 | `flingy.position`, 탑승 시 `related` | 쉬움 |
| vel / angle | 104–106 | UU:253–259 | `current_speed_x/y`, `facing_direction` | 쉬움 |
| hp / shield / energy | 107–109 | UU:175, 324–325 올림(/256) | `hitpoints`, `shields`, `energy` + units.dat | 쉬움 |
| order | 110 | UU:695 + `BWtoBWAPI.cpp` 매핑 | `Unit.order` + 같은 매핑표 | 보통(표 이식) |
| order_target / pos | 111–114 | UU:411–413 | `order_target` | 쉬움 |
| cooldowns | 115–117 | UU:206–225 subunit, 리버는 order_timer | 같은 필드 | 쉬움 |
| COMPLETED | 77 | UU:183, 363 변태 order면 false | `flags & 1` + order | 쉬움 |
| CONSTRUCTING | 78 | UU:370–384 + GU:263–290 보정 | `order`, `secondary_order`, `secondary_order_target.unit` | 보통 |
| SIEGED / BURROWED / CLOAKED | 79–81 | UL:336, UU:291–292 | `unit_id`, `flags` | 쉬움 |
| DETECTED | 82 | UU:94–108 | `sprite.visibility_mask`, `detection_status`, `flags & 0x100` | 보통 |
| LIFTED / STIMMED / CARRYING | 83–85 | UU:454, UL:348, UU:451 | `flags`, `stim_timer`, `carried_powerup_flags` | 쉬움 |
| LOADED / loaded_in | 86, 130 | UU:159–169 | `related`, `flags`, interceptor parent | 쉬움 |
| IDLE | 87 | UU:387–407 + GU 보정 | 원시 `order` | 보통 |
| UNDER_ATTACK | 88 | **UU:468 `recentlyAttacked = false`로 하드코딩 → 항상 0** | 0을 그대로 낸다(§4-1) | 쉬움 |
| GATHERING | 89 | UU:453 + US:449–520 | `flags`, `order`, `move_target.unit` | 보통 |
| TRAINING / BEING_CONSTRUCTED | 90–91 | UU:347–355, UL:235 | `build_queue`, `secondary_order` | 쉬움 |
| MOVING | 92 | UU:294 이동 플래그 \|\| order==Move. order는 직전 프레임 값 | `flingy_flags`, 직전 order 보관 | 쉬움. 1프레임 지연까지 맞출지 결정 |
| build_queue / remaining_* | 121–128 | UU:576–656, GU:300 | 같은 필드, 해처리는 larva_timer | 보통 |
| resources / status timers / blind / parasite / nuke_armed / has_addon | 129–137 | UU 각 줄 | 같은 필드 | 쉬움 |
| visibility / explored | 516–517 | MP:89–90 | `map_tile_flags` 하위 8비트 / 8–15비트(실게임 확인) | 쉬움 |
| creep | 518 (보이는 타일만) | BD:1225 | `map_tile_flags & 0x400000` | 쉬움 |
| effects: 탄환 | 521–526 | BI: CMI면 isVisible이 항상 true | `first_active_bullet` 체인 | 쉬움. 시야 필터는 §4-2 |
| effects: 핵 표시점 | 527–530 | GU:235–247 Ghost order로 추적 | Ghost `order`/`order_target.pos` | 보통 |
| effects: 스웜·웹 | convert.rs:133 | 중립 유닛 202/105 | 같은 유닛 | 쉬움 |
| events | 470–486 | §2 | 프레임마다 유닛 집합 비교 | 보통 |
| GameEnd | 436–468 | §2 | `Game.victory_state[8]`, `Player.player_type` | 보통 |
| PrivilegedSummary | 574–596 | – | realtime에서는 만들지 않는다(010-R10) | 해당 없음 |
| latency_frames | lib.rs:148 = 0 | – | 실측: 싱글 1, LAN 2(`m7_scr_spike.md`) | 쉬움 |

## 2. 재현해야 할 BWAPI 의미
- **적 유닛 시야 (shim:141).** 자기 유닛은 항상 넣는다. 그 밖의 유닛은 다음을 모두 만족해야 한다.
  - 살아 있다: sprite가 있고 `!(order==Die && order_state==1)`.
  - `sprite.visibility_mask & (1<<me)`가 켜져 있다.
  - 점유 타일 중 하나가 보인다.
  - 클록·버로우 유닛이면 탐지돼 있다.
- **스캐너 리스트.** BD:1333 `map_revealer_units`는 samase VarId에 없다. 유닛 배열 전체를 훑어 대신한다.
- **탐지 (UU:95–108).** RequiresDetection(0x100)인 유닛은 `detection_status == ~0 || detection_status & me`이면 탐지된 것이다.
- **isIdle (UU:387).** 생산·건설·변태 중이거나 연구·업그레이드 중이면 false다. 그 밖에는 원시 order가 대기 계열(PlayerGuard, Guard, Stop, Nothing 등)일 때만 true다.
- **이벤트.** CMI에서는 Discover가 Create와, Evade가 Destroy와 같은 프레임에 나간다(GU:161–216).
  - Morph는 접근 가능한 유닛의 타입이 바뀔 때 난다.
  - SC:R에서는 관측 프레임만이 아니라 매 프레임 살아 있는 유닛 집합을 비교해 쌓아야 한다.
  - 하류에서 쓰는 것은 Created·Destroyed·CommandFailed뿐이다.
- **이펙트.**
  - 스톰 157, EMP 151, 플레이그 168, 인스네어 170은 탄환이다.
  - 다크 스웜·디스럽션 웹은 중립 유닛이다.
  - 핵 표시점은 Ghost order에서 만든다.
  - Irradiate 이펙트는 OpenBW에서도 만들어지지 않는다.
- **게임 종료.**
  - GUP:48–87: `victory_state` 3 = 승리, 1·2·4·6 = 패배.
  - `player_type`이 10·11이 되거나, 중립이 아닌 플레이어가 7로 바뀌면 나간 것이다.
  - 둘 다 패배면 무승부(shim:453)다. 시간 제한·항복은 브리지가 판정한다.

## 3. 명령 → 네트워크 바이트
- shim은 그룹 명령을 12기씩 끊어 Select(1.16.1 `0x09`)를 보내고 명령 1개를 보낸다(GameImpl.cpp:686, UI:84).
- SC:R은 Select `0x63`[n, u32×n], RightClick `0x60`(x, y, u32 태그, u16 타입, queued), TargetedOrder `0x61`(x, y, u32, u16 타입, order, queued), Unload `0x62`(u32)를 쓴다. 나머지 opcode는 1.16.1과 같다(013).

| CmdOrder / Command | 1.16.1 바이트 (GC 줄) | SC:R |
|---|---|---|
| Move / Follow / Attack / AttackMove / Patrol | 0x15 order 6 / 49 / 8(건물 19) / 14 / 152 | 0x61 |
| Hold / Stop | 0x2B / 0x1A | 같음 |
| RightClick | 0x14 (204–206) | 0x60 |
| Siege·Unsiege·Cloak·Decloak·Burrow·Unburrow | 0x26/0x25/0x21/0x22/0x2C/0x2D | 같음 |
| Lift / Land | 0x2F / 0x0C (order 71) | 같음 |
| Load | 벙커 0x15 order 95, 수송선 94, 탑승 유닛 0x14 | 0x61 / 0x60 |
| UnloadAll / Unload unit | 벙커 0x28, 그 밖은 0x15 order 112 / 0x29 | 0x61 / 0x62 |
| Repair / Gather / ReturnCargo | 0x15 order 34 / 79 / 0x1E | 0x61 / 같음 |
| UseTech | 스팀 0x36, 그 밖은 0x15 tech order | 0x61 |
| Build / BuildAddon / Train / Morph / Research / Upgrade | 0x0C / 0x0C / 0x1F / 0x23·0x35 / 0x30 / 0x32 | 같음 |
| SetRally | 0x15 order 40 / 39 | 0x61 |
| Cancel ×6 | 0x18 / 0x20 / 0x31 / 0x33 / 0x34 / 0x19 | 같음 |
| 항복 채팅 / 나가기 | 0x5C / 0x57 | 채팅 0x5C는 `send_command`로 동작(스파이크 3차). 나가기는 0x57로 로컬 클라이언트가 빠지지 않음 |

## 4. OpenBW 경로와 맞추기 어려운 것 (학습·실전 불일치 위험)
1. **UNDER_ATTACK이 항상 0이다.** 모델은 0으로만 학습됐으므로 SC:R도 0을 낸다.
2. **탄환 이펙트에 시야 필터가 걸리지 않는다(010-R5 위반, 메인 확인).**
   - CMI에서 `BulletImpl::isVisible`이 모든 플레이어에게 true다(BI:45). 그래서 shim:522 필터가 아무것도 거르지 않는다.
   - feature 추출(`extract.rs:302`)도 시야를 보지 않고 hazard 채널에 그린다. BC shard와 코어 비교 학습에 시야 밖 탄환이 들어가 있다.
   - 사용자 결정 대상이다.
3. **NukeDetect 이벤트 위치가 시야와 관계없이 채워진다(GU:244–254).** 핵 표시점 이펙트는 shim:528에서 타일 시야로 거른다.
4. **Discover·Evade는 "시야 진입·이탈"이 아니라 "생성·사망"을 뜻한다.** Destroy는 탐지되지 않은 클록 유닛에도 나간다(shim:484).
5. **CommandFailed 오류 코드.**
   - BWAPI `canIssueCommand`(TP, 약 3,000줄)의 사전 검사가 만든다. SC:R에는 이 검사가 없으므로 이식하거나 근사해야 한다.
   - 기억 입력(memory.rs:126)에 쓰인다. 어려움.
6. **CMI 때문에 보이는 적 유닛의 내부 정보(생산 큐, 남은 시간)까지 들어간다(UI:202).** 원시 메모리로도 똑같이 만들 수 있지만, 사람이 볼 수 없는 정보다.
7. **수송선·벙커에 탄 적 유닛이 탑승 전 sprite 가시 비트를 유지해 관측될 수 있다.** SC:R도 같은 규칙을 쓰면 같은 결과가 나온다.
8. **미확인 항목:** 로비 종족 위치, `map_path`로 맵 파일을 해시할 수 있는지, 스캐너 리스트(배열 전체 스캔으로 대체).

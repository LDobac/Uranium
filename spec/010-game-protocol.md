# 010 — Game Protocol

- **Status**: Approved (2026-10-15. 2026-10-19 개정: R5 맵 공개 자원 예외(E-3, 사용자 결정, AMB-030-2·AMB-030-3). 같은 날 개정: R12 `GameEnd.final_frame`, `PROTOCOL_VERSION` 2(E-8, 사용자 결정 2026-10-19). 같은 날 개정: 034 `Unload_Unit` 추가로 `CmdOrder::Unload`, `PROTOCOL_VERSION` 3(R11, 사용자 결정 2026-10-19, 액션 감사 F7). 2026-10-22 개정: R5 효과는 중심 타일이 보이는 것만(AMB-030-4, 스키마 변경 없음). 2026-10-21 개정: 010-R13 `Act.surrender`, `PROTOCOL_VERSION` 4(사용자 결정, AMB-031-6). 2026-10-31 개정(사용자 결정, `reports/m6_action_audit.md` §11 U-2·U-6): R5 시야 밖 기억 적 건물 명령, 010-R14 `GameEvent::UnitRejected`, `PROTOCOL_VERSION` 5. 2026-10-31 리뷰 반영(메인 결정): R14 `command` 정의를 경우별(보낸 유닛·찾지 못한 유닛·잔상 명령)로 하나로 정함, 스키마 변경 없음)
- **Related**: [001](001-system-architecture.md), [011](011-bridge-openbw.md), [012](012-bridge-remastered.md), [013](013-version-mapping.md)

## Purpose
브리지와 에이전트 사이의 버전 중립 메시지 스키마와 전송 규약을 정의한다.

## Scope & Non-goals
- 범위: 상태 스냅샷, 명령, 핸드셰이크, 게임 종료, 지연 측정.
- 비범위: 모델 feature 형식([030](030-observation.md)에서 정의).

## Requirements
| ID | 요구사항 |
|---|---|
| 010-R1 | **MUST** 전송은 TCP 위에서 `u32 LE length + postcard payload` 프레이밍을 쓴다 |
| 010-R2 | **MUST** 첫 메시지는 `Hello{protocol_version, backend, game_version, mode}`이고, 버전이 다르면 연결을 거부한다 |
| 010-R3 | **MUST** 모든 ID(unit type, order, tech, upgrade, weapon)는 **정규 ID**(BWAPI 4.2 enum 값)로 표현한다. 네이티브 ID 변환은 브리지와 [013](013-version-mapping.md)이 맡는다 |
| 010-R4 | **MUST** 좌표는 픽셀 단위 `(x: u16, y: u16)`를 쓰고, 원점은 맵 좌상단이다 |
| 010-R5 | **MUST** RawFrame에는 자기 시야에서 관측 가능한 정보만 넣는다. 적 유닛은 현재 보이는 것만 넣는다. 기억 관리는 에이전트가 한다. **예외(2026-10-19 사용자 결정, E-3, AMB-030-2)**: 맵 파일에서 누구나 알 수 있는 중립 자원(미네랄 필드·베스핀 간헐천의 위치와 초기 잔량)은 시야 밖이어도 관측에 쓸 수 있다. 에이전트는 시야 밖 자원을 마지막으로 관측한 상태(한 번도 못 봤으면 맵 초기값)로 엔티티에 넣는다(030-R2). 초기값은 에이전트 쪽 맵 사전계산(`MapPrecompute`, CHK 유닛 배치, 031)에서 얻고, RawFrame·GameInfo 등 이 spec의 Interfaces는 바꾸지 않는다(AMB-030-3). 게임 시작 전에 맵에서 공개된 정보이므로 맵핵이 아니다. 시야 밖 자원의 현재 잔량이나 채취 상태, 적 유닛 정보는 이 예외에 들지 않는다. 한 번 본 시야 밖 자원은 명령 대상이 될 수 있고 한 번도 못 본 자원은 아니다(034 타깃 자격, 메인 결정 2026-10-19(게임 규칙)). 브리지는 이런 명령의 대상 id를 맵의 중립 자원에서 찾는다(011-R20, 액션 감사 F2). 이것은 에이전트가 이미 아는 id를 실행하는 것이고 관측에는 아무것도 더하지 않는다. **기억 적 건물 명령(2026-10-31 사용자 결정, 액션 감사 F18·U-2)**: 시야 밖이지만 본 적 있는 적 건물도 공격·우클릭·따라가기·랠리 대상이 될 수 있다(034 타깃 자격). 브리지는 그 대상의 실제 유닛을 찾지 않고, 자기가 관측해 둔 마지막 위치와 건물 종류로 잔상 명령을 보낸다(011-R20 ①-2, 012-R17). 사람이 안개 속 잔상을 클릭하는 것과 같은 명령이다. 명령 결과(실패 이벤트, 아군 order)가 그 건물의 현재 상태(남아 있는지, 옮겨졌는지)에 따라 갈리지 않으므로 관측에 아무것도 더하지 않는다. **효과(2026-10-22 사용자 결정, AMB-030-4)**: `effects`에는 중심 위치의 타일이 지금 자기 시야로 보이는 효과만 넣는다(사람 화면과 같은 규칙, 030 §3). 탄환 효과도 같다 |
| 010-R6 | **MUST** lockstep 모드에서 브리지는 `Act` 메시지를 받을 때까지 게임을 진행하지 않는다 |
| 010-R7 | **MUST** realtime 모드에서는 각 `Act`에 `valid_until_frame`을 붙이고, 지난 명령은 버리고 로그로 남긴다 |
| 010-R8 | **SHOULD** 메시지 크기는 프레임당 평균 64KB 이하로 유지한다 |
| 010-R9 | **MUST** `Hello`에 `game_id`와 `slot`(플레이어 위치)을 포함한다. 셀프플레이에서는 두 브리지가 같은 에이전트 서버에 접속하므로, 서버가 이 값으로 게임과 플레이어를 구분한다 |
| 010-R10 | **MUST** `PrivilegedSummary`는 **lockstep 학습 모드에서만** 송신한다. realtime(실전) 모드에서는 생성 코드가 동작하지 않아야 한다. 수신 측은 이 메시지를 value head 학습 입력으로만 쓴다(030-R9, 032-R11) |
| 010-R12 | **MUST** `GameEnd`에 마지막 `RawFrame` 이후의 최종 관측 `final_frame: Option<RawFrame>`을 싣는다(E-8, 사용자 결정 2026-10-19). 내용은 게임이 끝난 프레임의 자기 시야 상태(보이는 유닛, 시야·탐색 격자, 효과)와 마지막 `RawFrame` 뒤에 생긴 이벤트이고, 010-R5(맵핵 금지)를 그대로 따른다. `PrivilegedSummary` 정보는 넣지 않는다. 브리지가 만들 수 없을 때(shim 실패)와 리플레이 모드(011-R11)에서는 `None`이다. 에이전트는 이 프레임으로 마지막 결정 뒤의 dense 보상을 계산해 마지막 결정에 더한다(033-R7, 042 trajectory). 에이전트에 결정을 요구하지 않으므로 `Act` 응답은 없다. 스키마가 바뀌므로 `PROTOCOL_VERSION`을 2로 올리고, 버전 1인 상대는 010-R2로 거부된다. SC:R 브리지도 같은 필드를 보내야 한다(012-R10) |
| 010-R11 | **MUST** `Command`는 [034](034-action-table.md) 액션 테이블의 모든 `command` 매핑을 표현할 수 있어야 한다 |
| 010-R13 | **MUST** `Act.surrender: Option<String>`(2026-10-21 사용자 결정, AMB-031-6). `Some(text)`이면 에이전트가 항복한다(031-R12): `cmds`는 비어 있어야 하고(아니면 브리지가 거부), 브리지는 `text`가 비어 있지 않으면 채팅으로 보낸 뒤 게임을 떠난다(011-R21, 012-R11). `text`는 UTF-8 79바이트 이하이고(게임 채팅 버퍼 80바이트에 NUL 포함, BWAPI `sendText`와 같다) 넘으면 브리지가 거부한다. 떠난 쪽 `GameEnd`는 winner = 상대, reason = `Surrender`다. 떠난 쪽은 자기 승리를 보고하지 않으므로 020-R13 상호 퇴장 판정에 걸리지 않는다. 스크립트 봇과 재생 모드는 항상 `None`이다. 스키마가 바뀌므로 `PROTOCOL_VERSION`을 4로 올린다 |
| 010-R14 | **MUST** 유닛별 거절 이벤트 `GameEvent::UnitRejected { cmd_index: u16, unit: UnitId, unit_type: Option<u16>, command: u8, error: u16 }`(2026-10-31 사용자 결정, 액션 감사 F3·U-6). 직전 `Act`의 `cmd_index`번 `Command::Unit`에서 명령을 받지 못한 유닛 하나를 알린다. `unit`은 요청한 유닛 id다. `unit_type`은 그 유닛의 정규 유닛 종류이고, 유닛을 찾지 못했으면 `None`이다. `command`는 브리지가 그 유닛에 시도한 BWAPI `UnitCommandType` id다(2026-10-31 리뷰 반영, 이 정의 하나를 011이 따른다). 경우별로: ⓐ 브리지가 그 유닛에 명령을 내렸고 BWAPI가 거절했으면 유닛별 보정(011-R20 ③, 예: 비무장 유닛 Attack_Pos → Move) 뒤의 종류. ⓑ 유닛을 찾지 못했으면 대상 해석(011-R20 ①, 사라진 자원의 Move 변환 ①-1 포함) 뒤, 유닛별 보정 전의 종류. ⓒ 기억 적 건물 잔상 명령(011-R20 ①-2)이면 ⓐ·ⓑ와 관계없이 Attack은 `Attack_Unit`, 우클릭은 `Right_Click_Unit`(실제로 게임에 가는 것은 위치·유닛 종류를 실은 명령이지만 BWAPI 종류로는 이렇게 적는다). `error`는 BWAPI 오류 id다(`CommandFailed`와 같은 값). 거절된 유닛마다 하나씩 보낸다. 전원이 거절돼 `CommandFailed`를 보낼 때도 함께 보낸다. 명령 전체가 게임에 보내기 전에 실패하는 경우(대상·위치 없음, 011-R20 ②)에는 보내지 않고 `CommandFailed`만 보낸다. `Command::Unit` 외의 명령(Build·Train 등 유닛 하나)은 지금처럼 `CommandFailed`만 쓴다. 진단 전용이다. `CommandFailed`의 의미(1기 이상 받으면 성공, 031-R10)는 바뀌지 않고, 에이전트는 이 이벤트를 feature(`last_failed` 포함, 030 §4)에 쓰지 않고 세션 기록에만 남긴다(031 "유닛별 거절 기록"). SC:R 브리지는 보내지 않는다(012-R14). 리플레이 재생 모드는 명령이 없으므로 보내지 않는다. 마지막 variant로 붙이므로 기존 shard에 저장된 `RawFrame`은 그대로 읽힌다. 스키마가 바뀌므로 `PROTOCOL_VERSION`을 5로 올리고, 버전 4인 상대는 010-R2로 거부된다. SC:R 브리지도 버전 5로 다시 빌드한다 |

## Schema (Rust 정의 개요)
```rust
enum Backend { OpenBw, Remastered }
enum Mode { Lockstep, Realtime }

struct Hello { protocol_version: u32, backend: Backend, game_version: String, mode: Mode, game_id: u64, slot: u8, replay: bool /* 011-R11 */ }

struct GameInfo {
    map_name: String, map_hash: [u8; 32], map_size_tiles: (u16, u16),
    self_player: u8, players: Vec<PlayerInfo>,          // race, start location
    static_map: StaticMap,                               // walkability(8px), buildability(32px), ground height, 시작 위치
}

struct RawFrame {
    frame: u32,
    self_state: PlayerState,       // minerals, gas, supply used/total, upgrades[63], tech[47], researching
    units: Vec<UnitState>,         // 아군 전체 + 보이는 적/중립
    visibility: BitGrid,           // 32px 타일 단위, 현재 시야 / 탐색 여부
    creep: BitGrid,
    events: Vec<GameEvent>,        // UnitCreated/Destroyed/Morph/Discover/Evade, NukeDetect, CommandFailed, UnitRejected(010-R14)
    effects: Vec<Effect>,          // 보이는 스펠·투사체 효과 (030 hazard 채널)
}

enum GameEvent {
    UnitCreated(u32), UnitDestroyed(u32), UnitMorph(u32), UnitDiscover(u32), UnitEvade(u32), NukeDetect((u16, u16)),
    CommandFailed { cmd_index: u16, error: u16 },        // 031-R10: 그 명령을 받은 유닛이 없음, BWAPI 오류
    UnitRejected { cmd_index: u16, unit: u32, unit_type: Option<u16>, command: u8, error: u16 },
                                                         // 010-R14: 거절된 유닛 하나, 마지막 variant (PROTOCOL_VERSION 5)
}

struct Effect {                     // 자기 시야에서 보이는 것만
    kind: EffectKind,               // PsionicStorm, DarkSwarm, DisruptionWeb, NukeDot, EmpShockwave, Irradiate, Plague, Ensnare, Other(u16)
    pos: (u16, u16), radius: u16, remaining_frames: Option<u16>, source_owner: Option<u8>,
}

struct UnitState {
    id: u32,                        // 게임 내 수명 동안 고유 (재사용 시 generation 포함)
    owner: u8, unit_type: u16,
    pos: (u16, u16), vel: (i16, i16), angle: u8,
    hp: u16, shield: u16, energy: u16,    // 표시값(1/256 정밀도 버림)
    order: u16, order_target: Option<u32>, order_pos: Option<(u16,u16)>,
    ground_cd: u8, air_cd: u8, spell_cd: u8,
    flags: u32,       // completed, constructing, sieged, burrowed, cloaked, detected, lifted, stimmed, carrying, loaded, idle, under_attack...
    build_queue: [u16; 5], remaining_train: u16, remaining_build: u16, remaining_research: u16,
    resources: u16,   // 미네랄/가스 잔량 (자원 유닛)
    loaded_in: Option<u32>,
    status_timers: StatusTimers,  // lockdown, irradiate, matrix, stasis, plague, ensnare, maelstrom, stim (남은 프레임), blind, parasite (bool)
    spider_mines: u8, nuke_armed: bool, has_addon: bool,
}

enum Command {
    Unit { units: Vec<u32>, order: CmdOrder, queued: bool },   // 최대 12기 (원작 선택 제한)
    Build { worker: u32, unit_type: u16, tile: (u16,u16) },
    Train { building: u32, unit_type: u16 },
    Research { building: u32, tech: u16 }, Upgrade { building: u32, upgrade: u16 },
    CancelLast { unit: u32 }, ...
}
enum CmdOrder { Move(Pos), Attack(Target), Patrol(Pos), Hold, Stop, RightClick(Target),
                Siege, Unsiege, Stim, Cloak, Decloak, Lift, Land(Tile), Load(u32), UnloadAll(Option<Pos>),
                Repair(u32), Gather(u32), ReturnCargo, UseTech{tech:u16, target: Target}, ...,
                Unload(u32) /* 034 Unload_Unit: 탑승 유닛 1기 하차, 마지막 variant (PROTOCOL_VERSION 3) */ }

struct Act { cmds: Vec<Command>, advance_frames: u32, valid_until_frame: Option<u32>, surrender: Option<String> /* 010-R13, PROTOCOL_VERSION 4 */ }

/// 학습(lockstep) 모드 전용. 정책 입력에 쓰지 않는다 (010-R10).
struct PrivilegedSummary {
    frame: u32,
    enemy: PlayerState,                    // 상대 실제 자원, 서플라이, 업그레이드, 테크
    enemy_unit_counts: Vec<(u16, u16)>,    // (unit_type, 완성 수) 전체
    enemy_in_production: Vec<(u16, u16)>,  // (unit_type, 생산 중 수)
}
pub const PROTOCOL_VERSION: u32 = 5;   // 2: GameEnd.final_frame (010-R12), 3: CmdOrder::Unload (010-R11, 034 Unload_Unit, 2026-10-19), 4: Act.surrender (010-R13, 2026-10-21), 5: GameEvent::UnitRejected (010-R14, 2026-10-31)

struct GameEnd { winner: Option<u8>, frame: u32, reason: EndReason, stats: Option<BridgeStats>,
                 final_frame: Option<Box<RawFrame>> /* 010-R12: 마지막 RawFrame 이후 최종 관측(자기 시야) */ }
/// 브리지 측정값 (011-AC2/AC3). 2026-10-15 추가
struct BridgeStats { game_frames: u32, game_time_ms: f64, agent_wait_ms: f64, serialize_us_p99: f64, observations: u32 }
enum EndReason { Elimination, Surrender, TimeLimitDraw /* 020: 60분 초과 무승부 */, AgentTimeout, Aborted }
```

## 메시지 순서
```
Bridge → Agent : Hello, GameInfo
loop:
  Bridge → Agent : PrivilegedSummary   (lockstep 학습 모드에서만, 같은 프레임의 RawFrame보다 먼저)
  Bridge → Agent : RawFrame
  Agent → Bridge : Act
Bridge → Agent : GameEnd           (final_frame: 마지막 RawFrame 이후 이벤트와 종료 시점 자기 시야, 010-R12. 응답 없음)
```

## Acceptance Criteria
- 010-AC1: proto round-trip 테스트를 통과한다. 모든 메시지 타입에 대해 proptest를 1만 케이스 돌린다.
- 010-AC2: 버전이 다른 클라이언트의 연결 시도가 명시적 에러로 거부된다.
- 010-AC3: 200 유닛 상황 RawFrame의 직렬화 크기 ≤ 64KB, 직렬화+역직렬화 시간 ≤ 1ms.
- 010-AC4: [034](034-action-table.md)의 모든 action_type에 대해 executor 출력 `Command`가 존재하고 round-trip 된다.
- 010-AC5: realtime 모드 실행 시 `PrivilegedSummary`가 한 번도 송신되지 않는다.
- 010-AC7: `GameEnd.final_frame`(있음·없음)이 round-trip 되고(AC1 proptest), 버전 1 상대는 거부된다(AC2). 마지막 결정 뒤 보이는 적 유닛이 파괴되면 그 value 보상이 마지막 결정의 보상에 더해진다(`final_frame_reward_reaches_the_last_decision_010_r12`). 최종 프레임도 AC6 시야 검사를 받는다. OpenBW 실제 게임에서 최종 프레임이 오는지는 표본 단계에서 확인한다.
- 010-AC6: 모든 관측에서 자기 시야 밖의 적 유닛(가시 타일 및 인접 타일에 없거나, 은폐·버로우 상태인데 탐지되지 않은 유닛)이 0개다. 에이전트가 매 프레임 자동 검사한다(`visibility_violations`, 030-AC2 연동). 2026-10-15 변경: 011-R5에 따라 플래그를 항상 켜므로, 플래그 켜기·끄기 비교 대신 불변식 검사로 바꾼다.
- 010-AC8: `Act.surrender` 왕복 직렬화 테스트, 버전 3 상대 거부(010-R2), `cmds`가 있거나 79바이트를 넘는 `surrender` 거부.
- 010-AC9 (010-R14): `UnitRejected`(`unit_type` 있음·없음)가 왕복 직렬화되고(AC1 proptest), 버전 4 상대는 거부된다(AC2). OpenBW에서 일부 유닛만 받는 그룹 명령(예: 마린·파이어뱃 그룹의 공중 유닛 Attack_Unit)은 `CommandFailed` 없이 거절 유닛마다 `UnitRejected`가 오고, 전원이 거절되면 `CommandFailed`와 함께 온다. 대상이 없는 명령(011-R20 ②)에는 `UnitRejected`가 없다(031-AC1 시나리오).

## Risks & Open Questions
- 시야 BitGrid 해상도는 32px 타일로 확정한다(2026-10-17 사용자 확인). 엔진과 BWAPI(`isVisible`·`isExplored`)가 시야를 32px 타일 단위로만 기록하고, 모델 공간 입력도 32px 빌드타일이다(030-R6).

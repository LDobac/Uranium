# 010 — Game Protocol

- **Status**: Approved (2026-10-15. 2026-10-19 개정: R5 맵 공개 자원 예외(E-3, 사용자 결정, AMB-030-2·AMB-030-3))
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
| 010-R5 | **MUST** RawFrame에는 자기 시야에서 관측 가능한 정보만 넣는다. 적 유닛은 현재 보이는 것만 넣는다. 기억 관리는 에이전트가 한다. **예외(2026-10-19 사용자 결정, E-3, AMB-030-2)**: 맵 파일에서 누구나 알 수 있는 중립 자원(미네랄 필드·베스핀 간헐천의 위치와 초기 잔량)은 시야 밖이어도 관측에 쓸 수 있다. 에이전트는 시야 밖 자원을 마지막으로 관측한 상태(한 번도 못 봤으면 맵 초기값)로 엔티티에 넣는다(030-R2). 초기값은 에이전트 쪽 맵 사전계산(`MapPrecompute`, CHK 유닛 배치, 031)에서 얻고, RawFrame·GameInfo 등 이 spec의 Interfaces는 바꾸지 않는다(AMB-030-3). 게임 시작 전에 맵에서 공개된 정보이므로 맵핵이 아니다. 시야 밖 자원의 현재 잔량이나 채취 상태, 적 유닛 정보는 이 예외에 들지 않는다 |
| 010-R6 | **MUST** lockstep 모드에서 브리지는 `Act` 메시지를 받을 때까지 게임을 진행하지 않는다 |
| 010-R7 | **MUST** realtime 모드에서는 각 `Act`에 `valid_until_frame`을 붙이고, 지난 명령은 버리고 로그로 남긴다 |
| 010-R8 | **SHOULD** 메시지 크기는 프레임당 평균 64KB 이하로 유지한다 |
| 010-R9 | **MUST** `Hello`에 `game_id`와 `slot`(플레이어 위치)을 포함한다. 셀프플레이에서는 두 브리지가 같은 에이전트 서버에 접속하므로, 서버가 이 값으로 게임과 플레이어를 구분한다 |
| 010-R10 | **MUST** `PrivilegedSummary`는 **lockstep 학습 모드에서만** 송신한다. realtime(실전) 모드에서는 생성 코드가 동작하지 않아야 한다. 수신 측은 이 메시지를 value head 학습 입력으로만 쓴다(030-R9, 032-R11) |
| 010-R11 | **MUST** `Command`는 [034](034-action-table.md) 액션 테이블의 모든 `command` 매핑을 표현할 수 있어야 한다 |

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
    events: Vec<GameEvent>,        // UnitCreated/Destroyed/Morph/Discover/Evade, NukeDetect, CommandFailed
    effects: Vec<Effect>,          // 보이는 스펠·투사체 효과 (030 hazard 채널)
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
                Repair(u32), Gather(u32), ReturnCargo, UseTech{tech:u16, target: Target}, ... }

struct Act { cmds: Vec<Command>, advance_frames: u32, valid_until_frame: Option<u32> }

/// 학습(lockstep) 모드 전용. 정책 입력에 쓰지 않는다 (010-R10).
struct PrivilegedSummary {
    frame: u32,
    enemy: PlayerState,                    // 상대 실제 자원, 서플라이, 업그레이드, 테크
    enemy_unit_counts: Vec<(u16, u16)>,    // (unit_type, 완성 수) 전체
    enemy_in_production: Vec<(u16, u16)>,  // (unit_type, 생산 중 수)
}
pub const PROTOCOL_VERSION: u32 = 1;

struct GameEnd { winner: Option<u8>, frame: u32, reason: EndReason, stats: Option<BridgeStats> }
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
Bridge → Agent : GameEnd
```

## Acceptance Criteria
- 010-AC1: proto round-trip 테스트를 통과한다. 모든 메시지 타입에 대해 proptest를 1만 케이스 돌린다.
- 010-AC2: 버전이 다른 클라이언트의 연결 시도가 명시적 에러로 거부된다.
- 010-AC3: 200 유닛 상황 RawFrame의 직렬화 크기 ≤ 64KB, 직렬화+역직렬화 시간 ≤ 1ms.
- 010-AC4: [034](034-action-table.md)의 모든 action_type에 대해 executor 출력 `Command`가 존재하고 round-trip 된다.
- 010-AC5: realtime 모드 실행 시 `PrivilegedSummary`가 한 번도 송신되지 않는다.
- 010-AC6: 모든 관측에서 자기 시야 밖의 적 유닛(가시 타일 및 인접 타일에 없거나, 은폐·버로우 상태인데 탐지되지 않은 유닛)이 0개다. 에이전트가 매 프레임 자동 검사한다(`visibility_violations`, 030-AC2 연동). 2026-10-15 변경: 011-R5에 따라 플래그를 항상 켜므로, 플래그 켜기·끄기 비교 대신 불변식 검사로 바꾼다.

## Risks & Open Questions
- 시야 BitGrid 해상도는 32px 타일로 확정한다(2026-10-17 사용자 확인). 엔진과 BWAPI(`isVisible`·`isExplored`)가 시야를 32px 타일 단위로만 기록하고, 모델 공간 입력도 32px 빌드타일이다(030-R6).

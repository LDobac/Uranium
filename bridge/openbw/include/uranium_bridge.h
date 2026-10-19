/* C ABI between the BWAPI shim (C++) and the Rust bridge (crates/uranium-bridge-openbw).
 * Plain-old-data only; must match the #[repr(C)] structs in crates/uranium-bridge-openbw/src/ffi.rs.
 * Spec 011 / 001-R3: the shim only copies BWAPI state into these structs and executes commands. */
#ifndef URANIUM_BRIDGE_H
#define URANIUM_BRIDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define UR_NONE (-1)

typedef struct {
  uint8_t id;
  uint8_t race;      /* BWAPI Races: 0 Zerg, 1 Terran, 2 Protoss, 6 Random, else unknown */
  uint8_t is_self;
  uint8_t is_enemy;
  int32_t start_x;   /* tile, UR_NONE if unknown */
  int32_t start_y;
  char name[32];
} UrPlayer;

typedef struct {
  const char* map_name;
  const char* map_path;           /* file read by Rust for the SHA-256 */
  uint16_t width_tiles, height_tiles;
  const uint8_t* walkable;        /* (w*4)*(h*4) bytes, 0/1 */
  const uint8_t* buildable;       /* w*h bytes, 0/1 */
  const uint8_t* ground_height;   /* w*h bytes */
  const uint16_t* start_locations;/* x,y tile pairs */
  uint32_t n_start_locations;
  const UrPlayer* players;
  uint32_t n_players;
  uint8_t self_id;
  uint8_t privileged;             /* CompleteMapInformation enabled for PrivilegedSummary */
  uint8_t replay;                 /* 011-R10: replay playback, self_id is the perspective player */
} UrStatic;

typedef struct {
  int32_t minerals, gas;
  uint16_t supply_used, supply_total;
  uint8_t upgrades[63];
  uint64_t tech, researching, upgrading;
} UrPlayerState;

typedef struct {
  uint32_t id;
  uint8_t owner, angle;
  uint16_t type;
  uint16_t x, y;
  int16_t vx, vy;                 /* 1/256 px per frame */
  uint16_t hp, shield, energy, order;
  int32_t order_target;           /* UR_NONE */
  int32_t order_x, order_y;       /* UR_NONE */
  uint8_t ground_cd, air_cd, spell_cd, spider_mines;
  uint32_t flags;                 /* uranium_proto::flags */
  uint16_t build_queue[5];
  uint16_t remaining_train, remaining_build, remaining_research, resources;
  int32_t loaded_in;              /* UR_NONE */
  uint16_t timers[8];             /* lockdown, irradiate, matrix, stasis, plague, ensnare, maelstrom, stim */
  uint8_t blind, parasite, nuke_armed, has_addon;
} UrUnit;

enum { UR_EV_CREATE = 0, UR_EV_DESTROY, UR_EV_MORPH, UR_EV_DISCOVER, UR_EV_EVADE, UR_EV_NUKE, UR_EV_CMD_FAILED };

typedef struct {
  uint8_t kind;
  int32_t unit;                   /* or command index for UR_EV_CMD_FAILED */
  int32_t x, y;                   /* or error code in x for UR_EV_CMD_FAILED */
} UrEvent;

typedef struct {
  uint16_t bullet_type;           /* BWAPI BulletType id, 0xFFFF = nuke dot */
  uint16_t x, y;
  int32_t remaining;              /* frames, UR_NONE */
  int32_t owner;                  /* UR_NONE */
} UrEffect;

typedef struct {
  uint16_t type, count;
} UrTypeCount;

typedef struct {
  UrPlayerState enemy;
  const UrTypeCount* completed;
  uint32_t n_completed;
  const UrTypeCount* in_production;
  uint32_t n_in_production;
} UrPrivileged;

typedef struct {
  uint32_t frame;
  UrPlayerState self_state;
  const UrUnit* units;
  uint32_t n_units;
  const uint8_t* visible;         /* w*h tiles */
  const uint8_t* explored;
  const uint8_t* creep;
  const UrEvent* events;
  uint32_t n_events;
  const UrEffect* effects;
  uint32_t n_effects;
  const UrPrivileged* privileged; /* NULL unless UrStatic.privileged */
} UrFrame;

/* Command kinds */
enum { UR_CMD_UNIT = 0, UR_CMD_BUILD, UR_CMD_BUILD_ADDON, UR_CMD_TRAIN, UR_CMD_MORPH, UR_CMD_RESEARCH, UR_CMD_UPGRADE, UR_CMD_CANCEL, UR_CMD_SET_RALLY };
/* Orders for UR_CMD_UNIT */
enum {
  UR_ORD_MOVE = 0, UR_ORD_ATTACK, UR_ORD_PATROL, UR_ORD_HOLD, UR_ORD_STOP, UR_ORD_RIGHT_CLICK, UR_ORD_SIEGE, UR_ORD_UNSIEGE,
  UR_ORD_CLOAK, UR_ORD_DECLOAK, UR_ORD_BURROW, UR_ORD_UNBURROW, UR_ORD_LIFT, UR_ORD_LAND, UR_ORD_LOAD, UR_ORD_UNLOAD_ALL,
  UR_ORD_REPAIR, UR_ORD_GATHER, UR_ORD_RETURN_CARGO, UR_ORD_USE_TECH
};
/* Cancel kinds */
enum { UR_CANCEL_CONSTRUCTION = 0, UR_CANCEL_TRAIN, UR_CANCEL_RESEARCH, UR_CANCEL_UPGRADE, UR_CANCEL_ADDON, UR_CANCEL_MORPH };

typedef struct {
  uint8_t kind, order, queued, cancel_kind;
  int32_t unit;                   /* actor: worker / building / unit (single-unit commands) */
  int32_t target_unit;            /* UR_NONE */
  int32_t x, y;                   /* position (px) or tile; UR_NONE */
  uint16_t type;                  /* unit type / tech / upgrade */
  uint16_t n_units;               /* UR_CMD_UNIT: units in unit_pool[units_offset ..] */
  uint32_t units_offset;
} UrCmd;

typedef struct {
  const UrCmd* cmds;              /* valid until the next uranium_bridge_* call */
  uint32_t n_cmds;
  const int32_t* unit_pool;
  uint32_t advance_frames;
} UrAct;

/* Return 0 on success; non-zero means the shim must leave the game. */
int32_t uranium_bridge_start(const UrStatic* info);
int32_t uranium_bridge_step(const UrFrame* frame, UrAct* out);
/* reason: 0 elimination, 1 surrender/left, 2 time limit draw.
   last: own-vision state at the end and the events since the last step (010-R12), or NULL.
   Its privileged pointer is ignored. */
void uranium_bridge_end(int32_t winner, uint8_t reason, uint32_t frame, const UrFrame* last);
/* Next frame at which the shim must call uranium_bridge_step. */
uint32_t uranium_bridge_next_frame(void);
/* Max frames (URANIUM_MAX_FRAMES). The game seed is applied by OpenBW patch 0002. */
uint32_t uranium_bridge_max_frames(void);
uint8_t uranium_bridge_privileged_requested(void);
/* 011-R10: URANIUM_REPLAY_PERSPECTIVE player id, or -1 when unset. */
int32_t uranium_bridge_replay_perspective(void);
/* 011-R10: URANIUM_REPLAY_ENEMY player id, or -1 when unset. */
int32_t uranium_bridge_replay_enemy(void);

#ifdef __cplusplus
}
#endif
#endif

#ifndef URANIUM_BRIDGE_H
#define URANIUM_BRIDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define UR_NONE (-1)

typedef struct {
  uint8_t id;
  uint8_t race;
  uint8_t is_self;
  uint8_t is_enemy;
  int32_t start_x;
  int32_t start_y;
  char name[32];
} UrPlayer;

typedef struct {
  const char* map_name;
  const char* map_path;
  uint16_t width_tiles, height_tiles;
  const uint8_t* walkable;
  const uint8_t* buildable;
  const uint8_t* ground_height;
  const uint16_t* start_locations;
  uint32_t n_start_locations;
  const UrPlayer* players;
  uint32_t n_players;
  uint8_t self_id;
  uint8_t privileged;
  uint8_t replay;
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
  int16_t vx, vy;
  uint16_t hp, shield, energy, order;
  int32_t order_target;
  int32_t order_x, order_y;
  uint8_t ground_cd, air_cd, spell_cd, spider_mines;
  uint32_t flags;
  uint16_t build_queue[5];
  uint16_t remaining_train, remaining_build, remaining_research, resources;
  int32_t loaded_in;
  uint16_t timers[8];
  uint8_t blind, parasite, nuke_armed, has_addon;
} UrUnit;

enum { UR_EV_CREATE = 0, UR_EV_DESTROY, UR_EV_MORPH, UR_EV_DISCOVER, UR_EV_EVADE, UR_EV_NUKE, UR_EV_CMD_FAILED };
#define UR_EV_UNIT_REJECTED 7

typedef struct {
  uint8_t kind;
  int32_t unit;
  int32_t x, y;
  uint16_t unit_type;
  uint8_t command;
} UrEvent;

typedef struct {
  uint16_t bullet_type;
  uint16_t x, y;
  int32_t remaining;
  int32_t owner;
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
  const uint8_t* visible;
  const uint8_t* explored;
  const uint8_t* creep;
  const UrEvent* events;
  uint32_t n_events;
  const UrEffect* effects;
  uint32_t n_effects;
  const UrPrivileged* privileged;
} UrFrame;

enum { UR_CMD_UNIT = 0, UR_CMD_BUILD, UR_CMD_BUILD_ADDON, UR_CMD_TRAIN, UR_CMD_MORPH, UR_CMD_RESEARCH, UR_CMD_UPGRADE, UR_CMD_CANCEL, UR_CMD_SET_RALLY };
enum {
  UR_ORD_MOVE = 0, UR_ORD_ATTACK, UR_ORD_PATROL, UR_ORD_HOLD, UR_ORD_STOP, UR_ORD_RIGHT_CLICK, UR_ORD_SIEGE, UR_ORD_UNSIEGE,
  UR_ORD_CLOAK, UR_ORD_DECLOAK, UR_ORD_BURROW, UR_ORD_UNBURROW, UR_ORD_LIFT, UR_ORD_LAND, UR_ORD_LOAD, UR_ORD_UNLOAD_ALL,
  UR_ORD_REPAIR, UR_ORD_GATHER, UR_ORD_RETURN_CARGO, UR_ORD_USE_TECH,
  UR_ORD_UNLOAD
};
enum { UR_CANCEL_CONSTRUCTION = 0, UR_CANCEL_TRAIN, UR_CANCEL_RESEARCH, UR_CANCEL_UPGRADE, UR_CANCEL_ADDON, UR_CANCEL_MORPH };

typedef struct {
  uint8_t kind, order, queued, cancel_kind;
  int32_t unit;
  int32_t target_unit;
  int32_t x, y;
  uint16_t type;
  uint16_t n_units;
  uint32_t units_offset;
} UrCmd;

typedef struct {
  const UrCmd* cmds;
  uint32_t n_cmds;
  const int32_t* unit_pool;
  uint32_t advance_frames;
  uint8_t surrender;
  const char* surrender_chat;
} UrAct;

enum { UR_SURRENDER_LEAVE_FRAMES = 24 };

int32_t uranium_bridge_start(const UrStatic* info);
int32_t uranium_bridge_step(const UrFrame* frame, UrAct* out);
void uranium_bridge_end(int32_t winner, uint8_t reason, uint32_t frame, const UrFrame* last);
uint32_t uranium_bridge_next_frame(void);
uint32_t uranium_bridge_max_frames(void);
uint8_t uranium_bridge_privileged_requested(void);
int32_t uranium_bridge_replay_perspective(void);
int32_t uranium_bridge_replay_enemy(void);

#ifdef __cplusplus
}
#endif
#endif

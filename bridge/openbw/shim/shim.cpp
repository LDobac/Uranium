// Thin BWAPI AIModule shim (spec 011, 001-R3, CLAUDE.md C++ rule).
// Copies BWAPI state into the POD structs of uranium_bridge.h and executes commands returned
// by the Rust bridge. No decision logic lives here.
#include <BWAPI.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "uranium_bridge.h"

using namespace BWAPI;

namespace {

// Unit ids exposed to Rust are BWAPI replay ids: derived from the engine's unit index, so they
// are identical across processes (getID() depends on pointer order; 011-AC4) and match the
// unit ids in .rep command streams (040).
int32_t uid(Unit u) { return u->getReplayID(); }
// Observed player and its opponent: self()/enemy() in games, the 011-R10 perspective in replays.
Player g_me = nullptr, g_enemy = nullptr;

std::unordered_map<int32_t, Unit> g_units;  // replay id -> unit, rebuilt every observation
std::unordered_set<Unit> g_observed;         // the same units, by pointer

// Unit references (order target, transport) name observed units only. A reference to a dead
// or untracked unit carries a stale or uninitialized replayID (and exists() flag) in BWAPI,
// which differed between runs of one replay (030-R1, 040-R8); hidden ones would leak (010-R5).
int32_t id_or_none(Unit u) { return u && g_observed.count(u) ? uid(u) : UR_NONE; }
Unit find_unit(int32_t id) {
  auto it = g_units.find(id);
  return it == g_units.end() ? nullptr : it->second;
}

// Neutral mineral fields and geysers by replay id, out of vision included (rebuilt every
// observation). A command may target a resource the agent saw before and no longer sees (034
// target classes, 010-R5 exception: map-public resources only). Every other unit, enemy units
// above all, is resolved among the observed units alone (no map hack; 011-R20, audit F2).
std::unordered_map<int32_t, Unit> g_resources;
Unit find_target(int32_t id) {
  if (id == UR_NONE) return nullptr;
  if (Unit u = find_unit(id)) return u;
  auto it = g_resources.find(id);
  return it == g_resources.end() ? nullptr : it->second;
}
// Position of every map resource ever present (map-public, never cleared). A command on a
// remembered resource that no longer exists (mined out, geyser under a refinery) becomes a move
// there, like clicking a fog ghost sprite: no CommandFailed, so the agent learns nothing it could
// not see (011-R20, main decision 2026-10-19).
std::unordered_map<int32_t, Position> g_resource_pos;
bool is_map_resource(Unit u) {
  UnitType t = u->getType();
  return u->getPlayer() && u->getPlayer()->isNeutral() && (t.isMineralField() || t == UnitTypes::Resource_Vespene_Geyser);
}

void fill_player_state(Player p, UrPlayerState& s) {
  std::memset(&s, 0, sizeof s);
  s.minerals = p->minerals();
  s.gas = p->gas();
  s.supply_used = (uint16_t)p->supplyUsed(p->getRace());
  s.supply_total = (uint16_t)p->supplyTotal(p->getRace());
  for (int i = 0; i < 63; ++i) s.upgrades[i] = (uint8_t)p->getUpgradeLevel(UpgradeType(i));
  for (int i = 0; i < 47; ++i) {
    if (p->hasResearched(TechType(i))) s.tech |= 1ull << i;
    if (p->isResearching(TechType(i))) s.researching |= 1ull << i;
  }
  for (int i = 0; i < 63 && i < 64; ++i)
    if (p->isUpgrading(UpgradeType(i))) s.upgrading |= 1ull << i;
}

uint32_t unit_flags(Unit u) {
  uint32_t f = 0;
  auto set = [&](bool b, int bit) { if (b) f |= 1u << bit; };
  set(u->isCompleted(), 0);
  set(u->isConstructing(), 1);
  set(u->isSieged(), 2);
  set(u->isBurrowed(), 3);
  set(u->isCloaked(), 4);
  set(u->isDetected(), 5);
  set(u->isLifted(), 6);
  set(u->isStimmed(), 7);
  set(u->isCarryingMinerals() || u->isCarryingGas(), 8);
  set(u->isLoaded(), 9);
  set(u->isIdle(), 10);
  set(u->isUnderAttack(), 11);
  set(u->isGatheringMinerals() || u->isGatheringGas(), 12);
  set(u->isTraining(), 13);
  set(u->isBeingConstructed(), 14);
  set(u->isMoving(), 15);
  return f;
}

void fill_unit(Unit u, UrUnit& o) {
  std::memset(&o, 0, sizeof o);
  o.id = (uint32_t)uid(u);
  o.owner = (uint8_t)u->getPlayer()->getID();
  o.type = (uint16_t)u->getType().getID();
  Position p = u->getPosition();
  o.x = (uint16_t)std::max(0, p.x);
  o.y = (uint16_t)std::max(0, p.y);
  o.vx = (int16_t)std::lround(u->getVelocityX() * 256.0);
  o.vy = (int16_t)std::lround(u->getVelocityY() * 256.0);
  o.angle = (uint8_t)((int)std::lround(u->getAngle() * 256.0 / (2 * 3.14159265358979)) & 0xFF);
  o.hp = (uint16_t)std::max(0, u->getHitPoints());
  o.shield = (uint16_t)std::max(0, u->getShields());
  o.energy = (uint16_t)std::max(0, u->getEnergy());
  o.order = (uint16_t)u->getOrder().getID();
  o.order_target = id_or_none(u->getOrderTarget());
  Position op = u->getOrderTargetPosition();
  o.order_x = op.isValid() && op != Positions::None ? op.x : UR_NONE;
  o.order_y = op.isValid() && op != Positions::None ? op.y : UR_NONE;
  o.ground_cd = (uint8_t)std::min(255, u->getGroundWeaponCooldown());
  o.air_cd = (uint8_t)std::min(255, u->getAirWeaponCooldown());
  o.spell_cd = (uint8_t)std::min(255, u->getSpellCooldown());
  o.spider_mines = (uint8_t)u->getSpiderMineCount();
  o.flags = unit_flags(u);
  int i = 0;
  for (UnitType t : u->getTrainingQueue()) {
    if (i == 5) break;
    o.build_queue[i++] = (uint16_t)t.getID();
  }
  for (; i < 5; ++i) o.build_queue[i] = (uint16_t)UnitTypes::None.getID();
  o.remaining_train = (uint16_t)u->getRemainingTrainTime();
  o.remaining_build = (uint16_t)u->getRemainingBuildTime();
  o.remaining_research = (uint16_t)std::max(u->getRemainingResearchTime(), u->getRemainingUpgradeTime());
  o.resources = (uint16_t)std::max(0, u->getResources());
  o.loaded_in = id_or_none(u->getTransport());
  int t[8] = {u->getLockdownTimer(), u->getIrradiateTimer(), u->getDefenseMatrixTimer(), u->getStasisTimer(),
              u->getPlagueTimer(), u->getEnsnareTimer(), u->getMaelstromTimer(), u->getStimTimer()};
  for (int k = 0; k < 8; ++k) o.timers[k] = (uint16_t)std::max(0, t[k]);
  o.blind = u->isBlind();
  o.parasite = u->isParasited();
  o.nuke_armed = u->hasNuke();
  o.has_addon = u->getAddon() != nullptr;
}

// 010-R5 / 011-R5: the policy only sees own units and what our player can currently see.
bool visible_to_self(Unit u) {
  Player self = g_me;
  if (!u->exists() || !u->getPlayer()) return false;  // dying units
  if (u->getPlayer() == self) return true;
  if (!u->isVisible(self)) return false;
  // Belt and braces: at least one footprint tile must be visible to us (e.g. units handed to
  // the neutral player when their owner is defeated report isVisible() without being seen).
  TilePosition tp = u->getTilePosition();
  bool any = false;
  for (int y = tp.y; y < tp.y + u->getType().tileHeight() && !any; ++y)
    for (int x = tp.x; x < tp.x + u->getType().tileWidth() && !any; ++x) any = Broodwar->isVisible(x, y);
  if (!any) return false;
  if ((u->isCloaked() || u->isBurrowed()) && !u->isDetected()) return false;
  return true;
}

Position pos(const UrCmd& c) { return Position(c.x, c.y); }

// The command one unit of a selection gets. The unit is left null so that equal commands compare
// equal and go to the game as one group. `t` is the resolved unit target.
UnitCommand unit_command(Unit u, const UrCmd& c, Unit t) {
  bool q = c.queued != 0;
  Position p = pos(c);
  switch (c.order) {
    case UR_ORD_MOVE: return t ? UnitCommand::follow(nullptr, t, q) : UnitCommand::move(nullptr, p, q);
    case UR_ORD_ATTACK:
      if (t) return UnitCommand::attack(nullptr, t, q);
      // 034 Attack_Pos: the game attack-moves any unit that can move, and unarmed ones just
      // move; BWAPI only lets armed units and medics attack-move, so send those the move.
      if (!u->canAttackMove() && u->canMove()) return UnitCommand::move(nullptr, p, q);
      return UnitCommand::attack(nullptr, p, q);
    case UR_ORD_PATROL: return UnitCommand::patrol(nullptr, p, q);
    case UR_ORD_HOLD: return UnitCommand::holdPosition(nullptr, q);
    case UR_ORD_STOP: return UnitCommand::stop(nullptr, q);
    case UR_ORD_RIGHT_CLICK:
      if (!t) return UnitCommand::rightClick(nullptr, p, q);
      // Audit F17 (as AMB-041-1): the game turns the right click of a unit that cannot attack the
      // enemy target into a follow, and a medic's into a heal-move; BWAPI rejects both, so send a
      // follow, and the medic an attack-move to the target's position (the game makes it HealMove).
      if (t->getPlayer() && g_me->isEnemy(t->getPlayer()) && !u->canAttackUnit(t)) {
        if (u->getType() == UnitTypes::Terran_Medic && u->canAttackMove()) return UnitCommand::attack(nullptr, t->getPosition(), q);
        if (u->canFollow(t)) return UnitCommand::follow(nullptr, t, q);
      }
      return UnitCommand::rightClick(nullptr, t, q);
    case UR_ORD_SIEGE: return UnitCommand::siege(nullptr);
    case UR_ORD_UNSIEGE: return UnitCommand::unsiege(nullptr);
    case UR_ORD_CLOAK: return UnitCommand::cloak(nullptr);
    case UR_ORD_DECLOAK: return UnitCommand::decloak(nullptr);
    case UR_ORD_BURROW: return UnitCommand::burrow(nullptr);
    case UR_ORD_UNBURROW: return UnitCommand::unburrow(nullptr);
    case UR_ORD_LIFT: return UnitCommand::lift(nullptr);
    case UR_ORD_LAND: return UnitCommand::land(nullptr, TilePosition(c.x, c.y));
    case UR_ORD_LOAD: return UnitCommand::load(nullptr, t, q);
    case UR_ORD_UNLOAD_ALL: return c.x == UR_NONE ? UnitCommand::unloadAll(nullptr, q) : UnitCommand::unloadAll(nullptr, p, q);
    case UR_ORD_REPAIR: return UnitCommand::repair(nullptr, t, q);
    case UR_ORD_GATHER: return UnitCommand::gather(nullptr, t, q);
    case UR_ORD_RETURN_CARGO: return UnitCommand::returnCargo(nullptr, q);
    case UR_ORD_USE_TECH:
      if (t) return UnitCommand::useTech(nullptr, TechType(c.type), t);
      if (c.x != UR_NONE) return UnitCommand::useTech(nullptr, TechType(c.type), p);
      return UnitCommand::useTech(nullptr, TechType(c.type));
    // 034 Unload_Unit (2026-10-19): the transport drops one loaded unit (no queue argument).
    case UR_ORD_UNLOAD: return UnitCommand::unload(nullptr, t);
  }
  return UnitCommand();
}

// Arguments each order needs (011-R20): a unit, a unit or a position, or a position.
bool needs_unit(uint8_t o) { return o == UR_ORD_LOAD || o == UR_ORD_REPAIR || o == UR_ORD_GATHER || o == UR_ORD_UNLOAD; }
bool needs_unit_or_pos(uint8_t o) { return o == UR_ORD_MOVE || o == UR_ORD_ATTACK || o == UR_ORD_RIGHT_CLICK; }
bool needs_pos(uint8_t o) { return o == UR_ORD_PATROL || o == UR_ORD_LAND; }

// 011-R20 (audit F2/F3, user decision 2026-10-19): one command to the whole selection, as a
// player gives it. Units sharing a command go to the game together (`Game::issueCommand`: one
// select and one order, so the engine's group movement applies); units BWAPI cannot order as a
// group (lifted buildings) get it one by one. The command succeeds when at least one unit took it;
// otherwise the error is the first rejected unit's. A unit target that is neither observed nor a
// map resource, or a missing position, fails the command at once (no silent move to (-1, -1)).
bool exec_units(const UrCmd& c_in, const int32_t* pool) {
  UrCmd c = c_in;
  Unit t = find_target(c.target_unit);
  auto gone = g_resource_pos.find(c.target_unit);
  if (c.target_unit != UR_NONE && !t && gone != g_resource_pos.end() &&
      (c.order == UR_ORD_MOVE || c.order == UR_ORD_RIGHT_CLICK || c.order == UR_ORD_GATHER)) {
    c.order = UR_ORD_MOVE;
    c.target_unit = UR_NONE;
    c.x = gone->second.x;
    c.y = gone->second.y;
  }
  if ((c.target_unit != UR_NONE || needs_unit(c.order)) && !t) return Broodwar->setLastError(Errors::Unit_Does_Not_Exist);
  if ((needs_unit_or_pos(c.order) && !t && c.x == UR_NONE) || (needs_pos(c.order) && c.x == UR_NONE))
    return Broodwar->setLastError(Errors::Invalid_Parameter);
  std::vector<std::pair<UnitCommand, Unitset>> groups;
  Error first = Errors::None;
  bool accepted = false;
  for (uint32_t i = 0; i < c.n_units; ++i) {
    Unit u = find_unit(pool[c.units_offset + i]);
    if (!u) {
      if (first == Errors::None) first = Errors::Unit_Does_Not_Exist;
      continue;
    }
    UnitCommand cmd = unit_command(u, c, t);
    if (u->canIssueCommandGrouped(cmd)) {
      auto g = std::find_if(groups.begin(), groups.end(), [&](const std::pair<UnitCommand, Unitset>& x) { return x.first == cmd; });
      if (g == groups.end()) {
        groups.emplace_back(cmd, Unitset());
        g = groups.end() - 1;
      }
      g->second.insert(u);
      accepted = true;
      continue;
    }
    cmd.unit = u;
    if (u->issueCommand(cmd)) {
      accepted = true;
    } else if (first == Errors::None) {
      first = Broodwar->getLastError();
    }
  }
  // BWAPI's Unitset iterates by pointer hash, so the game may see the units in another order than
  // the pool; OpenBW's group handling does not depend on it (`action_order` and `calc_group_move`
  // sum over the selection).
  for (auto& g : groups) {
    g.first.unit = *g.second.begin();
    Broodwar->issueCommand(g.second, g.first);
  }
  // setLastError returns whether the error is None.
  return Broodwar->setLastError(accepted ? Errors::None : first == Errors::None ? Errors::Unit_Does_Not_Exist : first);
}

bool exec(const UrCmd& c, const int32_t* pool) {
  if (c.kind == UR_CMD_UNIT) return exec_units(c, pool);
  Unit u = find_unit(c.unit);
  if (!u) return Broodwar->setLastError(Errors::Unit_Does_Not_Exist);
  switch (c.kind) {
    case UR_CMD_BUILD: return u->build(UnitType(c.type), TilePosition(c.x, c.y));
    case UR_CMD_BUILD_ADDON: return u->buildAddon(UnitType(c.type));
    case UR_CMD_TRAIN: return u->train(UnitType(c.type));
    case UR_CMD_MORPH: return u->morph(UnitType(c.type));
    case UR_CMD_RESEARCH: return u->research(TechType(c.type));
    case UR_CMD_UPGRADE: return u->upgrade(UpgradeType(c.type));
    case UR_CMD_SET_RALLY: {
      Unit t = find_target(c.target_unit);
      auto gone = g_resource_pos.find(c.target_unit);
      if (c.target_unit != UR_NONE && !t && gone != g_resource_pos.end()) return u->setRallyPoint(gone->second);
      if (c.target_unit != UR_NONE && !t) return Broodwar->setLastError(Errors::Unit_Does_Not_Exist);
      if (!t && c.x == UR_NONE) return Broodwar->setLastError(Errors::Invalid_Parameter);
      return t ? u->setRallyPoint(t) : u->setRallyPoint(pos(c));
    }
    case UR_CMD_CANCEL:
      switch (c.cancel_kind) {
        case UR_CANCEL_CONSTRUCTION: return u->cancelConstruction();
        case UR_CANCEL_TRAIN: return u->cancelTrain();
        case UR_CANCEL_RESEARCH: return u->cancelResearch();
        case UR_CANCEL_UPGRADE: return u->cancelUpgrade();
        case UR_CANCEL_ADDON: return u->cancelAddon();
        case UR_CANCEL_MORPH: return u->cancelMorph();
      }
  }
  return false;
}

class UraniumModule : public AIModule {
 public:
  void onStart() override {
    // The game seed is applied to every client at game start by OpenBW patch 0002
    // (URANIUM_SEED). Do not call setRandomSeed() here: reseeding only our own client
    // desyncs LAN games against any other AI module (M3 finding).
    privileged_ = uranium_bridge_privileged_requested() != 0;
    replay_ = Broodwar->isReplay();
    g_me = Broodwar->self();
    g_enemy = Broodwar->enemy();
    if (replay_) {
      // 011-R10: observe as the perspective player; patch 0011 restricts tile vision and
      // detection to it. The opponent is URANIUM_REPLAY_ENEMY, else the lowest-id other player
      // that starts with units (players iterate in pointer order).
      int32_t pid = uranium_bridge_replay_perspective();
      g_me = pid >= 0 ? Broodwar->getPlayer(pid) : nullptr;
      if (!g_me) {
        fail_ = true;
        return;
      }
      for (Player p : Broodwar->getPlayers())
        if (p != g_me) Broodwar->setVision(p, false);
      int32_t eid = uranium_bridge_replay_enemy();
      g_enemy = eid >= 0 ? Broodwar->getPlayer(eid) : nullptr;
      if (!g_enemy)
        for (Player p : Broodwar->getPlayers())
          if (p != g_me && !p->isNeutral() && !p->getUnits().empty() && (!g_enemy || p->getID() < g_enemy->getID())) g_enemy = p;
    }
    // 030-R11 (E-6, 2026-10-19): read player races before CompleteMapInformation. Without the
    // flag BWAPI reports a lobby-Random player not seen yet as Races::Unknown; with it, the race
    // the game rolled would leak to the agent. Replays keep the played race (the shard's
    // `enemy_random` marks Random there).
    std::unordered_map<int, int> lobby_race;
    if (!replay_)
      for (Player p : Broodwar->getPlayers()) lobby_race[p->getID()] = p->getRace().getID();
    // 011-R5 (2026-10-15): always on in OpenBW lockstep mode. Needed for deterministic replay
    // ids; the policy observation is filtered by visible_to_self() regardless.
    Broodwar->enableFlag(Flag::CompleteMapInformation);
    Broodwar->setLocalSpeed(0);
    Broodwar->setFrameSkip(0);
    max_frames_ = uranium_bridge_max_frames();

    int w = Broodwar->mapWidth(), h = Broodwar->mapHeight();
    tiles_ = (size_t)w * h;
    std::vector<uint8_t> walk((size_t)w * 4 * h * 4), build(tiles_), height(tiles_);
    for (int y = 0; y < h * 4; ++y)
      for (int x = 0; x < w * 4; ++x) walk[(size_t)y * w * 4 + x] = Broodwar->isWalkable(x, y);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        build[(size_t)y * w + x] = Broodwar->isBuildable(x, y, false);
        height[(size_t)y * w + x] = (uint8_t)Broodwar->getGroundHeight(x, y);
      }
    std::vector<uint16_t> starts;
    for (TilePosition t : Broodwar->getStartLocations()) {
      starts.push_back((uint16_t)t.x);
      starts.push_back((uint16_t)t.y);
    }
    std::vector<UrPlayer> players;
    for (Player p : Broodwar->getPlayers()) {
      if (p->isNeutral()) continue;
      // 011-R10: a replay's GameInfo holds the perspective and its opponent only (1v1 form).
      if (replay_ && p != g_me && p != g_enemy) continue;
      UrPlayer o{};
      o.id = (uint8_t)p->getID();
      auto lr = lobby_race.find(p->getID());
      o.race = (uint8_t)(lr != lobby_race.end() ? lr->second : p->getRace().getID());
      o.is_self = p == g_me;
      o.is_enemy = g_me->isEnemy(p);
      TilePosition s = p->getStartLocation();
      bool known = s.isValid() && s != TilePositions::None && s != TilePositions::Unknown;
      o.start_x = known ? s.x : UR_NONE;
      o.start_y = known ? s.y : UR_NONE;
      std::strncpy(o.name, p->getName().c_str(), sizeof o.name - 1);
      players.push_back(o);
    }
    std::string name = Broodwar->mapName(), path = Broodwar->mapPathName();
    UrStatic info{};
    info.map_name = name.c_str();
    info.map_path = path.c_str();
    info.width_tiles = (uint16_t)w;
    info.height_tiles = (uint16_t)h;
    info.walkable = walk.data();
    info.buildable = build.data();
    info.ground_height = height.data();
    info.start_locations = starts.data();
    info.n_start_locations = (uint32_t)(starts.size() / 2);
    info.players = players.data();
    info.n_players = (uint32_t)players.size();
    info.self_id = (uint8_t)g_me->getID();
    info.privileged = privileged_;
    info.replay = replay_;
    if (uranium_bridge_start(&info) != 0) fail_ = true;
  }

  void onFrame() override {
    if (fail_ || done_) {
      if (!left_) { Broodwar->leaveGame(); left_ = true; }
      return;
    }
    int frame = Broodwar->getFrameCount();
    if ((uint32_t)frame >= max_frames_) {
      time_limit_ = true;
      done_ = true;
      Broodwar->leaveGame();
      left_ = true;
      return;
    }
    if (!replay_ && g_enemy) {
      enemy_buildings_.clear();
      for (Unit u : g_enemy->getUnits())
        if (u->getType().isBuilding()) enemy_buildings_.push_back(u);
    }
    if ((uint32_t)frame < uranium_bridge_next_frame()) return;
    step(frame);
  }

  // 011-R17: the enemy left (or was dropped) instead of being eliminated. A melee defeat needs
  // every building destroyed; leaving turns the buildings neutral, so they still exist.
  void onPlayerLeft(Player p) override {
    if (replay_ || p != g_enemy) return;
    for (Unit u : enemy_buildings_)
      if (u->exists()) { enemy_left_ = true; break; }
  }

  void onEnd(bool isWinner) override {
    int32_t winner = UR_NONE;
    uint8_t reason = 0;
    if (enemy_left_) reason = 1;
    if (time_limit_) reason = 2;
    else if (replay_) {
      // isWinner is meaningless in replays; report a winner only when one side was defeated.
      if (g_enemy && g_enemy->isDefeated() && !g_me->isDefeated()) winner = g_me->getID();
      else if (g_enemy && g_me->isDefeated() && !g_enemy->isDefeated()) winner = g_enemy->getID();
    } else if (isWinner) winner = g_me->getID();
    // Both eliminated on the same frame: draw (M3 finding).
    else if (g_enemy && !g_enemy->isDefeated()) winner = g_enemy->getID();
    // 010-R12 / 011-R19: the final own-vision frame carries what happened after the last step.
    int frame = Broodwar->getFrameCount();
    if (fail_ || replay_) {
      uranium_bridge_end(winner, reason, (uint32_t)frame, nullptr);
      return;
    }
    UrFrame f = observe(frame);
    uranium_bridge_end(winner, reason, (uint32_t)frame, &f);
    events_.clear();
  }

  void onUnitCreate(Unit u) override { event(UR_EV_CREATE, u); }
  void onUnitDestroy(Unit u) override { event(UR_EV_DESTROY, u); }
  void onUnitMorph(Unit u) override { event(UR_EV_MORPH, u); }
  void onUnitDiscover(Unit u) override { event(UR_EV_DISCOVER, u); }
  void onUnitEvade(Unit u) override { event(UR_EV_EVADE, u); }
  void onNukeDetect(Position p) override {
    if (p == Positions::Unknown) return;  // not visible to us
    events_.push_back(UrEvent{UR_EV_NUKE, UR_NONE, p.x, p.y});
  }

 private:
  void event(uint8_t kind, Unit u) {
    // With CompleteMapInformation (privileged mode) events fire for hidden units too; drop those.
    if (kind != UR_EV_DESTROY && kind != UR_EV_EVADE && !visible_to_self(u)) return;
    if ((kind == UR_EV_DESTROY || kind == UR_EV_EVADE) && u->getPlayer() != g_me && !u->isVisible(g_me)) return;
    events_.push_back(UrEvent{kind, uid(u), UR_NONE, UR_NONE});
  }

  // Own-vision state of `frame` (010-R5) with the events since the last step, into the buffers.
  UrFrame observe(int frame) {
    units_.clear();
    g_units.clear();
    g_observed.clear();
    g_resources.clear();
    for (Unit u : Broodwar->getAllUnits()) {
      if (u->exists() && is_map_resource(u)) {
        g_resources[uid(u)] = u;
        g_resource_pos[uid(u)] = u->getPosition();
      }
      if (!visible_to_self(u)) continue;
      g_units[uid(u)] = u;
      g_observed.insert(u);
    }
    // Second pass: references are resolved against the complete observed set.
    for (Unit u : Broodwar->getAllUnits()) {
      if (!g_observed.count(u)) continue;
      units_.emplace_back();
      fill_unit(u, units_.back());
    }
    int w = Broodwar->mapWidth(), h = Broodwar->mapHeight();
    visible_.resize(tiles_);
    explored_.resize(tiles_);
    creep_.resize(tiles_);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        size_t i = (size_t)y * w + x;
        visible_[i] = Broodwar->isVisible(x, y);
        explored_[i] = Broodwar->isExplored(x, y);
        creep_[i] = visible_[i] && Broodwar->hasCreep(x, y);
      }
    effects_.clear();
    for (Bullet b : Broodwar->getBullets()) {
      if (!b->isVisible(g_me)) continue;
      Position p = b->getPosition();
      effects_.push_back(UrEffect{(uint16_t)b->getType().getID(), (uint16_t)std::max(0, p.x), (uint16_t)std::max(0, p.y),
                                  b->getRemoveTimer(), b->getPlayer() ? b->getPlayer()->getID() : UR_NONE});
    }
    for (Position p : Broodwar->getNukeDots()) {
      if (!Broodwar->isVisible(TilePosition(p))) continue;
      effects_.push_back(UrEffect{0xFFFF, (uint16_t)p.x, (uint16_t)p.y, UR_NONE, UR_NONE});
    }

    UrFrame f{};
    f.frame = (uint32_t)frame;
    fill_player_state(g_me, f.self_state);
    f.units = units_.data();
    f.n_units = (uint32_t)units_.size();
    f.visible = visible_.data();
    f.explored = explored_.data();
    f.creep = creep_.data();
    f.events = events_.data();
    f.n_events = (uint32_t)events_.size();
    f.effects = effects_.data();
    f.n_effects = (uint32_t)effects_.size();
    return f;
  }

  void step(int frame) {
    UrFrame f = observe(frame);
    UrPrivileged priv{};
    if (privileged_ && g_enemy) {
      fill_privileged(priv);
      f.privileged = &priv;
    }

    UrAct act{};
    int32_t rc = uranium_bridge_step(&f, &act);
    events_.clear();
    if (rc != 0) {
      fail_ = true;
      return;
    }
    for (uint32_t i = 0; i < act.n_cmds; ++i) {
      if (!exec(act.cmds[i], act.unit_pool))
        events_.push_back(UrEvent{UR_EV_CMD_FAILED, (int32_t)i, Broodwar->getLastError().getID(), UR_NONE});
    }
  }

  void fill_privileged(UrPrivileged& p) {
    Player enemy = g_enemy;
    fill_player_state(enemy, p.enemy);
    completed_.assign(UnitTypes::allUnitTypes().size() + 1, 0);
    producing_.assign(completed_.size(), 0);
    for (Unit u : enemy->getUnits()) {
      int t = u->getType().getID();
      if (t < 0 || (size_t)t >= completed_.size()) continue;
      (u->isCompleted() ? completed_ : producing_)[t]++;
      for (UnitType q : u->getTrainingQueue())
        if (q.getID() >= 0 && (size_t)q.getID() < producing_.size()) producing_[q.getID()]++;
    }
    pc_.clear();
    pp_.clear();
    for (size_t t = 0; t < completed_.size(); ++t) {
      if (completed_[t]) pc_.push_back(UrTypeCount{(uint16_t)t, completed_[t]});
      if (producing_[t]) pp_.push_back(UrTypeCount{(uint16_t)t, producing_[t]});
    }
    p.completed = pc_.data();
    p.n_completed = (uint32_t)pc_.size();
    p.in_production = pp_.data();
    p.n_in_production = (uint32_t)pp_.size();
  }

  bool fail_ = false, done_ = false, left_ = false, time_limit_ = false, privileged_ = false, replay_ = false;
  bool enemy_left_ = false;
  std::vector<Unit> enemy_buildings_;
  uint32_t max_frames_ = 86400;
  size_t tiles_ = 0;
  std::vector<UrUnit> units_;
  std::vector<uint8_t> visible_, explored_, creep_;
  std::vector<UrEvent> events_;
  std::vector<UrEffect> effects_;
  std::vector<uint16_t> completed_, producing_;
  std::vector<UrTypeCount> pc_, pp_;
};

}  // namespace

#ifdef _MSC_VER
#define UR_EXPORT extern "C" __declspec(dllexport)
#else
#define UR_EXPORT extern "C" __attribute__((visibility("default")))
#endif
UR_EXPORT void gameInit(Game* game) { BroodwarPtr = game; }
UR_EXPORT AIModule* newAIModule() { return new UraniumModule(); }

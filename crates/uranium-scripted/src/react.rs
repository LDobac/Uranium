mod bunker;
mod geom;
mod marines;
mod observe;
mod params;
mod queries;
mod rules;
mod scout;
mod scvs;
mod state;

pub use uranium_env::types::FRAMES_PER_SECOND as FPS;

pub(crate) use self::bunker::bunker_order_stale;
pub use self::geom::{depot_center, ground_path};
pub(crate) use self::geom::{bunker_center, toward, within};
pub use self::params::{MapHints, ReactParams};
pub use self::rules::{RuleContext, RushRule, pool_start_frame, rush_rules};
pub(crate) use self::state::ReactState;
pub use self::state::{Branch, Harass, ReactStats};

const TILE: i64 = 32;
pub const BUNKER_HP: u16 = 350;
const CHOKE_MIN: i64 = 8;
const BUNKER_SEARCH_MAX: i64 = 9;
const ORDER_CONSTRUCTING: u16 = 33;
const ORDER_PLACE_BUILDING: u16 = 30;
const ORDER_MOVE: u16 = 6;
const ATTACK_ORDERS: [u16; 5] = [8, 9, 10, 11, 14];
const SCOUT_AWAY_TILES: i64 = 12;

#[cfg(feature = "testkit")]
pub mod testkit;

#[cfg(test)]
mod tests;

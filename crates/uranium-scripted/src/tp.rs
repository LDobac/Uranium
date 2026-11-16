mod army;
mod economy;
mod observe;
mod params;
mod rules;
mod state;

pub use self::params::TpParams;
pub use self::rules::{TpContext, TpRule, tp_rules};
pub(crate) use self::state::{TpState, pick_key, pick_key_id, snap};
pub use self::state::{Matchup, TpStats};

const SCV_HP: u32 = 60;
const OUT_TILES: i64 = crate::brain::DEFEND_RADIUS / 32;

#[cfg(feature = "testkit")]
pub mod testkit;

#[cfg(test)]
mod tests;

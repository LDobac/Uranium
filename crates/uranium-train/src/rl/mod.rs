//! Reinforcement learning and the league (spec 042).

pub mod config;
pub mod league;
pub mod learner;
pub mod returns;
pub mod run;
pub mod trajectory;

#[cfg(test)]
pub(crate) mod testkit;

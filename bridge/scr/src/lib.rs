//! SC:R bridge (spec 012): a samase plugin DLL that plays one side of an SC:R 1.23.10.13515 game
//! for the Uranium agent, speaking the same protocol (010) with the same observation meaning as
//! the OpenBW bridge (012-R12). Everything but `raw` and `samase` is plain logic over `snapshot`
//! values, tested without the game.

pub mod bw;
pub mod commands;
pub mod end;
pub mod events;
pub mod game;
pub mod io;
pub mod log;
pub mod observe;
pub mod raw;
pub mod schedule;
pub mod snapshot;
pub mod tables;

#[cfg(windows)]
mod samase;

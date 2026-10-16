//! `uranium-replay`: StarCraft replay parsing (040-R1), SC:R -> 1.16.1 conversion (013-R6..R8)
//! and dataset filtering (040).

pub mod chk;
pub mod cmd;
pub mod container;
pub mod header;
pub mod label;
pub mod pipeline;
pub mod play;
pub mod shard;
pub mod trace;

use anyhow::{Context, Result};
use serde::Serialize;

pub use container::Format;
pub use header::{Header, Player};

#[derive(Debug, Clone)]
pub struct Replay {
    pub format: Format,
    pub header: Header,
    pub raw_header: Vec<u8>,
    pub blocks: Vec<cmd::Block>,
    pub map: Vec<u8>,
    /// SC:R `LMTS` unit limit (013 tag remapping).
    pub unit_limit: Option<u32>,
}

pub fn parse(file: &[u8]) -> Result<Replay> {
    let s = container::read(file)?;
    let blocks = cmd::parse(&s.commands).context("commands")?;
    Ok(Replay { format: s.format, header: header::parse(&s.header), raw_header: s.header, blocks, map: s.map, unit_limit: s.unit_limit })
}

/// Milliseconds per frame at Fastest (screp `Frame.Milliseconds`).
pub const MS_PER_FRAME: u64 = 42;

#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct PlayerStats {
    pub id: u8,
    pub race: u8,
    pub cmds: u32,
    pub last_cmd_frame: u32,
    /// Same definition as screp: commands / minutes up to the player's last command.
    pub apm: u32,
    /// Train + build commands, used to tell players from observers.
    pub train_build: u32,
}

impl Replay {
    pub fn cmds(&self) -> impl Iterator<Item = &cmd::Cmd> {
        self.blocks.iter().flat_map(|b| b.cmds.iter())
    }

    pub fn player_stats(&self) -> Vec<PlayerStats> {
        self.header
            .players
            .iter()
            .map(|p| {
                let mine = self.cmds().filter(|c| c.player == p.id);
                let (mut n, mut tb, mut last) = (0u32, 0u32, 0u32);
                for c in mine {
                    n += 1;
                    if matches!(c.op, cmd::op::TRAIN | cmd::op::BUILD | cmd::op::UNIT_MORPH) {
                        tb += 1;
                    }
                    if c.frame <= self.header.frames {
                        last = c.frame;
                    }
                }
                let mins = (last as u64 * MS_PER_FRAME) as f64 / 60_000.0;
                let apm = if last == 0 { 0 } else { (n as f64 / mins + 0.5) as u32 };
                PlayerStats { id: p.id, race: p.race, cmds: n, last_cmd_frame: last, apm, train_build: tb }
            })
            .collect()
    }

    /// The 1.16.1 file OpenBW can load (013-R6). Legacy replays are re-encoded too, so every
    /// replay goes through the same path.
    pub fn to_legacy_file(&self) -> Result<Vec<u8>> {
        // 013: commands of ids outside the header players (observers, id >= 128) stop OpenBW's
        // replay loader ("player id not found"); they never affect the game, so drop them.
        let ids: Vec<u8> = self.header.players.iter().map(|p| p.id).collect();
        let blocks = self
            .blocks
            .iter()
            .map(|b| {
                let mut cmds = Vec::with_capacity(b.cmds.len());
                for c in b.cmds.iter().filter(|c| ids.contains(&c.player)) {
                    let c = cmd::to_legacy(c, self.unit_limit)?;
                    match cmd::openbw(c.op) {
                        cmd::OpenBw::Supported => cmds.push(c),
                        cmd::OpenBw::DropNoEffect => {}
                        // Kept for review instead of discarding the replay (user decision 2026-10-16).
                        cmd::OpenBw::Unhandled => anyhow::bail!("review: opcode {:#04x} at frame {} is not handled by OpenBW", c.op, c.frame),
                    }
                }
                Ok(cmd::Block { frame: b.frame, cmds })
            })
            .filter(|b: &Result<cmd::Block>| b.as_ref().map_or(true, |b| !b.cmds.is_empty()))
            .collect::<Result<Vec<_>>>()?;
        // 013 맵 CHK 정규화: SC:R map versions (STRx) become ones OpenBW's loader accepts.
        Ok(container::write_legacy(&self.raw_header, &cmd::encode(&blocks)?, &chk::to_legacy(&self.map)?))
    }

    /// 013-R7 / 013-AC4: the converted file parses back to exactly the converted commands
    /// (observer and no-effect commands dropped) with the same header and map.
    pub fn roundtrip_check(&self) -> Result<()> {
        let back = parse(&self.to_legacy_file()?)?;
        anyhow::ensure!(back.format == Format::Legacy, "converted file is {:?}", back.format);
        anyhow::ensure!(back.raw_header == self.raw_header, "header changed");
        anyhow::ensure!(back.map == chk::to_legacy(&self.map)?, "map changed");
        let ids: Vec<u8> = self.header.players.iter().map(|p| p.id).collect();
        let want: Vec<cmd::Cmd> = self
            .cmds()
            .filter(|c| ids.contains(&c.player))
            .map(|c| cmd::to_legacy(c, self.unit_limit))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .filter(|c| cmd::openbw(c.op) == cmd::OpenBw::Supported)
            .collect();
        let got: Vec<cmd::Cmd> = back.cmds().cloned().collect();
        anyhow::ensure!(got.len() == want.len(), "{} commands after round trip, expected {}", got.len(), want.len());
        if let Some((g, w)) = got.iter().zip(&want).find(|(g, w)| g != w) {
            anyhow::bail!("command differs at frame {}: {:?} vs {:?}", w.frame, g, w);
        }
        Ok(())
    }

    /// Players that actually played: humans with at least one train/build command
    /// (observers never issue them; same idea as screp's observer heuristic).
    pub fn active_players(&self) -> Vec<(Player, PlayerStats)> {
        let stats = self.player_stats();
        self.header.players.iter().cloned().zip(stats).filter(|(p, s)| p.kind == 2 && s.train_build > 0).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 013-R7: converting an SC:R command stream and parsing the result gives the same
    /// commands as converting command by command.
    #[test]
    fn conversion_round_trip_013_r7() {
        let mut h = vec![0u8; container::HEADER_SIZE];
        h[1..5].copy_from_slice(&100u32.to_le_bytes());
        h[0xa1 + 4] = 0; // player id 0
        h[0xa1 + 8] = 2; // human
        h[0xa1 + 9] = header::RACE_TERRAN;
        h[0xa1 + 11] = b'p';
        let blocks = vec![cmd::Block {
            frame: 3,
            cmds: vec![
                cmd::Cmd { frame: 3, player: 0, op: cmd::op::SELECT_121, data: vec![1, 7, 0, 0, 0] },
                cmd::Cmd { frame: 3, player: 0, op: cmd::op::TRAIN, data: vec![7, 0] },
            ],
        }];
        let rep = Replay { format: Format::Modern121, header: header::parse(&h), raw_header: h, blocks, map: vec![9; 10], unit_limit: None };
        let back = parse(&rep.to_legacy_file().unwrap()).unwrap();
        assert_eq!(back.format, Format::Legacy);
        let want: Vec<cmd::Cmd> = rep.cmds().map(|c| cmd::to_legacy(c, None).unwrap()).collect();
        assert_eq!(back.cmds().cloned().collect::<Vec<_>>(), want);
        assert_eq!(back.map, rep.map);
        assert_eq!(back.active_players().len(), 1);
        assert_eq!(back.player_stats()[0].cmds, 2);
    }
}

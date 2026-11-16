use std::collections::{HashSet, VecDeque};
use anyhow::Result;
use crate::shard::ShardGame;
use super::{StepSource, GameId};
use super::source::ShardSource;

pub const CHOSEN_PREFETCH_FILES: usize = 3;
pub const CHOSEN_PREFETCH_FILES_CUT: usize = 24;

pub struct ChosenGames {
    rx: std::sync::mpsc::Receiver<Result<ChosenFile>>,
    cur: VecDeque<(usize, ShardGame)>,
    cur_file: usize,
    last: Option<GameId>,
    thread: Option<std::thread::JoinHandle<()>>,
}

type ChosenFile = (usize, Vec<(usize, ShardGame)>);

fn panic_text(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned()).unwrap_or_else(|| "non-text panic".into())
}

pub fn cut_game(game: &mut ShardGame, until: u32) {
    let keep = game.steps.partition_point(|s| s.raw_frame.frame < until).max(1);
    game.steps.truncate(keep);
    game.steps.shrink_to_fit();
}

impl ChosenGames {
    pub fn new(src: ShardSource, ids: &[GameId]) -> Self {
        Self::with_until(src, ids, None)
    }

    pub fn with_until(src: ShardSource, ids: &[GameId], until: Option<u32>) -> Self {
        let ids: HashSet<GameId> = ids.iter().copied().collect();
        let files: std::collections::BTreeSet<usize> = ids.iter().map(|g| g.file).collect();
        let (tx, rx) = std::sync::mpsc::sync_channel(if until.is_some() { CHOSEN_PREFETCH_FILES_CUT } else { CHOSEN_PREFETCH_FILES });
        let thread = std::thread::spawn(move || {
            for k in files {
                let r = src.file_games(k).map(|games| {
                    let mut kept: Vec<(usize, ShardGame)> = games.into_iter().enumerate().filter(|(i, _)| ids.contains(&GameId { file: k, game: *i })).collect();
                    if let Some(f) = until {
                        kept.iter_mut().for_each(|(_, g)| cut_game(g, f));
                    }
                    (k, kept)
                });
                let failed = r.is_err();
                if tx.send(r).is_err() || failed {
                    return;
                }
            }
        });
        Self { rx, cur: VecDeque::new(), cur_file: 0, last: None, thread: Some(thread) }
    }
}

impl StepSource for ChosenGames {
    fn next_game(&mut self) -> Result<Option<ShardGame>> {
        while self.cur.is_empty() {
            let Ok(r) = self.rx.recv() else {
                if let Some(t) = self.thread.take() {
                    t.join().map_err(|p| anyhow::anyhow!("the chosen-games read-ahead thread panicked: {}", panic_text(&*p)))?;
                }
                return Ok(None);
            };
            let (k, games) = r?;
            self.cur_file = k;
            self.cur = games.into();
        }
        let (i, g) = self.cur.pop_front().expect("a game");
        self.last = Some(GameId { file: self.cur_file, game: i });
        Ok(Some(g))
    }

    fn last_id(&self) -> Option<GameId> {
        self.last
    }
}

#[cfg(test)]
mod chosen_tests {
    use super::*;

    #[test]
    fn chosen_games_read_ahead_panic_is_an_error_050_r9() {
        let (tx, rx) = std::sync::mpsc::sync_channel::<Result<ChosenFile>>(1);
        let thread = std::thread::spawn(move || {
            let _tx = tx;
            panic!("decode failed");
        });
        let mut c = ChosenGames { rx, cur: VecDeque::new(), cur_file: 0, last: None, thread: Some(thread) };
        let e = c.next_game().expect_err("a panic is an error");
        assert!(format!("{e:#}").contains("decode failed"));
    }
}

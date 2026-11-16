use std::io::Write;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};
use anyhow::{Context, Result};
use uranium_agent::{PolicySpec, T0Build, game_result};
use uranium_env::concede::{ConcedeGame, LeaveRule};
use uranium_env::reward::Outcome;
use uranium_env::scout::{PseudoMode, seconds};
use uranium_proto::control::{GameSpec, MapKey, PlayerKind, PlayerSpec};
use uranium_proto::{EndReason, Race};
use uranium_runner::{Job, JobOutcome, JobSource};
use crate::league::{AgentKind, Class, Opponent, Pick};
use crate::trajectory::{GameResult, Segment};
use super::{GameMeta, Shared};
use super::pool::prune_servers;
use super::hook::Hook;
use super::seed::game_seed;
use super::watch::bad_learning_session;
use super::csv::{slots_start_hit, start_columns, scout_columns};

pub(super) struct LeagueSource {
    pub(super) shared: Arc<Shared>,
    pub(super) hooks: Vec<Arc<Hook>>,
}

impl LeagueSource {
    fn job(&self, mut pick: Pick) -> Job {
        let s = &self.shared;
        let id = s.next_id.fetch_add(1, Ordering::SeqCst);
        let bridge = |race| PlayerSpec { race, kind: PlayerKind::Bridge { agent_addr: s.agent_addr.clone() }, name: None };
        let mut t0_builds = [T0Build::Standard; 2];
        let mut policies: [Option<PolicySpec>; 2] = [Some(PolicySpec::Learner(self.hooks[pick.agent.index()].clone())), None];
        let mut server = None;
        if matches!(pick.opponent, Opponent::Snapshot { .. } | Opponent::Anchor { .. }) {
            let got = s.snapshots.lock().unwrap().acquire(&pick.key);
            match got {
                Ok((h, server_id)) => {
                    policies[1] = Some(PolicySpec::Model(h));
                    server = Some(server_id);
                }
                Err(e) => {
                    let key = s.league.lock().unwrap().self_play_key();
                    uranium_store::elog!("[rl] game {id}: opponent {} cannot be served ({e:#}); playing the current Main instead (042)", pick.key);
                    pick = Pick { opponent: Opponent::SelfPlay, key, class: Class::Model, ..pick };
                }
            }
        }
        let (p1, enemy_race, opponent) = match &pick.opponent {
            Opponent::T0 { race, build } => {
                t0_builds[1] = *build;
                (bridge(*race), *race, format!("{}@{}", pick.key, env!("CARGO_PKG_VERSION")))
            }
            Opponent::T1 { name, race } => (PlayerSpec { race: *race, kind: PlayerKind::OssBot { name: name.clone() }, name: None }, *race, pick.key.clone()),
            Opponent::SelfPlay => {
                policies[1] = Some(PolicySpec::Model(s.handles[AgentKind::Main.index()].clone()));
                (bridge(Race::Terran), Race::Terran, pick.key.clone())
            }
            Opponent::Snapshot { .. } | Opponent::Anchor { .. } => (bridge(Race::Terran), Race::Terran, pick.key.clone()),
        };
        let players = [bridge(Race::Terran), p1];
        let random_players = players.iter().filter(|p| p.race == Race::Random).count() as u32;
        let share = if pick.class == Class::Model { None } else { s.league_cfg.start_mix.share_for(pick.ladder_group.as_deref()) };
        let seed = game_seed(id, share, random_players);
        let slots = uranium_runner::start_slots::start_slots(seed, random_players);
        let spec = GameSpec {
            game_id: id,
            map: if pick.fighting_spirit { MapKey::FightingSpirit } else { MapKey::CircuitBreaker },
            players,
            seed: Some(seed),
            max_frames: s.league_cfg.game_max_frames_of(pick.class == Class::Model, pick.ladder_group.as_deref()),
            privileged: true,
            save_replay: true,
            wall_timeout_s: s.game_wall_timeout_s,
        };
        s.games.lock().unwrap().insert(id, GameMeta { pick, self_player: None, enemy_race, server, slots: Some(slots) });
        if s.concede_cfg.enabled {
            s.concede.insert(id, Arc::new(ConcedeGame::new(s.concede_cfg)));
        }
        Job { matchup: 0, spec, t0_builds, models: [None, None], policies, opponent: Some(opponent) }
    }

    fn release_server(&self, meta: &GameMeta) {
        if let Some(id) = meta.server {
            self.shared.snapshots.lock().unwrap().release(id);
        }
        prune_servers(&self.shared);
    }

    fn collect(&self, game_id: u64) -> (Option<GameResult>, Option<Segment>) {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            {
                let rx = self.shared.results.lock().unwrap();
                let mut f = self.shared.finished.lock().unwrap();
                while let Ok(r) = rx.try_recv() {
                    f.insert(r.game_id, r);
                }
            }
            let r = self.shared.finished.lock().unwrap().remove(&game_id);
            if r.is_some() || Instant::now() > deadline {
                let seg = self.shared.held.lock().unwrap().remove(&game_id);
                return (r, seg);
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

impl JobSource for LeagueSource {
    fn next(&self) -> Option<Job> {
        if self.shared.stopping.load(Ordering::SeqCst) {
            return None;
        }
        let pick = self.shared.league.lock().unwrap().next();
        Some(self.job(pick))
    }

    fn requeue(&self, job: Job) {
        let meta = self.shared.games.lock().unwrap().remove(&job.spec.game_id);
        self.shared.concede.take(job.spec.game_id);
        if let Some(sil) = &self.shared.sil {
            sil.settle(job.spec.game_id, false, 0.0);
        }
        if let Some(m) = meta {
            self.release_server(&m);
        }
    }

    fn done(&self, job: &Job, outcome: JobOutcome, replay: Option<Vec<u8>>) {
        let s = &self.shared;
        let id = job.spec.game_id;
        let concede = s.concede.take(id);
        let Some(mut meta) = s.games.lock().unwrap().remove(&id) else { return };
        let agent = meta.pick.agent.index();
        if outcome != JobOutcome::Finished {
            s.queues[agent].revoke(id);
        }
        let (result, held) = self.collect(id);
        if outcome == JobOutcome::Finished {
            s.finished_games.fetch_add(1, Ordering::SeqCst);
            if result.is_none() {
                uranium_store::elog!("[rl] game {id}: finished without a report of the learning session; not counted for the no-decision streak (042-R20)");
            }
        }
        match bad_learning_session(outcome, result.as_ref()) {
            Some(true) => {
                s.bad_sessions.fetch_add(1, Ordering::SeqCst);
            }
            Some(false) => s.bad_sessions.store(0, Ordering::SeqCst),
            None => {}
        }
        if matches!(meta.enemy_race, Race::Random | Race::Unknown) {
            if let Some(r) = result.as_ref().and_then(|r| r.enemy_race) {
                meta.enemy_race = r;
            }
        }
        let ends = s.ends.take(id);
        let left = concede.and_then(|g| g.conceded());
        let conceded = left.filter(|c| c.rule == LeaveRule::Concede);
        let verdict = (outcome == JobOutcome::Finished).then(|| game_result(&ends)).flatten();
        let decided = match (&verdict, meta.self_player) {
            (Some((e, disagree)), Some(me)) => {
                if *disagree {
                    s.disagree.fetch_add(1, Ordering::SeqCst);
                }
                match e.reason {
                    EndReason::Aborted | EndReason::AgentTimeout => None,
                    EndReason::TimeLimitDraw => Some(Outcome::Draw),
                    _ => Some(match e.winner {
                        Some(w) if w == me => Outcome::Win,
                        Some(_) => Outcome::Loss,
                        None => Outcome::Draw,
                    }),
                }
            }
            _ => None,
        };
        let timed_out = verdict.as_ref().is_some_and(|(e, _)| e.reason == EndReason::TimeLimitDraw);
        let mut judge_scv = None;
        let decided = match decided {
            Some(o) if timed_out && s.league_cfg.rush_judged(meta.pick.class == Class::Model, meta.pick.ladder_group.as_deref()) => {
                let judge = &s.league_cfg.rush_judge;
                match result.as_ref().map(|r| r.scout.and_then(|m| m.scv_last)) {
                    Some(Some(n)) => {
                        let judged = judge.judge(n);
                        judge_scv = Some(n);
                        uranium_store::elog!("[rl] game {id}: 042-R17 rush judge at {} s: {n} own SCVs (win at >= {}) -> {judged:?} (bridge {o:?})", judge.at_s, judge.min_scv);
                        Some(judged)
                    }
                    Some(None) => {
                        uranium_store::elog!("[rl] game {id}: 042-R17 rush judge at {} s: no observation of the learning session to count SCVs; no result, not trained (042-R3)", judge.at_s);
                        None
                    }
                    None => {
                        uranium_store::elog!("[rl] game {id}: 042-R17 rush judge at {} s: no session report of the learning session (timeout); no result, not trained (042-R3)", judge.at_s);
                        None
                    }
                }
            }
            other => other,
        };
        let r_win = match decided {
            Some(Outcome::Win) => 1.0,
            Some(Outcome::Loss) => -1.0,
            _ => 0.0,
        };
        match (decided, held) {
            (Some(o), Some(mut seg)) => {
                if let Some(last) = seg.steps.last_mut() {
                    last.reward[0] += r_win;
                }
                if let Some(sil) = &s.sil
                    && meta.pick.agent == AgentKind::Main
                    && meta.pick.ladder_group.as_deref() == Some(sil.cfg.select_group.as_str())
                    && sil.cfg.may_select(&meta.pick.key, slots_start_hit(meta.slots))
                {
                    sil.keep(&seg);
                }
                s.queues[agent].push(seg);
                s.queues[agent].finish(id);
                let stage_up = s.league.lock().unwrap().record(&meta.pick, meta.enemy_race, o);
                if stage_up {
                    stage_advanced(s);
                }
            }
            (Some(o), None) => {
                s.queues[agent].finish(id);
                if s.league.lock().unwrap().record(&meta.pick, meta.enemy_race, o) {
                    stage_advanced(s);
                }
            }
            (None, _) => {
                s.queues[agent].revoke(id);
                s.aborted.fetch_add(1, Ordering::SeqCst);
            }
        }
        let end_reason = match (&verdict, decided) {
            (_, None) => String::new(),
            (_, Some(_)) if judge_scv.is_some() => "RushJudge".to_string(),
            (_, Some(_)) if conceded.is_some() => "Concede".to_string(),
            (Some((e, _)), Some(_)) => format!("{:?}", e.reason),
            (None, Some(_)) => String::new(),
        };
        if let Some(c) = conceded {
            let who = if c.side == 0 { "learning agent" } else { "opponent" };
            uranium_store::elog!("[rl] game {id}: 042-R14 collapse concede by the {who} side (players index {}) at frame {}; result {:?}", c.side, c.frame, decided);
        }
        if let Some(sil) = &s.sil {
            let scout = result.as_ref().and_then(|r| r.scout);
            let scv_4m = scout.and_then(|m| m.scv_4m);
            let army_s = scout.and_then(|m| m.army_frame).map(seconds);
            let selected = meta.pick.agent == AgentKind::Main && sil.cfg.selects(meta.pick.ladder_group.as_deref(), decided, scv_4m, conceded.is_some()) && sil.cfg.passes_filters(&meta.pick.key, army_s, slots_start_hit(meta.slots));
            sil.settle(id, selected, r_win);
        }
        let disagree = verdict.as_ref().is_some_and(|v| v.1);
        let (replay_path, replay_reason) = match (replay_reason(id, disagree, s.replay_every), replay) {
            (Some(why), Some(bytes)) => match keep_replay(&s.replay_dir, id, why, &bytes) {
                Ok(name) => (format!("replays/{name}"), why),
                Err(e) => {
                    uranium_store::elog!("[rl] replay of game {id} not kept: {e:#}");
                    (String::new(), "")
                }
            },
            _ => (String::new(), ""),
        };
        let (sum, decisions) = result.as_ref().map_or(([0.0; 4], 0), |r| (r.reward_sum, r.decisions));
        s.agent_steps.fetch_add(decisions, Ordering::SeqCst);
        s.games_done.fetch_add(1, Ordering::SeqCst);
        let extra = scout_columns(s.pseudo, result.as_ref(), meta.pick.ladder_group.as_deref());
        let start = start_columns(meta.slots, result.as_ref());
        let line = format!(
            "{},{},{},{},{:?},{},{},{:?},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            id,
            AgentKind::ALL[agent].name(),
            job.opponent.clone().unwrap_or_default(),
            match meta.pick.class {
                Class::T0 => "t0",
                Class::T1 => "t1",
                Class::Model => "model",
            },
            meta.enemy_race,
            if meta.pick.fighting_spirit { "fighting_spirit" } else { "circuit_breaker" },
            outcome.as_str(),
            decided,
            disagree,
            decisions,
            r_win + sum[0],
            sum[1],
            sum[2],
            sum[3],
            replay_path,
            replay_reason,
            end_reason,
            extra,
            judge_scv.map_or(String::new(), |n| n.to_string()),
            start
        );
        let _ = s.games_csv.lock().unwrap().write_all(line.as_bytes());
        self.release_server(&meta);
    }
}

fn stage_advanced(s: &Shared) {
    let stage = s.league.lock().unwrap().stage;
    uranium_store::elog!("[rl] curriculum advanced to {stage:?} (042-R5)");
    if let Some(w) = scout_stage_warning(s.pseudo, stage) {
        uranium_store::elog!("{w}");
    }
}

pub(super) fn scout_stage_warning(mode: PseudoMode, stage: crate::league::Stage) -> Option<&'static str> {
    (mode == PseudoMode::Scout && stage == crate::league::Stage::C2).then_some("[rl] WARNING: stage C2 with [reward] pseudo = \"scout\": no C2 decay is applied to r_scout / r_survive (033-R11: to be decided before C2)")
}

fn replay_reason(game_id: u64, disagree: bool, every: u64) -> Option<&'static str> {
    if disagree {
        Some("disagree")
    } else if every > 0 && game_id % every == 0 {
        Some("sample")
    } else {
        None
    }
}

fn keep_replay(dir: &Path, game_id: u64, reason: &str, bytes: &[u8]) -> Result<String> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let name = format!("g{game_id}-{reason}.rep");
    let tmp = dir.join(format!("{name}.part"));
    std::fs::write(&tmp, bytes).with_context(|| format!("writing {}", tmp.display()))?;
    if std::fs::read(&tmp)? != bytes {
        let _ = std::fs::remove_file(&tmp);
        anyhow::bail!("{} reads back different bytes", tmp.display());
    }
    std::fs::rename(&tmp, dir.join(&name))?;
    Ok(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::path::PathBuf;
    use uranium_agent::model_policy::LearnerHook;
    use uranium_env::scout::ScoutMetrics;
    use crate::config::RlConfig;
    use crate::sil::SilConfig;
    use crate::run::watch::BAD_SESSION_STREAK;
    use crate::run::csv::{GAMES_HEADER, open_csv};
    use crate::run::testkit::{fake_servers, source, source_cfg, opponent_handle, result};

    #[test]
    fn model_game_time_cap_042_r15() {
        let lc = |s: u32| crate::config::LeagueConfig { model_game_max_s: s, ..RlConfig::default_file().league };
        assert_eq!((lc(0).game_max_frames(true), lc(0).game_max_frames(false)), (86_400, 86_400));
        assert_eq!((lc(1_500).game_max_frames(true), lc(1_500).game_max_frames(false)), (36_000, 86_400));
        assert_eq!(lc(3_600).game_max_frames(true), 86_400);
        assert_eq!(lc(60).game_max_frames(true), 1_440);
        let frames = |cap: u32, c1: [f64; 3], t0: f64, fail: bool| {
            let mut cfg = RlConfig::default_file();
            cfg.curriculum.start_stage = crate::league::Stage::C1;
            let cc = &mut cfg.curriculum;
            (cc.c1_t0, cc.c1_t1, cc.c1_self, cc.c1_snapshot, cc.c1_anchor) = (t0, 0.0, c1[0], c1[1], c1[2]);
            cfg.league.anchor = "checkpoints/x/ckpt-9".into();
            cfg.league.model_game_max_s = cap;
            cfg.validate().unwrap();
            let csv = PathBuf::from(if cfg!(windows) { "NUL" } else { "/dev/null" });
            let src = source_cfg(cfg, fake_servers(fail).0, &csv);
            (0..30)
                .map(|_| {
                    let j = src.next().unwrap();
                    let class = src.shared.games.lock().unwrap()[&j.spec.game_id].pick.class;
                    (class, j.spec.max_frames)
                })
                .collect::<Vec<_>>()
        };
        for (c1, fail) in [([1.0, 0.0, 0.0], false), ([0.0, 0.0, 1.0], false), ([0.0, 0.0, 1.0], true), ([0.0, 1.0, 0.0], false)] {
            assert!(frames(1_500, c1, 0.0, fail).iter().all(|&(c, f)| c == Class::Model && f == 36_000), "{c1:?} {fail}");
            assert!(frames(0, c1, 0.0, fail).iter().all(|&(_, f)| f == 86_400));
        }
        let mixed = frames(1_500, [0.5, 0.0, 0.0], 0.5, false);
        assert!(mixed.iter().any(|&(c, _)| c == Class::T0) && mixed.iter().any(|&(c, _)| c == Class::Model));
        for (c, f) in mixed {
            assert_eq!(f, if c == Class::Model { 36_000 } else { 86_400 }, "{c:?}");
        }
    }

    #[test]
    fn replay_reason_samples_and_disagreements_042_r11() {
        assert_eq!(replay_reason(200, false, 100), Some("sample"));
        assert_eq!(replay_reason(201, false, 100), None);
        assert_eq!(replay_reason(201, true, 100), Some("disagree"));
        assert_eq!(replay_reason(200, true, 100), Some("disagree"));
        assert_eq!(replay_reason(200, false, 0), None, "0 = no regular sample");
        let kept = (1_000u64..2_000).filter(|id| replay_reason(*id, false, 100).is_some()).count();
        assert_eq!(kept, 10);
    }

    #[test]
    fn keep_replay_writes_verified_file_042_r11() {
        let dir = std::env::temp_dir().join(format!("uranium-rl-rep-{}", std::process::id()));
        let name = keep_replay(&dir, 7, "disagree", b"replay bytes").unwrap();
        assert_eq!(name, "g7-disagree.rep");
        assert_eq!(std::fs::read(dir.join(&name)).unwrap(), b"replay bytes");
        assert!(!dir.join("g7-disagree.rep.part").exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn unservable_model_opponent_becomes_self_play_042() {
        let (servers, _, _) = fake_servers(true);
        let src = source([0.0, 0.0, 1.0], "checkpoints/x/ckpt-9", 4, servers);
        for _ in 0..3 {
            let job = src.next().expect("a game, never the end of the slot");
            assert_eq!(job.opponent.as_deref(), Some("uranium-main@r1"));
            assert_eq!(opponent_handle(&job), "main", "the current Main's server");
            let games = src.shared.games.lock().unwrap();
            let m = &games[&job.spec.game_id];
            assert_eq!((&m.pick.opponent, m.pick.class, m.server), (&Opponent::SelfPlay, Class::Model, None));
        }
        let (servers, opened, _) = fake_servers(false);
        let src = source([0.0, 1.0, 0.0], "", 4, servers);
        {
            let mut l = src.shared.league.lock().unwrap();
            let id = l.add_snapshot(AgentKind::Main);
            l.mark_published(&id);
        }
        let job = src.next().unwrap();
        assert_eq!(job.opponent.as_deref(), Some("uranium-main@r1"));
        assert!(opened.lock().unwrap().is_empty());
    }

    #[test]
    fn concede_games_in_the_run_loop_042_r14() {
        use uranium_proto::EndReason;
        let dir = std::env::temp_dir().join(format!("uranium-rl-concede-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let csv = dir.join("games.csv");
        drop(open_csv(&csv, GAMES_HEADER).unwrap());
        let mut cfg = RlConfig::default_file();
        cfg.curriculum.start_stage = crate::league::Stage::C1;
        let cc = &mut cfg.curriculum;
        (cc.c1_t0, cc.c1_t1, cc.c1_self) = (0.0, 0.0, 1.0);
        cfg.concede.enabled = true;
        let (servers, _, _) = fake_servers(false);
        let src = source_cfg(cfg.clone(), servers, &csv);
        let play = |ends: &[(u8, u8, EndReason)], conceded: Option<u8>, outcome: JobOutcome| {
            let job = src.next().unwrap();
            let id = job.spec.game_id;
            let game = src.shared.concede.get(id).expect("on the board from dispatch");
            if let Some(side) = conceded {
                assert!(game.claim(side, 7_200, LeaveRule::Concede));
            }
            src.shared.games.lock().unwrap().get_mut(&id).unwrap().self_player = Some(0);
            for &(slot, winner, reason) in ends {
                src.shared.ends.push(id, uranium_agent::SessionEnd { slot, winner: Some(winner), frame: 7_210 + slot as u32, reason });
            }
            let result = match ends.iter().find(|e| e.0 == 0) {
                Some(&(_, w, _)) if outcome == JobOutcome::Finished => Some(if w == 0 { Outcome::Win } else { Outcome::Loss }),
                _ => None,
            };
            src.hooks[0].tx.send(GameResult { agent: 0, game_id: id, outcome: result, reward_sum: [0.0; 4], scout_sum: 0.0, survive_sum: 0.0, scout: None, decisions: 5, enemy_race: Some(Race::Terran) }).unwrap();
            src.done(&job, outcome, None);
            assert!(src.shared.concede.get(id).is_none(), "off the board once done");
            std::fs::read_to_string(&csv).unwrap().lines().last().unwrap().to_string()
        };
        let row = |line: &str| line.split(',').map(str::to_string).collect::<Vec<_>>();
        let r = row(&play(&[(0, 1, EndReason::Surrender), (1, 1, EndReason::Surrender)], Some(0), JobOutcome::Finished));
        assert_eq!((r[7].as_str(), r[10].as_str(), r[16].as_str()), ("Some(Loss)", "-1", "Concede"), "{r:?}");
        let r = row(&play(&[(1, 0, EndReason::Surrender), (0, 0, EndReason::Surrender)], Some(1), JobOutcome::Finished));
        assert_eq!((r[7].as_str(), r[10].as_str(), r[16].as_str()), ("Some(Win)", "1", "Concede"), "{r:?}");
        let r = row(&play(&[(0, 0, EndReason::Elimination), (1, 0, EndReason::Elimination)], None, JobOutcome::Finished));
        assert_eq!((r[7].as_str(), r[16].as_str()), ("Some(Win)", "Elimination"));
        let r = row(&play(&[(0, 1, EndReason::Surrender)], Some(0), JobOutcome::Aborted));
        assert_eq!((r[7].as_str(), r[16].as_str()), ("None", ""));
        assert_eq!(r.len(), GAMES_HEADER.trim().split(',').count());
        let job = src.next().unwrap();
        let id = job.spec.game_id;
        src.requeue(job);
        assert!(src.shared.concede.get(id).is_none());
        cfg.concede.enabled = false;
        let (servers, _, _) = fake_servers(false);
        let off = source_cfg(cfg, servers, &csv);
        let job = off.next().unwrap();
        assert!(off.shared.concede.get(job.spec.game_id).is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn bad_session_streak_in_the_run_loop_042_r20() {
        use uranium_proto::EndReason;
        let mut cfg = RlConfig::default_file();
        cfg.curriculum.start_stage = crate::league::Stage::C1;
        let cc = &mut cfg.curriculum;
        (cc.c1_t0, cc.c1_t1, cc.c1_self) = (0.0, 0.0, 1.0);
        let (servers, _, _) = fake_servers(false);
        let csv = PathBuf::from(if cfg!(windows) { "NUL" } else { "/dev/null" });
        let src = source_cfg(cfg, servers, &csv);
        let play = |decisions: u64, outcome: JobOutcome| {
            let job = src.next().unwrap();
            let id = job.spec.game_id;
            src.shared.games.lock().unwrap().get_mut(&id).unwrap().self_player = Some(0);
            for slot in [0u8, 1] {
                src.shared.ends.push(id, uranium_agent::SessionEnd { slot, winner: Some(0), frame: 100 + slot as u32, reason: EndReason::Elimination });
            }
            src.hooks[0].tx.send(GameResult { agent: 0, game_id: id, outcome: None, reward_sum: [0.0; 4], scout_sum: 0.0, survive_sum: 0.0, scout: None, decisions, enemy_race: Some(Race::Terran) }).unwrap();
            src.done(&job, outcome, None);
            src.shared.bad_sessions.load(Ordering::SeqCst)
        };
        assert_eq!(play(0, JobOutcome::Finished), 1);
        assert_eq!(play(0, JobOutcome::Finished), 2);
        assert_eq!(play(0, JobOutcome::Aborted), 2, "not finished: neither");
        assert_eq!(play(7, JobOutcome::Finished), 0, "a decision resets");
        for k in 1..=BAD_SESSION_STREAK {
            assert_eq!(play(0, JobOutcome::Finished), k);
        }
        assert_eq!(src.shared.finished_games.load(Ordering::SeqCst), 3 + BAD_SESSION_STREAK);
        assert_eq!(src.shared.games_done.load(Ordering::SeqCst), 4 + BAD_SESSION_STREAK);
    }

    #[test]
    fn sil_buffer_takes_selected_games_042_r16() {
        use crate::league::{LadderEntry, LadderKind};
        use uranium_agent::T0Build;
        use uranium_env::features::executor::HostAction;
        use uranium_proto::EndReason;
        let csv = PathBuf::from(if cfg!(windows) { "NUL" } else { "/dev/null" });
        for until_s in [0u32, 2] {
            let mut cfg = RlConfig::default_file();
            cfg.curriculum.start_stage = crate::league::Stage::C1;
            let cc = &mut cfg.curriculum;
            (cc.c1_t0, cc.c1_t1, cc.c1_self, cc.c1_ladder) = (0.0, 0.0, 0.0, 1.0);
            cc.ladder_groups = [("rush".to_string(), 0.5), ("macro".to_string(), 0.5)].into_iter().collect();
            let entry = |build, group: &str| LadderEntry { kind: LadderKind::T0, build: Some(build), name: None, race: Race::Zerg, group: group.into(), weight: None };
            cfg.ladder = vec![entry(T0Build::ZergRush, "rush"), entry(T0Build::Standard, "macro")];
            cfg.sil = SilConfig { enabled: true, until_s, lanes: 2, buffer_segments: 1_000, ..SilConfig::default() };
            cfg.validate().unwrap();
            let src = source_cfg(cfg, fake_servers(false).0, &csv);
            let info = crate::testkit::game_info();
            let obs = crate::testkit::observations(10);
            let cases = [(Outcome::Win, None, false), (Outcome::Loss, Some(12), false), (Outcome::Loss, Some(5), false), (Outcome::Win, Some(20), true)];
            let (mut want, mut groups) = (0usize, BTreeSet::new());
            for g in 0..24 {
                let (outcome, scv_4m, aborted) = cases[g % cases.len()];
                let job = src.next().unwrap();
                let id = job.spec.game_id;
                let group = src.shared.games.lock().unwrap()[&id].pick.ladder_group.clone();
                groups.insert(group.clone());
                let (_, mut o) = src.hooks[0].session(id, &info);
                let a = HostAction { action: 0, delay_bucket: 0, queued: false, selected: vec![], target_unit: None, target_loc: None };
                for (i, ob) in obs.iter().enumerate() {
                    let state = o.wants_state().then(uranium_agent::infer::HostState::default);
                    o.on_decision(&uranium_agent::model_policy::Decision { frame: 10 * i as u32, obs: ob, action: &a, log_prob: 0.0, head_log_prob: [0.0; uranium_agent::infer::HEADS], value: [0.0; 4], rejected: false, version: 0, reward_before: Default::default(), state });
                }
                o.on_end(Default::default(), Some(outcome));
                src.shared.results.lock().unwrap().try_iter().for_each(drop);
                src.shared.games.lock().unwrap().get_mut(&id).unwrap().self_player = Some(0);
                let winner = if outcome == Outcome::Win { 0 } else { 1 };
                for slot in [0u8, 1] {
                    src.shared.ends.push(id, uranium_agent::SessionEnd { slot, winner: Some(winner), frame: 9_000, reason: EndReason::Elimination });
                }
                let m = ScoutMetrics { scv_4m, ..Default::default() };
                src.hooks[0].tx.send(GameResult { outcome: Some(outcome), ..result(id, Some(m)) }).unwrap();
                src.done(&job, if aborted { JobOutcome::Aborted } else { JobOutcome::Finished }, None);
                let selected = group.as_deref() == Some("rush") && !aborted && (outcome == Outcome::Win || scv_4m.is_some_and(|n| n >= 12));
                if selected {
                    want += if until_s == 0 { 4 } else { 2 };
                }
            }
            assert!(groups.contains(&Some("rush".to_string())) && groups.contains(&Some("macro".to_string())), "{groups:?}");
            let sil = src.shared.sil.as_ref().unwrap();
            assert!(sil.pending.lock().unwrap().is_empty(), "nothing left pending");
            let mut buf = sil.buffer.lock().unwrap();
            assert!(want > 0);
            assert_eq!(buf.len(), want, "until_s {until_s}");
            let segs = buf.sample(want, crate::sil::SilMode::Sil, 4);
            assert_eq!(segs.iter().filter(|s| s.terminal).count(), if until_s == 0 { want / 4 } else { 0 });
            for s in &segs {
                let r = s.sil_return.unwrap();
                assert!(r == 1.0 || r == -1.0, "{r}");
                if until_s > 0 {
                    assert!(s.start_frame < until_s * 24, "{}", s.start_frame);
                }
                if s.terminal {
                    assert_eq!(s.steps.last().unwrap().reward[0], r, "terminal copy carries r_win");
                }
            }
        }
    }

    #[test]
    fn sil_opponent_and_contact_filters_in_the_run_loop_042_r16() {
        for select_start_hit in [None, Some(true)] {
            sil_filters_in_the_run_loop(select_start_hit);
        }
    }

    fn sil_filters_in_the_run_loop(select_start_hit: Option<bool>) {
        use crate::league::{LadderEntry, LadderKind};
        use uranium_agent::T0Build;
        use uranium_env::features::executor::HostAction;
        use uranium_proto::EndReason;
        let csv = PathBuf::from(if cfg!(windows) { "NUL" } else { "/dev/null" });
        let mut cfg = RlConfig::default_file();
        cfg.curriculum.start_stage = crate::league::Stage::C1;
        let cc = &mut cfg.curriculum;
        (cc.c1_t0, cc.c1_t1, cc.c1_self, cc.c1_ladder) = (0.0, 0.0, 0.0, 1.0);
        cc.ladder_groups = [("rush".to_string(), 0.8), ("macro".to_string(), 0.2)].into_iter().collect();
        let entry = |build, group: &str| LadderEntry { kind: LadderKind::T0, build: Some(build), name: None, race: Race::Zerg, group: group.into(), weight: None };
        cfg.ladder = vec![entry(T0Build::ZergRush, "rush"), entry(T0Build::ZergMuta, "rush"), entry(T0Build::Standard, "macro")];
        cfg.sil = SilConfig { enabled: true, until_s: 0, lanes: 2, buffer_segments: 1_000, wins_only: true, select_keys: Some(vec!["t0-zerg-rush".into()]), select_army_before_s: Some(225.0), select_start_hit, ..SilConfig::default() };
        cfg.validate().unwrap();
        let src = source_cfg(cfg, fake_servers(false).0, &csv);
        let info = crate::testkit::game_info();
        let obs = crate::testkit::observations(10);
        let armies = [Some(5_400u32), Some(5_401), None, Some(2_000)];
        let (mut want, mut keys, mut misses) = (0usize, BTreeSet::new(), 0);
        for g in 0..120 {
            let army_frame = armies[g % armies.len()];
            let job = src.next().unwrap();
            let id = job.spec.game_id;
            let (key, hit) = {
                let games = src.shared.games.lock().unwrap();
                (games[&id].pick.key.clone(), slots_start_hit(games[&id].slots))
            };
            assert!(hit.is_some(), "every league game has start slots");
            let start_ok = select_start_hit.is_none_or(|w| hit == Some(w));
            keys.insert(key.clone());
            let (_, mut o) = src.hooks[0].session(id, &info);
            let a = HostAction { action: 0, delay_bucket: 0, queued: false, selected: vec![], target_unit: None, target_loc: None };
            for (i, ob) in obs.iter().enumerate() {
                let state = o.wants_state().then(uranium_agent::infer::HostState::default);
                o.on_decision(&uranium_agent::model_policy::Decision { frame: 10 * i as u32, obs: ob, action: &a, log_prob: 0.0, head_log_prob: [0.0; uranium_agent::infer::HEADS], value: [0.0; 4], rejected: false, version: 0, reward_before: Default::default(), state });
            }
            o.on_end(Default::default(), Some(Outcome::Win));
            if key == "t0-zerg-rush" && !start_ok {
                misses += 1;
                assert!(!src.shared.sil.as_ref().unwrap().pending.lock().unwrap().contains_key(&id), "a miss keeps no copies");
            }
            src.shared.results.lock().unwrap().try_iter().for_each(drop);
            src.shared.games.lock().unwrap().get_mut(&id).unwrap().self_player = Some(0);
            for slot in [0u8, 1] {
                src.shared.ends.push(id, uranium_agent::SessionEnd { slot, winner: Some(0), frame: 9_000, reason: EndReason::Elimination });
            }
            src.hooks[0].tx.send(result(id, Some(ScoutMetrics { army_frame, ..Default::default() }))).unwrap();
            src.done(&job, JobOutcome::Finished, None);
            if key == "t0-zerg-rush" && army_frame.is_some_and(|f| f <= 5_400) && start_ok {
                want += 4;
            }
        }
        assert!(keys.len() == 3, "{keys:?}");
        assert_eq!(misses > 0, select_start_hit.is_some(), "{misses} listed-opponent misses");
        let sil = src.shared.sil.as_ref().unwrap();
        assert!(sil.pending.lock().unwrap().is_empty(), "nothing left pending");
        assert!(want > 0);
        assert_eq!(sil.buffer.lock().unwrap().len(), want);
    }

    #[test]
    fn scout_c2_warning_033_r11() {
        use crate::league::Stage;
        assert!(scout_stage_warning(PseudoMode::Scout, Stage::C2).is_some_and(|w| w.contains("WARNING")));
        for (m, st) in [(PseudoMode::Scout, Stage::C1), (PseudoMode::Scout, Stage::C0), (PseudoMode::Z, Stage::C2)] {
            assert!(scout_stage_warning(m, st).is_none(), "{m:?} {st:?}");
        }
    }

    #[test]
    fn sessions_and_games_csv_in_scout_mode_042_amb_042_16() {
        use crate::league::{LadderEntry, LadderKind};
        use uranium_agent::T0Build;
        use uranium_env::features::executor::HostAction;
        use uranium_proto::EndReason;
        let dir = std::env::temp_dir().join(format!("uranium-rl-scout-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let csv = dir.join("games.csv");
        drop(open_csv(&csv, GAMES_HEADER).unwrap());
        let mut cfg = RlConfig::default_file();
        cfg.curriculum.start_stage = crate::league::Stage::C1;
        let cc = &mut cfg.curriculum;
        (cc.c1_t0, cc.c1_t1, cc.c1_self, cc.c1_ladder) = (0.0, 0.0, 0.5, 0.5);
        cc.ladder_groups = [("rush".to_string(), 0.5), ("macro".to_string(), 0.5)].into_iter().collect();
        let entry = |build, group: &str| LadderEntry { kind: LadderKind::T0, build: Some(build), name: None, race: Race::Zerg, group: group.into(), weight: None };
        cfg.ladder = vec![entry(T0Build::ZergRush, "rush"), entry(T0Build::Standard, "macro")];
        cfg.reward.pseudo = PseudoMode::Scout;
        cfg.validate().unwrap();
        let (servers, _, _) = fake_servers(false);
        let src = source_cfg(cfg, servers, &csv);
        assert_eq!(src.hooks[0].reward_config().pseudo, PseudoMode::Scout);
        let info = crate::testkit::game_info();
        let obs = crate::testkit::observations(2);
        let (mut model, mut bot) = (0, 0);
        for _ in 0..40 {
            let job = src.next().unwrap();
            let id = job.spec.game_id;
            let (class, group) = {
                let g = src.shared.games.lock().unwrap();
                let p = &g.get(&id).unwrap().pick;
                (p.class, p.ladder_group.clone())
            };
            let (z, mut o) = src.hooks[0].session(id, &info);
            assert!(z.is_none());
            let a = HostAction { action: 0, delay_bucket: 0, queued: false, selected: vec![], target_unit: None, target_loc: None };
            for (i, ob) in obs.iter().enumerate() {
                let state = o.wants_state().then(uranium_agent::infer::HostState::default);
                o.on_decision(&uranium_agent::model_policy::Decision { frame: i as u32, obs: ob, action: &a, log_prob: 0.0, head_log_prob: [0.0; uranium_agent::infer::HEADS], value: [0.0; 4], rejected: false, version: 0, reward_before: Default::default(), state });
            }
            o.on_end(Default::default(), Some(Outcome::Win));
            let seg = src.shared.held.lock().unwrap().get(&id).map(|s| s.pseudo_scale).expect("held terminal segment");
            src.shared.results.lock().unwrap().try_iter().for_each(drop);
            if class == Class::Model {
                assert_eq!(seg, [0.5, 0.0], "model game: vs_model");
                model += 1;
            } else {
                assert_eq!(seg, [1.0, 1.0], "bot game");
                bot += 1;
            }
            src.shared.games.lock().unwrap().get_mut(&id).unwrap().self_player = Some(0);
            for slot in [0u8, 1] {
                src.shared.ends.push(id, uranium_agent::SessionEnd { slot, winner: Some(0), frame: 9_000, reason: EndReason::Elimination });
            }
            src.hooks[0].tx.send(result(id, Some(ScoutMetrics { main_frame: Some(2_400), types_4m: 1, types_6m: 3, ..Default::default() }))).unwrap();
            src.done(&job, JobOutcome::Finished, None);
            let line = std::fs::read_to_string(&csv).unwrap().lines().last().unwrap().to_string();
            let r: Vec<&str> = line.split(',').collect();
            assert_eq!(r.len(), GAMES_HEADER.trim().split(',').count(), "{line}");
            assert_eq!(&r[17..27], [&["0.6", "0.375", group.as_deref().unwrap_or("")][..], &["100.0", "1", "3", "", "", "", ""]].concat(), "{line}");
            let [a, b] = uranium_runner::start_slots::start_slots(job.spec.seed.unwrap(), 0);
            let hit = if uranium_runner::start_slots::start_hit(a, b) { "1" } else { "0" };
            assert_eq!(&r[27..], [a.to_string().as_str(), b.to_string().as_str(), hit, "", ""], "{line}");
            if let Some(g) = &group {
                assert_eq!(r[2].starts_with("t0-zerg-rush"), g == "rush", "{line}");
            }
        }
        assert!(model > 0 && bot > 0, "{model} {bot}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rush_judge_results_in_the_run_loop_042_r17() {
        use crate::config::RushJudgeConfig;
        use crate::league::{LadderEntry, LadderKind};
        use uranium_agent::T0Build;
        use uranium_env::features::executor::HostAction;
        let dir = std::env::temp_dir().join(format!("uranium-rl-rushjudge-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let csv = dir.join("games.csv");
        drop(open_csv(&csv, GAMES_HEADER).unwrap());
        let mut cfg = RlConfig::default_file();
        cfg.curriculum.start_stage = crate::league::Stage::C1;
        let cc = &mut cfg.curriculum;
        (cc.c1_t0, cc.c1_t1, cc.c1_self, cc.c1_ladder) = (0.0, 0.0, 0.0, 1.0);
        cc.ladder_groups = [("rush".to_string(), 0.5), ("macro".to_string(), 0.5)].into_iter().collect();
        let entry = |build, group: &str| LadderEntry { kind: LadderKind::T0, build: Some(build), name: None, race: Race::Zerg, group: group.into(), weight: None };
        cfg.ladder = vec![entry(T0Build::ZergRush, "rush"), entry(T0Build::Standard, "macro")];
        cfg.league.win_rate_decay = 0.9;
        cfg.league.rush_judge = RushJudgeConfig { groups: vec!["rush".into()], ..RushJudgeConfig::default() };
        cfg.sil = SilConfig { enabled: true, wins_only: true, lanes: 2, buffer_segments: 100, ..SilConfig::default() };
        cfg.concede.enabled = true;
        cfg.validate().unwrap();
        let src = source_cfg(cfg, fake_servers(false).0, &csv);
        let info = crate::testkit::game_info();
        let obs = crate::testkit::observations(2);
        struct Played {
            row: Vec<String>,
            win_delta: f64,
            terminal_r_win: Option<f32>,
            buffer: usize,
            aborted: u64,
        }
        let play = |group: &str, decisions: usize, reason: EndReason, winner: Option<u8>, scout: ScoutMetrics, concede: bool| {
            let job = loop {
                let j = src.next().unwrap();
                if src.shared.games.lock().unwrap()[&j.spec.game_id].pick.ladder_group.as_deref() == Some(group) {
                    break j;
                }
                src.requeue(j);
            };
            let id = job.spec.game_id;
            let key = src.shared.games.lock().unwrap()[&id].pick.key.clone();
            let before = src.shared.league.lock().unwrap().win_rate(AgentKind::Main, &key);
            let (_, mut o) = src.hooks[0].session(id, &info);
            let a = HostAction { action: 0, delay_bucket: 0, queued: false, selected: vec![], target_unit: None, target_loc: None };
            for (i, ob) in obs.iter().take(decisions).enumerate() {
                let state = o.wants_state().then(uranium_agent::infer::HostState::default);
                o.on_decision(&uranium_agent::model_policy::Decision { frame: 24 * i as u32, obs: ob, action: &a, log_prob: 0.0, head_log_prob: [0.0; uranium_agent::infer::HEADS], value: [0.0; 4], rejected: false, version: 0, reward_before: Default::default(), state });
            }
            let session_view = match winner {
                Some(0) => Outcome::Win,
                Some(_) => Outcome::Loss,
                None => Outcome::Draw,
            };
            o.on_end(Default::default(), Some(session_view));
            src.shared.results.lock().unwrap().try_iter().for_each(drop);
            src.shared.games.lock().unwrap().get_mut(&id).unwrap().self_player = Some(0);
            if concede {
                assert!(src.shared.concede.get(id).unwrap().claim(0, 7_200, LeaveRule::Concede));
            }
            for slot in [0u8, 1] {
                src.shared.ends.push(id, uranium_agent::SessionEnd { slot, winner, frame: 11_520, reason });
            }
            src.hooks[0].tx.send(GameResult { outcome: Some(session_view), ..result(id, Some(scout)) }).unwrap();
            let aborted = src.shared.aborted.load(Ordering::SeqCst);
            src.done(&job, JobOutcome::Finished, None);
            let row: Vec<String> = std::fs::read_to_string(&csv).unwrap().lines().last().unwrap().split(',').map(str::to_string).collect();
            assert_eq!(row.len(), GAMES_HEADER.trim().split(',').count(), "{row:?}");
            let terminal_r_win = src.shared.queues[0].try_take(1, 0, 1_000).map(|b| b[0].steps.last().unwrap().reward[0]);
            let win_delta = src.shared.league.lock().unwrap().win_rate(AgentKind::Main, &key) - before;
            let buffer = src.shared.sil.as_ref().unwrap().buffer.lock().unwrap().len();
            Played { row, win_delta, terminal_r_win, buffer, aborted: src.shared.aborted.load(Ordering::SeqCst) - aborted }
        };
        let m = |scv_last: Option<u32>, scv_4m: Option<u32>| ScoutMetrics { scv_last, scv_4m, ..Default::default() };
        let cols = |p: &Played| (p.row[7].clone(), p.row[10].clone(), p.row[16].clone(), p.row[26].clone());
        let s = |x: &str| x.to_string();
        let won = play("rush", 2, EndReason::TimeLimitDraw, None, m(Some(8), Some(12)), false);
        assert_eq!(cols(&won), (s("Some(Win)"), s("1"), s("RushJudge"), s("8")));
        assert_eq!(won.terminal_r_win, Some(1.0), "r_win on the last decision");
        assert!((won.win_delta - 0.05).abs() < 1e-9, "win table: a win (0.9 x 0.5 + 0.1 x 1)");
        assert_eq!(won.buffer, 1, "a won game: selected (its terminal segment)");
        let sampled = src.shared.sil.as_ref().unwrap().buffer.lock().unwrap().sample(1, crate::sil::SilMode::Sil, 4);
        assert_eq!(sampled[0].sil_return, Some(1.0), "the SIL return R is the judged r_win");
        let lost = play("rush", 2, EndReason::TimeLimitDraw, None, m(Some(7), Some(12)), false);
        assert_eq!(cols(&lost), (s("Some(Loss)"), s("-1"), s("RushJudge"), s("7")));
        assert_eq!(lost.terminal_r_win, Some(-1.0));
        assert!(lost.win_delta < 0.0, "win table: a loss");
        assert_eq!(lost.buffer, 1, "wins_only: a lost game with scv_4m >= 12 is not selected");
        let none = play("rush", 0, EndReason::TimeLimitDraw, None, m(None, None), false);
        assert_eq!(cols(&none), (s("None"), s("0"), s(""), s("")));
        assert_eq!((none.terminal_r_win, none.win_delta, none.aborted), (None, 0.0, 1));
        let elim = play("rush", 2, EndReason::Elimination, Some(0), m(Some(3), None), false);
        assert_eq!(cols(&elim), (s("Some(Win)"), s("1"), s("Elimination"), s("")));
        assert_eq!((elim.terminal_r_win, elim.buffer), (Some(1.0), 2), "a win before 8:00 is selected too");
        let gg = play("rush", 2, EndReason::Surrender, Some(1), m(Some(20), None), false);
        assert_eq!(cols(&gg), (s("Some(Loss)"), s("-1"), s("Surrender"), s("")));
        let conceded = play("rush", 2, EndReason::Surrender, Some(1), m(Some(20), Some(15)), true);
        assert_eq!(cols(&conceded), (s("Some(Loss)"), s("-1"), s("Concede"), s("")));
        let both = play("rush", 2, EndReason::TimeLimitDraw, None, m(Some(3), Some(15)), true);
        assert_eq!(cols(&both), (s("Some(Loss)"), s("-1"), s("RushJudge"), s("3")));
        assert_eq!(both.buffer, 2, "a judged loss is not selected");
        let macro_draw = play("macro", 2, EndReason::TimeLimitDraw, None, m(Some(20), Some(15)), false);
        assert_eq!(cols(&macro_draw), (s("Some(Draw)"), s("0"), s("TimeLimitDraw"), s("")));
        assert_eq!((macro_draw.terminal_r_win, macro_draw.buffer), (Some(0.0), 2));
        assert!(macro_draw.win_delta.abs() < 1e-9, "a draw from the 0.5 prior");
        assert!(src.shared.sil.as_ref().unwrap().pending.lock().unwrap().is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }
}

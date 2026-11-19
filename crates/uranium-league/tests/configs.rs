use uranium_match::concede::ConcedeConfig;
use uranium_env::scout::*;
use uranium_league::config::*;
use uranium_league::league::*;

#[test]
fn t1self_config_042_r5() {
    let t: RlConfig = toml::from_str(include_str!("../../../config/rl_t1self.toml")).unwrap();
    t.validate().unwrap();
    let cc = &t.curriculum;
    assert_eq!((cc.start_stage, cc.auto_advance), (Stage::C1, false));
    assert_eq!((cc.c1_t0, cc.c1_t1, cc.c1_self), (0.0, 0.5, 0.5));
    assert_eq!(cc.r_value_stages, vec![Stage::C0, Stage::C1]);
    let names: Vec<&str> = t.t1.iter().map(|b| b.name.as_str()).collect();
    assert_eq!(names, ["ualbertabot", "steamhammer", "stardust", "zzzkbot", "iron"], "McRave left out");
    for r in [uranium_proto::Race::Terran, uranium_proto::Race::Protoss, uranium_proto::Race::Zerg] {
        assert_eq!(t.t1.iter().filter(|b| b.races.contains(&r)).count(), 3, "{r:?}");
    }
    let mut d = RlConfig::default_file();
    let dc = &mut d.curriculum;
    (dc.start_stage, dc.auto_advance, dc.r_value_stages) = (cc.start_stage, cc.auto_advance, cc.r_value_stages.clone());
    (dc.c1_t0, dc.c1_t1, dc.c1_self) = (cc.c1_t0, cc.c1_t1, cc.c1_self);
    d.t1.retain(|b| b.name != "mcrave");
    assert_eq!(d, t);
}

#[test]
fn t1self2_config_042_amb_042_14() {
    use uranium_proto::Race;
    let t: RlConfig = toml::from_str(include_str!("../../../config/rl_t1self2.toml")).unwrap();
    t.validate().unwrap();
    let cc = &t.curriculum;
    assert_eq!((cc.c1_t0, cc.c1_t1, cc.c1_self, cc.c1_snapshot, cc.c1_anchor), (0.0, 0.5, 0.15, 0.25, 0.10));
    assert!(!cc.r_value_vs_model);
    assert_eq!(cc.r_value_stages, vec![Stage::C0, Stage::C1]);
    assert_eq!(t.league.anchor, "checkpoints/ac1-vtrace-s1/ckpt-3133569");
    assert_eq!(t.league.anchor_name().as_deref(), Some("ckpt-3133569"));
    assert_eq!((t.snapshot.main_every_updates, t.snapshot.main_candidates), (250, 4));
    let sh = t.t1.iter().find(|b| b.name == "steamhammer").unwrap();
    assert_eq!(sh.races, vec![Race::Terran, Race::Zerg]);
    for (r, n) in [(Race::Terran, 3), (Race::Protoss, 2), (Race::Zerg, 3)] {
        assert_eq!(t.t1.iter().filter(|b| b.races.contains(&r)).count(), n, "{r:?}");
    }
    let mut d: RlConfig = toml::from_str(include_str!("../../../config/rl_t1self.toml")).unwrap();
    let dc = &mut d.curriculum;
    (dc.c1_t1, dc.c1_self, dc.c1_snapshot, dc.c1_anchor, dc.r_value_vs_model) = (cc.c1_t1, cc.c1_self, cc.c1_snapshot, cc.c1_anchor, cc.r_value_vs_model);
    d.league.anchor = t.league.anchor.clone();
    (d.snapshot.main_every_updates, d.snapshot.main_candidates) = (250, 4);
    d.t1.iter_mut().find(|b| b.name == "steamhammer").unwrap().races = sh.races.clone();
    assert_eq!(d, t);
}

#[test]
fn rush1_config_042_amb_042_15() {
    use uranium_proto::Race;
    let t: RlConfig = toml::from_str(include_str!("../../../config/rl_rush1.toml")).unwrap();
    t.validate().unwrap();
    let cc = &t.curriculum;
    assert_eq!((cc.c1_ladder, cc.c1_anchor, cc.c1_snapshot, cc.c1_self, cc.c1_t1, cc.c1_t0), (0.70, 0.10, 0.15, 0.05, 0.0, 0.0));
    assert_eq!(cc.ladder_pick, LadderPickSetting::All(LadderPick::FVar));
    assert_eq!(t.concede, ConcedeConfig { enabled: true, ..ConcedeConfig::default() });
    let keys: Vec<String> = t.ladder.iter().map(|e| e.opponent().unwrap().1).collect();
    assert_eq!(keys, ["ualbertabot9pool-zerg", "t0-zerg-rush", "ualbertabot12poollings-zerg", "ualbertabot-zerg", "ualbertabot9poollings-zerg", "steamhammeroverpool-zerg", "zzzkbot-zerg"], "user decision 2026-11-03");
    assert!(t.ladder.iter().all(|e| e.race == Race::Zerg));
    let mut d: RlConfig = toml::from_str(include_str!("../../../config/rl_t1self2.toml")).unwrap();
    let dc = &mut d.curriculum;
    (dc.c1_ladder, dc.c1_anchor, dc.c1_snapshot, dc.c1_self, dc.c1_t1, dc.c1_t0) = (cc.c1_ladder, cc.c1_anchor, cc.c1_snapshot, cc.c1_self, cc.c1_t1, cc.c1_t0);
    d.ladder = t.ladder.clone();
    d.concede.enabled = true;
    assert_eq!(d, t, "everything else as in t1self-s2 ([[t1]] included)");
}

#[test]
fn rush2_config_042_amb_042_16() {
    use uranium_env::scout::{ScoutConfig, SurviveConfig};
    let t: RlConfig = toml::from_str(include_str!("../../../config/rl_rush2.toml")).unwrap();
    t.validate().unwrap();
    assert_eq!(t.league.win_rate_decay, 0.97);
    assert_eq!(t.league.anchor, "checkpoints/ac1-vtrace-s1/ckpt-3133569", "AC1 anchor kept");
    let groups: Vec<(&str, f64)> = t.curriculum.ladder_groups.iter().map(|(g, x)| (g.as_str(), *x)).collect();
    assert_eq!(groups, [("macro", 0.5), ("rush", 0.5)]);
    assert_eq!(t.reward, RewardConfig { pseudo: PseudoMode::Scout, scout: ScoutConfig::default(), survive: SurviveConfig::default() });
    let s = &t.reward.scout;
    assert_eq!((s.weight, s.cap, s.vs_model, s.full_until_s, s.zero_at_s, s.main_radius_tiles, s.workers_min), (0.15, 1.0, 0.5, 240.0, 480.0, 10.0, 4));
    assert_eq!((s.main, s.building_type, s.expansion, s.workers, s.army), (0.3, 0.1, 0.1, 0.1, 0.1));
    let v = &t.reward.survive;
    assert_eq!((v.weight, v.vs_model, v.from_s, v.until_s, v.scv_target), (0.1, 0.0, 180.0, 480.0, 20));
    let rush1: RlConfig = toml::from_str(include_str!("../../../config/rl_rush1.toml")).unwrap();
    let rush: Vec<&LadderEntry> = t.ladder.iter().filter(|e| e.group == "rush").collect();
    assert_eq!(rush.len(), 7);
    for (a, b) in rush.iter().zip(&rush1.ladder) {
        assert_eq!(LadderEntry { group: String::new(), ..(*a).clone() }, *b, "the rush-s1 entries as they were");
    }
    let macro_keys: Vec<String> = t.ladder.iter().filter(|e| e.group == "macro").map(|e| e.opponent().unwrap().1).collect();
    assert_eq!(macro_keys, ["t0-zerg", "t0-zerg-muta", "ualbertabot2hatchhydra-zerg", "ualbertabot3hatchmuta-zerg", "steamhammer2hatchmuta-zerg"], "user decision 2026-11-04 (AMB-042-16)");
    assert!(t.ladder.iter().all(|e| e.race == uranium_proto::Race::Zerg));
    assert_eq!(t.ladder.len(), 12, "7 rush + 5 macro");
    let mut d = rush1;
    d.league.win_rate_decay = 0.97;
    d.curriculum.ladder_groups = t.curriculum.ladder_groups.clone();
    d.ladder = t.ladder.clone();
    d.reward = t.reward.clone();
    assert_eq!(d, t, "everything else as in rush-s1");
}

#[test]
fn reward_and_group_keys_default_amb_042_16() {
    let c = RlConfig::default_file();
    assert_eq!(c.reward, RewardConfig::default());
    assert_eq!(c.pseudo(), PseudoMode::Z);
    assert!(c.curriculum.ladder_groups.is_empty());
    let full = include_str!("../../../config/rl.toml");
    let at = full.find("\n[reward]").unwrap() + 1;
    let end = at + full[at..].find("\n[concede]").unwrap() + 1;
    let old: RlConfig = toml::from_str(&format!("{}{}", &full[..at], &full[end..])).unwrap();
    assert_eq!(old, c, "absent = defaults");
    for f in [include_str!("../../../config/rl_rush1.toml"), include_str!("../../../config/rl_t1self2.toml")] {
        let o: RlConfig = toml::from_str(f).unwrap();
        o.validate().unwrap();
        assert_eq!((o.reward.clone(), o.curriculum.ladder_groups.is_empty()), (RewardConfig::default(), true));
        assert!(o.ladder.iter().all(|e| e.group.is_empty()));
    }
    let t: RlConfig = toml::from_str(include_str!("../../../config/rl_rush2.toml")).unwrap();
    let back: RlConfig = serde_json::from_str(&serde_json::to_string(&t).unwrap()).unwrap();
    assert_eq!(back, t);
}

#[test]
fn aux_and_monitor_tables_042_r21() {
    let c = RlConfig::default_file();
    assert!(c.aux.is_none() && c.monitor.is_none());
    let json = serde_json::to_value(&c).unwrap();
    assert!(json.get("aux").is_none() && json.get("monitor").is_none());
    let ck: RlConfig = toml::from_str(include_str!("../../../releases/checkpoint-1-rl/config/rl.toml")).unwrap();
    assert!(ck.aux.is_none() && ck.monitor.is_none());
    let back: RlConfig = serde_json::from_str(&serde_json::to_string(&ck).unwrap()).unwrap();
    assert_eq!(back, ck);
    let full = include_str!("../../../config/rl.toml").replace("\r\n", "\n");
    let on: RlConfig = toml::from_str(&format!("{full}\n[aux]\npriv = 0.2\njepa = 0.1\n\n[monitor]\nspatial_every = 100\n")).unwrap();
    on.validate().unwrap();
    assert_eq!(on.aux, Some(AuxWeights { privileged: Some(0.2), jepa: Some(0.1), adam_eps: None }));
    assert_eq!(on.monitor, Some(MonitorConfig { spatial_every: 100 }));
    let json = serde_json::to_value(&on).unwrap();
    assert_eq!((json["aux"]["priv"].as_f64().map(|x| x as f32), json["monitor"]["spatial_every"].as_u64()), (Some(0.2), Some(100)));
    let bad: RlConfig = toml::from_str(&format!("{full}\n[aux]\npriv = -0.2\n")).unwrap();
    assert!(bad.validate().is_err());
    let empty: RlConfig = toml::from_str(&format!("{full}
[aux]
")).unwrap();
    assert!(empty.validate().is_err(), "an empty [aux] table is refused");
    assert!(toml::from_str::<RlConfig>(&format!("{full}\n[aux]\nprivileged = 0.2\n")).is_err());
}

#[test]
fn rl_rejects_aux_adam_eps_042_r21() {
    let mut t: RlConfig = toml::from_str(include_str!("../../../config/rl_m2.toml")).unwrap();
    t.aux.as_mut().unwrap().adam_eps = Some(1e-8);
    let err = format!("{:#}", t.validate().unwrap_err());
    assert!(err.contains("adam_eps") && err.contains("042-R21"), "{err}");
    let full = include_str!("../../../config/rl.toml").replace("\r\n", "\n");
    let c: RlConfig = toml::from_str(&format!("{full}\n[aux]\npriv = 0.2\njepa = 0.1\nadam_eps = 1e-5\n")).unwrap();
    assert!(c.validate().is_err(), "041-R23 ⑦ is BC only, even at the code default");
}

#[test]
fn rl_m2_config_042_r21() {
    let mut t: RlConfig = toml::from_str(include_str!("../../../config/rl_m2.toml")).unwrap();
    t.validate().unwrap();
    assert_eq!(t.aux, Some(AuxWeights { privileged: Some(0.2), jepa: Some(0.1), adam_eps: None }), "AMB-041-13b / AMB-042-21");
    assert_eq!(t.monitor, Some(MonitorConfig { spatial_every: 100 }), "AMB-041-14");
    (t.aux, t.monitor) = (None, None);
    assert_eq!(t, RlConfig::default_file(), "everything else as config/rl.toml");
}

#[test]
fn amb_042_17_keys_default() {
    let c = RlConfig::default_file();
    assert_eq!(c.kl.head_scale, KlHeadScale::default());
    assert!(c.kl.head_scale.by_head().iter().all(|(_, x)| *x == 1.0));
    assert_eq!(c.league.model_game_max_s, 0);
    assert_eq!(c.curriculum.band, BandConfig { center: 0.5, width: 0.15, exclude_below: 0.10 });
    assert_eq!(c.curriculum.ladder_pick, LadderPickSetting::All(LadderPick::FVar));
    assert_eq!(c.sil, SilConfig::default());
    assert!(!c.sil.enabled);
    let s = &c.sil;
    assert_eq!((s.mode, s.select_group.as_str(), s.select_scv_4m, s.until_s, s.buffer_segments, s.lanes, s.w_pg, s.w_v, s.max_uses), (SilMode::Sil, "rush", 12, 480, 2_048, 4, 0.05, 0.01, 4), "042-R16 기본안(미확정)");
    let full = include_str!("../../../config/rl.toml");
    let cut = |text: &str, from: &str| {
        let at = text.find(from).unwrap();
        let end = at + 1 + text[at + 1..].find("\n[").unwrap();
        format!("{}{}", &text[..at], &text[end..])
    };
    let old = cut(&cut(full, "\n[kl.head_scale]"), "\n[curriculum.band]");
    let old: String = old.lines().filter(|l| !l.starts_with("model_game_max_s =")).map(|l| format!("{l}\n")).collect();
    assert!(!old.lines().any(|l| !l.starts_with('#') && (l.contains("head_scale") || l.starts_with("center =") || l.starts_with("model_game_max_s ="))));
    let o: RlConfig = toml::from_str(&old).unwrap();
    assert_eq!(o, c, "absent = defaults");
    for f in [include_str!("../../../config/rl_rush2.toml"), include_str!("../../../config/rl_rush1.toml")] {
        let o: RlConfig = toml::from_str(f).unwrap();
        o.validate().unwrap();
        assert_eq!((o.kl.head_scale.clone(), o.league.model_game_max_s, o.curriculum.band, o.sil.clone()), (KlHeadScale::default(), 0, BandConfig::default(), SilConfig::default()));
    }
}

#[test]
fn rush3_config_042_amb_042_17() {
    let t: RlConfig = toml::from_str(include_str!("../../../config/rl_rush3.toml")).unwrap();
    t.validate().unwrap();
    let cc = &t.curriculum;
    assert_eq!((cc.c1_ladder, cc.c1_anchor, cc.c1_snapshot, cc.c1_self, cc.c1_t1, cc.c1_t0), (0.90, 0.05, 0.05, 0.0, 0.0, 0.0));
    assert_eq!(cc.ladder_pick, LadderPickSetting::PerGroup([("rush".to_string(), LadderPick::Band), ("macro".to_string(), LadderPick::FVar)].into_iter().collect()));
    assert_eq!(cc.band, BandConfig { center: 0.5, width: 0.15, exclude_below: 0.10 });
    assert_eq!(t.league.model_game_max_s, 1_500);
    assert_eq!(t.league.game_max_frames(true), 36_000);
    assert_eq!((t.kl.init, t.kl.floor, t.kl.half_life_updates), (0.1, 0.01, 20_000.0), "rl.toml schedule (reverted 2026-11-04)");
    assert_eq!(t.kl, RlConfig::default_file().kl, "KL as in rl.toml");
    assert_eq!(t.kl.head_scale, KlHeadScale::default());
    assert_eq!(t.loss.td_lambda, 0.95);
    assert!(!t.sil.enabled);
    assert_eq!(t.sil, SilConfig::default());
    let rush2: RlConfig = toml::from_str(include_str!("../../../config/rl_rush2.toml")).unwrap();
    let rung = ["ualbertabot9pool10lings", "ualbertabot9pool12lings", "ualbertabot9poolhydra"];
    let added: Vec<&LadderEntry> = t.ladder.iter().filter(|e| !rush2.ladder.contains(e)).collect();
    let keys: Vec<String> = added.iter().map(|e| e.opponent().unwrap().1).collect();
    assert_eq!(keys, rung.map(|n| format!("{n}-zerg")));
    assert!(added.iter().all(|e| e.group == "rush" && e.race == uranium_proto::Race::Zerg && e.kind == uranium_league::league::LadderKind::T1));
    let kept: Vec<LadderEntry> = t.ladder.iter().filter(|e| rush2.ladder.contains(e)).cloned().collect();
    assert_eq!(kept, rush2.ladder, "the rush-s2 entries as they were");
    assert_eq!((t.ladder.iter().filter(|e| e.group == "rush").count(), t.ladder.iter().filter(|e| e.group == "macro").count()), (10, 5));
    let mut d = rush2;
    d.ladder = t.ladder.clone();
    let dc = &mut d.curriculum;
    (dc.c1_ladder, dc.c1_anchor, dc.c1_snapshot, dc.c1_self) = (0.90, 0.05, 0.05, 0.0);
    dc.ladder_pick = cc.ladder_pick.clone();
    d.league.model_game_max_s = 1_500;
    d.loss.td_lambda = 0.95;
    assert_eq!(d, t, "everything else as in rush-s2");
    let back: RlConfig = serde_json::from_str(&serde_json::to_string(&t).unwrap()).unwrap();
    assert_eq!(back, t);
}

#[test]
fn rush_judge_and_wins_only_default_042_r17() {
    assert_eq!(RushJudgeConfig::default(), RushJudgeConfig { groups: vec![], at_s: 480, min_scv: 8 });
    for f in [
        include_str!("../../../config/rl.toml"),
        include_str!("../../../config/rl_rush3.toml"),
        include_str!("../../../config/rl_rush2.toml"),
        include_str!("../../../config/rl_rush1.toml"),
    ] {
        let c: RlConfig = toml::from_str(f).unwrap();
        c.validate().unwrap();
        assert_eq!(c.league.rush_judge, RushJudgeConfig::default());
        assert!(!c.sil.wins_only);
        for g in [None, Some("rush"), Some("macro"), Some("warmup")] {
            assert!(!c.league.rush_judged(false, g));
            assert_eq!(c.league.game_max_frames_of(false, g), GAME_MAX_FRAMES, "{g:?}");
            assert_eq!(c.league.game_max_frames_of(true, g), c.league.game_max_frames(true));
        }
        let json = serde_json::to_value(&c).unwrap();
        assert!(json["league"].get("rush_judge").is_none(), "checkpoint config as before");
        assert_eq!(serde_json::from_value::<RlConfig>(json).unwrap(), c);
    }
    assert!(!include_str!("../../../config/rl.toml").lines().any(|l| l.starts_with("[league.rush_judge]") || l.starts_with("wins_only")), "rl.toml: not set");
}

#[test]
fn rush_judge_settings_are_checked_042_r17() {
    let base: RlConfig = toml::from_str(include_str!("../../../config/rl_rush4.toml")).unwrap();
    base.validate().unwrap();
    let with = |groups: &[&str], at_s: u32, min_scv: u32| {
        let mut c = base.clone();
        c.league.rush_judge = RushJudgeConfig { groups: groups.iter().map(|g| g.to_string()).collect(), at_s, min_scv };
        c.validate()
    };
    for (groups, at_s, min_scv, ok) in [
        (&["rush"][..], 480, 8, true),
        (&["rush", "warmup", "macro"][..], 480, 8, true),
        (&["rush"][..], 60, 1, true),
        (&["rush"][..], 3_600, 8, true),
        (&["rush"][..], 59, 8, false),
        (&["rush"][..], 3_601, 8, false),
        (&["rush"][..], 0, 8, false),
        (&["rush"][..], 480, 0, false),
        (&["late"][..], 480, 8, false),
        (&["rush", ""][..], 480, 8, false),
        (&[][..], 0, 0, true),
    ] {
        assert_eq!(with(groups, at_s, min_scv).is_ok(), ok, "{groups:?} {at_s} {min_scv}");
    }
    let mut c = RlConfig::default_file();
    c.league.rush_judge.groups = vec!["rush".into()];
    assert!(c.validate().is_err());
    let text = include_str!("../../../config/rl_rush4.toml").replace("\r\n", "\n");
    for (from, to) in [("\nmin_scv = 8\n", "\nmin_scvs = 8\n"), ("at_s = 480\nmin_scv", "at_s = -1\nmin_scv"), ("groups = [\"rush\", \"warmup\"]", "groups = \"rush\"")] {
        assert!(text.contains(from), "{from}");
        assert!(toml::from_str::<RlConfig>(&text.replacen(from, to, 1)).is_err(), "{to}");
    }
    let only_groups: RlConfig = toml::from_str(&text.replacen("at_s = 480\nmin_scv = 8\n", "", 1)).unwrap();
    assert_eq!(only_groups.league.rush_judge, base.league.rush_judge);
    let l = &base.league;
    for (vs_model, g, want) in [(false, Some("rush"), 11_520), (false, Some("warmup"), 11_520), (false, Some("macro"), 86_400), (false, None, 86_400), (true, None, 36_000), (true, Some("rush"), 36_000)] {
        assert_eq!(l.game_max_frames_of(vs_model, g), want, "{vs_model} {g:?}");
        assert_eq!(l.rush_judged(vs_model, g), want == 11_520);
    }
    let at = |s: u32| RushJudgeConfig { groups: vec!["rush".into()], at_s: s, min_scv: 8 }.max_frames();
    assert_eq!((at(60), at(480), at(3_600)), (1_440, 11_520, 86_400));
    use uranium_env::reward::Outcome;
    let j = &l.rush_judge;
    assert_eq!((j.judge(8), j.judge(7), j.judge(0), j.judge(50)), (Outcome::Win, Outcome::Loss, Outcome::Loss, Outcome::Win));
}

#[test]
fn rush4_config_042_amb_042_18() {
    let t: RlConfig = toml::from_str(include_str!("../../../config/rl_rush4.toml")).unwrap();
    t.validate().unwrap();
    let groups: Vec<(&str, f64)> = t.curriculum.ladder_groups.iter().map(|(g, x)| (g.as_str(), *x)).collect();
    assert_eq!(groups, [("macro", 0.20), ("rush", 0.65), ("warmup", 0.05)]);
    let cc = &t.curriculum;
    assert_eq!((cc.c1_ladder, cc.c1_anchor, cc.c1_snapshot, cc.c1_self, cc.c1_t1, cc.c1_t0), (0.90, 0.05, 0.05, 0.0, 0.0, 0.0));
    assert_eq!(cc.ladder_pick.for_group(Some("warmup")), LadderPick::FVar);
    assert_eq!(t.league.win_rate_decay, 0.9);
    assert_eq!(cc.band, BandConfig { center: 0.5, width: 0.30, exclude_below: 0.0 });
    assert_eq!(t.league.rush_judge, RushJudgeConfig { groups: vec!["rush".into(), "warmup".into()], at_s: 480, min_scv: 8 });
    assert_eq!(t.league.rush_judge.max_frames(), 11_520);
    assert_eq!(t.sil, SilConfig { enabled: true, mode: SilMode::Sil, lanes: 4, w_pg: 0.05, w_v: 0.01, until_s: 480, buffer_segments: 256, select_group: "rush".into(), wins_only: true, ..SilConfig::default() });
    let keys = |g: &str| t.ladder.iter().filter(|e| e.group == g).map(|e| e.opponent().unwrap().1).collect::<Vec<_>>();
    assert_eq!(keys("warmup"), ["ualbertabot9pool-zerg"]);
    assert_eq!(keys("rush"), ["t0-zerg-rush", "ualbertabot12poollings-zerg", "ualbertabot9pool10lings-zerg", "ualbertabot9poolhydra-zerg"]);
    assert_eq!(keys("macro"), ["t0-zerg", "t0-zerg-muta", "ualbertabot2hatchhydra-zerg", "ualbertabot3hatchmuta-zerg", "steamhammer2hatchmuta-zerg"]);
    assert_eq!(t.ladder.len(), 10);
    let mut d: RlConfig = toml::from_str(include_str!("../../../config/rl_rush3.toml")).unwrap();
    let left_out = ["ualbertabot9pool12lings", "ualbertabot9poollings", "ualbertabot", "steamhammeroverpool", "zzzkbot"];
    d.ladder.retain(|e| !e.name.as_deref().is_some_and(|n| left_out.contains(&n)));
    d.ladder.iter_mut().filter(|e| e.name.as_deref() == Some("ualbertabot9pool")).for_each(|e| e.group = "warmup".into());
    d.curriculum.ladder_groups = t.curriculum.ladder_groups.clone();
    d.league.win_rate_decay = 0.9;
    d.curriculum.band = BandConfig { center: 0.5, width: 0.30, exclude_below: 0.0 };
    d.league.rush_judge = t.league.rush_judge.clone();
    d.sil = t.sil.clone();
    assert_eq!(d, t);
    let back: RlConfig = serde_json::from_str(&serde_json::to_string(&t).unwrap()).unwrap();
    assert_eq!(back, t, "the serialised config (checkpoint) reads back the same");
}

#[test]
fn amb_042_19_keys_default() {
    assert_eq!(StartMixConfig::default(), StartMixConfig { rush_hit_share: None, groups: vec!["rush".into()] });
    for f in [
        include_str!("../../../config/rl.toml"),
        include_str!("../../../config/rl_rush1.toml"),
        include_str!("../../../config/rl_rush2.toml"),
        include_str!("../../../config/rl_rush3.toml"),
        include_str!("../../../config/rl_rush4.toml"),
    ] {
        let c: RlConfig = toml::from_str(f).unwrap();
        c.validate().unwrap();
        assert_eq!(c.league.start_mix, StartMixConfig::default());
        for g in [None, Some("rush"), Some("macro"), Some("warmup")] {
            assert_eq!(c.league.start_mix.share_for(g), None);
            assert_ne!(c.curriculum.ladder_pick.for_group(g), LadderPick::Fixed);
        }
        assert!(c.ladder.iter().all(|e| e.weight.is_none()));
        assert_eq!((c.sil.select_keys.as_ref(), c.sil.select_army_before_s, c.sil.select_start_hit), (None, None, None));
        let json = serde_json::to_value(&c).unwrap();
        assert!(json["league"].get("start_mix").is_none() && json["sil"].get("select_keys").is_none() && json["sil"].get("select_army_before_s").is_none() && json["sil"].get("select_start_hit").is_none());
        assert!(json["ladder"].as_array().unwrap().iter().all(|e| e.get("weight").is_none()), "checkpoint config as before");
    }
    let rl = include_str!("../../../config/rl.toml");
    assert!(!rl.lines().any(|l| ["[league.start_mix]", "rush_hit_share", "select_keys", "select_army_before_s", "select_start_hit", "[[ladder]]"].iter().any(|k| l.starts_with(k))), "rl.toml: not set");
}

#[test]
fn amb_042_19_settings_are_checked() {
    let base: RlConfig = toml::from_str(include_str!("../../../config/rl_rush5.toml")).unwrap();
    base.validate().unwrap();
    let check = |f: &dyn Fn(&mut RlConfig)| {
        let mut c = base.clone();
        f(&mut c);
        c.validate()
    };
    for (share, ok) in [(0.0, true), (1.0, true), (0.5, true), (-0.01, false), (1.01, false), (f64::NAN, false)] {
        assert_eq!(check(&|c| c.league.start_mix.rush_hit_share = Some(share)).is_ok(), ok, "{share}");
    }
    assert!(check(&|c| c.league.start_mix.groups = vec![]).is_err(), "empty groups");
    assert!(check(&|c| c.league.start_mix.groups = vec!["late".into()]).is_err(), "not in ladder_groups");
    check(&|c| c.league.start_mix.groups = vec!["rush".into(), "macro".into()]).unwrap();
    check(&|c| (c.league.start_mix.rush_hit_share, c.league.start_mix.groups) = (None, vec![])).unwrap();
    let mut no_groups = RlConfig::default_file();
    no_groups.league.start_mix.rush_hit_share = Some(0.5);
    assert!(no_groups.validate().is_err(), "no ladder_groups");
    fn rush(c: &mut RlConfig) -> Vec<&mut LadderEntry> {
        c.ladder.iter_mut().filter(|e| e.group == "rush").collect()
    }
    assert!(check(&|c| rush(c)[0].weight = Some(-0.1)).is_err(), "negative weight");
    assert!(check(&|c| rush(c)[0].weight = Some(f64::INFINITY)).is_err());
    assert!(check(&|c| rush(c).into_iter().for_each(|e| e.weight = Some(0.0))).is_err(), "fixed group sums to 0");
    check(&|c| rush(c)[0].weight = Some(0.0)).unwrap();
    check(&|c| rush(c).into_iter().for_each(|e| e.weight = None)).unwrap();
    assert!(check(&|c| c.ladder.iter_mut().find(|e| e.group == "macro").unwrap().weight = Some(1.0)).is_err(), "weight on a f_var group");
    assert!(check(&|c| c.curriculum.ladder_pick = LadderPickSetting::PerGroup([("rush".to_string(), LadderPick::Band)].into_iter().collect())).is_err(), "weights without fixed");
    check(&|c| c.curriculum.ladder_pick = LadderPickSetting::All(LadderPick::Fixed)).unwrap();
    let mut flat = RlConfig::default_file();
    flat.curriculum.c1_ladder = 0.5;
    flat.ladder = base.ladder.iter().map(|e| LadderEntry { group: String::new(), ..e.clone() }).collect();
    assert!(flat.validate().is_err(), "weights with the default f_var");
    flat.curriculum.ladder_pick = LadderPickSetting::All(LadderPick::Fixed);
    flat.validate().unwrap();
    assert!(check(&|c| c.sil.select_keys = Some(vec!["t0-zerg".into()])).is_err(), "a macro key");
    assert!(check(&|c| c.sil.select_keys = Some(vec!["nobody-zerg".into()])).is_err());
    assert!(check(&|c| c.sil.select_keys = Some(vec![])).is_err());
    assert!(check(&|c| c.sil.select_army_before_s = Some(-1.0)).is_err());
    check(&|c| c.sil.select_keys = Some(vec!["ualbertabot9poolhydra-zerg".into()])).unwrap();
    check(&|c| (c.sil.select_keys, c.sil.select_army_before_s) = (None, None)).unwrap();
    let text = include_str!("../../../config/rl_rush5.toml").replace("\r\n", "\n");
    for (from, to) in [("\nrush_hit_share = 0.5\n", "\nrush_hit_shares = 0.5\n"), ("\nrush_hit_share = 0.5\n", "\nrush_hit_share = \"half\"\n"), ("\nselect_army_before_s = 225\n", "\nselect_army_before = 225\n"), ("\nweight = 0.35\n", "\nweight = \"0.35\"\n"), ("\nladder_pick = { rush = \"fixed\"", "\nladder_pick = { rush = \"fix\"")] {
        assert!(text.contains(from), "{from}");
        assert!(toml::from_str::<RlConfig>(&text.replacen(from, to, 1)).is_err(), "{to}");
    }
}

#[test]
fn rush5_config_042_amb_042_19() {
    let t: RlConfig = toml::from_str(include_str!("../../../config/rl_rush5.toml")).unwrap();
    t.validate().unwrap();
    let groups: Vec<(&str, f64)> = t.curriculum.ladder_groups.iter().map(|(g, x)| (g.as_str(), *x)).collect();
    assert_eq!(groups, [("macro", 0.20), ("rush", 0.70)]);
    let cc = &t.curriculum;
    assert_eq!((cc.c1_ladder, cc.c1_anchor, cc.c1_snapshot, cc.c1_self, cc.c1_t1, cc.c1_t0), (0.90, 0.05, 0.05, 0.0, 0.0, 0.0));
    assert_eq!((cc.ladder_pick.for_group(Some("rush")), cc.ladder_pick.for_group(Some("macro"))), (LadderPick::Fixed, LadderPick::FVar));
    let weights: Vec<(String, Option<f64>)> = t.ladder.iter().filter(|e| e.group == "rush").map(|e| (e.opponent().unwrap().1, e.weight)).collect();
    assert_eq!(weights, [("t0-zerg-rush".to_string(), Some(0.35)), ("ualbertabot12poollings-zerg".into(), Some(0.30)), ("ualbertabot9pool10lings-zerg".into(), Some(0.20)), ("ualbertabot9poolhydra-zerg".into(), Some(0.15))]);
    assert!(t.ladder.iter().filter(|e| e.group == "macro").all(|e| e.weight.is_none()));
    assert_eq!(t.league.start_mix, StartMixConfig { rush_hit_share: Some(0.5), groups: vec!["rush".into()] });
    assert_eq!((t.league.start_mix.share_for(Some("rush")), t.league.start_mix.share_for(Some("macro")), t.league.start_mix.share_for(None)), (Some(0.5), None, None));
    assert_eq!(t.league.rush_judge, RushJudgeConfig { groups: vec!["rush".into()], at_s: 480, min_scv: 8 });
    assert_eq!(t.sil.select_keys, Some(vec!["t0-zerg-rush".to_string(), "ualbertabot12poollings-zerg".into(), "ualbertabot9pool10lings-zerg".into()]));
    assert_eq!(t.sil.select_army_before_s, Some(225.0));
    assert_eq!(t.sil.select_start_hit, Some(true));
    assert_eq!(t.ladder.len(), 9);
    let mut d: RlConfig = toml::from_str(include_str!("../../../config/rl_rush4.toml")).unwrap();
    d.ladder.retain(|e| e.group != "warmup");
    for e in d.ladder.iter_mut() {
        e.weight = t.ladder.iter().find(|x| x.opponent().unwrap().1 == e.opponent().unwrap().1).unwrap().weight;
    }
    d.curriculum.ladder_groups = t.curriculum.ladder_groups.clone();
    d.curriculum.ladder_pick = t.curriculum.ladder_pick.clone();
    d.league.start_mix = t.league.start_mix.clone();
    d.league.rush_judge.groups = vec!["rush".into()];
    (d.sil.select_keys, d.sil.select_army_before_s, d.sil.select_start_hit) = (t.sil.select_keys.clone(), t.sil.select_army_before_s, t.sil.select_start_hit);
    assert_eq!(d, t);
    let back: RlConfig = serde_json::from_str(&serde_json::to_string(&t).unwrap()).unwrap();
    assert_eq!(back, t, "the serialised config (checkpoint) reads back the same");
}

#[test]
fn rl_m2_race_config_044_r10() {
    use uranium_proto::Race;
    let text = include_str!("../../../config/rl_m2_race.toml").replace("\r\n", "\n");
    let s = "checkpoints/m2-dagger-r2/ckpt-1408";
    let filled = format!("\nzerg = \"{s}\"\n");
    assert!(text.contains(&filled), "S = r2 (041-R29 ④, 2026-11-14)");
    let empty: RlConfig = toml::from_str(&text.replacen(&filled, "\nzerg = \"\"\n", 1)).unwrap();
    let err = format!("{:#}", empty.validate().unwrap_err());
    assert!(err.contains("[kl.reference] zerg is empty"), "an empty S refuses to start: {err}");
    let t: RlConfig = toml::from_str(&text).unwrap();
    t.validate().unwrap_or_else(|e| panic!("{e:#}"));
    let base = "checkpoints/m2-s2e2/ckpt-11387";
    assert_eq!(t.kl.reference, KlReference { zerg: Some(s.into()), terran: Some(base.into()), protoss: Some(base.into()), random: Some(base.into()) });
    let cc = &t.curriculum;
    assert_eq!((cc.start_stage, cc.auto_advance), (Stage::C1, false));
    assert_eq!((cc.c1_ladder, cc.c1_t0, cc.c1_t1, cc.c1_self, cc.c1_snapshot, cc.c1_anchor), (1.0, 0.0, 0.0, 0.0, 0.0, 0.0));
    assert_eq!(cc.ladder_pick, LadderPickSetting::All(LadderPick::Fixed));
    let groups: Vec<(&str, f64)> = cc.ladder_groups.iter().map(|(g, x)| (g.as_str(), *x)).collect();
    assert_eq!(groups, [("protoss", 0.28), ("random", 0.20), ("terran", 0.24), ("zerg", 0.28)]);
    assert_eq!(cc.r_value_stages, vec![Stage::C0, Stage::C1]);
    let t1: Vec<(&str, &[Race])> = t.t1.iter().map(|b| (b.name.as_str(), b.races.as_slice())).collect();
    assert_eq!(t1, [("ualbertabot", &[Race::Terran, Race::Protoss, Race::Zerg][..]), ("steamhammer", &[Race::Terran, Race::Zerg][..]), ("stardust", &[Race::Protoss][..]), ("zzzkbot", &[Race::Zerg][..]), ("iron", &[Race::Terran][..])]);
    let race_group = |r: Race| match r {
        Race::Terran => "terran",
        Race::Protoss => "protoss",
        Race::Zerg => "zerg",
        _ => "random",
    };
    assert!(t.ladder.iter().all(|e| e.weight.is_some() && e.group == race_group(e.race)), "every entry weighted, in its race's group");
    assert!(t.ladder.iter().all(|e| e.build.is_none_or(|b| matches!(b, uranium_agent::T0Build::Standard | uranium_agent::T0Build::ZergRush))));
    assert!(t.ladder.iter().all(|e| !e.name.as_deref().unwrap_or_default().starts_with("mcrave")));
    assert_eq!(t.ladder.len(), 17);
    let mut d: RlConfig = toml::from_str(include_str!("../../../config/rl_m2.toml")).unwrap();
    d.kl.reference = t.kl.reference.clone();
    {
        let dc = &mut d.curriculum;
        (dc.start_stage, dc.auto_advance, dc.r_value_stages) = (cc.start_stage, cc.auto_advance, cc.r_value_stages.clone());
        (dc.c1_ladder, dc.c1_t0, dc.c1_t1, dc.c1_self) = (cc.c1_ladder, cc.c1_t0, cc.c1_t1, cc.c1_self);
        (dc.ladder_pick, dc.ladder_groups) = (cc.ladder_pick.clone(), cc.ladder_groups.clone());
    }
    (d.t1, d.ladder) = (t.t1.clone(), t.ladder.clone());
    assert_eq!(d, t);
    let back: RlConfig = serde_json::from_str(&serde_json::to_string(&t).unwrap()).unwrap();
    assert_eq!(back, t, "the serialised config (checkpoint) reads back the same");
}

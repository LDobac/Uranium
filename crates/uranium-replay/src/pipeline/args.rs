use std::collections::HashMap;
use std::path::{Path, PathBuf};
use anyhow::Result;

pub const BRANCH_SRC: &str = "branch";
pub const DISTILL_SRC: &str = "distill";

fn isolated(src: &str) -> bool {
    src == BRANCH_SRC || src == DISTILL_SRC
}

pub fn check_build_args(src: &str, relabel: bool, memory_log: Option<&Path>, out: Option<&Path>, bc_shards_root: Option<&Path>) -> Result<()> {
    anyhow::ensure!((src == "dagger") == relabel, "--relabel goes with --src dagger and only with it (041-R22)");
    if let Some(d) = memory_log {
        anyhow::ensure!(d.is_dir(), "--memory-log {}: not a folder (041-R28)", d.display());
        anyhow::ensure!(relabel || isolated(src), "--memory-log without --relabel goes with --src {BRANCH_SRC} or --src {DISTILL_SRC} only (041-R28: branch trunk shards are not BC data; 044-R2)");
    }
    anyhow::ensure!(src != DISTILL_SRC || memory_log.is_some(), "--src {DISTILL_SRC} needs --memory-log <the plan's memory folder> (044-R2: a step per logged model decision)");
    if isolated(src) {
        let Some(out) = out else {
            anyhow::bail!("--src {src} needs an explicit --out outside the BC shards (e.g. <shared>/eval/branch-<date>/shards, 041-R28; <shared>/datasets/replays/distill_pre/d<R>, 044-R2)")
        };
        if let Some(root) = bc_shards_root {
            anyhow::ensure!(!path_within(out, root), "--src {src}: --out {} is inside the BC shards {} (041-R28, 044-R2: these shards are not BC data)", out.display(), root.display());
        }
    }
    Ok(())
}

pub fn check_games_arg(src: &str, relabel: Option<uranium_scripted::T0Build>, games: bool) -> Result<()> {
    let tp = relabel == Some(uranium_scripted::T0Build::TerranTp);
    anyhow::ensure!(!tp || games, "--relabel terran_tp needs --games <the job's games.jsonl> (022-R2 / 022-AC3: lobby races; only players[0], the model, is a view)");
    anyhow::ensure!(!games || tp || (src == "botgen" && relabel.is_none()), "--games goes with --relabel terran_tp or --src botgen (the 022 teacher's botgen games, 022-R12) only");
    Ok(())
}

pub fn check_isolated_build(src: &str, records: &Path, time_limit: Option<u32>) -> Result<()> {
    if !isolated(src) {
        return Ok(());
    }
    anyhow::ensure!(time_limit.is_some(), "--src {src} needs --time-limit <the plan's max_frames> (051-R1, 051-R13 4: trunks end at the time limit; 044-R2: distill games end at the 044-R1 time limit 14400)");
    let name = records.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    anyhow::ensure!(name.starts_with(src) && name.ends_with(".jsonl"), "--records {}: the file name must start with `{src}` and end with .jsonl (the resume scan reads `<source>*.jsonl`; 051-R13 4, 044-R2)", records.display());
    let dir = records.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
    for other in ["branches.jsonl", "points.jsonl"] {
        anyhow::ensure!(!dir.join(other).exists(), "--records {}: its folder holds {other} (the 051-R12 output folder); give the build records a folder of their own (e.g. <out folder>/build/)", records.display());
    }
    Ok(())
}

pub fn path_within(p: &Path, root: &Path) -> bool {
    let abs = |x: &Path| std::path::absolute(x).unwrap_or_else(|_| x.to_path_buf());
    let (p, root) = (abs(p), abs(root));
    let (mut a, b) = (p.components(), root.components());
    b.into_iter().all(|rc| {
        a.next().is_some_and(|pc| if cfg!(windows) { pc.as_os_str().eq_ignore_ascii_case(rc.as_os_str()) } else { pc == rc })
    })
}

pub fn unique_stems(files: &[PathBuf]) -> Result<()> {
    let mut seen: HashMap<String, &PathBuf> = HashMap::new();
    for f in files {
        let stem = f.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        if let Some(first) = seen.insert(stem.clone(), f) {
            anyhow::bail!("--memory-log: {} and {} have the same name {stem}: build one plan folder (with its memory folder) at a time (041-R28)", first.display(), f.display());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::filter::filter_rules;

    #[test]
    fn memory_log_without_relabel_only_for_branch_051_r13_041_r28() {
        let dir = std::env::temp_dir();
        let root = dir.join("shared").join("datasets").join("replays").join("shards");
        let eval = dir.join("shared").join("eval").join("branch-20261113").join("shards");
        let ok = |src, relabel, mem: Option<&Path>, out: Option<&Path>| check_build_args(src, relabel, mem, out, Some(&root)).is_ok();
        assert!(ok("branch", false, Some(&dir), Some(&eval)), "a branch trunk build");
        assert!(ok("branch", false, Some(&dir), Some(&eval)) && check_build_args("branch", false, Some(&dir), Some(&eval), None).is_ok());
        assert!(ok("dagger", true, Some(&dir), None) && ok("dagger", true, None, None) && ok("botgen", false, None, None));
        for src in ["botgen", "bot", "scr_ladder", "human"] {
            assert!(!ok(src, false, Some(&dir), Some(&eval)), "{src}");
        }
        assert!(!ok("branch", false, Some(&dir), None));
        assert!(!ok("branch", false, Some(&dir), Some(&root.join("branch"))));
        assert!(!ok("branch", false, None, Some(&root)));
        if cfg!(windows) {
            let upper = PathBuf::from(root.to_string_lossy().to_uppercase()).join("x");
            assert!(!ok("branch", false, Some(&dir), Some(&upper)));
        }
        let rec = |n: &str| dir.join(n);
        assert!(check_isolated_build("branch", &rec("branch_build.jsonl"), Some(11_568)).is_ok());
        assert!(check_isolated_build("branch", &rec("branch_build.jsonl"), None).is_err());
        assert!(check_isolated_build("branch", &rec("build_records.jsonl"), Some(11_568)).is_err());
        assert!(check_isolated_build("botgen", &rec("x.jsonl"), None).is_ok(), "other sources unchanged");
        let o = dir.join(format!("u6-branch-records-{}", std::process::id()));
        std::fs::create_dir_all(o.join("build")).unwrap();
        std::fs::write(o.join("branches.jsonl"), "").unwrap();
        assert!(check_isolated_build("branch", &o.join("branch_build.jsonl"), Some(11_568)).is_err());
        assert!(check_isolated_build("branch", &o.join("build").join("branch_build.jsonl"), Some(11_568)).is_ok());
        std::fs::remove_dir_all(&o).ok();
        assert!(!ok("dagger", false, Some(&dir), None), "dagger needs --relabel (041-R22)");
        assert!(!ok("botgen", true, None, None), "--relabel only with dagger (041-R22)");
        assert!(!ok("branch", false, Some(Path::new("no/such/memory")), Some(&eval)));
        assert!(!path_within(&eval, &root) && path_within(&root, &root));
    }

    #[test]
    fn distill_build_args_and_filter_rules_044_r2_040_r24() {
        let dir = std::env::temp_dir();
        let root = dir.join("shared").join("datasets").join("replays").join("shards");
        let pre = dir.join("shared").join("datasets").join("replays").join("distill_pre").join("d1");
        let ok = |src, relabel, mem: Option<&Path>, out: Option<&Path>| check_build_args(src, relabel, mem, out, Some(&root)).is_ok();
        assert!(ok("distill", false, Some(&dir), Some(&pre)));
        assert!(!ok("distill", false, None, Some(&pre)), "the steps are the logged decisions");
        assert!(!ok("distill", false, Some(&dir), None), "explicit --out");
        assert!(!ok("distill", false, Some(&dir), Some(&root.join("distill_m2").join("d1"))), "not inside the BC shards");
        assert!(!ok("distill", false, Some(&dir), Some(&root)));
        assert!(!ok("distill", true, Some(&dir), Some(&pre)), "--relabel only with dagger");
        assert!(!ok("distill", false, Some(Path::new("no/such/memory")), Some(&pre)));
        let rec = |n: &str| dir.join(n);
        assert!(check_isolated_build("distill", &rec("distill_d1.jsonl"), Some(14_400)).is_ok());
        assert!(check_isolated_build("distill", &rec("distill_d1.jsonl"), None).is_err());
        assert!(check_isolated_build("distill", &rec("branch_d1.jsonl"), Some(14_400)).is_err());
        assert!(check_isolated_build("branch", &rec("distill_d1.jsonl"), Some(14_400)).is_err());
        assert_eq!(filter_rules("distill", false), (true, true));
        for (src, relabel) in [("stardata", false), ("scr_public", false), ("community", false), ("personal", false), ("user", false), ("scr_ladder", false), ("bot", false), ("botgen", false), ("dagger", true), ("branch", false)] {
            let before = (matches!(src, "bot" | "botgen" | "dagger" | BRANCH_SRC), relabel || src == BRANCH_SRC);
            assert_eq!(filter_rules(src, relabel), before, "{src}");
        }
    }

    #[test]
    fn games_arg_rules_022_r2() {
        use uranium_scripted::T0Build::{TerranReact, TerranTp};
        assert!(check_games_arg("dagger", Some(TerranTp), false).is_err());
        assert!(check_games_arg("dagger", Some(TerranTp), true).is_ok());
        assert!(check_games_arg("botgen", None, true).is_ok());
        assert!(check_games_arg("botgen", None, false).is_ok());
        assert!(check_games_arg("dagger", Some(TerranReact), true).is_err());
        assert!(check_games_arg("dagger", Some(TerranReact), false).is_ok());
        for src in ["distill", "branch", "stardata"] {
            assert!(check_games_arg(src, None, true).is_err(), "{src}");
        }
    }
}

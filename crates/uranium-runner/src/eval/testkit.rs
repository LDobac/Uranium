use std::collections::HashSet;
use std::path::PathBuf;
use super::ledger::Rec;

pub(super) fn rec(id: u64, outcome: &str, detail: Option<&str>) -> Rec {
    Rec { id, outcome: outcome.into(), detail: detail.map(str::to_string), line: serde_json::json!({}) }
}

pub(super) fn job_dir(tag: &str, games: u32) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("uranium-eval-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("plans")).unwrap();
    for (i, name) in ["a", "b"].iter().enumerate() {
        std::fs::write(dir.join(format!("plans/{i}.toml")), format!("name='{name}'\nsave_replay=true\n[[matchup]]\nmap='both'\ngames={games}\np0={{kind='t0',race='terran'}}\np1={{kind='t0',race='zerg'}}\n")).unwrap();
    }
    std::fs::write(dir.join("job.toml"), "stages=['plans/0.toml','plans/1.toml']\nhosts='hosts.toml'\n").unwrap();
    dir
}

pub(super) fn all_stages() -> HashSet<usize> {
    HashSet::from([0, 1])
}

//! Markdown summary of session records (M2 report: 011-AC1..AC4).

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use anyhow::Result;
use serde_json::Value;

fn pct(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * q).round() as usize]
}

pub fn summarize(input: &Path, out: &Path, title: &str) -> Result<()> {
    let text = std::fs::read_to_string(input)?;
    let recs: Vec<Value> = text.lines().filter(|l| !l.trim().is_empty()).map(serde_json::from_str).collect::<Result<_, _>>()?;

    let mut games: BTreeMap<u64, Vec<&Value>> = BTreeMap::new();
    for r in &recs {
        games.entry(r["game_id"].as_u64().unwrap_or(0)).or_default().push(r);
    }
    let mut reasons: BTreeMap<String, usize> = BTreeMap::new();
    let (mut frames, mut fps, mut wait_ms, mut ser_us, mut minutes) = (vec![], vec![], vec![], vec![], vec![]);
    let mut errors = Vec::new();
    let mut incomplete = 0;
    let mut slot_wins: BTreeMap<u64, usize> = BTreeMap::new();
    for (id, rs) in &games {
        if rs.len() != 2 {
            incomplete += 1;
        }
        // One line per game: take the record with stats from the first slot.
        let r = rs[0];
        *reasons.entry(r["reason"].as_str().unwrap_or("?").to_string()).or_default() += 1;
        if let Some(w) = r["winner"].as_u64() {
            *slot_wins.entry(w).or_default() += 1;
        }
        for r in rs {
            if let Some(e) = r["error"].as_str() {
                errors.push(format!("game {id} slot {}: {e}", r["slot"]));
            }
            let s = &r["stats"];
            if let (Some(gf), Some(gt)) = (s["game_frames"].as_f64(), s["game_time_ms"].as_f64()) {
                if gt > 0.0 {
                    fps.push(gf / (gt / 1e3));
                }
                frames.push(gf);
                minutes.push(gf * 42.0 / 60_000.0);
                if let Some(obs) = s["observations"].as_f64().filter(|o| *o > 0.0) {
                    wait_ms.push(s["agent_wait_ms"].as_f64().unwrap_or(0.0) / obs);
                }
                ser_us.push(s["serialize_us_p99"].as_f64().unwrap_or(0.0));
            }
        }
    }
    let first_minerals: Vec<u64> = recs.iter().filter_map(|r| r["first_frame_minerals"].as_u64()).collect();
    let failed: u64 = recs.iter().filter_map(|r| r["failed_commands"].as_u64()).sum();
    let violations: u64 = recs.iter().filter_map(|r| r["visibility_violations"].as_u64()).sum();
    let cmds: u64 = recs.iter().filter_map(|r| r["commands"].as_u64()).sum();

    let mut md = String::new();
    writeln!(md, "# {}\n", if title.is_empty() { "Selfmatch summary" } else { title })?;
    writeln!(md, "- source: `{}`", input.display())?;
    writeln!(md, "- games: {} (sessions {}, games missing a side: {incomplete})", games.len(), recs.len())?;
    writeln!(md, "- end reasons: {reasons:?}")?;
    writeln!(md, "- wins by player slot: {slot_wins:?}")?;
    writeln!(md, "- session errors: {}", errors.len())?;
    writeln!(md, "- game length (frames): p50 {:.0}, p90 {:.0}, max {:.0}", pct(&mut frames.clone(), 0.5), pct(&mut frames.clone(), 0.9), pct(&mut frames, 1.0))?;
    writeln!(md, "- game length (game minutes @fastest): p50 {:.1}, max {:.1}", pct(&mut minutes.clone(), 0.5), pct(&mut minutes, 1.0))?;
    writeln!(md, "- engine speed excluding agent wait (frames/s, 011-AC3): p10 {:.0}, p50 {:.0}, p90 {:.0}", pct(&mut fps.clone(), 0.1), pct(&mut fps.clone(), 0.5), pct(&mut fps, 0.9))?;
    writeln!(md, "- bridge serialize+send p99 per observation (µs, 011-AC2): p50 {:.0}, max {:.0}", pct(&mut ser_us.clone(), 0.5), pct(&mut ser_us, 1.0))?;
    writeln!(md, "- agent round trip per observation (ms): p50 {:.3}", pct(&mut wait_ms, 0.5))?;
    writeln!(md, "- commands {cmds}, failed {failed} ({:.2}%)", if cmds > 0 { failed as f64 * 100.0 / cmds as f64 } else { 0.0 })?;
    writeln!(md, "- visibility violations (enemy units outside our vision, 010-AC6): {violations}")?;
    writeln!(md, "- mineral fields visible in first observation: min {:?}, max {:?}", first_minerals.iter().min(), first_minerals.iter().max())?;
    if !errors.is_empty() {
        writeln!(md, "\n## Errors\n")?;
        for e in errors.iter().take(20) {
            writeln!(md, "- {e}")?;
        }
    }
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(out, md)?;
    Ok(())
}

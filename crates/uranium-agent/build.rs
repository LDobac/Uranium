use std::path::{Path, PathBuf};
use std::process::Command;

fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let o = Command::new("git").args(args).current_dir(dir).output().ok()?;
    if !o.status.success() {
        return None;
    }
    let s = String::from_utf8(o.stdout).ok()?.trim().to_string();
    (!s.is_empty()).then_some(s)
}

fn main() {
    let dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"));
    println!("cargo:rerun-if-changed=build.rs");
    let commit = git(&dir, &["rev-parse", "HEAD"]).filter(|c| c.len() == 40 && c.bytes().all(|b| b.is_ascii_hexdigit()));
    let mut watch = vec!["HEAD".to_string(), "logs/HEAD".to_string(), "packed-refs".to_string()];
    if let Some(r) = git(&dir, &["symbolic-ref", "-q", "HEAD"]) {
        watch.push(r);
    }
    for w in watch {
        if let Some(p) = git(&dir, &["rev-parse", "--git-path", &w]) {
            let p = PathBuf::from(p);
            let p = if p.is_absolute() { p } else { dir.join(p) };
            if p.exists() {
                println!("cargo:rerun-if-changed={}", p.display());
            }
        }
    }
    println!("cargo:rustc-env=URANIUM_GIT={}", commit.as_deref().unwrap_or("unknown"));
}

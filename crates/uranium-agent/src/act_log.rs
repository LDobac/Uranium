use std::io::BufWriter;
use std::path::PathBuf;
use serde::Serialize;
use uranium_store::elog;

#[derive(Serialize, Debug, Clone, Default, PartialEq)]
pub struct ActLogStatus {
    pub path: String,
    pub complete: bool,
    pub error: Option<String>,
}

pub(crate) struct ActLogFile {
    pub(crate) path: PathBuf,
    w: Option<uranium_env::branch::ActLogWriter<BufWriter<std::fs::File>>>,
    pub(crate) error: Option<String>,
    pub(crate) complete: bool,
}

impl ActLogFile {
    pub(crate) fn create(path: PathBuf) -> Self {
        let open = || -> std::io::Result<std::fs::File> {
            if let Some(d) = path.parent() {
                std::fs::create_dir_all(d)?;
            }
            std::fs::File::create(&path)
        };
        match open() {
            Ok(f) => Self { path, w: Some(uranium_env::branch::ActLogWriter::new(BufWriter::new(f))), error: None, complete: false },
            Err(e) => {
                let mut s = Self { path, w: None, error: None, complete: false };
                s.stop(format!("open: {e}"));
                s
            }
        }
    }

    fn stop(&mut self, why: String) {
        elog!("[agent] WARNING: act log {} stopped (051-R2): {why}", self.path.display());
        self.w = None;
        self.error.get_or_insert(why);
    }

    pub(crate) fn push(&mut self, line: &uranium_env::branch::ObsLine) {
        let Some(w) = self.w.as_mut() else { return };
        if let Err(e) = w.push(line) {
            self.stop(format!("{e:#}"));
        }
    }

    pub(crate) fn finish(&mut self, end: uranium_env::branch::ActLogEnd) {
        let Some(w) = self.w.take() else { return };
        let res = w.finish(end).and_then(|w| w.into_inner().map_err(|e| anyhow::anyhow!("{}", e.error()))).and_then(|f| f.sync_all().map_err(Into::into));
        match res {
            Ok(()) => self.complete = true,
            Err(e) => self.stop(format!("{e:#}")),
        }
    }

    pub(crate) fn abort(&mut self) {
        if let Some(w) = self.w.take() {
            drop(w);
        }
    }
}

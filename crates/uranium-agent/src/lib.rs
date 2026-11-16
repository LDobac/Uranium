pub mod broker;
pub mod infer;
pub mod model_policy;
pub mod summary;
#[cfg(feature = "testkit")]
pub mod testkit;
mod act_log;
mod record;
mod rejection;
mod routing;
mod scripted;
mod server;

use std::sync::Arc;
use anyhow::{Result, bail};
use uranium_proto::*;
use crate::infer::InferenceHandle;
use crate::model_policy::{LearnerHook, ModelPolicy, ModelStats};
pub use uranium_scripted::T0Build;
pub use uranium_match::branch::{BranchBoard, GameBranch, fnv1a};

pub use routing::{Routes, SessionEnd, game_result, mutual_leave, EndBoard, ConcedeBoard, T0Builds, spec_index, name_index};
pub use server::{end_outcome, ServeOptions, serve_addr, serve, session};
pub use scripted::{ScriptedStats, TpRecord, ReactRecord, ScriptedPolicy};
pub use act_log::ActLogStatus;
pub use record::{git_commit, SessionRecord, RecordSink, open_sink};
pub use rejection::RejectionLog;

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum PolicyKind {
    /// T0 rule-based bot for our own race, picked at game start.
    ScriptedT0,
    /// Checkpoint policy, Terran only.
    Model,
}

#[derive(Clone)]
pub enum PolicySpec {
    ScriptedT0,
    Model(InferenceHandle),
    Learner(Arc<dyn LearnerHook>),
}

impl std::fmt::Debug for PolicySpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ScriptedT0 => write!(f, "ScriptedT0"),
            Self::Model(h) => write!(f, "Model({h:?})"),
            Self::Learner(l) => write!(f, "Learner({:?})", l.handle()),
        }
    }
}

impl PolicySpec {
    pub fn label(&self) -> String {
        match self {
            Self::ScriptedT0 => format!("t0@{}", env!("CARGO_PKG_VERSION")),
            Self::Model(h) => infer::model_policy_label(&h.checkpoint_id, h.sample),
            Self::Learner(l) => format!("learner:{}:{:?}", l.handle().checkpoint_id, l.handle().sample).to_lowercase(),
        }
    }
}

pub trait Policy: Send {
    fn on_privileged(&mut self, _p: &PrivilegedSummary) {}
    fn on_frame(&mut self, f: &RawFrame) -> Result<Act>;
    fn model_stats(&mut self) -> Option<ModelStats> {
        None
    }
    fn scripted_stats(&mut self) -> Option<ScriptedStats> {
        None
    }
    fn on_end(&mut self, _outcome: Option<uranium_env::reward::Outcome>, _final_frame: Option<&RawFrame>) {}
    fn on_abort(&mut self) {
        self.on_end(None, None)
    }
    fn act_log_extra(&mut self) -> Option<model_policy::LineExtra> {
        None
    }
    fn branch_record(&mut self) -> Option<model_policy::BranchSessionPart> {
        None
    }
}

pub const REALTIME_ACT_FRAMES: u32 = 24;

#[allow(clippy::too_many_arguments)]
pub fn make_policy(spec: &PolicySpec, game_id: u64, info: &GameInfo, build: T0Build, realtime: bool, concede: Option<(Arc<uranium_match::concede::ConcedeGame>, u8)>, branch: Option<(Arc<GameBranch>, u8)>) -> Result<Box<dyn Policy>> {
    let model = |p: ModelPolicy| -> Result<ModelPolicy> {
        let p = if realtime { p.with_zero_last_failed() } else { p };
        let p = match concede.clone() {
            Some((game, side)) => p.with_concede(game, side),
            None => p,
        };
        match branch.clone() {
            Some((game, index)) => p.with_branch(game, index),
            None => Ok(p),
        }
    };
    let terran = || -> Result<()> {
        let me = info.players.iter().find(|p| p.is_self).map(|p| p.race);
        if me != Some(Race::Terran) {
            bail!("model policy plays Terran only, got {me:?}");
        }
        Ok(())
    };
    Ok(match spec {
        PolicySpec::ScriptedT0 => Box::new(ScriptedPolicy::new(info, build)?),
        PolicySpec::Model(h) => {
            terran()?;
            Box::new(model(ModelPolicy::new(h, info)?)?)
        }
        PolicySpec::Learner(l) => {
            terran()?;
            let (z, observer) = l.session(game_id, info);
            Box::new(model(ModelPolicy::new(l.handle(), info)?.with_observer(z, l.reward_config(), observer))?)
        }
    })
}

impl Policy for Box<dyn uranium_scripted::Bot> {
    fn on_frame(&mut self, f: &RawFrame) -> Result<Act> {
        Ok(self.act(f))
    }
}

impl Policy for ModelPolicy {
    fn on_privileged(&mut self, p: &PrivilegedSummary) {
        ModelPolicy::on_privileged(self, p)
    }

    fn on_frame(&mut self, f: &RawFrame) -> Result<Act> {
        ModelPolicy::on_frame(self, f)
    }

    fn model_stats(&mut self) -> Option<ModelStats> {
        Some(self.stats())
    }

    fn on_end(&mut self, outcome: Option<uranium_env::reward::Outcome>, final_frame: Option<&RawFrame>) {
        ModelPolicy::on_end(self, outcome, final_frame)
    }

    fn on_abort(&mut self) {
        ModelPolicy::on_abort(self)
    }

    fn act_log_extra(&mut self) -> Option<model_policy::LineExtra> {
        ModelPolicy::act_log_extra(self)
    }

    fn branch_record(&mut self) -> Option<model_policy::BranchSessionPart> {
        ModelPolicy::branch_record(self)
    }
}

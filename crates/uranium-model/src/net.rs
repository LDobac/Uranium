//! `UraniumNet` (032 §1, §6). Encoders and heads run over all B (or B·T) rows in parallel; only
//! the Core walks the time axis.

use burn::module::Module;
use burn::tensor::{Device, Tensor};

use crate::batch::{ActionBatch, ActionColumns, ActionTableTensors, ObsBatch, PrivBatch};
use crate::config::{CoreKind, UraniumNetConfig};
use crate::core::{Core, GtrxlCore, LstmCore};
use crate::entity::EntityEncoder;
use crate::heads::{Heads, HeadsOut, Mode};
use crate::scalar::{ScalarEncoder, ScalarOut};
use crate::spatial::{SpatialEncoder, SpatialOut};
use crate::value::ValueHeads;

/// Builds a core from the config (lets `UraniumNet::new` stay generic over `Core`, 032-AC6).
pub trait CoreInit: Core + Sized {
    const KIND: CoreKind;
    fn build(d_in: usize, cfg: &UraniumNetConfig, device: &Device) -> Self;
}

impl CoreInit for LstmCore {
    const KIND: CoreKind = CoreKind::Lstm;
    fn build(d_in: usize, cfg: &UraniumNetConfig, device: &Device) -> Self {
        LstmCore::new(d_in, &cfg.core, device)
    }
}

impl CoreInit for GtrxlCore {
    const KIND: CoreKind = CoreKind::Gtrxl;
    fn build(d_in: usize, cfg: &UraniumNetConfig, device: &Device) -> Self {
        GtrxlCore::new(d_in, &cfg.core, device)
    }
}

/// The `Module` bound must be spelled out for Burn's derive to treat `C` as a sub-module.
#[derive(Module, Debug)]
pub struct UraniumNet<C: Module + Core> {
    entity: EntityEncoder,
    spatial: SpatialEncoder,
    scalar: ScalarEncoder,
    core: C,
    heads: Heads,
    value: ValueHeads,
    table: ActionTableTensors,
}

#[derive(Debug, Clone)]
pub struct ActOutput<S> {
    pub action: ActionBatch,
    /// [B] joint log-prob of the used heads (032-R10), at the sampling temperature.
    pub log_prob: Tensor<1>,
    /// [B,6] per-head log-prob, unused heads 0 (032-R15, behaviour log-prob for 042).
    pub head_log_prob: Tensor<2>,
    /// [B] joint entropy of the used heads.
    pub entropy: Tensor<1>,
    /// [B,4] win, build, units, value (no privileged input).
    pub value: Tensor<2>,
    pub state: S,
}

#[derive(Debug, Clone)]
pub struct SeqOutput<S> {
    /// Rows in B·T order (row = b*T + t).
    pub heads: HeadsOut,
    /// [B·T,4]
    pub value: Tensor<2>,
    pub state: S,
}

/// Where [`UraniumNet::forward_seq_cut`] stops the gradient.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cut {
    None,
    /// Entity encoder outputs.
    Entity,
    /// Spatial encoder outputs.
    Spatial,
    /// All encoder outputs (backward runs through core, heads and value).
    Encoders,
    /// Encoders and the core output (backward runs through heads and value only).
    Core,
}

struct Encoded {
    core_in: Tensor<2>,
    e: Tensor<3>,
    sp: SpatialOut,
    sc: ScalarOut,
}

impl<C: CoreInit> UraniumNet<C> {
    pub fn new(cfg: &UraniumNetConfig, columns: &ActionColumns, device: &Device) -> Self {
        assert_eq!(columns.len(), cfg.input.actions, "action table size != config.input.actions");
        assert_eq!(cfg.input.map_size % 16, 0, "map_size must be a multiple of 16 (four stride-2 stages)");
        let d_e = cfg.entity.d_model;
        let d_in = d_e + cfg.spatial.pool + cfg.scalar.hidden;
        let core = C::build(d_in, cfg, device);
        let d_core = core.hidden();
        Self {
            entity: EntityEncoder::new(&cfg.input, &cfg.entity, device),
            spatial: SpatialEncoder::new(&cfg.input, &cfg.spatial, d_e, device),
            scalar: ScalarEncoder::new(&cfg.input, &cfg.scalar, device),
            heads: Heads::new(&cfg.input, &cfg.heads, &cfg.spatial, d_core, cfg.scalar.hidden, d_e, device),
            value: ValueHeads::new(&cfg.input, &cfg.value, d_core, cfg.scalar.z_out, cfg.scalar.race_emb, device),
            core,
            table: columns.to_tensors(device),
        }
    }
}

impl<C: Core> UraniumNet<C> {
    fn encode(&self, obs: &ObsBatch) -> Encoded {
        self.encode_cut(obs, Cut::None)
    }

    fn encode_cut(&self, obs: &ObsBatch, cut: Cut) -> Encoded {
        let mut ent = self.entity.forward(obs);
        crate::prof::mark("fwd.entity");
        if matches!(cut, Cut::Entity | Cut::Encoders | Cut::Core) {
            ent.e = ent.e.detach();
            ent.pool = ent.pool.detach();
        }
        let mut sp = self.spatial.forward(obs, ent.e.clone());
        crate::prof::mark("fwd.spatial");
        let mut sc = self.scalar.forward(obs);
        crate::prof::mark("fwd.scalar");
        if matches!(cut, Cut::Spatial | Cut::Encoders | Cut::Core) {
            sp.pool = sp.pool.detach();
            sp.s8 = sp.s8.detach();
            sp.skips = sp.skips.map(|s| s.detach());
        }
        if matches!(cut, Cut::Encoders | Cut::Core) {
            sc.c = sc.c.detach();
            sc.z = sc.z.detach();
            sc.race = sc.race.detach();
        }
        let core_in = Tensor::cat(vec![ent.pool, sp.pool.clone(), sc.c.clone()], 1);
        Encoded { core_in, e: ent.e, sp, sc }
    }

    pub fn init_state(&self, batch: usize, device: &Device) -> C::State {
        self.core.init_state(batch, device)
    }

    /// One inference step for B games; sampling happens inside this forward (032-R9). Takes no
    /// privileged input (032-R11). `temp` 0 picks the argmax of every head (032-R15, 050-R1).
    pub fn act(&self, obs: &ObsBatch, state: C::State, temp: f32) -> ActOutput<C::State> {
        let enc = self.encode(obs);
        let (h, state) = self.core.step(enc.core_in, state);
        let out = self.heads.forward(Mode::Sample { temp }, h.clone(), enc.sc.c.clone(), enc.e, &enc.sp, obs, &self.table);
        let value = self.value.forward(h, enc.sc.z, enc.sc.race, None);
        ActOutput { log_prob: out.logp.sum(), head_log_prob: out.logp.stack(), entropy: out.entropy.sum(), action: out.chosen, value, state }
    }

    /// Teacher-forced training pass over `t` steps per sequence. `obs`, `labels`, `privileged`
    /// hold B·T rows in b-major order (032 §6).
    pub fn forward_seq(&self, obs: &ObsBatch, labels: &ActionBatch, privileged: Option<&PrivBatch>, s0: C::State, t: usize) -> SeqOutput<C::State> {
        self.forward_seq_cut(obs, labels, privileged, s0, t, Cut::None)
    }

    /// [`UraniumNet::forward_seq`] with the gradient stopped at `cut` (profiling the backward
    /// pass module by module; the forward values are the same).
    pub fn forward_seq_cut(&self, obs: &ObsBatch, labels: &ActionBatch, privileged: Option<&PrivBatch>, s0: C::State, t: usize, cut: Cut) -> SeqOutput<C::State> {
        let bt = obs.batch();
        assert!(t > 0 && bt % t == 0, "batch rows must be B*T");
        let b = bt / t;
        let enc = self.encode_cut(obs, cut);
        let d = enc.core_in.dims()[1];
        let (h, state) = self.core.forward_seq(enc.core_in.reshape([b, t, d]), s0);
        let mut h = h.reshape([bt, self.core.hidden()]);
        crate::prof::mark("fwd.core");
        if cut == Cut::Core {
            h = h.detach();
        }
        let heads = self.heads.forward(Mode::Teacher(labels), h.clone(), enc.sc.c.clone(), enc.e, &enc.sp, obs, &self.table);
        let value = self.value.forward(h, enc.sc.z, enc.sc.race, privileged);
        crate::prof::mark("fwd.value");
        SeqOutput { heads, value, state }
    }

    pub fn core(&self) -> &C {
        &self.core
    }
}

"""Cue-flow diagnostics of a checkpoint (reports/jepa_survey.md 7, 10): `python -m uranium diag <cmd>`.

Read-only: teacher-forced passes over shard games, offline labels from the privileged columns (the
model input stays own-vision, 030-R9), no training and no change to the model.

Commands (`uranium.diag.cli`):
- `dump`: one pass per game (optionally with counterfactual passes), per-row features and labels,
  activations at the stages (focus token, e_pool, sp_pool, c, core input, GTrXL sub-points, h_t)
- `probes`: cue probes per enemy race on a dump (ridge + MLP(64), game-level CV, bootstrap CIs,
  shuffled-label control, gain over a nuisance baseline)
- `intervene`: counterfactual Δ P(bunker) / marines / 2nd barracks from a dump with `--cf`
- `spatial`: per-layer positive fraction and never-active channels (`uranium.model.monitor`)
- `core-split`: probes at the GTrXL sub-points and the gate decomposition
- `summary`: the per-lineage-point summary JSON (`uranium.diag.summary`)
"""

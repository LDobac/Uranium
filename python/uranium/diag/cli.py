"""`python -m uranium diag <cmd>` (uranium.diag; 050-R9 lineage measurements, 041-R24 cue probes):
the subcommands and their arguments. Data paths come from the arguments (`--dataset` names
resolve under URANIUM_SHARED_DIR, 041-R9; `--shards` are folders as given); nothing is read from a
fixed location. The command modules are imported when the parser is built (not at package import)."""

import importlib

COMMANDS = {
    "dump": ("dump", "teacher-forced pass: per-row features, labels and stage activations (--cf: counterfactual passes)"),
    "probes": ("probes", "cue probes per enemy race on a dump (ridge + MLP, game CV, bootstrap, nuisance gain)"),
    "intervene": ("intervene", "counterfactual effects on P(bunker) / marines / 2nd barracks from a dump with --cf"),
    "spatial": ("spatial", "032-R25 dead-channel statistics (pos / dead / const) of the ReLU-bearing layers"),
    "core-split": ("core_split", "probes at the GTrXL sub-points and the gate decomposition, on a dump"),
    "summary": ("summary", "the per-lineage-point summary JSON from the other commands' outputs"),
    "plan-dump": ("plan_dump", "jepa_planning S0 rows: h', JEPA target, privileged, own state, action, policy log-probs, g_k, win_open"),
    "plan-s0": ("plan_s0", "jepa_planning S0 predictors U / A / A+d / H / S and the M0-M3d tables on plan-dump folders"),
    "plan-branch": ("plan_branch", "051-R13 branch games: paired effects (trunk bootstrap, split-half), M3u and M2c against the S0b predictor A"),
}


def add_parser(sub) -> None:
    p = sub.add_parser("diag", help="cue-flow diagnostics of a checkpoint (reports/jepa_survey.md 7, 10)")
    dsub = p.add_subparsers(dest="diag_cmd", required=True)
    for name, (module, text) in COMMANDS.items():
        mod = importlib.import_module(f"uranium.diag.{module}")
        q = dsub.add_parser(name, help=text, description=(mod.__doc__ or text).strip().split("\n\n")[0])
        mod.add_args(q)
        q.set_defaults(fn=mod.run)

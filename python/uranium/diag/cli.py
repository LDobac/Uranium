import importlib

COMMANDS = {
    "dump": ("dump", "teacher-forced pass: per-row features, labels and stage activations (--cf: counterfactual passes)"),
    "probes": ("probes", "cue probes per enemy race on a dump (ridge + MLP, game CV, bootstrap, nuisance gain)"),
    "intervene": ("intervene", "counterfactual effects on P(bunker) / marines / 2nd barracks from a dump with --cf"),
    "spatial": ("spatial", "dead-channel statistics (pos / dead / const) of the ReLU-bearing layers"),
    "core-split": ("core_split", "probes at the GTrXL sub-points and the gate decomposition, on a dump"),
    "summary": ("summary", "the per-lineage-point summary JSON from the other commands' outputs"),
    "plan-dump": ("plan_dump", "JEPA planning S0 rows: h', JEPA target, privileged, own state, action, log-probs, g_k, win_open"),
    "plan-s0": ("plan_s0", "JEPA planning S0 predictors U / A / A+d / H / S and the M0-M3d tables on plan-dump folders"),
    "plan-branch": ("plan_branch", "branch games: paired effects (trunk bootstrap, split-half), M3u and M2c vs the S0b predictor A"),
}


def add_parser(sub) -> None:
    p = sub.add_parser("diag", help="cue-flow and planning diagnostics of a checkpoint")
    dsub = p.add_subparsers(dest="diag_cmd", required=True)
    for name, (module, text) in COMMANDS.items():
        mod = importlib.import_module(f"uranium.diag.{module}")
        q = dsub.add_parser(name, help=text, description=(mod.__doc__ or text).strip().split("\n\n")[0])
        mod.add_args(q)
        q.set_defaults(fn=mod.run)

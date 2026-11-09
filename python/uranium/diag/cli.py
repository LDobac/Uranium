"""`python -m uranium diag <cmd>` (uranium.diag): the subcommands and their arguments. Data paths
come from the arguments (`--dataset` names resolve under URANIUM_SHARED_DIR, 041-R9; `--shards`
are folders as given); nothing is read from a fixed location."""

from uranium.diag import core_split, dump, intervene, probes, spatial, summary

COMMANDS = {
    "dump": (dump, "teacher-forced pass: per-row features, labels and stage activations (--cf: counterfactual passes)"),
    "probes": (probes, "cue probes per enemy race on a dump (ridge + MLP, game CV, bootstrap, nuisance gain)"),
    "intervene": (intervene, "counterfactual effects on P(bunker) / marines / 2nd barracks from a dump with --cf"),
    "spatial": (spatial, "per-layer positive fraction and never-active channels of the ReLU-bearing layers"),
    "core-split": (core_split, "probes at the GTrXL sub-points and the gate decomposition, on a dump"),
    "summary": (summary, "the per-lineage-point summary JSON from the other commands' outputs"),
}


def add_parser(sub) -> None:
    p = sub.add_parser("diag", help="cue-flow diagnostics of a checkpoint (reports/jepa_survey.md 7, 10)")
    dsub = p.add_subparsers(dest="diag_cmd", required=True)
    for name, (mod, text) in COMMANDS.items():
        q = dsub.add_parser(name, help=text, description=(mod.__doc__ or text).strip().split("\n\n")[0])
        mod.add_args(q)
        q.set_defaults(fn=mod.run)

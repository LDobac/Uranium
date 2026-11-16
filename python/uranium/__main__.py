import argparse
import sys

from uranium import env


def main(argv: list[str] | None = None) -> int:
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetConsoleCtrlHandler(None, False)
    import signal

    if hasattr(signal, "SIGUSR1"):
        import faulthandler

        faulthandler.register(signal.SIGUSR1, all_threads=True)
    env.load()
    env.apply_malloc()
    env.apply_malloc_arenas()
    p = argparse.ArgumentParser(prog="python -m uranium", description="Uranium training and inference (PyTorch)")
    sub = p.add_subparsers(dest="cmd", required=True)
    from uranium.bc import eval as bc_eval
    from uranium.bc import feed_digest
    from uranium.bc import profile as bc_profile
    from uranium.bc import run as bc_run

    feed_digest.add_parser(sub)
    bc_run.add_parser(sub)
    bc_eval.add_parser(sub)
    bc_profile.add_parser(sub)
    from uranium.distill import label as distill_label

    distill_label.add_parser(sub)
    from uranium.agent import eval_worker, random_checkpoint
    from uranium.agent import serve as agent_serve

    agent_serve.add_parsers(sub)
    eval_worker.add_parsers(sub)
    random_checkpoint.add_parser(sub)
    from uranium.rl import run as rl_run

    rl_run.add_parser(sub)
    from uranium.bench import cli as bench_cli

    bench_cli.add_parser(sub)
    from uranium.diag import cli as diag_cli

    diag_cli.add_parser(sub)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

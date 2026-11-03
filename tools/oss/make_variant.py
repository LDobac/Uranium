"""021 T1 config variant (2026-11-03 user decision, AMB-042-15): copy a deployed T1 bot folder and
change its opening config only, so every game host gets the same variant without hand edits.

The variant folder is `<bot><variant>` beside the original, `VERSION` is
`<bot><variant>@<original version>-<variant>` (its 021-R3 opponent id); the binary stays the
original's (hash checked). One edit per run, either an exact text replacement that must occur once
or a 1-based inclusive line range replaced by new text. Prints what `reports/oss_bot_porting.md`
records per variant (original id, file, lines, before -> after, sha256 of the changed file).

Run with the stdlib only (Linux game hosts: `python3 -I`, Windows: the venv python):

  python3 -I tools/oss/make_variant.py --bot ualbertabot --variant 9pool \\
      --file bwapi-data/AI/UAlbertaBot_Config.txt \\
      --replace '"Zerg_ZerglingRush",' '"Zerg_9Pool",'
  python3 -I tools/oss/make_variant.py --bot steamhammer --variant overpool \\
      --file bwapi-data/AI/Steamhammer_5.3.6.json --lines 230-249 \\
      --text '            { "Weight" : 100, "Strategy" : "OverpoolSpeed" }'

The bots folder is `--bots`, else `URANIUM_OSS_BOTS_DIR`, else `build/oss_bots` of the repository.
An existing variant folder is left alone unless `--force` (then it is rebuilt in a `.part` folder
and replaced only once that is complete).
"""

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

BINARIES = ("module.dll", "module.so")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bots", type=Path, default=None)
    ap.add_argument("--bot", required=True, help="original bot folder name, e.g. steamhammer")
    ap.add_argument("--variant", required=True, help="variant tag, e.g. overpool (folder <bot><variant>)")
    ap.add_argument("--file", required=True, help="config file relative to the bot folder")
    edit = ap.add_mutually_exclusive_group(required=True)
    edit.add_argument("--replace", nargs=2, metavar=("OLD", "NEW"), help="exact text, must occur once")
    edit.add_argument("--lines", help="1-based inclusive range A-B, replaced by --text")
    ap.add_argument("--text", help="replacement lines for --lines (\\n separates lines)")
    ap.add_argument("--force", action="store_true", help="rebuild an existing variant folder")
    a = ap.parse_args()

    if not a.variant.isalnum() or not a.variant.islower():
        ap.error("--variant must be lowercase letters / digits (it is part of the folder name)")
    if a.lines is not None and a.text is None:
        ap.error("--lines needs --text")
    bots = a.bots or Path(os.environ.get("URANIUM_OSS_BOTS_DIR") or Path(__file__).resolve().parents[2] / "build" / "oss_bots")
    src = bots / a.bot
    dst = bots / f"{a.bot}{a.variant}"
    version = (src / "VERSION").read_text(encoding="utf-8").strip()
    name, _, ver = version.partition("@")
    if name != a.bot or not ver:
        sys.exit(f"{src / 'VERSION'}: expected '{a.bot}@<version>', got {version!r}")
    if dst.exists() and not a.force:
        sys.exit(f"{dst} exists (use --force to rebuild it)")

    tmp = dst.with_name(dst.name + ".part")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp)
    target = tmp / a.file
    raw = target.read_bytes()
    text = raw.decode("utf-8")
    nl = "\r\n" if "\r\n" in text else "\n"
    if a.replace:
        old, new = a.replace
        n = text.count(old)
        if n != 1:
            shutil.rmtree(tmp)
            sys.exit(f"{a.file}: {old!r} occurs {n} times, expected once")
        line = text[: text.index(old)].count("\n") + 1
        where, before, after = f"line {line}", old, new
        text = text.replace(old, new)
    else:
        lo, _, hi = a.lines.partition("-")
        lo, hi = int(lo), int(hi or lo)
        lines = text.splitlines(keepends=True)
        if not 1 <= lo <= hi <= len(lines):
            shutil.rmtree(tmp)
            sys.exit(f"{a.file}: lines {lo}-{hi} outside 1-{len(lines)}")
        before = "".join(lines[lo - 1 : hi]).rstrip("\r\n")
        after = a.text.replace("\\n", "\n")
        new_lines = [l + nl for l in after.split("\n")]
        text = "".join(lines[: lo - 1] + new_lines + lines[hi:])
        where = f"lines {lo}-{hi}"
    target.write_bytes(text.encode("utf-8"))
    variant_id = f"{a.bot}{a.variant}@{ver}-{a.variant}"
    (tmp / "VERSION").write_text(variant_id, encoding="utf-8")
    for b in BINARIES:
        if (src / b).exists() and sha256(src / b) != sha256(tmp / b):
            shutil.rmtree(tmp)
            sys.exit(f"{b} differs from the original after the copy")
    # Swap only once the new folder is complete: an existing variant is kept until then.
    old = dst.with_name(dst.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    if dst.exists():
        dst.rename(old)
    tmp.rename(dst)
    if old.exists():
        shutil.rmtree(old)

    print(f"[oss] variant {variant_id} from {version} in {dst}")
    print(f"  file: {a.file} ({where})")
    print(f"  before: {before}")
    print(f"  after:  {after}")
    print(f"  sha256 {a.file}: {sha256(dst / a.file)}")
    for b in BINARIES:
        if (dst / b).exists():
            print(f"  {b} sha256 (= original): {sha256(dst / b)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

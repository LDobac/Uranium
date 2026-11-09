"""G0 decision-granularity measurement (model-2 plan P0 "G0"; reports/m6_rl_rootcause.md section 4).

Question: is the rush-defence failure limited by the action space (031 delay buckets, the 12-unit
selection of 032/034, one command per decision) or by what the policy learned? Compared in the
window 2:30-4:30 (frames 3,600-6,480, 24 frames/s as everywhere in this repo) of rush games:

- people: S2 human shards (040-R9), TvZ on the target maps, labels as stored (040-R3, R4);
- the 021-R8 scripted teacher: its botgen rush games (replays), its own commands;
- Checkpoint 1 (bc-dagger-r1/ckpt-1137): its rush-check and DAgger-round-2 games (replays).

Replays go through the same 040 labeler as the shards (`uranium-replay granularity --replays`: one
label per command, same-frame commands become delay-1 chains, 031-R11 floor buckets), but with no
5-minute bar, so games lost early to a rush stay in. Everything is read from the per-game dumps of
`uranium-replay granularity` (labels + compact own/enemy states).

Metrics (per group of games):
1. delay-bucket histogram per action group (labelled delay = frames to the next command label);
2. 12-cap saturation: share of unit-command selections of exactly 12 units while more than 12
   eligible own SCVs/marines (completed, not loaded) were within 8 tiles of the selection centre;
3. same-frame bursts: share of command labels at most 2 frames after the previous one;
4. SCV micro: distinct SCVs given a combat command (move/attack/smart/repair/hold/stop/patrol)
   per 5 s bin with enemy army near the main; time from bunker completion to the first
   Repair/Load (also right clicks on the bunker);
5. command labels per 1,000 frames (and, for model games, the agent's own decisions/1,000 frames
   from the eval results).

Game groups: "early" = the 020-R21 seed condition (opponent's first target is our main,
`uranium._core.start_slots`) for plan games; for people (no seed) a rush is observed contact: at
least 2 zerglings within 20 tiles of our start by 3:30 and at least 4 by 4:30. "held" = at least 8
own SCVs at 8:00 (or won before 8:00), the RL mini-game criterion.

    # dumps (CPU; replays need the OpenBW launcher and bridge from .env)
    uranium-replay granularity <shard dirs> --matchup TvZ --stage S2 \\
        --families fighting_spirit,circuit_breaker --lo 2400 --out <out>/human_s2_tvz.jsonl.gz
    uranium-replay granularity --replays --list <list> --source <tag> --lo 2400 --out <out>/<tag>.jsonl.gz
    # report
    python tools/eval/granularity.py --human <out>/human_s2_tvz.jsonl.gz \\
        --teacher <out>/teacher_botgen.jsonl.gz:<plan.toml> \\
        --model <out>/model_check.jsonl.gz:<plan.toml>[:<results dir>] ... \\
        --out <out>/granularity.md --json <out>/granularity.json

Plan games are matched to seeds by the replay name `g<n>` (game n = seeds[n - 1], matchups in
order); `--opponents` keeps the opponents of the 050-R6 rush check (T0 zerg_rush, UAB 12-pool).
"""
import argparse
import collections
import gzip
import json
import statistics as st
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FPS = 24
WIN = (3600, 6480)  # 2:30-4:30
BIN = 5 * FPS  # 5 s
TILE = 32
NEAR_SEL = 8 * TILE  # 12-cap eligibility radius around the selection centre
NEAR_MAIN = 20 * TILE  # enemy army "near the main"
CONTACT_BY = 3 * 60 * FPS + 30 * FPS  # 3:30
AT_8M = 8 * 60 * FPS
BUCKETS = [(1, 1), (2, 2), (3, 4), (5, 8), (10, 16), (20, 32)]
GROUPS = {
    "Repair": {"Repair_Unit"},
    "Attack": {"Attack_Pos", "Attack_Unit"},
    "Move": {"Move_Pos", "Move_Unit"},
    "Smart": {"Smart_Pos", "Smart_Unit"},
    "Load": {"Load_Unit"},
    "Hold/Stop/Patrol": {"Hold", "Stop", "Patrol_Pos"},
    "Gather": {"Gather_Unit", "ReturnCargo"},
    "Build_Bunker": {"Build_Bunker"},
}
COMBAT = {"Repair", "Attack", "Move", "Smart", "Load", "Hold/Stop/Patrol"}
UNIT_CMD = COMBAT | {"Gather"}


def tables():
    acts = tomllib.loads((ROOT / "crates/uranium-env/tables/actions.toml").read_text(encoding="utf-8"))["action"]
    units = tomllib.loads((ROOT / "crates/uranium-env/tables/unit_types.toml").read_text(encoding="utf-8"))["rows"]
    name = {a["id"]: a["name"] for a in acts}
    uid = {u["name"]: u["id"] for u in units}
    worker_or_air = {u["id"] for u in units if "Worker" in u.get("flags", []) or "Flyer" in u.get("flags", []) or "Building" in u.get("flags", [])}
    army = {u["id"] for u in units if u.get("race") == "Zerg" and "Attack" in u.get("flags", []) and u["id"] not in worker_or_air}
    return name, uid, army


ANAME, UID, ZERG_ARMY = tables()
SCV, MARINE, BUNKER, LING = UID["Terran_SCV"], UID["Terran_Marine"], UID["Terran_Bunker"], UID["Zerg_Zergling"]
COMPLETED = 1


def group_of(action):
    n = ANAME.get(action, "?")
    if n.startswith("Train_"):
        return "Train"
    if n == "NoOp":
        return "NoOp"
    for g, names in GROUPS.items():
        if n in names:
            return g
    return "Other"


def d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def load(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def plan_info(plan_path):
    """game n (1-based) -> (seed, opponent)."""
    p = tomllib.loads(Path(plan_path).read_text(encoding="utf-8"))
    opp = []
    for m in p["matchup"]:
        p1 = m["p1"]
        opp += [p1.get("name") or p1.get("build")] * m["games"]
    return {n: (s, opp[n - 1] if n - 1 < len(opp) else None) for n, s in enumerate(p["seeds"], 1)}


def results_rows(results_dir):
    """game_id % 100000 -> eval result row of the model player (results/*.jsonl)."""
    rows = {}
    for f in sorted(Path(results_dir).glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("model"):
                    rows.setdefault(r["game_id"] // 100000, {})[r["game_id"] % 100000] = r
    return rows


class Game:
    """Per-game derived facts and metric contributions."""

    def __init__(self, d):
        self.d = d
        self.start = tuple(d["start"]) if d.get("start") else None
        self.states = d["states"]
        self.frames = [s["f"] for s in self.states]
        self.labels = d["labels"]
        self.contact, self.lings_max = self._contact()
        end_early = d["end_frame"] < AT_8M - 32
        self.held = (d["scv_at"] or 0) >= 8 or (end_early and d["result"] == 1)

    def state_at(self, f):
        """Last state at or before frame f."""
        import bisect
        i = bisect.bisect_right(self.frames, f) - 1
        return self.states[i] if i >= 0 else None

    def _contact(self):
        first, mx = None, 0
        if not self.start:
            return None, 0
        for s in self.states:
            n = sum(1 for (_, t, x, y) in s["enemy"] if t == LING and d2((x, y), self.start) <= NEAR_MAIN ** 2)
            if n >= 2 and first is None:
                first = s["f"]
            if s["f"] <= WIN[1]:
                mx = max(mx, n)
        return first, mx

    def army_near(self, s):
        return any(t in ZERG_ARMY and d2((x, y), self.start) <= NEAR_MAIN ** 2 for (_, t, x, y) in s["enemy"]) if self.start else False

    def bunker_done(self):
        for s in self.states:
            for u in s["own"]:
                if u[1] == BUNKER and u[4] & COMPLETED:
                    return s["f"], u[0]
        return None, None


def collect(games):
    m = collections.defaultdict(collections.Counter)
    delays = collections.defaultdict(list)
    sel = collections.Counter()
    sel_sizes = []
    bursts = collections.Counter()
    micro_bins, micro_all_bins = [], []
    fight_cmds, fight_delays = [], []
    bunker_resp, bunker_n = [], collections.Counter()
    cmd_rate = []
    for g in games:
        lab = [l for l in g.labels if WIN[0] <= l[0] < WIN[1]]
        cmds = [l for l in lab if group_of(l[1]) != "NoOp"]
        cmd_rate.append(1000 * len(cmds) / (WIN[1] - WIN[0]))
        prev = None
        for l in cmds:
            grp = group_of(l[1])
            delays[grp].append(l[2])
            if prev is not None:
                bursts["n"] += 1
                bursts["le2"] += l[0] - prev <= 2
            prev = l[0]
            if grp not in UNIT_CMD or not l[4]:
                continue
            s = g.state_at(l[0])
            if s is None:
                continue
            own = {u[0]: u for u in s["own"]}
            units = [own[i] for i in l[4] if i in own]
            if not units or any(u[1] not in (SCV, MARINE) for u in units):
                continue
            cx = sum(u[2] for u in units) / len(units)
            cy = sum(u[3] for u in units) / len(units)
            elig = sum(1 for u in s["own"] if u[1] in (SCV, MARINE) and u[4] & COMPLETED and not u[6] and d2((u[2], u[3]), (cx, cy)) <= NEAR_SEL ** 2)
            fight = g.army_near(s)
            for k in ("all", "fight") if fight else ("all",):
                sel[f"{k}_n"] += 1
                sel[f"{k}_eq12"] += len(l[4]) == 12
                sel[f"{k}_elig_gt12"] += elig > 12
                sel[f"{k}_sat12"] += len(l[4]) == 12 and elig > 12
                sel[f"{k}_ge6"] += len(l[4]) >= 6
            sel_sizes.append(len(l[4]))
        # SCV micro per 5 s bin with enemy army near the main.
        for b0 in range(WIN[0], WIN[1], BIN):
            st_in = [s for s in g.states if b0 <= s["f"] < b0 + BIN]
            if not any(g.army_near(s) for s in st_in):
                continue
            scvs, scvs_all = set(), set()
            n_cmd = 0
            for l in cmds:
                if b0 <= l[0] < b0 + BIN:
                    n_cmd += 1
                    if group_of(l[1]) in COMBAT:
                        fight_delays.append(l[2])
                if b0 <= l[0] < b0 + BIN and l[4]:
                    s = g.state_at(l[0])
                    own = {u[0]: u[1] for u in s["own"]} if s else {}
                    ids = {i for i in l[4] if own.get(i) == SCV}
                    scvs_all |= ids
                    if group_of(l[1]) in COMBAT:
                        scvs |= ids
            fight_cmds.append(n_cmd)
            micro_bins.append(len(scvs))
            micro_all_bins.append(len(scvs_all))
        # Bunker completion -> first Repair / Load (incl. right clicks on the bunker).
        bf, bid = g.bunker_done()
        if bf is not None and bf <= WIN[1]:
            bunker_n["done"] += 1
            first = {}
            for l in g.labels:
                if l[0] < bf:
                    continue
                n = ANAME.get(l[1], "")
                s = g.state_at(l[0])
                own = {u[0]: u[1] for u in s["own"]} if s else {}
                types = {own.get(i) for i in l[4]}
                if n == "Repair_Unit" or (n == "Smart_Unit" and l[5] == bid and SCV in types):
                    first.setdefault("repair", l[0] - bf)
                if n == "Load_Unit" or (n == "Smart_Unit" and l[5] == bid and MARINE in types):
                    first.setdefault("load", l[0] - bf)
            for k in ("repair", "load"):
                if k in first:
                    bunker_resp.append((k, first[k]))
                    bunker_n[k] += 1
    return dict(m=m, delays=delays, sel=sel, sel_sizes=sel_sizes, bursts=bursts, micro=micro_bins, micro_all=micro_all_bins, bunker_resp=bunker_resp, bunker_n=bunker_n, cmd_rate=cmd_rate, fight_cmds=fight_cmds, fight_delays=fight_delays)


def pct(a, b):
    return 100.0 * a / b if b else float("nan")


def med(xs):
    return st.median(xs) if xs else float("nan")


def mean(xs):
    return st.fmean(xs) if xs else float("nan")


def summarize(name, games, extra_rate=None):
    c = collect(games)
    out = {"group": name, "games": len(games), "held": sum(g.held for g in games)}
    hist = {}
    for grp in ["Repair", "Attack", "Move", "Smart", "Load", "Hold/Stop/Patrol", "Gather", "Build_Bunker", "Train", "Other"]:
        xs = c["delays"].get(grp, [])
        row = {"n": len(xs), "mean": mean(xs), "median": med(xs)}
        for lo, hi in BUCKETS:
            row[f"{lo}-{hi}" if lo != hi else str(lo)] = pct(sum(1 for x in xs if lo <= x <= hi), len(xs))
        hist[grp] = row
    out["delay"] = hist
    s = c["sel"]
    out["sel"] = {k: {"n": s[f"{k}_n"], "eq12": pct(s[f"{k}_eq12"], s[f"{k}_n"]), "elig_gt12": pct(s[f"{k}_elig_gt12"], s[f"{k}_n"]),
                      "sat12": pct(s[f"{k}_sat12"], s[f"{k}_n"]), "sat12_given_elig": pct(s[f"{k}_sat12"], s[f"{k}_elig_gt12"]),
                      "ge6": pct(s[f"{k}_ge6"], s[f"{k}_n"])} for k in ("all", "fight")}
    out["sel_size_mean"] = mean(c["sel_sizes"])
    out["burst_le2"] = pct(c["bursts"]["le2"], c["bursts"]["n"])
    out["micro_scv_per_5s"] = {"bins": len(c["micro"]), "combat_mean": mean(c["micro"]), "combat_p90": (sorted(c["micro"])[int(0.9 * (len(c["micro"]) - 1))] if c["micro"] else float("nan")), "any_mean": mean(c["micro_all"])}
    rep = [f for k, f in c["bunker_resp"] if k == "repair"]
    lod = [f for k, f in c["bunker_resp"] if k == "load"]
    out["bunker"] = {"done_by_4:30": c["bunker_n"]["done"], "repair": len(rep), "repair_s_median": med(rep) / FPS if rep else float("nan"),
                     "load": len(lod), "load_s_median": med(lod) / FPS if lod else float("nan")}
    out["cmd_per_1000f"] = mean(c["cmd_rate"])
    fd = c["fight_delays"]
    out["fight"] = {"cmds_per_5s": mean(c["fight_cmds"]), "combat_delay_mean": mean(fd), "combat_delay_le2": pct(sum(1 for x in fd if x <= 2), len(fd)),
                    "combat_delay_ge10": pct(sum(1 for x in fd if x >= 10), len(fd))}
    if extra_rate:
        out["agent_decisions_per_1000f"] = mean(extra_rate)
    return out


def fmt(x, d=1):
    return "-" if x != x else f"{x:.{d}f}"


def tables_md(rows):
    L = []
    L.append("### 1. Delay buckets per action group (share of labels, %; mean frames)\n")
    for r in rows:
        L.append(f"**{r['group']}** ({r['games']} games)\n")
        L.append("| group | n | mean | 1 | 2 | 3-4 | 5-8 | 10-16 | 20-32 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for grp, h in r["delay"].items():
            if h["n"]:
                L.append(f"| {grp} | {h['n']} | {fmt(h['mean'])} | " + " | ".join(fmt(h[k]) for k in ["1", "2", "3-4", "5-8", "10-16", "20-32"]) + " |")
        L.append("")
    L.append("### 2-5. Selection cap, bursts, SCV micro, rate\n")
    L.append("| group | games | held | sel n (fight) | =12 % | >12 eligible % | 12-cap sat. % (fight) | sat. given >12 eligible % | sel ≥6 % | mean sel size | burst ≤2f % | SCVs/5 s in fight (combat cmds) | p90 | SCVs/5 s any cmd | fight bins | bunker done ≤4:30 | → repair (n, median s) | → load (n, median s) | cmd labels /1000f | agent decisions /1000f | fight: cmds/5 s | fight: combat delay mean | ≤2 % | ≥10 % |")
    L.append("|---" * 24 + "|")
    for r in rows:
        a, f = r["sel"]["all"], r["sel"]["fight"]
        b = r["bunker"]
        mi = r["micro_scv_per_5s"]
        L.append(
            f"| {r['group']} | {r['games']} | {r['held']} | {a['n']} ({f['n']}) | {fmt(a['eq12'])} | {fmt(a['elig_gt12'])} | {fmt(a['sat12'])} ({fmt(f['sat12'])}) | "
            f"{fmt(a['sat12_given_elig'])} | {fmt(a['ge6'])} | {fmt(r['sel_size_mean'], 2)} | {fmt(r['burst_le2'])} | {fmt(mi['combat_mean'], 2)} | {mi['combat_p90']} | "
            f"{fmt(mi['any_mean'], 2)} | {mi['bins']} | {b['done_by_4:30']} | {b['repair']}, {fmt(b['repair_s_median'])} | {b['load']}, {fmt(b['load_s_median'])} | "
            f"{fmt(r['cmd_per_1000f'])} | {fmt(r.get('agent_decisions_per_1000f', float('nan')))} | "
            f"{fmt(r['fight']['cmds_per_5s'], 2)} | {fmt(r['fight']['combat_delay_mean'])} | {fmt(r['fight']['combat_delay_le2'])} | {fmt(r['fight']['combat_delay_ge10'])} |"
        )
    return "\n".join(L)


def plan_games(spec, opponents, kind):
    """`<dump.jsonl.gz>:<plan.toml>[:<results dir>]` -> [(Game, early, extra)]"""
    parts = spec.split(":")
    # Windows drive letters: rejoin "C" + ":\..." pieces.
    paths, cur = [], ""
    for p in parts:
        if len(p) == 1 and p.isalpha():
            cur = p + ":"
            continue
        paths.append(cur + p)
        cur = ""
    dump, plan = paths[0], paths[1]
    res = results_rows(paths[2]) if len(paths) > 2 else {}
    info = plan_info(plan)
    from uranium import _core
    out = []
    for d in load(dump):
        n = int(d["game"].lstrip("g"))
        seed, opp = info[n]
        if opponents and opp not in opponents:
            continue
        early = _core.start_slots(seed, 0)[2]
        extra = None
        for stage_rows in res.values():
            r = stage_rows.get(n)
            if r and r["model"].get("decisions"):
                extra = 1000 * r["model"]["decisions"] / max(1, r["end_frame"])
        out.append((Game(d), early, extra, opp))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--human", nargs="*", default=[])
    ap.add_argument("--teacher", nargs="*", default=[], help="<dump>:<plan>")
    ap.add_argument("--model", nargs="*", default=[], help="<dump>:<plan>[:<results dir>]")
    ap.add_argument("--opponents", default="zerg_rush,ualbertabot12poollings", help="plan opponents kept (empty = all)")
    ap.add_argument("--human-every", type=int, default=8, help="people all-games row: every n-th game (memory)")
    ap.add_argument("--out")
    ap.add_argument("--json")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    opps = {o for o in a.opponents.split(",") if o}
    rows = []
    # Streamed: every rush game is kept, of the rest only every --human-every-th (memory).
    human, rush, total = [], [], 0
    for h in a.human:
        for d in load(h):
            g = Game(d)
            if g.contact is not None and g.contact <= CONTACT_BY and g.lings_max >= 4:
                rush.append(g)
            if total % a.human_every == 0:
                human.append(g)
            total += 1
    if total:
        print(f"people: {total} games, {len(rush)} rush, {len(human)} in the all-games sample", file=sys.stderr)
        rows.append(summarize(f"people S2 TvZ (1/{a.human_every} sample of all)", human))
        rows.append(summarize("people rush (contact ≤3:30)", rush))
        rows.append(summarize("people rush, held", [g for g in rush if g.held]))
        rows.append(summarize("people rush, lost", [g for g in rush if not g.held]))
    for kind, specs in (("teacher", a.teacher), ("Checkpoint 1", a.model)):
        allg = [x for s in specs for x in plan_games(s, opps, kind)]
        if not allg:
            continue
        early = [x for x in allg if x[1]]
        late = [x for x in allg if not x[1]]
        rate = lambda xs: [x[2] for x in xs if x[2] is not None]
        rows.append(summarize(f"{kind} early", [x[0] for x in early], rate(early)))
        rows.append(summarize(f"{kind} early, held", [x[0] for x in early if x[0].held], rate([x for x in early if x[0].held])))
        rows.append(summarize(f"{kind} early, lost", [x[0] for x in early if not x[0].held], rate([x for x in early if not x[0].held])))
        rows.append(summarize(f"{kind} late", [x[0] for x in late], rate(late)))
        # Observed contact of the seed condition (sanity check of the people definition).
        obs = collections.Counter((x[1], x[0].contact is not None and x[0].contact <= CONTACT_BY and x[0].lings_max >= 4) for x in allg)
        print(f"{kind}: seed early x observed rush contact: {dict(obs)}", file=sys.stderr)
    md = tables_md(rows)
    print(md)
    if a.out:
        Path(a.out).write_text(md + "\n", encoding="utf-8")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

"""G0 decision-granularity measurement (model-2 plan P0 "G0"; reports/m6_rl_rootcause.md section 4).

Input of the deferred decision AMB-031-7 (031 "결정 단위", [SPEC DEFERRED(G0)]): does the 031-R3
12-unit selection cap bind in rush defence, or is the rush-defence failure set by what the policy
learned (031 delay buckets, one command per decision)? The measurement changes no action space;
whether to keep it is the user's AMB-031-7 decision. Compared in the window 2:30-4:30 (frames
3,600-6,480, 24 frames/s as everywhere in this repo) of rush games:

- people: S2 human shards (040-R9), TvZ on the target maps, labels as stored (040-R3, R4). These
  passed the 040 5-minute filter, so a person who lost to a rush in under 5 minutes is missing;
- the 021-R8 scripted teacher: its botgen rush games (replays), its own commands;
- Checkpoint 1 (bc-dagger-r1/ckpt-1137): its rush-check and DAgger-round-2 games (replays).

Replays go through the same 040 labeler as the shards (`uranium-replay granularity --replays`: one
label per command, same-frame commands become delay-1 chains, 031-R11 floor buckets), with no
5-minute bar, so games lost early to a rush stay in. Everything is read from the per-game dumps of
`uranium-replay granularity` (labels + compact own/enemy states).

Metrics (per group of games):
1. delay-bucket histogram per action group (labelled delay = frames to the next command label),
   with the games contributing, the most labels one game gives, and the per-game median of the
   game's mean delay;
2. 12-cap saturation: share of unit-command selections of exactly 12 units while more than 12
   eligible own SCVs/marines (completed, not loaded) were within 8 tiles of the selection centre;
3. same-frame bursts: share of command labels at most 2 frames after the previous one;
4. SCV micro: distinct SCVs given a combat command (move/attack/smart/repair/hold/stop/patrol;
   a right click on a neutral unit is a mining click = Gather, not combat) per 5 s bin with enemy
   army near the main; bunker completion -> first load (Load_Unit or a marine's right click on the
   bunker); the bunker's first HP drop -> first repair of that bunker (Repair_Unit or an SCV's right
   click on it), with the games that got no repair in the window (censored);
5. command labels per 1,000 frames in the window (and, for model games, the agent's own
   decisions per 1,000 frames over the whole game from the eval results).

Game groups: "early" = the 050-R6 / 020-R21 seed condition (opponent's first target is our main,
`eval_common.start_condition`) for plan games; people have no seed, so their "rush" is observed
contact: at least 2 zerglings within 20 tiles of our start by 3:30 and at least 4 by 4:30.
Held (docs/measurement_protocol.md, 050-R9): `held_8m` = at least 8 own SCVs at 8:00, a game that
ended before 8:00 fails; `held_8m_winpass` also passes a game won before 8:00. Both counts are
reported; held / lost groups are split by `held_8m_winpass`, because the teacher wins about half of
its botgen rush games before 8:00 (the opponent is eliminated or leaves) and `held_8m` would count
those defences as failures.

    # dumps (CPU; replays need the OpenBW launcher and bridge from .env)
    uranium-replay granularity <shard dirs> --matchup TvZ --stage S2 \\
        --families fighting_spirit,circuit_breaker --lo 2400 --out <out>/human_s2_tvz.jsonl.gz
    uranium-replay granularity --replays --list <list> --source <tag> --lo 2400 --out <out>/<tag>.jsonl.gz
    # report
    python tools/eval/granularity.py --human <out>/human_s2_tvz.jsonl.gz \\
        --teacher <out>/teacher_botgen.jsonl.gz <plan.toml> \\
        --model <out>/model_check.jsonl.gz <job>/plans/01-rush5check.toml <job> 1 ... \\
        --out <out>/granularity.md --json <out>/granularity.json

Plan games are matched to seeds by the replay name `g<n>` (game n = seeds[n - 1], matchups in
order); `--opponents` keeps the opponents of the 050-R6 rush check (T0 zerg_rush, UAB 12-pool).
"""
import argparse
import bisect
import collections
import gzip
import json
import statistics as st
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_common as ec  # noqa: E402

ROOT = ec.ROOT
FPS = ec.FPS
WIN = (3600, 6480)  # 2:30-4:30
BIN = 5 * FPS  # 5 s
TILE = 32
NEAR_SEL = 8 * TILE  # 12-cap eligibility radius around the selection centre
NEAR_MAIN = 20 * TILE  # enemy army "near the main"
CONTACT_BY = 3 * 60 * FPS + 30 * FPS  # 3:30
AT_8M = ec.CUT
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
GROUP_ORDER = ["Repair", "Attack", "Move", "Smart", "Load", "Hold/Stop/Patrol", "Gather", "Build_Bunker", "Train", "Other"]
COMBAT = {"Repair", "Attack", "Move", "Smart", "Load", "Hold/Stop/Patrol"}
UNIT_CMD = COMBAT | {"Gather"}
DEFAULT_OPPONENTS = "t0 zerg_rush,ualbertabot12poollings"


def load_tables():
    acts = tomllib.loads((ROOT / "crates/uranium-env/tables/actions.toml").read_text(encoding="utf-8"))["action"]
    units = tomllib.loads((ROOT / "crates/uranium-env/tables/unit_types.toml").read_text(encoding="utf-8"))["rows"]
    name = {a["id"]: a["name"] for a in acts}
    uid = {u["name"]: u["id"] for u in units}
    not_army = {u["id"] for u in units if {"Worker", "Flyer", "Building"} & set(u.get("flags", []))}
    army = {u["id"] for u in units if u.get("race") == "Zerg" and "Attack" in u.get("flags", []) and u["id"] not in not_army}
    return name, uid, army


ANAME, UID, ZERG_ARMY = load_tables()
AID = {v: k for k, v in ANAME.items()}
SCV, MARINE, BUNKER, LING = UID["Terran_SCV"], UID["Terran_Marine"], UID["Terran_Bunker"], UID["Zerg_Zergling"]
COMPLETED = 1
# Dump rows: label (frame, action, delay, queued, selected, target unit); own unit (id, type, x, y,
# flags, hp, loaded); enemy unit (id, type, x, y).


def group_of(label, state=None):
    """Action group of a label; a right click on a unit that is neither own nor enemy in the decision
    state (a mineral field or geyser) is a mining click (Gather)."""
    n = ANAME.get(label[1], "?")
    if n.startswith("Train_"):
        return "Train"
    if n == "NoOp":
        return "NoOp"
    if n == "Smart_Unit" and state is not None and label[5] is not None:
        if label[5] not in {u[0] for u in state["own"]} and label[5] not in {e[0] for e in state["enemy"]}:
            return "Gather"
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


class Game:
    """One dumped game and its derived facts."""

    def __init__(self, d):
        self.d = d
        self.start = tuple(d["start"]) if d.get("start") else None
        self.states = d["states"]
        self.frames = [s["f"] for s in self.states]
        self.labels = d["labels"]
        self.contact, self.lings_max = self._contact()
        scv8 = d["scv_at"]
        # docs/measurement_protocol.md: held_8m fails a game that ended before 8:00; the old
        # rush_summary.py passed one won before 8:00 (held_8m_winpass).
        self.held_8m = scv8 is not None and scv8 >= 8
        self.held_8m_winpass = self.held_8m or (d["end_frame"] < AT_8M - 32 and d["result"] == 1)

    @property
    def rush(self):
        """People rush rule (observed contact): >= 2 zerglings within 20 tiles of our start by 3:30
        and >= 4 at some point by 4:30."""
        return self.contact is not None and self.contact <= CONTACT_BY and self.lings_max >= 4

    def state_at(self, f):
        """Last state at or before frame f."""
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

    def bunker(self):
        """(completion frame, id, first HP-drop frame) of the first own bunker completed in the dump."""
        done, bid, hp, drop = None, None, None, None
        for s in self.states:
            for u in s["own"]:
                if bid is None and u[1] == BUNKER and u[4] & COMPLETED:
                    done, bid = s["f"], u[0]
                if bid is not None and u[0] == bid:
                    if hp is not None and u[5] < hp and drop is None:
                        drop = s["f"]
                    hp = u[5]
        return done, bid, drop


def game_stats(g):
    """Per-game contributions to every metric."""
    out = {"delays": collections.defaultdict(list), "sel": collections.Counter(), "sizes": [], "bursts": [0, 0],
           "micro": [], "micro_any": [], "fight_cmds": [], "fight_delays": [], "bunker": {}}
    lab = [l for l in g.labels if WIN[0] <= l[0] < WIN[1]]
    groups = [(l, group_of(l, g.state_at(l[0]))) for l in lab]
    cmds = [(l, grp) for l, grp in groups if grp != "NoOp"]
    out["cmd_rate"] = 1000 * len(cmds) / (WIN[1] - WIN[0])
    prev = None
    for l, grp in cmds:
        out["delays"][grp].append(l[2])
        if prev is not None:
            out["bursts"][0] += 1
            out["bursts"][1] += l[0] - prev <= 2
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
        for k in ("all", "fight") if g.army_near(s) else ("all",):
            c = out["sel"]
            c[f"{k}_n"] += 1
            c[f"{k}_eq12"] += len(l[4]) == 12
            c[f"{k}_elig_gt12"] += elig > 12
            c[f"{k}_sat12"] += len(l[4]) == 12 and elig > 12
            c[f"{k}_ge6"] += len(l[4]) >= 6
        out["sizes"].append(len(l[4]))
    # SCV micro per 5 s bin with enemy army near the main.
    for b0 in range(WIN[0], WIN[1], BIN):
        if not any(g.army_near(s) for s in g.states if b0 <= s["f"] < b0 + BIN):
            continue
        scvs, scvs_any, n_cmd = set(), set(), 0
        for l, grp in cmds:
            if not b0 <= l[0] < b0 + BIN:
                continue
            n_cmd += 1
            if grp in COMBAT:
                out["fight_delays"].append(l[2])
            s = g.state_at(l[0])
            own = {u[0]: u[1] for u in s["own"]} if s else {}
            ids = {i for i in l[4] if own.get(i) == SCV}
            scvs_any |= ids
            if grp in COMBAT:
                scvs |= ids
        out["fight_cmds"].append(n_cmd)
        out["micro"].append(len(scvs))
        out["micro_any"].append(len(scvs_any))
    # Bunker: completion -> first load; first HP drop -> first repair of that bunker.
    done, bid, drop = g.bunker()
    if done is not None and done <= WIN[1]:
        b = out["bunker"]
        b["done"] = True
        last = g.labels[-1][0] if g.labels else done
        for l in g.labels:
            n = ANAME.get(l[1], "")
            s = g.state_at(l[0])
            types = {({u[0]: u[1] for u in s["own"]} if s else {}).get(i) for i in l[4]}
            if l[0] >= done and "load" not in b and (n == "Load_Unit" or (n == "Smart_Unit" and l[5] == bid and MARINE in types)):
                b["load"] = l[0] - done
            if drop is not None and l[0] >= drop and "repair" not in b and l[5] == bid and (n == "Repair_Unit" or (n == "Smart_Unit" and SCV in types)):
                b["repair"] = l[0] - drop
        if drop is not None and drop <= last:
            b["damaged"] = True
    return out


def pct(a, b):
    return 100.0 * a / b if b else float("nan")


def med(xs):
    v = ec.median(xs)
    return float("nan") if v is None else v


def mean(xs):
    return st.fmean(xs) if xs else float("nan")


def summarize(name, games, extra_rate=None):
    per = [game_stats(g) for g in games]
    out = {"group": name, "games": len(games), "held_8m": sum(g.held_8m for g in games), "held_8m_winpass": sum(g.held_8m_winpass for g in games)}
    hist = {}
    for grp in GROUP_ORDER:
        xs = [x for p in per for x in p["delays"].get(grp, [])]
        counts = [len(p["delays"].get(grp, [])) for p in per]
        row = {"n": len(xs), "games": sum(1 for c in counts if c), "max_per_game": max(counts, default=0), "mean": mean(xs),
               "per_game_median_of_mean": med([mean(p["delays"][grp]) for p in per if p["delays"].get(grp)])}
        for lo, hi in BUCKETS:
            row[f"{lo}-{hi}" if lo != hi else str(lo)] = pct(sum(1 for x in xs if lo <= x <= hi), len(xs))
        hist[grp] = row
    out["delay"] = hist
    s = collections.Counter()
    for p in per:
        s.update(p["sel"])
    out["sel"] = {k: {"n": s[f"{k}_n"], "eq12": pct(s[f"{k}_eq12"], s[f"{k}_n"]), "elig_gt12": pct(s[f"{k}_elig_gt12"], s[f"{k}_n"]),
                      "sat12": pct(s[f"{k}_sat12"], s[f"{k}_n"]), "sat12_given_elig": pct(s[f"{k}_sat12"], s[f"{k}_elig_gt12"]),
                      "ge6": pct(s[f"{k}_ge6"], s[f"{k}_n"]), "games_with_sat12": sum(1 for p in per if p["sel"][f"{k}_sat12"])} for k in ("all", "fight")}
    out["sel_size_mean"] = mean([x for p in per for x in p["sizes"]])
    out["burst_le2"] = pct(sum(p["bursts"][1] for p in per), sum(p["bursts"][0] for p in per))
    micro = [x for p in per for x in p["micro"]]
    out["micro_scv_per_5s"] = {"bins": len(micro), "games": sum(1 for p in per if p["micro"]), "combat_mean": mean(micro),
                               "combat_p90": sorted(micro)[int(0.9 * (len(micro) - 1))] if micro else float("nan"),
                               "per_game_median": med([mean(p["micro"]) for p in per if p["micro"]]),
                               "any_mean": mean([x for p in per for x in p["micro_any"]])}
    bk = [p["bunker"] for p in per if p["bunker"]]
    load = [b["load"] for b in bk if "load" in b]
    dmg = [b for b in bk if b.get("damaged")]
    rep = [b["repair"] for b in dmg if "repair" in b]
    out["bunker"] = {"done_by_4:30": len(bk), "load": len(load), "load_s_median": med(load) / FPS if load else float("nan"),
                     "damaged": len(dmg), "repair": len(rep), "repair_censored": len(dmg) - len(rep),
                     "repair_s_median": med(rep) / FPS if rep else float("nan")}
    out["cmd_per_1000f"] = mean([p["cmd_rate"] for p in per])
    fd = [x for p in per for x in p["fight_delays"]]
    out["fight"] = {"cmds_per_5s": mean([x for p in per for x in p["fight_cmds"]]), "combat_delay_mean": mean(fd),
                    "combat_delay_le2": pct(sum(1 for x in fd if x <= 2), len(fd)), "combat_delay_ge10": pct(sum(1 for x in fd if x >= 10), len(fd))}
    if extra_rate:
        out["agent_decisions_per_1000f_whole_game"] = mean(extra_rate)
    return out


def fmt(x, d=1):
    return "-" if x != x else f"{x:.{d}f}"


def tables_md(rows):
    L = ["### 1. Delay buckets per action group (share of labels, %; mean frames; games = games with such labels, max/game = most labels from one game, game med = per-game median of the game's mean delay)\n"]
    for r in rows:
        L.append(f"**{r['group']}** ({r['games']} games)\n")
        L.append("| group | n | games | max/game | mean | game med | 1 | 2 | 3-4 | 5-8 | 10-16 | 20-32 |")
        L.append("|---" * 12 + "|")
        for grp, h in r["delay"].items():
            if h["n"]:
                L.append(f"| {grp} | {h['n']} | {h['games']} | {h['max_per_game']} | {fmt(h['mean'])} | {fmt(h['per_game_median_of_mean'])} | "
                         + " | ".join(fmt(h[k]) for k in ["1", "2", "3-4", "5-8", "10-16", "20-32"]) + " |")
        L.append("")
    L.append("### 2-5. Selection cap, bursts, SCV micro, rate\n")
    L.append("| group | games | held_8m (winpass) | sel n (fight) | =12 % | >12 eligible % | 12-cap sat. % (fight) | sat. given >12 eligible % | games with sat. | sel ≥6 % | mean sel size | burst ≤2f % "
             "| SCVs/5 s in fight (combat cmds) | per-game median | p90 | SCVs/5 s any cmd | fight bins (games) | bunker done ≤4:30 | → load (n, median s) "
             "| bunker damaged → repair (n / damaged, censored, median s) | cmd labels /1000f | agent decisions /1000f (whole game) | fight: cmds/5 s | fight: combat delay mean | ≤2 % | ≥10 % |")
    L.append("|---" * 26 + "|")
    for r in rows:
        a, f = r["sel"]["all"], r["sel"]["fight"]
        b, mi, fi = r["bunker"], r["micro_scv_per_5s"], r["fight"]
        L.append(
            f"| {r['group']} | {r['games']} | {r['held_8m']} ({r['held_8m_winpass']}) | {a['n']} ({f['n']}) | {fmt(a['eq12'])} | {fmt(a['elig_gt12'])} | "
            f"{fmt(a['sat12'])} ({fmt(f['sat12'])}) | {fmt(a['sat12_given_elig'])} | {a['games_with_sat12']} | {fmt(a['ge6'])} | {fmt(r['sel_size_mean'], 2)} | "
            f"{fmt(r['burst_le2'])} | {fmt(mi['combat_mean'], 2)} | {fmt(mi['per_game_median'], 2)} | {mi['combat_p90']} | {fmt(mi['any_mean'], 2)} | "
            f"{mi['bins']} ({mi['games']}) | {b['done_by_4:30']} | {b['load']}, {fmt(b['load_s_median'])} | "
            f"{b['repair']} / {b['damaged']}, {b['repair_censored']}, {fmt(b['repair_s_median'])} | {fmt(r['cmd_per_1000f'])} | "
            f"{fmt(r.get('agent_decisions_per_1000f_whole_game', float('nan')))} | {fmt(fi['cmds_per_5s'], 2)} | {fmt(fi['combat_delay_mean'])} | "
            f"{fmt(fi['combat_delay_le2'])} | {fmt(fi['combat_delay_ge10'])} |")
    return "\n".join(L)


def plan_games(dump, plan_path, opponents, job=None, stage=None):
    """[(Game, early, agent decisions per 1,000 frames or None)] of one dump of plan games."""
    plan = tomllib.loads(Path(plan_path).read_text(encoding="utf-8"))
    seeds, opps = ec.plan_seeds(plan), ec.game_opponents(plan)
    rows = ec.model_rows(ec.read_results(Path(job)), int(stage)) if job else {}
    out = []
    for d in load(dump):
        n = int(d["game"].lstrip("g"))
        if opponents and opps[n - 1] not in opponents:
            continue
        r = rows.get(n)
        rate = 1000 * r["model"]["decisions"] / max(1, r["end_frame"]) if r and r["model"].get("decisions") else None
        out.append((Game(d), ec.start_condition(seeds[n - 1]) == "early", rate))
    return out


def people_games(paths, every):
    """(rush games, every-th game of all by replay hash, total): streamed for memory; the sample is a
    stable function of the replay, not of the dump order."""
    sample, rush, total = [], [], 0
    for h in paths:
        for d in load(h):
            g = Game(d)
            total += 1
            if g.rush:
                rush.append(g)
            if int(d["replay_sha256"][:8], 16) % every == 0:
                sample.append(g)
    return rush, sample, total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--human", nargs="*", default=[], help="people shard dumps")
    ap.add_argument("--teacher", nargs=2, action="append", default=[], metavar=("DUMP", "PLAN"))
    ap.add_argument("--model", nargs=4, action="append", default=[], metavar=("DUMP", "PLAN", "JOB", "STAGE"),
                    help="model replay dump, its plan, the eval job folder (results/) and the plan's stage index")
    ap.add_argument("--opponents", default=DEFAULT_OPPONENTS, help="plan opponents kept, eval_common.opponent_label (empty = all)")
    ap.add_argument("--human-every", type=int, default=8, help="people all-games row: games with sha256 %% n == 0 (memory)")
    ap.add_argument("--out")
    ap.add_argument("--json")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    opps = {o for o in a.opponents.split(",") if o}
    rows = []
    if a.human:
        rush, sample, total = people_games(a.human, a.human_every)
        print(f"people: {total} games, {len(rush)} rush, {len(sample)} in the all-games sample", file=sys.stderr)
        rows.append(summarize(f"people S2 TvZ (sha256 % {a.human_every} sample of all)", sample))
        rows.append(summarize("people rush (contact ≤3:30)", rush))
        rows.append(summarize("people rush, held_8m_winpass", [g for g in rush if g.held_8m_winpass]))
        rows.append(summarize("people rush, not held_8m_winpass", [g for g in rush if not g.held_8m_winpass]))
    sources = [("teacher", [plan_games(d, p, opps) for d, p in a.teacher]),
               ("Checkpoint 1", [plan_games(d, p, opps, j, k) for d, p, j, k in a.model])]
    for name, parts in sources:
        allg = [x for part in parts for x in part]
        if not allg:
            continue
        rate = lambda xs: [x[2] for x in xs if x[2] is not None]  # noqa: E731
        early = [x for x in allg if x[1]]
        held = [x for x in early if x[0].held_8m_winpass]
        lost = [x for x in early if not x[0].held_8m_winpass]
        for label, xs in ((f"{name} early", early), (f"{name} early, held_8m_winpass", held),
                          (f"{name} early, not held_8m_winpass", lost), (f"{name} late", [x for x in allg if not x[1]])):
            rows.append(summarize(label, [x[0] for x in xs], rate(xs)))
        obs = collections.Counter((x[1], x[0].rush) for x in allg)
        print(f"{name}: (seed early, observed rush contact) -> games: {dict(obs)}", file=sys.stderr)
    md = tables_md(rows)
    print(md)
    if a.out:
        Path(a.out).write_text(md + "\n", encoding="utf-8")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

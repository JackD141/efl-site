"""League Rivals: rate the picks of the top managers in a Fantasy EFL league with our model.
python league_rivals.py      writes public/data/league_rivals.json (the League Rivals page)

For every listed manager and every finished gameweek, each pick's EXPECTED points next to what it scored:
- players: the model's expected points GIVEN the minutes he really played (backtest.py xp_mins: models refitted on games
  before the gameweek, closing odds for the fixtures; 60+ minutes = two appearance points and the clean sheet counts,
  under 60 = one, did not play = 0). So a manager is rated on who he picked, not on goals going in or not.
- captain: the captain's expected points again (the vice-captain's if the captain did not play). Max Captain chip: the
  expected top score of his seven, simulated from each player's expected points and the spread of real points around
  them (gamma with the variance seen in the backtest), because the top scorer is not known beforehand.
- clubs: expected club points from the closing odds (backtest.py club_x).
Inputs: data/2026_27/leagues/league_<id>.json (api/export-league.js: POST /api/export-league?league=<id>, then git
pull) and output/backtest_players.csv + backtest_clubs.csv (backtest.py, run before this).
Also the overall leaderboard (the EFL shows its top 400): data/2026_27/leagues/overall_top.json, built by
`python league_rivals.py --fetch-overall` (ten calls of POST /api/export-league?overall=<from>; no names are kept). It
is scored the same way and summarised on the page: where the account's own team stands on expected points.
Check printed each run: our rebuilt actual total per manager against the EFL's own total. They are equal unless he
picked a player whose games are not in our stats, so those games (and their expected points) are missing; the page
marks it. (Until 14 Nov 2026 the stats export dropped a player's games for a previous club; it now keeps them.)
"""
import json
import sys
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import attack_model as am

HERE = am.HERE
SEASON = "2026_27"
LEAGUES = {40: "Lincoln City", 7: "Derby County"}  # league ids on fantasy.efl.com (/league/<id>/ladder)
LEAGUE_DIR = am.REPO / "data" / SEASON / "leagues"
OVERALL = LEAGUE_DIR / "overall_top.json"
OVERALL_STEP, OVERALL_MAX = 40, 400  # as in api/export-league.js; the EFL's overall leaderboard stops at 400
SITE = "https://dexters-corner.vercel.app"
OUT = am.REPO / "public" / "data" / "league_rivals.json"
POS = ["GK", "DEF", "MID", "FWD"]
N_SIM = 4000


def fetch_overall():
    """Download the overall leaderboard's managers and teams in slices and save them as one file (commit it)."""
    rankings, teams, meta = [], {}, {}
    for start in range(0, OVERALL_MAX, OVERALL_STEP):
        req = urllib.request.Request(f"{SITE}/api/export-league?overall={start}", method="POST")
        d = json.load(urllib.request.urlopen(req, timeout=90))
        if d.get("failed"):
            raise SystemExit(f"slice {start}: {d['failed']} teams could not be fetched; run again")
        rankings += d["rankings"]
        teams.update(d["teams"])
        meta = {k: d[k] for k in ("fetchedAt", "completedRounds", "liveRounds")}
        print(f"  {start + len(d['rankings'])} managers", flush=True)
    if len({r["userId"] for r in rankings}) != len(rankings):
        raise SystemExit("the slices overlap: the leaderboard moved while it was being read; run again")
    OVERALL.write_text(json.dumps(dict(meta, rankings=rankings, teams=teams), separators=(",", ":")), encoding="utf-8")
    print("wrote", OVERALL)


def spread(bp):
    """Variance of real points around xp_mins for players who played: Var = a + b * xp (least squares on bands)."""
    d = bp[(bp["mins"] > 0) & (bp["xp_mins"] > 0)]
    band = pd.qcut(d["xp_mins"], 12, duplicates="drop")
    g = d.groupby(band, observed=True).apply(lambda x: pd.Series(dict(m=x["xp_mins"].mean(), v=((x["pts"] - x["xp_mins"]) ** 2).mean())))
    b, a = np.polyfit(g["m"], g["v"], 1)
    return float(max(a, 0.5)), float(max(b, 0.0))


def expected_max(evs, var, rng):
    """E[max] of independent gamma points with means evs and variance var[0] + var[1] * mean (players who played)."""
    m = np.array([e for e in evs if e > 0], float)
    if not len(m):
        return 0.0
    m = np.maximum(m, 0.3)
    v = var[0] + var[1] * m
    return float(rng.gamma(m ** 2 / v, v / m, size=(N_SIM, len(m))).max(axis=1).mean())


class Scorer:
    def __init__(self):
        bp = pd.read_csv(HERE / "output" / "backtest_players.csv")
        bc = pd.read_csv(HERE / "output" / "backtest_clubs.csv")
        self.gws = sorted(int(g) for g in bp["gw"].unique())
        self.P = {(int(r.gw), int(r.id)): (float(r.xp_mins), float(r.pts), float(r.mins)) for r in bp.itertuples()}
        self.C = {(int(r.gw), int(r.club)): (float(r.xp), float(r.pts)) for r in bc.itertuples()}
        self.var = spread(bp)
        self.rng = np.random.default_rng(0)
        self.mc_check = []

    def week(self, gw, ids, clubs, cap, vice, mc, oc=False, check=True):
        """One team in one gameweek -> expected and real points, split players / captain / clubs."""
        got = {i: self.P.get((gw, i), (0.0, 0.0, 0.0)) for i in ids}
        ev, pts, mins = ({i: v[k] for i, v in got.items()} for k in range(3))
        top_ev = expected_max(list(ev.values()), self.var, self.rng) if mc or check else 0.0
        if check:
            self.mc_check.append((top_ev, max(pts.values(), default=0.0)))
        if mc:
            dbl, cap_ev, cap_act = "max", top_ev, max(pts.values(), default=0.0)
        else:
            dbl = cap if mins.get(cap, 0) > 0 else (vice if mins.get(vice, 0) > 0 else None)
            cap_ev, cap_act = (ev[dbl], pts[dbl]) if dbl else (0.0, 0.0)
        club_ev = sum(self.C.get((gw, c), (0.0, 0.0))[0] for c in clubs)
        club_act = sum(self.C.get((gw, c), (0.0, 0.0))[1] for c in clubs)
        w = dict(gw=gw, p=ids, s=clubs, c=cap, v=vice, dbl=dbl, mc=bool(mc), oc=bool(oc),
                 evP=round(sum(ev.values()), 2), evC=round(cap_ev, 2), evS=round(club_ev, 2), actP=sum(pts.values()), actC=cap_act, actS=club_act)
        w.update(ev=round(w["evP"] + w["evC"] + w["evS"], 2), act=w["actP"] + cap_act + club_act)
        return w

    def efl_done(self, d, r):
        """The EFL's total over the gameweeks we scored (one gameweek ahead can be taken off with roundPoints)."""
        ahead = sorted((set(d.get("liveRounds", [])) | set(d.get("completedRounds", []))) - set(self.gws))
        return None if len(ahead) > 1 else r["overallPoints"] - ((r.get("roundPoints") or 0) if ahead else 0)


def overall_summary(sc, you):
    """The overall leaderboard's managers scored like the leagues; no names. Returns the page's summary (or None)."""
    if not OVERALL.exists():
        return None
    d = json.load(open(OVERALL, encoding="utf-8"))
    rows = []
    for r in d["rankings"]:
        tot = dict(ev=0.0, act=0.0, evP=0.0, evC=0.0, evS=0.0)
        for gw in sc.gws:
            t = d["teams"].get(str(r["userId"]), {}).get(str(gw))
            if t:
                w = sc.week(gw, [int(i) for i in t["p"]], [int(c) for c in t["s"]], t.get("c"), t.get("v"), t.get("mc"), check=False)
                for k in tot:
                    tot[k] += w[k]
        rows.append(dict(userId=r["userId"], rank=r["rank"], efl=sc.efl_done(d, r), **{k: round(v, 1) for k, v in tot.items()}))
    n = len(rows)
    ok = sum(1 for x in rows if x["efl"] is not None and abs(x["act"] - x["efl"]) <= 0.5)
    by_ev = sorted(rows, key=lambda x: -x["ev"])
    for i, x in enumerate(by_ev, 1):
        x["evRank"] = i
    ev = np.array([x["ev"] for x in rows])
    luck = np.array([x["act"] - x["ev"] for x in rows])
    me = next((x for x in rows if x["userId"] == you), None)
    place = lambda k: 1 + sum(x[k] > me[k] for x in rows)
    print(f"Overall leaderboard: {n} managers (ranks {rows[0]['rank']}-{rows[-1]['rank']}); rebuilt actual total equals the EFL's for {ok} of {n}")
    print(f"   expected points: best {ev.max():.0f}, median {np.median(ev):.0f}, lowest {ev.min():.0f}; luck mean {luck.mean():+.0f}, sd {luck.std():.0f}; "
          f"correlation of expected with actual {np.corrcoef(ev, [x['act'] for x in rows])[0, 1]:.2f}")
    if me:
        print(f"   the account's team: overall rank {me['rank']}, expected {me['ev']:.0f} = {me['evRank']} of {n} on expected points "
              f"(players {place('evP')}, captain {place('evC')}, clubs {place('evS')})")
    lo, hi = int(np.floor(ev.min() / 20) * 20), int(np.ceil((ev.max() + 0.01) / 20) * 20)
    hist = [dict(lo=b, n=int(((ev >= b) & (ev < b + 20)).sum())) for b in range(lo, hi, 20)]
    keep = lambda x: {k: x[k] for k in ("rank", "ev", "act", "evP", "evC", "evS", "evRank")}
    part = lambda k: dict(rank=place(k), value=me[k], median=round(float(np.median([x[k] for x in rows])), 1))
    return dict(n=n, fetchedAt=d["fetchedAt"], lowestRank=rows[-1]["rank"], matched=ok, median=round(float(np.median(ev)), 1), best=round(float(ev.max()), 1),
                meanLuck=round(float(luck.mean()), 1), sdLuck=round(float(luck.std()), 1), hist=hist, top=[keep(x) for x in by_ev[:10]],
                you=dict(keep(me), players=part("evP"), captain=part("evC"), clubs=part("evS")) if me else None)


def main():
    sc = Scorer()
    gws = sc.gws
    names = {p["id"]: p for p in json.load(open(HERE / "cache" / "players.json", encoding="utf-8"))}
    squads = {s["id"]: s for s in json.load(open(HERE / "cache" / "squads.json", encoding="utf-8"))}
    used_p, used_c, leagues, you = set(), set(), [], None
    for lid, lname in LEAGUES.items():
        f = LEAGUE_DIR / f"league_{lid}.json"
        if not f.exists():
            print(f"league {lid} ({lname}): no export yet ({f.name}); left out")
            continue
        d = json.load(open(f, encoding="utf-8"))
        you = (d.get("ladder", {}).get("user") or {}).get("userId") or you
        managers = []
        for r in d["rankings"]:
            weeks = []
            for gw in gws:
                t = (d["teams"].get(str(r["userId"]), {}).get(str(gw)) or {}).get("team")
                if not t:
                    weeks.append(dict(gw=gw, p=[], s=[], ev=0.0, act=0.0, none=True))
                    continue
                ids = [int(i) for pos in POS for i in t["lineup"]["players"].get(pos, []) if i]
                clubs = [int(c) for c in t["lineup"].get("squads", []) if c]
                weeks.append(sc.week(gw, ids, clubs, t.get("captainId"), t.get("viceCaptainId"), t.get("maxCaptainEnabled"), t.get("oneClubEnabled")))
                used_p.update((gw, i) for i in ids)
                used_c.update((gw, c) for c in clubs)
            tot = lambda k: round(sum(w.get(k, 0.0) for w in weeks), 1)
            managers.append(dict(userId=r["userId"], team=r["teamName"], name=r["displayName"], rank=r["rank"], eflPoints=r["overallPoints"],
                                 act=tot("act"), ev=tot("ev"), evP=tot("evP"), evC=tot("evC"), evS=tot("evS"),
                                 actP=tot("actP"), actC=tot("actC"), actS=tot("actS"), eflDone=sc.efl_done(d, r), weeks=weeks))
        off = [(m["name"], m["act"], m["eflDone"]) for m in managers if m["eflDone"] is not None and abs(m["act"] - m["eflDone"]) > 0.5]
        print(f"{lname}: {len(managers)} managers, GW{gws[0]}-{gws[-1]}; rebuilt actual total equals the EFL's for {len(managers) - len(off)} of {len(managers)}"
              + (f"; differ: {off} (a pick whose games are not in our stats)" if off else "")
              + ("" if any(m["eflDone"] is not None for m in managers) else "; the EFL totals are more than a gameweek ahead of the stats, not compared"))
        for m in sorted(managers, key=lambda m: -m["ev"])[:5]:
            print(f"   {m['name']:18s} expected {m['ev']:7.1f}  actual {m['act']:6.0f}  ({m['act'] - m['ev']:+.0f})  players {m['evP']:.0f} captain {m['evC']:.0f} clubs {m['evS']:.0f}")
        leagues.append(dict(id=lid, name=lname, fetchedAt=d["fetchedAt"], you=(d.get("ladder", {}).get("user") or {}).get("userId"),
                            entrants=(d.get("league") or {}).get("league", {}).get("numJoins"), managers=managers))
    mc = np.array(sc.mc_check)
    print(f"Max Captain check over {len(mc)} team-gameweeks: simulated expected top score {mc[:, 0].mean():.2f}, real top score {mc[:, 1].mean():.2f}")
    overall = overall_summary(sc, you)
    players = {i: [names[i]["displayName"], names[i]["position"], names[i]["squadId"]] if i in names else [str(i), "", None] for _, i in used_p}
    pg, cg = {}, {}
    for gw, i in used_p:
        pg.setdefault(gw, {})[i] = [round(v, 2) for v in sc.P.get((gw, i), (0.0, 0.0, 0.0))]
    for gw, c in used_c:
        cg.setdefault(gw, {})[c] = [round(v, 2) for v in sc.C.get((gw, c), (0.0, 0.0))]
    out = dict(generatedAt=datetime.now(timezone.utc).isoformat(timespec="seconds"), season="2026/27", gws=gws, leagues=leagues, overall=overall, players=players,
               clubs={c: squads[c]["shortName"] for _, c in used_c if c in squads}, pg=pg, cg=cg,
               maxCaptain=dict(n=len(mc), sim=round(float(mc[:, 0].mean()), 2), real=round(float(mc[:, 1].mean()), 2), var=[round(v, 3) for v in sc.var]))
    OUT.write_text(json.dumps(out, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print("wrote", OUT, f"({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    if "--fetch-overall" in sys.argv:
        fetch_overall()
    main()

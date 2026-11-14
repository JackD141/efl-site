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
Check printed each run: our rebuilt actual total per manager against the EFL's own total. They are equal unless he
picked a player who has since moved club or left the game: the stats export (api/export-player-stats.js) only keeps a
player's games for his current club, so those games (and their expected points) are missing; the page marks it.
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import attack_model as am

HERE = am.HERE
SEASON = "2026_27"
LEAGUES = {40: "Lincoln City", 7: "Derby County"}  # league ids on fantasy.efl.com (/league/<id>/ladder)
OUT = am.REPO / "public" / "data" / "league_rivals.json"
POS = ["GK", "DEF", "MID", "FWD"]
N_SIM = 4000


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


def main():
    bp = pd.read_csv(HERE / "output" / "backtest_players.csv")
    bc = pd.read_csv(HERE / "output" / "backtest_clubs.csv")
    gws = sorted(int(g) for g in bp["gw"].unique())
    P = {(int(r.gw), int(r.id)): (float(r.xp_mins), float(r.pts), float(r.mins)) for r in bp.itertuples()}
    C = {(int(r.gw), int(r.club)): (float(r.xp), float(r.pts)) for r in bc.itertuples()}
    names = {p["id"]: p for p in json.load(open(HERE / "cache" / "players.json", encoding="utf-8"))}
    squads = {s["id"]: s for s in json.load(open(HERE / "cache" / "squads.json", encoding="utf-8"))}
    var = spread(bp)
    rng = np.random.default_rng(0)
    used_p, used_c, leagues, mc_check = set(), set(), [], []
    for lid, lname in LEAGUES.items():
        f = am.REPO / "data" / SEASON / "leagues" / f"league_{lid}.json"
        if not f.exists():
            print(f"league {lid} ({lname}): no export yet ({f.name}); left out")
            continue
        d = json.load(open(f, encoding="utf-8"))
        # gameweeks in the EFL's totals that we have not scored: one (the live one) can be taken off with roundPoints
        ahead = sorted((set(d.get("liveRounds", [])) | set(d.get("completedRounds", []))) - set(gws))
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
                ev = {i: P.get((gw, i), (0.0, 0.0, 0.0))[0] for i in ids}
                pts = {i: P.get((gw, i), (0.0, 0.0, 0.0))[1] for i in ids}
                mins = {i: P.get((gw, i), (0.0, 0.0, 0.0))[2] for i in ids}
                cap, vice = t.get("captainId"), t.get("viceCaptainId")
                top_ev = expected_max(list(ev.values()), var, rng)
                mc_check.append((top_ev, max(pts.values(), default=0.0)))
                if t.get("maxCaptainEnabled"):
                    dbl, cap_ev, cap_act = "max", top_ev, max(pts.values(), default=0.0)
                else:
                    dbl = cap if mins.get(cap, 0) > 0 else (vice if mins.get(vice, 0) > 0 else None)
                    cap_ev, cap_act = (ev[dbl], pts[dbl]) if dbl else (0.0, 0.0)
                club_ev = sum(C.get((gw, c), (0.0, 0.0))[0] for c in clubs)
                club_act = sum(C.get((gw, c), (0.0, 0.0))[1] for c in clubs)
                used_p.update((gw, i) for i in ids)
                used_c.update((gw, c) for c in clubs)
                weeks.append(dict(gw=gw, p=ids, s=clubs, c=cap, v=vice, dbl=dbl, mc=bool(t.get("maxCaptainEnabled")), oc=bool(t.get("oneClubEnabled")),
                                  evP=round(sum(ev.values()), 2), evC=round(cap_ev, 2), evS=round(club_ev, 2),
                                  actP=sum(pts.values()), actC=cap_act, actS=club_act))
                weeks[-1].update(ev=round(weeks[-1]["evP"] + weeks[-1]["evC"] + weeks[-1]["evS"], 2), act=weeks[-1]["actP"] + cap_act + club_act)
            tot = lambda k: round(sum(w.get(k, 0.0) for w in weeks), 1)
            efl = None if len(ahead) > 1 else r["overallPoints"] - ((r.get("roundPoints") or 0) if ahead else 0)
            managers.append(dict(userId=r["userId"], team=r["teamName"], name=r["displayName"], rank=r["rank"], eflPoints=r["overallPoints"],
                                 act=tot("act"), ev=tot("ev"), evP=tot("evP"), evC=tot("evC"), evS=tot("evS"),
                                 actP=tot("actP"), actC=tot("actC"), actS=tot("actS"), eflDone=efl, weeks=weeks))
        off = [(m["name"], m["act"], m["eflDone"]) for m in managers if m["eflDone"] is not None and abs(m["act"] - m["eflDone"]) > 0.5]
        print(f"{lname}: {len(managers)} managers, GW{gws[0]}-{gws[-1]}; rebuilt actual total equals the EFL's for {len(managers) - len(off)} of {len(managers)}"
              + (f"; differ: {off} (a pick whose games are not in our stats: he has since moved club)" if off else "")
              + ("" if len(ahead) <= 1 else f"; the EFL totals are {len(ahead)} gameweeks ahead of the stats, not compared"))
        for m in sorted(managers, key=lambda m: -m["ev"])[:5]:
            print(f"   {m['name']:18s} expected {m['ev']:7.1f}  actual {m['act']:6.0f}  ({m['act'] - m['ev']:+.0f})  players {m['evP']:.0f} captain {m['evC']:.0f} clubs {m['evS']:.0f}")
        leagues.append(dict(id=lid, name=lname, fetchedAt=d["fetchedAt"], you=(d.get("ladder", {}).get("user") or {}).get("userId"),
                            entrants=(d.get("league") or {}).get("league", {}).get("numJoins"), managers=managers))
    mc = np.array(mc_check)
    print(f"Max Captain check over {len(mc)} team-gameweeks: simulated expected top score {mc[:, 0].mean():.2f}, real top score {mc[:, 1].mean():.2f}")
    players = {i: [names[i]["displayName"], names[i]["position"], names[i]["squadId"]] if i in names else [str(i), "", None] for _, i in used_p}
    pg, cg = {}, {}
    for gw, i in used_p:
        pg.setdefault(gw, {})[i] = [round(v, 2) for v in P.get((gw, i), (0.0, 0.0, 0.0))]
    for gw, c in used_c:
        cg.setdefault(gw, {})[c] = [round(v, 2) for v in C.get((gw, c), (0.0, 0.0))]
    out = dict(generatedAt=datetime.now(timezone.utc).isoformat(timespec="seconds"), season="2026/27", gws=gws, leagues=leagues, players=players,
               clubs={c: squads[c]["shortName"] for _, c in used_c if c in squads}, pg=pg, cg=cg,
               maxCaptain=dict(n=len(mc), sim=round(float(mc[:, 0].mean()), 2), real=round(float(mc[:, 1].mean()), 2), var=[round(v, 3) for v in var]))
    OUT.write_text(json.dumps(out, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print("wrote", OUT, f"({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

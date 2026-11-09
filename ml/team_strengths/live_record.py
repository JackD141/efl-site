"""The live record: what the site actually published before each gameweek, scored against what happened.
python live_record.py            prints the record so far (also run by weekly_update.py)

Why, when there is a backtest: the backtest rebuilds past predictions with today's code, so every model change rewrites
it, and it uses closing odds and no team news. The log is the opposite: written once, before the gameweek kicks off,
with the odds and injury flags we really had, and never changed. Model choices cannot be tuned to it after the fact.
- save(gw): data/2026_27/predictions/gw<N>.json from the plan JSONs (the same Python as check_site_maths.py, which the
  pages are checked against): per player position, expected points, expected points if he starts, start chance; per
  club its expected points. Only written while the gameweek's first game has not kicked off (a later re-run, e.g. for
  Tuesday's games of a double, must not overwrite the pre-deadline view).
- record(): for every logged gameweek whose stats are in, error of the expected points over all logged players (and
  of "expected points if he starts" over those who started), calibration of the ten highest per position, clubs.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import check_site_maths as csm

HERE = Path(__file__).parent
REPO = HERE.parents[1]
LOG = REPO / "data" / "2026_27" / "predictions"
PLANS = {"GK": ("keeper_plan.json", "keepers", csm.keepers), "DEF": ("defender_plan.json", "defenders", csm.defenders),
         "MID": ("mid_plan.json", "players", csm.attackers), "FWD": ("fwd_plan.json", "players", csm.attackers)}


def save(gw=None, now=None):
    """Write the gameweek's published predictions to the log. Returns the path, or None if the gameweek has started."""
    load = lambda f: json.loads((csm.DATA / f).read_text(encoding="utf-8"))
    club = load("club_plan.json")
    gw = gw or club["firstGw"]
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    kos = [pd.Timestamp(f["ko"]) for c in club["clubs"] for w in c["weeks"] if w["gw"] == gw for f in w["fx"]]
    if not kos or now >= min(kos):
        return None
    players = {}
    for pos, (fname, key, fn) in PLANS.items():
        p = load(fname)
        if p.get("firstGw") != gw:
            continue
        xp, xps = fn(p), fn(p, True)
        for d in p[key]:
            if d["id"] in xp:
                players[str(d["id"])] = [pos, d["club"], round(xp[d["id"]], 3), round(xps.get(d["id"], 0.0), 3), d.get("pStart", 0.0)]
    clubs = {str(c["id"]): w["xp"] for c in club["clubs"] for w in c["weeks"] if w["gw"] == gw and w["games"]}
    LOG.mkdir(parents=True, exist_ok=True)
    out = LOG / f"gw{gw}.json"
    out.write_text(json.dumps(dict(gw=gw, savedAt=now.isoformat(timespec="seconds"), firstKickOff=min(kos).isoformat(),
                                   fields=["pos", "club", "xp", "xpIfStarts", "pStart"], players=players, clubs=clubs), separators=(",", ":")), encoding="utf-8")
    return out


def record(verbose=True):
    """One row per logged gameweek with stats: how the published expected points did."""
    rows = []
    for f in sorted(LOG.glob("gw*.json"), key=lambda f: int(f.stem[2:])):
        log = json.loads(f.read_text(encoding="utf-8"))
        stats = REPO / "data" / "2026_27" / f"player_stats_gw{log['gw']}.csv"
        if not stats.exists() or stats.stat().st_size < 1000:
            continue
        s = pd.read_csv(stats)
        got = s.groupby("player_id").agg(pts=("points", "sum"), mins=("minutes_played", "sum"), first=("minutes_played", "first"))
        d = pd.DataFrame([dict(id=int(i), pos=v[0], xp=v[2], xps=v[3], p=v[4]) for i, v in log["players"].items()]).set_index("id")
        d = d.join(got, how="left").fillna(dict(pts=0.0, mins=0.0, first=0.0))
        top = d.sort_values("xp", ascending=False).groupby("pos").head(10)
        st = d[d["first"] >= 60]  # played 60+ in his club's first game: "if he starts" is the right number for him
        rows.append(dict(gw=log["gw"], savedAt=log["savedAt"][:16], players=len(d), mean_xp=d["xp"].mean(), mean_pts=d["pts"].mean(),
                         mse=((d["pts"] - d["xp"]) ** 2).mean(), rank_corr=d["xp"].corr(d["pts"], method="spearman"),
                         top10_xp=top["xp"].mean(), top10_pts=top["pts"].mean(), starters=len(st),
                         starters_xp=st["xps"].mean() if len(st) else np.nan, starters_pts=st["pts"].mean() if len(st) else np.nan))
    r = pd.DataFrame(rows)
    if verbose:
        print(f"\nLIVE RECORD (published before each gameweek, never rewritten): {len(r)} gameweek(s) scored")
        if len(r):
            print(r.round(3).to_string(index=False))
            print(f"  all: mean xP {np.average(r['mean_xp'], weights=r['players']):.3f} vs actual {np.average(r['mean_pts'], weights=r['players']):.3f}; "
                  f"ten highest xP per position: {r['top10_xp'].mean():.2f} expected vs {r['top10_pts'].mean():.2f} actual; "
                  f"started and played 60+: {np.nanmean(r['starters_xp']):.2f} expected (if he starts) vs {np.nanmean(r['starters_pts']):.2f} actual")
    return r


if __name__ == "__main__":
    record()

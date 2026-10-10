"""How do player and club numbers change when the league changes? Estimated from FotMob season pairs
(2024/25 -> 2025/26 and 2025/26 -> 2026/27 so far).
python league_steps.py      prints the tables and writes ml/team_strengths/models/league_steps.json

Two separate effects:
- ROLE (a player moving to a club in another league): his rate relative to same-position team-mates, next season vs
  this season. Compared with players who changed club within the same league (same selection effects, e.g. good
  seasons earning moves), so  m_role[step] = mean ratio for that step / mean ratio for same-league movers.
- TEAM STYLE (a club promoted or relegated): the club's same-position rate per 90, next season vs this season,
  compared with clubs staying in their league:  m_team[step] = ratio for promoted (or relegated) / ratio for stayers.
step = +1 up a league (e.g. League One -> Championship), -1 down. Ratios are minutes-weighted.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
FM = REPO / "data" / "fotmob"
LEVEL = {"Championship": 1, "League One": 2, "League Two": 3}
POS = {1.0: "DEF", 2.0: "MID", 3.0: "FWD"}
STATS = {"int": "Interceptions", "kp": "Chances created", "sot": "Shots on target", "goals": "Goals", "assists": "Assists",
         "npxg": "xG Non-penalty", "xa": "Expected assists (xA)", "shots": "Total shots", "clr": "Clearances", "blk": "Blocks",
         "tkl": "Tackles"}
MIN_MINS = 900   # minutes in each season for a player pair
MIN_MINS_CUR = 450  # the current, unfinished season


def load():
    frames = []
    for f in sorted(FM.glob("*/player_matches.csv")):
        p = pd.read_csv(f, low_memory=False)
        cols = ["season", "league", "fm_player_id", "team", "usual_position", "Minutes played"] + [c for c in STATS.values() if c in p]
        frames.append(p[cols])
    d = pd.concat(frames, ignore_index=True).rename(columns={"Minutes played": "mins"} | {v: k for k, v in STATS.items()})
    d["mins"] = pd.to_numeric(d["mins"], errors="coerce").fillna(0)
    d = d[(d["mins"] > 0) & d["usual_position"].isin(POS)].copy()
    d["pos"] = d["usual_position"].map(POS)
    for k in STATS:
        d[k] = pd.to_numeric(d.get(k), errors="coerce").fillna(0)
    return d


def season_tables(d):
    # club-position totals per season, and each player's main club (most minutes) per season
    team = d.groupby(["season", "team", "league", "pos"])[["mins"] + list(STATS)].sum().reset_index()
    pl = d.groupby(["season", "fm_player_id", "team", "league", "pos"])[["mins"] + list(STATS)].sum().reset_index()
    pl = pl.sort_values("mins").drop_duplicates(["season", "fm_player_id"], keep="last")
    pl = pl.merge(team, on=["season", "team", "league", "pos"], suffixes=("", "_team"))
    for k in STATS:
        mates90 = (pl[f"{k}_team"] - pl[k]) / ((pl["mins_team"] - pl["mins"]).clip(lower=90) / 90)
        pl[f"role_{k}"] = (pl[k] / (pl["mins"] / 90)) / mates90.replace(0, np.nan)
    return team, pl


def main():
    d = load()
    team, pl = season_tables(d)
    seasons = sorted(d["season"].unique())
    pairs = list(zip(seasons[:-1], seasons[1:]))
    out = {"role": {}, "team": {}, "pairs": [f"{a} -> {b}" for a, b in pairs]}
    for pos in ("MID", "FWD", "DEF"):
        # players
        rows = []
        for a, b in pairs:
            mb = MIN_MINS_CUR if b == seasons[-1] else MIN_MINS
            x = pl[(pl["season"] == a) & (pl["pos"] == pos) & (pl["mins"] >= MIN_MINS)].merge(
                pl[(pl["season"] == b) & (pl["mins"] >= mb)], on="fm_player_id", suffixes=("_1", "_2"))
            x["step"] = x["league_1"].map(LEVEL) - x["league_2"].map(LEVEL)
            x["moved"] = x["team_1"] != x["team_2"]
            rows.append(x)
        x = pd.concat(rows, ignore_index=True)
        x["w"] = np.minimum(x["mins_1"], x["mins_2"])
        res = {}
        for k in STATS:
            r1, r2 = x[f"role_{k}_1"], x[f"role_{k}_2"]
            ok = r1.notna() & r2.notna() & np.isfinite(r1) & np.isfinite(r2) & (r1 > 0)
            g = x[ok].assign(r1=r1[ok], r2=r2[ok])
            def ratio(m):
                s = g[m]
                return (s["r2"] * s["w"]).sum() / (s["r1"] * s["w"]).sum(), int(m.sum())
            base, nb = ratio(g["moved"] & (g["step"] == 0))
            res[k] = {}
            for step in (1, -1):
                r, n = ratio(g["moved"] & (g["step"] == step))
                res[k][step] = dict(mult=round(r / base, 3), n=n, n_base=nb)
        out["role"][pos] = res
        print(f"\n{pos} players changing club: role multiplier vs same-league movers (n up / n down / n base)")
        print("  " + " | ".join(f"{k}: up {v[1]['mult']:.2f} ({v[1]['n']}) down {v[-1]['mult']:.2f} ({v[-1]['n']}) [base {v[1]['n_base']}]" for k, v in res.items()))
        # clubs
        rows = []
        for a, b in pairs:
            t1, t2 = team[(team["season"] == a) & (team["pos"] == pos)], team[(team["season"] == b) & (team["pos"] == pos)]
            y = t1.merge(t2, on="team", suffixes=("_1", "_2"))
            y["step"] = y["league_1"].map(LEVEL) - y["league_2"].map(LEVEL)
            rows.append(y)
        y = pd.concat(rows, ignore_index=True)
        tres = {}
        for k in STATS:
            y["r1"], y["r2"] = y[f"{k}_1"] / (y["mins_1"] / 90), y[f"{k}_2"] / (y["mins_2"] / 90)
            def tratio(m):
                s = y[m]
                return (s["r2"] * s["mins_2"]).sum() / (s["r1"] * s["mins_2"]).sum(), int(m.sum())
            base, nb = tratio(y["step"] == 0)
            tres[k] = {}
            for step in (1, -1):
                r, n = tratio(y["step"] == step)
                tres[k][step] = dict(mult=round(r / base, 3), n=n, n_base=nb)
        out["team"][pos] = tres
        print(f"{pos} clubs changing league: style multiplier vs clubs staying (n clubs)")
        print("  " + " | ".join(f"{k}: up {v[1]['mult']:.2f} ({v[1]['n']}) down {v[-1]['mult']:.2f} ({v[-1]['n']})" for k, v in tres.items()))
    # year-to-year persistence of club style (same-league clubs, full seasons): slope of log rate on last season's,
    # within league; used to carry a club's style into the next season
    out["persistence"] = {}
    full = [(a, b) for a, b in pairs if b != seasons[-1]] or pairs
    for pos in ("MID", "FWD", "DEF"):
        ys = []
        for a, b in full:
            t1, t2 = team[(team["season"] == a) & (team["pos"] == pos)], team[(team["season"] == b) & (team["pos"] == pos)]
            y = t1.merge(t2, on="team", suffixes=("_1", "_2"))
            ys.append(y[y["league_1"] == y["league_2"]])
        y = pd.concat(ys)
        out["persistence"][pos] = {}
        for k in STATS:
            r1, r2 = np.log((y[f"{k}_1"] + 0.5) / (y["mins_1"] / 90)), np.log((y[f"{k}_2"] + 0.5) / (y["mins_2"] / 90))
            r1, r2 = r1 - r1.groupby(y["league_1"]).transform("mean"), r2 - r2.groupby(y["league_2"]).transform("mean")
            out["persistence"][pos][k] = round(float(np.clip(np.polyfit(r1, r2, 1)[0], 0, 1)), 3)
        print(f"{pos} club style persistence:", out["persistence"][pos])
    path = REPO / "ml" / "team_strengths" / "models" / "league_steps.json"
    path.write_text(json.dumps(out, indent=1, default=lambda o: int(o) if isinstance(o, np.integer) else str(o)), encoding="utf-8")
    print("\nwrote", path)


if __name__ == "__main__":
    main()

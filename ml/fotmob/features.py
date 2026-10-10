"""Player features from FotMob history for the stat models.

signal_rates(rows) adds, for each EFL row (needs player_id and date), the player's per-90 rates over his last N FotMob
appearances strictly BEFORE that date (any club, any of the three leagues, 2024/25 onwards), shrunk towards the league
rate with K pseudo-90s:  npxg (non-penalty xG), pxg (penalty xG: identifies penalty takers), xa, shots, chances.
Rows: FotMob player_matches.csv for every season pulled, linked to EFL ids with player_map.csv (link_fotmob.py).
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
FM = REPO / "data" / "fotmob"
N_GAMES = 20
SIGNALS = {"npxg": "xG Non-penalty", "xg": "Expected goals (xG)", "xa": "Expected assists (xA)", "shots": "Total shots",
           "chances": "Chances created", "sot": "Shots on target", "int": "Interceptions", "xgot": "Expected goals on target (xGOT)"}
RATES = ("npxg", "pxg", "xa", "shots", "chances", "sot", "int", "xgot")


def fotmob_history():
    mp = pd.read_csv(FM / "player_map.csv")[["fm_player_id", "player_id"]]
    frames = []
    for f in sorted(FM.glob("*/player_matches.csv")):
        pm = pd.read_csv(f, low_memory=False)
        keep = ["fm_player_id", "utc", "league", "Minutes played"] + [c for c in SIGNALS.values() if c in pm]
        frames.append(pm[keep])
    h = pd.concat(frames, ignore_index=True).merge(mp, on="fm_player_id")
    h = h.rename(columns={v: k for k, v in SIGNALS.items()} | {"Minutes played": "mins"})
    h["mins"] = pd.to_numeric(h["mins"], errors="coerce").fillna(0)
    h = h[h["mins"] > 0].copy()
    for k in SIGNALS:
        h[k] = pd.to_numeric(h.get(k), errors="coerce").fillna(0)  # FotMob leaves xG / shots out when there were none
    h["pxg"] = (h["xg"] - h["npxg"]).clip(lower=0)
    h["date"] = pd.to_datetime(h["utc"], utc=True).dt.tz_localize(None).dt.normalize()
    h["level"] = h["league"].map({"Championship": 1, "League One": 2, "League Two": 3}).fillna(2).astype(int)
    return h.sort_values(["player_id", "date"]).reset_index(drop=True)


def league_priors(h):
    m90 = h["mins"].sum() / 90
    return {k: float(h[k].sum() / m90) for k in RATES}


def signal_rates(rows, k=5.0, n=N_GAMES, hist=None, priors=None, prefix="fm_", step_mult=None):
    """rows: DataFrame with player_id and date (Timestamp, day). Returns rows + <prefix><signal> per-90 columns + <prefix>n90.
    step_mult {signal: {1: up, -1: down}}: games played in a different league from the row's own league ('level' column
    of rows: 1 Championship .. 3 League Two) count x up**steps (old game lower) or x down**steps (old game higher)."""
    h = fotmob_history() if hist is None else hist
    priors = priors or league_priors(h)
    g = h.groupby("player_id")
    roll = pd.DataFrame({"player_id": h["player_id"], "date": h["date"],
                         "m90": g["mins"].transform(lambda s: s.rolling(n, min_periods=1).sum()) / 90})
    for s in priors:
        if step_mult is None or s not in step_mult:
            roll[s] = g[s].transform(lambda x: x.rolling(n, min_periods=1).sum())
            continue
        for lvl in (1, 2, 3):  # a version of the rolling sum for each league the row's player is now in
            step = h["level"] - lvl
            w = np.where(step > 0, step_mult[s][1] ** step.clip(lower=0), np.where(step < 0, step_mult[s][-1] ** (-step).clip(lower=0), 1.0))
            roll[f"{s}@{lvl}"] = (h[s] * w).groupby(h["player_id"]).transform(lambda x: x.rolling(n, min_periods=1).sum())
    # last row per player per date (doubles on one day are rare), then as-of join strictly before the row's date
    roll = roll.drop_duplicates(["player_id", "date"], keep="last").sort_values("date")
    r = rows.copy()
    r["_ord"] = np.arange(len(r))
    r["date"] = pd.to_datetime(r["date"]).dt.tz_localize(None).dt.normalize() if getattr(pd.to_datetime(r["date"]).dt, "tz", None) else pd.to_datetime(r["date"]).dt.normalize()
    r = pd.merge_asof(r.sort_values("date"), roll, on="date", by="player_id", allow_exact_matches=False, direction="backward")
    r["m90"] = r["m90"].fillna(0)
    lvl_cols = [c for c in roll if "@" in c]
    for s, p in priors.items():
        if f"{s}@1" in roll:
            lv = r["level"].fillna(2).astype(int) if "level" in r else pd.Series(2, index=r.index)
            num = np.select([lv == 1, lv == 2, lv == 3], [r[f"{s}@1"], r[f"{s}@2"], r[f"{s}@3"]], r[f"{s}@2"])
            r[f"{prefix}{s}"] = (pd.Series(num, index=r.index).fillna(0) + k * p) / (r["m90"] + k)
        else:
            r[f"{prefix}{s}"] = (r[s].fillna(0) + k * p) / (r["m90"] + k)
    drop = [c for c in list(priors) + lvl_cols if c in r]
    r = r.rename(columns={"m90": f"{prefix}n90"}).drop(columns=drop).sort_values("_ord").drop(columns="_ord")
    return r.set_index(rows.index)

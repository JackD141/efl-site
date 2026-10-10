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
           "chances": "Chances created", "sot": "Shots on target", "int": "Interceptions", "xgot": "Expected goals on target (xGOT)",
           "clr": "Clearances", "blk": "Blocks", "tkl": "Tackles"}
RATES = ("npxg", "pxg", "xa", "shots", "chances", "sot", "int", "xgot", "clr", "blk", "tkl")


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


PRIOR_BEFORE = pd.Timestamp("2026-07-01")  # league-average shrinkage targets from completed seasons only (no leakage)


def league_priors(h):
    h = h[h["date"] < PRIOR_BEFORE]
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


# ---------- set pieces (set_pieces.py + the 'Corners' stat) ----------
SP_N = 40          # appearances looked back over
SP_PSEUDO = {"pen": 2.0, "corner": 10.0}  # pseudo-events at the typical share (shrinkage for small samples)


def setpiece_history():
    """Per player per FotMob appearance: penalties / corners / direct free kicks he took, and his team's penalties and
    corners in that match."""
    frames = []
    for f in sorted(FM.glob("*/player_matches.csv")):
        pm = pd.read_csv(f, low_memory=False)
        pm = pm[pd.to_numeric(pm["Minutes played"], errors="coerce").fillna(0) > 0]
        frames.append(pm[["match_id", "fm_player_id", "fm_team_id", "utc", "Minutes played"] + (["Corners"] if "Corners" in pm else [])])
    h = pd.concat(frames, ignore_index=True).rename(columns={"Minutes played": "mins", "Corners": "corners"})
    h["mins"] = pd.to_numeric(h["mins"], errors="coerce").fillna(0)
    h["corners"] = pd.to_numeric(h.get("corners"), errors="coerce").fillna(0)
    sp = pd.read_csv(FM / "set_pieces.csv")
    h = h.merge(sp.drop(columns=["fm_team_id"]), on=["match_id", "fm_player_id"], how="left")
    for c in ("pens", "pen_goals", "pen_xg", "fk_shots", "fk_xg"):
        h[c] = h[c].fillna(0)
    team = h.groupby(["match_id", "fm_team_id"]).agg(team_pens=("pens", "sum"), team_corners=("corners", "sum")).reset_index()
    h = h.merge(team, on=["match_id", "fm_team_id"])
    mp = pd.read_csv(FM / "player_map.csv")[["fm_player_id", "player_id"]]
    h = h.merge(mp, on="fm_player_id")
    h["date"] = pd.to_datetime(h["utc"], utc=True).dt.tz_localize(None).dt.normalize()
    return h.sort_values(["player_id", "date"]).reset_index(drop=True)


def setpiece_constants(h):
    """League constants from completed seasons only: typical share of his team's penalties / corners for a player in
    the game, penalties per team-match, penalty conversion."""
    h = h[h["date"] < PRIOR_BEFORE]
    team_matches = h.drop_duplicates(["match_id", "fm_team_id"])
    return dict(pen_share0=float(h["pens"].sum() / h["team_pens"].sum()), corner_share0=float(h["corners"].sum() / h["team_corners"].sum()),
                pens_per_team_match=float(team_matches["team_pens"].mean()), pen_conv=float(h["pen_goals"].sum() / h["pens"].sum()),
                fk_shots90=float(h["fk_shots"].sum() / (h["mins"].sum() / 90)), fk_xg90=float(h["fk_xg"].sum() / (h["mins"].sum() / 90)),
                corners90=float(h["corners"].sum() / (h["mins"].sum() / 90)))


def setpiece_features(rows, hist=None, const=None, n=SP_N):
    """As of each row's date (strictly before): fm_penshare, fm_penrate (expected penalty goals per 90 if he plays the
    game: share x league penalties per team-match x conversion), fm_cornershare, fm_corners (per 90), fm_fkxg (per 90)."""
    h = setpiece_history() if hist is None else hist
    c = const or setpiece_constants(h)
    g = h.groupby("player_id")
    roll = pd.DataFrame({"player_id": h["player_id"], "date": h["date"]})
    for col in ("mins", "pens", "team_pens", "corners", "team_corners", "fk_xg", "fk_shots"):
        roll["s_" + col] = g[col].transform(lambda s: s.rolling(n, min_periods=1).sum())
    roll = roll.drop_duplicates(["player_id", "date"], keep="last").sort_values("date")
    r = rows.copy()
    r["_ord"] = np.arange(len(r))
    r["date"] = pd.to_datetime(r["date"]).dt.normalize()
    r = pd.merge_asof(r.sort_values("date"), roll, on="date", by="player_id", allow_exact_matches=False, direction="backward")
    for col in [x for x in roll if x.startswith("s_")]:
        r[col] = r[col].fillna(0)
    pp, cp = SP_PSEUDO["pen"], SP_PSEUDO["corner"]
    r["fm_penshare"] = (r["s_pens"] + pp * c["pen_share0"]) / (r["s_team_pens"] + pp)
    r["fm_penrate"] = r["fm_penshare"] * c["pens_per_team_match"] * c["pen_conv"]
    r["fm_cornershare"] = (r["s_corners"] + cp * c["corner_share0"]) / (r["s_team_corners"] + cp)
    r["fm_corners"] = (r["s_corners"] + 3 * c["corners90"]) / (r["s_mins"] / 90 + 3)
    r["fm_fkxg"] = (r["s_fk_xg"] + 3 * c["fk_xg90"]) / (r["s_mins"] / 90 + 3)
    # raw counts over the window, for the "Pens" / "Corners" / "FKs" tags on the site
    r = r.rename(columns={"s_pens": "sp_pens", "s_team_pens": "sp_team_pens", "s_corners": "sp_corners",
                          "s_team_corners": "sp_team_corners", "s_fk_shots": "sp_fk_shots"})
    r = r.drop(columns=[x for x in roll if x.startswith("s_") and x in r]).sort_values("_ord").drop(columns="_ord")
    return r.set_index(rows.index)


def latest_team_and_pen_takers(h, before):
    """({player_id: his latest FotMob team id}, {team id: player_id who took that team's most recent penalty}) using
    matches before `before`. Takers often change after a miss, so the most recent taker is the best guess at the
    current one (our history only covers EFL games, e.g. not a player's Premier League seasons)."""
    h = h[h["date"] < pd.Timestamp(before)]
    team = h.sort_values("date").groupby("player_id")["fm_team_id"].last().to_dict()
    pens = h[h["pens"] > 0].sort_values("date")
    taker = pens.groupby("fm_team_id")["player_id"].last().to_dict()
    return team, taker


def setpiece_tags(r, recent_taker=False):
    """Tags with a reason, from the window's counts (last SP_N games): Pens (took his team's most recent penalty, or
    2+ and half or more of his team's penalties in his games), Corners (10+ and a quarter or more of his team's corners),
    FKs (3+ direct free-kick shots)."""
    tags = []
    pens, tpens = int(r.get("sp_pens", 0)), int(r.get("sp_team_pens", 0))
    if (pens >= 2 and r.get("fm_penshare", 0) >= 0.5) or (recent_taker and pens >= 1):
        why = f"took {pens} of his team's {tpens} penalties in his last {SP_N} games" + (", including the most recent" if recent_taker else "")
        tags.append(dict(t="Pens", why=why))
    corners, tcorners = int(r.get("sp_corners", 0)), int(r.get("sp_team_corners", 0))
    if corners >= 10 and r.get("fm_cornershare", 0) >= 0.25:
        tags.append(dict(t="Corners", why=f"took {corners} of his team's {tcorners} corners in his last {SP_N} games ({corners / max(tcorners, 1):.0%})"))
    fks = int(r.get("sp_fk_shots", 0))
    if fks >= 3:
        tags.append(dict(t="FKs", why=f"{fks} direct free-kick shots in his last {SP_N} games"))
    return tags

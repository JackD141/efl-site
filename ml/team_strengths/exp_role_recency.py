"""Experiment: does recency weighting (and down-weighting games at a previous club) improve the midfielder role feature?
Role today = flat sum over his last 20 appearances (any club). Variants: exponential decay with half-life h appearances,
and games at other clubs multiplied by c. Same protocol: fit on 2025/26 GW1-23, compare validation NB log-likelihood
(GW24-35) with the chosen feature set / window / shrinkage of each stat. Only the role feature changes.
python exp_role_recency.py
"""
import json
import warnings

import numpy as np
import pandas as pd

import attack_model as am
import defence_model as dm
from saves_model import nb_loglik

warnings.filterwarnings("ignore")
HALF_LIVES = [None, 5, 10, 20]   # None = flat last 20 (current)
CLUB_W = [1.0, 0.5, 0.25]


def role_weighted(d, col, prior, k, h, c):
    tg = d.groupby(["season", "squad_id", "gameweek"]).agg(tx=(col, "sum"), tm=("minutes_played", "sum")).reset_index()
    x = d.merge(tg, on=["season", "squad_id", "gameweek"])
    mates = (x["tx"] - x[col]) / ((x["tm"] - x["minutes_played"]).clip(lower=1) / 90)
    x["exp"] = mates.where(x["tm"] > x["minutes_played"], prior) * x["minutes_played"] / 90
    x = x.sort_values(["player_id", "season", "gameweek"]).reset_index(drop=True)
    role = np.empty(len(x))
    for pid, idx in x.groupby("player_id").indices.items():
        y, e, club = x.loc[idx, col].to_numpy(float), x.loc[idx, "exp"].to_numpy(float), x.loc[idx, "squad_id"].to_numpy()
        for j in range(len(idx)):
            lo = max(0, j - am.ROLE_GAMES)
            age = np.arange(j - lo, 0, -1)  # 1 = most recent earlier appearance
            w = np.ones(j - lo) if h is None else 0.5 ** ((age - 1) / h)
            w = w * np.where(club[lo:j] == club[j], 1.0, c)
            role[idx[j]] = (np.dot(w, y[lo:j]) + k * prior) / (np.dot(w, e[lo:j]) + k * prior)
    x["role"] = role
    return x


def main():
    art = json.loads(am.MODEL_PATH.read_text(encoding="utf-8"))["positions"]["MID"]
    d = am.load("MID")
    rows = []
    for stat, cfg in art.items():
        col, prior, k, n, feats = cfg["column"], cfg["prior"], cfg["k_role"], cfg["window"], cfg["features"]
        team, opp = dm.style_tables(d, col, n, prior)
        for h in HALF_LIVES:
            for c in CLUB_W:
                x = role_weighted(d, col, prior, k, h, c)
                x = x.merge(team, on=["season", "squad_id", "gameweek"], how="left").merge(opp, on=["season", "opponent_id", "gameweek"], how="left")
                x["team"], x["opp"] = x["team"].fillna(prior), x["opp"].fillna(prior)
                x["y"], x["t"] = x[col], x["minutes_played"] / 90
                x = x[x["minutes_played"] >= am.MIN_ROW]
                tr = x[(x["season"] == "2526") & (x["gameweek"] <= 23)]
                va = x[(x["season"] == "2526") & (x["gameweek"] > 23)]
                f = am.fit(tr, feats)
                mu = np.clip(am.predict(f, va), 1e-6, None)
                rows.append(dict(stat=stat, half_life=h or "flat", other_club=c, val_loglik=nb_loglik(va["y"].to_numpy(), mu, f["r"]),
                                 mse=float(((va["y"] - mu) ** 2).mean())))
        r = pd.DataFrame([q for q in rows if q["stat"] == stat])
        base = r[(r["half_life"] == "flat") & (r["other_club"] == 1.0)]["val_loglik"].iloc[0]
        r["gain_vs_current"] = r["val_loglik"] - base
        print(f"\n{stat}:"); print(r.sort_values("val_loglik", ascending=False).head(5).round(5).to_string(index=False))


if __name__ == "__main__":
    main()

"""Operational test: every week refit on all earlier odds in the season, predict the next round's matches.
usage: python run_rolling.py 2425   (tune)   |   python run_rolling.py 2526 <ridge> <half_life>   (held-out test)"""
import itertools
import sys
import warnings

import numpy as np
import pandas as pd

from per_season import RHO, fit_strengths, lam_for, market_targets, predict_probs
from strengths import DIVS, log_loss, outcome

warnings.filterwarnings("ignore")
START = 96  # begin predicting once ~8 gameweeks of odds exist


def rolling(season, ridge, half_life, team_home_ridge=None):
    rows = []
    for div in DIVS:
        d = lam_for(div, season).dropna(subset=["lamH"]).sort_values("Date").reset_index(drop=True)
        teams = sorted(set(d["HomeTeam"]) | set(d["AwayTeam"]))
        edge = d["Date"].iloc[START]
        while edge <= d["Date"].max():
            tr = d[d["Date"] < edge]
            te = d[(d["Date"] >= edge) & (d["Date"] < edge + pd.Timedelta(days=7))]
            if len(te):
                w = None if half_life is None else 0.5 ** ((edge - tr["Date"]).dt.days.to_numpy() / half_life)
                fit = fit_strengths(tr, tr["lamH"].to_numpy(), tr["lamA"].to_numpy(), ridge=ridge, weights=w, teams=teams, team_home_ridge=team_home_ridge)
                for _, r in te.iterrows():
                    P = predict_probs(fit, r["HomeTeam"], r["AwayTeam"], RHO)[0][:3]
                    rows.append((div, r["Date"], *P, *market_targets(r)[:3], int(outcome(r.to_frame().T.astype({"FTHG": int, "FTAG": int}))[0])))
            edge += pd.Timedelta(days=7)
    return pd.DataFrame(rows, columns=["div", "date", "mH", "mD", "mA", "kH", "kD", "kA", "y"])


def score(r):
    P, M, y = r[["mH", "mD", "mA"]].to_numpy(), r[["kH", "kD", "kA"]].to_numpy(), r["y"].to_numpy()
    return log_loss(P, y), log_loss(M, y), np.abs(P - M).mean()


if __name__ == "__main__":
    season = sys.argv[1]
    if len(sys.argv) > 2:
        grid = [(float(sys.argv[2]), None if sys.argv[3] == "none" else float(sys.argv[3]))]
    else:
        grid = list(itertools.product([0.03, 0.1, 0.3], [None, 60, 120, 240]))
    print(f"season {season}   model_ll  market_ll  mae_vs_market")
    for ridge, hl in grid:
        r = rolling(season, ridge, hl)
        a, b, c = score(r)
        per = {d: round(score(r[r['div'] == d])[0] - score(r[r['div'] == d])[1], 4) for d in DIVS}
        print(f"ridge={ridge:<5} half_life={str(hl):<5} {a:.4f}   {b:.4f}   {c:.4f}   gap by league {per}")

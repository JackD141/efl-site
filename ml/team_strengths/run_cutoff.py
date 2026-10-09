"""Simulate 'now': fit each season from only its first ~8 gameweeks of odds (+ optional last-season prior),
then score the remaining matches vs the bookmakers' closing odds.   usage: python run_cutoff.py 2425 2324"""
import itertools
import sys
import warnings

import numpy as np
import pandas as pd

from per_season import RHO, fit_strengths, lam_for, market_targets, predict_probs
from strengths import DIVS, log_loss, outcome

warnings.filterwarnings("ignore")
N_TRAIN = 96  # 8 gameweeks x 12 matches per league
RIDGES = [0.1, 1, 3, 10, 30]
MULTS = [0.0, 0.5, 0.7, 0.9]


def evaluate(season, prev):
    rows = []
    for div in DIVS:
        d = lam_for(div, season).dropna(subset=["lamH"]).sort_values("Date")
        p = lam_for(div, prev).dropna(subset=["lamH"])
        prior = fit_strengths(p, p["lamH"].to_numpy(), p["lamA"].to_numpy(), ridge=0.1)["teams"]
        tr, te = d.iloc[:N_TRAIN], d.iloc[N_TRAIN:]
        teams = sorted(set(d["HomeTeam"]) | set(d["AwayTeam"]))
        y = outcome(te)
        M = np.array([market_targets(r)[:3] for _, r in te.iterrows()])
        rows.append((div, "market", None, None, log_loss(M, y), 0.0, len(te)))
        for ridge, mult in itertools.product(RIDGES, MULTS):
            fit = fit_strengths(tr, tr["lamH"].to_numpy(), tr["lamA"].to_numpy(), ridge=ridge, prior=prior, prior_mult=mult, teams=teams)
            P = np.array([predict_probs(fit, h, a, RHO)[0][:3] for h, a in zip(te["HomeTeam"], te["AwayTeam"])])
            rows.append((div, "model", ridge, mult, log_loss(P, y), np.abs(P - M).mean(), len(te)))
        # last season's ratings alone: team effects pinned to the prior, home/base still fitted on the training matches
        fit = fit_strengths(tr, tr["lamH"].to_numpy(), tr["lamA"].to_numpy(), ridge=1e5, prior=prior, prior_mult=0.8, teams=teams)
        P = np.array([predict_probs(fit, h, a, RHO)[0][:3] for h, a in zip(te["HomeTeam"], te["AwayTeam"])])
        rows.append((div, "last-season only", None, None, log_loss(P, y), np.abs(P - M).mean(), len(te)))
    return pd.DataFrame(rows, columns=["div", "kind", "ridge", "mult", "ll", "mae_vs_market", "n"])


if __name__ == "__main__":
    season, prev = sys.argv[1], sys.argv[2]
    r = evaluate(season, prev)
    pd.set_option("display.width", 200)
    mk = r[r.kind == "market"].groupby("div").ll.first()
    print(f"season {season}: market closing logloss by league", mk.round(4).to_dict(), "mean %.4f" % mk.mean())
    lo = r[r.kind == "last-season only"].set_index("div")
    print("last-season ratings only: ll", lo.ll.round(4).to_dict(), "mean %.4f  mae_vs_market %.4f" % (lo.ll.mean(), lo.mae_vs_market.mean()))
    m = r[r.kind == "model"].groupby(["ridge", "mult"])[["ll", "mae_vs_market"]].mean()
    print("\nmean over leagues (model fitted on first 8 gameweeks of odds):")
    print(m["ll"].unstack().round(4)); print("MAE vs market:"); print(m["mae_vs_market"].unstack().round(4))
    print("\nno-prior (mult=0) best:", m.xs(0.0, level="mult").ll.idxmin(), round(m.xs(0.0, level="mult").ll.min(), 4))

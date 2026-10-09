"""Does knowing the *weekend's* odds (before it is played) sharpen predictions for the following midweek games?

baseline : fit on odds of matches up to 5 days before the midweek game (weekend odds not yet known)
anchored : also include the weekend's own odds (what we have before a double gameweek: weekend prices exist, midweek ones don't)
Scored against the bookmakers' closing odds for the midweek game.   usage: python run_midweek.py
"""
import warnings

import numpy as np
import pandas as pd

from per_season import RHO, fit_strengths, lam_for, market_targets, predict_probs
from strengths import DIVS, log_loss, outcome

warnings.filterwarnings("ignore")
START = 96


def run(season, half_life, ridge=0.1):
    rows = []
    for div in DIVS:
        d = lam_for(div, season).dropna(subset=["lamH"]).sort_values("Date").reset_index(drop=True)
        teams = sorted(set(d["HomeTeam"]) | set(d["AwayTeam"]))
        for D, te in d.iloc[START:].groupby("Date"):
            if D.weekday() not in (1, 2, 3):
                continue
            preds = {}
            for name, cut in (("base", D - pd.Timedelta(days=5)), ("anch", D)):
                tr = d[d["Date"] < cut]
                w = 0.5 ** ((D - tr["Date"]).dt.days.to_numpy() / half_life)
                fit = fit_strengths(tr, tr["lamH"].to_numpy(), tr["lamA"].to_numpy(), ridge=ridge, weights=w, teams=teams)
                preds[name] = fit
            for _, r in te.iterrows():
                M = market_targets(r)[:3]
                y = int(outcome(r.to_frame().T.astype({"FTHG": int, "FTAG": int}))[0])
                rows.append(dict(div=div, y=y, **{f"mk{i}": M[i] for i in range(3)},
                                 **{f"{n}{i}": predict_probs(f, r["HomeTeam"], r["AwayTeam"], RHO)[0][i] for n, f in preds.items() for i in range(3)}))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("midweek matches, scored vs bookmakers' closing odds")
    for hl in (60, 30, 20):
        parts = [run(s, hl) for s in ("2425", "2526")]
        r = pd.concat(parts, ignore_index=True)
        y = r["y"].to_numpy()
        M = r[["mk0", "mk1", "mk2"]].to_numpy()
        out = [f"half-life {hl:>2}  n={len(r)}  market ll {log_loss(M, y):.4f}"]
        for n in ("base", "anch"):
            P = r[[f"{n}0", f"{n}1", f"{n}2"]].to_numpy()
            out.append(f"{n}: ll {log_loss(P, y):.4f} mae {np.abs(P - M).mean() * 100:.2f}pp")
        print("  ".join(out))

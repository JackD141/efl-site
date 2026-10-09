"""How does accuracy degrade with horizon, and which memory length is best at each horizon?
Fit weekly (odds-implied strengths, as in run_rolling); predict matches 1..12 weeks ahead; compare with the
bookmakers' closing odds.   usage: python run_horizon.py"""
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from per_season import RHO, fit_strengths, lam_for, market_targets, predict_probs
from strengths import DIVS, outcome

warnings.filterwarnings("ignore")
START = 96
HLS = [15, 30, 60, 120, 400]
MAXK = 12


def blend(f_a, f_b, w):
    f = dict(f_a)
    f["coef"] = w * f_a["coef"] + (1 - w) * f_b["coef"]
    f["home"] = w * f_a["home"] + (1 - w) * f_b["home"]
    f["base"] = w * f_a["base"] + (1 - w) * f_b["base"]
    return f


def season_rows(season):
    rows = []
    for div in DIVS:
        d = lam_for(div, season).dropna(subset=["lamH"]).sort_values("Date").reset_index(drop=True)
        teams = sorted(set(d["HomeTeam"]) | set(d["AwayTeam"]))
        edge = d["Date"].iloc[START]
        while edge <= d["Date"].max():
            tr = d[d["Date"] < edge]
            fut = d[(d["Date"] >= edge) & (d["Date"] < edge + pd.Timedelta(days=7 * MAXK))]
            if len(fut):
                fits = {hl: fit_strengths(tr, tr["lamH"].to_numpy(), tr["lamA"].to_numpy(), ridge=0.1,
                                          weights=0.5 ** ((edge - tr["Date"]).dt.days.to_numpy() / hl), teams=teams) for hl in HLS}
                fits["15+120"] = blend(fits[15], fits[120], 0.5)
                for _, r in fut.iterrows():
                    k = (r["Date"] - edge).days // 7 + 1
                    M = market_targets(r)[:3]
                    y = int(outcome(r.to_frame().T.astype({"FTHG": int, "FTAG": int}))[0])
                    for name, f in fits.items():
                        P = predict_probs(f, r["HomeTeam"], r["AwayTeam"], RHO)[0][:3]
                        rows.append((season, div, str(name), k, y, *P, *M))
            edge += pd.Timedelta(days=7)
    return pd.DataFrame(rows, columns=["season", "div", "hl", "k", "y", "pH", "pD", "pA", "kH", "kD", "kA"])


if __name__ == "__main__":
    r = pd.concat(Parallel(n_jobs=2)(delayed(season_rows)(s) for s in ("2425", "2526")), ignore_index=True)
    r["mae"] = (r[["pH", "pD", "pA"]].to_numpy() - r[["kH", "kD", "kA"]].to_numpy()).__abs__().mean(axis=1) * 100
    P = r[["pH", "pD", "pA"]].to_numpy()
    r["ll"] = -np.log(np.clip(P[np.arange(len(r)), r["y"].to_numpy()], 1e-9, 1))
    r["kk"] = pd.cut(r["k"], [0, 1, 2, 3, 4, 6, 8, 12], labels=["1", "2", "3", "4", "5-6", "7-8", "9-12"])
    pd.set_option("display.width", 200)
    print("mean abs gap to market (pp), by weeks ahead:")
    print(r.pivot_table(index="kk", columns="hl", values="mae", aggfunc="mean").round(2).to_string())
    print("\nlog-loss vs results, by weeks ahead (lower is better):")
    print(r.pivot_table(index="kk", columns="hl", values="ll", aggfunc="mean").round(4).to_string())
    print("\nn per bucket:", r[r.hl == "15"].groupby("kk").size().to_dict())

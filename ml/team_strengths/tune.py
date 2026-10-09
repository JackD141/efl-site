"""Tune half-life / ridge on 2024/25 (train 2023/24 onward); report vs bookmaker closing odds."""
import itertools
import sys
import warnings

import numpy as np
from joblib import Parallel, delayed

from strengths import DIVS, load_div, log_loss, market_probs, outcome, rps, walk_forward

warnings.filterwarnings("ignore")
SEASON = sys.argv[1] if len(sys.argv) > 1 else "2425"
HALF_LIVES = [90, 180, 365, 1000]
ALPHAS = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]


def run(div, hl, a):
    df = load_div(div)
    res = walk_forward(df, SEASON, hl, a)
    y, P, M = outcome(res), res[["pH", "pD", "pA"]].to_numpy(), market_probs(res)
    ok = ~np.isnan(M).any(axis=1)
    return div, hl, a, log_loss(P[ok], y[ok]), rps(P[ok], y[ok]), log_loss(M[ok], y[ok]), rps(M[ok], y[ok]), ok.sum()


jobs = list(itertools.product(DIVS, HALF_LIVES, ALPHAS))
out = Parallel(n_jobs=-1)(delayed(run)(*j) for j in jobs)
import pandas as pd

r = pd.DataFrame(out, columns=["div", "half_life", "alpha", "ll", "rps", "mkt_ll", "mkt_rps", "n"])
r.to_csv("tune_results.csv", index=False)
pd.set_option("display.width", 200)
for d in DIVS:
    print(f"\n=== {d} {DIVS[d]} (market logloss {r[r['div']==d].mkt_ll.iloc[0]:.4f}, rps {r[r['div']==d].mkt_rps.iloc[0]:.4f}) ===")
    print(r[r["div"] == d].pivot(index="half_life", columns="alpha", values="ll").round(4))
avg = r.groupby(["half_life", "alpha"])[["ll", "rps"]].mean().round(4)
print("\n=== mean across leagues ===")
print(avg.sort_values("ll").head(8))
print("market mean ll %.4f rps %.4f" % (r.groupby("div").mkt_ll.first().mean(), r.groupby("div").mkt_rps.first().mean()))

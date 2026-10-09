"""Which extra parameters / memory lengths reduce the gap to the market?  Weekly-refit backtest, 2024/25 + 2025/26.
usage: python run_variants.py"""
import itertools
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from run_rolling import rolling

warnings.filterwarnings("ignore")
HALF_LIVES = [60, 30, 15, 8]
HOME_RIDGE = [None, 1.0, 3.0, 10.0]


def one(hl, thr):
    parts = [rolling(s, 0.1, hl, thr) for s in ("2425", "2526")]
    r = pd.concat(parts, ignore_index=True)
    y = r["y"].to_numpy().astype(int)
    P, M = r[["mH", "mD", "mA"]].to_numpy(), r[["kH", "kD", "kA"]].to_numpy()
    ll = lambda p: float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1))))
    return dict(half_life=hl, team_home_ridge=thr, n=len(r), model_ll=ll(P), market_ll=ll(M), gap=ll(P) - ll(M), mae_pp=float(np.abs(P - M).mean() * 100),
                big_gap_share=float((np.abs(P - M).max(axis=1) > 0.08).mean()))


if __name__ == "__main__":
    res = Parallel(n_jobs=-1)(delayed(one)(hl, thr) for hl, thr in itertools.product(HALF_LIVES, HOME_RIDGE))
    out = pd.DataFrame(res)
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))
    out.to_csv("variants_results.csv", index=False)

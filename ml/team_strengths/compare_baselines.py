"""Defender stat model vs simple baselines on 2026/27 (never used for choosing).   python compare_baselines.py

Baselines: the player's plain average count over his last X full games (85+ min, any club, both seasons; league average
if he has none). Model: the chosen per-stat GLM refitted on all of 2025/26 (as in defence_model.py's test step).
Metrics on full games: MAE and MSE of the count, and MAE of the points from that stat (floor(count / unit)).
"""
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import nbinom

import defence_model as dm

warnings.filterwarnings("ignore")


def main():
    art = json.loads(dm.MODEL_PATH.read_text(encoding="utf-8"))
    d = dm.attach_odds(dm.load_fantasy()).dropna(subset=["lam_own"]).reset_index(drop=True)
    rows = []
    for stat, cfg in art["stats"].items():
        col, unit = cfg["column"], cfg["unit"]
        x = dm.build(d, stat, cfg["window"], cfg["prior"])
        fit = dm.fit(x[(x["season"] == "2526") & (x["minutes_played"] >= dm.MIN_FULL)], cfg["features"])
        x["model"] = dm.predict(fit, x)
        # simple baselines: average count in his previous X full games
        x = x.sort_values(["player_id", "season", "gameweek"])
        fullcnt = x[col].where(x["minutes_played"] >= dm.MIN_FULL)
        for n in (3, 5, 10, 20):
            x[f"last{n}"] = fullcnt.groupby(x["player_id"]).transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())
        te = x[(x["season"] == "2627") & (x["minutes_played"] >= dm.MIN_FULL)].copy()
        y = te[col].to_numpy()
        k = np.arange(120)

        def pts(mu, r):
            pmf = nbinom.pmf(k[None, :], r, (r / (r + np.asarray(mu)))[:, None])
            return (pmf * (k // unit)).sum(1)

        preds = {"league average": np.full(len(te), cfg["prior"])}
        for n in (3, 5, 10, 20):
            preds[f"his last {n} full games"] = te[f"last{n}"].fillna(cfg["prior"]).to_numpy()
        preds["model"] = te["model"].to_numpy()
        for name, mu in preds.items():
            # points: the model uses its negative-binomial spread; baselines get the same spread for a fair comparison
            rows.append(dict(stat=col, predictor=name, MAE=np.abs(y - mu).mean(), MSE=((y - mu) ** 2).mean(),
                             MAE_points=np.abs(y // unit - pts(np.clip(mu, 1e-3, None), fit["r"])).mean(), n=len(y)))
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    return res


if __name__ == "__main__":
    main()

"""End-to-end check of midfielder xP with predicted minutes vs the old 90-or-0 rule, on 2026/27 (models fitted on 2025/26).
Every squad midfielder's fixture counts (including those he did not play in), as when picking a team.
python eval_xmins_xp.py
Note: stat rates for rows without minutes use the same role / style features, with the role window counted in fixtures
(zeros included) rather than appearances - close to, not identical to, the live pipeline.
"""
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
from sklearn.isotonic import IsotonicRegression

import attack_model as am
import defence_model as dm
import minutes_model as mm

warnings.filterwarnings("ignore")
POS = "MID"
PTS = {"goals": 6, "assists": 3, "sot": 1, "int": 2}
KEY = ["player_id", "season", "gameweek", "opponent_id"]


def main():
    art = json.loads(am.MODEL_PATH.read_text(encoding="utf-8"))["positions"][POS]
    # all MID rows, including 0 minutes
    raw = mm.load_rows(POS)
    d = dm.attach_odds(raw.drop(columns=["hid", "aid", "date"])).dropna(subset=["lam_own"]).reset_index(drop=True)
    rates = d[KEY + ["minutes_played", "points", "yellow_cards", "red_cards", "own_goals", "penalty_misses"]].copy()
    for stat, cfg in art.items():
        x = am.build(d, cfg["column"], cfg["window"], cfg["prior"], cfg["k_role"])
        fitrows = x[(x["season"] == "2526") & (x["minutes_played"] >= am.MIN_ROW)]
        f = am.fit(fitrows, cfg["features"])
        x["rate"] = f["model"].predict(am.design(x, f["feats"], f["scaler"])[0])
        x[f"r_{stat}"] = f["r"]
        rates = rates.merge(x[KEY + ["rate", f"r_{stat}"]].rename(columns={"rate": f"rate_{stat}"}).drop_duplicates(KEY), on=KEY, how="left")
    tr = d[d["season"] == "2526"]
    m90 = tr["minutes_played"].sum() / 90
    card90 = float((tr["yellow_cards"] + 3 * tr["red_cards"] + 3 * tr["own_goals"] + 3 * tr["penalty_misses"]).sum() / m90)
    # minutes: model fitted on 2025/26, old rule, and the actual minutes (oracle, for reference)
    feat = mm.add_features(raw)
    mod = mm.models()["gbm"]().fit(feat.loc[feat["season"] == "2526", mm.FEATURES], feat.loc[feat["season"] == "2526", "mins"])
    feat["xmins_model"] = np.clip(mod.predict(feat[mm.FEATURES]), 0, 90)
    va = feat[(feat["season"] == "2526") & (feat["gameweek"] > 23)]
    mod_tr = mm.models()["gbm"]().fit(feat.loc[(feat["season"] == "2526") & (feat["gameweek"] <= 23), mm.FEATURES], feat.loc[(feat["season"] == "2526") & (feat["gameweek"] <= 23), "mins"])
    iso = IsotonicRegression(y_min=0, y_max=2, out_of_bounds="clip").fit(np.clip(mod_tr.predict(va[mm.FEATURES]), 0, 90), mm.app_points(va["mins"]))
    rates = rates.merge(feat[KEY + ["xmins_model", "b_rule", "mins"]].drop_duplicates(KEY), on=KEY, how="left")
    te = rates[(rates["season"] == "2627")].dropna(subset=[f"rate_{s}" for s in art] + ["xmins_model"]).copy()
    k = np.arange(60)

    def xp(mins, app):
        t = mins / 90
        tot = app.copy()
        for s, p in PTS.items():
            tot += p * te[f"rate_{s}"] * t
        r = te["r_kp"].iloc[0]
        mu = (te["rate_kp"] * t).to_numpy()
        tot += (nbinom.pmf(k[None, :], r, (r / (r + mu))[:, None]) * (k // 2)).sum(1)
        mug = (te["rate_goals"] * t).to_numpy()
        tot += 5 * (1 - poisson.cdf(2, mug))
        return tot - card90 * t

    te["xp_model"] = xp(te["xmins_model"], iso.predict(te["xmins_model"]))
    te["xp_rule"] = xp(te["b_rule"], mm.app_points(te["b_rule"]).astype(float))
    te["xp_oracle"] = xp(te["mins"], mm.app_points(te["mins"]).astype(float))
    y = te["points"]
    print(f"{POS} 2026/27, all squad fixtures (n={len(te)}), actual points mean {y.mean():.2f}")
    for name in ("xp_rule", "xp_model", "xp_oracle"):
        p = te[name]
        print(f"  {name:10s} MAE {np.abs(y - p).mean():.3f}  RMSE {np.sqrt(((y - p) ** 2).mean()):.3f}  corr {np.corrcoef(p, y)[0, 1]:.3f}  mean {p.mean():.2f}")
    # picking: actual points of the top N by xP each gameweek (what the optimiser would choose)
    for n in (5, 10, 20, 40):
        res = {}
        for name in ("xp_rule", "xp_model", "xp_oracle"):
            top = te.sort_values(name, ascending=False).groupby("gameweek").head(n)
            res[name] = round(float(top["points"].mean()), 2)
        dm_, se, ng = paired(te, n)
        print(f"  top {n} per gameweek by xP -> mean actual points: {res}; model - rule {dm_:+.2f} (se {se:.2f}, {ng} gameweeks)")



def paired(te, n):
    """Per-gameweek mean actual points of the top n by each xP, and the paired difference with its standard error."""
    rows = []
    for gw, g in te.groupby("gameweek"):
        rows.append({name: g.nlargest(n, name)["points"].mean() for name in ("xp_rule", "xp_model")})
    r = pd.DataFrame(rows)
    diff = r["xp_model"] - r["xp_rule"]
    return diff.mean(), diff.std(ddof=1) / np.sqrt(len(diff)), len(diff)


if __name__ == "__main__":
    main()

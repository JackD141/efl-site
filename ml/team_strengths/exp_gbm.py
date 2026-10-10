"""Experiment: gradient-boosted stat models vs the GLMs (and their average), same features pipeline, same protocol.
python exp_gbm.py [MID FWD DEF]

For each position / stat: the live configuration's feature table (window, role shrinkage) from player_model.build, then
  GLM    the live feature set, refitted
  GBM    HistGradientBoostingRegressor, Poisson loss, minutes as exposure (target rate per 90, weight minutes/90), on the
         GLM's inputs plus related FotMob history (20 and 40 games) and league level; hyper-parameters from a small grid
         fixed in advance, chosen on the folds
  blend  mean of the two
scored on the two rolling validation folds inside 2025/26 (log-likelihood of the actual count). No 2026/27 data is used
for any choice here. Also prints a diagnostic: actual / predicted by minutes played (does a per-90 rate hold for short
spells?).
"""
import itertools
import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import player_model as pm
from saves_model import nb_r

warnings.filterwarnings("ignore")
HIST = {
    "goals": ["fm_npxg", "fm40_npxg", "fm_pxg", "fm_shots", "fm40_shots", "fm_xgot"],
    "assists": ["fm_xa", "fm40_xa", "fm_chances", "fm40_chances"],
    "sot": ["fm_shots", "fm40_shots", "fm_sot", "fm40_sot", "fm_npxg", "fm40_npxg"],
    "kp": ["fm_chances", "fm40_chances", "fm_xa", "fm40_xa"],
    "int": ["fm_int", "fm40_int"],
    "clr": ["fm_clr", "fm40_clr"], "blk": ["fm_blk", "fm40_blk"], "tkl": ["fm_tkl", "fm40_tkl"],
}
GRID = [dict(learning_rate=0.05, max_iter=it, max_leaf_nodes=leaves, min_samples_leaf=leaf, l2_regularization=1.0)
        for it, leaves, leaf in itertools.product((150, 400), (7, 15), (50, 200))]


def gbm_features(stat, glm_feats):
    base = ["role", "team", "mates", "opp", "lam_own", "lam_opp", "home", "level"]
    return list(dict.fromkeys(base + [f for f in glm_feats if f not in base] + HIST.get(stat, [])))


def fit_gbm(x, feats, spec, params):
    m = HistGradientBoostingRegressor(loss="poisson", random_state=0, **params)
    m.fit(x[feats], x["y"] / x["t"], sample_weight=x["t"])
    raw = m.predict(x[feats]) * x["t"]
    pen = (x["fm_pxg"] * x["t"]).to_numpy() if spec["pens"] else 0.0
    scale = 1.0 if spec["target"] == spec["column"] else float((x["cnt"].sum() - np.sum(pen)) / raw.sum())
    mu = scale * raw + pen
    return dict(model=m, feats=feats, scale=scale, r=nb_r(x["cnt"].to_numpy(), np.asarray(mu)))


def pred_gbm(f, x, spec):
    raw = f["model"].predict(x[f["feats"]]) * x["t"].to_numpy()
    pen = (x["fm_pxg"] * x["t"]).to_numpy() if spec["pens"] else 0.0
    return f["scale"] * raw + pen


def main(positions):
    out = []
    for pos in positions:
        d = pm.load(pos)
        art = json.loads(pm.model_path(pos).read_text(encoding="utf-8"))
        for stat, a in art.items():
            spec = pm.SPECS[pos][stat]
            x, prior, carry = pm.build(d, spec, a["window"], a["k_role"], pm.league_adjustments(pos)[stat])
            s25 = x[x["season"] == "2526"]
            gf = gbm_features(stat, a["features"])
            res = {"GLM": [], "blend": []}
            gbm_scores = {i: [] for i in range(len(GRID))}
            gbm_preds = {i: [] for i in range(len(GRID))}
            diag = []
            for lo, hi in pm.FOLDS:
                tr, va = s25[s25["gameweek"] <= lo], s25[(s25["gameweek"] > lo) & (s25["gameweek"] <= hi)]
                y = va["cnt"].to_numpy()
                g = pm.fit(tr, a["features"], spec)
                mg = pm.predict(g, va, spec)
                res["GLM"].append(pm.score(y, mg, g["r"])["loglik"])
                for i, params in enumerate(GRID):
                    b = fit_gbm(tr, gf, spec, params)
                    mb = pred_gbm(b, va, spec)
                    gbm_scores[i].append(pm.score(y, mb, b["r"])["loglik"])
                    gbm_preds[i].append((mb, g, mg, y, tr, b))
                diag.append(va.assign(mu=mg))
            best_i = max(gbm_scores, key=lambda i: np.mean(gbm_scores[i]))
            for mb, g, mg, y, tr, b in gbm_preds[best_i]:
                blend = 0.5 * (mb + mg)
                mu_tr = 0.5 * (pm.predict(g, tr, spec) + pred_gbm(b, tr, spec))
                res["blend"].append(pm.score(y, blend, nb_r(tr["cnt"].to_numpy(), np.asarray(mu_tr)))["loglik"])
            row = dict(pos=pos, stat=stat, GLM=np.mean(res["GLM"]), GBM=np.mean(gbm_scores[best_i]), blend=np.mean(res["blend"]),
                       gbm_params=GRID[best_i], n_val=sum(len(v) for v in diag))
            row["best"] = max(("GLM", "GBM", "blend"), key=lambda k: row[k])
            out.append(row)
            dg = pd.concat(diag)
            dg["mins_bucket"] = pd.cut(dg["minutes_played"], [9, 30, 59, 75, 90])
            cal = dg.groupby("mins_bucket", observed=True).apply(lambda q: q["cnt"].sum() / q["mu"].sum()).round(2).to_dict()
            print(f"{pos} {stat}: GLM {row['GLM']:.4f} | GBM {row['GBM']:.4f} {GRID[best_i]} | blend {row['blend']:.4f} -> best {row['best']}"
                  f" | GLM actual/predicted by minutes {cal}", flush=True)
    r = pd.DataFrame(out)
    r["gbm_gain"] = r["GBM"] - r["GLM"]
    r["blend_gain"] = r["blend"] - r["GLM"]
    print("\n" + r[["pos", "stat", "GLM", "GBM", "blend", "gbm_gain", "blend_gain", "best"]].round(5).to_string(index=False))
    (pm.HERE / "models" / "exp_gbm_results.json").write_text(r.to_json(orient="records", indent=1), encoding="utf-8")


if __name__ == "__main__":
    main([p.upper() for p in sys.argv[1:]] or ["MID", "FWD", "DEF"])

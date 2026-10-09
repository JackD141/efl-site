"""Experiment: do FotMob xG-type features improve the midfielder stat models?
For each stat, keep the chosen window / role shrinkage / club weighting and compare feature sets on validation
(2025/26 GW24-35, fitted on GW1-23), then test once on 2026/27 (fitted on all 2025/26).
python exp_xg.py [MID|FWD]
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd

import attack_model as am
import minutes_model as mm

sys.path.insert(0, str(am.REPO / "ml" / "fotmob"))
import features as fmf  # noqa: E402
from saves_model import nb_loglik  # noqa: E402

warnings.filterwarnings("ignore")
BASE = ["role", "team", "opp", "lam_own", "lam_opp", "home"]
CANDIDATES = {
    "goals": {"current": None, "+npxg": BASE + ["fm_npxg"], "+npxg+pens": BASE + ["fm_npxg", "fm_pxg"],
              "npxg+pens instead of role": BASE[1:] + ["fm_npxg", "fm_pxg"], "+npxg+pens+shots": BASE + ["fm_npxg", "fm_pxg", "fm_shots"]},
    "assists": {"current": None, "+xa": BASE + ["fm_xa"], "role+xa": ["role", "fm_xa"], "+xa+chances": BASE + ["fm_xa", "fm_chances"]},
    "sot": {"current": None, "+shots": BASE + ["fm_shots"], "+shots+npxg": BASE + ["fm_shots", "fm_npxg"]},
    "kp": {"current": None, "+chances": BASE + ["fm_chances"], "+chances+xa": BASE + ["fm_chances", "fm_xa"]},
    "int": {"current": None},
}


def load_with_signals(position, k):
    d = am.load(position).drop(columns=["hid", "aid"])
    d = mm.attach_dates(d)
    return fmf.signal_rates(d, k=k)


def main(position="MID"):
    art = json.loads(am.MODEL_PATH.read_text(encoding="utf-8"))["positions"][position]
    out = []
    for k_sig in (3.0, 8.0):
        d = load_with_signals(position, k_sig)
        cov = (d["fm_n90"] > 0).groupby(d["season"]).mean().round(3).to_dict()
        print(f"\nsignal shrinkage {k_sig}: rows with FotMob history by season {cov}")
        for stat, cfg in art.items():
            if stat not in CANDIDATES:
                continue
            x = am.build(d, cfg["column"], cfg["window"], cfg["prior"], cfg["k_role"], cfg.get("club_w", 1.0))
            x = x[x["minutes_played"] >= am.MIN_ROW]
            tr = x[(x["season"] == "2526") & (x["gameweek"] <= 23)]
            va = x[(x["season"] == "2526") & (x["gameweek"] > 23)]
            tv, te = x[x["season"] == "2526"], x[x["season"] == "2627"]
            for name, feats in CANDIDATES[stat].items():
                feats = feats or cfg["features"]
                if name == "current" and k_sig != 3.0:
                    continue
                f = am.fit(tr, feats)
                mv = np.clip(am.predict(f, va), 1e-6, None)
                f2 = am.fit(tv, feats)
                mt = np.clip(am.predict(f2, te), 1e-6, None)
                out.append(dict(stat=stat, k_sig=k_sig if name != "current" else "-", features=name,
                                val_loglik=nb_loglik(va["y"].to_numpy(), mv, f["r"]), val_mse=float(((va["y"] - mv) ** 2).mean()),
                                test_loglik=nb_loglik(te["y"].to_numpy(), mt, f2["r"]), test_mse=float(((te["y"] - mt) ** 2).mean()),
                                coefs=dict(zip(feats, np.round(f2["model"].coef_, 3)))))
    r = pd.DataFrame(out)
    for stat, g in r.groupby("stat", sort=False):
        base = g[g["features"] == "current"].iloc[0]
        g = g.assign(val_gain=g["val_loglik"] - base["val_loglik"], test_gain=g["test_loglik"] - base["test_loglik"])
        print(f"\n=== {position} {stat} (validation n={len(va)}, test n={len(te)}) ===")
        print(g.sort_values("val_loglik", ascending=False)[["features", "k_sig", "val_loglik", "val_gain", "val_mse", "test_loglik", "test_gain", "test_mse"]].round(4).to_string(index=False))
        best = g.sort_values("val_loglik", ascending=False).iloc[0]
        print("coefficients of best:", best["coefs"])
    return r


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "MID")

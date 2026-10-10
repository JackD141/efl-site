"""Experiment: set-piece features (features.setpiece_features) in the stat models. Validation folds inside 2025/26 only.
python exp_setpieces.py [MID FWD DEF]

goals:   penalty term = recent penalty xG per 90 (live) vs penalty SHARE x league penalty rate x conversion; and
         + direct free-kick xG per 90 as an npxG feature
assists / key passes: + corners per 90, + share of his team's corners
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd

import player_model as pm

sys.path.insert(0, str(pm.am.REPO / "ml" / "fotmob"))
import features as fmf  # noqa: E402

warnings.filterwarnings("ignore")


def folds_score(x, feats, spec, pen_col="fm_pxg"):
    x = x.assign(fm_pxg=x[pen_col]) if pen_col != "fm_pxg" else x
    s25 = x[x["season"] == "2526"]
    ll = []
    for lo, hi in pm.FOLDS:
        tr, va = s25[s25["gameweek"] <= lo], s25[(s25["gameweek"] > lo) & (s25["gameweek"] <= hi)]
        f = pm.fit(tr, feats, spec)
        ll.append(pm.score(va["cnt"].to_numpy(), pm.predict(f, va, spec), f["r"])["loglik"])
    return float(np.mean(ll))


def main(positions):
    hist = fmf.setpiece_history()
    const = fmf.setpiece_constants(hist)
    rows = []
    for pos in positions:
        d = fmf.setpiece_features(pm.load(pos), hist, const)
        art = json.loads(pm.model_path(pos).read_text(encoding="utf-8"))
        for stat, a in art.items():
            if stat not in ("goals", "assists", "kp"):
                continue
            spec = pm.SPECS[pos][stat]
            x, _, _ = pm.build(d, spec, a["window"], a["k_role"], pm.league_adjustments(pos)[stat])
            base = folds_score(x, a["features"], spec)
            cands = {}
            if stat == "goals":
                cands["penalty share term"] = (a["features"], "fm_penrate")
                cands["+ free-kick xG"] = (a["features"] + ["fm_fkxg"], "fm_pxg")
                cands["penalty share + free-kick xG"] = (a["features"] + ["fm_fkxg"], "fm_penrate")
            else:
                cands["+ corners per 90"] = (a["features"] + ["fm_corners"], "fm_pxg")
                cands["+ corner share"] = (a["features"] + ["fm_cornershare"], "fm_pxg")
            for name, (feats, pen) in cands.items():
                v = folds_score(x, feats, spec, pen)
                rows.append(dict(pos=pos, stat=stat, candidate=name, val=v, live=base, gain=v - base))
                print(f"{pos} {stat}: {name}: {v:.5f} vs live {base:.5f} (gain {v - base:+.5f})", flush=True)
    r = pd.DataFrame(rows)
    (pm.HERE / "models" / "exp_setpieces_results.json").write_text(r.to_json(orient="records", indent=1), encoding="utf-8")


if __name__ == "__main__":
    main([p.upper() for p in sys.argv[1:]] or ["MID", "FWD", "DEF"])

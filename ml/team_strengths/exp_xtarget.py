"""Experiment: model the EXPECTED stat (match npxG, xA from FotMob) instead of the noisy count, then turn it into goals /
assists. Compared on actual goals / assists (negative-binomial log-likelihood, MSE) with:
  current   - the live model (count target, role / style / odds)
  +signals  - count target plus FotMob xG history features (best from exp_xg.py)
  x-target  - target = match npxG (or xA) per 90 from role / club style / opponent style (all computed on the x stat), odds,
              and his FotMob history rate; predicted goals = c * xG + his penalty xG rate (c fitted on training)
Protocol as always: fit 2025/26 GW1-23, choose on GW24-35, test once on 2026/27 (fitted on all 2025/26).
python exp_xtarget.py
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

import attack_model as am
import minutes_model as mm

sys.path.insert(0, str(am.REPO / "ml" / "fotmob"))
import features as fmf  # noqa: E402
import link_fotmob  # noqa: E402
from saves_model import nb_loglik, nb_r  # noqa: E402

warnings.filterwarnings("ignore")
KEY = ["season", "player_id", "squad_id", "opponent_id", "is_home"]
BASE = ["role", "team", "opp", "lam_own", "lam_opp", "home"]
WINDOWS, KS = [12, 46], [3.0, 10.0, 30.0]


def load():
    d = am.load("MID").drop(columns=["hid", "aid"])
    d = fmf.signal_rates(mm.attach_dates(d), k=3.0)
    fm = link_fotmob.fotmob_for_efl()[KEY + ["fm_minutes", "npxg", "xg", "xa"]].rename(columns={"npxg": "m_npxg", "xg": "m_xg", "xa": "m_xa"})
    d = d.merge(fm, on=KEY, how="left")
    d["linked"] = d["fm_minutes"].notna()
    for c in ("m_npxg", "m_xg", "m_xa"):
        d[c] = d[c].where(~d["linked"], d[c].fillna(0))
    return d[d["linked"]].reset_index(drop=True)


def fit_rate(x, feats, ycol):
    X, sc = am.design(x, feats)
    m = PoissonRegressor(alpha=1e-4, max_iter=3000).fit(X, x[ycol] / x["t"], sample_weight=x["t"])
    return dict(model=m, scaler=sc, feats=feats)


def rate(f, x):
    return f["model"].predict(am.design(x, f["feats"], f["scaler"])[0])


def score(y, mu, r):
    mu = np.clip(mu, 1e-6, None)
    return dict(loglik=nb_loglik(y, mu, r), mse=float(((y - mu) ** 2).mean()), mean_pred=float(mu.mean()), mean=float(y.mean()))


def run_stat(d, stat, count_col, x_col, pen):
    art = json.loads(am.MODEL_PATH.read_text(encoding="utf-8"))["positions"]["MID"][stat]
    out = []
    # count-target references, on the same (linked) rows
    xc = am.build(d, count_col, art["window"], art["prior"], art["k_role"], art.get("club_w", 1.0))
    xc = xc[xc["minutes_played"] >= am.MIN_ROW]
    sig = BASE + (["fm_npxg", "fm_pxg", "fm_shots"] if stat == "goals" else ["fm_xa"])
    if stat == "assists":
        sig = ["role", "fm_xa"]
    splits = lambda x: (x[(x["season"] == "2526") & (x["gameweek"] <= 23)], x[(x["season"] == "2526") & (x["gameweek"] > 23)],
                        x[x["season"] == "2526"], x[x["season"] == "2627"])
    tr, va, tv, te = splits(xc)
    for name, feats in (("current", art["features"]), ("+signals", sig)):
        f, f2 = am.fit(tr, feats), am.fit(tv, feats)
        out.append(dict(model=name, **{f"val_{k}": v for k, v in score(va["y"].to_numpy(), am.predict(f, va), f["r"]).items()},
                        **{f"test_{k}": v for k, v in score(te["y"].to_numpy(), am.predict(f2, te), f2["r"]).items()}))
    # x-target models
    tr_m90 = tr["minutes_played"].sum() / 90
    for n in WINDOWS:
        for k in KS:
            prior = float(tr[x_col].sum() / tr_m90)
            xx = am.build(d, x_col, n, prior, k)
            xx = xx[xx["minutes_played"] >= am.MIN_ROW]
            xx["xy"] = xx[x_col]
            xx["goals_y"] = xx[count_col]
            a, b, ab, c = splits(xx)
            for fs_name, feats in (("x: role/style/odds", BASE), ("x: + history rate", BASE + ([f"fm_npxg"] if stat == "goals" else ["fm_xa"]))):
                res = {}
                for tag, fit_on, ev in (("val", a, b), ("test", ab, c)):
                    f = fit_rate(fit_on, feats, "xy")
                    pen_tr = fit_on["fm_pxg"] * fit_on["t"] if pen else 0
                    base_tr = rate(f, fit_on) * fit_on["t"]
                    scale = float((fit_on["goals_y"].sum() - (pen_tr.sum() if pen else 0)) / base_tr.sum())  # finishing / conversion factor
                    mu_tr = scale * base_tr + pen_tr
                    r = nb_r(fit_on["goals_y"].to_numpy(), np.asarray(mu_tr))
                    mu = scale * rate(f, ev) * ev["t"] + (ev["fm_pxg"] * ev["t"] if pen else 0)
                    res.update({f"{tag}_{kk}": v for kk, v in score(ev["goals_y"].to_numpy(), np.asarray(mu), r).items()})
                    res[f"{tag}_scale"] = scale
                out.append(dict(model=f"{fs_name} (window {n}, k {k:g})", **res))
    r = pd.DataFrame(out)
    print(f"\n=== MID {stat}: actual {count_col}; rows linked to FotMob ===")
    cols = ["model", "val_loglik", "val_mse", "test_loglik", "test_mse", "val_scale"]
    print(r.sort_values("val_loglik", ascending=False)[[c for c in cols if c in r]].round(4).head(8).to_string(index=False))
    return r


def main():
    d = load()
    print("linked MID rows:", len(d))
    run_stat(d, "goals", "goals_scored", "m_npxg", pen=True)
    run_stat(d, "assists", "assists", "m_xa", pen=False)


if __name__ == "__main__":
    main()

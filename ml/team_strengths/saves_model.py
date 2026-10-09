"""Team-level goalkeeper saves model.  python saves_model.py   (trains, evaluates, writes models/saves_model.json)

Target: saves by a team's keeper(s) in a match = opponent shots on target - goals conceded (football-data.co.uk), which
matches the fantasy saves in 82.5% of games exactly and 99.1% within one (see docs/goalkeeper-model-plan.md).

Features (all known before kick-off; rolling ones use only earlier matches of the same season):
  lam_opp, lam_own   expected goals for the opponent / the team, implied by the closing odds (market view)
  home               1 if the team is at home
  opp_sot_for        opponent's recent shots on target per game (attacking shot volume)
  sot_against        team's recent shots on target conceded per game (defensive shot suppression)
  save_rate          team's recent saves / shots on target faced (keeper + defence quality)
Rolling features are means over the last N games, shrunk towards the league average with K pseudo-games so the first
weeks of a season are sensible. N is chosen on the validation season only.

Protocol: train 2023/24 + 2024/25, choose the model and N on 2025/26 (validation; simplest model within TOL of the best),
report 2026/27 to date once (test),
then refit the chosen model on everything and save it. Models are Poisson GLMs (log link) on log/standardised features,
with a negative-binomial spread estimated from training residuals; a gradient-boosted model is a comparison.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import nbinom
from sklearn.linear_model import PoissonRegressor

from per_season import lam_for
from strengths import DIVS

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
MODEL_PATH = HERE / "models" / "saves_model.json"
TRAIN, VALID, TEST = ["2324", "2425"], ["2526"], ["2627"]
WINDOWS = [3, 6, 10, 46]  # 46 = season to date
K = 3  # pseudo-games of league average mixed into every rolling mean
TOL = 5e-4  # validation log-likelihood differences below this are treated as ties (prefer the simpler model)
ROLL_COLS = ["opp_sot_for", "sot_against", "save_rate"]
FEATURE_SETS = {
    "M1 odds only": ["lam_opp"],
    "M2 odds + home": ["lam_opp", "lam_own", "home"],
    "M3 + rolling shots": ["lam_opp", "lam_own", "home", "opp_sot_for", "sot_against"],
    "M4 + rolling save rate": ["lam_opp", "lam_own", "home", "opp_sot_for", "sot_against", "save_rate"],
}


def team_games(seasons):
    """One row per team per match (the team whose keeper makes the saves)."""
    rows = []
    for s in seasons:
        for div in DIVS:
            d = lam_for(div, s).dropna(subset=["lamH", "HST", "AST"])
            for side, opp_side in (("H", "A"), ("A", "H")):
                t = pd.DataFrame({
                    "season": s, "div": div, "date": d["Date"].to_numpy(),
                    "team": d["HomeTeam" if side == "H" else "AwayTeam"].to_numpy(),
                    "opp": d["AwayTeam" if side == "H" else "HomeTeam"].to_numpy(),
                    "home": int(side == "H"),
                    "lam_own": d["lamH" if side == "H" else "lamA"].to_numpy(),
                    "lam_opp": d["lamA" if side == "H" else "lamH"].to_numpy(),
                    "sot_for": d[f"{side}ST"].to_numpy(), "sot_ag": d[f"{opp_side}ST"].to_numpy(),
                    "gc": d["FTAG" if side == "H" else "FTHG"].to_numpy(),
                })
                rows.append(t)
    g = pd.concat(rows, ignore_index=True)
    g["saves"] = (g["sot_ag"] - g["gc"]).clip(lower=0)
    return g.sort_values(["season", "div", "date"]).reset_index(drop=True)


def shrunk_roll(series, n, prior, k=K):
    """Mean of the previous n values (strictly before this match), shrunk towards prior with k pseudo-values."""
    s = series.shift(1)
    tot = s.rolling(n, min_periods=0).sum()
    cnt = s.rolling(n, min_periods=0).count()
    return (tot + k * prior) / (cnt + k)


def add_rolling(g, n, priors):
    g = g.copy()
    grp = g.groupby(["season", "team"], sort=False)
    g["sot_against"] = grp["sot_ag"].transform(lambda x: shrunk_roll(x, n, priors["sot"]))
    # save rate as a ratio of rolling sums (shrunk), not a mean of ratios
    sv = grp["saves"].transform(lambda x: x.shift(1).rolling(n, min_periods=0).sum())
    sa = grp["sot_ag"].transform(lambda x: x.shift(1).rolling(n, min_periods=0).sum())
    g["save_rate"] = (sv + K * priors["sot"] * priors["save_rate"]) / (sa + K * priors["sot"])
    # opponent's attacking shot volume: the opponent's own rolling SOT-for, as of before this match
    own_for = grp["sot_for"].transform(lambda x: shrunk_roll(x, n, priors["sot"]))
    key = g[["season", "team", "date"]].assign(opp_sot_for=own_for.to_numpy())
    g = g.merge(key.rename(columns={"team": "opp"}), on=["season", "opp", "date"], how="left")
    g["opp_sot_for"] = g["opp_sot_for"].fillna(priors["sot"])
    return g


def design(g, feats, scaler=None):
    X = np.column_stack([np.log(g[f].clip(lower=1e-3)) if f != "home" else g[f].to_numpy() for f in feats]).astype(float)
    if scaler is None:
        scaler = (X.mean(0), X.std(0) + 1e-9)
    return (X - scaler[0]) / scaler[1], scaler


def nb_r(y, mu):
    """Negative-binomial size from Var = mu + mu^2/r (method of moments)."""
    num = (mu ** 2).sum()
    den = ((y - mu) ** 2 - mu).sum()
    return float(np.clip(num / den, 1.0, 1e4)) if den > 0 else 1e4


def nb_loglik(y, mu, r):
    p = r / (r + mu)
    return float(np.mean(gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r * np.log(p) + y * np.log1p(-p)))


def poisson_dev(y, mu):
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(np.mean(2 * (t - (y - mu))))


def save_points(mu, r, kmax=40):
    k = np.arange(kmax)
    pmf = nbinom.pmf(k[None, :], r, (r / (r + np.asarray(mu)))[:, None])
    return (pmf * (2 * (k // 3))[None, :]).sum(1)


def evaluate(y, mu, r):
    return dict(poisson_dev=poisson_dev(y, mu), nb_loglik=nb_loglik(y, mu, r),
                mae_saves=float(np.abs(y - mu).mean()), mae_save_pts=float(np.abs(2 * (y // 3) - save_points(mu, r)).mean()),
                mean_pred=float(mu.mean()), mean_actual=float(y.mean()))


def fit_glm(g, feats, alpha=1e-4):
    X, sc = design(g, feats)
    m = PoissonRegressor(alpha=alpha, max_iter=1000).fit(X, g["saves"])
    mu = m.predict(X)
    return dict(model=m, scaler=sc, feats=feats, r=nb_r(g["saves"].to_numpy(), mu))


def predict_glm(fit, g):
    X, _ = design(g, fit["feats"], fit["scaler"])
    return fit["model"].predict(X)


def main():
    raw_train, raw_valid, raw_test = team_games(TRAIN), team_games(VALID), team_games(TEST)
    priors = dict(sot=float(raw_train["sot_ag"].mean()), save_rate=float(raw_train["saves"].sum() / raw_train["sot_ag"].sum()))
    print(f"team-games: train {len(raw_train)}, validation {len(raw_valid)}, test {len(raw_test)}; priors {priors}")

    results, fits = [], {}
    yv = raw_valid["saves"].to_numpy()
    base_mu = np.full(len(yv), raw_train["saves"].mean())
    base_r = nb_r(raw_train["saves"].to_numpy(), np.full(len(raw_train), raw_train["saves"].mean()))
    results.append(dict(model="M0 league average", window="-", **evaluate(yv, base_mu, base_r)))
    for n in WINDOWS:
        tr, va = add_rolling(raw_train, n, priors), add_rolling(raw_valid, n, priors)
        for name, feats in FEATURE_SETS.items():
            if n != WINDOWS[0] and not any(f in ROLL_COLS for f in feats):
                continue  # window does not affect these
            f = fit_glm(tr, feats)
            fits[(name, n)] = f
            results.append(dict(model=name, window=n if any(c in ROLL_COLS for c in feats) else "-", **evaluate(yv, predict_glm(f, va), f["r"])))
        try:  # gradient boosting comparison on the richest features
            from xgboost import XGBRegressor
            feats = FEATURE_SETS["M4 + rolling save rate"]
            xgb = XGBRegressor(objective="count:poisson", n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                               colsample_bytree=0.8, min_child_weight=20, random_state=42)
            xgb.fit(tr[feats], tr["saves"])
            mu_tr = xgb.predict(tr[feats])
            results.append(dict(model="XGBoost (M4 features)", window=n, **evaluate(yv, xgb.predict(va[feats]), nb_r(tr["saves"].to_numpy(), mu_tr))))
        except ImportError:
            pass
    res = pd.DataFrame(results)
    pd.set_option("display.width", 220)
    print("\nVALIDATION 2025/26 (lower deviance / higher log-lik / lower MAE is better):")
    print(res.round(4).to_string(index=False))

    # pick the simplest model within TOL of the best validation log-likelihood (differences smaller than that are noise)
    glm = res[res["model"].isin(FEATURE_SETS)].copy()
    glm["complexity"] = glm["model"].map(list(FEATURE_SETS).index)
    best = glm[glm["nb_loglik"] >= glm["nb_loglik"].max() - TOL].sort_values(["complexity", "nb_loglik"], ascending=[True, False]).iloc[0]
    n_best = WINDOWS[0] if best["window"] == "-" else int(best["window"])
    print(f"\nchosen on validation: {best['model']} (window {best['window']})")

    # test once: refit on train + validation, score 2026/27 to date
    trva_raw = pd.concat([raw_train, raw_valid], ignore_index=True)
    trva, te = add_rolling(trva_raw, n_best, priors), add_rolling(raw_test, n_best, priors)
    f_tv = fit_glm(trva, FEATURE_SETS[best["model"]])
    yt = te["saves"].to_numpy()
    test = evaluate(yt, predict_glm(f_tv, te), f_tv["r"])
    test_base = evaluate(yt, np.full(len(yt), trva_raw["saves"].mean()), nb_r(trva_raw["saves"].to_numpy(), np.full(len(trva_raw), trva_raw["saves"].mean())))
    print(f"TEST 2026/27 to date ({len(yt)} team-games): model {json.dumps({k: round(v, 4) for k, v in test.items()})}")
    print(f"                              baseline {json.dumps({k: round(v, 4) for k, v in test_base.items()})}")
    te = te.assign(mu=predict_glm(f_tv, te))
    te["bucket"] = pd.qcut(te["mu"], 5, labels=False, duplicates="drop")
    cal = te.groupby("bucket").agg(pred=("mu", "mean"), actual=("saves", "mean"), n=("saves", "size"))
    print("test calibration by predicted quintile:\n" + cal.round(2).to_string())

    # production fit on everything
    all_raw = pd.concat([raw_train, raw_valid, raw_test], ignore_index=True)
    allg = add_rolling(all_raw, n_best, priors)
    final = fit_glm(allg, FEATURE_SETS[best["model"]])
    coefs = dict(zip(final["feats"], final["model"].coef_.tolist()))
    print(f"\nfinal fit on {len(allg)} team-games; coefficients on standardised log-features: {json.dumps({k: round(v, 3) for k, v in coefs.items()})}; NB r {final['r']:.1f}")
    MODEL_PATH.parent.mkdir(exist_ok=True)
    MODEL_PATH.write_text(json.dumps(dict(
        model=best["model"], features=final["feats"], window=n_best, k=K, priors=priors,
        intercept=float(final["model"].intercept_), coef=final["model"].coef_.tolist(),
        scaler_mean=final["scaler"][0].tolist(), scaler_sd=final["scaler"][1].tolist(), nb_r=final["r"],
        validation=res.round(5).to_dict(orient="records"), test=dict(model=test, baseline=test_base, n=int(len(yt))),
        trained_on=TRAIN + VALID + TEST,
    ), indent=1), encoding="utf-8")
    print(f"saved {MODEL_PATH}")


if __name__ == "__main__":
    main()

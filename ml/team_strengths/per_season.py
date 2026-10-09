"""Per-season team strengths fitted to bookmaker odds (and to results), one model per league.

Step 1: invert each match's closing odds (1X2 + over/under 2.5) into expected goals (lam_home, lam_away).
Step 2: per season, fit log(lam) = base + home*[home side] + attack[team] - defence_conceded[opp]  by ridge least squares.
        -> one attack + one defence number per team per season, plus league hyperparameters.
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import poisson

MAXG = 10
G = np.arange(MAXG + 1)
HG, AG = np.meshgrid(G, G, indexing="ij")


def tau_matrix(lh, la, rho):
    T = np.ones((MAXG + 1, MAXG + 1))
    T[0, 0] = 1 - lh * la * rho
    T[0, 1] = 1 + lh * rho
    T[1, 0] = 1 + la * rho
    T[1, 1] = 1 - rho
    return T


def score_matrix(lh, la, rho=0.0):
    P = np.outer(poisson.pmf(G, lh), poisson.pmf(G, la))
    if rho:
        P = P * tau_matrix(lh, la, rho)
    return P / P.sum()


def probs_from_lams(lh, la, rho=0.0):
    P = score_matrix(lh, la, rho)
    return np.array([P[HG > AG].sum(), P[HG == AG].sum(), P[HG < AG].sum(), P[HG + AG > 2].sum()])


def devig(*odds):
    p = 1 / np.array(odds, dtype=float)
    return p / p.sum()


def market_targets(row, prefix="AvgC"):
    """[pH, pD, pA, pOver2.5] de-vigged from average odds (closing columns by default; prefix="Avg" for upcoming fixtures)."""
    try:
        h = devig(row[f"{prefix}H"], row[f"{prefix}D"], row[f"{prefix}A"])
        o = devig(row[f"{prefix}>2.5"], row[f"{prefix}<2.5"])
    except Exception:
        return None
    t = np.array([*h, o[0]])
    return None if np.isnan(t).any() else t


def invert(target, rho):
    f = lambda x: probs_from_lams(np.exp(x[0]), np.exp(x[1]), rho) - target
    r = least_squares(f, x0=np.log([1.4, 1.1]), xtol=1e-8, ftol=1e-8)
    return np.exp(r.x), float(np.sqrt((r.fun ** 2).mean()))


def implied_lambdas(df, rho, prefix="AvgC"):
    out = []
    for _, row in df.iterrows():
        t = market_targets(row, prefix)
        if t is None:
            out.append((np.nan, np.nan, np.nan))
            continue
        lam, err = invert(t, rho)
        out.append((lam[0], lam[1], err))
    return pd.DataFrame(out, columns=["lamH", "lamA", "inv_err"], index=df.index)


def fit_strengths(df, lamH, lamA, ridge=0.5, weights=None, prior=None, prior_mult=0.0, teams=None, team_home_ridge=None):
    """Ridge least squares on log-lambda, optionally shrinking towards prior_mult * last season's ratings.

    prior: DataFrame with columns team, attack, conceded (last season's fit). Teams not in it shrink to 0.
    """
    teams = teams or sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    idx = {t: i for i, t in enumerate(teams)}
    n, k = len(teams), len(df)
    extra = n if team_home_ridge is not None else 0
    X = np.zeros((2 * k, 2 * n + 2 + extra))  # [att(n), dfn(n), home, base, (team home boost(n))]
    hi = df["HomeTeam"].map(idx).to_numpy()
    ai = df["AwayTeam"].map(idx).to_numpy()
    r = np.arange(k)
    X[r, hi] = 1; X[r, n + ai] = 1; X[r, 2 * n] = 1; X[r, 2 * n + 1] = 1
    if extra:
        X[r, 2 * n + 2 + hi] = 1
    X[k + r, ai] = 1; X[k + r, n + hi] = 1; X[k + r, 2 * n + 1] = 1
    y = np.log(np.concatenate([lamH, lamA]))
    w = np.ones(2 * k) if weights is None else np.concatenate([weights, weights])
    pen = np.zeros(2 * n + 2 + extra)
    pen[: 2 * n] = ridge
    if extra:
        pen[2 * n + 2:] = team_home_ridge
    c0 = np.zeros(2 * n + 2 + extra)
    if prior is not None and prior_mult:
        p = prior.set_index("team")
        c0[:n] = [prior_mult * p["attack"].get(t, 0.0) for t in teams]
        c0[n:2 * n] = [prior_mult * p["conceded"].get(t, 0.0) for t in teams]
    A = X.T @ (X * w[:, None]) + np.diag(pen)
    b = X.T @ (w * y) + pen * c0
    c = np.linalg.solve(A, b)
    pred = X @ c
    ss_res = (w * (y - pred) ** 2).sum()
    ss_tot = (w * (y - np.average(y, weights=w)) ** 2).sum()
    res = pd.DataFrame({"team": teams, "attack": c[:n], "conceded": c[n:2 * n]})
    res["defence"] = -res["conceded"]  # higher = better defence
    return dict(home=c[2 * n], base=c[2 * n + 1], r2=1 - ss_res / ss_tot, rmse=float(np.sqrt(ss_res / w.sum())),
                teams=res, coef=c, idx=idx)


def predict_probs(fit, home, away, rho=0.0):
    c, idx, n = fit["coef"], fit["idx"], len(fit["idx"])
    def g(team, off):
        return c[off + idx[team]] if team in idx else 0.0
    boost = c[2 * n + 2 + idx[home]] if len(c) > 2 * n + 2 and home in idx else 0.0
    lh = np.exp(fit["base"] + fit["home"] + boost + g(home, 0) + g(away, n))
    la = np.exp(fit["base"] + g(away, 0) + g(home, n))
    return probs_from_lams(lh, la, rho), lh, la


# ---- cached implied-lambda loader -------------------------------------------------------
from pathlib import Path  # noqa: E402

from strengths import load_div  # noqa: E402

CACHE = Path(__file__).parent / "cache"
CACHE.mkdir(exist_ok=True)
RHO = -0.06


def lam_for(div, season):
    """Season matches joined with odds-implied expected goals (cached to disk)."""
    f = CACHE / f"lam_{season}_{div}.csv"
    df = load_div(div)
    df = df[df["season"] == season]
    if f.exists() and len(pd.read_csv(f, index_col=0)) == len(df):
        lam = pd.read_csv(f, index_col=0)
    else:
        lam = implied_lambdas(df, RHO)
        lam.to_csv(f)
    return df.join(lam)

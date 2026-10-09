"""Per-league team strength model (Poisson attack/defence ratings, Dixon-Coles low-score fix).

One model per division (E1 Championship, E2 League 1, E3 League 2), fitted on football-data.co.uk
results with exponential time-decay and ridge shrinkage.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

DATA = Path(__file__).resolve().parents[2] / "data" / "football_data"
SEASONS = ["2324", "2425", "2526", "2627"]
DIVS = {"E1": "Championship", "E2": "League 1", "E3": "League 2"}
MAXG = 10


def load_div(div):
    frames = []
    for s in SEASONS:
        f = DATA / f"{s}_{div}.csv"
        d = pd.read_csv(f, encoding="utf-8-sig")
        d["season"] = s
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.dropna(subset=["FTHG", "FTAG", "HomeTeam", "AwayTeam"]).copy()
    df[["FTHG", "FTAG"]] = df[["FTHG", "FTAG"]].astype(int)
    return df.sort_values("Date").reset_index(drop=True)


def _design(m, idx):
    n, k = len(idx), len(m)
    hi = m["HomeTeam"].map(idx).to_numpy()
    ai = m["AwayTeam"].map(idx).to_numpy()
    X = np.zeros((2 * k, 2 * n + 1))
    r = np.arange(k)
    X[r, hi] = 1
    X[r, n + ai] = 1
    X[r, 2 * n] = 1
    X[k + r, ai] = 1
    X[k + r, n + hi] = 1
    y = np.concatenate([m["FTHG"].to_numpy(), m["FTAG"].to_numpy()])
    return X, y


class Model:
    def __init__(self, half_life=240, alpha=1e-3, dc=True):
        self.half_life, self.alpha, self.dc = half_life, alpha, dc

    def fit(self, matches, as_of):
        m = matches[matches["Date"] < as_of]
        self.teams = sorted(set(m["HomeTeam"]) | set(m["AwayTeam"]))
        idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)
        X, y = _design(m, idx)
        age = (as_of - m["Date"]).dt.days.to_numpy()
        w = np.tile(0.5 ** (age / self.half_life), 2)
        glm = PoissonRegressor(alpha=self.alpha, max_iter=500)
        glm.fit(X, y, sample_weight=w)
        c = glm.coef_
        self.att = dict(zip(self.teams, c[:n]))
        self.dfn = dict(zip(self.teams, c[n:2 * n]))
        self.home, self.b0 = c[2 * n], glm.intercept_
        self.rho = self._fit_rho(m, w[: len(m)]) if self.dc else 0.0
        return self

    def lambdas(self, home, away):
        lh = np.exp(self.b0 + self.home + self.att.get(home, 0) + self.dfn.get(away, 0))
        la = np.exp(self.b0 + self.att.get(away, 0) + self.dfn.get(home, 0))
        return lh, la

    @staticmethod
    def _tau(x, y, lh, la, rho):
        t = np.ones_like(lh, dtype=float)
        t = np.where((x == 0) & (y == 0), 1 - lh * la * rho, t)
        t = np.where((x == 0) & (y == 1), 1 + lh * rho, t)
        t = np.where((x == 1) & (y == 0), 1 + la * rho, t)
        t = np.where((x == 1) & (y == 1), 1 - rho, t)
        return t

    def _fit_rho(self, m, w):
        lh = np.array([self.lambdas(h, a)[0] for h, a in zip(m["HomeTeam"], m["AwayTeam"])])
        la = np.array([self.lambdas(h, a)[1] for h, a in zip(m["HomeTeam"], m["AwayTeam"])])
        x, y = m["FTHG"].to_numpy(), m["FTAG"].to_numpy()
        best, best_ll = 0.0, -np.inf
        for rho in np.linspace(-0.25, 0.15, 41):
            t = self._tau(x, y, lh, la, rho)
            if (t <= 0).any():
                continue
            ll = (w * np.log(t)).sum()
            if ll > best_ll:
                best, best_ll = rho, ll
        return best

    def score_matrix(self, home, away):
        lh, la = self.lambdas(home, away)
        g = np.arange(MAXG + 1)
        P = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
        if self.dc and self.rho:
            for x, y in ((0, 0), (0, 1), (1, 0), (1, 1)):
                P[x, y] *= float(self._tau(np.array(x), np.array(y), np.array(lh), np.array(la), self.rho))
        return P / P.sum(), lh, la

    def predict(self, home, away):
        P, lh, la = self.score_matrix(home, away)
        g = np.arange(MAXG + 1)
        hg, ag = np.meshgrid(g, g, indexing="ij")
        return dict(
            pH=P[hg > ag].sum(), pD=P[hg == ag].sum(), pA=P[hg < ag].sum(),
            xgH=lh, xgA=la,
            csH=P[:, 0].sum(),  # home team keeps a clean sheet (away scores 0)
            csA=P[0, :].sum(),
            pO25=P[hg + ag > 2].sum(),
        )

    def ratings(self):
        r = pd.DataFrame({"team": self.teams, "attack": [self.att[t] for t in self.teams],
                          "defence": [-self.dfn[t] for t in self.teams]})
        return r.sort_values("attack", ascending=False).reset_index(drop=True)


def market_probs(df, prefix="AvgC"):
    o = df[[f"{prefix}H", f"{prefix}D", f"{prefix}A"]].astype(float).to_numpy()
    p = 1 / o
    return p / p.sum(axis=1, keepdims=True)


def outcome(df):
    return np.where(df["FTHG"] > df["FTAG"], 0, np.where(df["FTHG"] == df["FTAG"], 1, 2))


def log_loss(p, y):
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1))))


def rps(p, y):
    cum_p = np.cumsum(p, axis=1)[:, :2]
    cum_o = np.cumsum(np.eye(3)[y], axis=1)[:, :2]
    return float(np.mean(((cum_p - cum_o) ** 2).sum(axis=1) / 2))


def walk_forward(df, season, half_life, alpha, dc=True, step_days=7):
    """Predict every match in `season`, refitting each week on strictly earlier matches."""
    test = df[df["season"] == season]
    start = test["Date"].min()
    rows = []
    edge = start
    while edge <= test["Date"].max():
        chunk = test[(test["Date"] >= edge) & (test["Date"] < edge + pd.Timedelta(days=step_days))]
        if len(chunk):
            mdl = Model(half_life, alpha, dc).fit(df, edge)
            for i, r in chunk.iterrows():
                p = mdl.predict(r["HomeTeam"], r["AwayTeam"])
                p["i"] = i
                rows.append(p)
        edge += pd.Timedelta(days=step_days)
    out = pd.DataFrame(rows).set_index("i")
    return test.join(out, how="inner")

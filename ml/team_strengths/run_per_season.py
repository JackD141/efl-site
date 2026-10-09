"""How well do per-season team strengths explain the closing odds?  (3 seasons x 3 leagues)"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from per_season import CACHE, RHO, fit_strengths, lam_for, market_targets, predict_probs
from strengths import DIVS, Model, load_div, log_loss, outcome

warnings.filterwarnings("ignore")
SEASONS = ["2324", "2425", "2526"]


def one(div, season):
    d = lam_for(div, season)
    d = d.dropna(subset=["lamH"])
    y = outcome(d)
    M = np.array([market_targets(r)[:3] for _, r in d.iterrows()])
    fit = fit_strengths(d, d["lamH"].to_numpy(), d["lamA"].to_numpy(), ridge=0.1)
    P = np.array([predict_probs(fit, h, a, RHO)[0][:3] for h, a in zip(d["HomeTeam"], d["AwayTeam"])])
    # goals-based fit for comparison (same structure, results only)
    g = Model(half_life=1e9, alpha=1e-3, dc=False).fit(d, d["Date"].max() + pd.Timedelta(days=1))
    gr = g.ratings().set_index("team")
    tt = fit["teams"].set_index("team")
    j = tt.join(gr, rsuffix="_goals")
    out = dict(
        div=div, season=season, n=len(d), inv_err=d["inv_err"].mean(),
        r2_loglam=fit["r2"], rmse_loglam=fit["rmse"],
        mae_pH=np.abs(P[:, 0] - M[:, 0]).mean(), mae_pD=np.abs(P[:, 1] - M[:, 1]).mean(), mae_pA=np.abs(P[:, 2] - M[:, 2]).mean(),
        ll_market=log_loss(M, y), ll_strengths_from_odds=log_loss(P, y),
        home_x=np.exp(fit["home"]), base_goals=np.exp(fit["base"]),
        sd_att=tt["attack"].std(), sd_def=tt["defence"].std(), corr_att_def=np.corrcoef(tt["attack"], tt["defence"])[0, 1],
        corr_odds_vs_goals_att=np.corrcoef(j["attack"], j["attack_goals"])[0, 1],
        corr_odds_vs_goals_def=np.corrcoef(j["defence"], j["defence_goals"])[0, 1],
    )
    return out, tt.reset_index().assign(div=div, season=season)


jobs = [(d, s) for d in DIVS for s in SEASONS]
res = Parallel(n_jobs=-1)(delayed(one)(*j) for j in jobs)
summ = pd.DataFrame([r[0] for r in res])
teams = pd.concat([r[1] for r in res], ignore_index=True)
teams.to_csv(CACHE / "season_ratings_from_odds.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 30)
print(summ.round(4).to_string(index=False))

# year-to-year persistence of odds-implied ratings (same league both years)
print("\nYear-to-year correlation of team ratings (teams in same league both seasons):")
for d in DIVS:
    for s0, s1 in zip(SEASONS[:-1], SEASONS[1:]):
        a = teams[(teams["div"] == d) & (teams["season"] == s0)].set_index("team")
        b = teams[(teams["div"] == d) & (teams["season"] == s1)].set_index("team")
        c = a.join(b, lsuffix="0", rsuffix="1", how="inner")
        print(f"  {d} {s0}->{s1} n={len(c):2d}  attack r={np.corrcoef(c.attack0,c.attack1)[0,1]:.2f}  defence r={np.corrcoef(c.defence0,c.defence1)[0,1]:.2f}")

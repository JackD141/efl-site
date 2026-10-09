"""Are 1X2 + over/under 2.5 odds enough to price the Fantasy EFL club bonuses?

For every match in 2023/24-2025/26 (E1-E3): invert the closing odds to expected goals, derive P(clean sheet),
P(2+ goals), P(4+ goals), P(win), P(draw) and expected club points, then compare with what actually happened.
Also compares how sharp each bookmaker's closing odds are.   usage: python check_markets.py
"""
import warnings

import numpy as np
import pandas as pd

from club_points import expected_points, side_probs
from per_season import RHO, devig, lam_for, score_matrix
from strengths import DIVS

warnings.filterwarnings("ignore")
SEASONS = ["2324", "2425", "2526"]


def actual_points(gf, ga, home):
    win = gf > ga
    return 5 * win + 3 * (gf == ga) + (2 * win if not home else 0) + 2 * (ga == 0) + 2 * (gf >= 2) + 2 * (gf >= 4)


def main():
    rows = []
    for div in DIVS:
        for s in SEASONS:
            d = lam_for(div, s).dropna(subset=["lamH"])
            for _, r in d.iterrows():
                M = score_matrix(r["lamH"], r["lamA"], RHO)
                for home in (True, False):
                    gf, ga = (r["FTHG"], r["FTAG"]) if home else (r["FTAG"], r["FTHG"])
                    e, p = expected_points(M, home)
                    rows.append(dict(div=div, season=s, home=home, exp_pts=e, act_pts=actual_points(gf, ga, home),
                                     p_win=p["win"], a_win=float(gf > ga), p_draw=p["draw"], a_draw=float(gf == ga),
                                     p_cs=p["clean_sheet"], a_cs=float(ga == 0), p_g2=p["two_goals"], a_g2=float(gf >= 2),
                                     p_g4=p["four_goals"], a_g4=float(gf >= 4)))
    cg = pd.DataFrame(rows)
    print(f"{len(cg)} club-games, {len(cg) // 2} matches\n")
    print("Calibration (mean predicted vs actual frequency), all club-games:")
    for k, n in (("win", "win"), ("draw", "draw"), ("cs", "clean sheet"), ("g2", "2+ goals scored"), ("g4", "4+ goals scored")):
        print(f"  {n:16s} predicted {cg['p_' + k].mean():.4f}   actual {cg['a_' + k].mean():.4f}")
    print(f"  {'club points':16s} predicted {cg.exp_pts.mean():.3f}   actual {cg.act_pts.mean():.3f}")
    print("\nBy home/away:")
    print(cg.groupby("home")[["exp_pts", "act_pts", "p_cs", "a_cs", "p_g4", "a_g4"]].mean().round(4).to_string())
    print("\nExpected vs actual club points by predicted-points bucket (does it hold at the high end?):")
    cg["bucket"] = pd.qcut(cg.exp_pts, 8, duplicates="drop")
    t = cg.groupby("bucket")[["exp_pts", "act_pts"]].mean().round(3)
    t["n"] = cg.groupby("bucket").size()
    print(t.to_string())
    print("\nClean-sheet probability by bucket:")
    cg["csb"] = pd.qcut(cg.p_cs, 6, duplicates="drop")
    print(cg.groupby("csb")[["p_cs", "a_cs"]].mean().round(3).to_string())
    print("\n4+ goals: predicted vs actual by bucket of predicted probability:")
    cg["g4b"] = pd.qcut(cg.p_g4, 5, duplicates="drop")
    print(cg.groupby("g4b")[["p_g4", "a_g4"]].mean().round(4).to_string())

    # how sharp is each bookmaker's closing 1X2?
    print("\nSharpness of closing 1X2 odds (log-loss vs results, lower = sharper):")
    ll = {k: [] for k in ("Pinnacle", "Betfair Exch", "Bet365", "Market avg")}
    cols = {"Pinnacle": ("PSCH", "PSCD", "PSCA"), "Betfair Exch": ("BFECH", "BFECD", "BFECA"), "Bet365": ("B365CH", "B365CD", "B365CA"), "Market avg": ("AvgCH", "AvgCD", "AvgCA")}
    from strengths import load_div
    for div in DIVS:
        df = load_div(div)
        df = df[df["season"].isin(SEASONS)]
        y = np.where(df.FTHG > df.FTAG, 0, np.where(df.FTHG == df.FTAG, 1, 2))
        ok = np.ones(len(df), bool)
        P = {}
        for k, c in cols.items():
            if not all(x in df.columns for x in c):
                ok &= False
                continue
            p = 1 / df[list(c)].astype(float).to_numpy()
            P[k] = p / p.sum(1, keepdims=True)
            ok &= ~np.isnan(P[k]).any(1)
        for k in P:
            ll[k].append(-np.log(P[k][ok][np.arange(ok.sum()), y[ok]]))
    for k, v in ll.items():
        if v:
            print(f"  {k:14s} {np.concatenate(v).mean():.4f}  (n={len(np.concatenate(v))})")


if __name__ == "__main__":
    main()

"""Experiment: a club starts exactly ten outfield players.  python exp_clubstart.py

The start model scores each player on his own history, so a club's start chances need not add up to ten. Logic for
making them: the line-up is a fixed-size choice among the squad, so when the model's chances for a club add up to 11.5
somebody is too high, and when team-mates are known to be out (EFL flag -> 0 on the site; suspensions here) the others'
chances must rise. Candidate: per club fixture, shift every available outfield player's start logit by the same amount
so the chances add up to the number of outfield starters (10).
Protocol as minutes_model.py: fit on 2025/26 GW1-23, validate GW24-35; 2026/27 shown for information only.
Prints the calibration of the raw start chances, the distribution of the club sums, and log-loss raw vs normalised.
"""
import json
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit

import minutes_model as mm

warnings.filterwarnings("ignore")
OUTFIELD = ("DEF", "MID", "FWD")
FIX = ["season", "squad_id", "gameweek", "opponent_id"]


def shift_to(p, total):
    """p shifted on the logit scale so it sums to total (if the squad can field that many)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    if len(p) <= total:
        return np.ones(len(p))
    z = logit(p)
    return expit(z + brentq(lambda d: expit(z + d).sum() - total, -30, 30))


def ll(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def rows_with_probs():
    parts = []
    for pos in OUTFIELD:
        d = mm.add_features(mm.load_rows(pos))
        feats = mm.FEATURES + mm.FM_FEATURES
        tr = d[(d["season"] == "2526") & (d["gameweek"] <= 23)]
        art = mm.fit_start(tr, feats)
        ps, pf, pp = mm.scenario_probs(art, d)
        parts.append(d[FIX + ["player_id", "start", "mins", "fm_linked", "date"]].assign(pos=pos, ps=ps, pf=pf, pp=pp, q=np.where(ps > 0, pf / ps, 0)))
    return pd.concat(parts, ignore_index=True)


def main():
    d = rows_with_probs()
    g = d.groupby(FIX)
    d["n_start"] = g["start"].transform("sum")
    d["sum_ps"] = g["ps"].transform("sum")
    # fixtures where the fantasy list + FotMob line-up account for a normal XI (10 outfield starters)
    ok = d["n_start"] == 10
    print(f"club fixtures: {g.ngroups}; outfield starters found: {d.drop_duplicates(FIX)['n_start'].value_counts().sort_index().to_dict()}")
    d["ps_norm"] = g["ps"].transform(lambda s: shift_to(s.to_numpy(), 10))
    # normalise P(full game) with the same per-player P(60+ | start)
    d["pf_norm"] = d["ps_norm"] * d["q"]
    res = {}
    for lab, m in (("validation 2025/26 GW24-35", (d["season"] == "2526") & (d["gameweek"] > 23)), ("2026/27 (information)", d["season"] == "2627")):
        s = d[m & ok]
        fx = s.drop_duplicates(FIX)
        full = (s["mins"] >= 60).astype(float)
        r = dict(rows=int(len(s)), fixtures=int(len(fx)), sum_mean=float(fx["sum_ps"].mean()), sum_sd=float(fx["sum_ps"].std()),
                 sum_p10=float(fx["sum_ps"].quantile(0.1)), sum_p90=float(fx["sum_ps"].quantile(0.9)),
                 start_ll_raw=ll(s["start"], s["ps"]), start_ll_norm=ll(s["start"], s["ps_norm"]),
                 p60_ll_raw=ll(full, s["pf"]), p60_ll_norm=ll(full, s["pf_norm"]))
        res[lab] = r
        print(f"\n{lab}: {r['rows']} rows, {r['fixtures']} club fixtures")
        print(f"  sum of start chances per club: mean {r['sum_mean']:.2f}, sd {r['sum_sd']:.2f}, 10%-90% {r['sum_p10']:.2f}-{r['sum_p90']:.2f} (truth: 10)")
        print(f"  P(start) log-loss raw {r['start_ll_raw']:.4f} -> normalised {r['start_ll_norm']:.4f} ({r['start_ll_norm'] - r['start_ll_raw']:+.4f})")
        print(f"  P(60+)   log-loss raw {r['p60_ll_raw']:.4f} -> normalised {r['p60_ll_norm']:.4f} ({r['p60_ll_norm'] - r['p60_ll_raw']:+.4f})")
        for col in ("ps", "ps_norm"):
            b = pd.cut(s[col], [0, .05, .2, .4, .6, .75, .85, .92, .96, 1.0001], include_lowest=True)
            t = s.groupby(b, observed=True).agg(pred=(col, "mean"), actual=("start", "mean"), n=("start", "size"))
            print(f"  reliability of {col}:\n" + t.round(3).to_string())
        by = s.groupby("pos").apply(lambda x: pd.Series(dict(raw=ll(x["start"], x["ps"]), norm=ll(x["start"], x["ps_norm"]))))
        print("  by position:\n" + by.round(4).to_string())
        # does the club sum itself predict the error? (if high sums go with fewer starts than predicted, normalising is right)
        s2 = s.assign(resid=s["start"] - s["ps"], hi=pd.cut(s["sum_ps"], [0, 9.5, 10.5, 30], labels=["<9.5", "9.5-10.5", ">10.5"]))
        print("  mean (actual - predicted) by club sum:", s2.groupby("hi", observed=True)["resid"].mean().round(4).to_dict())
    (mm.MODELS / "exp_clubstart_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

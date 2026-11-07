"""Experiment: model the chance he STARTS (and comes off the bench) directly, instead of expected minutes.
python exp_start.py

Same protocol as minutes_model.py: train 2025/26 GW1-23, validate GW24-35, choose on validation only; 2026/27 is reported
once for information. All candidates are scored on the same rows: games he was available for (inferred suspensions are
removed, because at pick time a ban is known and the site sets him to 0 anyway).

Candidates (deliberately few):
  current   ridge expected minutes -> isotonic P(60+) (live since 9 Oct)
  logit     logistic P(start) and P(sub appearance | not start) on the same features; P(60+) = P(start) x q, q = his
            position's share of starts reaching 60 (training data)
  +run      logit plus his current run of consecutive starts
  +susp     logit +run with suspended games also removed from his HISTORY (a ban does not count as being dropped)
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

import minutes_model as mm

warnings.filterwarnings("ignore")
FEATS = mm.FEATURES + mm.FM_FEATURES


def suspended(d):
    """Games a player most likely missed through suspension: 0 minutes in the club's next 1-3 games after a red card
    (stopping at his first appearance), or the next game after his 5th / 10th / 15th league yellow of the season
    (next 2 after the 10th). d sorted by player, club, date."""
    d = d.copy()
    d["ycum"] = d.groupby(["player_id", "season"])["yellow_cards"].cumsum()
    milestone = ((d["ycum"] // 5) > ((d["ycum"] - d["yellow_cards"]) // 5)) & (d["ycum"] >= 5)
    ban = np.where(d["red_cards"] > 0, 3, 0)
    ban = np.maximum(ban, np.where(milestone, np.where(d["ycum"] >= 10, 2, 1), 0))
    susp = np.zeros(len(d), bool)
    keys = (d["player_id"].astype(str) + "|" + d["squad_id"].astype(str)).to_numpy()
    mins = d["minutes_played"].to_numpy()
    for i in np.flatnonzero(ban > 0):
        for j in range(i + 1, min(i + 1 + ban[i], len(d))):
            if keys[j] != keys[i] or mins[j] > 0:
                break
            susp[j] = True
    return susp


def start_run(d):
    """Consecutive starts before this game at this club (capped at 10), log1p."""
    g = d.groupby(["player_id", "squad_id"], sort=False)["start"]
    def run(s):
        out, r = [], 0
        for v in s:
            out.append(r)
            r = r + 1 if v == 1 else 0
        return pd.Series(out, index=s.index)
    return np.log1p(g.transform(run).clip(upper=10))


def logloss(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def fit_logit(X, y):
    sc = StandardScaler().fit(X)
    return sc, LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X), y)


def run(position):
    raw = mm.load_rows(position).sort_values(["player_id", "squad_id", "date", "gameweek"]).reset_index(drop=True)
    raw["susp"] = suspended(raw)
    full = mm.add_features(raw)                      # history includes suspended games (current behaviour)
    clean = mm.add_features(raw[~raw["susp"]])       # history skips them
    out = {}
    for name, d in (("history incl. bans", full), ("history skips bans", clean)):
        d = d.copy()
        d["run"] = start_run(d)
        d["p60y"] = (d["minutes_played"] >= 60).astype(float)
        d["subapp"] = ((d["start"] == 0) & (d["minutes_played"] > 0)).astype(float)
        d = d[~d["susp"]] if "susp" in d else d
        tr = d[(d["season"] == "2526") & (d["gameweek"] <= 23)]
        va = d[(d["season"] == "2526") & (d["gameweek"] > 23)]
        te = d[d["season"] == "2627"]
        q = float(tr.loc[tr["start"] == 1, "p60y"].mean())
        res = {}
        if name == "history incl. bans":  # the live model
            m = Ridge(alpha=1.0).fit(tr[FEATS], tr["mins"])
            ptr = np.clip(m.predict(tr[FEATS]), 0, 90)
            # live fits the isotonic curve on validation; to keep validation clean fit it on training predictions
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(ptr, tr["p60y"])
            for lab, s in (("val", va), ("test", te)):
                pm_ = np.clip(m.predict(s[FEATS]), 0, 90)
                res[("current", lab)] = dict(p60_ll=logloss(s["p60y"], iso.predict(pm_)), mins_rmse=float(np.sqrt(((s["mins"] - pm_) ** 2).mean())))
        for cand, feats in (("logit", FEATS), ("logit +run", FEATS + ["run"])):
            sc, ms = fit_logit(tr[feats], tr["start"])
            ns = tr[tr["start"] == 0]
            sc2, mb = fit_logit(ns[feats], ns["subapp"])
            part_m = float(tr.loc[(tr["mins"] > 0) & (tr["mins"] < 60), "mins"].mean())
            full_m = float(tr.loc[tr["p60y"] == 1, "mins"].mean())
            for lab, s in (("val", va), ("test", te)):
                ps = ms.predict_proba(sc.transform(s[feats]))[:, 1]
                pb = mb.predict_proba(sc2.transform(s[feats]))[:, 1] * (1 - ps)
                p60 = ps * q
                xm = p60 * full_m + (ps * (1 - q) + pb) * part_m
                res[(cand, lab)] = dict(start_ll=logloss(s["start"], ps), p60_ll=logloss(s["p60y"], p60), mins_rmse=float(np.sqrt(((s["mins"] - xm) ** 2).mean())))
        out[name] = res
    return out


if __name__ == "__main__":
    allres = {}
    for pos in (sys.argv[1:] or ["DEF", "MID", "FWD", "GK"]):
        r = run(pos)
        print(f"\n=== {pos} ===")
        for hist, res in r.items():
            for (cand, lab), v in sorted(res.items(), key=lambda kv: (kv[0][1] != "val", kv[0][0])):
                print(f"  [{lab:4s}] {hist:20s} {cand:12s} " + "  ".join(f"{k} {x:.4f}" for k, x in v.items()))
        allres[pos] = {f"{h}|{c}|{l}": v for h, res in r.items() for (c, l), v in res.items()}
    (mm.MODELS / "exp_start_results.json").write_text(json.dumps(allres, indent=1), encoding="utf-8")

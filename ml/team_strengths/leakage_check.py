"""Automatic leakage test: no feature of a game may change when later results change.
python leakage_check.py            (about 2 minutes; also run by weekly_update.py before retraining; exit code 1 on FAIL)

For a cut-off date D (one inside each season), every outcome on or after D is altered (counts, expected goals / assists,
minutes, starts - in the fantasy data and in the FotMob history) and the features are rebuilt with the live code. Every
row dated D or earlier must come out identical: its features may only use games before it. This tests the code, not the
data: it catches a rolling window that includes the current game, a join on the wrong key, a pooled double gameweek.
(It found one on 9 Nov 2026: team-mates' counts were pooled per gameweek, so in a double the first game's "expected"
count contained the second game's outcomes.)
Covers: role / club style / opponent style / team-mates rate of every stat model (its own window and shrinkage), the
FotMob history rates (20 / 40 games, recency-weighted) and set-piece shares, and the start model's features.
Not covered (fitted constants, documented in docs/ml-methodology.md section 4): league-average shrinkage targets and
league step multipliers, which use completed seasons only.
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd

import minutes_model as mm
import player_model as pm

sys.path.insert(0, str(pm.am.REPO / "ml" / "fotmob"))
import features as fmf  # noqa: E402

warnings.filterwarnings("ignore")
COUNTS = ["goals_scored", "assists", "shots_on_target", "key_passes", "interceptions", "clearances", "blocks", "tackles", "m_npxg", "m_xa"]
RESULTS = []


def report(ok, name, detail):
    RESULTS.append((ok, name, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def cutoffs(d):
    """One date inside each season, on a day when clubs play the SECOND game of a double gameweek if there is one."""
    out = []
    for season, x in d.groupby("season"):
        second = x[x.groupby(["squad_id", "gameweek"])["date"].transform("min") < x["date"]]
        days = (second if len(second) else x)["date"].value_counts()
        days = days[(days.index > x["date"].quantile(0.55)) & (days.index < x["date"].quantile(0.95))] if len(days) > 3 else days
        out.append(pd.Timestamp(days.idxmax()))
    return out


def compare(a, b, cols, mask, name):
    worst, where = 0.0, ""
    for c in cols:
        d = (a.loc[mask, c].to_numpy(float) - b.loc[mask, c].to_numpy(float))
        d = np.nan_to_num(np.abs(d))
        if len(d) and d.max() > worst:
            worst, where = float(d.max()), f"{c} ({int((d > 1e-9).sum())} rows)"
    report(worst < 1e-9, name, f"{int(mask.sum())} rows on or before the cut-off, {len(cols)} features, largest change {worst:.2g} {where}")


def check_stats(pos):
    d = pm.load(pos)
    art = json.loads(pm.model_path(pos).read_text(encoding="utf-8"))
    for D in cutoffs(d):
        d2 = d.copy()
        late = d2["date"] >= D
        for c in COUNTS:
            d2.loc[late, c] = d2.loc[late, c] * 3 + 1
        d2.loc[late & (d2["minutes_played"] >= pm.MIN_ROW), "minutes_played"] = (d2["minutes_played"] * 0.7).clip(lower=pm.MIN_ROW).round()
        for stat, a in art.items():
            spec, adj = pm.SPECS[pos][stat], pm.league_adjustments(pos)[stat]
            x1 = pm.build(d, spec, a["window"], a["k_role"], adj)[0]
            x2 = pm.build(d2, spec, a["window"], a["k_role"], adj)[0]
            key = ["season", "player_id", "gameweek", "opponent_id"]
            m = x1[key + ["date", "role", "team", "opp", "mates"]].merge(x2[key + ["role", "team", "opp", "mates"]], on=key, suffixes=("", "_2"))
            cols = ["role", "team", "opp", "mates"]
            compare(m[cols], m[[c + "_2" for c in cols]].rename(columns=lambda c: c[:-2]), cols, m["date"] <= D, f"{pos} {stat} features, cut-off {D.date()}")


def check_fotmob():
    h = fmf.fotmob_history()
    priors = fmf.league_priors(h)
    sph = fmf.setpiece_history()
    const = fmf.setpiece_constants(sph)
    for D in (pd.Timestamp("2026-03-07"), pd.Timestamp("2026-10-17")):
        rows = h.loc[h["date"] <= D, ["player_id", "date"]].drop_duplicates().reset_index(drop=True)
        h2 = h.copy()
        late = h2["date"] >= D
        for c in list(fmf.RATES) + ["mins"]:
            h2.loc[late, c] = h2.loc[late, c] * 2 + 1
        for kw, name in ((dict(), "20 games"), (dict(n=40, prefix="fm40_"), "40 games"), (dict(n=40, prefix="fm40h20_", half_life=20), "40 games, recency-weighted")):
            a, b = fmf.signal_rates(rows, k=3.0, hist=h, priors=priors, **kw), fmf.signal_rates(rows, k=3.0, hist=h2, priors=priors, **kw)
            cols = [c for c in a.columns if c.startswith(kw.get("prefix", "fm_"))]
            compare(a, b, cols, pd.Series(True, index=a.index), f"FotMob history rates ({name}), cut-off {D.date()}")
        s2 = sph.copy()
        late = s2["date"] >= D
        for c in ("mins", "pens", "team_pens", "corners", "team_corners", "fk_xg", "fk_shots"):
            s2.loc[late, c] = s2.loc[late, c] * 2 + 1
        a, b = fmf.setpiece_features(rows, hist=sph, const=const), fmf.setpiece_features(rows, hist=s2, const=const)
        compare(a, b, [c for c in a.columns if c.startswith("fm_")], pd.Series(True, index=a.index), f"set-piece shares, cut-off {D.date()}")


def check_minutes(pos):
    d = mm.load_rows(pos, drop_suspended=False)
    feats = mm.FEATURES + mm.FM_FEATURES
    f1 = mm.add_features(d)
    for D in cutoffs(d):
        d2 = d.copy()
        late = d2["date"] >= D
        d2.loc[late, "minutes_played"] = 90 - d2.loc[late, "minutes_played"].clip(upper=90)
        d2.loc[late, "start"] = 1 - d2.loc[late, "start"]
        d2.loc[late, "suboff"] = 1 - d2.loc[late, "suboff"]
        f2 = mm.add_features(d2)
        key = ["season", "player_id", "squad_id", "gameweek", "opponent_id"]
        m = f1[key + ["date"] + feats].merge(f2[key + feats], on=key, suffixes=("", "_2"))
        compare(m[feats], m[[c + "_2" for c in feats]].rename(columns=lambda c: c[:-2]), feats, m["date"] <= D, f"{pos} start-model features, cut-off {D.date()}")


def run():
    RESULTS.clear()
    print("\n---------- leakage test: features must not change when later results change ----------")
    for pos in ("MID", "FWD", "DEF"):
        check_stats(pos)
    check_fotmob()
    for pos in ("GK", "DEF", "MID", "FWD"):
        check_minutes(pos)
    fails = [r for r in RESULTS if not r[0]]
    print(f"---------- {len(RESULTS)} leakage checks: {len(fails)} FAIL ----------")
    return fails


if __name__ == "__main__":
    if "--old-role" in sys.argv:  # demonstration: the per-gameweek pooling used until 9 Nov 2026 must FAIL
        pm.am.MATCH = ["season", "squad_id", "gameweek"]
    sys.exit(1 if run() else 0)

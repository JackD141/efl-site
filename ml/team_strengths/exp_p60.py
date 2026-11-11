"""Experiment: P(60+ | started) for players just back or taken off early.  python exp_p60.py

Reason (written before the numbers): out of sample the chance a starter reaches 60 minutes is 2-6 points too high when
he missed the previous game or came off before 60 in it (11 Nov 2026): the logistic model is linear in "minutes last
game" and "games missed", and those cases are a step, not a slope. And a manager who regularly takes a player off
early keeps doing it. Candidates for the P(60+ | started) model, fixed beforehand (the P(start) and P(comes on) models
are not touched):
  A  + two flags: came off before 60 last game; missed the previous game
  B  + his own record: share of his last 10 starts at the club in which he reached 60 (shrunk with 3 starts at the
     position average)
  C  A + B
Protocol as minutes_model.py: fit on 2025/26 GW1-23, validate on GW24-35 starters; pass = log-loss gain above one
paired standard error over gameweeks; 2026/27 shown for information. Writes models/exp_p60_results.json.
"""
import json
import warnings

import numpy as np
import pandas as pd

import minutes_model as mm

warnings.filterwarnings("ignore")


def extra_features(f):
    f = f.sort_values(["player_id", "squad_id", "date", "gameweek"]).reset_index(drop=True)
    f["last_short"] = ((f["last_min"] > 0) & (f["last_min"] * 90 < 60)).astype(float)
    f["back"] = (f["zero_run"] >= 1).astype(float)
    st = f[f["start"] == 1]
    g = st.groupby(["player_id", "squad_id"], sort=False)["full"]
    prev = g.shift(1)
    key = [st["player_id"], st["squad_id"]]
    s = prev.groupby(key, sort=False).rolling(10, min_periods=1).sum().reset_index(level=[0, 1], drop=True).reindex(st.index).fillna(0)
    n = prev.notna().astype(float).groupby(key, sort=False).rolling(10, min_periods=1).sum().reset_index(level=[0, 1], drop=True).reindex(st.index).fillna(0)
    q0 = float(st.loc[(st["season"] == "2526") & (st["gameweek"] <= 23), "full"].mean())
    f["q_hist"] = np.nan
    f.loc[st.index, "q_hist"] = (s + 3 * q0) / (n + 3)
    f["q_hist"] = f.groupby(["player_id", "squad_id"])["q_hist"].ffill().fillna(q0)  # rows he did not start: his latest value
    return f


def ll_rows(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    out = {}
    base = mm.FEATURES + mm.FM_FEATURES
    cands = {"live": base, "A flags": base + ["last_short", "back"], "B own record": base + ["q_hist"], "C both": base + ["last_short", "back", "q_hist"]}
    for pos in ("DEF", "MID", "FWD", "GK"):
        f = extra_features(mm.add_features(mm.load_rows(pos)))
        st = f[f["start"] == 1]
        tr = st[(st["season"] == "2526") & (st["gameweek"] <= 23)]
        sets = {"validation": st[(st["season"] == "2526") & (st["gameweek"] > 23)], "2026/27": st[st["season"] == "2627"]}
        fits = {n: mm._logit(tr[c], tr["full"].astype(int)) for n, c in cands.items()}
        print(f"\n=== {pos}: {len(tr)} training starts")
        for lab, s in sets.items():
            y = s["full"].to_numpy()
            rows = {n: ll_rows(y, fits[n][1].predict_proba(fits[n][0].transform(s[c]))[:, 1]) for n, c in cands.items()}
            gws = s["gameweek"].to_numpy()
            for n in list(cands)[1:]:
                d = pd.Series(rows["live"] - rows[n]).groupby(gws).agg(["sum", "size"])
                gain = d["sum"].sum() / d["size"].sum()
                se = float(np.sqrt(len(d) / (len(d) - 1) * ((d["sum"] - gain * d["size"]) ** 2).sum()) / d["size"].sum())
                p = fits[n][1].predict_proba(fits[n][0].transform(s[cands[n]]))[:, 1]
                p0 = fits["live"][1].predict_proba(fits["live"][0].transform(s[base]))[:, 1]
                grp = (s["last_short"] == 1) | (s["back"] == 1)
                print(f"   {lab:10s} {n:13s} log-loss {rows['live'].mean():.4f} -> {rows[n].mean():.4f}, gain {gain:+.5f} (se {se:.5f}) {'PASSES' if lab == 'validation' and gain > se else ''}"
                      f" | back / off early last time (n {int(grp.sum())}): predicted {p0[grp].mean():.3f} -> {p[grp].mean():.3f}, actual {y[grp].mean():.3f}")
                out[f"{pos} {lab} {n}"] = dict(gain=float(gain), se=se, live=float(rows["live"].mean()), new=float(rows[n].mean()),
                                               group=dict(n=int(grp.sum()), before=float(p0[grp].mean()), after=float(p[grp].mean()), actual=float(y[grp].mean())))
    (mm.MODELS / "exp_p60_results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

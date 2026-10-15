"""Experiment: recency weighting inside the history windows. Validation folds inside 2025/26 only.
python exp_recency.py [MID FWD DEF]

Live, every game in a window counts equally: role over his last 20 appearances, FotMob rates over his last 20 / 40.
Here a game a appearances back counts 0.5 ** (a / h) (h = half-life in appearances; same window caps, same shrinkage),
separately for the role and the FotMob rates, on the live structure (features, window, role shrinkage) of each stat.
Question it answers: does a player whose output changed (new role, new club) get picked up faster AND better?
"""
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd

import player_model as pm

sys.path.insert(0, str(pm.am.REPO / "ml" / "fotmob"))
import features as fmf  # noqa: E402

warnings.filterwarnings("ignore")
HALF_LIVES = [None, 40, 20, 10, 5]  # None = equal weights (live)
ORIG_ROLE = pm.role_table


def kernel(n, h):
    return np.ones(n) if h is None else 0.5 ** (np.arange(n) / h)


def norm_dates(s):
    s = pd.to_datetime(s)
    return (s.dt.tz_localize(None) if getattr(s.dt, "tz", None) else s).dt.normalize()


def decayed_fm(rows, hist, priors, signals, n, h, prefix, k=3.0):
    """As features.signal_rates (strictly before the row's date, k pseudo-90s at the league rate) with decayed weights."""
    ker, cols = kernel(n, h), ["mins"] + signals
    vals, sums = hist[cols].to_numpy(float), np.zeros((len(hist), len(cols)))
    for _, idx in hist.groupby("player_id").indices.items():  # hist sorted by player, date
        for c in range(len(cols)):
            sums[idx, c] = np.convolve(vals[idx, c], ker)[:len(idx)]
    roll = pd.DataFrame(sums, columns=["_m"] + [f"_{s}" for s in signals])
    roll["player_id"], roll["date"] = hist["player_id"].to_numpy(), hist["date"].to_numpy()
    roll = roll.drop_duplicates(["player_id", "date"], keep="last").sort_values("date")
    r = pd.DataFrame({"player_id": rows["player_id"].to_numpy(), "date": norm_dates(rows["date"]).to_numpy(), "_ord": np.arange(len(rows))})
    r = pd.merge_asof(r.sort_values("date"), roll, on="date", by="player_id", allow_exact_matches=False, direction="backward").sort_values("_ord")
    m90 = r["_m"].fillna(0).to_numpy() / 90
    return pd.DataFrame({f"{prefix}{s}": (r[f"_{s}"].fillna(0).to_numpy() + k * priors[s]) / (m90 + k) for s in signals}, index=rows.index)


def make_role(h):
    """player_model.role_table with a game a appearances back weighted 0.5 ** (a / h) (still the last ROLE_GAMES)."""
    def role_table(d, col, prior, k, mult):
        x = ORIG_ROLE(d, col, prior, k, mult)  # sorted by player / season / date; has 'exp' and 'level'
        ker = kernel(pm.ROLE_GAMES, h)
        y, e, lvl = x[col].to_numpy(float), x["exp"].to_numpy(float), x["level"].to_numpy()
        role = np.empty(len(x))
        for _, idx in x.groupby("player_id").indices.items():
            for j, i in enumerate(idx):
                prev = idx[max(0, j - pm.ROLE_GAMES):j]
                step = lvl[prev] - lvl[i]
                w = ker[:len(prev)][::-1]  # recency weight (both sides); the league step multiplier scales his output only
                m = np.where(step > 0, mult[1] ** np.clip(step, 0, None), np.where(step < 0, mult[-1] ** np.clip(-step, 0, None), 1.0))
                role[i] = (w @ (m * y[prev]) + k * prior) / (w @ e[prev] + k * prior)
        x["role"] = role
        return x
    return role_table


def fold_scores(x, feats, spec):
    s25 = x[x["season"] == "2526"]
    out = []
    for lo, hi in pm.FOLDS:
        tr, va = s25[s25["gameweek"] <= lo], s25[(s25["gameweek"] > lo) & (s25["gameweek"] <= hi)]
        f = pm.fit(tr, feats, spec)
        out.append(pm.score(va["cnt"].to_numpy(), pm.predict(f, va, spec), f["r"])["loglik"])
    return out


def test_score(x, feats, spec):
    tv, te = x[x["season"] == "2526"], x[x["season"] == "2627"]
    f = pm.fit(tv, feats, spec)
    return pm.score(te["cnt"].to_numpy(), pm.predict(f, te, spec), f["r"])["loglik"], len(te)


def main(positions):
    hist = fmf.fotmob_history()
    priors = fmf.league_priors(hist)
    results = []
    for pos in positions:
        t0 = time.time()
        d = pm.load(pos)
        d["_rid"] = np.arange(len(d))
        art = json.loads(pm.model_path(pos).read_text(encoding="utf-8"))
        used = sorted({f for a in art.values() for f in a["features"] if f.startswith(("fm_", "fm40_")) and f != "fm_cornershare"})
        groups = {}
        for f in used:
            pre, s = ("fm40_", f[5:]) if f.startswith("fm40_") else ("fm_", f[3:])
            groups.setdefault(pre, []).append(s)
        fmcols = {}
        for h in HALF_LIVES:
            fmcols[h] = pd.concat([decayed_fm(d, hist, priors, sig, 40 if pre == "fm40_" else 20, h, pre) for pre, sig in groups.items()], axis=1)
        diff = max(float(np.abs(fmcols[None][c] - d[c]).max()) for c in used)
        print(f"{pos}: equal-weight reimplementation of the FotMob rates matches live to {diff:.1e} ({time.time() - t0:.0f}s)", flush=True)
        for stat, a in art.items():
            spec, adj = pm.SPECS[pos][stat], pm.league_adjustments(pos)[stat]
            feats = a["features"]
            hs_fm = HALF_LIVES if any(f in used for f in feats) else [None]
            for hr in HALF_LIVES:
                pm.role_table = make_role(hr)
                try:
                    x, _, _ = pm.build(d, spec, a["window"], a["k_role"], adj)
                finally:
                    pm.role_table = ORIG_ROLE
                if hr is None:  # check the reimplemented role equals the live one
                    x0, _, _ = pm.build(d, spec, a["window"], a["k_role"], adj)
                    print(f"  {stat}: role reimplementation matches live to {float(np.abs(x0['role'].to_numpy() - x['role'].to_numpy()).max()):.1e}", flush=True)
                for hf in hs_fm:
                    xx = x.copy()
                    for c in used:
                        xx[c] = fmcols[hf][c].to_numpy()[xx["_rid"].to_numpy()]
                    ll = fold_scores(xx, feats, spec)
                    tl, n = test_score(xx, feats, spec)
                    results.append(dict(pos=pos, stat=stat, role_h=hr, fm_h=hf, val=float(np.mean(ll)), fold=ll, test=tl, n_test=n))
            r = pd.DataFrame([q for q in results if q["pos"] == pos and q["stat"] == stat])
            live = r[r["role_h"].isna() & r["fm_h"].isna()].iloc[0]
            r["gain"], r["test_gain"] = r["val"] - live["val"], r["test"] - live["test"]
            r["fold_gain"] = [[round(v - w, 5) for v, w in zip(f, live["fold"])] for f in r["fold"]]
            print(f"\n=== {pos} {stat} (features {feats}) live validation {live['val']:.5f} ===")
            print(r.sort_values("gain", ascending=False)[["role_h", "fm_h", "gain", "fold_gain", "test_gain"]].head(8).round(5).to_string(index=False), flush=True)
        print(f"{pos} done in {time.time() - t0:.0f}s", flush=True)
    (pm.HERE / "models" / "exp_recency_results.json").write_text(pd.DataFrame(results).to_json(orient="records", indent=1), encoding="utf-8")


if __name__ == "__main__":
    main([p.upper() for p in sys.argv[1:]] or ["MID", "FWD", "DEF"])

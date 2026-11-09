"""How much of the model selection is noise?  python exp_selection.py [MID FWD DEF]   (validation folds inside 2025/26 only)

The protocol picks, per stat, the most-preferred candidate within TOL = 0.0005 of the best mean validation log-likelihood,
and the best (window, role shrinkage) for it. TOL is a fixed number: it does not know how noisy a validation difference
is. This script measures that noise:
- per validation row log-likelihood of every (candidate, window, k), both folds pooled (the same rows for every config)
- paired difference between configurations, with a standard error from resampling whole GAMEWEEKS (rows in a gameweek
  share fixtures, odds and team-mates, so they are not independent; players repeat across gameweeks but the block
  bootstrap over the 18 validation gameweeks is the conservative one of the two)
- how often each candidate / each (window, k) would be chosen by the live rule if the validation gameweeks had come out
  differently (bootstrap of gameweeks): a choice that flips often is noise
Writes models/exp_selection_results.json.
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.special import gammaln

import player_model as pm

warnings.filterwarnings("ignore")
B = 400
RNG = np.random.default_rng(0)


def row_ll(y, mu, r):
    mu = np.clip(mu, 1e-6, None)
    p = r / (r + mu)
    return gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r * np.log(p) + y * np.log1p(-p)


def per_row(d, pos, stat):
    """{(candidate, window, k): Series of per-row validation log-likelihood indexed by (fold, row key)}, and the
    gameweek block of each row."""
    spec, adj, cands = pm.SPECS[pos][stat], pm.league_adjustments(pos)[stat], pm.CANDS[pos][stat]
    out, block = {}, None
    for k in pm.K_ROLES:
        for n in pm.WINDOWS:
            x, _, _ = pm.build(d, spec, n, k, adj)
            s25 = x[x["season"] == "2526"]
            for name, feats in cands.items():
                parts = []
                for fi, (a, b) in enumerate(pm.FOLDS):
                    tr, va = s25[s25["gameweek"] <= a], s25[(s25["gameweek"] > a) & (s25["gameweek"] <= b)]
                    f = pm.fit(tr, feats, spec)
                    ll = row_ll(va["cnt"].to_numpy(), pm.predict(f, va, spec), f["r"])
                    idx = pd.MultiIndex.from_arrays([va["gameweek"], va["player_id"], va["opponent_id"], va["is_home"]])
                    parts.append(pd.Series(ll, index=idx))
                s = pd.concat(parts)
                out[(name, n, k)] = s[~s.index.duplicated()].sort_index()
    return out


def choose(means, cands):
    """The live rule on a table of mean log-likelihoods {(name, n, k): mean}."""
    best = max(means.values())
    ok = {c: v for c, v in means.items() if v >= best - pm.TOL}
    pref = min(list(cands).index(c[0]) for c in ok)
    return max((c for c in ok if list(cands).index(c[0]) == pref), key=lambda c: ok[c])


def main(positions):
    res = {}
    for pos in positions:
        d = pm.load(pos)
        for stat in pm.SPECS[pos]:
            cands = pm.CANDS[pos][stat]
            rows = per_row(d, pos, stat)
            keys = list(rows)
            M = pd.concat([rows[c] for c in keys], axis=1, keys=range(len(keys))).dropna()
            gws = M.index.get_level_values(0).to_numpy()
            ugw = np.unique(gws)
            # per-gameweek sums and counts: a bootstrap replicate is a re-weighting of gameweeks
            S = np.vstack([M.to_numpy()[gws == g].sum(0) for g in ugw])
            N = np.array([(gws == g).sum() for g in ugw], float)
            mean = S.sum(0) / N.sum()
            live = choose(dict(zip(keys, mean)), cands)
            li = keys.index(live)
            first = list(cands)[0]
            # the plain reference: most-preferred candidate at the middle window / shrinkage
            ref = keys.index((first, pm.WINDOWS[1], pm.K_ROLES[1]))
            picks, diffs = [], []
            for _ in range(B):
                w = RNG.multinomial(len(ugw), np.ones(len(ugw)) / len(ugw))
                m = (w[:, None] * S).sum(0) / (w * N).sum()
                picks.append(choose(dict(zip(keys, m)), cands))
                diffs.append(m - m[ref])
            diffs = np.array(diffs)
            se = diffs.std(0)
            name_freq = pd.Series([p[0] for p in picks]).value_counts(normalize=True)
            cfg_freq = pd.Series([f"{p[1]}/{p[2]:g}" for p in picks if p[0] == live[0]]).value_counts(normalize=True)
            gain = mean[li] - mean[ref]
            best_of = {name: max((c for c in keys if c[0] == name), key=lambda c: mean[keys.index(c)]) for name in cands}
            print(f"\n=== {pos} {stat}: {len(M)} validation rows in {len(ugw)} gameweeks ===")
            print(f"live choice: {live[0]} (window {live[1]}, k {live[2]:g}); gain over '{first}' at window {pm.WINDOWS[1]} / k {pm.K_ROLES[1]:g}: "
                  f"{gain:+.5f} +- {se[li]:.5f} (z {gain / max(se[li], 1e-12):.1f})")
            for name, c in best_of.items():
                i = keys.index(c)
                g_ = mean[i] - mean[ref]
                print(f"   {name:48s} best at {c[1]:2d}/{c[2]:<4g} gain {g_:+.5f} +- {se[i]:.5f}  chosen in {name_freq.get(name, 0):5.1%} of resamples")
            print(f"   (window / k) picked for the live candidate across resamples: {cfg_freq.round(2).to_dict()}")
            # spread of the 9 (window, k) settings for the live candidate
            same = [keys.index(c) for c in keys if c[0] == live[0]]
            print(f"   the 9 settings of the live candidate span {mean[same].max() - mean[same].min():.5f}")
            res[f"{pos} {stat}"] = dict(live=dict(model=live[0], window=live[1], k=live[2]), gain_vs_reference=float(gain), se=float(se[li]),
                                        name_freq=name_freq.round(3).to_dict(), cfg_freq=cfg_freq.round(3).to_dict(),
                                        candidates={name: dict(window=c[1], k=c[2], gain=float(mean[keys.index(c)] - mean[ref]), se=float(se[keys.index(c)]))
                                                    for name, c in best_of.items()},
                                        setting_span=float(mean[same].max() - mean[same].min()), rows=int(len(M)), gameweeks=int(len(ugw)))
    out = pm.HERE / "models" / "exp_selection_results.json"
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    out.write_text(json.dumps(prev | res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main([p.upper() for p in sys.argv[1:]] or ["MID", "FWD", "DEF"])

"""Experiment: does recent output above / below what the model expected carry over?  python exp_form.py [MID FWD DEF]

Reason (written before the numbers): in the backtest, likely starters who beat our xP by a point a game over their
previous 3-5 gameweeks beat it by ~0.1 the next time (11 Nov 2026). If that is real skill or a changed situation the
stat models do not see (finishing, a new role, a hot team), a "form" feature should help the stat models; if it is
luck plus minutes, it should not. Feature, per stat:
  form = (his actual count over his last K appearances + c) / (what the live model expected in those games + c)
with the expectations from the live model fitted on the fold's training games only, c = 3 average games' worth of
expectation (shrinkage). Candidates, fixed beforehand: K = 5 and K = 10, added to each stat's frozen live structure.
Protocol: validation folds inside 2025/26; pass = gain above max(TOL, one paired standard error) and both folds positive.
Writes models/exp_form_results.json.
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd

import player_model as pm

warnings.filterwarnings("ignore")
KS = (5, 10)


def with_form(x, mu, k):
    """x + 'form': actual / expected over his previous k appearances (date order), shrunk."""
    x = x.assign(_mu=mu).sort_values(["player_id", "date"])
    prev = x.groupby("player_id", sort=False)[["cnt", "_mu"]].shift(1)
    s = prev.groupby(x["player_id"], sort=False).rolling(k, min_periods=1).sum().reset_index(level=0, drop=True).reindex(x.index).fillna(0)
    c = 3 * float(np.mean(mu))
    x["form"] = (s["cnt"] + c) / (s["_mu"] + c)
    return x.drop(columns="_mu")


def main(positions):
    out = {}
    for pos in positions:
        d = pm.load(pos)
        art = json.loads(pm.model_path(pos).read_text(encoding="utf-8"))
        for stat, a in art.items():
            spec = pm.SPECS[pos][stat]
            x, _, _ = pm.build(d, spec, a["window"], a["k_role"], pm.league_adjustments(pos)[stat])
            s25 = x[x["season"] == "2526"]
            base = a["features"]
            sums = {n: {} for n in ("live",) + KS}
            counts, folds = {}, {n: [] for n in ("live",) + KS}
            coefs = {}
            for lo, hi in pm.FOLDS:
                tr_mask = s25["gameweek"] <= lo
                f0 = pm.fit(s25[tr_mask], base, spec)
                mu_all = pm.predict(f0, s25, spec)
                for n in ("live",) + KS:
                    xx = s25 if n == "live" else with_form(s25, mu_all, n)
                    tr, va = xx[xx["gameweek"] <= lo], xx[(xx["gameweek"] > lo) & (xx["gameweek"] <= hi)]
                    f = f0 if n == "live" else pm.fit(tr, base + ["form"], spec)
                    ll = pm.row_loglik(va["cnt"].to_numpy(), pm.predict(f, va, spec), f["r"])
                    folds[n].append(float(ll.mean()))
                    for g, v in pd.Series(ll).groupby(va["gameweek"].to_numpy()):
                        sums[n][int(g)] = float(v.sum())
                        counts[int(g)] = int(len(v))
                    if n != "live":
                        coefs[n] = float(f["model"].coef_[-1])
            tot = sum(counts.values())
            live = sum(sums["live"].values()) / tot
            print(f"=== {pos} {stat}: live {live:.5f}")
            for n in KS:
                gain = sum(sums[n].values()) / tot - live
                se = pm.paired_se(sums[n], sums["live"], counts)
                fd = [round(u - v, 5) for u, v in zip(folds[n], folds["live"])]
                ok = gain > max(pm.TOL, se) and min(fd) > 0
                print(f"   + form over last {n:2d}: gain {gain:+.5f} (se {se:.5f}, folds {fd}) {'PASSES' if ok else ''} | coefficient {coefs[n]:+.3f}")
                out[f"{pos} {stat} form{n}"] = dict(gain=float(gain), se=float(se), folds=fd, passes=bool(ok), coef=coefs[n])
    f = pm.HERE / "models" / "exp_form_results.json"
    prev = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    f.write_text(json.dumps(prev | out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main([p.upper() for p in sys.argv[1:]] or ["MID", "FWD", "DEF"])

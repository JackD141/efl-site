"""Experiment: the job a player is actually given (FotMob lineup slots).  python exp_position.py [MID FWD DEF]

Reason (written before the numbers): the fantasy position is fixed for a season, but what a player produces depends on
his job. His history rates were earned in the jobs he held then; if he is now used differently (a striker moved to
attacking midfield: fewer shots, more interceptions; a full-back used as a wing-back), the history misleads until
enough new games accumulate. FotMob gives every starter's lineup slot (ml/fotmob/positions.py), so:
  jobmult_<stat>  what the jobs of his last 5 starts typically produce / what the jobs over his history window produce
  job_<stat>      what the jobs of his last 5 starts typically produce (helps players with thin history)
Candidates, fixed beforehand, on each stat's frozen live structure: + jobmult; + job; + both.
Protocol (docs/ml-methodology.md section 3): validation folds inside 2025/26; adopt only if the gain is more than
max(TOL, one paired standard error over gameweeks) AND positive in both folds. 2026/27 printed for information.
Writes models/exp_position_results.json.
"""
import json
import sys
import warnings

import numpy as np

import player_model as pm

sys.path.insert(0, str(pm.am.REPO / "ml" / "fotmob"))
import features as fmf  # noqa: E402

warnings.filterwarnings("ignore")
SIGNAL = dict(goals="npxg", assists="xa", sot="sot", kp="chances", int="int", clr="clr", blk="blk", tkl="tkl")


def main(positions):
    out = {}
    for pos in positions:
        d = pm.load(pos)
        art = json.loads(pm.model_path(pos).read_text(encoding="utf-8"))
        pf = fmf.position_features(d[["player_id", "date"]])
        for c in [c for c in pf.columns if c.startswith("job")]:
            d[c] = pf[c].to_numpy()
        print(f"\n######## {pos}: {len(d)} appearances; with a job on record {d['job_int'].notna().mean():.1%}; "
              f"job multiplier outside 0.9-1.1 (interceptions) {((d['jobmult_int'] - 1).abs() > 0.1).mean():.1%}")
        for stat, a in art.items():
            spec, sig = pm.SPECS[pos][stat], SIGNAL[stat]
            x, _, _ = pm.build(d, spec, a["window"], a["k_role"], pm.league_adjustments(pos)[stat])
            x = x.copy()
            x["jobmult"] = x[f"jobmult_{sig}"]
            x["job"] = x[f"job_{sig}"].fillna(x.loc[x["season"] == "2526", f"job_{sig}"].median())
            base = a["features"]
            cands = {"live": base, "+ job multiplier": base + ["jobmult"], "+ job": base + ["job"], "+ both": base + ["job", "jobmult"]}
            res = {n: pm.validate(x, f, spec) for n, f in cands.items()}
            s25, te = x[x["season"] == "2526"], x[x["season"] == "2627"]
            print(f"=== {pos} {stat}: live {res['live'][0]:.5f}")
            for n, f in cands.items():
                if n == "live":
                    continue
                gain = res[n][0] - res["live"][0]
                se = pm.paired_se(res[n][1], res["live"][1], res["live"][2])
                folds = [round(u - v, 5) for u, v in zip(res[n][3], res["live"][3])]
                fit, fit0 = pm.fit(s25, f, spec), pm.fit(s25, base, spec)
                mon = pm.score(te["cnt"].to_numpy(), pm.predict(fit, te, spec), fit["r"])["loglik"] - pm.score(te["cnt"].to_numpy(), pm.predict(fit0, te, spec), fit0["r"])["loglik"]
                coef = dict(zip(f, np.round(pm.fit(x, f, spec)["model"].coef_, 3)))
                ok = gain > max(pm.TOL, se) and min(folds) > 0
                print(f"   {n:18s} gain {gain:+.5f} (se {se:.5f}, folds {folds}) {'PASSES' if ok else '      '} | 2026/27 {mon:+.5f} | coef "
                      f"{ {k: coef[k] for k in ('job', 'jobmult') if k in coef} }")
                out[f"{pos} {stat} {n}"] = dict(gain=float(gain), se=float(se), folds=folds, passes=bool(ok), monitor=float(mon),
                                                coef={k: float(coef[k]) for k in ("job", "jobmult") if k in coef})
    f = pm.HERE / "models" / "exp_position_results.json"
    prev = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    f.write_text(json.dumps(prev | out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main([p.upper() for p in sys.argv[1:]] or ["MID", "FWD", "DEF"])

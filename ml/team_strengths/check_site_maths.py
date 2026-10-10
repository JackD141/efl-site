"""Reference xP from the plan JSONs, computed in Python exactly as the pages should (to check the browser maths).
python check_site_maths.py          writes public/data/check_xp.json (delete it afterwards; it is not committed)

Then, in the browser on each Player Picks tab, compare with the page's own numbers, e.g. on picks.html?pos=MID:
  const ref = await (await fetch('data/check_xp.json')).json();
  Math.max(...state.plan.players.filter(d => ref.MID[d.id] !== undefined).map(d => Math.abs(weekXp(d, state.plan.firstGw, d.xMins).xp - ref.MID[d.id])))
(DEF: weekXp(d, gw).xp90 with state.mins = {}; GK: kxp(k, gw)). Expect ~1e-6 (forwards' Poisson goals ~2e-5).
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import nbinom, poisson

DATA = Path(__file__).resolve().parents[2] / "public" / "data"
K = np.arange(300)


def pmf(mu, r):
    return poisson.pmf(K, mu) if r > 1e3 else nbinom.pmf(K, r, r / (r + mu))


def curve(c, m, key):
    return float(np.interp(m, c["mins"], c[key])) if m > 0 else 0.0


def glm(m, vals, floor):
    z = m["intercept"]
    for i, n in enumerate(m["features"]):
        v = vals[n] if n == "home" else math.log(max(vals[n], floor))
        z += m["coef"][i] * (v - m["mean"][i]) / m["sd"][i]
    return math.exp(z)


def vals_for(p, clubs, d, f, st):
    c = clubs[d["club"]]
    o = clubs.get(f["oppId"], c)
    v = dict(role=d["role"][st], team=c["style"][st]["team"], opp=o["style"][st]["opp"], lam_own=f["lamOwn"], lam_opp=f["lamOpp"], home=f["home"],
             **d.get("fm", {}))
    if d.get("mates"):
        v["mates"] = d["mates"][st]
    return v


def scenarios(p, m):
    p60 = curve(p["p60Curve"], m, "p")
    return p60, max(0.0, curve(p["appCurve"], m, "pts") - 2 * p60)


def attackers(p):
    clubs = {c["id"]: c for c in p["clubs"]}
    sc = p["scoring"]
    out = {}
    for d in p["players"]:
        if d["xMins"] <= 0:
            continue
        tot = 0.0
        for f in clubs[d["club"]]["weeks"][0]["fx"]:
            rate = {}
            for st, m in p["model"].items():
                rate[st] = glm(m, vals_for(p, clubs, d, f, st), 1e-4) * m.get("scale", 1) + (d["fm"].get(m.get("penCol", "fm_pxg"), 0) if m.get("pens") and d.get("fm") else 0)

            def at(mins, app):
                t = mins / 90
                mu = {k: v * t for k, v in rate.items()}
                r = d["rates"]
                x = app + sc["goal"] * mu["goals"] + sc["hatTrick"] * float(pmf(mu["goals"], p["model"]["goals"]["r"])[3:].sum()) + sc["assist"] * mu["assists"]
                x += sc["sot"] * mu["sot"] + float((pmf(mu["kp"], p["model"]["kp"]["r"]) * (K // sc["keyPassPer"])).sum())
                x += sc.get("interception", 0) * mu.get("int", 0)
                return x + t * (sc["yellow"] * r["y90"] + sc["red"] * r["r90"] + sc["penMiss"] * r["pm90"] + sc["ownGoal"] * r["og90"])
            if p.get("minsMix"):
                p60, part = scenarios(p, d["xMins"])
                tot += p60 * at(d.get("mFull", p["minsMix"]["full"]), sc["appearance60"]) + part * at(p["minsMix"]["part"], sc["appearance"])
            else:
                tot += at(d["xMins"], curve(p["appCurve"], d["xMins"], "pts"))
        out[d["id"]] = tot
    return out


def defenders(p):
    clubs = {c["id"]: c for c in p["clubs"]}
    sc = p["scoring"]
    units = {"clr": 4, "blk": 2, "tkl": 2}
    out = {}
    for d in p["defenders"]:
        if d["xMins"] <= 0:
            continue
        tot = 0.0
        for f in clubs[d["club"]]["weeks"][0]["fx"]:
            rate = {st: glm(p["model"][st], vals_for(p, clubs, d, f, st), p["model"][st].get("floor", 1e-3)) for st in list(units) + ["goals", "assists"]}
            mg = p["model"]["goals"]

            def at(mins, app, cs_w):
                t = mins / 90
                x = app + sc["cleanSheet"] * f["cs"] * cs_w + f["gcPts"] * t
                for st, u in units.items():
                    x += float((pmf(rate[st] * t, p["model"][st]["r"]) * (K // u)).sum())
                x += sc["goal"] * (rate["goals"] * mg["scale"] + (d["fm"].get(mg.get("penCol", "fm_pxg"), 0) if mg["pens"] else 0)) * t
                x += sc["assist"] * rate["assists"] * p["model"]["assists"]["scale"] * t
                return x + t * (sc["yellow"] * d["rates"]["y90"] + sc["red"] * d["rates"]["r90"] + p["other"])
            if p.get("minsMix"):
                p60, part = scenarios(p, d["xMins"])
                tot += p60 * at(d.get("mFull", p["minsMix"]["full"]), 2, 1.0) + part * at(p["minsMix"]["part"], 1, 0.0)
            else:
                tot += at(d["xMins"], curve(p["appCurve"], d["xMins"], "pts"), curve(p["p60Curve"], d["xMins"], "p"))
        out[d["id"]] = tot
    return out


def keepers(p):
    clubs = {c["id"]: c for c in p["clubs"]}
    out = {}
    for k in p["keepers"]:
        w = clubs[k["club"]]["weeks"][0]
        if p.get("minsMix"):
            p60, part = scenarios(p, k["xMins"])
            out[k["id"]] = p60 * w["xp"] + part * w["games"]
        else:
            out[k["id"]] = k["xMins"] / 90 * w["xp"]
    return out


def main():
    load = lambda f: json.loads((DATA / f).read_text(encoding="utf-8"))
    ref = {"MID": attackers(load("mid_plan.json")), "FWD": attackers(load("fwd_plan.json")),
           "DEF": defenders(load("defender_plan.json")), "GK": keepers(load("keeper_plan.json"))}
    (DATA / "check_xp.json").write_text(json.dumps({k: {str(i): v for i, v in d.items()} for k, d in ref.items()}), encoding="utf-8")
    print({k: len(v) for k, v in ref.items()}, "-> public/data/check_xp.json")


if __name__ == "__main__":
    main()

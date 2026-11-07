"""Experiment: playing-style features for midfielders' interceptions. Validation folds inside 2025/26 only.
python exp_intstyle.py

Logic: an interception is one of the OPPONENT's misplaced passes, and midfielders mostly cut out short / medium passes
(long balls go over them). So the opponent's misplaced short passes per game measure "teams that give the ball away in
midfield" directly, with many more events (less noise) than the live opponent feature (interceptions midfielders make
against them). And a side that sees less of the ball defends more: his own team's share of the passes in its games.
Both from FotMob team totals per match (sum over players), rolling over each team's previous 12 matches (any league,
FotMob team ids so promoted / relegated clubs keep their history), shrunk to the all-team mean with 3 pseudo-matches.

Candidates (fixed before looking at results): live; + opponent misplaced short passes; + own pass share; + both.
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd

import player_model as pm

sys.path.insert(0, str(pm.am.REPO / "ml" / "fotmob"))
import link_fotmob  # noqa: E402

warnings.filterwarnings("ignore")
FM = pm.am.REPO / "data" / "fotmob"
N, K = 12, 3.0


def team_matches():
    """One row per team per FotMob match: date, fm team id, passes, misplaced short passes, opponent's passes."""
    rows = []
    for f in sorted(FM.glob("*/player_matches.csv")):
        p = pd.read_csv(f, low_memory=False)
        num = lambda c: pd.to_numeric(p[c], errors="coerce").fillna(0)
        p = p.assign(att=num("Accurate passes (attempts)"), acc=num("Accurate passes"), latt=num("Accurate long balls (attempts)"),
                     lacc=num("Accurate long balls"))
        t = p.groupby(["match_id", "fm_team_id"]).agg(utc=("utc", "first"), att=("att", "sum"), acc=("acc", "sum"), latt=("latt", "sum"),
                                                     lacc=("lacc", "sum")).reset_index()
        rows.append(t)
    t = pd.concat(rows, ignore_index=True)
    t["mis_short"] = (t["att"] - t["acc"]) - (t["latt"] - t["lacc"])
    opp = t[["match_id", "fm_team_id", "att"]].rename(columns={"fm_team_id": "opp_fm", "att": "opp_att"})
    t = t.merge(opp, on="match_id")
    t = t[t["fm_team_id"] != t["opp_fm"]].copy()
    t["share"] = t["att"] / (t["att"] + t["opp_att"]).clip(lower=1)
    t["date"] = pd.to_datetime(t["utc"], utc=True, format="ISO8601").dt.tz_localize(None).dt.normalize()
    return t.sort_values(["fm_team_id", "date"]).reset_index(drop=True)


def rolling_style(t):
    """Per team and date: mean misplaced short passes and pass share over its previous N matches (strictly before),
    shrunk to the overall mean with K pseudo-matches."""
    prior = {c: float(t[c].mean()) for c in ("mis_short", "share")}
    g = t.groupby("fm_team_id")
    out = t[["fm_team_id", "date"]].copy()
    n = g.cumcount().clip(upper=N)
    for c in ("mis_short", "share"):
        s = g[c].transform(lambda x: x.shift(1).rolling(N, min_periods=1).sum()).fillna(0)
        out[c] = (s + K * prior[c]) / (n + K)
    return out, prior


def efl_to_fm():
    """{EFL squad id: FotMob team id} from both seasons' match names."""
    out = {}
    for efl_season, fm_season, squads_path in (("2025_26", "2025_2026", pm.am.REPO / "data" / "squads.json"),
                                               ("2026_27", "2026_2027", pm.HERE / "cache" / "squads.json")):
        m = pd.read_csv(FM / fm_season / "matches.csv")
        tm = link_fotmob.team_map(json.load(open(squads_path, encoding="utf-8")), set(m["home"]) | set(m["away"]))
        ids = dict(zip(m["home"], m["home_id"])) | dict(zip(m["away"], m["away_id"]))
        for name, sid in tm.items():
            out[int(sid)] = int(ids[name])
    return out


def attach(x, style, prior, sq2fm):
    """x rows (player game: squad_id, opponent_id, date) + opp_mis_short (opponent's style) and own_share (his team's)."""
    st = style.sort_values("date")
    x = x.copy()
    x["date"] = pd.to_datetime(x["date"]).dt.normalize()
    x["_o"] = np.arange(len(x))
    for col, side, name in (("mis_short", "opponent_id", "opp_mis_short"), ("share", "squad_id", "own_share")):
        k = x[[side, "date", "_o"]].assign(fm_team_id=x[side].map(sq2fm)).dropna(subset=["fm_team_id"])
        k["fm_team_id"] = k["fm_team_id"].astype(int)
        m = pd.merge_asof(k.sort_values("date"), st[["fm_team_id", "date", col]], on="date", by="fm_team_id",
                          allow_exact_matches=False, direction="backward").set_index("_o")[col]
        x[name] = x["_o"].map(m).fillna(prior[col])
    return x.drop(columns="_o")


def main():
    t = team_matches()
    style, prior = rolling_style(t)
    sq2fm = efl_to_fm()
    print(f"team-matches {len(t)}; mean misplaced short passes {prior['mis_short']:.1f}, pass share {prior['share']:.3f}; "
          f"EFL clubs mapped {len(sq2fm)}")
    d = pm.load("MID")
    if "date" not in d:
        d = pm.mm.attach_dates(d)
    art = json.loads(pm.model_path("MID").read_text(encoding="utf-8"))["int"]
    spec = pm.SPECS["MID"]["int"]
    x, _, _ = pm.build(d, spec, art["window"], art["k_role"], pm.league_adjustments("MID")["int"])
    x = attach(x, style, prior, sq2fm)
    print("feature coverage:", {c: float((x[c] != prior[c.replace('opp_', '').replace('own_', '')]).mean()) for c in ("opp_mis_short", "own_share")})
    print("correlation with the live opponent feature (log):", round(float(np.corrcoef(np.log(x["opp"]), np.log(x["opp_mis_short"]))[0, 1]), 3))
    base = art["features"]
    cands = {"live": base, "+ opponent misplaced short passes": base + ["opp_mis_short"], "+ own pass share": base + ["own_share"],
             "+ both": base + ["opp_mis_short", "own_share"]}
    s25, te = x[x["season"] == "2526"], x[x["season"] == "2627"]
    res = []
    for name, feats in cands.items():
        ll = []
        for lo, hi in pm.FOLDS:
            tr, va = s25[s25["gameweek"] <= lo], s25[(s25["gameweek"] > lo) & (s25["gameweek"] <= hi)]
            f = pm.fit(tr, feats, spec)
            ll.append(pm.score(va["cnt"].to_numpy(), pm.predict(f, va, spec), f["r"])["loglik"])
        f = pm.fit(s25, feats, spec)
        test = pm.score(te["cnt"].to_numpy(), pm.predict(f, te, spec), f["r"])["loglik"]
        coef = dict(zip(feats, np.round(pm.fit(x, feats, spec)["model"].coef_, 3)))
        res.append(dict(candidate=name, val=float(np.mean(ll)), folds=[round(v, 5) for v in ll], test=test, coef={k: coef[k] for k in feats if k not in base[:-1] or k == "opp"}))
    live = res[0]
    for r in res:
        print(f"{r['candidate']:36s} val {r['val']:.5f} (gain {r['val'] - live['val']:+.5f}, folds {[round(a - b, 5) for a, b in zip(r['folds'], live['folds'])]}) "
              f"| test {r['test']:.5f} ({r['test'] - live['test']:+.5f}) | coef {r['coef']}")
    (pm.HERE / "models" / "exp_intstyle_results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()

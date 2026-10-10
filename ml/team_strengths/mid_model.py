"""Midfielder stat models v2 (FotMob-based). Forwards still use attack_model.py.
python mid_model.py        trains, validates, tests, writes models/mid_model.json

Per stat, a Poisson GLM with minutes exposure on standardised logs of: role, club style, opponent style, odds
(own / opponent expected goals), home, and (optionally) FotMob history rates. Differences from attack_model.py v1:
- goals and assists are modelled through their EXPECTED versions: the GLM predicts match npxG (xA) per 90, then
  goals = c_g * npxG + his penalty-xG rate, assists = c_a * xA (c fitted on training; npxG / xA are far less noisy
  than goals / assists, so the model learns chance quality, not finishing luck)
- FotMob history (last 20 appearances, from 2024/25, shrunk): npxG, penalty xG, xA, shots, chances created
- league changes: games played in another league count with the measured step multiplier (league_steps.py,
  shrunk towards 1 by sample size) instead of a flat down-weight; e.g. npxG from a League One season counts ~0.8x
  for a player now in the Championship
- club style carries over from last season: the prior a club starts the season with is the league average times
  (last season's club rate / average) ^ persistence (measured year-to-year slope) times the promotion / relegation
  multiplier, instead of every club starting at the all-league average
Protocol as always: fit 2025/26 GW1-23, choose window / role shrinkage / features on GW24-35 (log-likelihood of the
ACTUAL count), test once on 2026/27, refit on everything.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

import attack_model as am
import defence_model as dm
import minutes_model as mm
from saves_model import nb_loglik, nb_r

sys.path.insert(0, str(am.REPO / "ml" / "fotmob"))
import features as fmf  # noqa: E402
import link_fotmob  # noqa: E402

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
MODEL_PATH = HERE / "models" / "mid_model.json"
STEPS_PATH = HERE / "models" / "league_steps.json"
SPEC = {
    "goals": dict(column="goals_scored", target="m_npxg", fm="npxg", pens=True, extra=["fm_npxg"]),
    "assists": dict(column="assists", target="m_xa", fm="xa", pens=False, extra=["fm_xa"]),
    "sot": dict(column="shots_on_target", target="shots_on_target", fm="sot", pens=False, extra=["fm_shots", "fm_npxg"]),
    "kp": dict(column="key_passes", target="key_passes", fm="kp", pens=False, extra=["fm_chances", "fm_xa"]),
    "int": dict(column="interceptions", target="interceptions", fm="int", pens=False, extra=[]),
}
BASE = ["role", "team", "opp", "lam_own", "lam_opp", "home"]
# "mates": his CURRENT team-mates' rate (same position, this club, last N gameweeks, excluding him), so role (his share
# relative to team-mates) multiplies the environment he actually plays in. "team" includes his own output, which
# double-counts high-role players (the Morris problem). fm40_* = FotMob history over 40 instead of 20 appearances.
BM = ["role", "mates", "opp", "lam_own", "lam_opp", "home"]
CANDIDATES = {  # in order of preference when within TOL (incumbent first, then the simpler alternatives)
    "goals": {"v2": BASE, "mates": BM, "mates + npxG hist": BM + ["fm_npxg"], "mates + npxG hist40": BM + ["fm40_npxg"]},
    "assists": {"v2": BASE + ["fm_xa"], "mates": BM + ["fm_xa"], "mates hist40": BM + ["fm40_xa"]},
    "sot": {"v2": BASE + ["fm_shots", "fm_npxg"], "mates": BM + ["fm_shots", "fm_npxg"], "mates + SOT hist": BM + ["fm_sot", "fm_npxg"],
            "mates + SOT + shots hist": BM + ["fm_sot", "fm_shots", "fm_npxg"], "mates + SOT hist40": BM + ["fm40_sot", "fm40_npxg"]},
    "kp": {"v2": BASE + ["fm_chances", "fm_xa"], "mates": BM + ["fm_chances", "fm_xa"], "mates hist40": BM + ["fm40_chances", "fm40_xa"]},
    "int": {"v2": ["role", "team", "opp"], "mates": ["role", "mates", "opp"], "mates + int hist": ["role", "mates", "opp", "fm_int"],
            "mates + int hist40": ["role", "mates", "opp", "fm40_int"]},
}
WINDOWS = [8, 12, 46]
K_ROLES = [3.0, 10.0, 30.0]
# rolling-origin validation inside 2025/26 (2026/27 is touched once, at the end)
FOLDS = [(17, 26), (26, 35)]  # (train up to GW a, validate a+1..b)
ROLE_GAMES, MIN_ROW, TOL = 20, 10, 5e-4
N0_ROLE, N0_TEAM = 30.0, 14.0  # shrink measured multipliers towards 1: weight n / (n + n0)
KEY = ["season", "player_id", "squad_id", "opponent_id", "is_home"]


def levels():
    """{season tag: {squad id: 1 Championship / 2 League One / 3 League Two}}"""
    lv = {10: 1, 11: 2, 12: 3}
    old = json.load(open(am.REPO / "data" / "squads.json", encoding="utf-8"))
    cur = json.load(open(HERE / "cache" / "squads.json", encoding="utf-8"))
    return {"2526": {s["id"]: lv[s["competitionId"]] for s in old}, "2627": {s["id"]: lv[s["competitionId"]] for s in cur}}


def league_adjustments():
    js = json.loads(STEPS_PATH.read_text(encoding="utf-8"))
    out = {}
    for stat, spec in SPEC.items():
        role = js["role"]["MID"][spec["fm"]]
        team = js["team"]["MID"][spec["fm"]]
        shrink = lambda e, n0: float(np.exp(e["n"] / (e["n"] + n0) * np.log(e["mult"]))) if e["mult"] == e["mult"] and e["mult"] > 0 else 1.0
        out[stat] = dict(role={1: shrink(role["1"], N0_ROLE), -1: shrink(role["-1"], N0_ROLE)},
                         team={1: shrink(team["1"], N0_TEAM), -1: shrink(team["-1"], N0_TEAM)},
                         persist=float(js.get("persistence", {}).get("MID", {}).get(spec["fm"], 0.35)))
    return out


def load():
    d = am.load("MID").drop(columns=["hid", "aid"])
    hist = fmf.fotmob_history()
    d = fmf.signal_rates(mm.attach_dates(d), k=3.0, hist=hist)
    d = fmf.signal_rates(d, k=3.0, n=40, hist=hist, prefix="fm40_")
    fm = link_fotmob.fotmob_for_efl()[KEY + ["fm_minutes", "npxg", "xa"]].rename(columns={"npxg": "m_npxg", "xa": "m_xa"})
    d = d.merge(fm, on=KEY, how="left")
    d["linked"] = d["fm_minutes"].notna()
    for c in ("m_npxg", "m_xa"):
        d[c] = d[c].fillna(0)  # FotMob omits xG / xA when there was none; unlinked rows (2%) are dropped from x-target fits
    lv = levels()
    d["level"] = [lv[s].get(c, 2) for s, c in zip(d["season"], d["squad_id"])]
    return d.reset_index(drop=True)


def role_table(d, col, prior, k, mult):
    """Role over his last ROLE_GAMES appearances vs same-position team-mates; games in another league count x mult[step]."""
    x = am.role_table(d, col, prior, k)  # sorted by player / season / gameweek, has 'exp' and the unadjusted role
    multi = x.groupby("player_id")["level"].transform("nunique") > 1
    for _, idx in x[multi].groupby("player_id").indices.items():
        rows = x[multi].iloc[idx]
        ix = rows.index.to_numpy()
        y, e, lvl = rows[col].to_numpy(float), rows["exp"].to_numpy(float), rows["level"].to_numpy()
        for j in range(len(ix)):
            lo = max(0, j - ROLE_GAMES)
            step = lvl[lo:j] - lvl[j]  # +1: the old game was a league below (he has moved up)
            w = np.where(step > 0, mult[1] ** step, np.where(step < 0, mult[-1] ** (-step), 1.0))
            x.at[ix[j], "role"] = (np.dot(w, y[lo:j]) + k * prior) / (np.dot(e[lo:j], np.ones(j - lo)) + k * prior)
    return x


def style_tables(d, col, n, prior, adj):
    """Club style (team) and opponent style per (season, club, gameweek) from earlier gameweeks this season, shrunk with
    K_TEAM pseudo-90s towards the club's carried-over prior (team) or the league average (opponent)."""
    team, opp = dm.style_tables(d, col, n, prior)
    # carried prior for each club's second season: last season's rate relative to average, ^ persistence, x step mult
    lv = levels()
    s1 = d[d["season"] == "2526"].groupby("squad_id").agg(x=(col, "sum"), m=("minutes_played", "sum"))
    avg = s1["x"].sum() / (s1["m"].sum() / 90)
    rel = ((s1["x"] + dm.K_TEAM * avg) / (s1["m"] / 90 + dm.K_TEAM)) / avg
    carry = {}
    for cid, r in rel.items():
        step = lv["2526"].get(cid, 2) - lv["2627"].get(cid, lv["2526"].get(cid, 2))
        m = adj["team"].get(int(np.sign(step)), 1.0) if step else 1.0
        carry[cid] = prior * (r ** adj["persist"]) * m
    # recompute team style for 2026/27 with the carried prior in place of the league average
    g = d[d["season"] == "2627"].groupby(["squad_id", "gameweek"]).agg(x=(col, "sum"), m90=("minutes_played", lambda s: s.sum() / 90)).reset_index()
    rows = []
    for cid, x in g.groupby("squad_id"):
        gws = np.arange(1, 47)
        xs = pd.Series(0.0, index=gws).add(x.set_index("gameweek")["x"], fill_value=0)
        ms = pd.Series(0.0, index=gws).add(x.set_index("gameweek")["m90"], fill_value=0)
        sx = xs.shift(1).rolling(n, min_periods=1).sum().fillna(0)
        sm = ms.shift(1).rolling(n, min_periods=1).sum().fillna(0)
        p = carry.get(cid, prior)
        rows.append(pd.DataFrame({"season": "2627", "squad_id": cid, "gameweek": gws, "team": (sx + dm.K_TEAM * p) / (sm + dm.K_TEAM)}))
    if rows:
        t27 = pd.concat(rows, ignore_index=True)
        team = pd.concat([team[team["season"] != "2627"], t27], ignore_index=True)
    return team, opp, carry


def mates_table(d, col, n, prior, carry):
    """Per (season, player, club, gameweek): his club's same-position rate per 90 over the previous n gameweeks of this
    season EXCLUDING his own minutes and output, shrunk with K_TEAM pseudo-90s to the club's prior (carried over for
    2026/27, league average for 2025/26)."""
    out = []
    gws = np.arange(1, 47)
    club = d.groupby(["season", "squad_id", "gameweek"]).agg(x=(col, "sum"), m=("minutes_played", "sum"))
    own = d.groupby(["season", "player_id", "squad_id", "gameweek"]).agg(x=(col, "sum"), m=("minutes_played", "sum"))

    def roll(series):
        return pd.Series(0.0, index=gws).add(series, fill_value=0).shift(1).rolling(n, min_periods=1).sum().fillna(0)
    club_roll = {}
    for (season, cid), g in club.groupby(level=[0, 1]):
        g = g.droplevel([0, 1])
        club_roll[(season, cid)] = (roll(g["x"]), roll(g["m"]))
    for (season, pid, cid), g in own.groupby(level=[0, 1, 2]):
        g = g.droplevel([0, 1, 2])
        cx, cm = club_roll[(season, cid)]
        ox, om = roll(g["x"]), roll(g["m"])
        p0 = carry.get(cid, prior) if season == "2627" else prior
        val = ((cx - ox).clip(lower=0) + dm.K_TEAM * p0) / ((cm - om).clip(lower=0) / 90 + dm.K_TEAM)
        out.append(pd.DataFrame({"season": season, "player_id": pid, "squad_id": cid, "gameweek": gws, "mates": val.to_numpy()}))
    return pd.concat(out, ignore_index=True)


def build(d, stat, n, k, adj):
    spec = SPEC[stat]
    tcol = spec["target"]
    rows = d if tcol == spec["column"] else d[d["linked"]]
    tr = rows[(rows["season"] == "2526") & (rows["gameweek"] <= 23)]
    prior = float(tr[tcol].sum() / (tr["minutes_played"].sum() / 90))
    team, opp, carry = style_tables(rows, tcol, n, prior, adj[stat])
    x = role_table(rows, tcol, prior, k, adj[stat]["role"])
    x = x.merge(team, on=["season", "squad_id", "gameweek"], how="left").merge(opp, on=["season", "opponent_id", "gameweek"], how="left")
    x["team"], x["opp"] = x["team"].fillna(prior), x["opp"].fillna(prior)
    x = x.merge(mates_table(rows, tcol, n, prior, carry), on=["season", "player_id", "squad_id", "gameweek"], how="left")
    x["mates"] = x["mates"].fillna(prior)
    x["y"], x["cnt"], x["t"] = x[tcol], x[spec["column"]], x["minutes_played"] / 90
    return x[x["minutes_played"] >= MIN_ROW], prior, carry


def fit(x, feats, stat):
    spec = SPEC[stat]
    X, sc = am.design(x, feats)
    m = PoissonRegressor(alpha=1e-4, max_iter=3000).fit(X, x["y"] / x["t"], sample_weight=x["t"])
    raw = m.predict(X) * x["t"]
    pen = (x["fm_pxg"] * x["t"]).to_numpy() if spec["pens"] else 0.0
    scale = 1.0 if spec["target"] == spec["column"] else float((x["cnt"].sum() - np.sum(pen)) / raw.sum())
    mu = scale * raw + pen
    return dict(model=m, scaler=sc, feats=feats, scale=scale, r=nb_r(x["cnt"].to_numpy(), np.asarray(mu)))


def predict(f, x, stat):
    raw = f["model"].predict(am.design(x, f["feats"], f["scaler"])[0]) * x["t"].to_numpy()
    pen = (x["fm_pxg"] * x["t"]).to_numpy() if SPEC[stat]["pens"] else 0.0
    return f["scale"] * raw + pen


def score(y, mu, r):
    mu = np.clip(mu, 1e-6, None)
    return dict(loglik=nb_loglik(y, mu, r), mse=float(((y - mu) ** 2).mean()), mae=float(np.abs(y - mu).mean()))


def main():
    d = load()
    adj = league_adjustments()
    prev = json.loads(MODEL_PATH.read_text(encoding="utf-8")) if MODEL_PATH.exists() else {}
    out = {}
    for stat, spec in SPEC.items():
        cands = CANDIDATES[stat]
        rows, built = [], {}
        for k in K_ROLES:
            for n in WINDOWS:
                x, prior, carry = build(d, stat, n, k, adj)
                built[(n, k)] = (x, prior, carry)
                s25 = x[x["season"] == "2526"]
                for name, feats in cands.items():
                    ll, mse = [], []
                    for a, b in FOLDS:
                        tr, va = s25[s25["gameweek"] <= a], s25[(s25["gameweek"] > a) & (s25["gameweek"] <= b)]
                        f = fit(tr, feats, stat)
                        sc = score(va["cnt"].to_numpy(), predict(f, va, stat), f["r"])
                        ll.append(sc["loglik"]); mse.append(sc["mse"])
                    rows.append(dict(model=name, window=n, k=k, loglik=float(np.mean(ll)), mse=float(np.mean(mse)), fold_ll=[round(v, 4) for v in ll]))
        res = pd.DataFrame(rows)
        res["pref"] = res["model"].map(list(cands).index)
        best = res[res["loglik"] >= res["loglik"].max() - TOL].sort_values(["pref", "loglik"], ascending=[True, False]).iloc[0]
        print(f"\n=== MID {stat} (target {spec['target']}): mean validation over {len(FOLDS)} folds, best per candidate ===")
        print(res.sort_values("loglik", ascending=False).groupby("model").head(1)[["model", "window", "k", "loglik", "mse", "fold_ll"]].round(4).to_string(index=False))
        feats, n_best, k_best = cands[best["model"]], int(best["window"]), float(best["k"])
        print(f"chosen: {best['model']}, window {n_best}, role shrinkage {k_best:g}")
        x, prior, carry = built[(n_best, k_best)]
        tv, te = x[x["season"] == "2526"], x[x["season"] == "2627"]
        f = fit(tv, feats, stat)
        test = score(te["cnt"].to_numpy(), predict(f, te, stat), f["r"])
        # incumbent (previous saved configuration) on the same test rows
        inc = prev.get(stat)
        test_inc = None
        if inc:
            xi = built.get((inc["window"], inc["k_role"]))
            if xi is None:
                xi = build(d, stat, inc["window"], inc["k_role"], adj)
            xi = xi[0]
            fi = fit(xi[xi["season"] == "2526"], inc["features"], stat)
            tei = xi[xi["season"] == "2627"]
            test_inc = score(tei["cnt"].to_numpy(), predict(fi, tei, stat), fi["r"])
        print(f"TEST 2026/27 (once; n={len(te)}): chosen loglik {test['loglik']:.4f} mse {test['mse']:.4f}"
              + (f" | previous v2 loglik {test_inc['loglik']:.4f} mse {test_inc['mse']:.4f}" if test_inc else "") + f" | conversion {f['scale']:.3f}")
        fin = fit(x, feats, stat)
        out[stat] = dict(column=spec["column"], target=spec["target"], pens=spec["pens"], model=best["model"], features=feats, window=n_best,
                         k_role=k_best, prior=prior, scale=fin["scale"], intercept=float(fin["model"].intercept_), coef=fin["model"].coef_.tolist(),
                         scaler_mean=fin["scaler"][0].tolist(), scaler_sd=fin["scaler"][1].tolist(), nb_r=fin["r"],
                         league=adj[stat], carry={str(c): v for c, v in carry.items()},
                         validation=res.drop(columns=["pref"]).to_dict("records"),
                         test=dict(v2=test, v1=test_inc or test, n=int(len(te))))
        print("final coefficients", {kk: round(v, 3) for kk, v in zip(feats, fin["model"].coef_)})
    MODEL_PATH.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\nsaved", MODEL_PATH)


if __name__ == "__main__":
    main()


# ---------- live inputs for the site (attackers.py) ----------
def live_inputs(players, art=None):
    """Current role per stat (last ROLE_GAMES appearances, other-league games x step multiplier relative to his CURRENT
    club's league), club / opponent style per stat (this season, carried-over prior), FotMob history rates as of today."""
    art = art or json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    d = load()
    lv = levels()["2627"]
    cur_club = {p["id"]: p["squadId"] for p in players}
    roles, styles, mates_out = {}, {}, {}
    last_gw = int(d.loc[d["season"] == "2627", "gameweek"].max())
    for stat, a in art.items():
        spec = SPEC[stat]
        tcol, prior, k, n = spec["target"], a["prior"], a["k_role"], a["window"]
        rows = d if tcol == spec["column"] else d[d["linked"]]
        tg = rows.groupby(["season", "squad_id", "gameweek"]).agg(tx=(tcol, "sum"), tm=("minutes_played", "sum")).reset_index()
        x = rows.merge(tg, on=["season", "squad_id", "gameweek"])
        mates = (x["tx"] - x[tcol]) / ((x["tm"] - x["minutes_played"]).clip(lower=1) / 90)
        x["exp"] = mates.where(x["tm"] > x["minutes_played"], prior) * x["minutes_played"] / 90
        last = x.sort_values(["player_id", "season", "gameweek"]).groupby("player_id").tail(ROLE_GAMES).copy()
        now_lvl = last["player_id"].map(lambda pid: lv.get(cur_club.get(pid), None))
        step = (last["level"] - now_lvl.fillna(last["level"])).astype(int)
        m = a["league"]["role"]
        up, down = float(m.get("1", m.get(1, 1.0))), float(m.get("-1", m.get(-1, 1.0)))
        last["w"] = np.where(step > 0, up ** step.clip(lower=0), np.where(step < 0, down ** (-step).clip(lower=0), 1.0))
        last["num"] = last[tcol] * last["w"]
        agg = last.groupby("player_id").agg(num=("num", "sum"), den=("exp", "sum"))
        for pid, r in agg.iterrows():
            roles.setdefault(int(pid), {})[stat] = round(float((r["num"] + k * prior) / (r["den"] + k * prior)), 4)
        cur = rows[rows["season"] == "2627"]
        w = cur[cur["gameweek"] > last_gw - n]
        carry = {int(c): v for c, v in a["carry"].items()}
        for key, name in (("squad_id", "team"), ("opponent_id", "opp")):
            g = w.groupby(key).agg(x=(tcol, "sum"), m=("minutes_played", "sum"))
            for cid, r in g.iterrows():
                p0 = carry.get(int(cid), prior) if name == "team" else prior
                styles.setdefault(int(cid), {}).setdefault(stat, {})[name] = round(float((r["x"] + dm.K_TEAM * p0) / (r["m"] / 90 + dm.K_TEAM)), 4)
        for cid, p0 in carry.items():  # clubs with no games yet in the window keep their carried prior
            styles.setdefault(cid, {}).setdefault(stat, {}).setdefault("team", round(float(p0), 4))
        # team-mates' rate excluding the player (his current club, same window)
        club = w.groupby("squad_id").agg(x=(tcol, "sum"), m=("minutes_played", "sum"))
        own = w.groupby(["player_id", "squad_id"]).agg(x=(tcol, "sum"), m=("minutes_played", "sum"))
        for p in players:
            cid, pid = p["squadId"], p["id"]
            cx, cm = (club.at[cid, "x"], club.at[cid, "m"]) if cid in club.index else (0.0, 0.0)
            ox, om = (own.at[(pid, cid), "x"], own.at[(pid, cid), "m"]) if (pid, cid) in own.index else (0.0, 0.0)
            p0 = carry.get(cid, prior)
            mates_out.setdefault(pid, {})[stat] = round(float((max(cx - ox, 0) + dm.K_TEAM * p0) / (max(cm - om, 0) / 90 + dm.K_TEAM)), 4)
    # FotMob history features, named as in the model (fm_* = last 20 appearances, fm40_* = last 40); fm_pxg = penalty xG
    today = pd.Timestamp.now().normalize()
    hist = fmf.fotmob_history()
    pr = fmf.signal_rates(pd.DataFrame({"player_id": [p["id"] for p in players], "date": today}), k=3.0, hist=hist)
    pr = fmf.signal_rates(pr, k=3.0, n=40, hist=hist, prefix="fm40_")
    used = sorted({f for a in art.values() for f in a["features"] if f.startswith("fm")} | {"fm_pxg"})
    fm = {int(r["player_id"]): {f: round(float(r[f]), 4) for f in used} for _, r in pr.iterrows()}
    return roles, styles, fm, mates_out

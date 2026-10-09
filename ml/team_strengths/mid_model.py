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
FEATURE_SETS = {"G1 role": ["role"], "G2 + team": ["role", "team"], "G3 + opp": ["role", "team", "opp"], "G4 + odds/home": BASE}
WINDOWS = [4, 8, 12, 46]
K_ROLES = [3.0, 10.0, 30.0]
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
    d = fmf.signal_rates(mm.attach_dates(d), k=3.0)
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
    old = json.loads(am.MODEL_PATH.read_text(encoding="utf-8"))["positions"]["MID"]
    out = {}
    for stat, spec in SPEC.items():
        sets = dict(FEATURE_SETS)
        if spec["extra"]:
            sets["G5 + FotMob history"] = BASE + spec["extra"]
        rows = []
        for k in K_ROLES:
            for n in WINDOWS:
                x, prior, _ = build(d, stat, n, k, adj)
                tr = x[(x["season"] == "2526") & (x["gameweek"] <= 23)]
                va = x[(x["season"] == "2526") & (x["gameweek"] > 23)]
                for name, feats in sets.items():
                    if n != WINDOWS[0] and not any(f in ("team", "opp") for f in feats):
                        continue
                    f = fit(tr, feats, stat)
                    rows.append(dict(model=name, window=n, k=k, **score(va["cnt"].to_numpy(), predict(f, va, stat), f["r"])))
        res = pd.DataFrame(rows)
        res["complexity"] = res["model"].map(list(sets).index)
        best = res[res["loglik"] >= res["loglik"].max() - TOL].sort_values(["complexity", "loglik"], ascending=[True, False]).iloc[0]
        feats, n_best, k_best = sets[best["model"]], int(best["window"]), float(best["k"])
        print(f"\n=== MID {stat} (target {spec['target']}): validation best per model ===")
        print(res.sort_values("loglik", ascending=False).groupby("model").head(1).round(4).to_string(index=False))
        print(f"chosen: {best['model']}, window {n_best}, role shrinkage {k_best:g}")
        x, prior, carry = build(d, stat, n_best, k_best, adj)
        tv, te = x[x["season"] == "2526"], x[x["season"] == "2627"]
        f = fit(tv, feats, stat)
        test = score(te["cnt"].to_numpy(), predict(f, te, stat), f["r"])
        # the live v1 model on the same rows, for comparison
        xo = am.build(d, spec["column"], old[stat]["window"], old[stat]["prior"], old[stat]["k_role"], old[stat].get("club_w", 1.0))
        xo = xo[xo["minutes_played"] >= MIN_ROW]
        keys = ["player_id", "season", "gameweek", "opponent_id"]
        xo = xo.merge(te[keys], on=keys)
        xall = am.build(d, spec["column"], old[stat]["window"], old[stat]["prior"], old[stat]["k_role"], old[stat].get("club_w", 1.0))
        fo = am.fit(xall[(xall["season"] == "2526") & (xall["minutes_played"] >= MIN_ROW)], old[stat]["features"])
        test_v1 = score(xo["y"].to_numpy(), am.predict(fo, xo), fo["r"])
        print(f"TEST 2026/27 (n={len(te)}): v2 loglik {test['loglik']:.4f} mse {test['mse']:.4f} | v1 (live) loglik {test_v1['loglik']:.4f} mse {test_v1['mse']:.4f}"
              f" (n={len(xo)}) | conversion {f['scale']:.3f}")
        fin = fit(x, feats, stat)
        out[stat] = dict(column=spec["column"], target=spec["target"], pens=spec["pens"], model=best["model"], features=feats, window=n_best,
                         k_role=k_best, prior=prior, scale=fin["scale"], intercept=float(fin["model"].intercept_), coef=fin["model"].coef_.tolist(),
                         scaler_mean=fin["scaler"][0].tolist(), scaler_sd=fin["scaler"][1].tolist(), nb_r=fin["r"],
                         league=adj[stat], carry={str(c): v for c, v in carry.items()}, test=dict(v2=test, v1=test_v1, n=int(len(te))))
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
    roles, styles = {}, {}
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
    today = pd.Timestamp.now().normalize()
    pr = fmf.signal_rates(pd.DataFrame({"player_id": [p["id"] for p in players], "date": today}), k=3.0)
    fm = {int(r.player_id): dict(npxg=round(r.fm_npxg, 4), pxg=round(r.fm_pxg, 4), xa=round(r.fm_xa, 4), shots=round(r.fm_shots, 4),
                                 chances=round(r.fm_chances, 4)) for r in pr.itertuples()}
    return roles, styles, fm

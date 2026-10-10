"""Outfield stat models (midfielders, forwards, defenders), FotMob-based. Replaces attack_model.py (MID / FWD) and the
clearance / block / tackle and goal / assist parts of defence_model.py once trained.
python player_model.py [MID|FWD|DEF ...]     trains, validates, tests, writes models/<pos>_model.json

Per stat, a Poisson GLM with minutes exposure on standardised logs of candidate features:
  role      his rate relative to same-position team-mates in the same games, last ROLE_GAMES appearances, shrunk to 1;
            games in another league count x the measured step multiplier (league_steps.py, shrunk by sample size)
  team      his club's same-position rate, last N gameweeks this season (prior carried over from last season)
  mates     the same EXCLUDING him, so his share multiplies the environment he actually plays in
  opp       what same-position players do against this week's opponent
  lam_own / lam_opp / home    the match odds (expected goals each way) and home advantage
  fm_* / fm40_*  his FotMob per-90 history over the last 20 / 40 appearances (from 2024/25, any EFL club)
Goals and assists are modelled through their expected versions: the GLM predicts match npxG (xA) per 90, then
goals = c * npxG + his penalty-xG rate and assists = c * xA, c fitted on training data.
Protocol: choose window / role shrinkage / feature set on two rolling validation folds inside 2025/26 (log-likelihood of
the ACTUAL count, simplest candidate within TOL of the best), test once on 2026/27 against the incumbent, refit on all.
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
STEPS_PATH = HERE / "models" / "league_steps.json"
BASE = ["role", "team", "opp", "lam_own", "lam_opp", "home"]
BM = ["role", "mates", "opp", "lam_own", "lam_opp", "home"]
G = dict(column="goals_scored", target="m_npxg", fm="npxg", pens=True, pen_col="fm_penrate")  # + penalty share x league pen rate
A = dict(column="assists", target="m_xa", fm="xa", pens=False)
SPECS = {
    "MID": {"goals": G, "assists": A,
            "sot": dict(column="shots_on_target", target="shots_on_target", fm="sot", pens=False),
            "kp": dict(column="key_passes", target="key_passes", fm="kp", pens=False),
            "int": dict(column="interceptions", target="interceptions", fm="int", pens=False)},
    "FWD": {"goals": G, "assists": A,
            "sot": dict(column="shots_on_target", target="shots_on_target", fm="sot", pens=False),
            "kp": dict(column="key_passes", target="key_passes", fm="kp", pens=False)},
    "DEF": {"goals": G, "assists": A,
            "clr": dict(column="clearances", target="clearances", fm="clr", pens=False),
            "blk": dict(column="blocks", target="blocks", fm="blk", pens=False),
            "tkl": dict(column="tackles", target="tackles", fm="tkl", pens=False)},
}
# candidate feature sets, in order of preference when within TOL (incumbent structure first)
ATTACK_CANDS = {
    "goals": {"base": BASE, "mates": BM, "mates + npxG hist": BM + ["fm_npxg"], "mates + npxG hist40": BM + ["fm40_npxg"]},
    "assists": {"base + xA hist": BASE + ["fm_xa"], "mates + xA hist": BM + ["fm_xa"], "mates + xA hist40": BM + ["fm40_xa"]},
    "sot": {"base + shots/npxG hist": BASE + ["fm_shots", "fm_npxg"], "mates + shots/npxG hist": BM + ["fm_shots", "fm_npxg"],
            "mates + shots/npxG hist40": BM + ["fm40_shots", "fm40_npxg"]},
    "kp": {"base + chances/xA hist": BASE + ["fm_chances", "fm_xa"], "mates + chances/xA hist": BM + ["fm_chances", "fm_xa"],
           "mates + chances/xA hist40": BM + ["fm40_chances", "fm40_xa"],
           "mates + chances/xA hist40 + corner share": BM + ["fm40_chances", "fm40_xa", "fm_cornershare"]},
    "int": {"base": ["role", "team", "opp"], "mates": ["role", "mates", "opp"], "mates + int hist": ["role", "mates", "opp", "fm_int"],
            "mates + int hist40": ["role", "mates", "opp", "fm40_int"]},
}
DEF_CANDS = {
    "goals": {"base": BASE, "mates + npxG hist": BM + ["fm_npxg"], "mates + npxG hist40": BM + ["fm40_npxg"]},
    "assists": {"base": BASE, "mates + xA hist": BM + ["fm_xa"], "mates + xA hist40": BM + ["fm40_xa"]},
    **{st: {"base": BASE, "mates": BM, f"mates + {st} hist": BM + [f"fm_{st}"], f"mates + {st} hist40": BM + [f"fm40_{st}"]}
       for st in ("clr", "blk", "tkl")},
}
CANDS = {"MID": ATTACK_CANDS, "FWD": {k: v for k, v in ATTACK_CANDS.items() if k != "int"}, "DEF": DEF_CANDS}
WINDOWS = [8, 12, 46]
K_ROLES = [3.0, 10.0, 30.0]
FOLDS = [(17, 26), (26, 35)]  # rolling-origin validation inside 2025/26: (train up to GW a, validate a+1..b)
ROLE_GAMES, MIN_ROW, TOL = 20, 10, 5e-4
N0_ROLE, N0_TEAM = 30.0, 14.0  # shrink measured multipliers towards 1: weight n / (n + n0)
KEY = ["season", "player_id", "squad_id", "opponent_id", "is_home"]


def model_path(pos):
    return HERE / "models" / f"{pos.lower()}_model.json"


def levels():
    """{season tag: {squad id: 1 Championship / 2 League One / 3 League Two}}"""
    lv = {10: 1, 11: 2, 12: 3}
    old = json.load(open(am.REPO / "data" / "squads.json", encoding="utf-8"))
    cur = json.load(open(HERE / "cache" / "squads.json", encoding="utf-8"))
    return {"2526": {s["id"]: lv[s["competitionId"]] for s in old}, "2627": {s["id"]: lv[s["competitionId"]] for s in cur}}


def league_adjustments(pos):
    js = json.loads(STEPS_PATH.read_text(encoding="utf-8"))
    shrink = lambda e, n0: float(np.exp(e["n"] / (e["n"] + n0) * np.log(e["mult"]))) if e["mult"] == e["mult"] and e["mult"] > 0 else 1.0
    out = {}
    for stat, spec in SPECS[pos].items():
        role, team = js["role"][pos][spec["fm"]], js["team"][pos][spec["fm"]]
        out[stat] = dict(role={1: shrink(role["1"], N0_ROLE), -1: shrink(role["-1"], N0_ROLE)},
                         team={1: shrink(team["1"], N0_TEAM), -1: shrink(team["-1"], N0_TEAM)},
                         persist=float(js.get("persistence", {}).get(pos, {}).get(spec["fm"], 0.35)))
    return out


def load(pos):
    d = am.load(pos).drop(columns=["hid", "aid"])
    hist = fmf.fotmob_history()
    d = fmf.signal_rates(mm.attach_dates(d), k=3.0, hist=hist)
    d = fmf.signal_rates(d, k=3.0, n=40, hist=hist, prefix="fm40_")
    d = fmf.setpiece_features(d)  # penalty / corner shares, free kicks (exp_setpieces.py)
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
            x.at[ix[j], "role"] = (np.dot(w, y[lo:j]) + k * prior) / (e[lo:j].sum() + k * prior)
    return x


def carried_priors(d, col, prior, adj):
    """Each club's starting prior for 2026/27: league average x (last season's rate / average) ^ persistence x step mult."""
    lv = levels()
    s1 = d[d["season"] == "2526"].groupby("squad_id").agg(x=(col, "sum"), m=("minutes_played", "sum"))
    avg = s1["x"].sum() / (s1["m"].sum() / 90)
    rel = ((s1["x"] + dm.K_TEAM * avg) / (s1["m"] / 90 + dm.K_TEAM)) / avg
    carry = {}
    for cid, r in rel.items():
        step = lv["2526"].get(cid, 2) - lv["2627"].get(cid, lv["2526"].get(cid, 2))
        m = adj["team"].get(int(np.sign(step)), 1.0) if step else 1.0
        carry[int(cid)] = prior * (r ** adj["persist"]) * m
    return carry


def style_tables(d, col, n, prior, adj):
    """Club style (team) and opponent style per (season, club, gameweek) from earlier gameweeks this season, shrunk with
    K_TEAM pseudo-90s towards the club's carried-over prior (team, 2026/27) or the league average."""
    team, opp = dm.style_tables(d, col, n, prior)
    carry = carried_priors(d, col, prior, adj)
    g = d[d["season"] == "2627"].groupby(["squad_id", "gameweek"]).agg(x=(col, "sum"), m90=("minutes_played", lambda s: s.sum() / 90)).reset_index()
    rows, gws = [], np.arange(1, 47)
    for cid, x in g.groupby("squad_id"):
        xs = pd.Series(0.0, index=gws).add(x.set_index("gameweek")["x"], fill_value=0)
        ms = pd.Series(0.0, index=gws).add(x.set_index("gameweek")["m90"], fill_value=0)
        sx, sm = xs.shift(1).rolling(n, min_periods=1).sum().fillna(0), ms.shift(1).rolling(n, min_periods=1).sum().fillna(0)
        p = carry.get(int(cid), prior)
        rows.append(pd.DataFrame({"season": "2627", "squad_id": cid, "gameweek": gws, "team": (sx + dm.K_TEAM * p) / (sm + dm.K_TEAM)}))
    if rows:
        team = pd.concat([team[team["season"] != "2627"], pd.concat(rows, ignore_index=True)], ignore_index=True)
    return team, opp, carry


def mates_table(d, col, n, prior, carry):
    """Per (season, player, club, gameweek): his club's same-position rate per 90 over the previous n gameweeks of this
    season EXCLUDING his own minutes and output, shrunk with K_TEAM pseudo-90s to the club's prior."""
    out, gws = [], np.arange(1, 47)
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
        p0 = carry.get(int(cid), prior) if season == "2627" else prior
        val = ((cx - ox).clip(lower=0) + dm.K_TEAM * p0) / ((cm - om).clip(lower=0) / 90 + dm.K_TEAM)
        out.append(pd.DataFrame({"season": season, "player_id": pid, "squad_id": cid, "gameweek": gws, "mates": val.to_numpy()}))
    return pd.concat(out, ignore_index=True)


def build(d, spec, n, k, adj):
    tcol = spec["target"]
    rows = d if tcol == spec["column"] else d[d["linked"]]
    tr = rows[(rows["season"] == "2526") & (rows["gameweek"] <= 23)]
    prior = float(tr[tcol].sum() / (tr["minutes_played"].sum() / 90))
    team, opp, carry = style_tables(rows, tcol, n, prior, adj)
    x = role_table(rows, tcol, prior, k, adj["role"])
    x = x.merge(team, on=["season", "squad_id", "gameweek"], how="left").merge(opp, on=["season", "opponent_id", "gameweek"], how="left")
    x["team"], x["opp"] = x["team"].fillna(prior), x["opp"].fillna(prior)
    x = x.merge(mates_table(rows, tcol, n, prior, carry), on=["season", "player_id", "squad_id", "gameweek"], how="left")
    x["mates"] = x["mates"].fillna(prior)
    x["y"], x["cnt"], x["t"] = x[tcol], x[spec["column"]], x["minutes_played"] / 90
    return x[x["minutes_played"] >= MIN_ROW], prior, carry


def fit(x, feats, spec):
    X, sc = am.design(x, feats)
    m = PoissonRegressor(alpha=1e-4, max_iter=3000).fit(X, x["y"] / x["t"], sample_weight=x["t"])
    raw = m.predict(X) * x["t"]
    pen = (x[spec.get("pen_col", "fm_pxg")] * x["t"]).to_numpy() if spec["pens"] else 0.0
    scale = 1.0 if spec["target"] == spec["column"] else float((x["cnt"].sum() - np.sum(pen)) / raw.sum())
    mu = scale * raw + pen
    return dict(model=m, scaler=sc, feats=feats, scale=scale, r=nb_r(x["cnt"].to_numpy(), np.asarray(mu)))


def predict(f, x, spec):
    raw = f["model"].predict(am.design(x, f["feats"], f["scaler"])[0]) * x["t"].to_numpy()
    pen = (x[spec.get("pen_col", "fm_pxg")] * x["t"]).to_numpy() if spec["pens"] else 0.0
    return f["scale"] * raw + pen


def score(y, mu, r):
    mu = np.clip(mu, 1e-6, None)
    return dict(loglik=nb_loglik(y, mu, r), mse=float(((y - mu) ** 2).mean()), mae=float(np.abs(y - mu).mean()))


def incumbent_test(d, pos, stat, spec, te, prev):
    """The model live before this run, scored on the same 2026/27 rows."""
    keys = ["player_id", "season", "gameweek", "opponent_id"]
    if prev.get(stat):  # an earlier run of this script
        return None  # compared inside main (same infrastructure)
    if pos in ("MID", "FWD"):  # attack_model.py v1
        old = json.loads(am.MODEL_PATH.read_text(encoding="utf-8"))["positions"][pos].get(stat)
        if not old:
            return None
        xall = am.build(d, spec["column"], old["window"], old["prior"], old["k_role"], old.get("club_w", 1.0))
        fo = am.fit(xall[(xall["season"] == "2526") & (xall["minutes_played"] >= MIN_ROW)], old["features"])
        xo = xall[xall["minutes_played"] >= MIN_ROW].merge(te[keys], on=keys)
        return score(xo["y"].to_numpy(), am.predict(fo, xo), fo["r"])
    if stat in ("goals", "assists"):  # defence_model / defenders.py: his rate over 40 appearances x team expected goals
        col = spec["column"]
        x = d.sort_values(["player_id", "season", "gameweek"]).copy()
        tr = x[x["season"] == "2526"]
        prior = tr[col].sum() / (tr["minutes_played"].sum() / 90)
        g = x.groupby("player_id")
        num = g[col].transform(lambda s: s.shift(1).rolling(dm.RATE_GAMES, min_periods=1).sum()).fillna(0)
        m90 = g["minutes_played"].transform(lambda s: s.shift(1).rolling(dm.RATE_GAMES, min_periods=1).sum()).fillna(0) / 90
        x["mu"] = (num + dm.K_RATE * prior) / (m90 + dm.K_RATE) * x["lam_own"] / tr["lam_own"].mean() * x["minutes_played"] / 90
        x = x[x["minutes_played"] >= MIN_ROW]
        r = nb_r(x.loc[x["season"] == "2526", col].to_numpy(), x.loc[x["season"] == "2526", "mu"].to_numpy())
        xo = x.merge(te[keys], on=keys)
        return score(xo[col].to_numpy(), xo["mu"].to_numpy(), r)
    return None  # DEF clearances / blocks / tackles: the "base" candidate has defence_model.py's structure


def main(pos):
    d = load(pos)
    path = model_path(pos)
    prev = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    out = {}
    for stat, spec in SPECS[pos].items():
        adj = league_adjustments(pos)[stat]
        cands, rows, built = CANDS[pos][stat], [], {}
        for k in K_ROLES:
            for n in WINDOWS:
                x, prior, carry = build(d, spec, n, k, adj)
                built[(n, k)] = (x, prior, carry)
                s25 = x[x["season"] == "2526"]
                for name, feats in cands.items():
                    ll, mse = [], []
                    for a, b in FOLDS:
                        tr, va = s25[s25["gameweek"] <= a], s25[(s25["gameweek"] > a) & (s25["gameweek"] <= b)]
                        f = fit(tr, feats, spec)
                        sc = score(va["cnt"].to_numpy(), predict(f, va, spec), f["r"])
                        ll.append(sc["loglik"]); mse.append(sc["mse"])
                    rows.append(dict(model=name, window=n, k=k, loglik=float(np.mean(ll)), mse=float(np.mean(mse)), fold_ll=[round(v, 4) for v in ll]))
        res = pd.DataFrame(rows)
        res["pref"] = res["model"].map(list(cands).index)
        best = res[res["loglik"] >= res["loglik"].max() - TOL].sort_values(["pref", "loglik"], ascending=[True, False]).iloc[0]
        print(f"\n=== {pos} {stat} (target {spec['target']}): mean validation over {len(FOLDS)} folds, best per candidate ===")
        print(res.sort_values("loglik", ascending=False).groupby("model").head(1)[["model", "window", "k", "loglik", "mse", "fold_ll"]].round(4).to_string(index=False))
        feats, n_best, k_best = cands[best["model"]], int(best["window"]), float(best["k"])
        print(f"chosen: {best['model']}, window {n_best}, role shrinkage {k_best:g}")
        x, prior, carry = built[(n_best, k_best)]
        tv, te = x[x["season"] == "2526"], x[x["season"] == "2627"]
        f = fit(tv, feats, spec)
        test = score(te["cnt"].to_numpy(), predict(f, te, spec), f["r"])
        # incumbent on the same test rows: previous run of this script, else the older model, else the base candidate
        inc_name, test_inc = None, None
        if prev.get(stat):
            p = prev[stat]
            xi = built.get((p["window"], p["k_role"])) or build(d, spec, p["window"], p["k_role"], adj)
            fi = fit(xi[0][xi[0]["season"] == "2526"], p["features"], spec)
            tei = xi[0][xi[0]["season"] == "2627"]
            inc_name, test_inc = "previous run", score(tei["cnt"].to_numpy(), predict(fi, tei, spec), fi["r"])
        else:
            test_inc = incumbent_test(d, pos, stat, spec, te, prev)
            inc_name = "old model" if test_inc else None
            if test_inc is None:
                bx = built[(n_best, k_best)][0]
                fb = fit(bx[bx["season"] == "2526"], list(cands.values())[0], spec)
                teb = bx[bx["season"] == "2627"]
                inc_name, test_inc = "base structure", score(teb["cnt"].to_numpy(), predict(fb, teb, spec), fb["r"])
        print(f"TEST 2026/27 (once; n={len(te)}): chosen loglik {test['loglik']:.4f} mse {test['mse']:.4f} | {inc_name} loglik "
              f"{test_inc['loglik']:.4f} mse {test_inc['mse']:.4f} | conversion {f['scale']:.3f}")
        fin = fit(x, feats, spec)
        out[stat] = dict(column=spec["column"], target=spec["target"], pens=spec["pens"], pen_col=spec.get("pen_col", "fm_pxg"),
                         model=best["model"], features=feats, window=n_best,
                         k_role=k_best, prior=prior, scale=fin["scale"], intercept=float(fin["model"].intercept_), coef=fin["model"].coef_.tolist(),
                         scaler_mean=fin["scaler"][0].tolist(), scaler_sd=fin["scaler"][1].tolist(), nb_r=fin["r"],
                         league=adj, carry={str(c): v for c, v in carry.items()}, validation=res.drop(columns=["pref"]).to_dict("records"),
                         test=dict(v2=test, v1=test_inc, v1_name=inc_name, n=int(len(te))))
        print("final coefficients", {kk: round(v, 3) for kk, v in zip(feats, fin["model"].coef_)})
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\nsaved", path)


# ---------- live inputs for the site (attackers.py / defenders.py) ----------
def live_inputs(players, art, pos, d=None, as_of=None):
    """Current role / team-mates rate per stat (relative to his CURRENT club and league), club / opponent style per stat
    (this season, carried-over prior), FotMob history rates as of today. Returns roles, styles, fm, mates.
    For a backtest pass d already cut to games before the gameweek and as_of = that gameweek's first match date."""
    d = load(pos) if d is None else d
    lv = levels()["2627"]
    plist = [p for p in players if p["position"] == pos]
    cur_club = {p["id"]: p["squadId"] for p in plist}
    roles, styles, mates_out = {}, {}, {}
    cur_rows = d.loc[d["season"] == "2627", "gameweek"]
    last_gw = int(cur_rows.max()) if len(cur_rows) else 0
    for stat, a in art.items():
        spec = SPECS[pos][stat]
        tcol, prior, k, n = spec["target"], a["prior"], a["k_role"], a["window"]
        rows = d if tcol == spec["column"] else d[d["linked"]]
        tg = rows.groupby(["season", "squad_id", "gameweek"]).agg(tx=(tcol, "sum"), tm=("minutes_played", "sum")).reset_index()
        x = rows.merge(tg, on=["season", "squad_id", "gameweek"])
        others = (x["tx"] - x[tcol]) / ((x["tm"] - x["minutes_played"]).clip(lower=1) / 90)
        x["exp"] = others.where(x["tm"] > x["minutes_played"], prior) * x["minutes_played"] / 90
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
        club = w.groupby("squad_id").agg(x=(tcol, "sum"), m=("minutes_played", "sum"))
        own = w.groupby(["player_id", "squad_id"]).agg(x=(tcol, "sum"), m=("minutes_played", "sum"))
        for p in plist:
            cid, pid = p["squadId"], p["id"]
            cx, cm = (club.at[cid, "x"], club.at[cid, "m"]) if cid in club.index else (0.0, 0.0)
            ox, om = (own.at[(pid, cid), "x"], own.at[(pid, cid), "m"]) if (pid, cid) in own.index else (0.0, 0.0)
            p0 = carry.get(cid, prior)
            mates_out.setdefault(pid, {})[stat] = round(float((max(cx - ox, 0) + dm.K_TEAM * p0) / (max(cm - om, 0) / 90 + dm.K_TEAM)), 4)
    # FotMob history features, named as in the model (fm_* = last 20 appearances, fm40_* = last 40); fm_pxg = penalty xG
    today = pd.Timestamp.now().normalize() if as_of is None else pd.Timestamp(as_of).normalize()
    hist = fmf.fotmob_history()
    pr = fmf.signal_rates(pd.DataFrame({"player_id": [p["id"] for p in plist], "date": today}), k=3.0, hist=hist)
    pr = fmf.signal_rates(pr, k=3.0, n=40, hist=hist, prefix="fm40_")
    pr = fmf.setpiece_features(pr)
    used = sorted({f for a in art.values() for f in a["features"] if f.startswith("fm")} | {a.get("pen_col", "fm_pxg") for a in art.values()})
    fm = {int(r["player_id"]): {f: round(float(r[f]), 4) for f in used} for _, r in pr.iterrows()}
    tags = {int(r["player_id"]): fmf.setpiece_tags(r) for _, r in pr.iterrows()}
    return roles, styles, fm, mates_out, tags


def refit(d, pos, art, before_gw, built=None):
    """The artifact's configuration (features, window, role shrinkage) refitted on 2025/26 + 2026/27 games before
    `before_gw` only (for the backtest). built: optional cache {stat: build output} of the full feature table."""
    out = {}
    for stat, a in art.items():
        spec = SPECS[pos][stat]
        if built is not None and stat in built:
            x, prior, carry = built[stat]
        else:
            x, prior, carry = build(d, spec, a["window"], a["k_role"], league_adjustments(pos)[stat])
            if built is not None:
                built[stat] = (x, prior, carry)
        tr = x[(x["season"] == "2526") | (x["gameweek"] < before_gw)]
        f = fit(tr, a["features"], spec)
        out[stat] = dict(a, scale=f["scale"], intercept=float(f["model"].intercept_), coef=f["model"].coef_.tolist(),
                         scaler_mean=f["scaler"][0].tolist(), scaler_sd=f["scaler"][1].tolist(), nb_r=f["r"])
    return out


def fm_defaults(fm):
    """FotMob feature values for a player with no history: what the shrinkage tends to (league rates / typical shares)."""
    pri = fmf.league_priors(fmf.fotmob_history())
    sp = fmf.setpiece_features(pd.DataFrame({"player_id": [-1], "date": [pd.Timestamp.now().normalize()]})).iloc[0]
    keys = next(iter(fm.values())).keys() if fm else []
    return {f: round(float(sp[f]) if f in sp.index else pri[f.split("_", 1)[1]], 4) for f in keys}


if __name__ == "__main__":
    for position in (sys.argv[1:] or ["MID", "FWD", "DEF"]):
        main(position.upper())

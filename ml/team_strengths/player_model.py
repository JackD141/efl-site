"""Outfield stat models (midfielders, forwards, defenders), FotMob-based. Replaces attack_model.py (MID / FWD) and the
clearance / block / tackle and goal / assist parts of defence_model.py once trained.
python player_model.py [MID|FWD|DEF ...]            refits the coefficients of the CHOSEN models on everything (weekly)
python player_model.py --select [MID|FWD|DEF ...]   re-makes the choices on the validation folds (a deliberate step: log
                                                    the outcome in docs/ml-methodology.md), then refits
Both write models/<pos>_model.json.

Per stat, a Poisson GLM with minutes exposure on standardised logs of candidate features:
  role      his rate relative to same-position team-mates in the same games, last ROLE_GAMES appearances, shrunk to 1;
            games in another league count x the measured step multiplier (league_steps.py, shrunk by sample size)
  team      his club's same-position rate, last N gameweeks this season (prior carried over from last season)
  mates     the same EXCLUDING him, so his share multiplies the environment he actually plays in
  opp       what same-position players do against this week's opponent
  lam_own / lam_opp / home    the match odds (expected goals each way) and home advantage
  fm_* / fm40_*  his FotMob per-90 history over the last 20 / 40 appearances (from 2024/25, any EFL club)
  fm40h20_*      the same over the last 40 with recent games weighted more (half-life 20 appearances; exp_recency.py)
Goals and assists are modelled through their expected versions: the GLM predicts match npxG (xA) per 90, then
goals = c * npxG + his penalty-xG rate and assists = c * xA, c fitted on training data.
Protocol (docs/ml-methodology.md section 3): choose feature set / window / role shrinkage on two rolling validation
folds inside 2025/26 by the log-likelihood of the ACTUAL count, with the one-standard-error rule: the simplest
configuration whose validation score is within max(TOL, one standard error) of the best one, the standard error being
that of the paired difference, with whole gameweeks as the independent units. The choices are then FROZEN in the model
file; the weekly run only refits coefficients. 2026/27 is a monitor (looked at every week), never used to choose.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln
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
# every candidate on the 40-game history also gets a recency-weighted version, LEAST preferred (chosen only if it beats the
# equal-weight version by more than TOL on the validation folds; exp_recency.py found that for MID key passes, DEF clearances)
RECENT = ("fm40_", "fm40h20_")
for _c in (ATTACK_CANDS, DEF_CANDS):
    for _st, _cs in _c.items():
        for _name, _f in list(_cs.items()):
            if any(x.startswith(RECENT[0]) for x in _f):
                _cs[_name + " (recency-weighted)"] = [x.replace(*RECENT) for x in _f]
CANDS = {"MID": ATTACK_CANDS, "FWD": {k: v for k, v in ATTACK_CANDS.items() if k != "int"}, "DEF": DEF_CANDS}
WINDOWS = [8, 12, 46]
K_ROLES = [3.0, 10.0, 30.0]
DEFAULT_SETTING = (12, 10.0)  # the plain (window, role shrinkage): kept unless another setting is better beyond the noise
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
    d = fmf.signal_rates(d, k=3.0, n=40, hist=hist, prefix="fm40h20_", half_life=20)
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
    def prev_window(keys):
        """(groups, X, M): for each group and gameweek 1..46, the sum of the stat / the minutes over the previous n
        gameweeks (not the current one). Dense arrays + cumulative sums instead of a Python loop per player."""
        t = d.groupby(keys + ["gameweek"]).agg(x=(col, "sum"), m=("minutes_played", "sum")).reset_index()
        codes, groups = pd.factorize(pd.MultiIndex.from_frame(t[keys]))
        gw = t["gameweek"].to_numpy().astype(int)
        out = []
        for v in ("x", "m"):
            a = np.zeros((len(groups), 47))
            np.add.at(a, (codes, gw), t[v].to_numpy(float))
            c = np.cumsum(a, axis=1)  # c[:, g] = gameweeks 1..g (column 0 is empty)
            out.append(np.column_stack([c[:, g - 1] - c[:, max(g - 1 - n, 0)] for g in range(1, 47)]))
        return groups, out[0], out[1]
    clubs, cx, cm = prev_window(["season", "squad_id"])
    own, ox, om = prev_window(["season", "player_id", "squad_id"])
    club_row = {k: i for i, k in enumerate(clubs)}
    ci = np.array([club_row[(s, c)] for s, _, c in own])
    p0 = np.array([carry.get(int(c), prior) if s == "2627" else prior for s, _, c in own])[:, None]
    val = (np.clip(cx[ci] - ox, 0, None) + dm.K_TEAM * p0) / (np.clip(cm[ci] - om, 0, None) / 90 + dm.K_TEAM)
    rep = lambda j: np.repeat(np.array([k[j] for k in own]), 46)
    return pd.DataFrame({"season": rep(0), "player_id": rep(1), "squad_id": rep(2), "gameweek": np.tile(np.arange(1, 47), len(own)), "mates": val.ravel()})


def build(d, spec, n, k, adj):
    tcol = spec["target"]
    rows = d if tcol == spec["column"] else d[d["linked"]]
    tr = rows[(rows["season"] == "2526") & (rows["gameweek"] <= FOLDS[0][0])]  # the first fold's training games: no validation game in the league rate
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


def row_loglik(y, mu, r):
    """Negative-binomial log-likelihood of each row."""
    mu = np.clip(np.asarray(mu, float), 1e-6, None)
    p = r / (r + mu)
    return gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r * np.log(p) + y * np.log1p(-p)


def validate(x, feats, spec):
    """Validation log-likelihood of one configuration over the rolling folds: (mean, {gameweek: sum over its rows},
    {gameweek: rows}, mean per fold, MSE). Gameweeks are the units for the standard errors (rows of a gameweek share
    fixtures, odds and team-mates)."""
    s25 = x[x["season"] == "2526"]
    sums, counts, fold_ll, mse = {}, {}, [], []
    for a, b in FOLDS:
        tr, va = s25[s25["gameweek"] <= a], s25[(s25["gameweek"] > a) & (s25["gameweek"] <= b)]
        f = fit(tr, feats, spec)
        mu = predict(f, va, spec)
        ll = row_loglik(va["cnt"].to_numpy(), mu, f["r"])
        fold_ll.append(float(ll.mean()))
        mse.append(float(((va["cnt"].to_numpy() - mu) ** 2).mean()))
        for g, v in pd.Series(ll).groupby(va["gameweek"].to_numpy()):
            sums[int(g)], counts[int(g)] = float(v.sum()), int(len(v))
    n = sum(counts.values())
    return sum(sums.values()) / n, sums, counts, fold_ll, float(np.mean(mse))


def paired_se(sa, sb, counts):
    """Standard error of the difference in mean validation log-likelihood between two configurations scored on the same
    rows, treating gameweeks as independent clusters (ratio-estimator linearisation)."""
    gws = sorted(counts)
    d = np.array([sa[g] - sb[g] for g in gws])
    n = np.array([counts[g] for g in gws], float)
    m = d.sum() / n.sum()
    return float(np.sqrt(len(gws) / (len(gws) - 1) * ((d - m * n) ** 2).sum()) / n.sum())


def select(d, pos, stat, verbose=True):
    """Choose (candidate, window, role shrinkage) for one stat by the one-standard-error rule. Order of preference:
    candidates as listed in CANDS (simplest / incumbent structure first); within a candidate the DEFAULT_SETTING, then
    its best setting. The first configuration in that order within max(TOL, paired standard error) of the best wins."""
    spec, adj, cands = SPECS[pos][stat], league_adjustments(pos)[stat], CANDS[pos][stat]
    res = {}
    for k in K_ROLES:
        for n in WINDOWS:
            x, _, _ = build(d, spec, n, k, adj)
            for name, feats in cands.items():
                res[(name, n, k)] = validate(x, feats, spec)
    best = max(res, key=lambda c: res[c][0])
    counts = res[best][2]
    gap = {c: (res[best][0] - v[0], paired_se(res[best][1], v[1], counts)) for c, v in res.items()}
    ok = lambda c: gap[c][0] <= max(TOL, gap[c][1])
    chosen = None
    for name in cands:
        default = (name, *DEFAULT_SETTING)
        top = max((c for c in res if c[0] == name), key=lambda c: res[c][0])
        chosen = default if ok(default) else top if ok(top) else None
        if chosen:
            break
    table = [dict(model=c[0], window=c[1], k=c[2], loglik=round(v[0], 5), behind_best=round(gap[c][0], 5), se=round(gap[c][1], 5),
                  mse=round(v[4], 5), fold_ll=[round(f, 4) for f in v[3]]) for c, v in res.items()]
    if verbose:
        print(f"\n=== {pos} {stat} (target {spec['target']}): validation over {len(FOLDS)} folds, {sum(counts.values())} rows in {len(counts)} gameweeks ===")
        print(f"best: {best[0]} (window {best[1]}, k {best[2]:g}) {res[best][0]:.5f}")
        for name in cands:
            for c in ((name, *DEFAULT_SETTING), max((c for c in res if c[0] == name), key=lambda c: res[c][0])):
                print(f"   {name:60s} window {c[1]:2d} k {c[2]:<4g} behind best {gap[c][0]:.5f} (se {gap[c][1]:.5f}) {'within the noise' if ok(c) else ''}")
        print(f"chosen: {chosen[0]}, window {chosen[1]}, role shrinkage {chosen[2]:g}")
    return dict(model=chosen[0], window=int(chosen[1]), k_role=float(chosen[2]), rule=f"one-standard-error rule, floor {TOL}",
                best=dict(model=best[0], window=int(best[1]), k=float(best[2]), loglik=round(res[best][0], 5)),
                behind_best=round(gap[chosen][0], 5), se=round(gap[chosen][1], 5)), table


def main(pos, reselect=False):
    """Refit the position's stat models on everything. The choice of features / window / shrinkage is read from the
    existing model file (frozen) unless reselect=True or the stat has no choice yet."""
    d = load(pos)
    path = model_path(pos)
    prev = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    out = {}
    for stat, spec in SPECS[pos].items():
        adj = league_adjustments(pos)[stat]
        cands, old = CANDS[pos][stat], prev.get(stat)
        if reselect or not old or "selection" not in old or old["model"] not in cands:
            choice, table = select(d, pos, stat)
            if old and (old["model"], old["window"], old["k_role"]) != (choice["model"], choice["window"], choice["k_role"]):
                print(f"CHANGED from {old['model']}, window {old['window']}, role shrinkage {old['k_role']:g}: log it in docs/ml-methodology.md")
            choice["selected_on"] = pd.Timestamp.now().strftime("%Y-%m-%d")
        else:
            choice, table = old["selection"], old["validation"]
        name, n_best, k_best = choice["model"], choice["window"], choice["k_role"]
        feats = cands[name]
        x, prior, carry = build(d, spec, n_best, k_best, adj)
        tv, te = x[x["season"] == "2526"], x[x["season"] == "2627"]
        # monitor: the chosen configuration and the plainest candidate, both fitted on 2025/26 only, on 2026/27 to date
        f, fb = fit(tv, feats, spec), fit(tv, list(cands.values())[0], spec)
        mon, mon_base = score(te["cnt"].to_numpy(), predict(f, te, spec), f["r"]), score(te["cnt"].to_numpy(), predict(fb, te, spec), fb["r"])
        ratio = float(te["cnt"].sum() / predict(f, te, spec).sum())
        print(f"{pos} {stat}: {name} (window {n_best}, k {k_best:g}) | 2026/27 monitor, fitted on 2025/26 (n={len(te)}): log-lik {mon['loglik']:.4f} "
              f"(plainest candidate {mon_base['loglik']:.4f}), actual / predicted {ratio:.3f}")
        fin = fit(x, feats, spec)
        out[stat] = dict(column=spec["column"], target=spec["target"], pens=spec["pens"], pen_col=spec.get("pen_col", "fm_pxg"),
                         model=name, features=feats, window=n_best,
                         k_role=k_best, prior=prior, scale=fin["scale"], intercept=float(fin["model"].intercept_), coef=fin["model"].coef_.tolist(),
                         scaler_mean=fin["scaler"][0].tolist(), scaler_sd=fin["scaler"][1].tolist(), nb_r=fin["r"],
                         league=adj, carry={str(c): v for c, v in carry.items()}, selection=choice, validation=table,
                         monitor=dict(chosen=mon, plainest=mon_base, ratio=ratio, n=int(len(te))))
        print("   coefficients", {kk: round(v, 3) for kk, v in zip(feats, fin["model"].coef_)}, f"conversion {fin['scale']:.3f}")
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("saved", path)


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
        tg = rows.groupby(am.MATCH).agg(tx=(tcol, "sum"), tm=("minutes_played", "sum")).reset_index()
        x = rows.merge(tg, on=am.MATCH)
        others = (x["tx"] - x[tcol]) / ((x["tm"] - x["minutes_played"]).clip(lower=1) / 90)
        x["exp"] = others.where(x["tm"] > x["minutes_played"], prior) * x["minutes_played"] / 90
        last = x.sort_values(["player_id", "season", "date", "gameweek"] if "date" in x else ["player_id", "season", "gameweek"]).groupby("player_id").tail(ROLE_GAMES).copy()
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
    pr = fmf.signal_rates(pr, k=3.0, n=40, hist=hist, prefix="fm40h20_", half_life=20)
    pr = fmf.setpiece_features(pr)
    used = sorted({f for a in art.values() for f in a["features"] if f.startswith("fm")} | {a.get("pen_col", "fm_pxg") for a in art.values()})
    fm = {int(r["player_id"]): {f: round(float(r[f]), 4) for f in used} for _, r in pr.iterrows()}
    sph = fmf.setpiece_history()
    last_team, pen_taker = fmf.latest_team_and_pen_takers(sph, today)
    tags = {int(r["player_id"]): fmf.setpiece_tags(r, pen_taker.get(last_team.get(int(r["player_id"]))) == int(r["player_id"]))
            for _, r in pr.iterrows()}
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


_FM_DEFAULTS = {}


def fm_defaults(fm):
    """FotMob feature values for a player with no history: what the shrinkage tends to (league rates / typical shares).
    Constants of completed seasons, so worked out once per set of features."""
    keys = tuple(next(iter(fm.values())).keys()) if fm else ()
    if keys not in _FM_DEFAULTS:
        pri = fmf.league_priors(fmf.fotmob_history())
        sp = fmf.setpiece_features(pd.DataFrame({"player_id": [-1], "date": [pd.Timestamp.now().normalize()]})).iloc[0]
        _FM_DEFAULTS[keys] = {f: round(float(sp[f]) if f in sp.index else pri[f.split("_", 1)[1]], 4) for f in keys}
    return dict(_FM_DEFAULTS[keys])


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for position in (args or ["MID", "FWD", "DEF"]):
        main(position.upper(), reselect="--select" in sys.argv)

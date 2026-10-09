"""Defender clearances / blocks / tackles model.  python defence_model.py  (trains, evaluates, writes models/defence_model.json)

Only defenders have these stats in the fantasy feed, so a club's "style" is what its defenders do, and an opponent's
style is what defenders facing it do. For a defender playing the whole game, expected count of stat X:

  log mu = b0 + b1 log(role) + b2 log(team style) + b3 log(opponent style) + b4 log(lam_opp) + b5 log(lam_own) + b6 home

  role          the player's X relative to his team-mates' X per 90 in the same games, over his last ROLE_GAMES
                appearances at any club (so it travels with him when he moves), shrunk towards 1
  team style    his current club's defenders' X per defender-90 over the last N gameweeks (time window, this season)
  opp. style    X per defender-90 made by defenders facing this week's opponent over the last N gameweeks
  lam_opp/own   expected goals implied by the match odds (pressure on the defence / game state)
Every rolling quantity uses only earlier gameweeks and is shrunk towards the league average, so early weeks are sane.

Protocol (time-based, no leakage): train 2025/26 GW1-23, choose model and window N on 2025/26 GW24-35, test once on
2026/27 to date, then refit on everything. Candidates per stat: player's own recent per-90 rate (the usual baseline),
then nested Poisson GLMs adding role, team style, opponent style, odds; plus gradient boosting. The simplest model within
TOL of the best validation log-likelihood is kept. Counts are over-dispersed: negative binomial spread per stat.
"""
import glob
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

from per_season import lam_for
from predict_gw import resolver
from saves_model import nb_loglik, nb_r, poisson_dev
from strengths import DIVS

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
REPO = HERE.parents[1]
MODEL_PATH = HERE / "models" / "defence_model.json"
STATS = {"clr": ("clearances", 4), "blk": ("blocks", 2), "tkl": ("tackles", 2)}  # stat -> (column, units per point)
WINDOWS = [4, 8, 12, 46]  # gameweeks; 46 = season to date
ROLE_GAMES = 20
K_TEAM = 12.0  # pseudo defender-90s of league average in team / opponent style
K_ROLE = 3.0   # pseudo games at role 1.0
TOL = 5e-4
MIN_FULL = 85  # rows used for fitting: defender played (about) the whole game
FEATURE_SETS = {
    "G1 role": ["role"],
    "G2 role + team style": ["role", "team"],
    "G3 + opponent style": ["role", "team", "opp"],
    "G4 + odds + home": ["role", "team", "opp", "lam_opp", "lam_own", "home"],
}


def load_fantasy():
    frames = []
    for season, tag in (("2025_26", "2526"), ("2026_27", "2627")):
        for f in glob.glob(str(REPO / "data" / season / "player_stats_gw*.csv")):
            try:
                frames.append(pd.read_csv(f).assign(season=tag))
            except pd.errors.EmptyDataError:
                pass
    d = pd.concat(frames, ignore_index=True)
    return d[(d["position"] == "DEF") & (d["minutes_played"] > 0)].copy()


def attach_odds(d):
    """Closing-odds expected goals for every fantasy row (team = squad_id, opponent = opponent_id)."""
    cur = {s["id"]: s for s in json.load(open(HERE / "cache" / "squads.json", encoding="utf-8"))}
    old = {s["id"]: s for s in json.load(open(REPO / "data" / "squads.json", encoding="utf-8"))}
    parts = []
    for tag, squads in (("2526", old), ("2627", cur)):
        res = resolver(squads)
        for div in DIVS:
            m = lam_for(div, tag).dropna(subset=["lamH"])
            parts.append(pd.DataFrame({"season": tag, "hid": m["HomeTeam"].map(res), "aid": m["AwayTeam"].map(res),
                                       "lamH": m["lamH"], "lamA": m["lamA"]}))
    odds = pd.concat(parts, ignore_index=True)
    home = d["is_home"] == "H"
    d["hid"] = np.where(home, d["squad_id"], d["opponent_id"])
    d["aid"] = np.where(home, d["opponent_id"], d["squad_id"])
    d = d.merge(odds, on=["season", "hid", "aid"], how="left")
    d["home"] = home.astype(int).to_numpy()
    d["lam_own"] = np.where(d["home"] == 1, d["lamH"], d["lamA"])
    d["lam_opp"] = np.where(d["home"] == 1, d["lamA"], d["lamH"])
    return d


def style_tables(d, col, n, prior):
    """Per (season, club, gameweek): club style and opponent style as of BEFORE that gameweek (window n gameweeks)."""
    def rolling(key):
        g = d.groupby(["season", key, "gameweek"]).agg(x=(col, "sum"), m90=("minutes_played", lambda s: s.sum() / 90)).reset_index()
        out = []
        for (season, k), x in g.groupby(["season", key]):
            gws = np.arange(1, 47)
            xs = pd.Series(0.0, index=gws).add(x.set_index("gameweek")["x"], fill_value=0)
            ms = pd.Series(0.0, index=gws).add(x.set_index("gameweek")["m90"], fill_value=0)
            sx = xs.shift(1).rolling(n, min_periods=1).sum().fillna(0)
            sm = ms.shift(1).rolling(n, min_periods=1).sum().fillna(0)
            out.append(pd.DataFrame({"season": season, key: k, "gameweek": gws, "val": (sx + K_TEAM * prior) / (sm + K_TEAM)}))
        return pd.concat(out, ignore_index=True)
    team = rolling("squad_id").rename(columns={"val": "team"})
    opp = rolling("opponent_id").rename(columns={"val": "opp"})
    return team, opp


def role_table(d, col, prior):
    """Player role as of before each of his appearances: his X vs team-mates' X per 90 in the same games."""
    tg = d.groupby(["season", "squad_id", "gameweek"]).agg(tx=(col, "sum"), tm=("minutes_played", "sum")).reset_index()
    x = d.merge(tg, on=["season", "squad_id", "gameweek"])
    mates_rate = (x["tx"] - x[col]) / ((x["tm"] - x["minutes_played"]).clip(lower=1) / 90)
    x["exp"] = mates_rate.where(x["tm"] > x["minutes_played"], prior) * x["minutes_played"] / 90
    x = x.sort_values(["player_id", "season", "gameweek"])
    g = x.groupby("player_id")
    num = g[col].transform(lambda s: s.shift(1).rolling(ROLE_GAMES, min_periods=1).sum()).fillna(0)
    den = g["exp"].transform(lambda s: s.shift(1).rolling(ROLE_GAMES, min_periods=1).sum()).fillna(0)
    x["role"] = (num + K_ROLE * prior) / (den + K_ROLE * prior)
    # baseline: the player's own recent per-90 rate, shrunk to the league average
    m90 = g["minutes_played"].transform(lambda s: s.shift(1).rolling(ROLE_GAMES, min_periods=1).sum()).fillna(0) / 90
    x["own_rate"] = (num + K_ROLE * prior) / (m90 + K_ROLE)
    return x


def build(d, stat, n, prior):
    col = STATS[stat][0]
    team, opp = style_tables(d, col, n, prior)
    x = role_table(d, col, prior)
    x = x.merge(team, on=["season", "squad_id", "gameweek"], how="left").merge(opp, on=["season", "opponent_id", "gameweek"], how="left")
    x["team"] = x["team"].fillna(prior)
    x["opp"] = x["opp"].fillna(prior)
    x["y"] = x[col]
    return x


def design(x, feats, scaler=None):
    X = np.column_stack([np.log(np.clip(x[f], 1e-3, None)) if f != "home" else x[f].to_numpy() for f in feats]).astype(float)
    if scaler is None:
        scaler = (X.mean(0), X.std(0) + 1e-9)
    return (X - scaler[0]) / scaler[1], scaler


def fit(x, feats):
    X, sc = design(x, feats)
    m = PoissonRegressor(alpha=1e-4, max_iter=2000).fit(X, x["y"])
    return dict(model=m, scaler=sc, feats=feats, r=nb_r(x["y"].to_numpy(), m.predict(X)))


def predict(f, x):
    return f["model"].predict(design(x, f["feats"], f["scaler"])[0])


def points_mae(y, mu, r, unit):
    from scipy.stats import nbinom
    k = np.arange(80)
    pmf = nbinom.pmf(k[None, :], r, (r / (r + np.asarray(mu)))[:, None])
    return float(np.abs((y // unit) - (pmf * (k // unit)).sum(1)).mean())


def scores(y, mu, r, unit):
    return dict(poisson_dev=poisson_dev(y, mu), nb_loglik=nb_loglik(y, mu, r), mae=float(np.abs(y - mu).mean()),
                mae_pts=points_mae(y, mu, r, unit), mean_pred=float(np.mean(mu)), mean_actual=float(np.mean(y)))


def main():
    d = attach_odds(load_fantasy())
    d = d.dropna(subset=["lam_own"])  # rows we could match to odds (cup ties etc. drop out)
    is_train = (d["season"] == "2526") & (d["gameweek"] <= 23)
    is_valid = (d["season"] == "2526") & (d["gameweek"] > 23)
    is_test = d["season"] == "2627"
    full = d["minutes_played"] >= MIN_FULL
    print(f"defender rows matched to odds: {len(d)} (train {int((is_train & full).sum())}, validation {int((is_valid & full).sum())}, test {int((is_test & full).sum())} full games)")
    artifact = dict(stats={}, role_games=ROLE_GAMES, k_team=K_TEAM, k_role=K_ROLE, min_full=MIN_FULL)
    pd.set_option("display.width", 230)
    for stat, (col, unit) in STATS.items():
        prior = float(d.loc[is_train, col].sum() / (d.loc[is_train, "minutes_played"].sum() / 90))
        rows, cache = [], {}
        for n in WINDOWS:
            x = build(d, stat, n, prior)
            tr = x[(x["season"] == "2526") & (x["gameweek"] <= 23) & (x["minutes_played"] >= MIN_FULL)]
            va = x[(x["season"] == "2526") & (x["gameweek"] > 23) & (x["minutes_played"] >= MIN_FULL)]
            yv = va["y"].to_numpy()
            if n == WINDOWS[0]:
                rows.append(dict(model="B0 league average", window="-", **scores(yv, np.full(len(yv), prior), nb_r(tr["y"].to_numpy(), np.full(len(tr), prior)), unit)))
                rows.append(dict(model="B1 player's own recent rate", window="-", **scores(yv, va["own_rate"].to_numpy(), nb_r(tr["y"].to_numpy(), tr["own_rate"].to_numpy()), unit)))
            for name, feats in FEATURE_SETS.items():
                uses_window = any(f in ("team", "opp") for f in feats)
                if n != WINDOWS[0] and not uses_window:
                    continue
                f = fit(tr, feats)
                cache[(name, n)] = f
                rows.append(dict(model=name, window=n if uses_window else "-", **scores(yv, predict(f, va), f["r"], unit)))
            try:
                from xgboost import XGBRegressor
                feats = FEATURE_SETS["G4 + odds + home"]
                xg = XGBRegressor(objective="count:poisson", n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8, min_child_weight=20, random_state=42).fit(tr[feats], tr["y"])
                rows.append(dict(model="XGBoost (G4 features)", window=n, **scores(yv, xg.predict(va[feats]), nb_r(tr["y"].to_numpy(), xg.predict(tr[feats])), unit)))
            except ImportError:
                pass
        res = pd.DataFrame(rows)
        print(f"\n=== {col} (1 point per {unit}); league average {prior:.2f} per 90 ===\nVALIDATION 2025/26 GW24-35:")
        print(res.round(4).to_string(index=False))
        glm = res[res["model"].isin(FEATURE_SETS)].copy()
        glm["complexity"] = glm["model"].map(list(FEATURE_SETS).index)
        best = glm[glm["nb_loglik"] >= glm["nb_loglik"].max() - TOL].sort_values(["complexity", "nb_loglik"], ascending=[True, False]).iloc[0]
        n_best = WINDOWS[0] if best["window"] == "-" else int(best["window"])
        feats = FEATURE_SETS[best["model"]]
        print(f"chosen: {best['model']} (window {best['window']})")

        x = build(d, stat, n_best, prior)
        tv = x[(x["season"] == "2526") & (x["minutes_played"] >= MIN_FULL)]
        te = x[(x["season"] == "2627") & (x["minutes_played"] >= MIN_FULL)]
        f_tv = fit(tv, feats)
        yt = te["y"].to_numpy()
        t_model = scores(yt, predict(f_tv, te), f_tv["r"], unit)
        t_base = scores(yt, te["own_rate"].to_numpy(), nb_r(tv["y"].to_numpy(), tv["own_rate"].to_numpy()), unit)
        print(f"TEST 2026/27 ({len(yt)} full games): model mae_pts {t_model['mae_pts']:.4f} loglik {t_model['nb_loglik']:.4f} | own-rate baseline mae_pts {t_base['mae_pts']:.4f} loglik {t_base['nb_loglik']:.4f}")
        te = te.assign(mu=predict(f_tv, te))
        te["b"] = pd.qcut(te["mu"], 5, labels=False, duplicates="drop")
        print("test calibration by predicted quintile:", te.groupby("b").agg(pred=("mu", "mean"), actual=("y", "mean")).round(2).to_dict("list"))

        allx = x[x["minutes_played"] >= MIN_FULL]
        fin = fit(allx, feats)
        print(f"final coefficients: {json.dumps({k: round(v, 3) for k, v in zip(feats, fin['model'].coef_)})}; NB r {fin['r']:.1f}")
        artifact["stats"][stat] = dict(column=col, unit=unit, model=best["model"], features=feats, window=n_best, prior=prior,
                                       intercept=float(fin["model"].intercept_), coef=fin["model"].coef_.tolist(),
                                       scaler_mean=fin["scaler"][0].tolist(), scaler_sd=fin["scaler"][1].tolist(), nb_r=fin["r"],
                                       test=dict(model=t_model, baseline=t_base, n=int(len(yt))),
                                       validation=res.round(5).to_dict(orient="records"))
    MODEL_PATH.parent.mkdir(exist_ok=True)
    MODEL_PATH.write_text(json.dumps(artifact, indent=1), encoding="utf-8")
    print(f"\nsaved {MODEL_PATH}")


if __name__ == "__main__" and "--total" not in __import__("sys").argv:
    main()


# ---------- end-to-end check of total defender xP (all components) ----------
RATE_GAMES = 40   # recent appearances used for goals / assists / cards rates
K_RATE = 40.0     # pseudo full games of the defender average (chosen on 2025/26 GW24-35: 5/10/20/40/80 tried)


def player_rates(d, prior):
    """Goals, assists, yellow and red cards per 90 as of before each appearance (any club, both seasons), shrunk to prior."""
    x = d.sort_values(["player_id", "season", "gameweek"]).copy()
    g = x.groupby("player_id")
    m90 = g["minutes_played"].transform(lambda s: s.shift(1).rolling(RATE_GAMES, min_periods=1).sum()).fillna(0) / 90
    for col, key in (("goals_scored", "g90"), ("assists", "a90"), ("yellow_cards", "y90"), ("red_cards", "r90")):
        num = g[col].transform(lambda s: s.shift(1).rolling(RATE_GAMES, min_periods=1).sum()).fillna(0)
        x[key] = (num + K_RATE * prior[key]) / (m90 + K_RATE)
    return x


def evaluate_total():
    from scipy.stats import nbinom
    from per_season import RHO, score_matrix
    art = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    d = attach_odds(load_fantasy()).dropna(subset=["lam_own"]).reset_index(drop=True)
    tr = (d["season"] == "2526")
    full = d["minutes_played"] >= MIN_FULL
    key = ["player_id", "season", "gameweek", "opponent_id"]
    out = d[key + ["points", "minutes_played", "lam_own", "lam_opp", "home", "goals_scored", "assists"]].copy()
    k = np.arange(80)
    for stat, cfg in art["stats"].items():
        x = build(d, stat, cfg["window"], cfg["prior"])
        f = fit(x[(x["season"] == "2526") & (x["minutes_played"] >= MIN_FULL)], cfg["features"])  # 2025/26 only: honest test
        x["mu"] = predict(f, x)
        pmf = nbinom.pmf(k[None, :], f["r"], (f["r"] / (f["r"] + x["mu"].to_numpy()))[:, None])
        x[f"{stat}_pts"] = (pmf * (k // cfg["unit"])).sum(1)
        out = out.merge(x[key + [f"{stat}_pts"]].drop_duplicates(key), on=key, how="left")
    prior = {"g90": d.loc[tr, "goals_scored"].sum() / (d.loc[tr, "minutes_played"].sum() / 90),
             "a90": d.loc[tr, "assists"].sum() / (d.loc[tr, "minutes_played"].sum() / 90),
             "y90": d.loc[tr, "yellow_cards"].sum() / (d.loc[tr, "minutes_played"].sum() / 90),
             "r90": d.loc[tr, "red_cards"].sum() / (d.loc[tr, "minutes_played"].sum() / 90)}
    lam_avg = float(d.loc[tr, "lam_own"].mean())
    rates = player_rates(d, prior)
    out = out.merge(rates[key + ["g90", "a90", "y90", "r90"]].drop_duplicates(key), on=key, how="left")
    cs, gcp = [], []
    for lo, lp, h in zip(out["lam_own"], out["lam_opp"], out["home"]):
        M = score_matrix(lo, lp, RHO) if h else score_matrix(lp, lo, RHO)
        conc = M.sum(axis=0) if h else M.sum(axis=1)
        cs.append(conc[0]); gcp.append(-(conc * (np.arange(len(conc)) // 2)).sum())
    out["cs"], out["gc_pts"] = cs, gcp
    other = float((-3 * d.loc[tr, "own_goals"] - 3 * d.loc[tr, "penalty_misses"] + 5 * d.loc[tr, "hat_tricks"]).mean())
    out["xp"] = (2 + 5 * out["cs"] + out["gc_pts"] + out["clr_pts"] + out["blk_pts"] + out["tkl_pts"]
                 + (7 * out["g90"] + 3 * out["a90"]) * out["lam_own"] / lam_avg - out["y90"] - 3 * out["r90"] + other)
    # baseline: the player's average points in his previous 20 full games (any club), shrunk to the defender average
    o = out.sort_values(["player_id", "season", "gameweek"])
    gp = o.groupby("player_id")["points"]
    fullpts = o["points"].where(o["minutes_played"] >= MIN_FULL)
    s20 = fullpts.groupby(o["player_id"]).transform(lambda s: s.shift(1).rolling(20, min_periods=1).sum()).fillna(0)
    n20 = fullpts.groupby(o["player_id"]).transform(lambda s: s.shift(1).rolling(20, min_periods=1).count()).fillna(0)
    mean_full = float(out.loc[(out["season"] == "2526") & (out["minutes_played"] >= MIN_FULL), "points"].mean())
    o["base"] = (s20 + 5 * mean_full) / (n20 + 5)
    te = o[(o["season"] == "2627") & (o["minutes_played"] >= MIN_FULL)].dropna(subset=["xp"])
    y = te["points"]
    for lab, p in (("defender average", np.full(len(te), mean_full)), ("player's recent average (baseline)", te["base"]), ("component model", te["xp"])):
        e = y - p
        print(f"  {lab:36s} MAE {np.abs(e).mean():.3f}  RMSE {np.sqrt((e ** 2).mean()):.3f}  corr {np.corrcoef(p, y)[0, 1] if np.std(p) > 0 else 0:.3f}  mean {np.mean(p):.2f} (actual {y.mean():.2f})")
    te = te.assign(b=pd.qcut(te["xp"], 5, labels=False, duplicates="drop"))
    print("  calibration by xP quintile:", te.groupby("b").agg(xp=("xp", "mean"), actual=("points", "mean"), n=("points", "size")).round(2).to_dict("list"))
    # which component is off? predicted vs actual points by component, per xP quintile
    te = te.merge(d[key + ["clean_sheet", "goals_conceded", "clearances", "blocks", "tackles", "yellow_cards"]].drop_duplicates(key), on=key, how="left")
    comp = pd.DataFrame({"b": te["b"],
                         "cs_pred": 5 * te["cs"], "cs_act": 5 * (te["clean_sheet"] > 0),
                         "gc_pred": te["gc_pts"], "gc_act": -(te["goals_conceded"] // 2),
                         "clr_pred": te["clr_pts"], "clr_act": te["clearances"] // 4,
                         "tkl_pred": te["tkl_pts"], "tkl_act": te["tackles"] // 2,
                         "att_pred": (7 * te["g90"] + 3 * te["a90"]) * te["lam_own"] / lam_avg, "att_act": 7 * te["goals_scored"] + 3 * te["assists"]})
    print("  components by xP quintile (pred / actual):")
    print(comp.groupby("b").mean().round(2).to_string())
    return dict(prior=prior, lam_avg=lam_avg, other=other)


if __name__ == "__main__" and "--total" in __import__("sys").argv:
    print("End-to-end check, 2026/27 full games (component models fitted on 2025/26 only):")
    evaluate_total()

"""Midfielder / forward stat models (goals, assists, shots on target, key passes; interceptions for midfielders).
python attack_model.py        trains, validates, tests, writes models/attack_model.json
python attack_model.py --total   end-to-end points check on 2026/27

No expected-goals (xG) data yet (see docs/TODO.md), so goal threat is built from what the fantasy feed has:
  rate per 90 = exp(b0 + b1 log(role) + b2 log(team style) + b3 log(opponent style) + b4 log(own exp. goals)
                    + b5 log(opponent exp. goals) + b6 home)
  role          the player's count vs team-mates in the SAME position per 90 in the same games, last ROLE_GAMES
                appearances at any club, shrunk towards 1 with K pseudo-games (K chosen per stat on validation)
  team style    his club's same-position count per 90 over the last N gameweeks (this season)
  opp. style    count per 90 that players of this position make against this week's opponent, last N gameweeks
  exp. goals    from the match odds (market) - the main team-level signal for attacking returns
Attackers are substituted a lot (only 42% of midfielder and 26% of forward appearances are full 90s), so the models use
ALL appearances with minutes as exposure (Poisson rate per 90, weighted by minutes/90), and the page scales by minutes.

Protocol as for defenders: train 2025/26 GW1-23, choose model, window N and role shrinkage K on GW24-35, test once on
2026/27 to date, refit on all. Baselines: league rate, the player's own recent per-90 rate (minutes-adjusted), and the
plain average count over his last 10 appearances. Simplest model within TOL of the best validation log-likelihood is kept.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom
from sklearn.linear_model import PoissonRegressor

import defence_model as dm
from saves_model import nb_loglik, nb_r

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
REPO = HERE.parents[1]
MODEL_PATH = HERE / "models" / "attack_model.json"
STATS = {
    "MID": {"goals": "goals_scored", "assists": "assists", "sot": "shots_on_target", "kp": "key_passes", "int": "interceptions"},
    "FWD": {"goals": "goals_scored", "assists": "assists", "sot": "shots_on_target", "kp": "key_passes"},
}
WINDOWS = [4, 8, 12, 46]
K_ROLES = [3.0, 10.0, 30.0]  # pseudo full games at role 1.0 (rare stats like goals may need more shrinkage)
ROLE_GAMES = 20
MIN_ROW = 10  # minutes; very short cameos are too noisy to learn rates from
TOL = 5e-4
# Weight of games at a PREVIOUS club in the role (current club = 1). Chosen on 2026/27 summer movers (136 midfielders,
# 757 appearances; exp_role_recency.py + notes in docs/attacker-model.md): creative stats depend on the new team's system,
# goals / shots / interceptions travel with the player. Recency decay did not help (flat last ROLE_GAMES kept).
CLUB_W = {"kp": 0.25, "assists": 0.25}
FEATURE_SETS = {
    "G1 role": ["role"],
    "G2 role + team style": ["role", "team"],
    "G3 + opponent style": ["role", "team", "opp"],
    "G4 + odds + home": ["role", "team", "opp", "lam_own", "lam_opp", "home"],
}


def load(position):
    frames = []
    for season, tag in (("2025_26", "2526"), ("2026_27", "2627")):
        for f in sorted(Path(REPO / "data" / season).glob("player_stats_gw*.csv")):
            try:
                frames.append(pd.read_csv(f).assign(season=tag))
            except pd.errors.EmptyDataError:
                pass
    d = pd.concat(frames, ignore_index=True)
    d = d[(d["position"] == position) & (d["minutes_played"] > 0)].copy()
    return dm.attach_odds(d).dropna(subset=["lam_own"]).reset_index(drop=True)


def role_table(d, col, prior, k, club_w=1.0):
    """As defence_model.role_table, but team-mates = same position, shrinkage k is a parameter, and games at a previous
    club can be down-weighted (club_w) relative to the club of the row being predicted."""
    tg = d.groupby(["season", "squad_id", "gameweek"]).agg(tx=(col, "sum"), tm=("minutes_played", "sum")).reset_index()
    x = d.merge(tg, on=["season", "squad_id", "gameweek"])
    mates_rate = (x["tx"] - x[col]) / ((x["tm"] - x["minutes_played"]).clip(lower=1) / 90)
    x["exp"] = mates_rate.where(x["tm"] > x["minutes_played"], prior) * x["minutes_played"] / 90
    order = ["player_id", "season", "date", "gameweek"] if "date" in x else ["player_id", "season", "gameweek"]  # date: doubles
    x = x.sort_values(order).reset_index(drop=True)
    g = x.groupby("player_id")
    num = g[col].transform(lambda s: s.shift(1).rolling(ROLE_GAMES, min_periods=1).sum()).fillna(0)
    den = g["exp"].transform(lambda s: s.shift(1).rolling(ROLE_GAMES, min_periods=1).sum()).fillna(0)
    m90 = g["minutes_played"].transform(lambda s: s.shift(1).rolling(ROLE_GAMES, min_periods=1).sum()).fillna(0) / 90
    x["role"] = (num + k * prior) / (den + k * prior)
    if club_w != 1.0:
        x["role"] = club_weighted_role(x, col, prior, k, club_w)
    x["own_rate"] = (num + k * prior) / (m90 + k)
    x["last10_avg"] = g[col].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    return x


def club_weighted_role(x, col, prior, k, c):
    """Role over the last ROLE_GAMES appearances with games at other clubs weighted c (x sorted by player, season, gw)."""
    role = np.empty(len(x))
    for _, idx in x.groupby("player_id").indices.items():
        rows = x.iloc[idx]
        y, e, club = rows[col].to_numpy(float), rows["exp"].to_numpy(float), rows["squad_id"].to_numpy()
        for j in range(len(idx)):
            lo = max(0, j - ROLE_GAMES)
            w = np.where(club[lo:j] == club[j], 1.0, c)
            role[idx[j]] = (np.dot(w, y[lo:j]) + k * prior) / (np.dot(w, e[lo:j]) + k * prior)
    return role


def build(d, col, n, prior, k, club_w=1.0):
    team, opp = dm.style_tables(d, col, n, prior)
    x = role_table(d, col, prior, k, club_w)
    x = x.merge(team, on=["season", "squad_id", "gameweek"], how="left").merge(opp, on=["season", "opponent_id", "gameweek"], how="left")
    x["team"] = x["team"].fillna(prior)
    x["opp"] = x["opp"].fillna(prior)
    x["y"] = x[col]
    x["t"] = x["minutes_played"] / 90  # exposure
    return x


def design(x, feats, scaler=None):
    X = np.column_stack([np.log(np.clip(x[f], 1e-4, None)) if f != "home" else x[f].to_numpy() for f in feats]).astype(float)
    if scaler is None:
        scaler = (X.mean(0), X.std(0) + 1e-9)
    return (X - scaler[0]) / scaler[1], scaler


def fit(x, feats):
    X, sc = design(x, feats)
    # Poisson with exposure: rate = y / t, weight t (same estimates as a log(t) offset)
    m = PoissonRegressor(alpha=1e-4, max_iter=3000).fit(X, x["y"] / x["t"], sample_weight=x["t"])
    mu = m.predict(X) * x["t"]
    return dict(model=m, scaler=sc, feats=feats, r=nb_r(x["y"].to_numpy(), mu.to_numpy()))


def predict(f, x):
    return f["model"].predict(design(x, f["feats"], f["scaler"])[0]) * x["t"].to_numpy()


def scores(y, mu, r):
    mu = np.clip(np.asarray(mu, float), 1e-6, None)
    return dict(nb_loglik=nb_loglik(y, mu, r), mae=float(np.abs(y - mu).mean()), mse=float(((y - mu) ** 2).mean()),
                mean_pred=float(mu.mean()), mean_actual=float(np.mean(y)))


def train_position(position):
    d = load(position)
    tr_mask = (d["season"] == "2526") & (d["gameweek"] <= 23)
    out = {}
    print(f"\n######## {position}: {len(d)} appearances matched to odds")
    for stat, col in STATS[position].items():
        prior = float(d.loc[tr_mask, col].sum() / (d.loc[tr_mask, "minutes_played"].sum() / 90))
        rows, best_cfg = [], None
        for k in K_ROLES:
            for n in WINDOWS:
                x = build(d, col, n, prior, k, CLUB_W.get(stat, 1.0) if position == "MID" else 1.0)
                x = x[x["minutes_played"] >= MIN_ROW]
                tr = x[(x["season"] == "2526") & (x["gameweek"] <= 23)]
                va = x[(x["season"] == "2526") & (x["gameweek"] > 23)]
                yv = va["y"].to_numpy()
                if k == K_ROLES[0] and n == WINDOWS[0]:
                    base_r = nb_r(tr["y"].to_numpy(), (prior * tr["t"]).to_numpy())
                    rows.append(dict(model="B0 league rate", window="-", k="-", **scores(yv, prior * va["t"], base_r)))
                    rows.append(dict(model="B2 his last-10 average", window="-", k="-", **scores(yv, va["last10_avg"].fillna(prior * va["t"].mean()), base_r)))
                if n == WINDOWS[0]:
                    rows.append(dict(model="B1 his own recent rate", window="-", k=k, **scores(yv, va["own_rate"] * va["t"], nb_r(tr["y"].to_numpy(), (tr["own_rate"] * tr["t"]).to_numpy()))))
                for name, feats in FEATURE_SETS.items():
                    uses_w = any(f in ("team", "opp") for f in feats)
                    if n != WINDOWS[0] and not uses_w:
                        continue
                    f = fit(tr, feats)
                    rows.append(dict(model=name, window=n if uses_w else "-", k=k, **scores(yv, predict(f, va), f["r"])))
        res = pd.DataFrame(rows)
        glm = res[res["model"].isin(FEATURE_SETS)].copy()
        glm["complexity"] = glm["model"].map(list(FEATURE_SETS).index)
        best = glm[glm["nb_loglik"] >= glm["nb_loglik"].max() - TOL].sort_values(["complexity", "nb_loglik"], ascending=[True, False]).iloc[0]
        n_best = WINDOWS[0] if best["window"] == "-" else int(best["window"])
        k_best = float(best["k"])
        feats = FEATURE_SETS[best["model"]]
        summary = res.sort_values("nb_loglik", ascending=False).groupby("model").head(1).sort_values("nb_loglik", ascending=False)
        print(f"\n=== {position} {col}: league rate {prior:.3f} per 90 | validation 2025/26 GW24-35 (best row per model) ===")
        print(summary.round(4).to_string(index=False))
        print(f"chosen: {best['model']} (window {best['window']}, role shrinkage {k_best:g})")
        # test once
        club_w = CLUB_W.get(stat, 1.0) if position == "MID" else 1.0
        x = build(d, col, n_best, prior, k_best, club_w)
        x = x[x["minutes_played"] >= MIN_ROW]
        tv, te = x[x["season"] == "2526"], x[x["season"] == "2627"]
        f_tv = fit(tv, feats)
        yt = te["y"].to_numpy()
        test = dict(model=scores(yt, predict(f_tv, te), f_tv["r"]),
                    own_rate=scores(yt, te["own_rate"] * te["t"], f_tv["r"]),
                    last10=scores(yt, te["last10_avg"].fillna(prior * te["t"].mean()), f_tv["r"]), n=int(len(yt)))
        print(f"TEST 2026/27 ({len(yt)} appearances): MAE model {test['model']['mae']:.4f} / own rate {test['own_rate']['mae']:.4f} / last-10 avg {test['last10']['mae']:.4f}"
              f" | MSE {test['model']['mse']:.4f} / {test['own_rate']['mse']:.4f} / {test['last10']['mse']:.4f}"
              f" | log-lik {test['model']['nb_loglik']:.4f} / {test['own_rate']['nb_loglik']:.4f} / {test['last10']['nb_loglik']:.4f}")
        fin = fit(x, feats)
        out[stat] = dict(column=col, model=best["model"], features=feats, window=n_best, k_role=k_best, club_w=club_w, prior=prior,
                         intercept=float(fin["model"].intercept_), coef=fin["model"].coef_.tolist(),
                         scaler_mean=fin["scaler"][0].tolist(), scaler_sd=fin["scaler"][1].tolist(), nb_r=fin["r"], test=test)
        print(f"final coefficients {json.dumps({kk: round(v, 3) for kk, v in zip(feats, fin['model'].coef_)})}; NB r {fin['r']:.1f}")
    return out


def main():
    art = {"role_games": ROLE_GAMES, "k_team": dm.K_TEAM, "positions": {p: train_position(p) for p in STATS}}
    MODEL_PATH.parent.mkdir(exist_ok=True)
    MODEL_PATH.write_text(json.dumps(art, indent=1), encoding="utf-8")
    print(f"\nsaved {MODEL_PATH}")


# ---------- end-to-end points check ----------
POINTS = {"MID": {"goals": 6, "assists": 3, "sot": 1, "int": 2}, "FWD": {"goals": 5, "assists": 3, "sot": 1}}


def evaluate_total(position):
    art = json.loads(MODEL_PATH.read_text(encoding="utf-8"))["positions"][position]
    d = load(position)
    key = ["player_id", "season", "gameweek", "opponent_id"]
    out = d[key + ["points", "minutes_played", "yellow_cards", "red_cards"]].copy()
    k = np.arange(60)
    for stat, cfg in art.items():
        x = build(d, cfg["column"], cfg["window"], cfg["prior"], cfg["k_role"])
        f = fit(x[(x["season"] == "2526") & (x["minutes_played"] >= MIN_ROW)], cfg["features"])  # 2025/26 only
        x["mu"] = predict(f, x)
        if stat == "kp":
            pmf = nbinom.pmf(k[None, :], f["r"], (f["r"] / (f["r"] + x["mu"].to_numpy()))[:, None])
            x["pts_kp"] = (pmf * (k // 2)).sum(1)
        if stat == "goals":
            x["p_hat"] = 1 - nbinom.cdf(2, f["r"], f["r"] / (f["r"] + x["mu"].to_numpy()))
        cols = ["mu"] + (["pts_kp"] if stat == "kp" else []) + (["p_hat"] if stat == "goals" else [])
        out = out.merge(x[key + cols].rename(columns={"mu": f"mu_{stat}", "p_hat": "p_hat"}).drop_duplicates(key), on=key, how="left")
    tr = d["season"] == "2526"
    m90 = d.loc[tr, "minutes_played"].sum() / 90
    card = (d.loc[tr, "yellow_cards"].sum() + 3 * d.loc[tr, "red_cards"].sum()) / m90
    other = float((-3 * d.loc[tr, "own_goals"] - 3 * d.loc[tr, "penalty_misses"]).sum() / m90)
    pts = POINTS[position]
    out["xp"] = (np.where(out["minutes_played"] >= 60, 2, 1) + sum(pts[s] * out[f"mu_{s}"] for s in pts) + out["pts_kp"]
                 + 5 * out["p_hat"] - (card - other) * out["minutes_played"] / 90)
    o = out.sort_values(["player_id", "season", "gameweek"])
    full = o["points"].where(o["minutes_played"] >= 60)
    s10 = full.groupby(o["player_id"]).transform(lambda s: s.shift(1).rolling(10, min_periods=1).sum()).fillna(0)
    n10 = full.groupby(o["player_id"]).transform(lambda s: s.shift(1).rolling(10, min_periods=1).count()).fillna(0)
    mean60 = float(o.loc[(o["season"] == "2526") & (o["minutes_played"] >= 60), "points"].mean())
    o["base"] = (s10 + 3 * mean60) / (n10 + 3)
    te = o[(o["season"] == "2627") & (o["minutes_played"] >= 60)].dropna(subset=["xp"])
    y = te["points"]
    print(f"\n{position}: 2026/27 appearances of 60+ minutes ({len(te)}), model given actual minutes:")
    for lab, p in ((f"{position} average", np.full(len(te), mean60)), ("his last-10 average points (60+ games)", te["base"]), ("component model", te["xp"])):
        e = y - p
        print(f"  {lab:40s} MAE {np.abs(e).mean():.3f}  RMSE {np.sqrt((e ** 2).mean()):.3f}  corr {np.corrcoef(p, y)[0, 1] if np.std(p) > 0 else 0:.3f}  mean {np.mean(p):.2f} (actual {y.mean():.2f})")
    te = te.assign(b=pd.qcut(te["xp"], 5, labels=False, duplicates="drop"))
    print("  calibration by xP quintile:", te.groupby("b").agg(xp=("xp", "mean"), actual=("points", "mean")).round(2).to_dict("list"))
    return dict(card=card, other=other)


if __name__ == "__main__":
    if "--total" in sys.argv:
        for p in STATS:
            evaluate_total(p)
    else:
        main()

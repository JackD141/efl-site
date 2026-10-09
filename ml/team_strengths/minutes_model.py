"""Expected-minutes (xMins) model, midfielders first.
python minutes_model.py            train (2025/26 GW1-23), choose on validation (GW24-35), test once on 2026/27, refit on all,
                                   write models/minutes_model_<pos>.joblib + a JSON summary

One row per player per club fixture while he is registered at the club (the fantasy data lists every squad player,
including 0-minute ones). Target: minutes played (0-90). Features use only his EARLIER games at the same club (spells
carry across seasons): recent minutes and 60+ / appearance rates on several decays, last game's minutes, run of 0-minute
games, games at the club so far (new signings), rest days and whether it is the club's second game of the gameweek.
Baselines: the site's old rule (90 if 60+ in the club's previous game, else 0) and his average over the last 5 club games.
Expected appearance points as a function of expected minutes are fitted on validation (isotonic), so the page can turn
any xMins - predicted or typed in - into appearance points.
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge

from per_season import load_div
from predict_gw import resolver
from strengths import DIVS

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
REPO = HERE.parents[1]
MODELS = HERE / "models"
HALF_LIVES = [1, 3, 8]
FEATURES = (["last_min", "zero_run", "log_n_club", "rest_days", "second_in_gw", "season_first"]
            + [f"{k}_ewm{h}" for k in ("min", "full", "app") for h in HALF_LIVES])
# from FotMob line-ups (link_fotmob.py): started vs came off the bench, and subbed off early
FM_FEATURES = [f"{k}_ewm{h}" for k in ("start", "suboff") for h in HALF_LIVES]


def load_rows(position):
    frames = []
    for season, tag in (("2025_26", "2526"), ("2026_27", "2627")):
        for f in sorted((REPO / "data" / season).glob("player_stats_gw*.csv")):
            try:
                frames.append(pd.read_csv(f).assign(season=tag))
            except pd.errors.EmptyDataError:
                pass
    d = pd.concat(frames, ignore_index=True).drop_duplicates(["season", "player_id", "gameweek", "squad_id", "opponent_id"])
    return attach_fotmob(attach_dates(d[d["position"] == position].copy()))


def attach_fotmob(d):
    """started / subbed-off flags from FotMob line-ups; rows FotMob does not cover fall back to 60+ = started."""
    try:
        sys.path.insert(0, str(REPO / "ml" / "fotmob"))
        import link_fotmob
        fm = link_fotmob.fotmob_for_efl()[["season", "player_id", "squad_id", "opponent_id", "is_home", "started", "sub_out"]]
    except (FileNotFoundError, ImportError, ValueError):
        fm = pd.DataFrame(columns=["season", "player_id", "squad_id", "opponent_id", "is_home", "started", "sub_out"])
    d = d.merge(fm, on=["season", "player_id", "squad_id", "opponent_id", "is_home"], how="left")
    played = d["minutes_played"] > 0
    d["fm_linked"] = d["started"].notna() | ~played
    d["start"] = np.where(d["started"].notna(), d["started"].astype(float), np.where(played, (d["minutes_played"] >= 60).astype(float), 0.0))
    d["suboff"] = ((d["start"] == 1) & d["sub_out"].notna() & (d["minutes_played"] < 80)).astype(float)
    return d.drop(columns=["started", "sub_out"])


def attach_dates(d):
    """Match date for each row: football-data results, then the EFL rounds (this season) for anything missing."""
    old = {s["id"]: s for s in json.load(open(REPO / "data" / "squads.json", encoding="utf-8"))}
    cur = {s["id"]: s for s in json.load(open(HERE / "cache" / "squads.json", encoding="utf-8"))}
    parts = []
    for tag, squads in (("2526", old), ("2627", cur)):
        res = resolver(squads)
        for div in DIVS:
            m = load_div(div)
            m = m[m["season"] == tag]
            parts.append(pd.DataFrame({"season": tag, "hid": m["HomeTeam"].map(res), "aid": m["AwayTeam"].map(res), "date": pd.to_datetime(m["Date"])}))
    rounds = json.load(open(HERE / "cache" / "rounds.json", encoding="utf-8"))
    parts.append(pd.DataFrame([dict(season="2627", hid=g["homeId"], aid=g["awayId"], date=pd.Timestamp(g["date"][:10]))
                               for r in rounds if r.get("gameMode", "season") == "season" for g in r["games"]]))
    dates = pd.concat(parts).dropna().drop_duplicates(["season", "hid", "aid"])
    home = d["is_home"] == "H"
    d["hid"] = np.where(home, d["squad_id"], d["opponent_id"])
    d["aid"] = np.where(home, d["opponent_id"], d["squad_id"])
    d = d.merge(dates, on=["season", "hid", "aid"], how="left")
    # fallback ordering for anything unmatched: season start + gameweek weeks
    start = {"2526": pd.Timestamp("2025-08-01"), "2627": pd.Timestamp("2026-08-07")}
    d["date"] = d["date"].fillna(d["season"].map(start) + pd.to_timedelta(7 * (d["gameweek"] - 1), unit="D"))
    return d


def add_features(d):
    """Features from strictly earlier games of the same player at the same club."""
    d = d.sort_values(["player_id", "squad_id", "date", "gameweek"]).reset_index(drop=True)
    d["mins"] = d["minutes_played"].clip(upper=90)
    d["full"] = (d["minutes_played"] >= 60).astype(float)
    d["app"] = (d["minutes_played"] > 0).astype(float)
    g = d.groupby(["player_id", "squad_id"], sort=False)
    d["n_club"] = g.cumcount()
    d["log_n_club"] = np.log1p(d["n_club"])
    d["last_min"] = g["mins"].shift(1).fillna(-1) / 90  # -1/90 = no earlier game at this club (new signing)
    prior = {"min": 30.0, "full": 0.3, "app": 0.5, "start": 0.35, "suboff": 0.1}
    for k, col in (("min", "mins"), ("full", "full"), ("app", "app"), ("start", "start"), ("suboff", "suboff")):
        for h in HALF_LIVES:
            e = g[col].transform(lambda s: s.shift(1).ewm(halflife=h, min_periods=1).mean())
            d[f"{k}_ewm{h}"] = e.fillna(prior[k]) / (90 if k == "min" else 1)
    # run of consecutive 0-minute games just before this one
    def zero_run(s):
        out, run = [], 0
        for v in s:
            out.append(run)
            run = run + 1 if v == 0 else 0
        return out
    d["zero_run"] = np.minimum(g["mins"].transform(lambda s: pd.Series(zero_run(s.to_numpy()), index=s.index)), 10)
    # club calendar: rest days and second game in the same gameweek
    cal = d[["season", "squad_id", "gameweek", "date"]].drop_duplicates().sort_values(["squad_id", "date"])
    cal["rest_days"] = cal.groupby("squad_id")["date"].diff().dt.days.clip(upper=14).fillna(14)
    cal["second_in_gw"] = (cal.groupby(["season", "squad_id", "gameweek"]).cumcount() > 0).astype(int)
    cal["season_first"] = (cal.groupby(["season", "squad_id"]).cumcount() == 0).astype(int)
    d = d.merge(cal, on=["season", "squad_id", "gameweek", "date"], how="left")
    # baselines
    gg = d.sort_values(["player_id", "squad_id", "date"]).groupby(["player_id", "squad_id"], sort=False)
    d["b_rule"] = (gg["full"].shift(1).fillna(0) * 90).reindex(d.index)
    d["b_last5"] = gg["mins"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).fillna(0).reindex(d.index)
    return d


def predict_next(position, players, rounds):
    """Expected minutes in each player's club's next fixture: {player_id: minutes}. Injured / suspended -> 0."""
    art = joblib.load(MODELS / f"minutes_model_{position.lower()}.joblib")
    hist = load_rows(position)
    now = pd.Timestamp.now(tz="UTC")
    nxt = {}
    for r in sorted((r for r in rounds if r.get("gameMode", "season") == "season"), key=lambda r: r["roundNumber"]):
        for g in sorted(r["games"], key=lambda g: g["date"]):
            if pd.Timestamp(g["date"]) <= now:
                continue
            for cid, oid, ha in ((g["homeId"], g["awayId"], "H"), (g["awayId"], g["homeId"], "A")):
                nxt.setdefault(cid, (r["roundNumber"], oid, ha, pd.Timestamp(g["date"][:10])))
    synth = []
    for p in players:
        if p["position"] != position or p["status"] == "eliminated" or p["squadId"] not in nxt:
            continue
        gw, oid, ha, date = nxt[p["squadId"]]
        synth.append(dict(player_id=p["id"], squad_id=p["squadId"], season="2627", gameweek=gw, opponent_id=oid, is_home=ha,
                          date=date, minutes_played=0, position=position, start=0.0, suboff=0.0, _next=True))
    d = add_features(pd.concat([hist.assign(_next=False), pd.DataFrame(synth)], ignore_index=True))
    x = d[d["_next"] == True]  # noqa: E712
    pred = np.clip(art["model"].predict(x[art["features"]]), 0, 90)
    out = dict(zip(x["player_id"].astype(int), pred.round(1)))
    for p in players:
        if p["id"] in out and (p["status"] != "playing" or p.get("injuryDetails") or p.get("suspensionDetails")):
            out[p["id"]] = 0.0
    return out


def models():
    return {
        "ridge": lambda: Ridge(alpha=1.0),
        "gbm": lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0),
    }


def evaluate(y, p):
    p = np.clip(p, 0, 90)
    return dict(mae=float(np.abs(y - p).mean()), rmse=float(np.sqrt(((y - p) ** 2).mean())), mean_pred=float(p.mean()), mean_actual=float(y.mean()))


def app_points(m):
    return np.where(m >= 60, 2, np.where(m > 0, 1, 0))


def main(position="MID"):
    d = add_features(load_rows(position))
    tr = d[(d["season"] == "2526") & (d["gameweek"] <= 23)]
    va = d[(d["season"] == "2526") & (d["gameweek"] > 23)]
    te = d[d["season"] == "2627"]
    print(f"{position}: {len(d)} player-fixtures (train {len(tr)}, validation {len(va)}, test {len(te)})")
    rows = [dict(model="old site rule (90 if 60+ last game)", **evaluate(va["mins"], va["b_rule"])),
            dict(model="his last-5 average", **evaluate(va["mins"], va["b_last5"]))]
    feature_sets = {"": FEATURES, " + FotMob starts": FEATURES + FM_FEATURES}
    for name, make in models().items():
        for tag, feats in feature_sets.items():
            m = make().fit(tr[feats], tr["mins"])
            rows.append(dict(model=name + tag, **evaluate(va["mins"], m.predict(va[feats]))))
    res = pd.DataFrame(rows)
    print("\nValidation 2025/26 GW24-35 (minutes):")
    print(res.round(2).to_string(index=False))
    best_row = res.iloc[2:].sort_values("rmse").iloc[0]["model"]
    best, feats_used = best_row.split(" + ")[0], (FEATURES + FM_FEATURES if " + FotMob" in best_row else FEATURES)
    print("chosen (lowest validation RMSE):", best_row)
    # test once, model fitted on all of 2025/26
    tv = d[d["season"] == "2526"]
    m = models()[best]().fit(tv[feats_used], tv["mins"])
    pt = np.clip(m.predict(te[feats_used]), 0, 90)
    test = {"model": evaluate(te["mins"], pt), "old_rule": evaluate(te["mins"], te["b_rule"]), "last5": evaluate(te["mins"], te["b_last5"]), "n": int(len(te))}
    print("\nTEST 2026/27:", json.dumps({k: (v if k == "n" else {kk: round(vv, 2) for kk, vv in v.items()}) for k, v in test.items()}))
    # expected appearance points as a function of expected minutes (fitted on 2025/26 out-of-sample predictions)
    pv = np.clip(models()[best]().fit(tr[feats_used], tr["mins"]).predict(va[feats_used]), 0, 90)
    iso = IsotonicRegression(y_min=0, y_max=2, out_of_bounds="clip").fit(np.r_[pv, 0, 90], np.r_[app_points(va["mins"]), 0, 2])
    grid = np.arange(0, 91, 5)
    curve = [round(float(v), 3) for v in iso.predict(grid)]
    print("\nexpected appearance points by xMins:", dict(zip(grid.tolist(), curve)))
    te_app = iso.predict(pt)
    print(f"test: appearance points MAE {np.abs(app_points(te['mins']) - te_app).mean():.3f} (mean {te_app.mean():.2f} vs actual {app_points(te['mins']).mean():.2f})")
    # calibration of predicted minutes on test
    te = te.assign(p=pt, b=pd.cut(pt, [-1, 5, 20, 45, 70, 85, 91]))
    print("test calibration (predicted bucket -> mean predicted / actual minutes, n):")
    print(te.groupby("b", observed=True).agg(pred=("p", "mean"), actual=("mins", "mean"), n=("p", "size")).round(1).to_string())
    final = models()[best]().fit(d[feats_used], d["mins"])
    MODELS.mkdir(exist_ok=True)
    joblib.dump(dict(model=final, features=feats_used, half_lives=HALF_LIVES), MODELS / f"minutes_model_{position.lower()}.joblib")
    (MODELS / f"minutes_model_{position.lower()}.json").write_text(json.dumps(dict(
        model=best_row, features=feats_used, validation=res.to_dict("records"), test=test, app_curve=dict(mins=grid.tolist(), pts=curve)), indent=1), encoding="utf-8")
    if hasattr(final, "coef_"):
        print("coefficients:", {f: round(c, 2) for f, c in zip(feats_used, final.coef_)})
    return final


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "MID")

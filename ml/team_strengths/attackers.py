"""Midfielder / forward inputs for the site: public/data/mid_plan.json and fwd_plan.json (attackers.js computes xP).

Per fixture with expected minutes m (counts scale with minutes; appearance = 2 if m >= 60, 1 if 0 < m < 60):
  goals x (6 MID / 5 FWD) + 5 x P(3+ goals) + 3 x assists + shots on target + floor(key passes / 2) [+ 2 x interceptions, MID]
  - cards - 3 x missed penalties - 3 x own goals (player rates per 90, shrunk)
Rates per 90 for goals/assists/SOT/key passes/interceptions come from attack_model.py (role x club style x opponent
style x odds). Default minutes: the minutes model's prediction for his club's next game (minutes_model.py; 0 if injured or
suspended), with appearance points from its expected-minutes curve (appCurve); positions without a minutes model use
90 if he played 60+ in his club's latest game, else 0, and appearance 2 if 60+ / 1 if under.
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import attack_model as am
import minutes_model as mm
from keepers import REPO
from predict_gw import predict_gameweek

SITE = {"MID": REPO / "public" / "data" / "mid_plan.json", "FWD": REPO / "public" / "data" / "fwd_plan.json"}
K_RATE = 40.0
POINTS = {"MID": dict(goal=6, assist=3, sot=1, interception=2), "FWD": dict(goal=5, assist=3, sot=1, interception=0)}


def _roles(d, cfgs, current_club):
    out = {}
    for stat, cfg in cfgs.items():
        col, prior, k, c = cfg["column"], cfg["prior"], cfg["k_role"], cfg.get("club_w", 1.0)
        tg = d.groupby(["season", "squad_id", "gameweek"]).agg(tx=(col, "sum"), tm=("minutes_played", "sum")).reset_index()
        x = d.merge(tg, on=["season", "squad_id", "gameweek"])
        mates = (x["tx"] - x[col]) / ((x["tm"] - x["minutes_played"]).clip(lower=1) / 90)
        x["exp"] = mates.where(x["tm"] > x["minutes_played"], prior) * x["minutes_played"] / 90
        last = x.sort_values(["player_id", "season", "gameweek"]).groupby("player_id").tail(am.ROLE_GAMES).copy()
        last["w"] = np.where(last["squad_id"] == last["player_id"].map(current_club).fillna(last["squad_id"]), 1.0, c)
        last["num"], last["den"] = last[col] * last["w"], last["exp"] * last["w"]
        a = last.groupby("player_id").agg(num=("num", "sum"), den=("den", "sum"))
        for pid, r in a.iterrows():
            out.setdefault(int(pid), {})[stat] = round(float((r["num"] + k * prior) / (r["den"] + k * prior)), 4)
    return out


def _styles(cur, cfgs):
    styles = {}
    last_gw = int(cur["gameweek"].max())
    for stat, cfg in cfgs.items():
        col, prior, n = cfg["column"], cfg["prior"], cfg["window"]
        w = cur[cur["gameweek"] > last_gw - n]
        for key, name in (("squad_id", "team"), ("opponent_id", "opp")):
            g = w.groupby(key).agg(x=(col, "sum"), m=("minutes_played", "sum"))
            for cid, r in g.iterrows():
                styles.setdefault(int(cid), {}).setdefault(stat, {})[name] = round(float((r["x"] + am.dm.K_TEAM * prior) / (r["m"] / 90 + am.dm.K_TEAM)), 4)
    return styles


def _player_rates(d):
    m90 = d["minutes_played"].sum() / 90
    priors = {k: d[c].sum() / m90 for c, k in (("yellow_cards", "y90"), ("red_cards", "r90"), ("penalty_misses", "pm90"), ("own_goals", "og90"))}
    last = d.sort_values(["player_id", "season", "gameweek"]).groupby("player_id").tail(40)
    a = last.groupby("player_id").agg(m=("minutes_played", "sum"), y=("yellow_cards", "sum"), r=("red_cards", "sum"),
                                      pm=("penalty_misses", "sum"), og=("own_goals", "sum"))
    rates = {int(pid): {k: round(float((r[c] + K_RATE * priors[k]) / (r["m"] / 90 + K_RATE)), 5)
                        for c, k in (("y", "y90"), ("r", "r90"), ("pm", "pm90"), ("og", "og90"))} for pid, r in a.iterrows()}
    return rates, {k: round(float(v), 5) for k, v in priors.items()}


def _starters(cur, rounds):
    date_of = {(r["roundNumber"], g["homeId"], g["awayId"]): g["date"] for r in rounds if r.get("gameMode", "season") == "season" for g in r["games"]}
    games = cur.drop_duplicates(["squad_id", "gameweek", "opponent_id"]).copy()
    games["date"] = [date_of.get((r.gameweek, r.squad_id, r.opponent_id) if r.is_home == "H" else (r.gameweek, r.opponent_id, r.squad_id), "")
                     for r in games.itertuples()]
    latest = games.sort_values(["gameweek", "date"]).groupby("squad_id").tail(1)[["squad_id", "gameweek", "opponent_id"]]
    x = cur.merge(latest, on=["squad_id", "gameweek", "opponent_id"])
    return set(x.loc[x["minutes_played"] >= 60, "player_id"]), int(cur["gameweek"].max())


def build_plan(position, rounds, squads, players, fits, id2fd, book, book_src, market_lam, league_names):
    # expected minutes: the minutes model where one exists for this position, else the old rule (90 if 60+ last game)
    mins_json = mm.MODELS / f"minutes_model_{position.lower()}.json"
    xmins = mm.predict_next(position, players, rounds) if mins_json.exists() else None
    app_curve = json.loads(mins_json.read_text(encoding="utf-8"))["app_curve"] if xmins is not None else None
    art = json.loads(am.MODEL_PATH.read_text(encoding="utf-8"))["positions"][position]
    d = am.load(position)
    cur = d[d["season"] == "2627"]
    roles = _roles(d, art, {p["id"]: p["squadId"] for p in players})
    styles = _styles(cur, art)
    rates, rate_priors = _player_rates(d)
    starters, local_gw = _starters(cur, rounds)
    apps = cur.groupby("player_id").agg(apps=("minutes_played", "size"), mins=("minutes_played", "sum"),
                                        full=("minutes_played", lambda s: int((s >= 60).sum())))
    now = pd.Timestamp.now(tz="UTC")
    season = sorted((r for r in rounds if r.get("gameMode", "season") == "season" and r["status"] != "completed"
                     and any(pd.Timestamp(g["date"]) > now for g in r["games"])), key=lambda r: r["roundNumber"])
    completed = [r["roundNumber"] for r in rounds if r.get("gameMode", "season") == "season" and r["status"] == "completed"]
    clubs = {cid: dict(id=cid, name=s["name"], short=s.get("shortName") or s["name"], league=league_names[id2fd[cid][1]],
                       style={st: {"team": styles.get(cid, {}).get(st, {}).get("team", round(cfg["prior"], 4)),
                                   "opp": styles.get(cid, {}).get(st, {}).get("opp", round(cfg["prior"], 4))} for st, cfg in art.items()},
                       weeks=[])
             for cid, s in squads.items() if cid in id2fd}
    gameweeks = []
    for rnd in season:
        gw = rnd["roundNumber"]
        fx = predict_gameweek(rounds, squads, gw, fits, id2fd, book, book_src, market_lam)
        per = {}
        for r in fx.itertuples():
            for home in (True, False):
                cid, oid = (r.home_id, r.away_id) if home else (r.away_id, r.home_id)
                lo, lp = (r.xgH, r.xgA) if home else (r.xgA, r.xgH)
                per.setdefault(int(cid), []).append(dict(oppId=int(oid), opp=r.away if home else r.home, ha="H" if home else "A",
                                                         date=r.date, ko=r.kickoff, src=r.source, home=int(home),
                                                         lamOwn=round(lo, 4), lamOpp=round(lp, 4)))
        for cid, c in clubs.items():
            f = sorted(per.get(cid, []), key=lambda x: x["date"])
            c["weeks"].append(dict(gw=gw, games=len(f), fx=f))
        dates = sorted(g["date"][:10] for g in rnd["games"])
        gameweeks.append(dict(gw=gw, start=dates[0], end=dates[-1], games=len(rnd["games"]), lockout=rnd["lockoutDate"],
                              marketGames=int((fx.source == "market").sum())))
    plist = []
    for p in players:
        if p["position"] != position or p["status"] == "eliminated" or p["squadId"] not in clubs:
            continue
        avail = p["status"] == "playing" and not p.get("injuryDetails") and not p.get("suspensionDetails")
        inj = p.get("injuryDetails") or {}
        a = apps.loc[p["id"]] if p["id"] in apps.index else None
        plist.append(dict(
            id=p["id"], name=p.get("displayName") or f"{p['firstName']} {p['lastName']}", club=p["squadId"],
            xMins=(int(round(xmins.get(p["id"], 0))) if xmins is not None else 90 if (avail and p["id"] in starters) else 0), status=p["status"], injury=inj.get("type") if inj else None,
            suspended=bool(p.get("suspensionDetails")), startedLast=p["id"] in starters,
            role={st: roles.get(p["id"], {}).get(st, 1.0) for st in art},
            rates=rates.get(p["id"], rate_priors),
            apps=int(a["apps"]) if a is not None else 0, starts60=int(a["full"]) if a is not None else 0,
            mins=int(a["mins"]) if a is not None else 0, totalPoints=p.get("totalPoints", 0)))
    model = {st: dict(features=c["features"], intercept=c["intercept"], coef=c["coef"], mean=c["scaler_mean"], sd=c["scaler_sd"],
                      r=c["nb_r"], prior=c["prior"], window=c["window"], test=c["test"]) for st, c in art.items()}
    return dict(position=position, generatedAt=datetime.now(timezone.utc).isoformat(timespec="seconds"), season="2026/27",
                firstGw=gameweeks[0]["gw"] if gameweeks else None, gameweeks=gameweeks, clubs=list(clubs.values()), players=plist,
                startersFromGw=local_gw, latestCompletedGw=max(completed) if completed else 0,
                model=model, appCurve=app_curve, scoring=dict(appearance60=2, appearance=1, hatTrick=5, yellow=-1, red=-3, penMiss=-3, ownGoal=-3,
                                          keyPassPer=2, **POINTS[position]))


def write_site_json(plan):
    path = SITE[plan["position"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    return path

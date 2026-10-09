"""Defender inputs for the site; writes public/data/defender_plan.json for defenders.html.

The page computes each defender's xP from these so that editing expected minutes updates it straight away:
  xP if he plays the whole game = appearance 2 + 5 P(clean sheet) - E[floor(goals conceded / 2)]
     + E[floor(clearances / 4)] + E[floor(blocks / 2)] + E[floor(tackles / 2)]      (defence_model.py, negative binomial)
     + (7 x goals/90 + 3 x assists/90) x (team expected goals / league average) - yellows/90 - 3 x reds/90 + other
  xP = (expected minutes / 90) x that.   Default expected minutes: 90 if he played 60+ in his club's latest game, else 0.
Clean sheet and goals conceded come from the same per-fixture expected goals as the Club Planner (market odds or model).
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import defence_model as dm
from keepers import REPO, fixture_points
from predict_gw import predict_gameweek

SITE_JSON = REPO / "public" / "data" / "defender_plan.json"


def _rates_and_roles(d, art):
    """Each defender's current role per stat and goals/assists/cards rates (latest appearances, any club)."""
    out = {}
    d = d.sort_values(["player_id", "season", "gameweek"])
    for stat, cfg in art["stats"].items():
        col, prior = cfg["column"], cfg["prior"]
        tg = d.groupby(["season", "squad_id", "gameweek"]).agg(tx=(col, "sum"), tm=("minutes_played", "sum")).reset_index()
        x = d.merge(tg, on=["season", "squad_id", "gameweek"])
        mates = (x["tx"] - x[col]) / ((x["tm"] - x["minutes_played"]).clip(lower=1) / 90)
        x["exp"] = mates.where(x["tm"] > x["minutes_played"], prior) * x["minutes_played"] / 90
        last = x.sort_values(["player_id", "season", "gameweek"]).groupby("player_id").tail(art["role_games"])
        a = last.groupby("player_id").agg(num=(col, "sum"), den=("exp", "sum"))
        for pid, r in a.iterrows():
            out.setdefault(pid, {})[stat] = (r["num"] + dm.K_ROLE * prior) / (r["den"] + dm.K_ROLE * prior)
    m90_all = d["minutes_played"].sum() / 90
    priors = {k: d[c].sum() / m90_all for c, k in (("goals_scored", "g90"), ("assists", "a90"), ("yellow_cards", "y90"), ("red_cards", "r90"))}
    last = d.groupby("player_id").tail(dm.RATE_GAMES)
    a = last.groupby("player_id").agg(m=("minutes_played", "sum"), g=("goals_scored", "sum"), a=("assists", "sum"),
                                      y=("yellow_cards", "sum"), r=("red_cards", "sum"))
    rates = {pid: {k: (r[c] + dm.K_RATE * priors[k]) / (r["m"] / 90 + dm.K_RATE) for c, k in (("g", "g90"), ("a", "a90"), ("y", "y90"), ("r", "r90"))}
             for pid, r in a.iterrows()}
    return out, rates, priors


def _styles(cur, art):
    """Club style and opponent style per stat, this season, over each stat's window of recent gameweeks."""
    styles = {}
    last_gw = int(cur["gameweek"].max())
    for stat, cfg in art["stats"].items():
        col, prior, n = cfg["column"], cfg["prior"], cfg["window"]
        w = cur[cur["gameweek"] > last_gw - n]
        for key, name in (("squad_id", "team"), ("opponent_id", "opp")):
            g = w.groupby(key).agg(x=(col, "sum"), m=("minutes_played", "sum"))
            for cid, r in g.iterrows():
                styles.setdefault(int(cid), {}).setdefault(stat, {})[name] = (r["x"] + dm.K_TEAM * prior) / (r["m"] / 90 + dm.K_TEAM)
    return styles


def _latest_starters(cur, rounds):
    """Defenders with 60+ minutes in their club's latest game this season."""
    date_of = {(r["roundNumber"], g["homeId"], g["awayId"]): g["date"] for r in rounds if r.get("gameMode", "season") == "season" for g in r["games"]}
    games = cur.drop_duplicates(["squad_id", "gameweek", "opponent_id"]).copy()
    games["date"] = [date_of.get((r.gameweek, r.squad_id, r.opponent_id) if r.is_home == "H" else (r.gameweek, r.opponent_id, r.squad_id), "")
                     for r in games.itertuples()]
    latest = games.sort_values(["gameweek", "date"]).groupby("squad_id").tail(1)[["squad_id", "gameweek", "opponent_id"]]
    x = cur.merge(latest, on=["squad_id", "gameweek", "opponent_id"])
    return set(x.loc[x["minutes_played"] >= 60, "player_id"]), int(cur["gameweek"].max())


def build_defender_plan(rounds, squads, players, fits, id2fd, book, book_src, market_lam, league_names):
    art = json.loads(dm.MODEL_PATH.read_text(encoding="utf-8"))
    d = dm.load_fantasy()
    cur = d[d["season"] == "2627"]
    roles, rates, priors = _rates_and_roles(d, art)
    styles = _styles(cur, art)
    starters, local_gw = _latest_starters(cur, rounds)
    apps = cur.groupby("player_id").agg(apps=("minutes_played", "size"), mins=("minutes_played", "sum"),
                                        full=("minutes_played", lambda s: int((s >= 60).sum())))
    now = pd.Timestamp.now(tz="UTC")
    season = sorted((r for r in rounds if r.get("gameMode", "season") == "season" and r["status"] != "completed"
                     and r.get("lockoutDate") and pd.Timestamp(r["lockoutDate"]) > now), key=lambda r: r["roundNumber"])
    completed = [r["roundNumber"] for r in rounds if r.get("gameMode", "season") == "season" and r["status"] == "completed"]
    no_saves = dict(pen=0.0, cards=0.0)
    clubs = {cid: dict(id=cid, name=s["name"], short=s.get("shortName") or s["name"], league=league_names[id2fd[cid][1]],
                       style={st: {k: round(v, 4) for k, v in styles.get(cid, {}).get(st, {}).items()} for st in art["stats"]}, weeks=[])
             for cid, s in squads.items() if cid in id2fd}
    for cid, c in clubs.items():  # clubs without data this season fall back to the league average
        for st, cfg in art["stats"].items():
            c["style"][st].setdefault("team", round(cfg["prior"], 4))
            c["style"][st].setdefault("opp", round(cfg["prior"], 4))
    gameweeks = []
    for rnd in season:
        gw = rnd["roundNumber"]
        fx = predict_gameweek(rounds, squads, gw, fits, id2fd, book, book_src, market_lam)
        per = {}
        for r in fx.itertuples():
            for home in (True, False):
                cid, oid = (r.home_id, r.away_id) if home else (r.away_id, r.home_id)
                lo, lp = (r.xgH, r.xgA) if home else (r.xgA, r.xgH)
                p = fixture_points(lo, lp, home, 1.0, 10.0, no_saves)  # clean sheet / goals-conceded parts (saves ignored)
                per.setdefault(int(cid), []).append(dict(oppId=int(oid), opp=r.away if home else r.home, ha="H" if home else "A", date=r.date,
                                                         src=r.source, home=int(home), lamOwn=round(lo, 4), lamOpp=round(lp, 4),
                                                         cs=round(p["cs"], 4), xgc=round(p["xgc"], 3), gcPts=round(p["gcPts"], 4),
                                                         pg2=round(p["pg2"], 4), pg4=round(p["pg4"], 4)))
        for cid, c in clubs.items():
            f = sorted(per.get(cid, []), key=lambda x: x["date"])
            c["weeks"].append(dict(gw=gw, games=len(f), fx=f))
        dates = sorted(g["date"][:10] for g in rnd["games"])
        gameweeks.append(dict(gw=gw, start=dates[0], end=dates[-1], games=len(rnd["games"]),
                              marketGames=int((fx.source == "market").sum())))
    defenders = []
    for p in players:
        if p["position"] != "DEF" or p["status"] == "eliminated" or p["squadId"] not in clubs:
            continue
        avail = p["status"] == "playing" and not p.get("injuryDetails") and not p.get("suspensionDetails")
        inj = p.get("injuryDetails") or {}
        rr = rates.get(p["id"], priors)
        a = apps.loc[p["id"]] if p["id"] in apps.index else None
        defenders.append(dict(
            id=p["id"], name=p.get("displayName") or f"{p['firstName']} {p['lastName']}", club=p["squadId"],
            xMins=90 if (avail and p["id"] in starters) else 0, status=p["status"], injury=inj.get("type") if inj else None,
            suspended=bool(p.get("suspensionDetails")), startedLast=p["id"] in starters,
            role={st: round(roles.get(p["id"], {}).get(st, 1.0), 4) for st in art["stats"]},
            rates={k: round(float(v), 5) for k, v in rr.items()},
            apps=int(a["apps"]) if a is not None else 0, starts60=int(a["full"]) if a is not None else 0,
            mins=int(a["mins"]) if a is not None else 0, totalPoints=p.get("totalPoints", 0)))
    model = {st: dict(features=c["features"], intercept=c["intercept"], coef=c["coef"], mean=c["scaler_mean"], sd=c["scaler_sd"], prior=c["prior"],
                      r=c["nb_r"], unit=c["unit"], window=c["window"], test=c["test"]) for st, c in art["stats"].items()}
    lam_avg = float(np.mean([f["lamOwn"] for c in clubs.values() for w in c["weeks"] for f in w["fx"]] or [1.3]))
    other = float((-3 * d["own_goals"] - 3 * d["penalty_misses"] + 5 * d["hat_tricks"]).sum() / len(d))
    return dict(
        generatedAt=datetime.now(timezone.utc).isoformat(timespec="seconds"), season="2026/27",
        firstGw=gameweeks[0]["gw"] if gameweeks else None, gameweeks=gameweeks, clubs=list(clubs.values()), defenders=defenders,
        startersFromGw=local_gw, latestCompletedGw=max(completed) if completed else 0,
        model=model, lamAvg=round(lam_avg, 4), other=round(other, 4),
        scoring=dict(appearance=2, cleanSheet=5, goal=7, assist=3, yellow=-1, red=-3))


def write_site_json(plan, path=SITE_JSON):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    return path

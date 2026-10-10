"""Goalkeeper expected points per fixture and per keeper; writes public/data/keeper_plan.json for keepers.html.

xP if he starts = appearance 2 (99.6% of starting keepers play 60+)
                + 5 x P(clean sheet) - E[floor(goals conceded / 2)]      (from the match score matrix: market odds or model)
                + 2 x E[floor(saves / 3)]                                (saves_model.py: negative binomial, mean from odds)
                + penalty saves and cards                                (keeper averages from the fantasy data)
Keeper xP = (expected minutes / 90) x xP if he plays the whole game. Default expected minutes: 90 for the keeper who
started his club's last game (if available), 0 for everyone else; editable on the page.
Discrete scoring is handled with full distributions: saves points = 2 x [P(3+) + P(6+) + ...] from the negative binomial,
goals-conceded points = -[P(2+) + P(4+) + ...] from the score matrix.
"""
import glob
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom

import saves_model as sm
from per_season import RHO, score_matrix
from predict_gw import predict_gameweek

HERE = Path(__file__).parent
REPO = HERE.parents[1]
SITE_JSON = REPO / "public" / "data" / "keeper_plan.json"
APPEARANCE = 2
CLEAN_SHEET = 5
K_SAVES = np.arange(41)


def load_model():
    return json.loads(sm.MODEL_PATH.read_text(encoding="utf-8"))


def predict_saves(model, rows):
    """Expected saves for rows with the model's feature columns (same transform as in training)."""
    X = np.column_stack([np.log(np.clip(rows[f], 1e-3, None)) if f != "home" else rows[f] for f in model["features"]]).astype(float)
    X = (X - np.array(model["scaler_mean"])) / np.array(model["scaler_sd"])
    return np.exp(model["intercept"] + X @ np.array(model["coef"]))


def current_rolling(model, season="2627"):
    """Each team's rolling features as of now (after its latest match); only needed if the model uses them."""
    need = [f for f in model["features"] if f in sm.ROLL_COLS]
    if not need:
        return {}
    g = sm.team_games([season])
    n, k, pr = model["window"], model["k"], model["priors"]
    out = {}
    for team, x in g.groupby("team"):
        last = x.tail(n)
        out[team] = dict(
            sot_against=(last["sot_ag"].sum() + k * pr["sot"]) / (len(last) + k),
            sot_for=(last["sot_for"].sum() + k * pr["sot"]) / (len(last) + k),
            save_rate=(last["saves"].sum() + k * pr["sot"] * pr["save_rate"]) / (last["sot_ag"].sum() + k * pr["sot"]))
    return out


def fixture_points(lam_own, lam_opp, home, mu_saves, r, consts):
    lh, la = (lam_own, lam_opp) if home else (lam_opp, lam_own)
    M = score_matrix(lh, la, RHO)
    conceded = M.sum(axis=0) if home else M.sum(axis=1)  # distribution of the opponent's goals
    g = np.arange(len(conceded))
    p_cs = float(conceded[0])
    gc_pts = -float((conceded * (g // 2)).sum())
    pmf = nbinom.pmf(K_SAVES, r, r / (r + mu_saves))
    save_pts = float((pmf * 2 * (K_SAVES // 3)).sum())
    xp = APPEARANCE + CLEAN_SHEET * p_cs + gc_pts + save_pts + consts["pen"] + consts["cards"]
    at_least = lambda dist, n: float(dist[n:].sum())
    return dict(cs=p_cs, xgc=float((conceded * g).sum()), gcPts=gc_pts, pg2=at_least(conceded, 2), pg4=at_least(conceded, 4),
                saves=float(mu_saves), savePts=save_pts, ps3=at_least(pmf, 3), ps6=at_least(pmf, 6), ps9=at_least(pmf, 9), xp=xp)


# ---------- fantasy data: constants, starters ----------
def fantasy_gk_rows():
    frames = []
    for season in ("2025_26", "2026_27"):
        for f in glob.glob(str(REPO / "data" / season / "player_stats_gw*.csv")):
            try:
                d = pd.read_csv(f)
            except pd.errors.EmptyDataError:
                continue
            frames.append(d[d["position"] == "GK"].assign(season=season))
    return pd.concat(frames, ignore_index=True)


def starts_table(gk):
    """One row per team-game: the keeper with most minutes (the starter)."""
    on = gk[gk["minutes_played"] > 0].sort_values("minutes_played", ascending=False)
    return on.groupby(["season", "squad_id", "gameweek", "opponent_id"], as_index=False).head(1)


def keeper_constants(gk):
    st = starts_table(gk).sort_values(["season", "squad_id", "gameweek"])
    st["prev"] = st.groupby(["season", "squad_id"])["player_id"].shift(1)
    kept = st[st["prev"].notna()]
    return dict(pen=float(5 * st["penalty_saves"].mean()),
                cards=float(-(st["yellow_cards"] + 3 * st["red_cards"]).mean()),
                p_keep=float((kept["player_id"] == kept["prev"]).mean()),
                n_starts=int(len(st)))


def model_minutes(players, rounds):
    """Minutes model (minutes_model.py, GK) for each keeper's next game, scaled within each club so the keepers' expected
    minutes add up to 90 (only one keeper plays); injured / suspended keepers get 0. None if no GK minutes model."""
    import minutes_model as mm
    if not (mm.MODELS / "minutes_model_gk.joblib").exists():
        return None
    pred = mm.predict_next("GK", players, rounds)
    by_club = {}
    for p in players:
        if p["position"] == "GK" and p["id"] in pred:
            by_club.setdefault(p["squadId"], []).append(p["id"])
    out = {}
    for ids in by_club.values():
        tot = sum(pred[i] for i in ids)
        for i in ids:
            out[i] = 90.0 * pred[i] / tot if tot > 0 else 0.0
    return out


def keeper_roster(players, squads, rounds, gk, consts):
    cur = gk[gk["season"] == "2026_27"]
    date_of = {}
    for r in rounds:
        if r.get("gameMode", "season") != "season":
            continue
        for g_ in r["games"]:
            date_of[(r["roundNumber"], g_["homeId"], g_["awayId"])] = g_["date"]
    def game_date(row):
        h, a = (row.squad_id, row.opponent_id) if row.is_home == "H" else (row.opponent_id, row.squad_id)
        return date_of.get((row.gameweek, h, a), "")
    st = starts_table(cur).copy()
    st["date"] = [game_date(r) for r in st.itertuples()]
    last = st.sort_values(["gameweek", "date"]).groupby("squad_id").tail(1).set_index("squad_id")
    starts = st.groupby("player_id").size()
    mins = cur.groupby("player_id")["minutes_played"].sum()
    last_start = st.sort_values(["gameweek", "date"]).groupby("player_id").tail(1).set_index("player_id")

    keepers = [p for p in players if p["position"] == "GK" and p["status"] != "eliminated" and p["squadId"] in squads]
    by_club = {}
    for p in keepers:
        by_club.setdefault(p["squadId"], []).append(p)
    out = []
    xm = model_minutes(players, rounds)
    for club, ks in by_club.items():
        def available(p):
            return p["status"] == "playing" and not p.get("injuryDetails") and not p.get("suspensionDetails")
        avail = [p for p in ks if available(p)]
        starter = None
        if club in last.index and any(p["id"] == last.at[club, "player_id"] for p in avail):
            starter = int(last.at[club, "player_id"])
        elif avail:  # last starter unavailable or gone: the available keeper with most starts, then minutes
            starter = max(avail, key=lambda p: (starts.get(p["id"], 0), mins.get(p["id"], 0)))["id"]
        for p in ks:
            ps = 1.0 if (available(p) and p["id"] == starter) else 0.0
            if xm is not None:  # minutes model (start chance) where trained; the rule above stays as the fallback
                ps = xm.get(p["id"], 0.0) / 90 if available(p) else 0.0
            inj = p.get("injuryDetails") or {}
            ls = last_start.loc[p["id"]] if p["id"] in last_start.index else None
            out.append(dict(
                id=p["id"], name=p.get("displayName") or f"{p['firstName']} {p['lastName']}", club=club, pStart=round(ps, 3),
                xMins=int(round(90 * ps)), status=p["status"], injury=inj.get("type") if inj else None,
                injuryStatus=inj.get("status") if inj else None, suspended=bool(p.get("suspensionDetails")),
                starts=int(starts.get(p["id"], 0)), mins=int(mins.get(p["id"], 0)),
                lastStartGw=int(ls["gameweek"]) if ls is not None else None,
                startedLast=bool(club in last.index and last.at[club, "player_id"] == p["id"]),
                totalPoints=p.get("totalPoints", 0)))
    return out, int(cur["gameweek"].max()) if len(cur) else 0


# ---------- the plan ----------
def build_keeper_plan(rounds, squads, players, fits, id2fd, book, book_src, market_lam, league_names):
    model = load_model()
    gk = fantasy_gk_rows()
    consts = keeper_constants(gk)
    roster, local_gw = keeper_roster(players, squads, rounds, gk, consts)
    rolling = current_rolling(model)
    now = pd.Timestamp.now(tz="UTC")
    season = sorted((r for r in rounds if r.get("gameMode", "season") == "season" and r["status"] != "completed"
                     and any(pd.Timestamp(g["date"]) > now for g in r["games"])), key=lambda r: r["roundNumber"])
    completed = [r["roundNumber"] for r in rounds if r.get("gameMode", "season") == "season" and r["status"] == "completed"]
    clubs = {cid: dict(id=cid, name=s["name"], short=s.get("shortName") or s["name"], league=league_names[id2fd[cid][1]], weeks=[])
             for cid, s in squads.items() if cid in id2fd}
    gameweeks = []
    for rnd in season:
        gw = rnd["roundNumber"]
        fx = predict_gameweek(rounds, squads, gw, fits, id2fd, book, book_src, market_lam)
        rows = []
        for r in fx.itertuples():
            for home in (True, False):
                cid, oid = (r.home_id, r.away_id) if home else (r.away_id, r.home_id)
                rows.append(dict(club=int(cid), opp_id=int(oid), opp=r.away if home else r.home, ha="H" if home else "A", date=r.date, ko=r.kickoff,
                                 lam_own=r.xgH if home else r.xgA, lam_opp=r.xgA if home else r.xgH, home=int(home), src=r.source,
                                 team_fd=id2fd[cid][0], opp_fd=id2fd[oid][0]))
        rows = pd.DataFrame(rows)
        if rolling:  # model uses recent-form features: attach each team's current values
            rows["sot_against"] = rows["team_fd"].map(lambda t: rolling.get(t, {}).get("sot_against", model["priors"]["sot"]))
            rows["save_rate"] = rows["team_fd"].map(lambda t: rolling.get(t, {}).get("save_rate", model["priors"]["save_rate"]))
            rows["opp_sot_for"] = rows["opp_fd"].map(lambda t: rolling.get(t, {}).get("sot_for", model["priors"]["sot"]))
        rows["mu"] = predict_saves(model, rows) if len(rows) else []
        per_club = {}
        for r in rows.itertuples():
            pts = fixture_points(r.lam_own, r.lam_opp, bool(r.home), r.mu, model["nb_r"], consts)
            per_club.setdefault(r.club, []).append(dict(opp=r.opp, ha=r.ha, date=r.date, ko=r.ko, src=r.src,
                                                        **{k: round(v, 4) for k, v in pts.items()}))
        for cid, c in clubs.items():
            f = sorted(per_club.get(cid, []), key=lambda x: x["date"])
            c["weeks"].append(dict(gw=gw, xp=round(sum(x["xp"] for x in f), 3), games=len(f), fx=f))
        dates = sorted(g_["date"][:10] for g_ in rnd["games"])
        gameweeks.append(dict(gw=gw, start=dates[0], end=dates[-1], games=len(rnd["games"]), lockout=rnd["lockoutDate"],
                              marketGames=int((fx.source == "market").sum()), modelGames=int((fx.source == "model").sum())))
    return dict(
        generatedAt=datetime.now(timezone.utc).isoformat(timespec="seconds"), season="2026/27",
        firstGw=gameweeks[0]["gw"] if gameweeks else None, gameweeks=gameweeks, clubs=list(clubs.values()), keepers=roster,
        startersFromGw=local_gw, latestCompletedGw=max(completed) if completed else 0,
        constants=dict(appearance=APPEARANCE, cleanSheet=CLEAN_SHEET, pen=round(consts["pen"], 3), cards=round(consts["cards"], 3),
                       pKeep=round(consts["p_keep"], 3), nStarts=consts["n_starts"]),
        savesModel=dict(name=model["model"], features=model["features"], nbR=round(model["nb_r"], 2),
                        test=model["test"]))


def write_site_json(plan, path=SITE_JSON):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    return path

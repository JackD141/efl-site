"""Backtest of the whole pick pipeline on 2026/27: for each completed gameweek, refit the models on games BEFORE it, pick
the team the site would have picked, and score it with the real fantasy points.
python backtest.py      writes public/data/backtest.json (the Backtest page)

Per gameweek g (all inputs as of g's first kick-off):
- fixtures: expected goals from the closing odds (the market's pre-match view); games without odds in our data use the
  team-strength model fitted on matches before g (predict_gw.fit_current)
- players: every registered player of that gameweek. Stat models (player_model.py, same features / window / shrinkage as
  live, coefficients refitted on 2025/26 + 2026/27 before g), inputs via the same live_inputs code cut to games before g;
  start chances from the start model (minutes_model.py) refitted before g, each club's outfield chances adding up to ten
  starters (no injury news: harder than live; the second game of a double uses the first game's start chance). Keepers:
  odds-based clean sheet / goals conceded / saves, start chances scaled to add up to 1 per club
- team: exact optimum (scipy milp) of sum xP + captain xP; formations 1-2-2-2 / 1-2-3-1 / 1-3-2-1, at most 2 per club
- clubs: the two highest-xP clubs with picks left (each club 5 times a season)
- score: real points; captain doubled (vice-captain if the captain did not play); club points from the real results
Comparisons: a "form" picker (average points over each player's last 5 club games, same rules and same clubs) and the
best possible team in hindsight.
Scorecard (in backtest.json, shown on the Backtest page): how good the expected points are over EVERY player-gameweek,
not just the seven picked: error vs the form baseline, calibration by xP band and of the top ten per position, start
chances vs what happened, expected vs actual count of every scoring stat, and how uncertain the points totals are.
Gameweeks after DEV_UNTIL were never looked at while the models were being designed: the honest record. Leakage: none known (league step multipliers and FotMob shrinkage targets use completed
seasons only, the saves model is refitted before each gameweek, feature choices were made on 2025/26 only).
"""
import json
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.stats import nbinom, poisson

import attack_model as am
import club_points as cp
import keepers as kp
import minutes_model as mm
import player_model as pm
import saves_model as sm
from per_season import lam_for, predict_probs, score_matrix
from predict_gw import RHO, fit_current, resolver
from strengths import DIVS

warnings.filterwarnings("ignore")
HERE = am.HERE
OUT = am.REPO / "public" / "data" / "backtest.json"
SCORING = {"MID": dict(goal=6, assist=3, sot=1, interception=2), "FWD": dict(goal=5, assist=3, sot=1, interception=0)}
UNITS = {"clr": 4, "blk": 2, "tkl": 2}
FORMATIONS = [(1, 2, 2, 2), (1, 2, 3, 1), (1, 3, 2, 1)]
POS = ["GK", "DEF", "MID", "FWD"]
K = np.arange(200)
DEV_UNTIL = 8  # 2026/27 GW1-8 were looked at while the models were designed (docs/ml-methodology.md section 3)
ALT = "start chances without the club total of ten"  # the previous method, carried as a comparison (xp_alt)
STAT_COLS = {"goals": "goals_scored", "assists": "assists", "sot": "shots_on_target", "kp": "key_passes", "int": "interceptions",
             "clr": "clearances", "blk": "blocks", "tkl": "tackles"}


def nb_pts(mu, r, per):
    if mu <= 0:
        return 0.0
    pmf = poisson.pmf(K, mu) if r > 1e3 else nbinom.pmf(K, r, r / (r + mu))
    return float((pmf * (K // per)).sum())


def p_at_least(mu, r, n):
    if mu <= 0:
        return 0.0
    return float(1 - (poisson.cdf(n - 1, mu) if r > 1e3 else nbinom.cdf(n - 1, r, r / (r + mu))))


def rate(a, vals):
    z = a["intercept"]
    for i, f in enumerate(a["features"]):
        v = vals[f] if f == "home" else np.log(max(vals[f], 1e-4))
        z += a["coef"][i] * (v - a["scaler_mean"][i]) / a["scaler_sd"][i]
    return float(np.exp(z))


def curve(c, mins, key):
    return float(np.interp(mins, c["mins"], c[key])) if mins > 0 else 0.0


# ---------- fixtures ----------
def fixtures_for(rounds, squads, gw):
    rnd = next(r for r in rounds if r["roundNumber"] == gw and r.get("gameMode", "season") == "season")
    start = pd.Timestamp(min(g["date"] for g in rnd["games"])[:10])
    res = resolver(squads)
    market = {}
    for div in DIVS:
        m = lam_for(div, "2627").dropna(subset=["lamH"])
        for r in m.itertuples():
            market[(res(r.HomeTeam), res(r.AwayTeam))] = (r.lamH, r.lamA)
    try:
        fits, id2fd, _ = fit_current(start, squads)
    except (IndexError, ValueError):  # no matches before the first gameweek: league averages from last season
        fits = None
    avg = {}
    for div in DIVS:
        m = lam_for(div, "2526").dropna(subset=["lamH"])
        for t in m["HomeTeam"]:
            avg[res(t)] = (float(m["lamH"].mean()), float(m["lamA"].mean()))
    fx = []
    for g in rnd["games"]:
        key = (g["homeId"], g["awayId"])
        if key in market:
            lh, la, src = *market[key], "market"
        elif fits is not None:
            hn, div = id2fd[g["homeId"]]
            an, _ = id2fd[g["awayId"]]
            _, lh, la = predict_probs(fits[div], hn, an, RHO)
            src = "model"
        else:
            lh, la = avg.get(g["homeId"], (1.4, 1.1))
            src = "average"
        fx.append(dict(home=g["homeId"], away=g["awayId"], date=g["date"][:10], lh=float(lh), la=float(la), src=src,
                       hs=g.get("homeScore"), as_=g.get("awayScore")))
    return fx, start


def club_realised(gf, ga, home):
    p = 0
    if gf > ga:
        p += cp.POINTS["win"] + (0 if home else cp.POINTS["away_win"])
    elif gf == ga:
        p += cp.POINTS["draw"]
    p += cp.POINTS["clean_sheet"] * (ga == 0) + cp.POINTS["two_goals"] * (gf >= 2) + cp.POINTS["four_goals"] * (gf >= 4)
    return p


# ---------- minutes, refitted before the gameweek ----------
def starts_before(pos, feat, gw, mins_json):
    """{player: (P(start), P(full game), P(part game))} for gameweek gw from the start model refitted on games before it.
    feat: minutes_model.add_features(load_rows(pos)) (suspension games already left out)."""
    tr = feat[(feat["season"] == "2526") | ((feat["season"] == "2627") & (feat["gameweek"] < gw))]
    art = mm.fit_start(tr, mins_json["features"])
    rows = feat[(feat["season"] == "2627") & (feat["gameweek"] == gw)].sort_values("date")
    first = rows.drop_duplicates("player_id")  # first game of a double: features do not know the first game's outcome
    ps, pf, pp = mm.scenario_probs(art, first)
    return {int(i): (float(a), float(b), float(c)) for i, a, b, c in zip(first["player_id"], ps, pf, pp)}, art["q"]


K_CARD = 40.0  # pseudo-90s of the position average, as on the pages (attackers.K_RATE, defence_model.K_RATE)


def card_costs(hist):
    """({player: expected points lost per 90 to cards, own goals and missed penalties}, the position average) as the
    pages work it out: his last 40 appearances before the gameweek, shrunk towards the position's rates with K_CARD
    pseudo-90s. hist: the position's rows before the gameweek."""
    h = hist[hist["minutes_played"] > 0]
    cost = {"yellow_cards": 1, "red_cards": 3, "own_goals": 3, "penalty_misses": 3}
    m90 = h["minutes_played"].sum() / 90
    pri = {c: h[c].sum() / m90 for c in cost}
    a = h.sort_values(["season", "date"]).groupby("player_id").tail(40).groupby("player_id")[list(cost) + ["minutes_played"]].sum()
    per = sum(w * (a[c] + K_CARD * pri[c]) / (a["minutes_played"] / 90 + K_CARD) for c, w in cost.items())
    return per.to_dict(), float(sum(w * pri[c] for c, w in cost.items()))


def club_total(sp, club_of, total=mm.OUTFIELD_STARTERS):
    """sp {player: (P(start), P(full game), P(part game))} for every outfield player of the gameweek -> the same with
    each club's start chances adding up to ten (minutes_model.shift_to: one shift of all the club's start logits);
    his P(60+ | started) and P(comes on | not started) are unchanged."""
    by = {}
    for i in sp:
        by.setdefault(club_of[i], []).append(i)
    out = {}
    for ids in by.values():
        for i, s2 in zip(ids, mm.shift_to([sp[i][0] for i in ids], total)):
            ps, pf, pp = sp[i]
            q = pf / ps if ps > 0 else 0.0
            pb = (pp - ps * (1 - q)) / (1 - ps) if ps < 1 else 0.0
            out[i] = (float(s2), float(s2 * q), float(s2 * (1 - q) + (1 - s2) * max(pb, 0.0)))
    return out


# ---------- expected points per player for one gameweek ----------
def outfield_xp(pos, art, inputs, players, fx_by_club, sp, mj, card90, m_full, mix=True, counts=None):
    """Gameweek xP per player. sp: {player: (P(start), P(full game), P(part game))}. mix=True: P(full) x his full game
    (m_full minutes, appearance 2, clean sheet counts) + P(part) x a typical part game (appearance 1); mix=False: straight
    scaling by the implied expected minutes / 90 (for comparison). card90: (per player, default) from card_costs().
    counts: a dict to fill with each player's expected count of every stat over the gameweek, for the scorecard."""
    roles, styles, fm, mates = inputs
    fm_def = pm.fm_defaults(fm) if fm else {}
    out = {}
    for p in players:
        pid, cid = p["id"], p["squadId"]
        _, pf, pp = sp.get(pid, (0.0, 0.0, 0.0))
        mfull, mpart = m_full.get(pid, mj["bucket_minutes"]["full"]), mj["bucket_minutes"]["part"]
        mins = pf * mfull + pp * mpart
        tot = 0.0
        for f in fx_by_club.get(cid, []):
            if mins <= 0:
                break
            home = f["home"] == cid
            oid = f["away"] if home else f["home"]
            lo, lp = (f["lh"], f["la"]) if home else (f["la"], f["lh"])
            rate90 = {}
            for stat, a in art.items():
                vals = dict(role=roles.get(pid, {}).get(stat, 1.0), team=styles.get(cid, {}).get(stat, {}).get("team", a["prior"]),
                            opp=styles.get(oid, {}).get(stat, {}).get("opp", a["prior"]), mates=mates.get(pid, {}).get(stat, a["prior"]),
                            lam_own=lo, lam_opp=lp, home=int(home), **fm.get(pid, fm_def))
                rate90[stat] = rate(a, vals) * a["scale"] + (fm.get(pid, fm_def).get(a.get("pen_col", "fm_pxg"), 0) if a["pens"] else 0)
            if counts is not None:
                c = counts.setdefault(pid, {})
                for stat, v in rate90.items():
                    c[stat] = c.get(stat, 0.0) + v * mins / 90
            pt = kp.fixture_points(lo, lp, home, 1.0, 10.0, dict(pen=0.0, cards=0.0)) if pos == "DEF" else None

            def at(m, app, cs_w):
                t = m / 90
                mu = {k: v * t for k, v in rate90.items()}
                if pos == "DEF":
                    x = app + 5 * pt["cs"] * cs_w + pt["gcPts"] * t + sum(nb_pts(mu[s_], art[s_]["nb_r"], u) for s_, u in UNITS.items())
                    x += 7 * mu["goals"] + 3 * mu["assists"]
                else:
                    sc = SCORING[pos]
                    x = app + sc["goal"] * mu["goals"] + 5 * p_at_least(mu["goals"], art["goals"]["nb_r"], 3) + sc["assist"] * mu["assists"]
                    x += sc["sot"] * mu["sot"] + nb_pts(mu["kp"], art["kp"]["nb_r"], 2) + (sc["interception"] * mu["int"] if "int" in mu else 0)
                return x - card90[0].get(pid, card90[1]) * t
            if mix:
                tot += pf * at(mfull, 2, 1.0) + pp * at(mpart, 1, 0.0)
            else:
                tot += at(mins, 2 * pf + pp, pf if pos == "DEF" else 0.0)
        out[pid] = tot
    return out


def keeper_xp(players, fx_by_club, sp, saves_model, consts, mj, mix=True):
    out = {}
    for p in players:
        tot = 0.0
        for f in fx_by_club.get(p["squadId"], []):
            home = f["home"] == p["squadId"]
            lo, lp = (f["lh"], f["la"]) if home else (f["la"], f["lh"])
            row = pd.DataFrame([dict(lam_own=lo, lam_opp=lp, home=int(home))])
            mu = float(kp.predict_saves(saves_model, row)[0])
            tot += kp.fixture_points(lo, lp, home, mu, saves_model["nb_r"], consts)["xp"]
        _, pf, pp = sp.get(p["id"], (0.0, 0.0, 0.0))
        if mix:  # P(full game) x full-game xP + P(part game) x one appearance point per game
            out[p["id"]] = pf * tot + pp * len(fx_by_club.get(p["squadId"], []))
        else:
            out[p["id"]] = tot * (pf * 90 + pp * mj["bucket_minutes"]["part"]) / 90
    return out


# ---------- team choice ----------
def best_team(cands, value, cap_value=None):
    """Exact best 7 (+ captain) by sum value + captain's value, under the formations and 2-per-club. Returns (ids, captain)."""
    cap_value = value if cap_value is None else cap_value
    n = len(cands)
    if n == 0:
        return [], None
    v = np.array([value[c["id"]] for c in cands]); cv = np.array([cap_value[c["id"]] for c in cands])
    best = None
    for form in FORMATIONS:
        # variables: x (picked) then c (captain)
        cost = -np.r_[v, cv]
        A, lo, hi = [], [], []
        def row(xc, cc=None):
            return np.r_[xc, np.zeros(n) if cc is None else cc]
        for pos, k in zip(POS, form):
            A.append(row(np.array([c["pos"] == pos for c in cands], float))); lo.append(k); hi.append(k)
        for club in {c["squadId"] for c in cands}:
            A.append(row(np.array([c["squadId"] == club for c in cands], float))); lo.append(0); hi.append(2)
        A.append(row(np.zeros(n), np.ones(n))); lo.append(1); hi.append(1)
        for i in range(n):  # captain must be picked
            e = np.zeros(2 * n); e[n + i] = 1; e[i] = -1
            A.append(e); lo.append(-np.inf); hi.append(0)
        r = milp(cost, constraints=LinearConstraint(np.array(A), lo, hi), integrality=np.ones(2 * n), bounds=Bounds(0, 1))
        if r.status == 0 and (best is None or r.fun < best[0]):
            best = (r.fun, r.x)
    x = best[1]
    ids = [cands[i]["id"] for i in range(n) if x[i] > 0.5]
    cap = next(cands[i]["id"] for i in range(n) if x[n + i] > 0.5)
    return ids, cap


def main():
    rounds = json.load(open(HERE / "cache" / "rounds.json", encoding="utf-8"))
    squads = {s["id"]: s for s in json.load(open(HERE / "cache" / "squads.json", encoding="utf-8"))}
    # finished gameweeks: every game completed (or postponed); the EFL can leave a round "playing" for days after it ends
    done = sorted(r["roundNumber"] for r in rounds if r.get("gameMode", "season") == "season" and r["games"]
                  and all(g.get("status") in ("completed", "postponed") for g in r["games"]) and any(g.get("status") == "completed" for g in r["games"]))
    # data loaded once; every step below only uses rows before the gameweek
    data = {pos: pm.load(pos) for pos in ("MID", "FWD", "DEF")}
    arts = {pos: json.loads(pm.model_path(pos).read_text(encoding="utf-8")) for pos in ("MID", "FWD", "DEF")}
    built = {pos: {} for pos in arts}
    mins_feat = {pos: mm.add_features(mm.load_rows(pos)) for pos in POS}
    mins_json = {pos: json.loads((mm.MODELS / f"minutes_model_{pos.lower()}.json").read_text(encoding="utf-8")) for pos in POS}
    saves_model = kp.load_model()  # configuration only; coefficients are refitted before each gameweek below
    saves_games = sm.team_games(["2324", "2425", "2526", "2627"])
    gk_rows = kp.fantasy_gk_rows()
    consts = kp.keeper_constants(gk_rows[gk_rows["season"] == "2025_26"])
    rows_all = pd.concat([mm.load_rows(pos).assign(pos=pos) for pos in POS], ignore_index=True)
    uses, form_uses = {}, {}
    calib, stat_rows = [], []
    weeks = []
    for gw in done:
        fx, start = fixtures_for(rounds, squads, gw)
        fx_by_club = {}
        for f in fx:
            fx_by_club.setdefault(f["home"], []).append(f)
            fx_by_club.setdefault(f["away"], []).append(f)
        # keeper saves model refitted on matches before this gameweek (the saved model was fitted including 2026/27)
        fs = sm.fit_glm(saves_games[saves_games["date"] < start], saves_model["features"])
        saves_gw = dict(saves_model, intercept=float(fs["model"].intercept_), coef=fs["model"].coef_.tolist(),
                        scaler_mean=fs["scaler"][0].tolist(), scaler_sd=fs["scaler"][1].tolist(), nb_r=fs["r"])
        cur = rows_all[(rows_all["season"] == "2627") & (rows_all["gameweek"] == gw)]
        players = [dict(id=int(pid), squadId=int(g["squad_id"].iloc[0]), pos=g["pos"].iloc[0],
                        name=f"{g['first_name'].iloc[0]} {g['last_name'].iloc[0]}") for pid, g in cur.groupby("player_id")]
        players = [p for p in players if p["squadId"] in fx_by_club]
        real = cur.groupby("player_id").agg(points=("points", "sum"), mins=("minutes_played", "sum"))
        xp, xp_lin = {}, {}
        # start chances of every position first: a club's outfield chances add up to ten starters
        sp_gw = {pos: starts_before(pos, mins_feat[pos], gw, mins_json[pos]) for pos in POS}
        club_of = {p["id"]: p["squadId"] for p in players}
        raw = {i: v for pos in mm.OUTFIELD for i, v in sp_gw[pos][0].items() if i in club_of}
        totalled = club_total(raw, club_of)
        first = rows_all[(rows_all["season"] == "2627") & (rows_all["gameweek"] == gw)].sort_values("date").drop_duplicates("player_id").set_index("player_id")
        start_chance, counts = {}, {}
        for pos in POS:
            plist = [dict(p, position=pos) for p in players if p["pos"] == pos]
            xm, q = sp_gw[pos]
            xm_raw = xm
            if pos != "GK":
                xm = {i: totalled[i] for i in xm if i in totalled}
            if pos == "GK":  # one keeper starts: scale each club's keepers' start chances to add up to 1
                byc = {}
                for p in plist:
                    byc.setdefault(p["squadId"], []).append(p["id"])
                for ids in byc.values():
                    tot = sum(xm.get(i, (0, 0, 0))[0] for i in ids)
                    for i in ids:
                        ps, pf, pp = xm.get(i, (0.0, 0.0, 0.0))
                        ps2 = ps / tot if tot > 0 else 0.0
                        qi = pf / ps if ps > 0 else q  # his P(60+ | started)
                        pb = (pp - ps * (1 - qi)) / (1 - ps) if ps < 1 else 0.0  # P(sub | not started)
                        xm[i] = (ps2, ps2 * qi, ps2 * (1 - qi) + (1 - ps2) * max(pb, 0.0))
                xp.update(keeper_xp(plist, fx_by_club, xm, saves_gw, consts, mins_json[pos]))
                xp_lin.update(keeper_xp(plist, fx_by_club, xm, saves_gw, consts, mins_json[pos]))
                start_chance.update({i: v[0] for i, v in xm.items()})
                continue
            start_chance.update({i: v[0] for i, v in xm.items()})
            art = pm.refit(data[pos], pos, arts[pos], gw, built[pos])
            d_cut = data[pos][(data[pos]["season"] == "2526") | (data[pos]["gameweek"] < gw)]
            inputs = pm.live_inputs(plist, art, pos, d=d_cut, as_of=start)[:4]
            # his full-game minutes: average of his last 10 games of 60+ at this club BEFORE the gameweek (shrunk)
            f60 = mins_feat[pos]
            f60 = f60[((f60["season"] == "2526") | (f60["gameweek"] < gw)) & (f60["mins"] >= 60)].sort_values("date")
            prior_full = mins_json[pos]["bucket_minutes"]["full"]
            m_full = {}
            for p_ in plist:
                g = f60[(f60["player_id"] == p_["id"]) & (f60["squad_id"] == p_["squadId"])]["mins"].tail(10)
                m_full[p_["id"]] = (g.sum() + 3 * prior_full) / (len(g) + 3)
            cards = card_costs(rows_all[(rows_all["pos"] == pos) & ((rows_all["season"] == "2526") | (rows_all["gameweek"] < gw))])
            xp.update(outfield_xp(pos, art, inputs, plist, fx_by_club, xm, mins_json[pos], cards, m_full, counts=counts))
            xp_lin.update(outfield_xp(pos, art, inputs, plist, fx_by_club, xm_raw, mins_json[pos], cards, m_full))
            got = cur[cur["pos"] == pos].groupby("player_id")[[STAT_COLS[st] for st in art]].sum()
            for st in art:  # expected vs actual count of each stat over everyone registered
                stat_rows.append(dict(gw=gw, pos=pos, stat=st, exp=sum(counts.get(p_["id"], {}).get(st, 0.0) for p_ in plist),
                                      act=float(got[STAT_COLS[st]].sum())))
        # form baseline: average points over the player's last 5 club games before this gameweek (incl. not playing)
        hist = rows_all[(rows_all["season"] == "2526") | (rows_all["gameweek"] < gw)].sort_values(["season", "gameweek"])
        form = {}
        for p in players:
            h = hist[(hist["player_id"] == p["id"]) & (hist["squad_id"] == p["squadId"])].tail(5)
            form[p["id"]] = (h["points"].mean() if len(h) else 0.0) * len(fx_by_club.get(p["squadId"], []))
        realised = {p["id"]: float(real.at[p["id"], "points"]) if p["id"] in real.index else 0.0 for p in players}
        for p in players:  # calibration record: every eligible player, the live xP and the comparison (ALT)
            i = p["id"]
            calib.append(dict(gw=gw, pos=p["pos"], id=i, club=p["squadId"], xp=xp[i], xp_alt=xp_lin[i], pts=realised[i], form=form[i],
                              games=len(fx_by_club.get(p["squadId"], [])), p_start=start_chance.get(i),
                              started=float(first.at[i, "start"]) if i in first.index and i in start_chance else None))
        played = {p["id"]: float(real.at[p["id"], "mins"]) if p["id"] in real.index else 0.0 for p in players}

        def score_team(ids, value):
            order = sorted(ids, key=lambda i: -value[i])
            cap = next((i for i in order if played[i] > 0), None)  # captain, else vice (if the captain did not play)
            cap = cap if cap in order[:2] else None
            return sum(realised[i] for i in ids) + (realised[cap] if cap else 0.0), (order[0] if order else None)
        ids, cap = best_team(players, xp)
        pts, _ = score_team(ids, xp)
        f_ids, f_cap = best_team(players, form)
        f_pts, _ = score_team(f_ids, form)
        h_ids, h_cap = best_team(players, realised)
        h_pts = sum(realised[i] for i in h_ids) + realised[h_cap]
        # clubs: expected club points from the fixtures' expected goals, real points from the results
        club_x, club_r = {}, {}
        for f in fx:
            M = score_matrix(f["lh"], f["la"], RHO)
            for cid, home in ((f["home"], True), (f["away"], False)):
                club_x[cid] = club_x.get(cid, 0.0) + cp.expected_points(M, home)[0]
                if f["hs"] is not None:
                    gf, ga = (f["hs"], f["as_"]) if home else (f["as_"], f["hs"])
                    club_r[cid] = club_r.get(cid, 0) + club_realised(gf, ga, home)
        avail = sorted((c for c in club_x if uses.get(c, 0) < 5), key=lambda c: -club_x[c])[:2]
        for c in avail:
            uses[c] = uses.get(c, 0) + 1
        clubs_pts = sum(club_r.get(c, 0) for c in avail)
        best_clubs = sorted(club_r, key=lambda c: -club_r[c])[:2]
        byid = {p["id"]: p for p in players}
        team = [dict(id=i, name=byid[i]["name"], pos=byid[i]["pos"], club=byid[i]["squadId"], clubName=squads[byid[i]["squadId"]]["shortName"],
                     xp=round(xp[i], 2), points=realised[i], mins=played[i], captain=i == cap) for i in sorted(ids, key=lambda i: POS.index(byid[i]["pos"]))]
        weeks.append(dict(
            gw=gw, start=str(start.date()), team=team,
            clubs=[dict(id=c, name=squads[c]["shortName"], xp=round(club_x[c], 2), points=club_r.get(c, 0)) for c in avail],
            model=dict(xp=round(sum(xp[i] for i in ids) + xp[cap] + sum(club_x[c] for c in avail), 2), players=pts, clubs=clubs_pts, total=pts + clubs_pts),
            form=dict(players=f_pts, total=f_pts + clubs_pts),
            hindsight=dict(players=h_pts, clubs=sum(club_r[c] for c in best_clubs), total=h_pts + sum(club_r[c] for c in best_clubs)),
            marketGames=sum(f["src"] == "market" for f in fx), games=len(fx)))
        print(f"GW{gw}: model {pts + clubs_pts:.0f} (players {pts:.0f} + clubs {clubs_pts}) | xP {weeks[-1]['model']['xp']:.1f} | form picker "
              f"{f_pts + clubs_pts:.0f} | hindsight best {weeks[-1]['hindsight']['total']:.0f}", flush=True)
    c = pd.DataFrame(calib)
    c.to_csv(HERE / "output" / "backtest_players.csv", index=False)  # every player-gameweek, for a closer look (not committed)
    card = scorecard(c, pd.DataFrame(stat_rows), weeks)
    OUT.write_text(json.dumps(dict(generatedAt=datetime.now(timezone.utc).isoformat(timespec="seconds"), season="2026/27", weeks=weeks, scorecard=card,
                                   totals=dict(model=sum(w["model"]["total"] for w in weeks), form=sum(w["form"]["total"] for w in weeks),
                                               hindsight=sum(w["hindsight"]["total"] for w in weeks), xp=round(sum(w["model"]["xp"] for w in weeks), 1))),
                              separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print("wrote", OUT)
    print_scorecard(card)


# ---------- scorecard: how good are the expected points, over every player-gameweek ----------
def _mse(a, b):
    return float(((a - b) ** 2).mean())


def scorecard(c, stats, weeks, n_boot=2000):
    """c: one row per registered player per gameweek (xp, xp_alt, pts, form, p_start, started); stats: expected / actual
    count per gameweek, position and stat; weeks: the per-gameweek results. Everything here is out of sample (each
    gameweek's models were refitted on games before it)."""
    rng = np.random.default_rng(0)
    live = c[(c["xp"] > 0) | (c["xp_alt"] > 0) | (c["form"] > 0)]

    def block(d):
        e = dict(n=int(len(d)), mse=_mse(d["pts"], d["xp"]), mse_alt=_mse(d["pts"], d["xp_alt"]), mse_form=_mse(d["pts"], d["form"]),
                 mean_xp=float(d["xp"].mean()), mean_pts=float(d["pts"].mean()),
                 rank_corr=float(d["xp"].corr(d["pts"], method="spearman")), rank_corr_form=float(d["form"].corr(d["pts"], method="spearman")))
        e["skill_vs_form"] = 1 - e["mse"] / e["mse_form"]
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in e.items()}
    periods = {"all": live, "after_dev": live[live["gw"] > DEV_UNTIL]}
    out = dict(devUntil=DEV_UNTIL, alt=ALT, overall={k: block(d) for k, d in periods.items() if len(d)},
               byPos={pos: block(d) for pos, d in live.groupby("pos")})
    # gameweek-block bootstrap of the MSE difference vs the comparison and vs form (gameweeks share fixtures: resample them whole)
    gws = sorted(live["gw"].unique())
    per = live.assign(d_alt=(live["pts"] - live["xp_alt"]) ** 2 - (live["pts"] - live["xp"]) ** 2,
                      d_form=(live["pts"] - live["form"]) ** 2 - (live["pts"] - live["xp"]) ** 2).groupby("gw").agg(a=("d_alt", "sum"), f=("d_form", "sum"), n=("pts", "size"))
    draws = rng.integers(0, len(gws), (n_boot, len(gws)))
    for key, col in (("mseGainVsAlt", "a"), ("mseGainVsForm", "f")):
        v = per[col].to_numpy()[draws].sum(1) / per["n"].to_numpy()[draws].sum(1)
        out[key] = dict(mean=round(float(per[col].sum() / per["n"].sum()), 4), lo=round(float(np.quantile(v, 0.05)), 4), hi=round(float(np.quantile(v, 0.95)), 4))
    # calibration by xP band (all positions) and of the ten highest xP per position each gameweek (the ones that get picked)
    bands = [0, 0.5, 1, 2, 3, 4, 5, 6, 8, 100]
    b = live.groupby(pd.cut(live["xp"], bands, right=False), observed=True).agg(xp=("xp", "mean"), pts=("pts", "mean"), n=("pts", "size"))
    out["calibration"] = [dict(band=f"{iv.left:g}-{iv.right:g}" if iv.right < 100 else f"{iv.left:g}+", xp=round(r.xp, 3), pts=round(r.pts, 3), n=int(r.n)) for iv, r in b.iterrows()]
    top = live.sort_values("xp", ascending=False).groupby(["gw", "pos"]).head(10)
    out["top10"] = {pos: dict(xp=round(float(d["xp"].mean()), 3), pts=round(float(d["pts"].mean()), 3), n=int(len(d)),
                              form_pts=round(float(live[live["pos"] == pos].sort_values("form", ascending=False).groupby("gw").head(10)["pts"].mean()), 3))
                    for pos, d in top.groupby("pos")}
    # start chances vs what happened (first game of the gameweek)
    s = c.dropna(subset=["p_start", "started"])
    sb = s.groupby(pd.cut(s["p_start"], [0, .05, .2, .4, .6, .75, .85, .92, 1.0001], include_lowest=True), observed=True).agg(
        p=("p_start", "mean"), started=("started", "mean"), n=("started", "size"))
    out["starts"] = [dict(band=f"{max(iv.left, 0):.0%}-{min(iv.right, 1):.0%}", p=round(r.p, 3), started=round(r.started, 3), n=int(r.n)) for iv, r in sb.iterrows()]
    pc = np.clip(s["p_start"], 1e-4, 1 - 1e-4)
    out["startLogLoss"] = round(float(-(s["started"] * np.log(pc) + (1 - s["started"]) * np.log(1 - pc)).mean()), 4)
    # every scoring stat: expected vs actual count over all registered players
    if len(stats):
        g = stats.groupby(["pos", "stat"]).agg(exp=("exp", "sum"), act=("act", "sum")).reset_index()
        late = stats[stats["gw"] > DEV_UNTIL].groupby(["pos", "stat"]).agg(exp=("exp", "sum"), act=("act", "sum"))
        out["stats"] = [dict(pos=r.pos, stat=r.stat, exp=round(r.exp, 1), act=round(r.act, 1), ratio=round(r.act / r.exp, 3),
                             ratioAfterDev=round(float(late.at[(r.pos, r.stat), "act"] / late.at[(r.pos, r.stat), "exp"]), 3) if (r.pos, r.stat) in late.index else None)
                        for r in g.itertuples()]
    # the points totals: how sure is "model beats form"? resample gameweeks
    d = np.array([w["model"]["total"] - w["form"]["total"] for w in weeks], float)
    v = d[rng.integers(0, len(d), (n_boot, len(d)))].sum(1)
    out["points"] = dict(modelMinusForm=float(d.sum()), lo=float(np.quantile(v, 0.05)), hi=float(np.quantile(v, 0.95)), weeksAhead=int((d > 0).sum()), weeks=int(len(d)),
                         afterDev=dict(model=float(sum(w["model"]["total"] for w in weeks if w["gw"] > DEV_UNTIL)),
                                       form=float(sum(w["form"]["total"] for w in weeks if w["gw"] > DEV_UNTIL)),
                                       xp=round(float(sum(w["model"]["xp"] for w in weeks if w["gw"] > DEV_UNTIL)), 1)))
    return out


def print_scorecard(card):
    o = card["overall"]["all"]
    print(f"\nSCORECARD: {o['n']} player-gameweeks. xP error (MSE) {o['mse']:.4f} | form baseline {o['mse_form']:.4f} (skill {o['skill_vs_form']:.1%}) | "
          f"{card['alt']} {o['mse_alt']:.4f}. Mean xP {o['mean_xp']:.3f} vs actual {o['mean_pts']:.3f}; rank correlation {o['rank_corr']:.3f} (form {o['rank_corr_form']:.3f})")
    if "after_dev" in card["overall"]:
        a = card["overall"]["after_dev"]
        print(f"  gameweeks after GW{card['devUntil']} only: MSE {a['mse']:.4f} vs form {a['mse_form']:.4f} (skill {a['skill_vs_form']:.1%}); mean xP {a['mean_xp']:.3f} vs {a['mean_pts']:.3f}")
    g = card["mseGainVsAlt"]
    print(f"  MSE gain over '{card['alt']}': {g['mean']:+.4f} (90% interval {g['lo']:+.4f} to {g['hi']:+.4f}; positive = live is better)")
    print("  by position (MSE / form MSE / mean xP / mean points):", {p: (v["mse"], v["mse_form"], v["mean_xp"], v["mean_pts"]) for p, v in card["byPos"].items()})
    print("  calibration by xP band:", [(r["band"], r["xp"], r["pts"], r["n"]) for r in card["calibration"]])
    print("  ten highest xP per position each gameweek (xP / actual / the form picker's ten):", {p: (v["xp"], v["pts"], v["form_pts"]) for p, v in card["top10"].items()})
    print(f"  start chances (log-loss {card['startLogLoss']}):", [(r["band"], r["p"], r["started"], r["n"]) for r in card["starts"]])
    print("  stats, actual / expected (all gameweeks | after dev):", {f"{r['pos']} {r['stat']}": (r["ratio"], r["ratioAfterDev"]) for r in card.get("stats", [])})
    p = card["points"]
    print(f"  points: model - form {p['modelMinusForm']:+.0f} (90% interval {p['lo']:+.0f} to {p['hi']:+.0f}), ahead in {p['weeksAhead']} of {p['weeks']} gameweeks; "
          f"after GW{card['devUntil']}: model {p['afterDev']['model']:.0f} (expected {p['afterDev']['xp']:.0f}) vs form {p['afterDev']['form']:.0f}")


if __name__ == "__main__":
    main()

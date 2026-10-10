"""Backtest of the whole pick pipeline on 2026/27: for each completed gameweek, refit the models on games BEFORE it, pick
the team the site would have picked, and score it with the real fantasy points.
python backtest.py      writes public/data/backtest.json (the Backtest page)

Per gameweek g (all inputs as of g's first kick-off):
- fixtures: expected goals from the closing odds (the market's pre-match view); games without odds in our data use the
  team-strength model fitted on matches before g (predict_gw.fit_current)
- players: every registered player of that gameweek. Stat models (player_model.py, same features / window / shrinkage as
  live, coefficients refitted on 2025/26 + 2026/27 before g), inputs via the same live_inputs code cut to games before g;
  expected minutes from the minutes model refitted before g (no injury news: harder than live). Keepers: odds-based clean
  sheet / goals conceded / saves, minutes scaled to 90 per club
- team: exact optimum (scipy milp) of sum xP + captain xP; formations 1-2-2-2 / 1-2-3-1 / 1-3-2-1, at most 2 per club
- clubs: the two highest-xP clubs with picks left (each club 5 times a season)
- score: real points; captain doubled (vice-captain if the captain did not play); club points from the real results
Comparisons: a "form" picker (average points over each player's last 5 club games, same rules and same clubs) and the
best possible team in hindsight. Leakage: none known (league step multipliers and FotMob shrinkage targets use completed
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
def minutes_before(pos, feat, gw, mins_json):
    name = mins_json["model"]
    kind = name.split(" + ")[0]
    feats = mins_json["features"]
    tr = feat[(feat["season"] == "2526") | ((feat["season"] == "2627") & (feat["gameweek"] < gw))]
    m = mm.models()[kind]().fit(tr[feats], tr["mins"])
    rows = feat[(feat["season"] == "2627") & (feat["gameweek"] == gw)].sort_values("date")
    first = rows.drop_duplicates("player_id")  # first game of a double: features do not know the first game's outcome
    return dict(zip(first["player_id"].astype(int), np.clip(m.predict(first[feats]), 0, 90)))


# ---------- expected points per player for one gameweek ----------
def outfield_xp(pos, art, inputs, players, fx_by_club, xmins, app_curve, p60_curve, card90):
    roles, styles, fm, mates = inputs
    fm_def = pm.fm_defaults(fm) if fm else {}
    out = {}
    for p in players:
        pid, cid = p["id"], p["squadId"]
        mins = xmins.get(pid, 0.0)
        t = mins / 90
        tot = 0.0
        for f in fx_by_club.get(cid, []):
            if mins <= 0:
                break
            home = f["home"] == cid
            oid = f["away"] if home else f["home"]
            lo, lp = (f["lh"], f["la"]) if home else (f["la"], f["lh"])
            mu = {}
            for stat, a in art.items():
                vals = dict(role=roles.get(pid, {}).get(stat, 1.0), team=styles.get(cid, {}).get(stat, {}).get("team", a["prior"]),
                            opp=styles.get(oid, {}).get(stat, {}).get("opp", a["prior"]), mates=mates.get(pid, {}).get(stat, a["prior"]),
                            lam_own=lo, lam_opp=lp, home=int(home), **fm.get(pid, fm_def))
                mu[stat] = (rate(a, vals) * a["scale"] + (fm.get(pid, fm_def).get("fm_pxg", 0) if a["pens"] else 0)) * t
            app = curve(app_curve, mins, "pts")
            if pos == "DEF":
                pt = kp.fixture_points(lo, lp, home, 1.0, 10.0, dict(pen=0.0, cards=0.0))
                x = app + 5 * pt["cs"] * curve(p60_curve, mins, "p") + pt["gcPts"] * t
                x += sum(nb_pts(mu[s], art[s]["nb_r"], u) for s, u in UNITS.items())
                x += 7 * mu["goals"] + 3 * mu["assists"]
            else:
                sc = SCORING[pos]
                x = app + sc["goal"] * mu["goals"] + 5 * p_at_least(mu["goals"], art["goals"]["nb_r"], 3) + sc["assist"] * mu["assists"]
                x += sc["sot"] * mu["sot"] + nb_pts(mu["kp"], art["kp"]["nb_r"], 2) + (sc["interception"] * mu["int"] if "int" in mu else 0)
            tot += x - card90 * t
        out[pid] = tot
    return out


def keeper_xp(players, fx_by_club, xmins, saves_model, consts):
    out = {}
    for p in players:
        tot = 0.0
        for f in fx_by_club.get(p["squadId"], []):
            home = f["home"] == p["squadId"]
            lo, lp = (f["lh"], f["la"]) if home else (f["la"], f["lh"])
            row = pd.DataFrame([dict(lam_own=lo, lam_opp=lp, home=int(home))])
            mu = float(kp.predict_saves(saves_model, row)[0])
            tot += kp.fixture_points(lo, lp, home, mu, saves_model["nb_r"], consts)["xp"]
        out[p["id"]] = tot * xmins.get(p["id"], 0.0) / 90
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
    done = sorted(r["roundNumber"] for r in rounds if r.get("gameMode", "season") == "season" and r["status"] == "completed")
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
    card90 = {}
    for pos in POS:
        r = rows_all[(rows_all["pos"] == pos) & (rows_all["season"] == "2526")]
        m90 = r["minutes_played"].sum() / 90
        card90[pos] = float((r["yellow_cards"] + 3 * r["red_cards"] + 3 * r["own_goals"] + 3 * r["penalty_misses"]).sum() / m90)
    uses, form_uses = {}, {}
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
        xp = {}
        for pos in POS:
            plist = [dict(p, position=pos) for p in players if p["pos"] == pos]
            xm = minutes_before(pos, mins_feat[pos], gw, mins_json[pos])
            if pos == "GK":  # one keeper plays: scale each club's keepers to 90
                byc = {}
                for p in plist:
                    byc.setdefault(p["squadId"], []).append(p["id"])
                for ids in byc.values():
                    tot = sum(xm.get(i, 0) for i in ids)
                    for i in ids:
                        xm[i] = 90 * xm.get(i, 0) / tot if tot > 0 else 0.0
                xp.update(keeper_xp(plist, fx_by_club, xm, saves_gw, consts))
                continue
            art = pm.refit(data[pos], pos, arts[pos], gw, built[pos])
            d_cut = data[pos][(data[pos]["season"] == "2526") | (data[pos]["gameweek"] < gw)]
            inputs = pm.live_inputs(plist, art, pos, d=d_cut, as_of=start)
            xp.update(outfield_xp(pos, art, inputs, plist, fx_by_club, xm, mins_json[pos]["app_curve"],
                                  mins_json[pos].get("p60_curve"), card90[pos]))
        # form baseline: average points over the player's last 5 club games before this gameweek (incl. not playing)
        hist = rows_all[(rows_all["season"] == "2526") | (rows_all["gameweek"] < gw)].sort_values(["season", "gameweek"])
        form = {}
        for p in players:
            h = hist[(hist["player_id"] == p["id"]) & (hist["squad_id"] == p["squadId"])].tail(5)
            form[p["id"]] = (h["points"].mean() if len(h) else 0.0) * len(fx_by_club.get(p["squadId"], []))
        realised = {p["id"]: float(real.at[p["id"], "points"]) if p["id"] in real.index else 0.0 for p in players}
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
    OUT.write_text(json.dumps(dict(generatedAt=datetime.now(timezone.utc).isoformat(timespec="seconds"), season="2026/27", weeks=weeks,
                                   totals=dict(model=sum(w["model"]["total"] for w in weeks), form=sum(w["form"]["total"] for w in weeks),
                                               hindsight=sum(w["hindsight"]["total"] for w in weeks), xp=round(sum(w["model"]["xp"] for w in weeks), 1))),
                              separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()

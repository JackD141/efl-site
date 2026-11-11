"""Start-chance model (since 7 Nov 2026; before that an expected-minutes ridge regression).
python minutes_model.py [POS ...]  train (2025/26 GW1-23), compare on validation (GW24-35), test once on 2026/27, refit on
                                   all, write models/minutes_model_<pos>.joblib + a JSON summary

What the pages need is not "expected minutes" but the chance of each minutes scenario: he starts and plays his usual
full game, he starts but comes off before 60, he comes off the bench, or he does not play. So three logistic models on the
same features: P(start), P(60+ | started) (his own: a 90-minute regular is not docked for team-mates' early substitutions)
and P(sub appearance | not started) (exp_start.py: better P(60+) log-loss than the ridge +
isotonic curve for every position, on validation and test). Games he most likely missed through SUSPENSION (inferred
from red cards and yellow-card milestones, see suspended()) are left out of training and of his history: at pick time a
ban is known (EFL marks him suspended) and it is not the same as being dropped.
Doubles: the chance he starts the second game is predicted twice, as if he started the first and as if he did not.
A club starts exactly ten outfield players (since 9 Nov 2026, exp_clubstart.py): each club's available defenders,
midfielders and forwards have their start logits shifted by the same amount so the chances add up to ten. So when
team-mates are flagged out by the EFL the others' chances rise, and a crowded squad's chances fall.

One row per player per club fixture while he is registered at the club (the fantasy data lists every squad player,
including 0-minute ones). Target: minutes played (0-90). Features use only his EARLIER games at the same club (spells
carry across seasons): recent minutes and 60+ / appearance rates on several decays, last game's minutes, run of 0-minute
games, games at the club so far (new signings), rest days and whether it is the club's second game of the gameweek.
Baselines: the site's old rule (90 if 60+ in the club's previous game, else 0) and his average over the last 5 club games.
Expected appearance points as a function of expected minutes are fitted on validation (isotonic), so the page can turn
any xMins - predicted or typed in - into appearance points.
"""
import json
import re
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import brentq
from scipy.special import expit, logit
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
# extra features of the P(60+ | started) model only (exp_p60.py, 11 Nov 2026): his own record of lasting 60 minutes in
# his last 10 starts at the club. Midfielders and forwards (the ones managers take off): validation log-loss 0.2972 ->
# 0.2966 and 0.3645 -> 0.3634; nothing for defenders and keepers.
FULL_EXTRA = {"MID": ["q_hist"], "FWD": ["q_hist"]}
Q_STARTS, Q_PSEUDO = 10, 3.0


def suspended(d):
    """Games a player most likely missed through suspension: 0 minutes in the club's next 1-3 games after a red card
    (stopping at his first appearance), or the next game after his 5th / 10th / 15th league yellow of the season (the
    next 2 after the 10th). Checked on 2025/26-26/27: 89% of red-carded players miss the next game, ~45% the next two or
    three; 29% miss the game after a 5th yellow vs 11% of regulars generally. d sorted by player, club, date."""
    d = d.copy()
    d["ycum"] = d.groupby(["player_id", "season"])["yellow_cards"].cumsum()
    milestone = ((d["ycum"] // 5) > ((d["ycum"] - d["yellow_cards"]) // 5)) & (d["ycum"] >= 5)
    ban = np.maximum(np.where(d["red_cards"] > 0, 3, 0), np.where(milestone, np.where(d["ycum"] >= 10, 2, 1), 0))
    susp = np.zeros(len(d), bool)
    keys = (d["player_id"].astype(str) + "|" + d["squad_id"].astype(str)).to_numpy()
    mins = d["minutes_played"].to_numpy()
    for i in np.flatnonzero(ban > 0):
        for j in range(i + 1, min(i + 1 + ban[i], len(d))):
            if keys[j] != keys[i] or mins[j] > 0:
                break
            susp[j] = True
    return susp


def load_rows(position, drop_suspended=True):
    """Every player-fixture row for the position; inferred suspension games dropped unless drop_suspended=False (the
    "last 5" display wants them)."""
    frames = []
    for season, tag in (("2025_26", "2526"), ("2026_27", "2627")):
        for f in sorted((REPO / "data" / season).glob("player_stats_gw*.csv")):
            try:
                frames.append(pd.read_csv(f).assign(season=tag))
            except pd.errors.EmptyDataError:
                pass
    d = pd.concat(frames, ignore_index=True).drop_duplicates(["season", "player_id", "gameweek", "squad_id", "opponent_id"])
    d = attach_fotmob(attach_dates(d[d["position"] == position].copy()))
    if not drop_suspended:
        return d
    d = d.sort_values(["player_id", "squad_id", "date", "gameweek"]).reset_index(drop=True)
    return d[~suspended(d)].reset_index(drop=True)


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
    # his own record of reaching 60 minutes when he starts: share of his last Q_STARTS starts at the club BEFORE this game
    # (whether or not he starts this one: the first version differed for starters, which leakage_check.py caught),
    # shrunk with Q_PSEUDO starts at a fixed 0.9 (about nine starts in ten reach 60; nothing estimated, nothing can leak)
    st = d[d["start"] == 1]
    roll = st.groupby(["player_id", "squad_id"], sort=False)["full"].rolling(Q_STARTS, min_periods=1)
    flat = lambda r: r.reset_index(level=[0, 1], drop=True).reindex(st.index)
    d["_q"] = np.nan
    d.loc[st.index, "_q"] = (flat(roll.sum()) + Q_PSEUDO * 0.9) / (flat(roll.count()) + Q_PSEUDO)  # after each start, that start included
    d["_q"] = d.groupby(["player_id", "squad_id"], sort=False)["_q"].ffill()
    d["q_hist"] = d.groupby(["player_id", "squad_id"], sort=False)["_q"].shift(1).fillna(0.9)  # as of before this game
    d = d.drop(columns="_q")
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


def _next_fixtures(rounds):
    """{club: [(round, opponent, H/A, date), ...]} for each club's games in the first round with a game still to play,
    in date order (two entries for a double)."""
    now = pd.Timestamp.now(tz="UTC")
    out, first_round = {}, {}
    for r in sorted((r for r in rounds if r.get("gameMode", "season") == "season"), key=lambda r: r["roundNumber"]):
        for g in sorted(r["games"], key=lambda g: g["date"]):
            if pd.Timestamp(g["date"]) <= now:
                continue
            for cid, oid, ha in ((g["homeId"], g["awayId"], "H"), (g["awayId"], g["homeId"], "A")):
                first_round.setdefault(cid, r["roundNumber"])
                if r["roundNumber"] == first_round[cid]:
                    out.setdefault(cid, []).append((r["roundNumber"], oid, ha, pd.Timestamp(g["date"][:10])))
    return out


def start_probs(art, hist, upcoming):
    """Start-model probabilities for upcoming games. hist: earlier rows (load_rows); upcoming: [{player_id, squad_id,
    position, fixtures: [(round, opp, H/A, date), ...]}]. Returns {player: dict(pStart, pSubN[, pStart2c, pStart2n])}:
    first game; P(sub appearance | not started); second game of a double if he started / did not start the first."""
    def synth(u, fx, start, tag=None):
        return dict(player_id=u["player_id"], squad_id=u["squad_id"], season="2627", gameweek=fx[0], opponent_id=fx[1], is_home=fx[2],
                    date=fx[3], minutes_played=90 if start else 0, position=u["position"], start=float(start), suboff=0.0,
                    yellow_cards=0, red_cards=0, _tag=tag)
    feats = art["features"]

    def predict(rows):
        d = add_features(pd.concat([hist.assign(_tag=None), pd.DataFrame(rows)], ignore_index=True))
        x = d[d["_tag"].notna()]
        ps = art["start"][1].predict_proba(art["start"][0].transform(x[feats]))[:, 1]
        pb = art["sub"][1].predict_proba(art["sub"][0].transform(x[feats]))[:, 1]
        pq = art["full"][1].predict_proba(art["full"][0].transform(x[art.get("full_features", feats)]))[:, 1]
        return {t: (float(a), float(b), float(c)) for t, a, b, c in zip(x["_tag"], ps, pb, pq)}
    p1 = predict([synth(u, u["fixtures"][0], False, f"{u['player_id']}|1") for u in upcoming])
    out = {u["player_id"]: dict(pStart=p1[f"{u['player_id']}|1"][0], pSubN=p1[f"{u['player_id']}|1"][1], pFull=p1[f"{u['player_id']}|1"][2])
           for u in upcoming}
    dbl = [u for u in upcoming if len(u["fixtures"]) > 1]
    for started, key in ((True, "pStart2c"), (False, "pStart2n")):
        rows = []
        for u in dbl:
            rows.append(synth(u, u["fixtures"][0], started))  # the first game, as if he started it / did not
            rows.append(synth(u, u["fixtures"][1], False, f"{u['player_id']}|2"))
        if rows:
            p2 = predict(rows)
            for u in dbl:
                out[u["player_id"]][key] = p2[f"{u['player_id']}|2"][0]
    return out


OUTFIELD, OUTFIELD_STARTERS = ("DEF", "MID", "FWD"), 10
_PCT = re.compile(r"\((\d+)%\)")
_STARTS_CACHE = {}


def availability(p):
    """Chance he is available at all, from the EFL's flags: 1 if playing; 0 if suspended, or injured with "Out (0%)" or
    no detail; the EFL's own percentage when it gives one ("1 | Questionable (50%)" -> 0.5)."""
    if p.get("suspensionDetails") or p["status"] == "suspended":
        return 0.0
    inj = p.get("injuryDetails")
    if inj:
        m = _PCT.search(inj.get("status") or "")
        return int(m.group(1)) / 100 if m else 0.0
    return 1.0 if p["status"] == "playing" else 0.0


def _shift(f, total):
    """delta with f(delta) = total for an increasing f; +-30 (everybody / nobody) if the total cannot be reached."""
    g = lambda d: float(f(d)) - total
    return brentq(g, -30, 30) if g(30) > 0 > g(-30) else (30.0 if g(30) <= 0 else -30.0)


def shift_to(p, total, a=None):
    """Start chances p (one club, one game) shifted by a common amount on the logit scale so that a x p adds up to
    total (a: availability, default 1)."""
    a = np.ones(len(p)) if a is None else np.asarray(a, float)
    z = logit(np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6))
    return a * expit(z + _shift(lambda d: (a * expit(z + d)).sum(), total))


def normalise_club(pred, avail, total=OUTFIELD_STARTERS):
    """One club's outfield players: {id: dict(pStart, pSubN, pFull[, pStart2c, pStart2n])} -> the same with the start
    chances adding up to `total` (ten outfield starters). avail {id: availability}: 0 -> out (start / bench chances 0),
    0.5 -> start / bench chances halved. The second game of a double: both conditional chances are shifted so the
    unconditional chances (as the pages compute them: s1 x c + (1 - s1) x n) add up to `total` as well."""
    out = {i: dict(v) for i, v in pred.items()}
    ids = [i for i in pred if avail.get(i, 1.0) > 0]
    for i in pred:
        if i not in ids:
            out[i] = {k: (v if k == "pFull" else 0.0) for k, v in pred[i].items()}
    if not ids:
        return out
    a = np.array([avail.get(i, 1.0) for i in ids])
    lg = lambda key: logit(np.clip(np.array([pred[i][key] for i in ids], float), 1e-6, 1 - 1e-6))
    s1 = shift_to([pred[i]["pStart"] for i in ids], total, a)
    for i, v, ai in zip(ids, s1, a):
        out[i]["pStart"], out[i]["pSubN"] = float(v), float(ai * pred[i]["pSubN"])
    if all("pStart2c" in pred[i] for i in ids):
        zc, zn = lg("pStart2c"), lg("pStart2n")
        d2 = _shift(lambda d: (s1 * expit(zc + d) + (1 - s1) * expit(zn + d)).sum(), total)
        for i, c, n in zip(ids, expit(zc + d2), expit(zn + d2)):
            out[i]["pStart2c"], out[i]["pStart2n"] = float(c), float(n)
    return out


def _model_starts(position, players, rounds):
    """The start model's own chances (no flags, no club total) for the position's players with a game to come."""
    art = joblib.load(MODELS / f"minutes_model_{position.lower()}.joblib")
    nxt = _next_fixtures(rounds)
    ups = [dict(player_id=p["id"], squad_id=p["squadId"], position=position, fixtures=nxt[p["squadId"]])
           for p in players if p["position"] == position and p["status"] != "eliminated" and p["squadId"] in nxt]
    return start_probs(art, load_rows(position), ups)


def predict_starts(position, players, rounds):
    """{player id: dict(pStart, pSubN, pFull[, pStart2c, pStart2n])} for each player's club's next round (pFull: P(60+ |
    started)). EFL flags: injured / suspended -> 0, "questionable (50%)" -> halved (availability()). Outfield positions:
    each club's chances add up to ten starters (normalise_club; all three positions are predicted together and cached
    for the run). Keepers: flags only here; keepers.model_minutes scales each club's keepers to one starter."""
    avail = {p["id"]: availability(p) for p in players}
    rnd = lambda out: {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in out.items()}
    if position not in OUTFIELD:
        out = _model_starts(position, players, rounds)
        return rnd({i: {k: (v if k == "pFull" else v * avail.get(i, 1.0)) for k, v in pr.items()} for i, pr in out.items()})
    key = (tuple(sorted((p["id"], p["squadId"], avail[p["id"]]) for p in players if p["status"] != "eliminated")),
           tuple(sorted((c, tuple(map(str, f))) for c, f in _next_fixtures(rounds).items())))
    if key not in _STARTS_CACHE:
        _STARTS_CACHE.clear()
        raw = {pos: _model_starts(pos, players, rounds) for pos in OUTFIELD}
        club = {p["id"]: p["squadId"] for p in players}
        by_club = {}
        for pos in OUTFIELD:
            for i, pr in raw[pos].items():
                by_club.setdefault(club[i], {})[i] = pr
        done = {}
        for preds in by_club.values():
            done.update(normalise_club(preds, avail))
        _STARTS_CACHE[key] = {pos: rnd({i: done[i] for i in raw[pos]}) for pos in OUTFIELD}
    return _STARTS_CACHE[key][position]


def expected_minutes(pr, q, full, part):
    """Expected minutes in the first game from the start probabilities (display and data checks)."""
    ps, q = pr["pStart"], pr.get("pFull", q)
    return ps * (q * full + (1 - q) * part) + (1 - ps) * pr["pSubN"] * part


def plan_fields(pr, m_full, sm):
    """Per-player plan fields: start probabilities (rounded) and the expected minutes they imply (his own full-game
    minutes m_full). sm = startModel (q, full, part)."""
    if pr is None:
        return dict(pStart=0.0, pSubN=0.0, xMins=0)
    out = {k: round(v, 3) for k, v in pr.items()}
    out["xMins"] = int(round(expected_minutes(pr, sm["q"], m_full, sm["part"])))
    return out


def predict_next(position, players, rounds):
    """Expected minutes in each player's club's next fixture (derived from the start model): {player id: minutes}."""
    js = json.loads((MODELS / f"minutes_model_{position.lower()}.json").read_text(encoding="utf-8"))
    b = js["bucket_minutes"]
    return {k: expected_minutes(v, js["q60"], b["full"], b["part"]) for k, v in predict_starts(position, players, rounds).items()}


def recent_minutes(position, players, n=5):
    """Minutes in each player's last n games for his CURRENT club (0 = in the squad but did not play), oldest first."""
    d = load_rows(position, drop_suspended=False).sort_values(["date", "gameweek"])
    cur = {p["id"]: p["squadId"] for p in players if p["position"] == position}
    d = d[d["player_id"].map(cur) == d["squad_id"]]
    return {int(pid): [int(m) for m in g["minutes_played"].tail(n)] for pid, g in d.groupby("player_id")}


def full_minutes(position, players, prior, n=10, k=3.0):
    """His average minutes in his last n games of 60+ at his CURRENT club, shrunk to the position average with k
    pseudo-games: the minutes of the "plays the full game" scenario (a nailed 90-minute man gets ~90, not ~84)."""
    d = load_rows(position).sort_values(["date", "gameweek"])
    cur = {p["id"]: p["squadId"] for p in players if p["position"] == position}
    d = d[(d["player_id"].map(cur) == d["squad_id"]) & (d["minutes_played"] >= 60)]
    out = {}
    for pid, g in d.groupby("player_id"):
        m = g["minutes_played"].clip(upper=90).tail(n)
        out[int(pid)] = round(float((m.sum() + k * prior) / (len(m) + k)), 1)
    return out


def mixture_inputs(position, players):
    """(plan fields, {player id: full-game minutes}) for the start-scenario mixture used by the pages: startModel =
    q (share of starts reaching 60), full / part (average minutes in a 60+ game / a 1-59 game)."""
    js = json.loads((MODELS / f"minutes_model_{position.lower()}.json").read_text(encoding="utf-8"))
    b = js["bucket_minutes"]
    return dict(startModel=dict(q=js["q60"], full=b["full"], part=b["part"]), minsMix=b), full_minutes(position, players, b["full"])


def models():
    return {
        "ridge": lambda: Ridge(alpha=1.0),
        "gbm": lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0,
                                                     random_state=0),
    }


def evaluate(y, p):
    p = np.clip(p, 0, 90)
    return dict(mae=float(np.abs(y - p).mean()), rmse=float(np.sqrt(((y - p) ** 2).mean())), mean_pred=float(p.mean()), mean_actual=float(y.mean()))


def app_points(m):
    return np.where(m >= 60, 2, np.where(m > 0, 1, 0))


def _logit(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    return sc, LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X), y)


def _ll(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def fit_start(tr, feats, full_extra=()):
    """Start model fitted on rows tr: dict(start=(scaler, logit) P(start), full=(scaler, logit) P(60+ | started) on
    starters (features + full_extra), sub=(scaler, logit) P(appearance | not started) on non-starters, q = average
    P(60+ | started), features, full_features)."""
    ns, st = tr[tr["start"] == 0], tr[tr["start"] == 1]
    ff = list(feats) + list(full_extra)
    return dict(start=_logit(tr[feats], tr["start"]), sub=_logit(ns[feats], (ns["mins"] > 0).astype(int)),
                full=_logit(st[ff], (st["mins"] >= 60).astype(int)), features=feats, full_features=ff, q=float((st["mins"] >= 60).mean()))


def scenario_probs(art, x):
    """(P(start), P(full game), P(part game)) for rows x."""
    X = x[art["features"]]
    ps = art["start"][1].predict_proba(art["start"][0].transform(X))[:, 1]
    pb = art["sub"][1].predict_proba(art["sub"][0].transform(X))[:, 1]
    q = art["full"][1].predict_proba(art["full"][0].transform(x[art.get("full_features", art["features"])]))[:, 1] if "full" in art else art["q"]
    return ps, ps * q, ps * (1 - q) + (1 - ps) * pb


def main(position="MID"):
    d = add_features(load_rows(position))  # suspension games dropped (load_rows)
    feats = FEATURES + FM_FEATURES
    tr = d[(d["season"] == "2526") & (d["gameweek"] <= 23)]
    va = d[(d["season"] == "2526") & (d["gameweek"] > 23)]
    te = d[d["season"] == "2627"]
    print(f"{position}: {len(d)} player-fixtures (train {len(tr)}, validation {len(va)}, test {len(te)}), suspension games left out")
    p60 = lambda s_: (s_["mins"] >= 60).astype(float)
    # the earlier model (ridge expected minutes -> isotonic P(60+)), for the record
    rid = Ridge(alpha=1.0).fit(tr[feats], tr["mins"])
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(np.clip(rid.predict(tr[feats]), 0, 90), p60(tr))
    art = fit_start(tr, feats, FULL_EXTRA.get(position, ()))
    bucket = dict(full=round(float(tr.loc[tr["mins"] >= 60, "mins"].mean()), 1),
                  part=round(float(tr.loc[(tr["mins"] > 0) & (tr["mins"] < 60), "mins"].mean()), 1))
    res = {}
    for lab, s_ in (("validation", va), ("test", te)):
        ps, pf, pp = scenario_probs(art, s_)
        xm = pf * bucket["full"] + pp * bucket["part"]
        res[lab] = dict(start_ll=_ll(s_["start"], ps), p60_ll=_ll(p60(s_), pf), p60_ll_ridge=_ll(p60(s_), iso.predict(np.clip(rid.predict(s_[feats]), 0, 90))),
                        mins_rmse=float(np.sqrt(((s_["mins"] - xm) ** 2).mean())), mean_start_pred=float(ps.mean()), mean_start=float(s_["start"].mean()))
        print(f"  {lab}: P(start) log-loss {res[lab]['start_ll']:.4f}; P(60+) log-loss {res[lab]['p60_ll']:.4f} (ridge+isotonic "
              f"{res[lab]['p60_ll_ridge']:.4f}); minutes RMSE {res[lab]['mins_rmse']:.2f}; mean P(start) {ps.mean():.3f} vs {s_['start'].mean():.3f}")
    final = fit_start(d, feats, FULL_EXTRA.get(position, ()))
    MODELS.mkdir(exist_ok=True)
    joblib.dump(final, MODELS / f"minutes_model_{position.lower()}.joblib")
    (MODELS / f"minutes_model_{position.lower()}.json").write_text(json.dumps(dict(
        model="logistic P(start) + P(sub | not started)", features=feats, full_extra=FULL_EXTRA.get(position, []), q60=round(final["q"], 4), bucket_minutes=bucket, results=res,
        suspensions_left_out=True), indent=1), encoding="utf-8")
    print(f"  q (starts reaching 60) {final['q']:.3f}; minutes in a 60+ / 1-59 game {bucket}")
    return final


if __name__ == "__main__":
    for pos_ in (sys.argv[1:] or ["MID"]):
        main(pos_.upper())

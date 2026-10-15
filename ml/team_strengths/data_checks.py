"""Data integrity checks for everything we stitch together (EFL fantasy stats, EFL fixtures, football-data.co.uk odds and
results, FotMob, the site JSON). Run by weekly_update.py; on its own:

    python data_checks.py            # inputs + outputs, for the next gameweek
    python data_checks.py --inputs   # inputs only

Each check prints PASS / WARN / FAIL with a reason. FAIL means something is joined or parsed wrongly and the predictions
must not be published until it is fixed; WARN means look at it. Checks compare sources against each other, so a name,
date, timestamp or orientation (home / away) slip in one source shows up as a disagreement.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "ml" / "fotmob"))
SEASON_DIR = REPO / "data" / "2026_27"
COMP_DIV = {10: "E1", 11: "E2", 12: "E3"}
RESULTS = []


def report(status, name, detail=""):
    RESULTS.append((status, name, detail))
    print(f"  [{status}] {name}" + (f": {detail}" if detail else ""), flush=True)


def load_efl():
    rounds = [r for r in json.load(open(HERE / "cache" / "rounds.json", encoding="utf-8")) if r.get("gameMode", "season") == "season"]
    squads = {s["id"]: s for s in json.load(open(HERE / "cache" / "squads.json", encoding="utf-8"))}
    return rounds, squads


def finished_rounds(rounds):
    return [r for r in rounds if r["games"] and all(g.get("status") in ("completed", "postponed") for g in r["games"])
            and any(g.get("status") == "completed" for g in r["games"])]


def fantasy_rows():
    frames = []
    for f in sorted(SEASON_DIR.glob("player_stats_gw*.csv")):
        try:
            frames.append(pd.read_csv(f))
        except pd.errors.EmptyDataError:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------- fantasy points recomputed from the stats (scoring verified 9 Oct 2026) ----------
def recompute_points(d):
    m = d["minutes_played"]
    app = np.where(m >= 60, 2, np.where(m > 0, 1, 0))
    common = 3 * d["assists"] + 5 * d["hat_tricks"] - 3 * d["penalty_misses"] - 3 * d["own_goals"] - d["yellow_cards"] - 3 * d["red_cards"]
    cs = 5 * ((d["clean_sheet"] > 0) & (m >= 60))
    gc = -(d["goals_conceded"] // 2)
    pos = d["position"]
    pts = np.select(
        [pos == "GK", pos == "DEF", pos == "MID", pos == "FWD"],
        [cs + gc + 2 * (d["saves"] // 3) + 5 * d["penalty_saves"] + 10 * d["goals_scored"],
         cs + gc + d["clearances"] // 4 + d["blocks"] // 2 + d["tackles"] // 2 + 7 * d["goals_scored"],
         6 * d["goals_scored"] + 2 * d["interceptions"] + d["key_passes"] // 2 + d["shots_on_target"],
         5 * d["goals_scored"] + d["key_passes"] // 2 + d["shots_on_target"]], 0)
    return np.where(m > 0, app + pts + common, d["points"])  # players who did not play: whatever the EFL says (0)


def check_fantasy(rounds, squads):
    print("EFL fantasy stats")
    d = fantasy_rows()
    fin = finished_rounds(rounds)
    missing = [r["roundNumber"] for r in fin if not (SEASON_DIR / f"player_stats_gw{r['roundNumber']}.csv").exists()
               or (SEASON_DIR / f"player_stats_gw{r['roundNumber']}.csv").stat().st_size < 1000]
    report("FAIL" if missing else "PASS", "every finished gameweek has a stats file", f"missing / empty: {missing}" if missing else f"{len(fin)} gameweeks")
    bad = []
    for r in fin:
        want = set()
        for g in r["games"]:
            if g.get("status") == "completed":
                want |= {(g["homeId"], g["awayId"], "H"), (g["awayId"], g["homeId"], "A")}
        got = set(map(tuple, d.loc[d["gameweek"] == r["roundNumber"], ["squad_id", "opponent_id", "is_home"]].drop_duplicates().to_numpy().tolist()))
        if want - got or got - want:
            bad.append(f"GW{r['roundNumber']}: {len(want - got)} club-games missing, {len(got - want)} unexpected")
    report("FAIL" if bad else "PASS", "stats cover exactly the finished games (both clubs, right home / away)", "; ".join(bad) or "all match the EFL fixtures")
    dup = d.duplicated(["player_id", "gameweek", "squad_id", "opponent_id"]).sum()
    report("FAIL" if dup else "PASS", "no duplicate player-game rows", f"{dup} duplicates" if dup else "")
    unknown = sorted(set(d["squad_id"]) - set(squads))
    report("FAIL" if unknown else "PASS", "every club id is a current EFL squad", f"unknown {unknown}" if unknown else "")
    played = d[d["minutes_played"] > 0]
    mism = (recompute_points(played) != played["points"]).mean() if len(played) else 0
    report("WARN" if mism > 0.01 else "PASS", "fantasy points recompute from the stats", f"{mism:.2%} of appearances differ" + (" (EFL corrections or a scoring change?)" if mism > 0.01 else ""))
    return d


# ---------- football-data (odds / results) vs the EFL ----------
def check_football_data(rounds, squads, fixtures_csv, gw):
    from per_season import lam_for
    from predict_gw import market_lambdas, resolver
    from strengths import DIVS
    print("football-data.co.uk (results, odds) against the EFL")
    res = resolver(squads)
    name_to_id, problems, wrong_div = {}, [], []
    for div in DIVS:
        m = lam_for(div, "2627")
        for n in pd.concat([m["HomeTeam"], m["AwayTeam"]]).unique():
            i = res(n)
            if i is None:
                problems.append(f"{n} ({div}) does not match any EFL club")
                continue
            name_to_id[n] = i
            if COMP_DIV.get(squads[i].get("competitionId")) != div:
                wrong_div.append(f"{n} -> {squads[i]['name']} is not in {div}")
    rev = {}
    for n, i in name_to_id.items():
        rev.setdefault(i, []).append(n)
    dupes = [f"{squads[i]['name']} <- {ns}" for i, ns in rev.items() if len(ns) > 1]
    report("FAIL" if problems or dupes else "PASS", "every football-data team name maps to exactly one EFL club",
           "; ".join(problems + dupes) or f"{len(name_to_id)} names")
    report("FAIL" if wrong_div else "PASS", "each team's football-data division is its EFL league", "; ".join(wrong_div[:5]))
    # results: same date and same score as the EFL, for every finished game both sources have
    efl = {}
    for r in rounds:
        for g in r["games"]:
            if g.get("status") == "completed":
                efl[(g["homeId"], g["awayId"])] = (pd.Timestamp(g["date"]).tz_convert("Europe/London").date(), g.get("homeScore"), g.get("awayScore"))
    date_bad, score_bad, n_both = [], [], 0
    for div in DIVS:
        m = lam_for(div, "2627")
        for row in m.itertuples():
            key = (res(row.HomeTeam), res(row.AwayTeam))
            if key not in efl:
                continue
            n_both += 1
            dt, hs, as_ = efl[key]
            if abs((pd.Timestamp(row.Date).date() - dt).days) > 1:
                date_bad.append(f"{row.HomeTeam} v {row.AwayTeam}: {pd.Timestamp(row.Date).date()} vs EFL {dt}")
            if hs is not None and (int(row.FTHG), int(row.FTAG)) != (int(hs), int(as_)):
                score_bad.append(f"{row.HomeTeam} v {row.AwayTeam}: {int(row.FTHG)}-{int(row.FTAG)} vs EFL {hs}-{as_}")
    report("FAIL" if date_bad else "PASS", "football-data result dates match the EFL fixture dates", "; ".join(date_bad[:5]) or f"{n_both} games")
    report("FAIL" if score_bad else "PASS", "football-data scores match the EFL scores (catches swapped / mis-named teams)", "; ".join(score_bad[:5]))
    # upcoming odds: every priced game of the gameweek is a real EFL fixture, same orientation and date
    fx = pd.read_csv(fixtures_csv, encoding="utf-8-sig")
    rnd = next((r for r in rounds if r["roundNumber"] == gw), None)
    games = {(g["homeId"], g["awayId"]): pd.Timestamp(g["date"]).tz_convert("Europe/London").date() for g in rnd["games"]} if rnd else {}
    reversed_, date_off, unknown = [], [], []
    for row in fx[fx["Div"].isin(DIVS)].itertuples():
        h, a = res(row.HomeTeam), res(row.AwayTeam)
        if h is None or a is None:
            unknown.append(f"{row.HomeTeam} v {row.AwayTeam}")
            continue
        dt = pd.to_datetime(row.Date, dayfirst=True).date()
        if (a, h) in games and (h, a) not in games and abs((dt - games[(a, h)]).days) <= 3:
            reversed_.append(f"{row.HomeTeam} v {row.AwayTeam}")
        if (h, a) in games and abs((dt - games[(h, a)]).days) > 1:
            date_off.append(f"{row.HomeTeam} v {row.AwayTeam}: {dt} vs EFL {games[(h, a)]}")
    report("FAIL" if unknown else "PASS", "every fixture-odds team name resolves", "; ".join(unknown[:5]))
    # names are matched by unique prefix: a name never seen in the results could match the wrong club, so flag it
    new = sorted({n for n in pd.concat([fx.loc[fx["Div"].isin(DIVS), "HomeTeam"], fx.loc[fx["Div"].isin(DIVS), "AwayTeam"]])
                  if n not in name_to_id and res(n) is not None} )
    report("WARN" if new else "PASS", "fixture-odds team names are ones already seen in the results",
           "; ".join(f"{n} -> {squads[res(n)]['name']}" for n in new[:8]) + (" (check these are the right clubs)" if new else ""))
    report("FAIL" if reversed_ else "PASS", "no priced fixture has home / away swapped vs the EFL", "; ".join(reversed_[:5]))
    report("FAIL" if date_off else "PASS", f"priced GW{gw} fixtures are on the EFL date", "; ".join(date_off[:5]))
    ml = market_lambdas(fixtures_csv, squads)
    odd = [f"{squads[k[0]]['name']} v {squads[k[1]]['name']}: {v[0]:.2f}-{v[1]:.2f}" for k, v in ml.items() if not (0.2 <= v[0] <= 4.5 and 0.2 <= v[1] <= 4.5)]
    report("FAIL" if odd else "PASS", "odds-implied expected goals are in a sane range (0.2 to 4.5)", "; ".join(odd[:5]) or f"{len(ml)} priced games")


# ---------- FotMob vs the EFL ----------
def check_fotmob(rounds, squads, fantasy):
    import link_fotmob as lf
    print("FotMob against the EFL")
    fm_dir = REPO / "data" / "fotmob" / "2026_2027"
    if not (fm_dir / "matches.csv").exists():
        report("FAIL", "FotMob data present", "data/fotmob/2026_2027 missing (run fetch_fotmob.py)")
        return
    mt = pd.read_csv(fm_dir / "matches.csv")
    tm = lf.team_map(list(squads.values()), set(mt["home"]) | set(mt["away"]))
    unmapped = sorted((set(mt["home"]) | set(mt["away"])) - set(tm))
    report("FAIL" if unmapped else "PASS", "every FotMob team maps to an EFL club", f"unmapped {unmapped}" if unmapped else f"{len(tm)} teams")
    ts = pd.to_datetime(mt["utc"], utc=True, format="ISO8601", errors="coerce")
    report("FAIL" if ts.isna().any() else "PASS", "FotMob kick-off timestamps parse", f"{int(ts.isna().sum())} unparseable" if ts.isna().any() else "")
    mt["hid"], mt["aid"], mt["day"] = mt["home"].map(tm), mt["away"].map(tm), ts.dt.tz_convert("Europe/London").dt.date
    fmk = {(r.hid, r.aid): r for r in mt.itertuples()}
    missing, date_bad, score_bad = [], [], []
    for r in finished_rounds(rounds) + [x for x in rounds if x["status"] != "completed"]:
        for g in r["games"]:
            if g.get("status") != "completed":
                continue
            m = fmk.get((g["homeId"], g["awayId"]))
            name = f"{squads[g['homeId']]['shortName']} v {squads[g['awayId']]['shortName']}"
            if m is None:
                missing.append(name)
                continue
            if abs((m.day - pd.Timestamp(g["date"]).tz_convert("Europe/London").date()).days) > 1:
                date_bad.append(f"{name}: FotMob {m.day}")
            if isinstance(m.score, str) and "-" in m.score and g.get("homeScore") is not None:
                hs, as_ = [int(x) for x in m.score.replace(" ", "").split("-")]
                if (hs, as_) != (int(g["homeScore"]), int(g["awayScore"])):
                    score_bad.append(f"{name}: FotMob {hs}-{as_} vs EFL {g['homeScore']}-{g['awayScore']}")
    report("WARN" if missing else "PASS", "every finished EFL game is in FotMob (same home / away)", f"{len(missing)} missing: {missing[:5]}" if missing else "")
    report("FAIL" if date_bad else "PASS", "FotMob match dates match the EFL", "; ".join(date_bad[:5]))
    report("FAIL" if score_bad else "PASS", "FotMob scores match the EFL scores", "; ".join(score_bad[:5]))
    # player links: coverage and agreement on minutes / goals
    f = lf.fotmob_for_efl()
    e = fantasy[fantasy["minutes_played"] > 0].assign(season="2627")
    j = e.merge(f, on=["season", "player_id", "squad_id", "opponent_id", "is_home"], how="left")
    cover = j["fm_minutes"].notna().mean()
    linked = j[j["fm_minutes"].notna()]
    mins_ok = ((linked["fm_minutes"] - linked["minutes_played"]).abs() <= 3).mean()
    goals_ok = (linked["fm_goals"].fillna(0) == linked["goals_scored"]).mean()
    worst = j.groupby("gameweek")["fm_minutes"].apply(lambda s: s.notna().mean()).sort_values().head(1)
    report("FAIL" if cover < 0.9 else "WARN" if cover < 0.95 else "PASS", "EFL appearances linked to a FotMob player",
           f"{cover:.1%} (lowest gameweek: GW{worst.index[0]} {worst.iloc[0]:.1%})")
    report("FAIL" if mins_ok < 0.95 else "PASS", "linked players' minutes agree (within 3)", f"{mins_ok:.1%}")
    report("FAIL" if goals_ok < 0.98 else "PASS", "linked players' goals agree", f"{goals_ok:.1%}")


# ---------- model data: dates ----------
def check_model_dates():
    import minutes_model as mm
    print("dates used by the models")
    d = mm.load_rows("MID")
    cur = d[d["season"] == "2627"]
    start = pd.Timestamp("2026-08-07")
    fallback = (cur["date"] == start + pd.to_timedelta(7 * (cur["gameweek"] - 1), unit="D")) & (cur["gameweek"] > 1)
    report("WARN" if fallback.mean() > 0.01 else "PASS", "2026/27 fantasy rows matched to a real match date (not the gameweek fallback)",
           f"{fallback.mean():.1%} fell back" if fallback.any() else "all matched")


# ---------- outputs: the site JSON ----------
def check_outputs(rounds, squads, gw):
    print("site data (public/data)")
    load = lambda f: json.loads((REPO / "public" / "data" / f).read_text(encoding="utf-8"))
    plans = {f: load(f) for f in ("club_plan.json", "keeper_plan.json", "defender_plan.json", "mid_plan.json", "fwd_plan.json")}
    first = {f: p.get("firstGw") for f, p in plans.items()}
    report("FAIL" if len(set(first.values())) != 1 or gw not in first.values() else "PASS", f"every plan starts at GW{gw}", str(first))
    rnd = next(r for r in rounds if r["roundNumber"] == gw)
    want = {}
    for g in rnd["games"]:
        for cid, oid, ha in ((g["homeId"], g["awayId"], "H"), (g["awayId"], g["homeId"], "A")):
            want.setdefault(cid, []).append((squads[oid]["name"], ha, pd.Timestamp(g["date"])))
    bad, ko_bad = [], []
    for c in plans["club_plan.json"]["clubs"]:
        w = next((x for x in c["weeks"] if x["gw"] == gw), None)
        got = sorted((f["opp"], f["ha"]) for f in (w["fx"] if w else []))
        exp = sorted((o, ha) for o, ha, _ in want.get(c["id"], []))
        if got != exp:
            bad.append(f"{c['short']}: site {got} vs EFL {exp}")
        for f in (w["fx"] if w else []):
            hits = [t for o, ha, t in want.get(c["id"], []) if o == f["opp"] and ha == f["ha"]]
            if hits and f.get("ko") and pd.Timestamp(f["ko"]) != hits[0]:
                ko_bad.append(f"{c['short']} v {f['opp']}: {f['ko']} vs EFL {hits[0]}")
    report("FAIL" if bad else "PASS", f"each club's GW{gw} fixtures on the site match the EFL (opponent, home / away)", "; ".join(bad[:3]) or f"{len(want)} clubs")
    report("FAIL" if ko_bad else "PASS", "kick-off times on the site match the EFL", "; ".join(ko_bad[:3]))
    probs = [abs(f["win"] + f["draw"] + (1 - f["win"] - f["draw"]) - 1) for c in plans["club_plan.json"]["clubs"] for w in c["weeks"] for f in w["fx"]]
    sane = [f for c in plans["club_plan.json"]["clubs"] for w in c["weeks"] for f in w["fx"] if not (0 <= f["win"] <= 1 and 0 <= f["draw"] <= 1 and f["win"] + f["draw"] <= 1.0001)]
    report("FAIL" if sane else "PASS", "club win / draw probabilities are valid", f"{len(sane)} invalid" if sane else f"{len(probs)} fixtures")
    mins_bad = [(f, x["name"]) for f, p in plans.items() if f != "club_plan.json"
                for x in (p.get("keepers") or p.get("defenders") or p.get("players") or []) if not 0 <= x["xMins"] <= 90]
    report("FAIL" if mins_bad else "PASS", "expected minutes are between 0 and 90", str(mins_bad[:3]))
    m = plans["mid_plan.json"]
    starters = sum(x["xMins"] >= 60 for x in m["players"])
    report("WARN" if not 100 <= starters <= 300 else "PASS", "plausible number of midfielders expected to play 60+", str(starters))


def run(inputs=True, outputs=True, gw=None, fixtures_csv=None):
    import fetch_data
    RESULTS.clear()
    rounds, squads = load_efl()
    if gw is None:
        now = pd.Timestamp.now(tz="UTC")
        gw = next(r["roundNumber"] for r in rounds if r["status"] != "completed" and any(pd.Timestamp(g["date"]) > now for g in r["games"]))
    print(f"\n---------- data checks (GW{gw}) ----------")
    if inputs:
        fantasy = check_fantasy(rounds, squads)
        check_football_data(rounds, squads, fixtures_csv or fetch_data.CACHE / "fd_fixtures.csv", gw)
        check_fotmob(rounds, squads, fantasy)
        check_model_dates()
    if outputs:
        check_outputs(rounds, squads, gw)
    fails = [r for r in RESULTS if r[0] == "FAIL"]
    warns = [r for r in RESULTS if r[0] == "WARN"]
    print(f"---------- {len(RESULTS)} checks: {len(fails)} FAIL, {len(warns)} WARN ----------")
    (HERE / "output").mkdir(exist_ok=True)
    (HERE / "output" / "data_checks.txt").write_text("\n".join(f"[{s}] {n}: {d}" for s, n, d in RESULTS), encoding="utf-8")
    return fails, warns


if __name__ == "__main__":
    fails, _ = run(inputs="--outputs" not in sys.argv, outputs="--inputs" not in sys.argv)
    sys.exit(1 if fails else 0)

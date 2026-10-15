"""The whole weekly refresh, in order. See docs/weekly-runbook.md for the steps around it (export, commit, push).

    python weekly_update.py                     # next gameweek with a game still to kick off
    python weekly_update.py 10                  # a given gameweek
    python weekly_update.py --override overrides/gw10_betfair.csv     # hand-pasted prices for this run
    python weekly_update.py --no-retrain --no-backtest                # quick re-run (e.g. after a late team-news edit)
    python weekly_update.py --allow-missing-odds                      # go ahead although the odds are not up yet

Before running: the fantasy stats for every finished game must be in data/2026_27 (Export Stats / the export API, then
git pull). This script checks that and stops if a finished gameweek is missing or empty.
1. odds + results (football-data.co.uk), EFL rounds / squads / players. STOPS if fewer than 80% of the gameweek's first set
   of games (a double gameweek has two) have odds yet: ask Jack whether to wait for football-data, paste Betfair prices
   (--override) or go ahead on the team-strength model (--allow-missing-odds)
2. FotMob: new matches -> link players -> set pieces; then DATA CHECKS (data_checks.py): EFL stats complete and
   consistent (points recompute), every team name maps to one club in the right league, football-data and FotMob dates
   and scores agree with the EFL, priced fixtures are real EFL fixtures (same home / away, same date), FotMob links
   cover 95%+ with agreeing minutes and goals. Any FAIL stops the run
3. retrain on everything to date: minutes models (all positions), stat models (MID / FWD / DEF). The model CHOICES are
   re-made on the 2025/26 validation folds exactly as before (deterministic); only the coefficients see the new games
4. the gameweek's predictions: club_plan / keeper / defender / mid / fwd JSON for the site (run_gameweek.py), then
   checks on them: same gameweek everywhere, each club's fixtures and kick-off times equal the EFL's, valid probabilities,
   minutes in 0-90
5. backtest over every finished gameweek (each refitted on games before it) -> backtest.json
6. reference xP for checking the pages' maths (check_site_maths.py -> public/data/check_xp.json, not committed)
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "ml" / "fotmob"))


def step(title):
    print(f"\n==================== {title} ====================", flush=True)
    return time.time()


def check_stats(rounds):
    """Every gameweek whose games have all finished must have a non-empty stats file."""
    missing = []
    for r in rounds:
        if r.get("gameMode", "season") != "season":
            continue
        finished = r["games"] and all(g.get("status") in ("completed", "postponed") for g in r["games"]) and any(g.get("status") == "completed" for g in r["games"])
        if not finished:
            continue
        f = REPO / "data" / "2026_27" / f"player_stats_gw{r['roundNumber']}.csv"
        if not f.exists() or f.stat().st_size < 1000:
            missing.append(r["roundNumber"])
    return missing


MIN_PRICED = 0.8  # share of the first set of fixtures that must have market odds before we run without asking


def odds_coverage(rounds, squads, gw, override_files=()):
    """Which of the gameweek's games have market odds (football-data fixtures, plus any override CSVs). The gameweek is
    split into sets at gaps of 2+ days (a double gameweek: weekend + midweek). Returns (gw, [(set label, games, priced)])."""
    import fetch_data
    import overrides as ov
    from predict_gw import market_lambdas
    season = [r for r in rounds if r.get("gameMode", "season") == "season"]
    if gw is None:
        now = pd.Timestamp.now(tz="UTC")
        gw = next(r["roundNumber"] for r in season if r["status"] != "completed" and any(pd.Timestamp(g["date"]) > now for g in r["games"]))
    csv = fetch_data.CACHE / "fd_fixtures.csv"
    if override_files:
        csv, _ = ov.apply_overrides(csv, override_files, squads, fetch_data.CACHE / "fd_fixtures_overridden.csv")
    priced = set(market_lambdas(csv, squads))
    games = sorted(next(r for r in season if r["roundNumber"] == gw)["games"], key=lambda g: g["date"])
    sets, cur = [], []
    for g in games:
        if cur and (pd.Timestamp(g["date"][:10]) - pd.Timestamp(cur[-1]["date"][:10])).days >= 2:
            sets.append(cur); cur = []
        cur.append(g)
    if cur:
        sets.append(cur)
    out = []
    for i, st in enumerate(sets):
        label = f"set {i + 1} ({st[0]['date'][:10]} to {st[-1]['date'][:10]})"
        out.append((label, len(st), sum((g["homeId"], g["awayId"]) in priced for g in st)))
    return gw, out


def main(gw=None, overrides=(), retrain=True, backtest=True, allow_missing_odds=False):
    import fetch_data
    t = step("1. odds, results, EFL rounds / squads / players")
    fetch_data.football_data()
    rounds, _ = fetch_data.efl()
    fetch_data.efl_players()
    missing = check_stats(rounds)
    if missing:
        raise SystemExit(f"Fantasy stats missing or empty for finished gameweek(s) {missing}: run the export (docs/weekly-runbook.md "
                         "step 1), git pull, then re-run.")
    print(f"  done in {time.time() - t:.0f}s; fantasy stats present for every finished gameweek")
    squads = {s_["id"]: s_ for s_ in _}
    gw_check, cover = odds_coverage(rounds, squads, gw, overrides)
    print(f"  GW{gw_check} market odds: " + "; ".join(f"{lab}: {p} of {n} games" for lab, n, p in cover))
    first_label, first_n, first_p = cover[0]
    if first_p < MIN_PRICED * first_n and not allow_missing_odds:
        raise SystemExit(
            f"\nSTOP: only {first_p} of {first_n} games in GW{gw_check} {first_label} have market odds yet. Ask Jack (docs/weekly-runbook.md):\n"
            "  - wait for football-data.co.uk to post them (usually by Friday for weekend games), then re-run;\n"
            "  - or paste Betfair prices: overrides.py import ..., then re-run with --override overrides/<file>.csv;\n"
            "  - or go ahead with the team-strength model for the unpriced games: re-run with --allow-missing-odds.")

    import fetch_fotmob
    import link_fotmob
    import set_pieces
    t = step("2. FotMob: new matches, player links, set pieces")
    fetch_fotmob.run(["2026/2027"])
    link_fotmob.build_map()
    set_pieces.main()
    print(f"  done in {time.time() - t:.0f}s")

    import data_checks
    t = step("2b. data checks: stitching EFL / football-data / FotMob (stops on FAIL)")
    import fetch_data as fd_
    fixtures_csv = fd_.CACHE / ("fd_fixtures_overridden.csv" if overrides else "fd_fixtures.csv")
    fails, warns = data_checks.run(inputs=True, outputs=False, gw=gw_check, fixtures_csv=fixtures_csv)
    if fails:
        raise SystemExit(f"STOP: {len(fails)} data check(s) failed (output/data_checks.txt). Fix the join / parse before publishing.")

    if retrain:
        import minutes_model
        import player_model
        t = step("3. retrain minutes models and stat models on everything to date")
        for pos in ("MID", "FWD", "DEF", "GK"):
            minutes_model.main(pos)
        for pos in ("MID", "FWD", "DEF"):
            player_model.main(pos)
        print(f"  done in {time.time() - t:.0f}s")

    import run_gameweek
    t = step("4. predictions for the site")
    gw, fx, clubs = run_gameweek.main(gw, list(overrides))
    print(f"  GW{gw}: done in {time.time() - t:.0f}s")
    t = step("4b. checks on the site data (stops on FAIL)")
    fails, warns = data_checks.run(inputs=False, outputs=True, gw=gw)
    if fails:
        raise SystemExit(f"STOP: {len(fails)} check(s) on the site data failed (output/data_checks.txt). Do not push.")

    if backtest:
        import backtest as bt
        t = step("5. backtest over finished gameweeks")
        bt.main()
        print(f"  done in {time.time() - t:.0f}s")

    import check_site_maths
    step("6. reference xP for checking the pages")
    check_site_maths.main()
    print("\nNext (docs/weekly-runbook.md): check the pages locally (python -m http.server 8765 --directory public), then commit\n"
          "public/data/*.json, ml/team_strengths/models/*, and push to deploy. Delete public/data/check_xp.json afterwards.")
    return gw


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Weekly refresh: data, retrain, predictions, backtest.")
    ap.add_argument("gw", nargs="?", type=int)
    ap.add_argument("--override", nargs="+", default=[], metavar="CSV")
    ap.add_argument("--no-retrain", action="store_true")
    ap.add_argument("--no-backtest", action="store_true")
    ap.add_argument("--allow-missing-odds", action="store_true", help="run even if most of the first set of games has no odds yet")
    a = ap.parse_args()
    main(a.gw, a.override, retrain=not a.no_retrain, backtest=not a.no_backtest, allow_missing_odds=a.allow_missing_odds)

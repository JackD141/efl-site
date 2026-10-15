"""The whole weekly refresh, in order. See docs/weekly-runbook.md for the steps around it (export, commit, push).

    python weekly_update.py                     # next gameweek with a game still to kick off
    python weekly_update.py 10                  # a given gameweek
    python weekly_update.py --override overrides/gw10_betfair.csv     # hand-pasted prices for this run
    python weekly_update.py --no-retrain --no-backtest                # quick re-run (e.g. after a late team-news edit)

Before running: the fantasy stats for every finished game must be in data/2026_27 (Export Stats / the export API, then
git pull). This script checks that and stops if a finished gameweek is missing or empty.
1. odds + results (football-data.co.uk), EFL rounds / squads / players
2. FotMob: new matches -> link players -> set pieces
3. retrain on everything to date: minutes models (all positions), stat models (MID / FWD / DEF). The model CHOICES are
   re-made on the 2025/26 validation folds exactly as before (deterministic); only the coefficients see the new games
4. the gameweek's predictions: club_plan / keeper / defender / mid / fwd JSON for the site (run_gameweek.py)
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


def main(gw=None, overrides=(), retrain=True, backtest=True):
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

    import fetch_fotmob
    import link_fotmob
    import set_pieces
    t = step("2. FotMob: new matches, player links, set pieces")
    fetch_fotmob.run(["2026/2027"])
    link_fotmob.build_map()
    set_pieces.main()
    print(f"  done in {time.time() - t:.0f}s")

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
    a = ap.parse_args()
    main(a.gw, a.override, retrain=not a.no_retrain, backtest=not a.no_backtest)

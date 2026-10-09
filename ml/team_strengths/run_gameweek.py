"""Weekly run.   python run_gameweek.py [gameweek]      (no argument = the first gameweek with a game still to kick off)

1. download latest results/odds (football-data.co.uk) + EFL rounds/squads
2. refit this season's per-league team strengths from the odds so far
3. predict every fixture of the gameweek: games with market odds use what those odds imply; the rest use the strengths model
4. convert to expected Fantasy EFL club points
Optional: --override overrides/<file>.csv swaps pasted prices (see overrides.py) in for that run.
Also writes public/data/keeper_plan.json, defender_plan.json, mid_plan.json and fwd_plan.json (keeper / defender pages; need data/2026_27 stats up to date for starters).
Outputs to output/: gw<N>_fixtures.csv, gw<N>_club_points.csv, ratings_<season>.csv, and public/data/club_plan.json
(every remaining gameweek, for the site's Club Planner page)
"""
import argparse
from pathlib import Path

import pandas as pd

import attackers
import defenders
import fetch_data
import keepers
import overrides
import plot_ratings
import season_plan
from predict_gw import CURRENT, OUT, bookmaker_probs, bookmaker_sources, club_table, fit_current, market_lambdas, predict_gameweek
from strengths import DIVS

pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200, "display.max_colwidth", 80)


def main(gw=None, override_files=()):
    print("Downloading data ...")
    fetch_data.football_data()
    rounds, squads_list = fetch_data.efl()
    squads = {s["id"]: s for s in squads_list}
    season_rounds = [r for r in rounds if r.get("gameMode", "season") == "season"]
    if gw is None:
        now = pd.Timestamp.now(tz="UTC")  # first gameweek that still has a game to kick off (locking is game by game)
        gw = next(r["roundNumber"] for r in season_rounds if r["status"] != "completed" and any(pd.Timestamp(g["date"]) > now for g in r["games"]))
    as_of = pd.Timestamp.today().normalize()
    print(f"Gameweek {gw}; fitting strengths as of {as_of.date()} ...")
    fixtures_csv = fetch_data.CACHE / "fd_fixtures.csv"
    if override_files:  # swap in hand-pasted prices (e.g. Betfair Exchange) for the matching fixtures, this run only
        fixtures_csv, report = overrides.apply_overrides(fixtures_csv, override_files, squads, fetch_data.CACHE / "fd_fixtures_overridden.csv")
        print(f"Overrides from {', '.join(Path(f).name for f in override_files)}:")
        for r in report:
            if r["status"] != "overridden":
                print(f"  {r['match']}: {r['status']}")
            else:
                d = max(abs(a - b) for a, b in zip(r["before"], r["after"])) * 100
                print(f"  {r['match']}: mid odds {r['mid_odds'][0]:.3f} / {r['mid_odds'][1]:.3f} / {r['mid_odds'][2]:.3f}  (largest change vs bookmaker average {d:.1f}pp)")
    fits, id2fd, ratings = fit_current(as_of, squads, fixtures_csv=fixtures_csv)
    ratings.to_csv(OUT / f"ratings_{CURRENT}.csv", index=False)
    plot_ratings.plot(ratings, squads, OUT / f"strengths_{CURRENT}.png", f"20{CURRENT[:2]}/{CURRENT[2:]}")
    book = bookmaker_probs(fixtures_csv, squads)
    book_src = bookmaker_sources(fixtures_csv, squads)
    market_lam = market_lambdas(fixtures_csv, squads)
    fx = predict_gameweek(rounds, squads, gw, fits, id2fd, book, book_src, market_lam)
    plan = season_plan.build_plan(rounds, squads, fits, id2fd, book, as_of, DIVS, book_src, market_lam)
    print(f"Season plan: {len(plan['gameweeks'])} gameweeks -> {season_plan.write_site_json(plan)}")
    players = fetch_data.efl_players()
    kp = keepers.build_keeper_plan(rounds, squads, players, fits, id2fd, book, book_src, market_lam, DIVS)
    print(f"Keeper plan: {len(kp['keepers'])} keepers -> {keepers.write_site_json(kp)}")
    dp = defenders.build_defender_plan(rounds, squads, players, fits, id2fd, book, book_src, market_lam, DIVS)
    print(f"Defender plan: {len(dp['defenders'])} defenders ({sum(x['xMins'] > 0 for x in dp['defenders'])} expected starters) -> {defenders.write_site_json(dp)}")
    for pos in ("MID", "FWD"):
        ap = attackers.build_plan(pos, rounds, squads, players, fits, id2fd, book, book_src, market_lam, DIVS)
        print(f"{pos} plan: {len(ap['players'])} players ({sum(x['xMins'] > 0 for x in ap['players'])} expected starters) -> {attackers.write_site_json(ap)}")
    if kp["latestCompletedGw"] > kp["startersFromGw"]:
        print(f"  WARNING: starters are based on local stats up to GW{kp['startersFromGw']} but GW{kp['latestCompletedGw']} has finished."
              " Press Export Stats on the Player Stats page and git pull, then re-run.")
    games, clubs = club_table(fx)
    fx.to_csv(OUT / f"gw{gw}_fixtures.csv", index=False)
    clubs.to_csv(OUT / f"gw{gw}_club_points.csv", index=False)
    return gw, fx, clubs


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Weekly club expected-points run.")
    ap.add_argument("gw", nargs="?", type=int, help="gameweek (default: next one whose deadline has not passed)")
    ap.add_argument("--override", nargs="+", default=[], metavar="CSV", help="override CSV(s) made by `overrides.py import` (e.g. Betfair prices)")
    args = ap.parse_args()
    gw, fx, clubs = main(args.gw, args.override)
    n_mkt = int((fx["source"] == "market").sum())
    print(f"\nOdds coverage: {n_mkt}/{len(fx)} games have bookmaker odds in football-data's fixtures file; the rest are model estimates.")
    for d_, g_ in fx[fx["source"] == "model"].groupby("date"):
        print(f"  model-only on {d_}: {len(g_)} games ({', '.join(sorted(g_['league'].unique()))})  -> re-run once football-data lists them")
    show = fx[["date", "league", "home", "away", "source", "oddsH", "oddsD", "oddsA"]].round(2)
    print(f"\nGW{gw} model decimal odds (fair, no margin)\n{show.to_string(index=False)}")
    print(f"\nTop clubs by expected points, GW{gw}\n{clubs.head(15).round(2).to_string(index=False)}")

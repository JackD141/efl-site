"""Weekly run.   python run_gameweek.py [gameweek]      (no argument = the next gameweek whose pick deadline has not passed)

1. download latest results/odds (football-data.co.uk) + EFL rounds/squads
2. refit this season's per-league team strengths from the odds so far
3. predict every fixture of the gameweek (works for midweek games with no bookmaker odds yet)
4. convert to expected Fantasy EFL club points
Outputs to output/: gw<N>_fixtures.csv, gw<N>_club_points.csv, ratings_<season>.csv, and public/data/club_plan.json
(every remaining gameweek, for the site's Club Planner page)
"""
import sys
from pathlib import Path

import pandas as pd

import fetch_data
import plot_ratings
import season_plan
from predict_gw import CURRENT, OUT, club_table, fit_current, bookmaker_probs, predict_gameweek
from strengths import DIVS

pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200, "display.max_colwidth", 80)


def main(gw=None):
    print("Downloading data ...")
    fetch_data.football_data()
    rounds, squads_list = fetch_data.efl()
    squads = {s["id"]: s for s in squads_list}
    season_rounds = [r for r in rounds if r.get("gameMode", "season") == "season"]
    if gw is None:
        now = pd.Timestamp.now(tz="UTC")  # the gameweek you can still pick for: first one whose deadline has not passed
        gw = next(r["roundNumber"] for r in season_rounds if r["status"] != "completed" and pd.Timestamp(r["lockoutDate"]) > now)
    as_of = pd.Timestamp.today().normalize()
    print(f"Gameweek {gw}; fitting strengths as of {as_of.date()} ...")
    fits, id2fd, ratings = fit_current(as_of, squads, fixtures_csv=fetch_data.CACHE / "fd_fixtures.csv")
    ratings.to_csv(OUT / f"ratings_{CURRENT}.csv", index=False)
    plot_ratings.plot(ratings, squads, OUT / f"strengths_{CURRENT}.png", f"20{CURRENT[:2]}/{CURRENT[2:]}")
    book = bookmaker_probs(fetch_data.CACHE / "fd_fixtures.csv", squads)
    fx = predict_gameweek(rounds, squads, gw, fits, id2fd, book)
    plan = season_plan.build_plan(rounds, squads, fits, id2fd, book, as_of, DIVS)
    print(f"Season plan: {len(plan['gameweeks'])} gameweeks -> {season_plan.write_site_json(plan)}")
    games, clubs = club_table(fx)
    fx.to_csv(OUT / f"gw{gw}_fixtures.csv", index=False)
    clubs.to_csv(OUT / f"gw{gw}_club_points.csv", index=False)
    return gw, fx, clubs


if __name__ == "__main__":
    gw, fx, clubs = main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
    n_mkt = int((fx["source"] == "market").sum())
    print(f"\nOdds coverage: {n_mkt}/{len(fx)} games have bookmaker odds in football-data's fixtures file; the rest are model estimates.")
    for d_, g_ in fx[fx["source"] == "model"].groupby("date"):
        print(f"  model-only on {d_}: {len(g_)} games ({', '.join(sorted(g_['league'].unique()))})  -> re-run once football-data lists them")
    show = fx[["date", "league", "home", "away", "source", "oddsH", "oddsD", "oddsA"]].round(2)
    print(f"\nGW{gw} model decimal odds (fair, no margin)\n{show.to_string(index=False)}")
    print(f"\nTop clubs by expected points, GW{gw}\n{clubs.head(15).round(2).to_string(index=False)}")

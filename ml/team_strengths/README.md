# Team strengths and expected club points

Full context for the whole Club Planner feature (decisions, evidence, gotchas, state of play): `docs/club-planner.md`.

Our own model of how strong every EFL club is, used to price fixtures and to work out how many Fantasy EFL points
each club is expected to score in a gameweek. It exists because bookmaker odds are not available for every fixture in
time (e.g. midweek games in a double gameweek), and because the fantasy club scoring needs more than win/draw/loss:
it needs clean-sheet and goals-scored probabilities.

## Run it every gameweek

```bash
cd ml/team_strengths
../../venv/Scripts/python.exe run_gameweek.py        # next gameweek that has not completed
../../venv/Scripts/python.exe run_gameweek.py 9      # or a specific one
```

Takes about 5 seconds. It does steps 1–5 below and writes to `output/`:

| File | Contents |
|---|---|
| `gw<N>_fixtures.csv` | every fixture: model win/draw/loss probabilities and fair decimal odds, expected goals, clean-sheet / 2+ / 4+ goal probabilities, expected club points for each side, de-vigged bookmaker probabilities where they exist |
| `gw<N>_club_points.csv` | one row per club: games, expected points, expected clean sheets/wins, fixture list with per-game expected points |
| `public/data/club_plan.json` | every remaining gameweek for every club (xP, rank among all 72 clubs, fixtures, market/model flag) and each club's top-5 weeks; read by the site's Club Planner page (`public/clubs.html`) |
| `ratings_2627.csv` | attack and defence strength for every club |
| `strengths_2627.png` | attack vs defence chart per league (`plot_ratings.py`) |

## The process

1. **Data** (`fetch_data.py`).
   - Results and closing odds per division (E1 Championship, E2 League 1, E3 League 2) from football-data.co.uk,
     `mmz4281/<season>/<div>.csv`. Three past seasons (2023/24–2025/26) are downloaded once; the current season is
     refreshed every run. These files hold goals, shots, cards and odds from many bookmakers.
   - Upcoming fixtures with odds: `football-data.co.uk/fixtures.csv`. It only lists the next few days, so midweek
     games of a double gameweek usually have no odds yet. That gap is the reason for the model.
   - EFL Fantasy `rounds.json` and `squads.json` (public, no login) for the gameweek's fixtures and squad ids.
   - Club names differ between sources; `ALIASES` in `predict_gw.py` fixes the ten that don't match.
2. **Odds to expected goals** (`per_season.py`). For every played match, de-vig the closing average 1X2 and
   over/under 2.5 odds, then find the home and away expected goals (a Poisson score model with the Dixon–Coles
   low-score correction, rho = -0.06) that reproduce those four probabilities.
3. **Strengths** (`per_season.py: fit_strengths`). One model per league, one fit per season; team strengths are
   not carried over between seasons. Weighted ridge regression on the log expected goals:
   `log(lambda) = base + home_advantage*[home side] + attack[team] - conceded[opponent]`.
   Matches are weighted by recency with a **15-day half-life** (team strength moves fast: injuries, form, lineups)
   and a light ridge (0.1). Upcoming fixtures that already have bookmaker odds (the weekend, before a double
   gameweek's midweek games are priced) are added as observations too, so the ratings reflect the market's latest
   view. The output is an attack and a defence number per team plus the league's home advantage.
4. **Predict** (`predict_gw.py`). For each fixture: expected goals for both sides from the two teams' strengths,
   the full score-probability matrix, then win/draw/loss, clean sheet, 2+ and 4+ goals, over 2.5.
5. **Expected club points** (`club_points.py`). See scoring below. A club with two fixtures scores both.
6. **Use the market where it exists.** For games that already have bookmaker or exchange odds, trust those. The
   model's job is the games without odds (midweek) and the clean-sheet / goals probabilities the fantasy scoring
   needs. Because the weekend odds are fed into the fit, model prices for those games are *not* independent of the
   bookmakers' prices.

## Club Planner page

`run_gameweek.py` also writes `public/data/club_plan.json`: expected club points for every remaining gameweek (blank
weeks count 0, doubles sum both games), each club's rank among all 72 clubs that week, and its 5 best weeks.
`public/clubs.html` + `clubs.js` show the best clubs for any gameweek, a season-at-a-glance table, and a popup per
club with its top 5 weeks (rank badge: 1 = the best club that week) and an xP-by-gameweek bar chart. To refresh the
live page: run `run_gameweek.py`, then commit and push `public/data/club_plan.json`.

**Weekly rhythm.** The page always works on the gameweek you can still pick for: the first one whose deadline (first
kick-off) has not passed. Suggestions are for that gameweek only; once you have added both picks it says the picks are
set and waits. After the deadline passes, re-run and push: the page then moves on to the next gameweek. Entered picks
persist in the browser, so each week is two clicks ("Add to picks" on each suggested club). In a double gameweek,
re-run again once the midweek games are priced to replace model estimates with market odds.
Hovering a fixture shows the model's (and, where listed, the bookmakers') decimal win/draw/lose odds; hovering an xP
number shows where it comes from (each outcome probability x its points).

**My picks & season plan** (Jack and John profiles, saved in the browser's local storage; export/import, and a passphrase-protected
**cloud backup** via `api/club-picks.js` so picks can be loaded in another browser or a private window; see
`docs/club-planner.md`):
enter the two clubs picked in each gameweek so far. That gives each club's picks left (shown as a "Picks left" column)
and drives an optimiser: choose 2 clubs per remaining gameweek, each club at most its picks left, to maximise total
xP. It is an assignment problem solved exactly (min-cost flow in the browser, checked against an independent LP solver
to the decimal in two scenarios). It reports the suggested picks for this week, the other options this week with the
season-plan cost of each, the full plan, and the gain over picking greedily week by week. It ignores risk: see the
caveats on the page (fixtures change, strength estimates change, variance, covariance e.g. two clubs facing each
other).

## Fantasy EFL club scoring

Source: fantasy.efl.com help centre, *Game Guidelines*. Checked against the public club results
(`/json/fantasy/squad_profiles/<id>.json`): 0 mismatches over all 574 club-games of 2026/27 GW1–8.

| Action | Points |
|---|---|
| Win | +5 |
| Draw | +3 |
| Away win | +2 (on top of the win) |
| Clean sheet | +2 |
| 2+ goals scored | +2 |
| 4+ goals scored | +2 (stacks with 2+, so 4+ goals = +4) |

Maximum in one game is 13 (away win, clean sheet, 4+ goals). You pick 2 clubs a week, each club at most 5 times a
season; no captain on clubs. A club in a double gameweek earns two sets of points.

## How we got here (what was tested)

1. Plain rolling Poisson attack/defence model on goals only. No better than predicting league-average results.
2. Switched to your framing: fit strengths per season, fitted to the bookmakers' odds rather than to goals.
   Strengths plus home advantage explain 85–90% of the variation in odds-implied expected goals; reproduced win
   probabilities are within 3–4pp of the market. Home advantage is about x1.2 expected goals in every league.
3. Team ratings persist year to year (correlation about 0.7), but once 8 gameweeks of odds exist, last season's
   ratings add nothing, so no prior is used.
4. Operational test (refit weekly from earlier odds only, predict the next round; 2024/25 + 2025/26, 2,751
   matches, scored against the bookmakers' closing odds). With a 60-day memory the model averaged **2.7pp** from the
   market's probabilities and 10% of games were more than 8pp off; log-loss was worse than the market by 0.008.
   **It does not beat the bookmakers.** An early "within 2.4pp" figure compared an average over games against the
   football-data bookmaker average and hid single-game gaps of 7–10pp. Quote the spread, not just the mean.
5. Shorter memory is the biggest single improvement: half-life 60d -> 15d cuts the mean gap to **2.2pp** and the share
   of games more than 8pp off to 5.5%. Below 15 days there is little further gain.
6. Team-specific home advantage (a boost per club): no help, dropped.
7. Adding the odds of upcoming fixtures: for midweek games, knowing the weekend's odds cut the gap a little
   (2.74pp -> 2.63pp at a 60-day memory). On GW9 against Betfair mid-prices (12 Championship games): 3.1pp before,
   2.1pp after, worst single probability 10.7pp -> 5.9pp; the football-data bookmaker average itself is 1.5pp from
   Betfair, so that is the floor for a one-source comparison. Small sample; the backtest is the better guide.
8. What is left (about 2pp on average) is match-specific information (lineups, injuries, rest, derbies) that a
   per-club strength cannot know. Do not expect more from tuning; the next gains are more information (below).
9. Horizon test (predict 1–12 weeks ahead, 2024/25 + 2025/26): the 15-day memory is the most accurate at every
   horizon, so one set of current ratings drives the whole-season plan. The gap to the market grows with distance:
   2.2pp next week, 2.5–2.9pp 2–4 weeks out, 3.1–3.4pp at 5–8 weeks, 3.8pp at 9–12 weeks. Blending in a slower
   rating (120 days) did not help. Beyond 12 weeks is untested; there the schedule (doubles, blanks) matters more
   than the strengths.
10. Cross-checks of the scoring formula and the expected-points maths (simulation agrees to 0.003).

Experiment scripts: `tune.py`, `run_per_season.py`, `run_cutoff.py`, `run_rolling.py`, `run_midweek.py`, `run_variants.py`, `run_horizon.py`, `check_markets.py`, `strengths.py`
(the original goals-only model, still used for the check in `run_per_season.py`).

## Odds source: football-data.co.uk is enough

The weekly pull already gets everything from football-data's `fixtures.csv`: for each upcoming game, 1X2 odds from
Pinnacle (`PP`), Betfair Exchange (`BFE`), Bet365 and a market average/max, over/under 2.5 (Bet365, Betfair Exchange,
average) and Asian handicap. No account, key or scraping needed.

What it does **not** have, and why that is fine:

- **Clean sheet, over 4.5, team totals, both-teams-to-score: not in the file.** `check_markets.py` shows they are not
  needed: inverting closing 1X2 + over/under 2.5 to expected goals and reading off the club bonuses is well
  calibrated over 4,968 matches (2023/24–2025/26): expected club points 4.36 vs actual 4.31, with the match holding
  across low- to high-scoring clubs. Small biases: clean sheet predicted 28.1% vs 27.0% actual, 4+ goals 5.2% vs 4.6%
  (about 0.05 points per club-game in total, ignorable next to the 6-point spread between best and worst fixtures).
- **Sharper source: not needed.** Closing log-loss vs results: Pinnacle 1.0350, Betfair Exchange 1.0357, Bet365
  1.0361, market average 1.0357. All within 1.5pp of each other on GW9 (Betfair Exchange 1.1pp, Pinnacle 1.6pp,
  average 1.5pp from the Betfair prices you pasted). We use the average.
- **Games not yet listed.** The file only lists the next few days (as of Friday 16:45 it held the 36 weekend games and
  none of the Tuesday/Wednesday ones). Those games are model estimates until football-data lists them.

**Weekly routine for a double gameweek**

1. Run `run_gameweek.py` at the start of the gameweek. Games that football-data lists use market odds
   (`source = market`) and feed the strengths fit; the rest are model estimates (`source = model`). The run prints
   how many games have market odds and which dates are still model-only.
2. Re-run it each day (or at least once the weekend has been played) until the later games appear in the file;
   each run swaps model estimates for market odds as they appear. We do not yet know how many days before kick-off
   football-data lists midweek games; the coverage line shows it, so note when they first appear.
3. If a game matters and is still model-only, check Betfair/Pinnacle by hand; if you paste prices in, we can add a
   small override file.

Other routes (Betfair or Pinnacle APIs, The Odds API) would only add markets we showed we do not need, or earlier
midweek prices. Pinnacle's own API has been closed to new users since July 2025 (access by request to
api@pinnacle.com) and pinnacle.com is blocked in the tooling here, so it is not planned.

## Automating the refresh (not built yet)

Idea: a GitHub Actions workflow that runs `run_gameweek.py` and commits the regenerated `public/data/club_plan.json`
(Vercel then redeploys). Steps and open questions:

1. First a manually triggered workflow (a "Run workflow" button on GitHub), so a refresh needs no local terminal.
2. Needs a small requirements file for the pipeline (numpy, pandas, scipy, scikit-learn, requests, matplotlib) and
   permission for the workflow to commit to `main`.
3. Open question to test: whether fantasy.efl.com's public `rounds.json`/`squads.json` and football-data.co.uk answer
   requests from GitHub's runners (they work from a normal connection).
4. Then a schedule: shortly after each gameweek's deadline (to move the page on), plus daily in double-gameweek
   weeks until football-data lists the midweek games. We still need to see how many days before kick-off
   football-data lists them (the run prints this).
5. Saved picks live in each browser. Moving them to a shared store (for example a small file in the repo written by
   an API route, like the stats export does) would make them follow Jack and John across devices.

## Known limits

- No lineup, injury, suspension or rotation information; the bookmakers have it. Expect an average gap of about
  2pp to the market, with single games occasionally 5pp+ off. Re-check midweek games against real odds once they
  appear, and treat the model's midweek prices as an estimate, not a price.
- Early in a season each team has few matches (about 8 at GW9), so ratings are noisy. Promoted clubs have no history.
- Independent Poisson goals with a low-score correction: draws and totals are approximate.
- Ratings are only comparable within a league (no cross-division matches).
- football-data.co.uk is a free third-party source; keep use light and credit it.

## Ideas for next steps

- Betfair exchange mid-prices are sharper than the football-data average and could replace it as the anchor; there
  is no clean free feed, so for now they would have to be pasted in.
- Add team news (starting lineups, injuries) as an input; that is the information the model is missing.
- Add shots on target (available in the same files) to the strength fit; less noisy than goals.
- A "slow strength + fast form" split instead of a single recency weight.
- Feed clean-sheet and goals probabilities into the defender/goalkeeper and attacker points models.
- Show this on the picks page and refresh it automatically each gameweek.

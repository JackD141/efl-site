# Club Planner: context, decisions and state

Written so a new session can pick up without the conversation. Status at time of writing (9 Oct 2026, GW9 week):
Club Planner is live at https://dexters-corner.vercel.app/clubs.html and Jack is happy with it. Process detail and model
results are also in `ml/team_strengths/README.md`.

## 1. What it is and why

Fantasy EFL lets you pick **2 clubs per gameweek** (plus 7 players), and **each club at most 5 times a season**. Clubs
score from results, so the question each week is "which two clubs, and which weeks should I save each club for?".
The Club Planner answers that with expected points (xP) from our own team-strength model, for every remaining
gameweek, plus a per-person picks tracker and an optimiser.

Club scoring (fantasy.efl.com help centre, "Game Guidelines"): win +5, draw +3, away win +2 (on top of the win),
clean sheet +2, 2+ goals scored +2, 4+ goals scored +2 (stacks with 2+, max 13 a game). A club with two fixtures in a
gameweek scores both (doubles); a club with no fixture scores 0 (blanks). Pick deadline for a gameweek = its first
kick-off (`lockoutDate` in `rounds.json`); clubs lock individually once they have played. 39 regular rounds,
many with doubles/blanks (e.g. GW20 Boxing Day: every club plays twice). Play-off rounds are ignored.
**Verified** against the public club results (`https://fantasy.efl.com/json/fantasy/squad_profiles/<squadId>.json`,
no login): the formula reproduces all 574 club-games of 2026/27 GW1-8 with zero mismatches.

## 2. How the numbers are produced (`ml/team_strengths/`)

Weekly: `run_gameweek.py` = download data, fit strengths, predict fixtures, expected club points, write
`public/data/club_plan.json` (+ CSVs/PNG in `output/`, git-ignored).

1. **Data** (`fetch_data.py`): football-data.co.uk results+odds per division (E1 Championship, E2 League 1, E3
   League 2) for 2023/24-2026/27, `fixtures.csv` (upcoming games with odds), and EFL `rounds.json`/`squads.json`.
   Team names differ between sources; `ALIASES` in `predict_gw.py` maps the ten that do not match.
2. **Odds to expected goals** (`per_season.py`): de-vig closing 1X2 and over/under 2.5 odds, solve for the home/away
   expected goals of a Poisson score model with a Dixon-Coles low-score correction (rho -0.06) that reproduce them.
3. **Strengths**: one model per league, one fit per season (strengths are NOT carried between seasons). Weighted ridge
   regression `log(lambda) = base + home + attack[team] - conceded[opponent]` on the odds-implied expected goals,
   recency half-life **15 days**, ridge 0.1. Upcoming fixtures that already have odds are included as observations.
4. **Predict** each fixture: score matrix -> win/draw/lose, clean sheet, 2+ and 4+ goals. A game that has market odds uses
   the expected goals those odds imply (the market beats the model); only games without odds use the strengths model.
5. **Expected club points** (`club_points.py`) from those probabilities and the scoring table.
6. **Season plan** (`season_plan.py`): every gameweek whose deadline has not passed; per club per week xP, rank among
   all 72 clubs, fixtures with their probabilities, market/model flag, and the club's top-5 weeks.

### Decisions and the evidence behind them

| Decision | Why (evidence) |
|---|---|
| Own model rather than only bookmaker odds | Midweek games of a double gameweek have no odds when picks must be made; the model fills the gap and gives clean-sheet / 2+ / 4+ probabilities the scoring needs |
| Odds source = football-data.co.uk `fixtures.csv` | Free, no key, already includes Pinnacle (`PP`), Betfair Exchange (`BFE`), Bet365, average/max 1X2, O/U 2.5 and Asian handicap for the next few days. All sources equally sharp (closing log-loss Pinnacle 1.0350, BFE 1.0357, Bet365 1.0361, avg 1.0357) and within ~1.5pp of Jack's pasted Betfair prices. We use the average |
| No clean-sheet / over 4.5 / team-total markets | `check_markets.py`: over 4,968 matches (2023/24-2025/26), club points implied by closing 1X2 + O/U 2.5 average 4.36 vs 4.31 actual, calibrated from low to high buckets; clean sheet predicted 28.1% vs 27.0%, 4+ goals 5.2% vs 4.6% (about 0.05 pts per club-game). Extra markets are not needed |
| Not Pinnacle's site/API | Pinnacle closed its public API to new users in July 2025 (access by request to api@pinnacle.com); pinnacle.com is blocked in the browser tool and automated scraping is likely against its terms. Do not scrape |
| Per-season fits, no last-season prior | Team strength correlates ~0.7 year to year, but once ~8 gameweeks of odds exist, last season's ratings add nothing (tested) |
| 15-day recency half-life | Weekly-refit backtest (2024/25 + 2025/26, 2,751 matches): 60d -> 2.66pp mean gap to closing odds, 15d -> 2.24pp; games >8pp off fell from 10% to 5.5%. Best at every horizon tested (1-12 weeks), blending a slower rating did not help |
| No team-specific home advantage | Tested, no gain |
| Games with odds use market-implied expected goals, not the model | Market beats the model in every backtest. Found on 9 Oct 2026: the pipeline had been labelling games "market" while using model-smoothed numbers; fixed alongside the Betfair override (model-vs-market xP had no net bias: 4.374 vs 4.381 over 36 games) |
| Add upcoming-fixture odds to the fit | Midweek backtest small gain (2.74 -> 2.63pp at 60d); on GW9 vs Betfair (12 Championship games) mean gap 3.1pp -> 2.1pp, worst 10.7 -> 5.9pp. Side effect: model prices for games that have odds are not independent of the bookmakers |
| One set of current ratings for the whole season plan | Horizon test above: error grows gently (2.2pp next week, 3.8pp 9-12 weeks out); beyond 12 weeks untested, where schedule (doubles/blanks) matters more than strengths |

**Accuracy statement to use (and not exceed):** the model does **not** beat the bookmakers (log-loss ~0.008-0.012 worse
than closing odds); typical gap to market ~2.2pp next week, up to ~4pp three months out, with occasional single
games 5pp+ off. It has no lineup/injury information. An earlier summary said it was "within 2.4pp" on average;
Jack compared against Betfair, found single-game gaps (Sheffield Utd v Lincoln model 54% vs Betfair 43%), and the
claim was corrected. Always quote the spread and worst cases.

## 3. The page (`public/clubs.html`, `clubs.js`, styles `cp-*` in `style.css`)

- **Best clubs for any gameweek** (selector, league filter, search): rank among all 72 (gold/silver/bronze badges, 1 =
  best that week), fixtures with xP each, total xP, **Picks left** column (for the active profile), "Top-5 week" star
  (is this one of the club's 5 best weeks), the club's best week, tags (double / locked / picked / plan pick),
  a dot per fixture (green = bookmaker odds, hollow = model estimate).
- **Hover a fixture**: stacked decimal win/draw/lose odds, model and (where listed) market. **Hover an xP number**: each
  outcome probability x points = the xP (for doubles it shows expected counts, not summed percentages).
- **Club popup**: the club's top 5 weeks with rank badge, fixtures, xP, and an xP-by-gameweek bar chart.
- **Season at a glance**: top 3 clubs each gameweek.
- **My picks & season plan** (profiles **Jack** and **John**): enter the two clubs picked per gameweek so far; gives picks
  left per club. Stored in `localStorage` key `efl_club_planner_v1` ({active, picks:{Jack:{gw:[id,id]}, John:{...}}}).
  Export/Import (copy-paste JSON) and **Cloud backup** (section 5). Picks from storage/import/cloud are sanitised: unknown
  club ids are dropped (an earlier version crashed the page on them).
- **Optimiser**: choose 2 clubs per remaining gameweek, each club at most its picks left, maximising total xP. It is an
  assignment problem; solved **exactly with min-cost flow in the browser** (source -> club [capacity = picks left] ->
  gameweek [profit = xP] -> sink [open slots]). Verified against scipy `linprog` to the decimal: 601.20 (all picks
  left) and 575.81 (limited-picks scenario with history entered). It also reports the gain over greedy week-by-week
  choice (about +5.4 xP), the "other options this week" table (season-plan total if you take that club now), and a
  collapsed full-season plan labelled provisional.
- **Which gameweeks show**: every gameweek that still has a game to kick off (Fantasy EFL locks game by game, so a
  gameweek stays editable for clubs that have not played until its last kick-off). Fixtures carry kick-off times and the
  pages lock a club for that week once its game has kicked off (live clock, no refresh needed); locked clubs are tagged
  and never suggested. (Originally the plan dropped a gameweek at its first kick-off; Jack pointed out he can still edit
  it, so this was changed on 9 Oct 2026.)
- **Suggestions are for the current gameweek** (`plan.firstGw`); once its two picks are entered and its first game has
  kicked off (when Fantasy EFL opens the next gameweek) they move to the next gameweek. Previously:
  Each suggested club has **Add to GW n picks** (plus Add both); once both are in, the page says the picks are set and
  that the next gameweek's suggestions appear after the next refresh. Jack asked for this: he does not want GW n+1
  suggested before the refresh that follows the GW n deadline.
- **Caveat box** (keep it): fixtures can change; strength estimates change; variance (one club game has sd ~3.8 points,
  a double ~5.4); covariance (two picks that play each other cannot both win and are flagged; a double's two games share
  form/injury risk; strength errors persist across weeks). The plan maximises expected points and ignores risk.

## 4. Weekly workflow (Jack runs it)

1. After a pick deadline passes: `cd ml/team_strengths && ../../venv/Scripts/python.exe run_gameweek.py`, commit and
   push `public/data/club_plan.json`. The page moves to the next gameweek (`firstGw`).
2. In a double gameweek, re-run once the midweek games are priced in football-data (the run prints which dates are
   model-only). We do not yet know how many days before kick-off football-data lists midweek games.
3. On the page, click **Add to picks** for the two suggested clubs (or choose differently).
3a. **Betfair overrides on request**: when Jack pastes Betfair Exchange prices (first-round games), save the text under
   `ml/team_strengths/overrides/`, run `python overrides.py import <paste> <csv>`, then `run_gameweek.py <gw> --override <csv>`
   (mid-price = average of back and lay). Details in the README, "Odds overrides". Not built into the website by his choice.
4. Jack does **not** want the refresh automated for now. If he asks later, the plan is in the README ("Automating the
   refresh"): a GitHub Actions workflow, first manual-trigger, then scheduled; test that EFL/football-data answer from
   GitHub runners first.

## 5. Cloud backup of picks (`api/club-picks.js`)

Why: picks in `localStorage` vanish in a private window or on another device, and Jack did not want to re-enter
them. Design:
- `POST /api/club-picks` with `{action: 'save'|'load', name, passphrase, picks}`. Names allowed: `Jack`, `John`,
  `Test` (Test only exists for checking the live route; its record on the branch is harmless).
- Stored in `club-picks.json` on the **`picks-store` branch** via the GitHub contents API using the existing
  `GITHUB_TOKEN`. A separate branch so saving does not redeploy production; `vercel.json` also has
  `git.deploymentEnabled: {"picks-store": false}` (believed to stop preview builds; not independently verified).
- The first save for a name sets its passphrase (PBKDF2-SHA256, 100k iterations, random salt; only salt+hash stored,
  never the passphrase). Later saves/loads need it. Wrong passphrase gives 401. Picks are validated (gameweek 1-60, two
  club ids or null) so junk cannot be stored. A write conflict is retried once.
- **The repo is public, so stored picks and the passphrase hash are publicly readable**; the UI says to use a passphrase
  not used elsewhere. First-come claims a name's passphrase (fine for two friends; to reset, edit the file on the branch).
- The branch was created from `main` once with `git push origin HEAD:refs/heads/picks-store`; the API does not create it.
- Tested: logic against an in-memory GitHub fake (save/load, wrong passphrase, validation, conflict retry, plaintext
  never stored), then live (save, load, wrong passphrase, unknown profile, GET 405) and confirmed one commit on
  `picks-store` with `main` untouched.

## 6. Testing without Node

No Node on this machine, so API functions cannot be run locally. What worked: serve the repo with
`python -m http.server`, open the page in the browser pane, `fetch('/api/<file>.js')` to get the real source, run it
with `new Function('require','module','process','fetch', src)` using a shim (`require('crypto')` ->
`{webcrypto: window.crypto}`) and a fake GitHub `fetch`, and route the page's `/api/...` calls to the handler by
overriding `window.fetch`. Handlers therefore avoid Node-only APIs where cheap (WebCrypto, `Buffer` feature-detected).
Keep a temporary harness out of the repo.

## 7. Gotchas met

- The site's global table CSS (striping, hover) leaks into any new table, including tooltips: override with `.cp-tip tbody ...`.
- Tooltips are hidden on scroll; when testing hover in the browser pane, scroll first then dispatch `mouseover`.
- The page re-renders by replacing `innerHTML`; `render()` restores scroll position and the cloud passphrase is kept in
  state (not in storage).
- Browser pane screenshots sometimes time out; retry. Static server for local testing: `python -m http.server` (the
  preview tool wants a `.claude/launch.json` in the session's own folder, which was an unrelated folder).
- football-data club names: `QPR`, `Wolves`, `Peterboro`, `Bristol Rvs`, `Sheffield Weds`, `West Brom`, `West Ham`,
  `Milton Keynes Dons` need aliases. Promoted clubs (e.g. York City) are not in older squads files: use the live squads.
- JSON object keys are strings: gameweeks in picks are string keys, club ids numbers.

## 8. Not done / ideas

- Automated refresh (GitHub Actions) and a schedule (deadline-driven plus daily in double-gameweek weeks).
- Team news (lineups, injuries) as a model input; shots on target in the strength fit; "slow strength + fast form".
- Feeding clean-sheet/goal probabilities into the defender/goalkeeper/attacker player-points models in `ml/`.
- Optionally let the optimiser trade expected points against risk; it currently ignores variance and correlation.
- Player-level data for 2026/27 accumulates in `data/2026_27/` (Export Stats button) for future modelling.

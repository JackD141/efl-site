# Player models (v2, 10 Oct 2026): all positions

Code: `ml/team_strengths/player_model.py` (MID / FWD / DEF stats), `minutes_model.py` (expected minutes, all positions),
`ml/fotmob/` (FotMob pull, linking, history features, league steps). Artifacts in `ml/team_strengths/models/`
(`mid_model.json`, `fwd_model.json`, `def_model.json`, `minutes_model_<pos>.*`, `league_steps.json`). The site pages compute
xP in the browser from these (checked equal to Python for every player).

## Data
- EFL fantasy per-match stats (2025/26 GW1-35, 2026/27 to date): the targets (what scores).
- FotMob per-player match stats for the Championship, League One and League Two, 2024/25 onwards (`fetch_fotmob.py`;
  FotMob's terms and robots.txt disallow automated access: Jack chose to use it, 9 Oct 2026). Linked to EFL players by
  club, match and name (`link_fotmob.py`, 98% of appearances; minutes and goals agree exactly).
- football-data.co.uk closing odds -> expected goals per match (team strength).

## Stat models (per position, per scoring stat)
Poisson GLM, minutes as exposure, standardised logs of: role (his rate relative to same-position team-mates, last 20
appearances), team-mates' rate excluding him, opponent style, odds (own / opponent expected goals), home, and FotMob
history per 90 over his last 20 or 40 appearances. Negative-binomial spread for the points thresholds.
- Goals and assists are predicted through npxG and xA, then converted: goals = c x npxG + his penalty xG (c 1.00 MID,
  0.98 FWD, 0.85 DEF), assists = c x xA (1.13 MID, 1.55 FWD, 1.04 DEF; fantasy assists are more generous than Opta xA).
- League changes: numbers from another league count x the measured step-up / step-down multiplier (FotMob movers,
  shrunk towards 1 by sample size). Club style carries over from last season (measured year-to-year persistence, with
  promotion / relegation multipliers) instead of resetting to the league average.
- Clean sheets / goals conceded (DEF, GK) and saves (GK) come from the match odds as before.

## Minutes
Ridge regression on: recent minutes / starts / appearances at his club (several decays), FotMob started / subbed-off
history, run of 0-minute games, games at the club, rest days, second game of a double. Expected appearance points and
P(60+ minutes) are isotonic curves of expected minutes (fitted out of sample). Keepers are scaled so each club sums to 90.
Test RMSE (minutes) vs the old 90-or-0 rule: MID 25.6 vs 34.3, FWD 25.1 vs 35.9, DEF 29.2 vs 35.7, GK 22.3 vs 24.0.

## Protocol (do not deviate)
Choices (features, window, role shrinkage) are made on two rolling validation folds inside 2025/26 (train to GW17 ->
GW18-26, train to GW26 -> GW27-35), simplest candidate within 0.0005 log-likelihood of the best. 2026/27 is a one-off
test against the incumbent; then refit on everything. Things tried and rejected for lack of validation gain:
shots-on-target history (MID), heavier shrinkage of history, league-adjusting FotMob history, recency decay in role.

## Test results, 2026/27 (negative-binomial log-likelihood per appearance; closer to 0 is better)
| Pos | Stat | New | Previous |
|---|---|---|---|
| MID | goals / assists / SOT / key passes / interceptions | -0.290 / -0.305 / -0.635 / -1.194 / -0.999 | -0.296 / -0.312 / -0.645 / -1.198 / -1.008 |
| FWD | goals / assists / SOT / key passes | -0.485 / -0.353 / -0.924 / -1.070 | -0.490 / -0.355 / -0.931 / -1.077 |
| DEF | goals / assists / clearances / blocks / tackles | -0.170 / -0.210 / -2.220 / -0.916 / -1.477 | -0.171 / -0.212 / -2.230 / -0.928 / -1.482 |

## Weekly
`python ml/fotmob/fetch_fotmob.py 2026/2027` then `python ml/fotmob/link_fotmob.py`, then `run_gameweek.py`. Retrain
(`player_model.py`, `minutes_model.py <POS>`) every few weeks as data accumulates; keep the protocol above.

## Backtest (`ml/team_strengths/backtest.py` -> public/data/backtest.json -> backtest.html)
For each completed 2026/27 gameweek: models refitted on games before it (same feature choices), inputs built with the
live code cut to that point, closing odds for fixtures, exact team optimiser (scipy milp), two best clubs with picks left;
scored with real points (captain x2, vice if the captain did not play). GW1-8 (10 Oct 2026): model 594, form picker
(last-5 average points, same clubs) 559, hindsight best 1383; model expected 646 (8% optimistic: winner's curse).
Re-run after each gameweek: `python backtest.py` (about 4 minutes).

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
history per 90 over his last 20 or 40 appearances (for MID key passes and DEF clearances the 40 are recency-weighted,
half-life 20 appearances: chosen on validation, 15 Oct). Negative-binomial spread for the points thresholds.
- Goals and assists are predicted through npxG and xA, then converted: goals = c x npxG + his penalty xG (c 1.00 MID,
  0.98 FWD, 0.85 DEF), assists = c x xA (1.13 MID, 1.55 FWD, 1.04 DEF; fantasy assists are more generous than Opta xA).
- League changes: numbers from another league count x the measured step-up / step-down multiplier (FotMob movers,
  shrunk towards 1 by sample size). Club style carries over from last season (measured year-to-year persistence, with
  promotion / relegation multipliers) instead of resetting to the league average.
- Clean sheets / goals conceded (DEF, GK) and saves (GK) come from the match odds as before.

## Set pieces (10 Oct)
`ml/fotmob/set_pieces.py` reads penalties and direct free kicks from FotMob shot maps; corners come from the match stats.
As of each match (last 40 appearances): share of his team's penalties (shrunk with 2 pseudo-penalties at the typical
share), share of his team's corners, free-kick xG per 90. Goals add share x 0.094 penalties per team-match x 75%
conversion per 90; key passes use corner share. Tags (with a reason shown on hover): Pens (took his club's most recent penalty, or 2+ and half or more of his team's; our history is EFL-only, so a recent arrival from the Premier League needs the first), Corners (10+ and
a quarter or more), FKs (3+ free-kick shots).

## Minutes: the start model (since 7 Nov 2026)
What scoring depends on is not "expected minutes" but the chance of each minutes scenario, so three logistic models
(minutes_model.py) on the same features (recent minutes / starts / appearances at his club on several decays, FotMob
started / subbed-off history, run of 0-minute games, games at the club, rest days, second game of a double):
P(start), P(60+ | started) (his own: a 90-minute regular is not docked for team-mates' substitutions) and P(comes on |
not started). A game is then P(full game) = start x P(60+|start) (appearance 2, clean sheet, his usual full-game minutes
mFull) + P(part game) = start x (1 - P(60+|start)) + (1 - start) x P(comes on) (appearance 1, ~24 minutes).
- Suspensions are inferred from cards (0 minutes in the 1-3 games after a red, the game after a 5th / 10th / 15th
  league yellow) and left out of training and of his history: a ban is known at pick time (EFL marks him suspended)
  and is not being dropped.
- Doubles: the second game's start chance is predicted as if he started the first (pStart2c) and as if not (pStart2n).
- Keepers: a club's keepers' start chances are scaled to add up to 1 per game.
- Validation 2025/26 GW24-35, P(60+) log-loss, old ridge + isotonic curve vs start model: GK 0.1535 / 0.1489, DEF 0.3436
  / 0.3421, MID 0.3093 / 0.3064, FWD 0.3399 / 0.3353 (test 2026/27: MID 0.384 / 0.360, GK 0.211 / 0.194).
- Regulars (5 straight 90s): start ~0.91-0.93, P(60+|start) ~0.97, ~87 minutes if he starts. Real data: regulars start
  the next game 83-88% of the time (8% do not play at all: injury, ban, rotation, sale) - so ~80 expected minutes is
  honest BEFORE team news. Known bias: EFL injury flags are not in the history, so known injuries are counted twice for
  everyone else (the site also sets flagged players to 0); weekly snapshots of the flags start 7 Nov to measure it.
- Pages: "Start %" (editable after team news) replaces expected minutes; "xP if starts" (default sort) puts his club's
  first game of the gameweek at 100% and a double's second at pStart2c; the homepage has "Assume my picks start".
  Python reference in check_site_maths.py.

## Protocol (do not deviate)
Choices (features, window, role shrinkage) are made on two rolling validation folds inside 2025/26 (train to GW17 ->
GW18-26, train to GW26 -> GW27-35), simplest candidate within 0.0005 log-likelihood of the best. 2026/27 is a one-off
test against the incumbent; then refit on everything. Things tried and rejected for lack of validation gain:
shots-on-target history (MID), heavier shrinkage of history, league-adjusting FotMob history, recency decay in role, recency weights in the FotMob history for the other stats
(`exp_recency.py`).

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
scored with real points (captain x2, vice if the captain did not play). GW1-8 (10 Oct 2026, after the leakage fixes): model 587, form
picker (last-5 average points, same clubs) 559, hindsight best 1383; model expected 646 (~9% optimistic: winner's curse).
Re-run after each gameweek: `python backtest.py` (about 4 minutes).

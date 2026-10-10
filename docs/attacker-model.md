# Midfielder and forward expected-points model (v1, 9 Oct 2026)

Live at https://dexters-corner.vercel.app/midfielders.html and /forwards.html. Code: `ml/team_strengths/attack_model.py`
(training and checks, artifact `models/attack_model.json`), `attackers.py` (page data `public/data/mid_plan.json`,
`fwd_plan.json`), `public/attackers.js` (one script for both pages; `<body data-pos="MID|FWD">`). The page computes xP
from the exported model, so editing minutes recalculates instantly. It matches the Python calculation to 0.0000005 (MID)
and 0.00003 (FWD, where the goals spread is effectively Poisson) for every expected starter.

## Scoring (2026/27, verified)
- MID: appearance 1 (under 60 min) / 2 (60+), goal 6, assist 3, +1 per shot on target, +1 per 2 key passes,
  +2 per interception, hat-trick 5, yellow -1, red -3, missed pen -3, own goal -3.
- FWD: the same, with goal 5 and no interception points.

## Data
Only the EFL fantasy data (2025/26 GW1-35, 2026/27 GW1-8) has key passes and interceptions. Shots on target and goals per
player are also only there. Odds-implied expected goals per match come from football-data.co.uk closing odds.
**There is no expected-goals (xG) data per player yet. It is on the to-do list (`docs/TODO.md`).**

## Model (one per position and stat: goals, assists, shots on target, key passes, interceptions for MID)
Rate per 90 = exp(b0 + b1 log(role) + b2 log(team style) + b3 log(opponent style) + b4 log(own expected goals)
+ b5 log(opponent expected goals) + b6 home). The expected count for a game is rate x minutes / 90, with a
negative-binomial spread. Fitted as a Poisson GLM with exposure (minutes / 90 as weight) on standardised log features.
- **Role**: his count over his last 20 appearances relative to what his same-position team-mates did per 90 in those
  games, shrunk towards 1 (3, 10 or 30 pseudo-games, chosen on validation). It moves with him between clubs.
- **Team style**: his club's same-position players' count per 90 over the last N gameweeks.
- **Opponent style**: what same-position players facing this week's opponent did per 90 over the last N gameweeks.
- All rolling values use strictly earlier gameweeks and are shrunk to the league average (12 pseudo-90s).

Protocol (same as defenders): train 2025/26 GW1-23, choose features / window (4/8/12/season) / role shrinkage on GW24-35,
test once on 2026/27 GW1-8, refit on all. The simplest model within 0.0005 validation log-likelihood of the best is kept.

## Test results (2026/27, never seen in training; mean squared error, lower is better)
Baselines: "own rate" = his per-90 rate over recent games, shrunk to the league average; "last 10" = his plain average
count over his last 10 appearances.

| Pos | Stat | Chosen | Model | Own rate | Last 10 |
|---|---|---|---|---|---|
| MID | Goals | full, 12-GW window | 0.093 | 0.094 | 0.106 |
| MID | Assists | role only | 0.093 | 0.092 | 0.106 |
| MID | Shots on target | full, season | 0.304 | 0.311 | 0.348 |
| MID | Key passes | full, season | 1.046 | 1.065 | 1.267 |
| MID | Interceptions | role + team + opponent, 12 GW | 0.729 | 0.725 | 0.855 |
| FWD | Goals | full, 12 GW | 0.183 | 0.184 | 0.214 |
| FWD | Assists | full, 8 GW | 0.111 | 0.117 | 0.132 |
| FWD | Shots on target | full, season | 0.548 | 0.555 | 0.673 |
| FWD | Key passes | full, season | 0.818 | 0.829 | 0.972 |

Mostly level with the player's own rate, slightly better for key passes and FWD assists, and clearly better than a plain
recent average (which is worst on every stat). The odds (own expected goals) carry real weight for goals and shots on
target. Role is the biggest coefficient except for FWD assists, where the team's expected goals matter more.

## End-to-end check (2026/27 appearances of 60+ minutes, model given actual minutes; `python attack_model.py --total`)
| Pos | n | Model MAE / corr | His last-10 average points MAE / corr |
|---|---|---|---|
| MID | 1,825 | 2.53 / 0.19 | 2.64 / 0.07 |
| FWD | 1,233 | 2.65 / 0.24 | 2.72 / 0.13 |

Calibration by xP quintile: MID 4.16/4.66/5.04/5.42/6.13 predicted vs 3.93/4.79/5.24/5.71/5.70 actual (top fifth a
little high); FWD 3.63/4.10/4.44/4.82/5.62 vs 3.71/4.33/4.40/4.91/5.88.

## Expected minutes
90 if he played 60+ in his club's latest game (and is not injured or suspended), else 0. The box is editable. Unlike
the keeper and defender pages, minutes here are **the minutes he plays**. Counts scale with minutes, and the appearance is
2 points at 60+ and 1 below 60. So a 30-minute sub gets 1 + a third of the counts.

## Ideas
- Historical player xG / xA (see `docs/TODO.md`) should help goals and assists most.
- Penalty takers (a goals boost); a starter / minutes model from rotation history.

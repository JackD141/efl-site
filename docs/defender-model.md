# Defender expected-points model (v1, 9 Oct 2026)

Live at https://efl-site.vercel.app/defenders.html. Code: `ml/team_strengths/defence_model.py` (training + checks),
`defenders.py` (page data), `public/defenders.html` + `defenders.js` (the page computes xP from the exported model, so
editing expected minutes recalculates instantly; checked equal to the Python calculation to 0.00005 for all 276 starters).

## Scoring (2026/27, verified)
Defender points = appearance (1 up to 59 min / 2 for 60+) + clean sheet 5 (60+ min) - 1 per 2 goals conceded
+ 1 per 4 clearances + 1 per 2 blocks + 1 per 2 tackles + goal 7 + assist 3 - yellow 1 - red 3 (- own goal 3, - missed pen 3,
+ hat-trick 5). Reproduces 13,350 of 13,351 defender appearances in 2025/26 and all 2,910 in 2026/27.
Average per appearance 2025/26: appearance 1.78, clean sheet 1.11, clearances 0.81, tackles 0.44, goals 0.27,
assists 0.17, blocks 0.11, goals conceded -0.31, cards -0.17. Only 65% of defender appearances are full 90s.

## Data
Clearances, blocks and tackles exist only in the EFL fantasy data, and only for defenders (midfielders and forwards
show 0). So this model trains on fantasy data only: 2025/26 GW1-35 and 2026/27 GW1-8 (football-data.co.uk has no
clearances). Odds-implied expected goals per match come from football-data.co.uk closing odds. (The keeper saves model
can use 2023/24 onwards because saves = opponent shots on target - goals conceded, which football-data has.)

## Model for clearances / tackles / blocks (per stat)
For a defender playing the whole game: log mu = b0 + b1 log(role) + b2 log(team style) + b3 log(opponent style)
+ b4 log(opponent expected goals) + b5 log(own expected goals) + b6 home, with a negative-binomial spread (clearances
are very over-dispersed: variance about 2.7x the mean).
- **Role**: his count relative to his team-mates' per-90 in the same games, over his last 20 appearances at any club,
  shrunk towards 1. Being relative to team-mates, it carries over when he changes club.
- **Team style**: his current club's defenders' count per defender-90 over the last N gameweeks of this season.
- **Opponent style**: count per defender-90 made by defenders facing this week's opponent over the last N gameweeks
  (for example long-ball sides force more clearances).
- All rolling values use strictly earlier gameweeks and are shrunk towards the league average (12 pseudo defender-90s).

Protocol: train 2025/26 GW1-23, choose model and window (4/8/12 gameweeks or season) on GW24-35, test once on
2026/27 GW1-8, refit on all. Candidates: league average, the player's own recent per-90 rate (the usual approach),
nested GLMs (role; + team style; + opponent style; + odds and home), XGBoost. The simplest model within 0.0005
validation log-likelihood of the best is kept.

| Stat | Chosen | Validation log-lik, model vs own-rate baseline | Test 2026/27 points error, model vs baseline |
|---|---|---|---|
| Clearances | full model, 12-GW window | -2.491 vs -2.503 | 0.635 vs 0.646 (log-lik -2.415 vs -2.425) |
| Blocks | full model, season window | -1.010 vs -1.020 | 0.257 vs 0.251 (log-lik -1.061 vs -1.073) |
| Tackles | role + team + opponent style, season | -1.594 vs -1.590 | 0.579 vs 0.573 (log-lik -1.576 vs -1.577) |

Team and opponent style clearly help clearances (each step improves validation log-likelihood); for tackles and
blocks the model is about level with the player's own recent rate. XGBoost was worse than the GLMs throughout.
Clearance coefficients (standardised logs): role 0.34, team style 0.13, opponent style 0.09, odds small.

### Against simple baselines (2026/27 full games, n=1,909; `python compare_baselines.py`)
Baseline = the player's plain average count in his last X full games (any club). MAE/MSE are on the count; points MAE
on floor(count/unit) (baselines given the same negative-binomial spread).

| Stat | Predictor | MAE | MSE | Points MAE |
|---|---|---|---|---|
| Clearances | league average | 2.90 | 13.08 | 0.689 |
| | his last 5 games | 2.65 | 12.17 | 0.686 |
| | his last 10 games | 2.56 | 11.21 | 0.664 |
| | his last 20 games | 2.53 | 10.87 | 0.659 |
| | **model** | **2.38** | **9.62** | **0.635** |
| Tackles | his last 20 games | 1.10 | 2.05 | 0.573 |
| | model | 1.07 | 1.90 | 0.579 |
| Blocks | his last 20 games | 0.69 | 0.83 | 0.254 |
| | model | 0.69 | 0.76 | 0.257 |

Clearances: the model cuts MSE by 12% against the best simple baseline. Tackles and blocks: slightly better MAE/MSE on
counts, level on points (scored in steps of 2, which hides small gains). Short windows (3-5 games) are worse than
longer ones: single-game counts are noisy.

## Other parts
- Clean sheet and goals conceded: same per-fixture score matrix as the Club Planner and keepers, full distribution, so
  goals-conceded points = -[P(2+) + P(4+) + ...]; clearance points = P(4+) + P(8+) + ...; tackles/blocks = P(2+) + P(4+) + ...
- Goals and assists: the player's rate per 90 over his last 40 appearances (any club), shrunk with 40 pseudo-games of the
  defender average (chosen on validation from 5/10/20/40/80; 10 overrated prolific scorers), scaled by his team's
  expected goals relative to the league average. Cards: his rate, same shrinkage.
- Expected minutes: 90 if he played 60+ in his club's latest game (and is not injured/suspended), else 0; editable on
  the page (xP scales with minutes/90, so 45 = a 50% chance he plays).

## End-to-end check (2026/27 full games, component models fitted on 2025/26 only, 1,909 games)
| Predictor | Mean abs error | Correlation |
|---|---|---|
| Defender average | 2.81 | 0.00 |
| Player's recent average points | 2.81 | 0.10 |
| Component model | **2.69** | **0.17** |

Calibration by xP quintile: 3.88 / 4.46 / 4.84 / 5.19 / 5.80 predicted vs 3.88 / 4.43 / 5.04 / 5.13 / 5.23 actual. The top
fifth is too high, mostly from clean sheets (1.60 vs 1.32, likely small-sample luck: the odds are well calibrated over
three seasons) and clearances (1.40 vs 1.24). `python defence_model.py --total` reruns this check.

## Ideas
- Recalibrate the clearance model's top end (fit a power on mu using validation); starter probability from rotation history.
- Retrain monthly as 2026/27 data accumulates (`python defence_model.py`).

# ML methodology, leakage rules and experiment log (handover)

Read this before changing any model. Companion docs: `docs/player-models.md` (what the live models are),
`docs/club-planner.md` (team strengths / club points), `ml/team_strengths/README.md` (weekly run).

## 1. The prediction problem
Expected Fantasy EFL points per player per gameweek = sum over his fixtures of the expected points of each scoring
component. Components are modelled separately (never points directly):
- minutes: `minutes_model.py` (one model per position) -> expected minutes, expected appearance points and P(60+) curves,
  average minutes of a full (60+) and a part (1-59) game, and each player's own full-game minutes
- **minutes scenarios** (since 10 Oct): a fixture's xP = P(60+) x xP if he plays his usual full game (his own average
  minutes in his last 10 games of 60+, shrunk; appearance 2; clean sheet counts) + P(1-59) x xP of a typical part game
  (appearance 1). P(1-59) = expected appearance points - 2 P(60+). Thresholds (clearances per 4, key passes per 2,
  saves per 3, goals conceded per 2) are therefore evaluated on minutes he would actually play, not on the average
  (E[f(minutes)] instead of f(E[minutes])). Pages show "60+: x%" under the expected minutes.
- per-90 rates of each scoring stat: `player_model.py` (MID, FWD, DEF), Poisson GLM with minutes as exposure
- goals / assists through their expected versions (npxG / xA from FotMob), converted with a fitted factor
- clean sheets / goals conceded / keeper saves: from the match odds (`keepers.py`, `saves_model.py`)
- club points: `club_points.py` from the odds-implied score distribution
The browser pages recompute xP from exported coefficients (`attackers.js`, `defenders.js`, `keepers.js`, `xp-core.js`)
so minute edits are instant. Any model change must be mirrored there and checked (section 6).

## 2. Data and what is known when
| Data | Source / script | Known at prediction time? |
|---|---|---|
| Fantasy per-match stats (targets) 2025/26 GW1-35, 2026/27 to date | EFL API via the site's Export Stats button -> `data/<season>/` | only earlier gameweeks |
| FotMob per-player match stats 2024/25+ (xG, xA, started, ...) | `ml/fotmob/fetch_fotmob.py`, linked by `link_fotmob.py` | only earlier matches (as-of join on date, strictly before) |
| Closing odds -> expected goals | football-data.co.uk (`fetch_data.py`, `per_season.lam_for`) | yes (pre-match market view) |
| Team-strength fits for games without odds | `predict_gw.fit_current(as_of)` | yes (matches before as_of) |
| League of each club per season | EFL squads `competitionId` | yes (known before the season) |

FotMob data is NOT in the repo (`data/fotmob/` is gitignored: FotMob's terms disallow automated access and the files are
large). It lives in this machine's working copy. To rebuild from scratch (about an hour, polite rate):
`cd ml/fotmob && python fetch_fotmob.py 2024/2025 2025/2026 2026/2027 && python link_fotmob.py && python set_pieces.py`.
Weekly: `python fetch_fotmob.py 2026/2027 && python link_fotmob.py && python set_pieces.py`, then `run_gameweek.py`.

## 3. Protocol (do not deviate)
- **Selection** (features, windows, role shrinkage, model class): two rolling-origin validation folds inside 2025/26:
  train to GW17 -> validate GW18-26; train to GW26 -> validate GW27-35. Metric: negative-binomial log-likelihood of the
  ACTUAL count (also MSE). Rule: the simplest / most-preferred candidate within TOL = 0.0005 of the best.
- **Test**: 2026/27 to date, once, against the incumbent. NOTE: 2026/27 GW1-8 has now been looked at many times while
  developing (9-10 Oct 2026). It is no longer a pristine holdout: never select on it. The honest out-of-sample record
  from now on is the **backtest on gameweeks after GW8** (re-run `backtest.py` after each gameweek and compare).
- **Refit** on everything for production after the choice is made.
- Shrinkage everywhere small samples meet: role (pseudo-games), club styles (K_TEAM pseudo-90s), FotMob history
  (3 pseudo-90s plus the GLM coefficient), league multipliers (n / (n + n0)).
- Random seeds fixed (`random_state=0`) so reruns reproduce.

## 4. Leakage rules and audit (10 Oct 2026)
Every feature for a row must use only information from before that match. Checked:
- role / own rates: rolling windows over earlier appearances (`shift(1)`), ordered by **date** (fixed 10 Oct: was
  gameweek order, which could put the second game of a double first)
- club style, opponent style, team-mates rate: rolling over earlier gameweeks of the same season (`shift(1)` by gameweek,
  so the current gameweek is excluded entirely); carried-over priors use the full previous season only
- FotMob history: `merge_asof(..., allow_exact_matches=False)` on match date (strictly before)
- FotMob shrinkage targets (league averages): seasons before 2026/27 only (fixed 10 Oct: used all data)
- league step multipliers / persistence (`league_steps.py`): completed season pairs only (fixed 10 Oct: included
  2025/26 -> 2026/27). Add that pair once 2026/27 is complete (set `CURRENT_SEASON`).
- scalers, NB dispersion, conversion factors: fitted on the training rows of each fit
- minutes model: features from earlier games at the club; appearance / P(60+) curves fitted on out-of-sample validation
  predictions within 2025/26
- backtest (`backtest.py`): every model refitted on games before each gameweek, inputs built by the live code on data
  cut before the gameweek, FotMob as of the first kick-off, keeper saves model refitted before each gameweek (fixed
  10 Oct: the saved one includes 2026/27). Remaining known optimism: none in data; the feature CHOICES were partly
  informed by looking at 2026/27 results during development (see section 3).
Re-run this audit whenever a feature is added: write down, for the new feature, why it is strictly pre-match.

## 5. Experiment log (validation = 2025/26 folds unless stated; numbers are mean NB log-likelihood per appearance)
| Date | Change | Result | Decision |
|---|---|---|---|
| 9 Oct | Minutes model vs "90 if 60+ last game" | MID test RMSE 25.4 vs 34.3 | adopted |
| 9 Oct | Recency decay in role (half-life 5/10/20 apps) | gains <= 0.0004 | rejected |
| 9 Oct | Down-weight previous-club games x0.25 (kp, assists) | small gain on movers | replaced 10 Oct by measured league multipliers (ad hoc) |
| 9 Oct | FotMob history features (MID) | all attacking stats gain on val and test | adopted |
| 9 Oct | Goals / assists via npxG / xA targets | assists better, goals tied | adopted (principled: less noisy target) |
| 9 Oct | League step multipliers + carried club style | part of v2; small gains | adopted |
| 10 Oct | Team-mates rate excluding the player ("mates") | coefficient ~0, harmless | kept where chosen |
| 10 Oct | 40-game vs 20-game FotMob history | goals, kp, int gain (int -0.9854 vs -0.9911) | adopted where chosen |
| 10 Oct | FotMob SOT history for shots on target (MID) | worse (-0.6422 vs -0.6381) | rejected |
| 10 Oct | Heavier shrinkage of history (k 10 / 20) | no gain | rejected |
| 10 Oct | League-adjusting FotMob history | +-0.0001 | rejected |
| 10 Oct | FotMob started / subbed-off in minutes model | val RMSE 22.6 vs 22.9 (MID) | adopted |
| 10 Oct | Forwards / defenders on the same framework | every stat beats the old model on test | adopted |
| 10 Oct | GBM (HistGradientBoosting, Poisson, exposure) vs GLM | GBM worse on 10/13 stats (up to -0.006) | rejected |
| 10 Oct | 50/50 GLM + GBM blend | better on 9/13 but > TOL only for MID goals (+0.0006), MID SOT (+0.0007), DEF clearances (+0.0022) | not adopted: tiny gains, would need precomputed rates on the site; revisit with more data |
| 10 Oct | Leakage fixes (section 4) and retrain | 12/13 identical choices; MID SOT now "mates + shots/npxG hist40" (test -0.6320 vs -0.6351) | adopted |
| 10 Oct | Set pieces (`ml/fotmob/set_pieces.py`, `features.setpiece_features`): corner share -> key passes | val +0.0011 MID, +0.0019 FWD; test -1.1909 vs -1.1942 MID, -1.0649 vs -1.0705 FWD | adopted |
| 10 Oct | Penalty term = penalty SHARE x league penalties per team-match x conversion (instead of recent penalty xG) | val +0.0001 to +0.0005, never worse; same complexity | adopted |
| 10 Oct | Direct free-kick xG per 90 in goals; corners in assists | no gain / negative | rejected |
| 10 Oct | Minutes scenarios (E[f(minutes)]) vs straight scaling by expected minutes / 90 | backtest GW1-8, 13,529 player-gameweeks: MSE 8.543 vs 8.563 (better for every position), mean xP 2.404 vs 2.415 actual (2.389 before) | adopted (principled + evidence) |
Set-piece numbers: `ml/team_strengths/models/exp_setpieces_results.json` (`exp_setpieces.py`).
Full GBM numbers: `ml/team_strengths/models/exp_gbm_results.json`.

## 6. Checks before shipping
- JS equals Python: `python ml/team_strengths/check_site_maths.py` writes the reference (public/data/check_xp.json,
  gitignored); compare on each Player Picks tab and the homepage (`playerWeek`) as described in its docstring. 10 Oct:
  max difference 1e-9 (MID / DEF), 2e-5 (FWD, Poisson goals), keepers 5e-4 on the homepage (JSON rounding).
- Pages load with no error on all positions, the homepage optimiser runs, backtest page renders.
- Backtest (`python backtest.py`, ~7 min): GW1-8, 10 Oct after set pieces + minutes scenarios: model 611, form picker
  559, hindsight best 1383, model expected about 635. It also prints xP calibration over all player-gameweeks (MSE /
  MAE, mixture vs straight scaling). Single gameweeks are noise; judge totals and the calibration lines.

## 7. Known issues and next steps (ranked)
1. DONE 10 Oct: set pieces (penalty share, corner share, free kicks) + "Pens" / "Corners" / "FKs" tags on the pages and
   the homepage. Possible next: team penalty rate scaled by the team's expected goals; corner share for assists of
   defenders (val +0.0004, below TOL: retest with more data).
2. **Minutes-dependence of per-90 rates**: actual / predicted by REALISED minutes is ~1.2-1.4 for 10-30 minute cameos,
   ~0.7-0.9 for 30-75, ~1.0 for 75+ (partly selection: players are subbed when it is not going well). Test the
   decision-relevant version: calibration by EXPECTED minutes (minutes model out of sample) end-to-end; if rotation
   players are off, fit a correction curve on validation.
3. **End-to-end optimism** (~9% in the backtest): mostly selection ("winner's curse"); could shrink xP towards the
   positional mean before optimising (test on the backtest, not on 2026/27 GW1-8).
4. Over-performers: test on 2025/26 whether players beating the model over a run keep beating it (decides how much
   recent form should count beyond the current shrinkage).
5. Injury / team news in minutes (live only; the backtest cannot use it).
6. Retrain monthly; add the 2025/26 -> 2026/27 pair to `league_steps.py` at season end.

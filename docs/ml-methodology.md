# ML methodology, leakage rules and experiment log (handover)

Read this before changing any model. Companion docs: `docs/player-models.md` (what the live models are),
`docs/club-planner.md` (team strengths / club points), `ml/team_strengths/README.md` (weekly run).

## 1. The prediction problem
Expected Fantasy EFL points per player per gameweek = sum over his fixtures of the expected points of each scoring
component. Components are modelled separately (never points directly):
- start chances: `minutes_model.py` (one model per position): P(start), P(60+ | started), P(comes on | not started);
  each club's outfield start chances add up to ten starters (EFL flags: out -> 0, "questionable (50%)" -> halved)
- **minutes scenarios** (since 10 Oct): a fixture's xP = P(full game) x xP if he plays his usual full game (his own
  average minutes in his last 10 games of 60+, shrunk; appearance 2; clean sheet counts) + P(part game) x xP of a
  typical part game (appearance 1). Thresholds (clearances per 4, key passes per 2, saves per 3, goals conceded per 2)
  are therefore evaluated on minutes he would actually play, not on the average (E[f(minutes)], not f(E[minutes])).
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
| FotMob lineup slot of every starter (his job: CB, FB, DM, ... ST) | `ml/fotmob/positions.py` from the cached match files | only earlier starts (the coming game's line-up is not known) |
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
  ACTUAL count.
- **Rule: one standard error (since 9 Nov 2026; before that a fixed tolerance of 0.0005).** Candidates are listed
  simplest first, and for each candidate the plain setting (window 12, role shrinkage 10) comes before its best
  setting. The first configuration within max(0.0005, one standard error) of the best one wins. The standard error is
  that of the PAIRED difference to the best (same rows), with the 18 validation gameweeks as the independent units
  (`player_model.paired_se`). Why: 0.0005 turned out to be about half a standard error for most stats, so near-ties
  were decided by noise, and the nine window / shrinkage settings differ by less than a standard error for 10 of 13
  stats (`exp_selection.py`, section 5).
- **Choices are frozen** in `models/<pos>_model.json` (`selection`). The weekly run refits coefficients only.
  Re-selecting is a deliberate step, `python player_model.py --select`: it prints every change; log it in section 5.
- **Adopting a new idea** (feature, data source, model class) needs all of: a football reason written down before the
  numbers are looked at; a short list of candidates fixed beforehand; a validation gain of more than one standard error
  (paired, gameweek clusters), positive in both folds; a pass of the leakage test (section 4). Log every experiment in
  section 5, failures included: the number of things tried is part of the evidence, because the best of twenty noisy
  tries always looks good.
- **2026/27 is a monitor, not a test set.** It has been looked at many times (GW1-8 while developing, 9-10 Oct, and
  every week since). `player_model.py` prints, per stat, the log-likelihood and actual / predicted on 2026/27 for the
  chosen model fitted on 2025/26 only. Never choose on it. The honest out-of-sample records are (a) the backtest
  scorecard for gameweeks after GW8 and (b) the live record (section 6), which no later change can rewrite.
- **Refit** on everything for production after the choice is made.
- Shrinkage everywhere small samples meet: role (pseudo-games), club styles (K_TEAM pseudo-90s), FotMob history
  (3 pseudo-90s plus the GLM coefficient), league multipliers (n / (n + n0)).
- Random seeds fixed (`random_state=0`) so reruns reproduce.

## 4. Leakage rules and audit (10 Oct 2026)
Every feature for a row must use only information from before that match.
**Automatic test (9 Nov 2026): `leakage_check.py`**, run by `weekly_update.py` before every retrain (FAIL stops). It
alters every outcome on or after a cut-off date (counts, xG / xA, minutes, starts; fantasy data and FotMob history),
rebuilds the features with the live code and requires every row on or before the cut-off to be unchanged: 44 checks
over the stat features of all 14 models, the FotMob rates, the set-piece shares and the start-model features. On its
first run it found a real leak: team-mates' counts in the role feature were pooled per GAMEWEEK, so in a double (42% of
rows) the first game's "expected" count contained the second game's outcomes, including the target of the row being
predicted. Fixed (team-mates per match); `python leakage_check.py --old-role` shows the test failing on the old code.
Also 9 Nov: the league rate each stat is shrunk towards now comes from GW1-17 (the first fold's training games), not
GW1-23, which included validation games of the first fold.
Checked by hand (10 Oct):
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
| 15 Oct | Recency weights inside the history windows (`exp_recency.py`): a game a appearances back counts 0.5^(a/h), h = 40 / 20 / 10 / 5, separately for the role (last 20) and the FotMob rates (last 20 / 40), on each stat's live structure (25 combinations per stat) | > TOL with both folds positive only for MID key passes (+0.0009) and DEF clearances (+0.0008), both with FotMob h = 20 and a smooth surface around it; everything else within +-0.0004 (role decay again ~0, as on 9 Oct; MID interceptions +0.0002) | adopted as "fm40h20_*" candidates (each 40-game candidate gets a recency-weighted twin, least preferred) and re-selected by the normal protocol: chosen for MID kp (test -1.1873 vs -1.1875) and DEF clr (test -2.2170 vs -2.2197) only. Side effect: FWD goals moved from fm_npxg to fm40_npxg (it was 0.0005 = TOL behind; the new candidate nudged the best up). Backtest GW1-9, same code, old vs new models: player xP MSE 8.6702 -> 8.6687 (DEF 9.1132 -> 9.1089, MID 9.2169 -> 9.2165, FWD 9.2925 -> 9.2929); picked team 743 -> 729 (GW1 / 3 / 6 / 7 swaps between near-tied players: noise) |
| 7 Nov | Compared with 72 Projections (GW13, Championship, per minute for regular starters) | they give ~86-89 minutes to regulars vs our ~80; per minute our shots / key passes / interceptions match theirs, our defenders' clearances are 0.8x theirs, xG / xA 0.7-0.8x. Checked against OUR out-of-sample record instead of their numbers: 2026/27 actual / predicted per stat (model fitted on 2025/26) is 0.93-1.07 for every stat (clearances 0.97; top-quintile clearance predictions 0.90, i.e. already slightly high) | stat models unchanged; their higher numbers track season averages without regression. The real difference is minutes |
| 7 Nov | Start model (`exp_start.py`, then `minutes_model.py`): logistic P(start) + P(sub \| not started), suspension games inferred from cards and left out | P(60+) log-loss better than ridge + isotonic for every position on validation and test (table in docs/player-models.md); leaving bans out of the history helps every position | adopted |
| 7 Nov | + current run of consecutive starts | val +0.0005 DEF (= TOL), worse for MID / FWD / GK | rejected (the decayed start features already carry it) |
| 7 Nov | P(60+ \| started) per player instead of the position average | val P(60+) log-loss better for DEF / MID / FWD (e.g. FWD 0.3373 -> 0.3353), GK equal | adopted (a 90-minute regular is no longer docked for team-mates' substitutions) |
| 7 Nov | MID interceptions + the match odds (does a dominant side intercept less?) | val -0.0003 (both folds worse); odds coefficients ~0 | rejected |
| 7 Nov | MID interceptions + playing style from FotMob team passing (`exp_intstyle.py`): opponent's misplaced short passes per game, own team's pass share (previous 12 matches, shrunk) | val +0.00007 / +0.00009 / both +0.00017 (fold 1 ~0); test +0.0006 (info). Diagnostics: within a match the mechanism is real (team interceptions vs opponent misplaced short passes r 0.26, vs own pass share r -0.29) and team style is very persistent (odd vs even games r 0.84 / 0.89), but the outcome-based features already carry it (team-mates' interception rate, interceptions conceded by the opponent; a team's own interceptions persist r 0.63) | rejected (below TOL); interceptions are mostly the player (FotMob history is the largest coefficient) |
| 9 Nov | How noisy is the selection? (`exp_selection.py`: per-row validation log-likelihood of every configuration, gameweeks resampled 400 times, old rule re-applied) | the big structural choices are solid (FotMob history: gains 2-10 standard errors, chosen in 70-98% of resamples) but the near-ties are not: paired standard errors are 0.0005-0.0015, i.e. the 0.0005 tolerance is about half of one; MID key passes' full structure chosen in 59% of resamples, FWD goals 20- vs 40-game history 42 / 56%, MID shots recency twin 31%; the nine window / shrinkage settings span less than one standard error for 10 of 13 stats | one-standard-error rule, plain setting preferred, choices frozen (section 3) |
| 9 Nov | Re-selection under the new rule (with the role fix) | 9 of 14 stats change, 8 of them only to the plain setting (window 12, shrinkage 10); MID key passes drops corner share (0.0008 behind the best, standard error 0.0008) and keeps the recency-weighted history. Monitor 2026/27, old -> new choices: 12 stats within +-0.0005, MID key passes -1.1824 -> -1.1845, MID interceptions -1.0080 -> -1.0076. Backtest GW1-12 with the same code, old vs new choices: player xP MSE 8.9715 -> 8.9704 (gain +0.001, 90% interval over gameweeks -0.0004 to +0.0025) | adopted: accuracy unchanged within noise, fewer arbitrary settings, no weekly re-selection |
| 9 Nov | Club total (`exp_clubstart.py`): a club starts ten outfield players, so each club's start logits are shifted by one common amount until the chances add up to ten | raw sums: mean 10.30 (sd 0.44) on validation, 10.35 (sd 0.68) on 2026/27, and where the sum is above 10.5 players start 2.5-3.7 points less often than predicted. P(start) log-loss, validation 0.3160 -> 0.3143, 2026/27 0.3656 -> 0.3621 (every position better); calibration by band now within 1 point almost everywhere (e.g. 40-60% band: 46.7% -> 49.6% started vs 49.9% predicted). Live it also moves start chances up when team-mates are flagged out | adopted (`minutes_model.normalise_club`). Backtest GW1-12 (each gameweek refitted): player xP MSE 8.988 -> 8.970, gain +0.017 with a 90% interval over gameweeks of +0.006 to +0.030; start chances out of sample within 1 point of what happened in every band up to 92% (92%+: 94.3% predicted, 92.3% started; the backtest has no injury news) |
| 9 Nov | First read of the backtest scorecard (GW1-12, 22,631 player-gameweeks; also faster code: backtest 4 minutes) | xP MSE 8.97 vs form 11.78 (24% lower; 22% in GW9-12); mean xP 2.45 vs 2.50 scored; calibrated in every xP band (largest gap: 5-6 band, 5.42 vs 5.12); ten highest per position: DEF 8.00 expected / 7.96 scored, MID 9.14 / 8.77, FWD 7.74 / 7.67, GK 6.33 / 6.38, and the form picker's ten score less in every position (6.77 / 8.56 / 7.02 / 5.55); stats actual / expected 0.97-1.08, forwards' four stats all 1.06-1.08, which sits in rotation players (20-60% start chance: xP 2.74 vs 3.08 scored; 85%+: 5.51 vs 5.61); picked team 1089 vs expected 1107 vs form picker 1056: +33, 90% interval -60 to +119 | no change. The picked-team total cannot tell methods apart at this sample size; the scorecard can |
| 9 Nov | EFL flag "1 \| Questionable (50%)" (18 of 284 flagged players): was treated as out | no history of flags to test on (snapshots start 7 Nov) | start / bench chances halved (the EFL's own figure) instead of 0: an assumption, check against the snapshots when there are ~8 weeks |
| 11 Nov | Compared with 72 Projections again (GW14, League One / Two, their top 50 MID / 30 FWD / 30 DEF / 14 GK "as a starter" against our xP if he starts; nothing of theirs saved) | (1) **Postponed games**: 8 GW14 games were off and we projected them (their list had no Luton, Barnet, Huddersfield players). (2) Level: MID 5.72 theirs vs 5.48 ours, FWD 5.21 vs 5.05, GK 4.25 vs 4.23, DEF 6.01 vs 5.29. (3) Minutes if he starts: they give ~3 more (84.7 vs 81.5 MID); about half of that is our part-game simplification (known issue 2). Per minute MID / FWD / GK stats agree within ~5% (FWD key passes 10%). (4) Defenders: their clearances are 22% above what the same players have done this season (7.67 vs 6.50 per 90; ours 6.39). Clearances fell 11% league-wide this season (fantasy 5.61 -> 5.00 per 90, FotMob 5.77 -> 5.21 for the same players), and their numbers sit near last season's level. (5) In-form players (Keillor-Dunn 7.1 vs 6.2): they follow the recent run; our shots / interceptions lean on 40 games, and his fantasy position changed (FWD -> MID), so his role history restarted. Backtest check: likely starters who beat our xP by 1 point a game over their previous 3-5 gameweeks beat it by ~0.1 the next (correlation 0.05; forwards 0.17-0.20) | postponed games now left out (`fetch_data.drop_postponed`, data check added). No model change from the rest: three ideas logged in section 7 (items 7-9) to test by the protocol |
| 11 Nov | **The job a player is given** (`exp_position.py`; FotMob lineup slot of every starter, `ml/fotmob/positions.py`: 84,524 starts, all three seasons; jobs = line x central / wide: CB, FB, DM, WB, CM, WM, AM, W, ST). Reason: the fantasy position is fixed for a season, the job is not. Candidates fixed beforehand on each stat's live structure: + job multiplier (what his recent jobs produce / what the jobs over his history window produce), + job (league rate of the jobs of his last 5 starts), + both | job passes (gain above one paired standard error, both folds positive) for 9 of 14 stats: DEF clearances +0.0062 (se 0.0013), blocks +0.0050 (0.0011), tackles +0.0017; MID assists +0.0008, shots on target +0.0012, key passes +0.0010, interceptions +0.0012 (se 0.0012); FWD goals +0.0009, shots on target +0.0016. The multiplier alone: nothing (only DEF clearances, +0.0006); both: no better than job. 2026/27 agrees in every passing case (+0.001 to +0.009) | job added as a candidate twin and re-selected by the one-standard-error rule: chosen for MID assists / shots on target / key passes (which takes corner share back), FWD shots on target, DEF clearances / blocks / tackles; FWD goals moves to the 40-game history. Monitor old -> new: DEF clr -2.2221 -> -2.2127, blk -0.9133 -> -0.9061, MID kp -1.1840 -> -1.1805, MID sot -0.6438 -> -0.6423. By job on 2026/27 (fitted on 2025/26): blocks actual / predicted CB 1.09 -> 1.03, FB 0.90 -> 1.01, WB 0.54 -> 0.69; tackles CB 0.96 -> 1.00, FB 1.09 -> 1.03. Backtest GW1-13, old vs new: player xP MSE 8.8830 -> 8.8801 (gain +0.003, 90% interval -0.003 to +0.009; MID +0.011, FWD +0.006, DEF -0.006, all inside their intervals): adopted on the stat-level evidence, neutral on points so far |
| 11 Nov | **Form** (`exp_form.py`): his actual count over his last 5 / 10 appearances divided by what the live model expected (fitted on the fold's training games), shrunk; added to each stat's structure | no stat passes (gains <= 0.0003) except DEF clearances: +0.0028 (se 0.0005), still there with the job feature in. So the 0.1 carry-over of points residuals seen in the backtest is not stat rates (probably start chances and jobs) | rejected for 13 stats. DEF clearances passes but is not implemented: the effect is ~3% of a defender's clearances (~0.04 points) and needs a two-stage model in the weekly run and the backtest; revisit if clearances become the weak spot |
| 11 Nov | **P(60+ given he starts)** (`exp_p60.py`), candidates fixed beforehand for that model only: A flags (came off before 60 last game; missed the previous game), B his own record (share of his last 10 starts at the club that reached 60, shrunk), C both | A: nothing (the 2-6 point over-prediction for those groups is not there in the training period, so a flag cannot learn it). B: MID 0.2972 -> 0.2966 (se 0.0003), FWD 0.3645 -> 0.3634 (se 0.0005) on validation, FWD also +0.0011 on 2026/27; DEF / GK +0.00003 | B adopted for MID and FWD (`minutes_model.FULL_EXTRA`); small. The leakage test caught the first version (defined differently for games he started) before it shipped |
Set-piece numbers: `ml/team_strengths/models/exp_setpieces_results.json` (`exp_setpieces.py`).
Recency numbers: `ml/team_strengths/models/exp_recency_results.json` (`exp_recency.py`).
Full GBM numbers: `ml/team_strengths/models/exp_gbm_results.json`.

## 6. Checks before shipping
- **Data integrity** (`ml/team_strengths/data_checks.py`, run by `weekly_update.py`; FAIL stops the run): we join EFL
  fantasy stats, EFL fixtures, football-data.co.uk and FotMob, so every join is cross-checked against another source
  (stats cover exactly the finished games and their points recompute; one club per team name, right league; dates and
  scores agree across sources for every finished game; priced fixtures are real EFL fixtures, right way round, right
  date; sane odds-implied expected goals; FotMob timestamps parse; FotMob links 95%+ with agreeing minutes / goals;
  site fixtures and kick-offs equal the EFL's). Each check was tested by corrupting a copy of the data. Add a check
  whenever a new source or join is added. Known past slips this would have caught: EFL round left "playing" (empty
  export), stale football-data fixture list, FotMob timestamp format change (15 Oct 2026).
- **Leakage test** (`leakage_check.py`, ~2 minutes, in `weekly_update.py` before retraining): section 4.
- **Backtest scorecard** (`backtest.py`, printed and on the Backtest page): over every registered player and gameweek,
  each refitted on games before it: xP error against the form baseline and against the previous method (`ALT`), with a
  gameweek-resampled interval; calibration by xP band; the ten highest xP per position (how much the top picks
  disappoint); start chances against what happened; actual / expected count of every stat; and an interval for "model
  minus form picker" points. Totals of the picked team alone are too noisy to judge a change (the interval is wide).
- **Live record** (`live_record.py`): every weekly run logs what is being published for the gameweek
  (data/2026_27/predictions/gw<N>.json: xP, xP if he starts, start chance, club xP), only before its first kick-off, and
  scores the logged gameweeks once their stats are in. Unlike the backtest it uses the odds and injury flags we really
  had, and it cannot be rewritten by a later model change. Starts with GW14 (9 Nov 2026).
- JS equals Python: `python ml/team_strengths/check_site_maths.py` writes the reference (public/data/check_xp.json,
  gitignored); compare on each Player Picks tab and the homepage (`playerWeek`) as described in its docstring. 10 Oct:
  max difference 1e-9 (MID / DEF), 2e-5 (FWD, Poisson goals), keepers 5e-4 on the homepage (JSON rounding).
- Pages load with no error on all positions, the homepage optimiser runs, backtest page renders.
- Backtest (`python backtest.py`, ~7 min): GW1-8, 10 Oct after set pieces + minutes scenarios: model 611, form picker
  559, hindsight best 1383, model expected about 635. It also prints xP calibration over all player-gameweeks (MSE /
  MAE, mixture vs straight scaling). Single gameweeks are noise; judge totals and the calibration lines.

## 7. Known issues and next steps (ranked)
0. **Injury flags counted twice** (7 Nov): the start model learns from all absences, including injuries EFL had flagged
   at pick time, and the site also sets flagged players to 0. Partly dealt with on 9 Nov: a club's chances now add up
   to ten, so the others rise when team-mates are flagged. Still to do when ~8 weeks of flag snapshots exist
   (data/2026_27/status/, saved every run): refit with flagged games left out (as suspensions are), check the
   "questionable (50%)" assumption, and compare on the backtest.
1. DONE 10 Oct: set pieces (penalty share, corner share, free kicks) + "Pens" / "Corners" / "FKs" tags on the pages and
   the homepage. Possible next: team penalty rate scaled by the team's expected goals; corner share for assists of
   defenders (val +0.0004, below TOL: retest with more data).
2. **Part games are one scenario** (measured 9 Nov): starters who come off before 60 play 44-48 minutes (5.5% of
   defenders' starts, 8% of midfielders', 11% of forwards') and substitutes about 20, but both are valued as one
   "part game" of ~24 minutes at the full-game rate; substitutes' per-90 rates are also higher than starters' (actual /
   predicted ~1.2-1.4 for 10-30 minute cameos). Effect: xP if he starts is under by less than 0.1 points; rotation
   players (20-60% start chance) are under-predicted by 5-10% (forwards 11%). Fix: three scenarios (started and off
   early / came on) with their own minutes and a rate factor fitted on validation. It changes the page maths in four
   files, and picks assume a start, so the value is low: do it only with an automatic page-maths check in place.
3. **End-to-end optimism**: 9% on 10 Oct, 1.7% now (picked team 1089 vs 1107 expected, GW1-12), and the ten highest
   xP per position score within 1% of expected (midfielders 4% under). Watch it on the scorecard; no shrinkage needed.
4. Over-performers: test on 2025/26 whether players beating the model over a run keep beating it (decides how much
   recent form should count beyond the current shrinkage).
5. Injury / team news in minutes (live only; the backtest cannot use it).
7. DONE 11 Nov (tested): form features do not help the stat models (see section 5); the small carry-over of points
   residuals is not in the rates. Open: a two-stage form term for defenders' clearances (passes, ~0.04 points).
8. DONE 11 Nov: the job a player is given (FotMob lineup slots) is a model input and a tag on the pages. Possible next:
   the job feature for MID interceptions (+0.0012, exactly one standard error: not chosen), and using the opponent's
   formation.
9. P(60+ | started): each player's own record added for MID / FWD (11 Nov). The remaining 2-6 point over-prediction
   for players just back or taken off early is drift between periods, not something a flag fixes.
6. Retrain monthly; add the 2025/26 -> 2026/27 pair to `league_steps.py` at season end.

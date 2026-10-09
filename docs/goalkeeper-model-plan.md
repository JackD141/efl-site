# Goalkeeper expected-points model

## Status: v1 built (9 Oct 2026), live at https://efl-site.vercel.app/keepers.html

What was built, versus the plan below:
- `ml/team_strengths/saves_model.py` trains the saves model; `keepers.py` turns it into keeper xP and expected minutes;
  `run_gameweek.py` writes `public/data/keeper_plan.json`; `public/keepers.html` + `keepers.js` show it.
- **Saves model** (time-based protocol: train 2023/24 + 2024/25, choose on 2025/26, test once on 2026/27 to date, then
  refit on all 10,510 team-games). Candidates: league average; odds only (opponent expected goals); odds + own expected
  goals + home; plus rolling opponent shots-on-target-for and team shots-on-target-against (windows 3/6/10/season, shrunk
  to the league average with 3 pseudo-games, strictly earlier matches only); plus rolling save rate; XGBoost on the
  richest set. **The odds carry almost all the signal; rolling shots/save-rate features add nothing at any window, and
  XGBoost is worse than the GLM.** Rule: simplest model within 0.0005 validation log-likelihood of the best, which
  chose "odds + home" (Poisson GLM on standardised logs, negative-binomial spread r ~ 14.6).
  Test (574 team-games, unseen): NB log-likelihood -1.942 vs -1.976 baseline, saves-points MAE 1.153 vs 1.207,
  calibration by quintile 2.27/2.56/2.81/3.06/3.54 predicted vs 2.32/2.35/2.68/2.98/3.63 actual.
  The rolling-feature code stays in place (and `keepers.py` supports it) in case a retrain picks it.
- **Keeper xP if he starts** = 2 + 5 P(CS) - E[floor(GC/2)] + 2 E[floor(S/3)] + 0.097 (penalty saves) - 0.056 (cards),
  using the same per-fixture expected goals as the Club Planner (market odds where they exist, else the strengths model).
- **Expected minutes** = 90 x P(start). The keeper who started his club's latest game keeps it with the measured rate
  (93.7% over 3,236 starts); available backups share the rest; keepers with an injury/suspension flag or status
  "injured" in the EFL players feed get 0. Starters come from local `data/2026_27` stats, so press Export Stats and
  `git pull` before refreshing; the run and the page warn if the stats lag the latest completed gameweek.
- Page: expected starters by default (P(start) >= 0.5), "Show backups and injured" toggle, league filter, search,
  hover breakdowns, next-5-gameweeks total. Defenders deliberately not started.

---

# Original plan (written before building)

Written 9 Oct 2026 after exploring the data. Nothing here is implemented; the numbers below are from quick checks
(scripts were throwaway). Builds on `docs/club-planner.md` and `ml/team_strengths/`.

## 1. Approach: model the components, not the points

Keeper points are a sum of a few countable things, so model each and add the expectations (expectation is linear, so no
covariance is needed for xP; covariance only matters for variance). The 2026/27 goalkeeper scoring (fantasy.efl.com
help centre) is: appearance +1 (up to 59 min) / +2 (60+), clean sheet +5 (needs 60+ min), every 2 goals conceded -1,
every 3 saves +2, penalty saved +5, goal +10, assist +3, own goal -3, missed penalty -3, yellow -1, red -3.
**Verified:** this reproduces every goalkeeper appearance exactly, 2,694 in 2025/26 and 573 so far in 2026/27, zero
mismatches, so scoring is unchanged between seasons and the components fully determine the points.

## 2. What the data says (2025/26, 2,694 keeper appearances, mean 4.08 points, sd 2.84)

| Component | Mean pts | Share of variance (R2 with total) | Where the expectation comes from |
|---|---|---|---|
| Appearance | 1.99 | 1% | 2 if he starts (99.6% of starting keepers play 60+) |
| Clean sheet (+5) | 1.34 | **67%** | P(opponent scores 0) from the match score matrix: **already built** (club planner) |
| Goals conceded (-1 per 2) | -0.39 | 23% | E[floor(goals/2)] from the same score matrix: **already built** |
| Saves (+2 per 3) | 1.09 | 14% | **new**: saves model (below) |
| Penalty saved (+5) | 0.10 | 8% | 2.1% of games have one; constant, refine later |
| Cards | -0.06 | 0% | keeper rate, small |
| Goals / assists / own goals | 0.02 | 2% | ignore |

Consequences: most of the keeper model already exists (clean sheet + goals conceded from market odds or the strengths
model). The one genuinely new piece is **saves**.

### Saves can be modelled from shots on target

football-data's shots on target against (`HST`/`AST`) tie to the fantasy saves: `saves = SOT against - goals conceded`
exactly in 82.5% of team-games, within 1 in 99.1%, correlation 0.97 (2,667 of 2,669 team-games matched, goals agree
in 99.8%). Mean 3.9 SOT against, 1.27 goals conceded, 2.6 saves, save rate 66.6%.
Consequently saves ground truth exists for **every match since 2023/24 (5,255 matches = 10,510 team-games)** without
needing keeper-level fantasy data, which only exists for 2025/26 onward.
Saves are mildly over-dispersed (variance/mean 1.27; negative-binomial r about 15 after conditioning on the mean) and
points are non-linear (`floor(S/3)`), so model the **distribution**, not just the mean.
First look: saves ~ 0.54 + 1.61 x opponent expected goals (R2 only 0.07, but it separates games: predicted 2.0 / 2.3 /
2.6 / 2.8 / 3.3 saves vs actual 2.0 / 2.3 / 2.6 / 2.8 / 3.4; saves points 0.74 to 1.58 a game across quintiles).

## 3. Minutes: model "if he starts", then multiply by P(start)

- When a keeper appears he plays 90 in 98.2% of appearances; the starter plays the full 90 in 99.1% of team-games
  (starter under 60 min in 0.4%). Two keepers appear in only 0.9% of team-games. So **assume 90 given a start**; no
  minutes model needed.
- The real uncertainty is **who starts**: the starting keeper differs from the previous game's in 5.9% of team-games;
  regular keepers (1,500+ min, n=70) played 60+ in a median 88% of their team's games, 10th percentile 54% (injuries,
  loans, mid-season changes).
- Plan: xP = P(start) x xP(if he starts). P(start) from recent starts (last ~5 team games) plus the EFL players feed's
  `injuryDetails` / `suspensionDetails` / `status`, with a manual override for team news. Non-starters score about 0.

## 4. Model design

1. **Inputs per fixture** (already produced by the planner): market-implied or strengths-model expected goals for both
   sides, and the full score matrix -> goals-conceded distribution -> P(clean sheet), E[floor(GC/2)].
2. **Shots-on-target strengths (new)**: the same per-league, per-season ridge/Poisson machinery as goals, fitted to
   SOT (about 3x the events of goals, so less noisy): each team gets "SOT for" and "SOT against" strengths plus home
   effect. Gives expected SOT against for any fixture, including ones with no odds.
3. **Saves distribution (new)**: saves = SOT - goals; model the mean from expected SOT against (and opponent expected
   goals) with a negative-binomial spread; E[2 x floor(S/3)] by summing the pmf. Compare candidates by out-of-sample
   negative-binomial log-likelihood: constant, opponent-goals only (above), SOT-strength based, plus keeper save-rate
   skill (expected tiny: about 160 shots per keeper-season gives a standard error of about 3.7pp on save rate).
4. **Penalty saves, cards**: constants first (+0.10, -0.06); refine if worth it.
5. **Combine**: xP = 2 + 5 P(CS) - E[floor(GC/2)] + 2 E[floor(S/3)] + 0.10 - 0.06, per fixture, summed over doubles;
   then x P(start) per keeper. Output next to the club xP in `club_plan.json` (a per-club keeper xP by gameweek), so
   the planner's rank/top-weeks machinery can be reused for the keeper slot.

## 5. How to judge it, and honest expectations

First end-to-end check (using closing-odds expected goals as inputs; saves link fitted on the same data, so slightly
optimistic): xP calibrates well by bucket, **but per-game R2 is about 0.006** (league-average baseline 0.000).

| xP bucket (6 equal groups) | predicted | actual |
|---|---|---|
| lowest | 3.85 | 3.83 |
| ... | 3.96 / 4.06 / 4.16 / 4.28 | 4.06 / 3.74 / 4.26 / 4.31 |
| highest | 4.51 | 4.53 |

Component calibration: clean sheet 1.43 vs 1.36 actual, goals conceded 0.42 vs 0.39, saves 1.08 vs 1.10.
So: **expect good calibration and a modest spread (about +/-0.3 typical, 3 to 6 at the extremes), not predictive power
for single games.** Single-game keeper points are mostly noise (a clean sheet is a coin flip weighted 15-60%). Do not
judge by per-game R2 or MAE; judge by calibration by bucket, by ranking (do top-xP fixtures beat bottom-xP ones by the
predicted margin), and by the saves log-likelihood.
Also learned: **a keeper's trailing average points is worse than the league average** as a predictor (R2 -0.10), which
is why the old direct points model (`ml/predict_points.ipynb`, GK R2 about 0) found nothing: past points mostly reflect
luck and team, and it never used odds-based goal expectations. Fixture information is the signal.
Test rolling-origin by time, and measure the degradation when inputs are *forecast* expected goals (strengths model)
rather than closing odds, which is what we will actually have for later weeks.

## 6. Build order

1. `keepers.py`: keeper xP from a score matrix + expected saves (constants for saves at first) and wire it into
   `predict_gameweek` / `season_plan` output. Quick, immediately useful (clean sheet + goals conceded + appearance).
2. SOT strengths + saves distribution, validated on 2023/24-2025/26 team-games (10k+).
3. P(start) from recent starts and the players feed; manual override file like the odds overrides.
4. Site: a keeper view (or add to the picks page), with the same hover breakdown by component.
5. Defenders next: they share clean sheet (+5) and goals conceded (-1 per 2) exactly, so only the bonus actions
   (every 4 clearances +1, 2 blocks +1, 2 tackles +1, goals +7) need new rate models.

## 7. Open questions for Jack

- Where should this surface: a keeper section on the Club Planner, or on the Optimal Picks (player) page?
- P(start): are you happy for the model to estimate it from recent starts plus the injury flags, with you overriding
  when you know the lineup?
- Do keeper picks need the same "picks left / plan" treatment? (Players have no per-player cap, only 2 per club.)
- Build defenders together, since they share two of the three big components?

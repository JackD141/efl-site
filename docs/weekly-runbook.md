# Weekly runbook: refresh predictions after a round of games

Run this after each batch of games (weekend, or midweek before the next weekend) and before the next pick deadline.
Total about 25 minutes, mostly unattended. Jack initiates it (or asks Claude to); it is not automated.

## 1. Fantasy stats for the games just played (EFL site -> repo)
Either press **Export Stats** on the site's Player Stats page, or call the same function:
```bash
curl -s -N -X POST https://dexters-corner.vercel.app/api/export-player-stats --max-time 300 | grep -A1 "event: complete"
```
It reads every player's game log from the EFL fantasy API (logged in with the credentials stored on Vercel) and commits
`data/2026_27/player_stats_gw<N>.csv` for every gameweek with finished games to `main`. Then pull:
```bash
git pull origin main
```
Check the newest file is not empty (`wc -l data/2026_27/player_stats_gw*.csv`). Since 15 Oct the export includes finished
games even when the EFL still marks the round "playing" (it did for GW9 for days), and never writes an empty file.

## 2. Everything else, in order: one command
```bash
cd ml/team_strengths
../../venv/Scripts/python.exe weekly_update.py                     # next gameweek with a game still to kick off
../../venv/Scripts/python.exe weekly_update.py --override overrides/gw<N>_betfair.csv   # if Jack pasted Betfair prices
```
`weekly_update.py` does, in order:
1. downloads odds and results (football-data.co.uk) and EFL rounds / squads / players; **stops** if a finished gameweek's
   stats file is missing or empty (go back to step 1)
   - **odds check: stop and ask Jack.** It prints how many games have market odds, per set of fixtures (a double gameweek
     has two sets, e.g. Saturday and Tuesday). If fewer than 80% of the FIRST set are priced it stops. Tell Jack and ask:
     wait for football-data.co.uk (fixture odds usually appear by Friday for weekend games), paste Betfair prices
     (`--override`, see "Odds overrides" in ml/team_strengths/README.md), or go ahead on the team-strength model for the
     unpriced games (`--allow-missing-odds`). The second set is normally priced after the weekend: re-run then
     (`--no-retrain` is enough) so the midweek games use market odds before their deadline.
2. FotMob: fetches new matches (`ml/fotmob/fetch_fotmob.py 2026/2027`), re-links players (`link_fotmob.py`), rebuilds
   set pieces (`set_pieces.py`). FotMob data lives only on this machine (`data/fotmob/`, gitignored)
3. retrains on everything to date: `minutes_model.py` (MID, FWD, DEF, GK) and `player_model.py` (MID, FWD, DEF). The
   feature choices are re-made on the 2025/26 validation folds exactly as before (same result); only the coefficients
   learn from the new games. Never change a choice because of this season's numbers (see docs/ml-methodology.md)
4. builds the gameweek's predictions for the site (`run_gameweek.py`): club_plan / keeper / defender / mid / fwd JSON
5. re-runs the backtest over every finished gameweek (`backtest.py`, each gameweek refitted on games before it)
6. writes the reference xP for checking the pages (`check_site_maths.py` -> `public/data/check_xp.json`)
Flags: `--no-retrain` and `--no-backtest` for a quick re-run (e.g. after a team-news minutes edit).

## 3. Check
- Serve locally: `venv/Scripts/python.exe -m http.server 8765 --directory public`, open the homepage and each Player Picks tab:
  no error message, sensible top players, gameweek shown is the one being picked.
- Page maths equal Python: see the docstring of `check_site_maths.py` (max difference about 1e-6).
- Read the backtest line for the newest gameweek (printed, and on the Backtest page). That gameweek is genuinely
  out of sample: it is the honest record of how the model is doing.

## 4. Commit and push (deploys)
```bash
rm public/data/check_xp.json
git add public/data/*.json ml/team_strengths/models/*.json ml/team_strengths/models/*.joblib
git commit -m "GW<N> predictions: retrained to GW<N-1>"
git push origin main
```
Vercel deploys `main` in about a minute: https://dexters-corner.vercel.app

## Notes
- Gameweeks open for picking as soon as the previous one kicks off, and lock game by game, so the "next gameweek" is
  the first one with a game still to kick off.
- If Jack pastes Betfair prices: `overrides.py` (see ml/team_strengths/README.md "Odds overrides").
- Football-data.co.uk updates results and odds about twice a week; games without odds use the team-strength model.

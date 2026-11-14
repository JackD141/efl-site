# To-do

## DONE (9-10 Oct 2026): historical player xG / xA data -> FotMob, see docs/player-models.md

## (old) Get historical player xG / xA data (midfielders and forwards)
The attacker model (`docs/attacker-model.md`) has no expected-goals data per player. Goals and assists are predicted from
past goals, shots on target and the match odds only. Historical xG and xA (expected assists) per player per match for the
Championship, League One and League Two would help model goals and assists.
- Find a source covering all three EFL leagues for 2025/26 and 2026/27, ideally per match. Possible places to check:
  FBref / Opta, Understat (top leagues only?), FotMob, Sofascore, WhoScored. Check coverage and terms of use before
  relying on any of them.
- Map their player names / IDs to EFL fantasy player IDs (same approach as `ALIASES` in `predict_gw.py` for clubs).
- Add rolling xG / xA per 90 (and team xG) as features in `attack_model.py`, then re-run the same train / validate / test.

## Other open items
- DONE: minutes model (minutes_model.py). Next: use injury news / predicted line-ups.
- Defenders: recalibrate the clearance model's top end (top xP fifth is over-predicted).
- Penalty takers: a goals boost for designated takers.
- Optional: automate the weekly refresh (currently Jack runs `run_gameweek.py` by hand).

## Stats export loses games played for a previous club (found and fixed in code 14 Nov 2026)
`api/export-player-stats.js` looked up each result's game among the fixtures of the player's CURRENT club and skipped it
when it was not there, so a player who moved between EFL clubs during the season had no rows for his old club (C. O'Hare:
nothing before GW3 2026/27). Fixed: the game is found by `gameId` among all the round's games and the club is the
result's own `squadId`. Second bug fixed with it: games were keyed by `roundNumber`, and the playoff rounds (ids 40-42,
numbered 1-3) emptied gameweeks 1-3, so those files had not been rewritten since 15 Oct. Tested with the real handler
on a download of every public profile (14 Nov): all 30,850 results exported, 243 rows added for 83 players in GW1-13
(238 of them in GW1-3), no existing row lost or changed apart from `fixture_difficulty`.
- STILL TO DO: push (deploys the function), run the export, `git pull`, then the weekly update (docs/weekly-runbook.md).
- Playoffs (May): their results now go to `player_stats_gw40..42.csv` (the round ids) instead of being dropped. The
  loaders glob `player_stats_gw*.csv`, so decide then whether the models should read them.
- Transfer gameweeks: a player can have rows for two clubs in one gameweek (25 cases in GW1-13). `backtest.py` takes
  the club he played most for; his start chance still comes from his first game by date, whichever club that was.

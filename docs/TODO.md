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

## Stats export loses games played for a previous club (found 14 Nov 2026)
`api/export-player-stats.js` looks up each result's game among the fixtures of the player's CURRENT club and skips it
when it is not there, so a player who moved between EFL clubs during the season has no rows for his old club (C. O'Hare:
nothing before GW3 2026/27). Effects: a few missing training rows, and the League Rivals page cannot score a pick of such
a player (its totals check shows it). Fix: find the game by `gameId` among all the round's games and take the club from
the side he played for (check what the profile's result row offers), then re-export.

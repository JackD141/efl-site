# To-do

## Get historical player xG / xA data (midfielders and forwards)
The attacker model (`docs/attacker-model.md`) has no expected-goals data per player. Goals and assists are predicted from
past goals, shots on target and the match odds only. Historical xG and xA (expected assists) per player per match for the
Championship, League One and League Two would help model goals and assists.
- Find a source covering all three EFL leagues for 2025/26 and 2026/27, ideally per match. Possible places to check:
  FBref / Opta, Understat (top leagues only?), FotMob, Sofascore, WhoScored. Check coverage and terms of use before
  relying on any of them.
- Map their player names / IDs to EFL fantasy player IDs (same approach as `ALIASES` in `predict_gw.py` for clubs).
- Add rolling xG / xA per 90 (and team xG) as features in `attack_model.py`, then re-run the same train / validate / test.

## Other open items
- Starter / minutes model (rotation history, injuries) instead of "90 if he played 60+ last game".
- Defenders: recalibrate the clearance model's top end (top xP fifth is over-predicted).
- Penalty takers: a goals boost for designated takers.
- Optional: automate the weekly refresh (currently Jack runs `run_gameweek.py` by hand).

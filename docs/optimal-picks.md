# Optimal Picks (homepage, v2, 9 Oct 2026)

https://efl-site.vercel.app/ (`public/index.html` + `optimal.js`). It replaced the old heuristic `picks.html`, which
now redirects here. The League Table page was removed (9 Oct 2026; it is in git history; `api/league.js` is now unused).

## What it shows
For a chosen gameweek: a pitch graphic in the Fantasy EFL style with the best 7 players, captain (C) and vice-captain
(V), the 2 clubs to pick, total expected points, each formation's best total, and the next-best options per position.

## Rules applied (Fantasy EFL help centre, checked 9 Oct 2026)
- 7 players: GK 1, DEF 2-3, MID 2-3, FWD 1-2. Formations 1-2-2-2, 1-2-3-1, 1-3-2-1.
- At most 2 players from one club per gameweek. The **One Club** chip (toggle) removes the limit.
- 2 clubs per gameweek; each club at most 5 times a season. Club picks do not count towards the player limit.
- Captain scores double. Vice-captain scores double only if the captain does not play. Clubs cannot be captained.
- Locking is game by game. You cannot switch captain to a player who has already played. Doubles score twice, blanks 0.
- **Max Captain** chip (twice a season): the captain becomes the actual top scorer. Not modelled; the page says so.

## How it works
- Player xP per gameweek uses the same data and maths as the position pages (`keeper_plan.json`, `defender_plan.json`,
  `mid_plan.json`, `fwd_plan.json`). Expected minutes are read from the minutes edits those pages save in this browser
  (`efl_keeper_mins_v1`, `efl_defender_mins_v1`, `efl_mid_mins_v1`, `efl_fwd_mins_v1`).
  - **The defender and attacker xP maths is copied in `optimal.js`.** If a model changes, update both places.
- Players: an exact branch and bound maximising sum of xP + captain's xP (captain = highest xP) for each formation, over the
  top 40 per position. Games that have already kicked off are **not** excluded (Jack's choice, 9 Oct 2026): the page
  shows the best team for the whole gameweek.
- Pin (must include) per player and the One Club chip apply to the gameweek they were set in only; exclude (x) applies to every gameweek. Saved in this browser (`efl_optimal_v1`: pinnedByGw, oneClubGws, excluded).
- Clubs: simply the two highest-xP clubs that gameweek. There are no Jack / John profiles here and picks left are
  ignored; the Club Planner handles picks left and season planning.
- Shirts: our own SVG shirt coloured with each club's colours from the EFL squad data (`backgroundColor`, `textColor`,
  `abbreviation`, written into `club_plan.json` by `season_plan.py`), plus a `KITS` table in `optimal.js` for stripes,
  hoops, halves, quarters and sleeves of well-known kits (approximate). Keepers get a generic keeper kit with club-colour
  sleeves. The EFL squad data also has official shirt image URLs (`jersey`); we do not use them (their artwork).

## Checks (9 Oct 2026, GW9)
- Player xP equals the position pages for every expected starter (GW9, 10, 12): defenders and forwards exactly, keepers
  within 0.0005 (rounding in the JSON). Midfielders use the same code as forwards.
- Branch and bound equals brute force on a restricted pool for all three formations; the full solve takes about 4 ms.

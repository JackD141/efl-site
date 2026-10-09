# efl-site ("Dexter's Corner")

Personal helper site for Fantasy EFL (English Football League fantasy game), by Jack Dexter. Static site plus Vercel
serverless functions, deployed from `main` at https://efl-site.vercel.app (pushing `main` deploys). Repo is public:
`JackD141/efl-site`. Two people use it: **Jack** (the repo owner, who you are talking to) and **John**, a friend.

## Layout

| Path | What |
|---|---|
| `public/` | the site (vanilla HTML/JS/CSS, no build step): `index.html` league table, `players.html` player stats + "Export Stats" button, `picks.html` player picks (simple heuristic), `clubs.html` **Club Planner**, `keepers.html` **Keeper Picks** (see docs) |
| `api/*.js` | Vercel functions (Node, `module.exports = async function handler(req, res)`): proxies to `fantasy.efl.com` JSON, `export-player-stats.js` (commits per-gameweek CSVs to GitHub), `club-picks.js` (cloud backup of club picks) |
| `data/<season>/player_stats_gwN.csv` | per-gameweek player stats, one folder per season (`2025_26`, `2026_27`). Written by the Export Stats button |
| `data/season_2025_26_final_totals.json` | final season-aggregate stats for last season, snapshotted before the API reset |
| `ml/` | notebooks for player-points models (WIP, not wired into the site) and `ml/team_strengths/` (team strengths + club expected points, **the active work**) |
| `vercel.json` | static output dir `public`, function limits, and `git.deploymentEnabled` off for the `picks-store` branch |

## Read these first

- `docs/club-planner.md`: what the Club Planner is, every design decision and why, test results, gotchas, state of play.
- `ml/team_strengths/README.md`: how to run the weekly pipeline and the model process/results.
- `docs/goalkeeper-model-plan.md`: the keeper model (v1 built: saves model, keeper xP, expected minutes) and the original plan; defenders not started.

## Commands

```bash
# weekly refresh (Jack runs this himself, then commits + pushes the JSON; do not automate unless asked)
# when he pastes Betfair prices: see ml/team_strengths/README.md "Odds overrides" (run_gameweek.py <gw> --override <csv>)
cd ml/team_strengths && ../../venv/Scripts/python.exe run_gameweek.py        # next pick-able gameweek
git add public/data/club_plan.json && git commit && git push                  # deploys the new data
```
Use the repo's venv: `./venv/Scripts/python.exe` (Windows). The shell is Git Bash on Windows (CRLF warnings are normal).
There is **no Node** on this machine, so API functions cannot be run locally: test their logic in the browser pane with
mocks (see `docs/club-planner.md`, "Testing").

## Hard-won facts

- **EFL's API resets at each season start.** Per-player game logs (`player_profiles/<id>.json` `results`) and `rounds.json`
  switch to the new season, so previous-season data must be captured before it rolls over. We only have last season's
  GW1-35 (and final aggregate totals). Stat files are now in season folders because the old flat `player_stats_gwN.csv`
  names were overwritten by the new season's GW1-3 (recovered from git history).
- The export function authenticates with env vars on Vercel (`EFL_EMAIL`, `EFL_PASSWORD`, `GITHUB_TOKEN`); none are
  available locally. Never put credentials in the repo.
- The built-in browser pane blocks pinnacle.com (and similar); do not try to scrape around it.
- Use dedicated tools; before committing run `git status` and add **specific files**. There is unrelated work in
  progress in the tree (untracked `ml/midfielder/*.ipynb`, `ml/odds_expected_points.ipynb`, `data/gw36_pred_*.csv`,
  a modified `data/opp_interceptions_by_league.png`). Do not `git add .`.
- Pushes to `main` redeploy the site. The `picks-store` branch is only written by `api/club-picks.js`; do not merge it.

## Working with Jack

- Wants concise answers, verified work (he checks numbers against Betfair), and honest reporting: an early claim that the
  model "reconciles closely" with bookmaker odds was overstated and he caught it; quote spreads and worst cases, not just means.
- He initiates refreshes and decides about commits/pushes; he has said "put it on our site" for the Club Planner, so
  pushing site changes for that feature is expected. Ask before anything broader (new workflows, new secrets, branches).
- Football context: he supports Lincoln City; irrelevant to the maths, just do not be surprised.

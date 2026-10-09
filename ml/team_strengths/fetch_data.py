"""Download everything the weekly run needs: football-data.co.uk results/odds and the EFL Fantasy rounds/squads."""
import json
from pathlib import Path

import requests

HERE = Path(__file__).parent
FD_DIR = HERE.parents[1] / "data" / "football_data"
CACHE = HERE / "cache"
SEASONS = ["2324", "2425", "2526", "2627"]
CURRENT = "2627"
DIVS = ["E1", "E2", "E3"]
UA = {"User-Agent": "Mozilla/5.0"}
EFL_HEADERS = {"Referer": "https://fantasy.efl.com/", **UA}


def _get(url, headers):
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    return r


def football_data(refresh_current=True):
    """Past seasons are static (downloaded once); the current season is refreshed every run."""
    FD_DIR.mkdir(parents=True, exist_ok=True)
    for s in SEASONS:
        for d in DIVS:
            f = FD_DIR / f"{s}_{d}.csv"
            if f.exists() and not (s == CURRENT and refresh_current):
                continue
            f.write_bytes(_get(f"https://www.football-data.co.uk/mmz4281/{s}/{d}.csv", UA).content)
    # upcoming fixtures with odds (only lists the next few days; midweek games usually missing)
    (CACHE).mkdir(exist_ok=True)
    (CACHE / "fd_fixtures.csv").write_bytes(_get("https://www.football-data.co.uk/fixtures.csv", UA).content)


def efl():
    CACHE.mkdir(exist_ok=True)
    out = {}
    for name in ("rounds", "squads"):
        data = _get(f"https://fantasy.efl.com/json/fantasy/{name}.json", EFL_HEADERS).json()
        (CACHE / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")
        out[name] = data
    return out["rounds"], out["squads"]


def efl_players():
    """All players with position, club, status and injury/suspension details (public, no login)."""
    data = _get("https://fantasy.efl.com/json/fantasy/players.json", EFL_HEADERS).json()
    (CACHE / "players.json").write_text(json.dumps(data), encoding="utf-8")
    return data

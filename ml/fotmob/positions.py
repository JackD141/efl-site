"""Where each starter lined up, from the cached FotMob match files.   python positions.py   -> data/fotmob/positions.csv

One row per starter per match: FotMob's formation slot (positionId), its place on the pitch diagram (depth: 0 = own
goal line, 1 = opponent's; side: 0.5 = central, 0 / 1 = the touchlines; `wide` = distance from the centre, 0-0.5) and
the team's formation. Substitutes have no slot in the lineup (FotMob only gives their usual position), so they are not
listed. Used by features.position_features (exp_position.py): the fantasy game's position (DEF / MID / FWD) is fixed for
a season, but what a player produces depends on the job he is actually given (a striker moved to attacking midfield,
a full-back used as a wing-back).
"""
import gzip
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
FM = REPO / "data" / "fotmob"
OUT = FM / "positions.csv"


def main():
    old = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["match_id"])
    done = set(old["match_id"])
    rows = []
    for f in sorted((FM / "raw").glob("*.json.gz")):
        mid = int(f.name.split(".")[0])
        if mid in done:
            continue
        try:
            lu = json.load(gzip.open(f, "rt", encoding="utf-8"))["content"]["lineup"]
        except (KeyError, TypeError, OSError, json.JSONDecodeError):
            continue
        for side in ("homeTeam", "awayTeam"):
            t = (lu or {}).get(side) or {}
            for p in t.get("starters") or []:
                lay = p.get("horizontalLayout") or {}
                if p.get("positionId") is None or "x" not in lay:
                    continue
                rows.append(dict(match_id=mid, fm_team_id=t.get("id"), fm_player_id=p["id"], position_id=p["positionId"], depth=lay["x"], side=lay["y"],
                                 wide=round(abs(lay["y"] - 0.5), 3), formation=t.get("formation")))
    out = pd.concat([old, pd.DataFrame(rows)], ignore_index=True) if rows else old
    out.to_csv(OUT, index=False)
    print(f"lineup positions: {len(out)} starters in {out['match_id'].nunique()} matches ({len(rows)} new) -> {OUT}")
    return out


if __name__ == "__main__":
    main()

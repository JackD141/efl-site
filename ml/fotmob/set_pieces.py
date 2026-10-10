"""Set-piece involvement per player per match from the cached FotMob match files (data/fotmob/raw/*.json.gz).
python set_pieces.py      writes data/fotmob/set_pieces.csv

From the shot map (each shot has a 'situation'): penalties taken / scored / their xG, direct free-kick shots and their xG.
Corners taken come from the per-match player stats ('Corners' in player_matches.csv). Features built from these
(features.py) use only matches before the one being predicted.
"""
import gzip
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
FM = REPO / "data" / "fotmob"


def main():
    rows = []
    for f in sorted((FM / "raw").glob("*.json.gz")):
        d = json.loads(gzip.decompress(f.read_bytes()))
        mid = int(d.get("general", {}).get("matchId") or f.name.split(".")[0])
        per = {}
        for s in ((d.get("content") or {}).get("shotmap") or {}).get("shots") or []:
            if s.get("isOwnGoal"):
                continue
            key = (s.get("playerId"), s.get("teamId"))
            r = per.setdefault(key, dict(pens=0, pen_goals=0, pen_xg=0.0, fk_shots=0, fk_xg=0.0))
            xg = float(s.get("expectedGoals") or 0)
            if s.get("situation") == "Penalty":
                r["pens"] += 1; r["pen_goals"] += s.get("eventType") == "Goal"; r["pen_xg"] += xg
            elif s.get("situation") == "FreeKick":
                r["fk_shots"] += 1; r["fk_xg"] += xg
        for (pid, tid), r in per.items():
            if pid is not None and (r["pens"] or r["fk_shots"]):
                rows.append(dict(match_id=mid, fm_player_id=int(pid), fm_team_id=int(tid), **r))
    out = pd.DataFrame(rows)
    out.to_csv(FM / "set_pieces.csv", index=False)
    print(f"{len(out)} player-match rows with a penalty or direct free kick; penalties {int(out['pens'].sum())} "
          f"(scored {int(out['pen_goals'].sum())}), free-kick shots {int(out['fk_shots'].sum())}")


if __name__ == "__main__":
    main()

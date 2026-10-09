"""Expected Fantasy EFL club points for every remaining gameweek, ranked, exported for the site's Club Planner page.

Uses the same fitted strengths for all future gameweeks (15-day memory was the most accurate at every horizon tested,
see README); weeks further out are less certain but the schedule (doubles/blanks) drives most of the difference.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from predict_gw import predict_gameweek

SITE_JSON = Path(__file__).resolve().parents[2] / "public" / "data" / "club_plan.json"


def _club_rows(fx):
    """One row per club-fixture for a gameweek's fixtures dataframe."""
    out = []
    for r in fx.itertuples():
        for side in ("H", "A"):
            h = side == "H"
            out.append(dict(
                club_id=int(r.home_id if h else r.away_id), opp_id=int(r.away_id if h else r.home_id),
                opp=r.away if h else r.home, ha=side, date=r.date, time=r.time,
                xp=r.ptsH if h else r.ptsA, win=r.pH if h else r.pA, draw=r.pD,
                cs=r.csH if h else r.csA, g2=r.g2H if h else r.g2A, g4=r.g4H if h else r.g4A,
                xg_for=r.xgH if h else r.xgA, xg_ag=r.xgA if h else r.xgH, src=r.source,
                bk=None if pd.isna(r.bookH) else [r.bookH, r.bookD, r.bookA] if h else [r.bookA, r.bookD, r.bookH]))
    return pd.DataFrame(out)


def build_plan(rounds, squads, fits, id2fd, book, as_of, league_names):
    season = sorted((r for r in rounds if r.get("gameMode", "season") == "season" and r["status"] != "completed"),
                    key=lambda r: r["roundNumber"])
    clubs = {cid: dict(id=cid, name=s["name"], short=s.get("shortName") or s["name"], league=league_names[id2fd[cid][1]], weeks=[])
             for cid, s in squads.items() if cid in id2fd}
    gameweeks, weekly = [], {}
    for rnd in season:
        gw = rnd["roundNumber"]
        fx = predict_gameweek(rounds, squads, gw, fits, id2fd, book)
        cr = _club_rows(fx)
        started = {g["homeId"] for g in rnd["games"] if g["status"] != "scheduled"} | {g["awayId"] for g in rnd["games"] if g["status"] != "scheduled"}
        per = cr.groupby("club_id").agg(xp=("xp", "sum"), games=("opp", "count"), cs=("cs", "sum"), win=("win", "sum"))
        week = pd.DataFrame(index=list(clubs), data=dict(xp=0.0, games=0, cs=0.0, win=0.0))
        week.loc[per.index, ["xp", "games", "cs", "win"]] = per[["xp", "games", "cs", "win"]].to_numpy()
        week["rank"] = week["xp"].rank(ascending=False, method="min").astype(int)
        for cid, row in week.iterrows():
            fixtures = []
            if row["games"]:
                for f in cr[cr.club_id == cid].sort_values(["date", "time"]).itertuples():
                    fixtures.append(dict(opp=f.opp, ha=f.ha, date=f.date, xp=round(f.xp, 2), win=round(f.win, 4), draw=round(f.draw, 4),
                                         cs=round(f.cs, 4), g2=round(f.g2, 4), g4=round(f.g4, 4), src=f.src,
                                         bk=None if f.bk is None else [round(x, 4) for x in f.bk]))
            clubs[cid]["weeks"].append(dict(gw=gw, xp=round(row["xp"], 2), rank=int(row["rank"]), games=int(row["games"]),
                                            locked=bool(cid in started), fx=fixtures))
        dates = sorted(g["date"][:10] for g in rnd["games"])
        gameweeks.append(dict(gw=gw, lockout=rnd["lockoutDate"], start=dates[0], end=dates[-1], games=len(rnd["games"]),
                              marketGames=int((fx.source == "market").sum()), modelGames=int((fx.source == "model").sum()),
                              doubles=int((week["games"] >= 2).sum()), blanks=int((week["games"] == 0).sum())))
        top = week[week["games"] > 0].sort_values("xp", ascending=False).head(10)
        weekly[str(gw)] = [dict(club=int(c), xp=round(r["xp"], 2), rank=int(r["rank"])) for c, r in top.iterrows()]
    for c in clubs.values():
        played = [w for w in c["weeks"] if w["games"] > 0]
        best = sorted(played, key=lambda w: w["xp"], reverse=True)[:5]
        c["top5"] = [w["gw"] for w in best]
        c["top5Total"] = round(sum(w["xp"] for w in best), 2)
    return dict(
        generatedAt=datetime.now(timezone.utc).isoformat(timespec="seconds"), asOf=str(as_of.date()), season="2026/27",
        firstGw=gameweeks[0]["gw"], lastGw=gameweeks[-1]["gw"], gameweeks=gameweeks, clubs=list(clubs.values()), weekly=weekly,
        scoring=dict(win=5, draw=3, awayWin=2, cleanSheet=2, twoGoals=2, fourGoals=2),
        notes=dict(
            picksPerWeek=2, maxUsesPerClub=5,
            horizon="Expected points come from our team-strength model fitted to bookmaker odds so far this season. "
                    "Accuracy falls with distance: about 2pp average error on win probabilities next week, 3.8pp 9-12 weeks out. "
                    "Games with bookmaker odds are marked market; the rest are model estimates."))


def write_site_json(plan, path=SITE_JSON):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, separators=(",", ":")), encoding="utf-8")
    return path

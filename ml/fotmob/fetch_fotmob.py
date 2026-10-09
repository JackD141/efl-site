"""Pull per-player match stats (xG, xA, minutes, started, shots, ...) for the Championship, League One and League Two
from FotMob, for building the player models.

    python fetch_fotmob.py                      # all seasons in SEASONS, only matches not fetched yet
    python fetch_fotmob.py 2026/2027            # one season (run weekly to add new matches)

Polite by design: one request at a time with a pause, every match fetched once and cached (gzip) in data/fotmob/raw/
(gitignored). Output: data/fotmob/<season>/player_matches.csv (one row per player per match, all leagues) and
matches.csv (one row per match). FotMob's terms and robots.txt do not allow automated access; Jack chose to use it
anyway (9 Oct 2026) for this personal project. If FotMob blocks or challenges the requests, stop: do not work around it.
"""
import gzip
import os
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "fotmob"
RAW = OUT / "raw"
LEAGUES = {48: "Championship", 108: "League One", 109: "League Two"}
SEASONS = ["2024/2025", "2025/2026", "2026/2027"]
PAUSE = float(os.environ.get("FOTMOB_PAUSE", "1.5"))  # seconds between requests (per process)
API = "https://www.fotmob.com/api/data"
HEADERS = {"User-Agent": "Mozilla/5.0 (personal fantasy-football research; low rate)"}

session = requests.Session()
session.headers.update(HEADERS)


def get_json(url, params):
    for attempt in range(3):
        r = session.get(url, params=params, timeout=30)
        time.sleep(PAUSE)
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"):
            return r.json()
        if r.status_code in (403, 429):
            raise SystemExit(f"FotMob refused the request ({r.status_code}) - stopping. Try again later; do not work around it.")
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed: {url} {params} -> {r.status_code}")


def season_matches(league_id, season):
    d = get_json(f"{API}/leagues", {"id": league_id, "season": season})
    out = []
    for m in d["fixtures"]["allMatches"]:
        st = m["status"]
        out.append(dict(match_id=int(m["id"]), league=LEAGUES[league_id], season=season, round=m.get("round"),
                        utc=st.get("utcTime"), home=m["home"]["name"], home_id=int(m["home"]["id"]),
                        away=m["away"]["name"], away_id=int(m["away"]["id"]), finished=bool(st.get("finished")),
                        cancelled=bool(st.get("cancelled")), score=st.get("scoreStr")))
    return out


def match_details(match_id):
    f = RAW / f"{match_id}.json.gz"
    if f.exists():
        return json.loads(gzip.decompress(f.read_bytes()))
    d = get_json(f"{API}/matchDetails", {"matchId": match_id})
    RAW.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(json.dumps(d).encode("utf-8")))
    tmp.replace(f)  # atomic, so a parallel reader never sees half a file
    return d


def player_rows(match, d):
    c = d.get("content") or {}
    lineup = c.get("lineup") or {}
    started, sub_in, sub_out = set(), {}, {}
    for side in ("homeTeam", "awayTeam"):
        t = lineup.get(side) or {}
        for p in t.get("starters") or []:
            started.add(p["id"])
        for p in (t.get("starters") or []) + (t.get("subs") or []):
            for ev in ((p.get("performance") or {}).get("substitutionEvents") or []):
                (sub_in if ev.get("type") == "subIn" else sub_out)[p["id"]] = ev.get("time")
    rows = []
    for p in (c.get("playerStats") or {}).values():
        row = dict(match_id=match["match_id"], league=match["league"], season=match["season"], utc=match["utc"],
                   fm_player_id=p["id"], opta_id=p.get("optaId"), name=p["name"], fm_team_id=p.get("teamId"),
                   team=p.get("teamName"), is_gk=bool(p.get("isGoalkeeper")), usual_position=p.get("usualPosition"),
                   started=p["id"] in started, sub_in=sub_in.get(p["id"]), sub_out=sub_out.get(p["id"]))
        for blk in p.get("stats") or []:
            for key, v in (blk.get("stats") or {}).items():
                stat = v.get("stat") or {}
                if key in ("Shotmap",) or key in row:
                    continue
                row[key] = stat.get("value")
                if stat.get("total") is not None and stat.get("type") == "fractionWithPercentage":
                    row[key + " (attempts)"] = stat.get("total")
        rows.append(row)
    return rows


def run(seasons):
    for season in seasons:
        tag = season.replace("/", "_")
        (OUT / tag).mkdir(parents=True, exist_ok=True)
        matches = []
        for lid in LEAGUES:
            matches += season_matches(lid, season)
        mdf = pd.DataFrame(matches)
        mdf.to_csv(OUT / tag / "matches.csv", index=False)
        pm_path = OUT / tag / "player_matches.csv"
        old = pd.read_csv(pm_path) if pm_path.exists() else pd.DataFrame()
        done = set(old["match_id"]) if len(old) else set()
        todo = [m for m in matches if m["finished"] and not m["cancelled"] and m["match_id"] not in done]
        print(f"{season}: {len(matches)} matches, {len(todo)} to fetch", flush=True)
        new = []
        for i, m in enumerate(todo, 1):
            try:
                new += player_rows(m, match_details(m["match_id"]))
            except RuntimeError as e:
                print("  skip", m["match_id"], e, flush=True)
            if i % 50 == 0 or i == len(todo):
                out = pd.concat([old, pd.DataFrame(new)], ignore_index=True)
                out.to_csv(pm_path, index=False)
                print(f"  {i}/{len(todo)} matches, {len(out)} player rows", flush=True)


def prefetch(seasons):
    """Only download raw match files, newest match first (run alongside run() to speed it up; run() then reads the cache)."""
    for season in seasons:
        mdf = pd.read_csv(OUT / season.replace("/", "_") / "matches.csv")
        todo = mdf[mdf["finished"] & ~mdf["cancelled"]].sort_values("utc", ascending=False)["match_id"]
        for i, mid in enumerate(todo, 1):
            if not (RAW / f"{mid}.json.gz").exists():
                match_details(int(mid))
            if i % 100 == 0:
                print(f"  prefetch {season}: {i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--prefetch"]:
        prefetch(sys.argv[2:])
    else:
        run(sys.argv[1:] or SEASONS)

"""Past seasons in the EFL fantasy format, rebuilt from FotMob (21 Nov 2026).

    python build_fantasy_history.py 2024/2025          # writes data/fotmob/fantasy_format/2024_25/player_stats_gw<N>.csv
    python build_fantasy_history.py 2025/2026 --check  # rebuilds a season we have from the EFL and compares, row by row

Why: the EFL fantasy API only serves the current season, so our own fantasy-format data starts in 2025/26. FotMob has
the same per-player match stats (Opta) for earlier seasons. Measured on 55,449 appearances of 2025/26-26/27 where we have
both: minutes 96-98% identical, goals 99.9%, shots on target 99%, key passes = FotMob "chances created" 97%, blocks 96%,
saves 94%, keeper goals conceded 99.6%, MID interceptions 91% (ratio 1.01), DEF tackles 88%, DEF clearances 73% exact /
92% within one (FotMob 4% higher). Assists: FotMob's are the real ones; the fantasy game adds extra assists (1.9% of
appearances differ, the EFL 25-38% higher in total), so assists here are LOW against the EFL's.

Same columns as the EFL export (api/export-player-stats.js). How each one is made:
- one row per player per club fixture for every player in the club's matchday squads that season, from his first to his
  last squad listing (the EFL lists every registered player, FotMob only the matchday squad; 0-minute rows in between)
- player_id: the EFL id when the FotMob player is linked (player_map.csv), else 10,000,000 + FotMob id
- position: his EFL fantasy position when linked (the CURRENT one: positions can change between seasons), else FotMob's
  usual position (0 GK, 1 DEF, 2 MID, 3 FWD)
- squad_id / opponent_id: EFL squad ids by name (link_fotmob.team_map), else 9,000,000 + FotMob team id (clubs no longer
  in the EFL)
- gameweek: weeks from Friday to Thursday from the season's first Friday (a midweek game falls in the same gameweek as
  the weekend before it, as fantasy gameweeks do, giving doubles)
- stats as FotMob reports them; goals conceded = opponents' goals while he was on the pitch (match events + his sub
  minutes); clean sheet = none conceded while on and 60+ minutes; cards from the events (a second yellow = 1 red,
  0 yellows, as the EFL records it); hat-trick = 3+ goals
- points: team_strengths/data_checks.recompute_points (the EFL's scoring, checked against every 2025/26+ appearance)
FotMob data is not in the repo (data/fotmob/ is gitignored); neither is this output.
"""
import glob
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
REPO = HERE.parents[1]
FM = REPO / "data" / "fotmob"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "ml" / "team_strengths"))
import link_fotmob as lf  # noqa: E402

COLS = ["player_id", "first_name", "last_name", "display_name", "position", "squad_id", "gameweek", "opponent_id", "opponent_name",
        "fixture_difficulty", "is_home", "minutes_played", "goals_scored", "hat_tricks", "assists", "penalty_misses", "own_goals",
        "yellow_cards", "red_cards", "saves", "penalty_saves", "clean_sheet", "goals_conceded", "clearances", "blocks", "tackles",
        "interceptions", "key_passes", "shots_on_target", "points"]
STAT = {"goals_scored": "Goals", "assists": "Assists", "penalty_misses": "Missed penalty", "own_goals": "Own goal", "saves": "Saves",
        "penalty_saves": "Saved penalties", "clearances": "Clearances", "blocks": "Blocks", "tackles": "Tackles",
        "interceptions": "Interceptions", "key_passes": "Chances created", "shots_on_target": "Shots on target"}
POS = {0: "GK", 1: "DEF", 2: "MID", 3: "FWD"}
END = 200  # "still on at the end"


def season_dir(fm_season):
    """Live seasons live in data/fotmob/<season>/, older ones in data/fotmob/history/<season>/ (fetch_history.py)."""
    return FM / fm_season if (FM / fm_season / "player_matches.csv").exists() else FM / "history" / fm_season


def events(match_id):
    """(goals [(minute, scoring side 'H'/'A')], cards {fm player id: (yellows, red)}, red minute {fm player id: minute})."""
    f = FM / "raw" / f"{match_id}.json.gz"
    if not f.exists():
        f = FM / "history" / "raw" / f"{match_id}.json.gz"
    if not f.exists():
        return None
    j = json.load(gzip.open(f))
    ev = (((j.get("content") or {}).get("matchFacts") or {}).get("events") or {}).get("events") or []
    goals, cards, red_at = [], {}, {}
    for e in ev:
        t = (e.get("time") or 0) + ((e.get("overloadTime") or 0) / 100)
        if e.get("type") == "Goal" and not e.get("isPenaltyShootoutEvent"):
            side = "H" if e.get("isHome") else "A"
            goals.append((t, side))
        elif e.get("type") == "Card" and e.get("playerId"):
            pid, c = int(e["playerId"]), str(e.get("card") or "")
            y, r = cards.get(pid, (0, 0))
            if c == "Yellow":
                y += 1
            elif c in ("Red", "YellowRed"):
                r = 1
                red_at[pid] = t
            cards[pid] = (y, r)
    for pid, (y, r) in list(cards.items()):
        if y >= 2:
            r, red_at[pid] = 1, red_at.get(pid, END)
        cards[pid] = (0, 1) if r else (min(y, 1), 0)
    return goals, cards, red_at


def strict_team_map(squads, fm_names):
    """{FotMob team name: EFL squad id}: exact normalised name / short name, or a known alias; otherwise unmapped (a club
    not in the EFL now gets its own id). No fuzzy matching: link_fotmob.team_map's similarity fallback mapped Carlisle
    United to Cambridge United for 2024/25."""
    efl = {}
    for s_ in squads:
        for k in (s_["name"], s_.get("shortName") or s_["name"]):
            efl[lf.norm(k)] = s_["id"]
    for a, b in lf.TEAM_ALIASES.items():  # both directions
        if a in efl:
            efl.setdefault(b, efl[a])
        if b in efl:
            efl.setdefault(a, efl[b])
    return {n_: efl[lf.norm(n_)] for n_ in fm_names if lf.norm(n_) in efl}


def season_rows(fm_season):
    pm = pd.read_csv(season_dir(fm_season) / "player_matches.csv", low_memory=False)
    mt = pd.read_csv(season_dir(fm_season) / "matches.csv")
    mt = mt[mt["finished"] & ~mt["cancelled"]].copy()
    mt["utc"] = pd.to_datetime(mt["utc"], utc=True)
    first_fri = (mt["utc"].min().tz_convert("Europe/London").normalize().tz_localize(None))
    first_fri -= pd.Timedelta(days=(first_fri.weekday() - 4) % 7)
    mt["date"] = mt["utc"].dt.tz_convert("Europe/London").dt.tz_localize(None).dt.normalize()
    mt["gameweek"] = ((mt["date"] - first_fri).dt.days // 7 + 1).astype(int)
    squads = json.load(open(REPO / "ml" / "team_strengths" / "cache" / "squads.json", encoding="utf-8"))
    old = json.load(open(REPO / "data" / "squads.json", encoding="utf-8"))
    names = {s["id"]: s for s in old}
    names.update({s["id"]: s for s in squads})
    tm = strict_team_map(list(names.values()), set(mt["home"]) | set(mt["away"]))
    name_of = {**dict(zip(mt["home_id"], mt["home"])), **dict(zip(mt["away_id"], mt["away"]))}  # FotMob team id -> name
    club_of_id = {int(i): (int(tm[n]) if n in tm else 9_000_000 + int(i)) for i, n in name_of.items()}
    club = lambda n: club_of_id[int(n)]
    mp = pd.read_csv(FM / "player_map.csv")[["fm_player_id", "player_id"]].drop_duplicates("fm_player_id")
    efl_players = {p["id"]: p for p in json.load(open(REPO / "ml" / "team_strengths" / "cache" / "players.json", encoding="utf-8"))}
    pm = pm.merge(mt[["match_id", "home_id", "away_id", "date", "gameweek", "score"]], on="match_id").merge(mp, on="fm_player_id", how="left")
    pm["team"] = pm["fm_team_id"].astype(int)
    pm["linked"] = pm["player_id"].notna()
    pm["player_id"] = pm["player_id"].fillna(10_000_000 + pm["fm_player_id"]).astype(int)
    pm["squad_id"] = pm["team"].map(club)
    home = pm["team"] == pm["home_id"]
    opp = np.where(home, pm["away_id"], pm["home_id"]).astype(int)
    pm["opponent_name"] = [name_of[o] for o in opp]
    pm["opponent_id"] = [club(o) for o in opp]
    pm["is_home"] = np.where(home, "H", "A")
    pm["mins"] = pd.to_numeric(pm["Minutes played"], errors="coerce").fillna(0)
    # goals conceded while on the pitch, cards
    ev = {m: events(m) for m in pm["match_id"].unique()}
    gc, yel, red = np.zeros(len(pm)), np.zeros(len(pm)), np.zeros(len(pm))
    for i, r in enumerate(pm.itertuples()):
        e = ev.get(r.match_id)
        if e is None or r.mins <= 0:
            continue
        goals, cards, red_at = e
        on = 0 if r.started else (r.sub_in if r.sub_in == r.sub_in else 0)
        off = r.sub_out if r.sub_out == r.sub_out else END
        off = min(off, red_at.get(int(r.fm_player_id), END))
        gc[i] = sum(1 for t, side in goals if side != r.is_home and on <= t <= off + 0.99)
        yel[i], red[i] = cards.get(int(r.fm_player_id), (0, 0))
    pm["goals_conceded"], pm["yellow_cards"], pm["red_cards"] = gc, yel, red
    for k, v in STAT.items():
        pm[k] = pd.to_numeric(pm.get(v), errors="coerce").fillna(0)
    pm["hat_tricks"] = (pm["goals_scored"] >= 3).astype(int)
    pm["clean_sheet"] = ((pm["goals_conceded"] == 0) & (pm["mins"] >= 60)).astype(int)
    pm["minutes_played"] = pm["mins"].astype(int)
    pos_fm = pd.to_numeric(pm["usual_position"], errors="coerce").map(POS).fillna("MID")
    pm["position"] = [efl_players[p]["position"] if lk and p in efl_players else f for p, lk, f in zip(pm["player_id"], pm["linked"], pos_fm)]
    nm = pm["name"].fillna("").str.split(" ", n=1)
    pm["first_name"], pm["last_name"] = nm.str[0], nm.str[1].fillna("")
    pm["display_name"] = [efl_players[p]["displayName"] if p in efl_players else (f"{a[0]}. {b}" if a and b else a) for p, a, b in zip(pm["player_id"], pm["first_name"], pm["last_name"])]
    pm["fixture_difficulty"] = ""
    played = pm.copy()
    # 0-minute rows for the club's fixtures between his first and last matchday squad listing (FotMob lists only squads)
    fx = pd.concat([pd.DataFrame({"match_id": mt["match_id"], "team": mt["home_id"].astype(int), "opp": mt["away_id"].astype(int), "is_home": "H", "date": mt["date"], "gameweek": mt["gameweek"]}),
                    pd.DataFrame({"match_id": mt["match_id"], "team": mt["away_id"].astype(int), "opp": mt["home_id"].astype(int), "is_home": "A", "date": mt["date"], "gameweek": mt["gameweek"]})])
    span = played.groupby(["player_id", "team"]).agg(a=("date", "min"), b=("date", "max")).reset_index()
    full = span.merge(fx, on="team")
    full = full[(full["date"] >= full["a"]) & (full["date"] <= full["b"])]
    have = set(zip(played["player_id"], played["match_id"]))
    miss = full[[(p, m) not in have for p, m in zip(full["player_id"], full["match_id"])]]
    info = played.drop_duplicates(["player_id", "team"]).set_index(["player_id", "team"])
    z = miss.join(info[["first_name", "last_name", "display_name", "position", "squad_id"]], on=["player_id", "team"])
    z = z.assign(opponent_name=z["opp"].map(name_of), opponent_id=z["opp"].map(club), fixture_difficulty="",
                 **{c: 0 for c in ["minutes_played", "goals_scored", "hat_tricks", "assists", "penalty_misses", "own_goals", "yellow_cards",
                                   "red_cards", "saves", "penalty_saves", "clean_sheet", "goals_conceded", "clearances", "blocks", "tackles",
                                   "interceptions", "key_passes", "shots_on_target"]})
    out = pd.concat([played, z], ignore_index=True)
    from data_checks import recompute_points
    out["points"] = 0
    out["points"] = recompute_points(out)
    return out.sort_values(["gameweek", "squad_id", "player_id"])


def check(fm_season, efl_season):
    """Rebuild a season we also have from the EFL; compare stats and points on the same player-matches."""
    b = season_rows(fm_season)
    e = pd.concat([pd.read_csv(f) for f in glob.glob(str(REPO / "data" / efl_season / "player_stats_gw*.csv")) if Path(f).stat().st_size > 100])
    key = ["player_id", "squad_id", "opponent_id", "is_home"]
    e2 = e[e["minutes_played"] > 0].drop_duplicates(key)
    m = e2.merge(b[b["minutes_played"] > 0].drop_duplicates(key), on=key, suffixes=("", "_fm"))
    print(f"{efl_season}: EFL appearances {len(e2)}, rebuilt from FotMob {int((b['minutes_played'] > 0).sum())}, matched {len(m)}")
    for c in ["minutes_played", "goals_scored", "assists", "yellow_cards", "red_cards", "goals_conceded", "clean_sheet", "saves", "clearances",
              "blocks", "tackles", "interceptions", "key_passes", "shots_on_target", "points"]:
        rel = m if c not in ("goals_conceded", "clean_sheet") else m[m["position"].isin(["GK", "DEF"])]
        if c == "interceptions":
            rel = m[m["position"] == "MID"]
        if c in ("clearances", "blocks", "tackles"):
            rel = m[m["position"] == "DEF"]
        if c in ("key_passes", "shots_on_target"):
            rel = m[m["position"].isin(["MID", "FWD"])]
        if c == "saves":
            rel = m[m["position"] == "GK"]
        x, y = rel[c + "_fm"], rel[c]
        print(f"   {c:16s} n {len(rel):6d}  exact {(x == y).mean():6.1%}  within 1 {((x - y).abs() <= 1).mean():6.1%}  EFL total {y.sum():8.0f}  FotMob {x.sum():8.0f}")
    for pos in ("GK", "DEF", "MID", "FWD"):
        r = m[m["position"] == pos]
        print(f"   points {pos}: exact {(r['points_fm'] == r['points']).mean():.1%}, mean EFL {r['points'].mean():.3f} vs rebuilt {r['points_fm'].mean():.3f}, "
              f"correlation {np.corrcoef(r['points'], r['points_fm'])[0, 1]:.3f}")
    # coverage of 0-minute rows (the start model trains on them)
    per_fx_e = e.groupby(["squad_id", "gameweek", "opponent_id"]).size().mean()
    per_fx_b = b.groupby(["squad_id", "gameweek", "opponent_id"]).size().mean()
    print(f"   rows per club fixture: EFL {per_fx_e:.1f}, rebuilt {per_fx_b:.1f} (the EFL lists every registered player)")


def main(season, do_check=False):
    fm_season = season.replace("/", "_")
    tag = f"{season[:4]}_{season[-2:]}"
    if do_check:
        check(fm_season, tag)
        return
    out = season_rows(fm_season)
    d = FM / "fantasy_format" / tag
    d.mkdir(parents=True, exist_ok=True)
    for gw, g in out.groupby("gameweek"):
        g[COLS].to_csv(d / f"player_stats_gw{gw}.csv", index=False)
    print(f"wrote {out['gameweek'].nunique()} gameweeks, {len(out)} rows ({int((out['minutes_played'] > 0).sum())} appearances) to {d}")


if __name__ == "__main__":
    main(sys.argv[1], "--check" in sys.argv)

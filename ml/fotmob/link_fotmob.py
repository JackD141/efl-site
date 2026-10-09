"""Link FotMob player-match rows to the EFL fantasy rows (same player, same match).
python link_fotmob.py     writes data/fotmob/player_map.csv and prints coverage

1. clubs: FotMob team name -> EFL squad id (normalised names + aliases), per season
2. matches: (season, home squad, away squad) is unique in a league season
3. players: within each linked match and club, pair FotMob and EFL players by name similarity and equal minutes;
   a FotMob id maps to the EFL id it pairs with most often (EFL ids are stable across seasons and clubs).
fotmob_for_efl(position) returns the EFL rows with FotMob stats attached (xg, npxg, xa, shots, ...).
"""
import difflib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
FM = REPO / "data" / "fotmob"
SEASONS = {"2025_26": "2025_2026", "2026_27": "2026_2027"}
TEAM_ALIASES = {"mkdons": "miltonkeynesdons", "wolves": "wolverhamptonwanderers", "qpr": "queensparkrangers",
                "westbrom": "westbromwichalbion", "sheffieldutd": "sheffieldunited", "sheffieldwed": "sheffieldwednesday",
                "manutd": "manchesterunited", "prestonnorthend": "preston", "leytonorient": "orient"}
STATS = {"Expected goals (xG)": "xg", "xG Non-penalty": "npxg", "Expected assists (xA)": "xa", "Total shots": "shots",
         "Shots on target": "fm_sot", "Chances created": "chances", "Touches in opposition box": "box_touches",
         "Touches": "touches", "Minutes played": "fm_minutes", "Interceptions": "fm_int", "Goals": "fm_goals",
         "Assists": "fm_assists", "Expected goals on target (xGOT)": "xgot"}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z]", "", s.replace("&", "and"))
    for w in ("afc", "fc"):
        s = s.replace(w, "") if len(s) > len(w) + 3 else s
    return TEAM_ALIASES.get(s, s)


def team_map(squads, fm_names):
    """{fotmob team name: squad id} by best normalised-name match (prefix / similarity)."""
    efl = {norm(s["name"]): s["id"] for s in squads}
    efl.update({norm(s.get("shortName") or s["name"]): s["id"] for s in squads})
    out = {}
    for name in fm_names:
        n = norm(name)
        if n in efl:
            out[name] = efl[n]; continue
        cands = [k for k in efl if k.startswith(n) or n.startswith(k)]
        if not cands:
            cands = difflib.get_close_matches(n, list(efl), n=1, cutoff=0.75)
        if cands:
            out[name] = efl[sorted(cands, key=lambda k: -difflib.SequenceMatcher(None, n, k).ratio())[0]]
    return out


def name_score(fm_name, first, last, display):
    a = norm(fm_name)
    best = 0.0
    for b in (norm(f"{first}{last}"), norm(last), norm(display)):
        if not b:
            continue
        r = difflib.SequenceMatcher(None, a, b).ratio()
        if norm(last) and a.endswith(norm(last)):
            r = max(r, 0.85)
        best = max(best, r)
    return best


def load_efl(season):
    frames = []
    for f in sorted((REPO / "data" / season).glob("player_stats_gw*.csv")):
        try:
            frames.append(pd.read_csv(f))
        except pd.errors.EmptyDataError:
            pass
    return pd.concat(frames, ignore_index=True).drop_duplicates(["player_id", "gameweek", "squad_id", "opponent_id"])


def build_map():
    pairs = []
    for efl_season, fm_season in SEASONS.items():
        pm_path = FM / fm_season / "player_matches.csv"
        if not pm_path.exists():
            print(f"{fm_season}: no FotMob data yet"); continue
        pm = pd.read_csv(pm_path)
        matches = pd.read_csv(FM / fm_season / "matches.csv")
        squads = json.load(open(REPO / "data" / "squads.json", encoding="utf-8")) if efl_season == "2025_26" else \
            json.load(open(REPO / "ml" / "team_strengths" / "cache" / "squads.json", encoding="utf-8"))
        tm = team_map(squads, set(matches["home"]) | set(matches["away"]))
        missing = sorted(set(matches["home"]) - set(tm))
        if missing:
            print(f"{fm_season}: unmapped FotMob teams {missing}")
        matches["hid"], matches["aid"] = matches["home"].map(tm), matches["away"].map(tm)
        efl = load_efl(efl_season)
        efl = efl[efl["minutes_played"] > 0]
        efl["hid"] = efl["squad_id"].where(efl["is_home"] == "H", efl["opponent_id"])
        efl["aid"] = efl["opponent_id"].where(efl["is_home"] == "H", efl["squad_id"])
        pm = pm.merge(matches[["match_id", "hid", "aid"]], on="match_id")
        pm["squad_id"] = pm["team"].map(tm)
        pm = pm[pm["Minutes played"].fillna(0) > 0]
        for (hid, aid, sq), g in pm.groupby(["hid", "aid", "squad_id"]):
            e = efl[(efl["hid"] == hid) & (efl["aid"] == aid) & (efl["squad_id"] == sq)]
            if e.empty:
                continue
            for r in g.to_dict("records"):
                score = e.apply(lambda q: name_score(r["name"], q["first_name"], q["last_name"], q["display_name"]), axis=1)
                score = score + 0.3 * ((e["minutes_played"] - r["Minutes played"]).abs() <= 2)
                j = score.idxmax()
                if score[j] >= 0.9:
                    pairs.append(dict(fm_player_id=r["fm_player_id"], fm_name=r["name"], player_id=int(e.at[j, "player_id"]),
                                      efl_name=f"{e.at[j, 'first_name']} {e.at[j, 'last_name']}", score=float(score[j])))
    p = pd.DataFrame(pairs)
    votes = p.groupby(["fm_player_id", "player_id"]).agg(n=("score", "size"), score=("score", "mean"), fm_name=("fm_name", "first"), efl_name=("efl_name", "first")).reset_index()
    best = votes.sort_values("n", ascending=False).drop_duplicates("fm_player_id").drop_duplicates("player_id")
    best.to_csv(FM / "player_map.csv", index=False)
    print(f"linked {len(best)} players ({p['fm_player_id'].nunique()} FotMob ids seen in pairs)")
    return best


def fotmob_for_efl():
    """FotMob stats per EFL row key (season tag, player_id, squad_id, opponent_id, is_home)."""
    mp = pd.read_csv(FM / "player_map.csv")[["fm_player_id", "player_id"]]
    out = []
    for efl_season, fm_season in SEASONS.items():
        pm_path = FM / fm_season / "player_matches.csv"
        if not pm_path.exists():
            continue
        pm = pd.read_csv(pm_path)
        matches = pd.read_csv(FM / fm_season / "matches.csv")
        squads = json.load(open(REPO / "data" / "squads.json", encoding="utf-8")) if efl_season == "2025_26" else \
            json.load(open(REPO / "ml" / "team_strengths" / "cache" / "squads.json", encoding="utf-8"))
        tm = team_map(squads, set(matches["home"]) | set(matches["away"]))
        pm = pm.merge(matches[["match_id", "home", "away"]], on="match_id").merge(mp, on="fm_player_id")
        pm["squad_id"] = pm["team"].map(tm)
        home = pm["team"] == pm["home"]
        pm["opponent_id"] = pm["away"].map(tm).where(home, pm["home"].map(tm))
        pm["is_home"] = home.map({True: "H", False: "A"})
        pm["season"] = {"2025_26": "2526", "2026_27": "2627"}[efl_season]
        keep = ["season", "player_id", "squad_id", "opponent_id", "is_home", "started", "sub_in", "sub_out"] + [c for c in STATS if c in pm]
        out.append(pm[keep].rename(columns=STATS))
    d = pd.concat(out, ignore_index=True)
    for c in STATS.values():
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.drop_duplicates(["season", "player_id", "squad_id", "opponent_id", "is_home"])


if __name__ == "__main__":
    best = build_map()
    f = fotmob_for_efl()
    for season, tag in (("2025_26", "2526"), ("2026_27", "2627")):
        e = load_efl(season)
        e = e[e["minutes_played"] > 0].assign(season=tag)
        m = e.merge(f, on=["season", "player_id", "squad_id", "opponent_id", "is_home"], how="left")
        if m["xg"].notna().any():
            cov = m.groupby("position")["xg"].apply(lambda s: s.notna().mean()).round(3).to_dict()
            agree = (m["fm_minutes"] - m["minutes_played"]).abs().le(3).mean()
            print(f"{season}: appearances with FotMob xG by position {cov}; minutes agree (±3) on {agree:.1%}; "
                  f"goals agree {(m['fm_goals'] == m['goals_scored']).mean():.1%}")

"""Player features from FotMob history for the stat models.

signal_rates(rows) adds, for each EFL row (needs player_id and date), the player's per-90 rates over his last N FotMob
appearances strictly BEFORE that date (any club, any of the three leagues, 2024/25 onwards), shrunk towards the league
rate with K pseudo-90s:  npxg (non-penalty xG), pxg (penalty xG: identifies penalty takers), xa, shots, chances.
Rows: FotMob player_matches.csv for every season pulled, linked to EFL ids with player_map.csv (link_fotmob.py).
"""
import functools
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
FM = REPO / "data" / "fotmob"
N_GAMES = 20
SIGNALS = {"npxg": "xG Non-penalty", "xg": "Expected goals (xG)", "xa": "Expected assists (xA)", "shots": "Total shots",
           "chances": "Chances created", "sot": "Shots on target", "int": "Interceptions", "xgot": "Expected goals on target (xGOT)",
           "clr": "Clearances", "blk": "Blocks", "tkl": "Tackles"}
RATES = ("npxg", "pxg", "xa", "shots", "chances", "sot", "int", "xgot", "clr", "blk", "tkl")


def _files_signature():
    """Changes whenever a FotMob file is rewritten (fetch / link / set pieces), so cached tables are rebuilt."""
    files = sorted(FM.glob("*/player_matches.csv")) + [FM / "player_map.csv", FM / "set_pieces.csv"]
    return tuple((str(f), f.stat().st_mtime_ns, f.stat().st_size) for f in files if f.exists())


def group_rolling_sum(h, cols, n):
    """Per player (h sorted by player then date, default index): each column summed over his last n rows, current one
    included. One vectorised pass (the per-player Python loop it replaces was most of the run time)."""
    r = h.groupby("player_id", sort=False)[list(cols)].rolling(n, min_periods=1).sum()
    return r.reset_index(level=0, drop=True).reindex(h.index)


def fotmob_history():
    """Every linked FotMob appearance (cached for the process; a copy, so callers may alter it)."""
    return _fotmob_history(_files_signature()).copy()


@functools.lru_cache(maxsize=1)
def _fotmob_history(_signature):
    mp = pd.read_csv(FM / "player_map.csv")[["fm_player_id", "player_id"]]
    frames = []
    for f in sorted(FM.glob("*/player_matches.csv")):
        pm = pd.read_csv(f, low_memory=False)
        keep = ["fm_player_id", "utc", "league", "Minutes played"] + [c for c in SIGNALS.values() if c in pm]
        frames.append(pm[keep])
    h = pd.concat(frames, ignore_index=True).merge(mp, on="fm_player_id")
    h = h.rename(columns={v: k for k, v in SIGNALS.items()} | {"Minutes played": "mins"})
    h["mins"] = pd.to_numeric(h["mins"], errors="coerce").fillna(0)
    h = h[h["mins"] > 0].copy()
    for k in SIGNALS:
        h[k] = pd.to_numeric(h.get(k), errors="coerce").fillna(0)  # FotMob leaves xG / shots out when there were none
    h["pxg"] = (h["xg"] - h["npxg"]).clip(lower=0)
    h["date"] = pd.to_datetime(h["utc"], utc=True, format="ISO8601").dt.tz_localize(None).dt.normalize()
    h["level"] = h["league"].map({"Championship": 1, "League One": 2, "League Two": 3}).fillna(2).astype(int)
    return h.sort_values(["player_id", "date"]).reset_index(drop=True)


PRIOR_BEFORE = pd.Timestamp("2026-07-01")  # league-average shrinkage targets from completed seasons only (no leakage)


def league_priors(h):
    h = h[h["date"] < PRIOR_BEFORE]
    m90 = h["mins"].sum() / 90
    return {k: float(h[k].sum() / m90) for k in RATES}


def recency_sum(h, col, n, half_life):
    """Per player, sum of col over his last n rows (h sorted by player, date), the game a rows back weighted
    0.5 ** (a / half_life)."""
    ker, out, v = 0.5 ** (np.arange(n) / half_life), np.zeros(len(h)), h[col].to_numpy(float)
    for _, idx in h.groupby("player_id").indices.items():
        out[idx] = np.convolve(v[idx], ker)[:len(idx)]
    return out


def signal_rates(rows, k=5.0, n=N_GAMES, hist=None, priors=None, prefix="fm_", step_mult=None, half_life=None):
    """rows: DataFrame with player_id and date (Timestamp, day). Returns rows + <prefix><signal> per-90 columns + <prefix>n90.
    step_mult {signal: {1: up, -1: down}}: games played in a different league from the row's own league ('level' column
    of rows: 1 Championship .. 3 League Two) count x up**steps (old game lower) or x down**steps (old game higher).
    half_life: within the last n, the game a appearances back counts 0.5 ** (a / half_life) (exp_recency.py); None = equal."""
    h = fotmob_history() if hist is None else hist
    priors = priors or league_priors(h)
    assert half_life is None or step_mult is None, "recency weights and league step multipliers are not combined"
    if half_life is None:
        sums = group_rolling_sum(h, ["mins"] + list(priors), n)
        total = lambda c: sums[c]
    else:
        total = lambda c: recency_sum(h, c, n, half_life)
    roll = pd.DataFrame({"player_id": h["player_id"], "date": h["date"], "m90": total("mins") / 90})
    for s in priors:
        if step_mult is None or s not in step_mult:
            roll[s] = total(s)
            continue
        for lvl in (1, 2, 3):  # a version of the rolling sum for each league the row's player is now in
            step = h["level"] - lvl
            w = np.where(step > 0, step_mult[s][1] ** step.clip(lower=0), np.where(step < 0, step_mult[s][-1] ** (-step).clip(lower=0), 1.0))
            roll[f"{s}@{lvl}"] = (h[s] * w).groupby(h["player_id"]).transform(lambda x: x.rolling(n, min_periods=1).sum())
    # last row per player per date (doubles on one day are rare), then as-of join strictly before the row's date
    roll = roll.drop_duplicates(["player_id", "date"], keep="last").sort_values("date")
    r = rows.copy()
    r["_ord"] = np.arange(len(r))
    r["date"] = pd.to_datetime(r["date"]).dt.tz_localize(None).dt.normalize() if getattr(pd.to_datetime(r["date"]).dt, "tz", None) else pd.to_datetime(r["date"]).dt.normalize()
    r = pd.merge_asof(r.sort_values("date"), roll, on="date", by="player_id", allow_exact_matches=False, direction="backward")
    r["m90"] = r["m90"].fillna(0)
    lvl_cols = [c for c in roll if "@" in c]
    for s, p in priors.items():
        if f"{s}@1" in roll:
            lv = r["level"].fillna(2).astype(int) if "level" in r else pd.Series(2, index=r.index)
            num = np.select([lv == 1, lv == 2, lv == 3], [r[f"{s}@1"], r[f"{s}@2"], r[f"{s}@3"]], r[f"{s}@2"])
            r[f"{prefix}{s}"] = (pd.Series(num, index=r.index).fillna(0) + k * p) / (r["m90"] + k)
        else:
            r[f"{prefix}{s}"] = (r[s].fillna(0) + k * p) / (r["m90"] + k)
    drop = [c for c in list(priors) + lvl_cols if c in r]
    r = r.rename(columns={"m90": f"{prefix}n90"}).drop(columns=drop).sort_values("_ord").drop(columns="_ord")
    return r.set_index(rows.index)


# ---------- set pieces (set_pieces.py + the 'Corners' stat) ----------
SP_N = 40          # appearances looked back over
SP_PSEUDO = {"pen": 2.0, "corner": 10.0}  # pseudo-events at the typical share (shrinkage for small samples)


def setpiece_history():
    """Per player per FotMob appearance: penalties / corners / direct free kicks he took, and his team's penalties and
    corners in that match (cached for the process; a copy)."""
    return _setpiece_history(_files_signature()).copy()


@functools.lru_cache(maxsize=1)
def _setpiece_history(_signature):
    frames = []
    for f in sorted(FM.glob("*/player_matches.csv")):
        pm = pd.read_csv(f, low_memory=False)
        pm = pm[pd.to_numeric(pm["Minutes played"], errors="coerce").fillna(0) > 0]
        frames.append(pm[["match_id", "fm_player_id", "fm_team_id", "utc", "Minutes played"] + (["Corners"] if "Corners" in pm else [])])
    h = pd.concat(frames, ignore_index=True).rename(columns={"Minutes played": "mins", "Corners": "corners"})
    h["mins"] = pd.to_numeric(h["mins"], errors="coerce").fillna(0)
    h["corners"] = pd.to_numeric(h.get("corners"), errors="coerce").fillna(0)
    sp = pd.read_csv(FM / "set_pieces.csv")
    h = h.merge(sp.drop(columns=["fm_team_id"]), on=["match_id", "fm_player_id"], how="left")
    for c in ("pens", "pen_goals", "pen_xg", "fk_shots", "fk_xg"):
        h[c] = h[c].fillna(0)
    team = h.groupby(["match_id", "fm_team_id"]).agg(team_pens=("pens", "sum"), team_corners=("corners", "sum")).reset_index()
    h = h.merge(team, on=["match_id", "fm_team_id"])
    mp = pd.read_csv(FM / "player_map.csv")[["fm_player_id", "player_id"]]
    h = h.merge(mp, on="fm_player_id")
    h["date"] = pd.to_datetime(h["utc"], utc=True, format="ISO8601").dt.tz_localize(None).dt.normalize()
    return h.sort_values(["player_id", "date"]).reset_index(drop=True)


def setpiece_constants(h):
    """League constants from completed seasons only: typical share of his team's penalties / corners for a player in
    the game, penalties per team-match, penalty conversion."""
    h = h[h["date"] < PRIOR_BEFORE]
    team_matches = h.drop_duplicates(["match_id", "fm_team_id"])
    return dict(pen_share0=float(h["pens"].sum() / h["team_pens"].sum()), corner_share0=float(h["corners"].sum() / h["team_corners"].sum()),
                pens_per_team_match=float(team_matches["team_pens"].mean()), pen_conv=float(h["pen_goals"].sum() / h["pens"].sum()),
                fk_shots90=float(h["fk_shots"].sum() / (h["mins"].sum() / 90)), fk_xg90=float(h["fk_xg"].sum() / (h["mins"].sum() / 90)),
                corners90=float(h["corners"].sum() / (h["mins"].sum() / 90)))


def setpiece_features(rows, hist=None, const=None, n=SP_N):
    """As of each row's date (strictly before): fm_penshare, fm_penrate (expected penalty goals per 90 if he plays the
    game: share x league penalties per team-match x conversion), fm_cornershare, fm_corners (per 90), fm_fkxg (per 90)."""
    h = setpiece_history() if hist is None else hist
    c = const or setpiece_constants(h)
    cols = ["mins", "pens", "team_pens", "corners", "team_corners", "fk_xg", "fk_shots"]
    roll = pd.concat([h[["player_id", "date"]], group_rolling_sum(h, cols, n).add_prefix("s_")], axis=1)
    roll = roll.drop_duplicates(["player_id", "date"], keep="last").sort_values("date")
    r = rows.copy()
    r["_ord"] = np.arange(len(r))
    r["date"] = pd.to_datetime(r["date"]).dt.normalize()
    r = pd.merge_asof(r.sort_values("date"), roll, on="date", by="player_id", allow_exact_matches=False, direction="backward")
    for col in [x for x in roll if x.startswith("s_")]:
        r[col] = r[col].fillna(0)
    pp, cp = SP_PSEUDO["pen"], SP_PSEUDO["corner"]
    r["fm_penshare"] = (r["s_pens"] + pp * c["pen_share0"]) / (r["s_team_pens"] + pp)
    r["fm_penrate"] = r["fm_penshare"] * c["pens_per_team_match"] * c["pen_conv"]
    r["fm_cornershare"] = (r["s_corners"] + cp * c["corner_share0"]) / (r["s_team_corners"] + cp)
    r["fm_corners"] = (r["s_corners"] + 3 * c["corners90"]) / (r["s_mins"] / 90 + 3)
    r["fm_fkxg"] = (r["s_fk_xg"] + 3 * c["fk_xg90"]) / (r["s_mins"] / 90 + 3)
    # raw counts over the window, for the "Pens" / "Corners" / "FKs" tags on the site
    r = r.rename(columns={"s_pens": "sp_pens", "s_team_pens": "sp_team_pens", "s_corners": "sp_corners",
                          "s_team_corners": "sp_team_corners", "s_fk_shots": "sp_fk_shots"})
    r = r.drop(columns=[x for x in roll if x.startswith("s_") and x in r]).sort_values("_ord").drop(columns="_ord")
    return r.set_index(rows.index)


def latest_team_and_pen_takers(h, before):
    """({player_id: his latest FotMob team id}, {team id: player_id who took that team's most recent penalty}) using
    matches before `before`. Takers often change after a miss, so the most recent taker is the best guess at the
    current one (our history only covers EFL games, e.g. not a player's Premier League seasons)."""
    h = h[h["date"] < pd.Timestamp(before)]
    team = h.sort_values("date").groupby("player_id")["fm_team_id"].last().to_dict()
    pens = h[h["pens"] > 0].sort_values("date")
    taker = pens.groupby("fm_team_id")["player_id"].last().to_dict()
    return team, taker


def setpiece_tags(r, recent_taker=False):
    """Tags with a reason, from the window's counts (last SP_N games): Pens (took his team's most recent penalty, or
    2+ and half or more of his team's penalties in his games), Corners (10+ and a quarter or more of his team's corners),
    FKs (3+ direct free-kick shots)."""
    tags = []
    pens, tpens = int(r.get("sp_pens", 0)), int(r.get("sp_team_pens", 0))
    if (pens >= 2 and r.get("fm_penshare", 0) >= 0.5) or (recent_taker and pens >= 1):
        why = f"took {pens} of his team's {tpens} penalties in his last {SP_N} games" + (", including the most recent" if recent_taker else "")
        tags.append(dict(t="Pens", why=why))
    corners, tcorners = int(r.get("sp_corners", 0)), int(r.get("sp_team_corners", 0))
    if corners >= 10 and r.get("fm_cornershare", 0) >= 0.25:
        tags.append(dict(t="Corners", why=f"took {corners} of his team's {tcorners} corners in his last {SP_N} games ({corners / max(tcorners, 1):.0%})"))
    fks = int(r.get("sp_fk_shots", 0))
    if fks >= 3:
        tags.append(dict(t="FKs", why=f"{fks} direct free-kick shots in his last {SP_N} games"))
    return tags


# ---------- the job a player is given (positions.py: lineup slots of every starter) ----------
JOB_STATS = ("npxg", "xa", "shots", "sot", "chances", "int", "clr", "blk", "tkl")
CUR_STARTS, HIST_APPS = 5, 40  # "now" = his last 5 starts; "then" = the starts among his last 40 appearances (the history window)


def job_of(depth, wide):
    """Lineup slot -> job: line (back / deep midfield / midfield / attacking midfield / forward) x central or wide."""
    line = np.select([depth < 0.2, depth < 0.42, depth < 0.55, depth < 0.65, depth < 0.78], ["gk", "back", "deep", "mid", "att"], "fwd")
    return np.where(line == "gk", "gk", np.char.add(line, np.where(wide >= 0.3, "-wide", "")))


@functools.lru_cache(maxsize=1)
def _position_history(_signature):
    frames = []
    for f in sorted(FM.glob("*/player_matches.csv")):
        pm = pd.read_csv(f, low_memory=False)
        keep = ["match_id", "fm_player_id", "utc", "started", "Minutes played"] + [c for c in SIGNALS.values() if c in pm]
        frames.append(pm[keep])
    h = pd.concat(frames, ignore_index=True).rename(columns={v: k for k, v in SIGNALS.items()} | {"Minutes played": "mins"})
    h["mins"] = pd.to_numeric(h["mins"], errors="coerce").fillna(0)
    h = h[h["mins"] > 0].merge(pd.read_csv(FM / "player_map.csv")[["fm_player_id", "player_id"]], on="fm_player_id")
    for k in JOB_STATS:
        h[k] = pd.to_numeric(h.get(k), errors="coerce").fillna(0)
    h = h.merge(pd.read_csv(FM / "positions.csv")[["match_id", "fm_player_id", "depth", "wide"]], on=["match_id", "fm_player_id"], how="left")
    h["date"] = pd.to_datetime(h["utc"], utc=True, format="ISO8601").dt.tz_localize(None).dt.normalize()
    h["job"] = np.where(h["depth"].notna(), job_of(h["depth"].fillna(0).to_numpy(), h["wide"].fillna(0).to_numpy()), None)
    return h.sort_values(["player_id", "date"]).reset_index(drop=True)


def position_history():
    """Every linked FotMob appearance with the job he started in (None for substitutes); cached, a copy."""
    return _position_history(_files_signature() + ((str(FM / "positions.csv"), (FM / "positions.csv").stat().st_mtime_ns),)).copy()


def job_rates(h):
    """{job: {stat: per-90 rate of starters in that job}} from completed seasons only (constants, like league_priors)."""
    s = h[(h["date"] < PRIOR_BEFORE) & h["job"].notna()]
    g = s.groupby("job")
    m90 = g["mins"].sum() / 90
    return {job: {k: float(g[k].sum()[job] / m90[job]) for k in JOB_STATS} for job in m90.index}


def position_features(rows, hist=None, rates=None):
    """As of each row's date (strictly before), per stat in JOB_STATS:
      job_<stat>     what the jobs he started his last CUR_STARTS games in typically produce per 90 (league rates)
      jobmult_<stat> that, divided by what the jobs he held over his history window (starts among his last HIST_APPS
                     appearances) produce: above 1 = he has been moved to a job that yields more of the stat than the
                     jobs his history was earned in. 1 when he has no starts on record.
    job_<stat> for a player with no starts is NaN (the caller fills the position's average)."""
    h = position_history() if hist is None else hist
    rates = rates or job_rates(h)
    for k in JOB_STATS:
        h["jr_" + k] = h["job"].map(lambda j: rates.get(j, {}).get(k, np.nan) if j is not None else np.nan)
    cols = ["jr_" + k for k in JOB_STATS]
    then = h.groupby("player_id", sort=False)[cols].rolling(HIST_APPS, min_periods=1).mean().reset_index(level=0, drop=True).reindex(h.index)
    st = h[h["job"].notna()]
    now = st.groupby("player_id", sort=False)[cols].rolling(CUR_STARTS, min_periods=1).mean().reset_index(level=0, drop=True).reindex(h.index)
    now = now.groupby(h["player_id"]).ffill()  # a substitute appearance keeps the job of his latest starts
    roll = pd.concat([h[["player_id", "date"]], now.add_prefix("now_"), then.add_prefix("then_")], axis=1)
    roll = roll.drop_duplicates(["player_id", "date"], keep="last").sort_values("date")
    r = rows.copy()
    r["_ord"] = np.arange(len(r))
    r["date"] = pd.to_datetime(r["date"]).dt.normalize()
    r = pd.merge_asof(r.sort_values("date"), roll, on="date", by="player_id", allow_exact_matches=False, direction="backward")
    for k in JOB_STATS:
        r["job_" + k] = r[f"now_jr_{k}"]
        r["jobmult_" + k] = (r[f"now_jr_{k}"] / r[f"then_jr_{k}"]).fillna(1.0)
    r = r.drop(columns=[c for c in r.columns if c.startswith("now_jr_") or c.startswith("then_jr_")]).sort_values("_ord").drop(columns="_ord")
    return r.set_index(rows.index)


JOB_LABEL = {"back": ("CB", "centre-back"), "back-wide": ("FB", "full-back"), "deep": ("DM", "deep midfield"), "deep-wide": ("WB", "wing-back"),
             "mid": ("CM", "central midfield"), "mid-wide": ("WM", "wide midfield / wing-back"), "att": ("AM", "attacking midfield"),
             "att-wide": ("W", "winger"), "fwd": ("ST", "striker"), "fwd-wide": ("ST", "striker")}


def recent_jobs(player_ids, before=None, n=CUR_STARTS):
    """{player id: dict(t=short label, why=sentence)}: the job he started most often in his last n starts before `before`
    (FotMob lineup slots), for the tag on the Player Picks pages. Players with no start on record are left out."""
    h = position_history()
    h = h[h["job"].notna() & h["player_id"].isin(set(player_ids))]
    if before is not None:
        h = h[h["date"] < pd.Timestamp(before)]
    out = {}
    for pid, g in h.sort_values("date").groupby("player_id"):
        jobs = g["job"].tail(n)
        c = jobs.map(lambda j: JOB_LABEL[j][1]).value_counts()
        top = jobs.value_counts().index[0]
        out[int(pid)] = dict(t=JOB_LABEL[top][0], why=f"Last {len(jobs)} starts (FotMob line-ups): " + ", ".join(f"{k} {v}" for k, v in c.items())
                             + ". The model uses what these jobs typically produce.")
    return out

"""Fit this season's strengths per league from the odds so far and predict an EFL Fantasy gameweek."""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from club_points import expected_points
from per_season import RHO, devig, fit_strengths, implied_lambdas, lam_for, predict_probs, score_matrix
from strengths import DIVS

warnings.filterwarnings("ignore")
HALF_LIFE, RIDGE = 15, 0.1  # weekly-refit backtest over 2024/25 + 2025/26: 60d -> 2.66pp from market, 15d -> 2.24pp (see README)
CURRENT = "2627"
OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
# football-data team names that differ from EFL squad names
ALIASES = {"Bristol Rvs": "Bristol Rovers", "Milton Keynes Dons": "MK Dons", "Sheffield Weds": "Sheffield Wednesday",
           "West Brom": "West Bromwich Albion", "West Ham": "West Ham United", "QPR": "Queens Park Rangers",
           "Wolves": "Wolverhampton Wanderers", "Peterboro": "Peterborough United"}


def resolver(squads):
    by_name = {s["name"].lower(): s["id"] for s in squads.values()}
    def f(n):
        k = ALIASES.get(n, n).lower()
        return by_name.get(k) or next((i for nm, i in by_name.items() if nm.startswith(k)), None)
    return f


def upcoming_lambdas(fixtures_csv, div):
    """Expected goals implied by the *current* odds of not-yet-played fixtures (same inversion as for played games)."""
    df = pd.read_csv(fixtures_csv, encoding="utf-8-sig")
    df = df[df["Div"] == div].copy()
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.join(implied_lambdas(df, RHO, prefix="Avg"))
    return df.dropna(subset=["lamH"])[["Date", "HomeTeam", "AwayTeam", "lamH", "lamA"]]


def fit_current(as_of, squads, fixtures_csv=None, half_life=HALF_LIFE):
    """One model per league, fitted on this season's odds-implied expected goals (recency-weighted).

    Played games contribute their closing odds; if fixtures_csv is given, upcoming fixtures that have odds
    contribute their current odds too, so the ratings reflect the market's latest view (lineups, injuries).
    """
    resolve = resolver(squads)
    fits, id2fd, ratings = {}, {}, []
    for div in DIVS:
        d = lam_for(div, CURRENT).dropna(subset=["lamH"])
        d = d[d["Date"] < as_of][["Date", "HomeTeam", "AwayTeam", "lamH", "lamA"]]
        n_up = 0
        if fixtures_csv is not None:
            up = upcoming_lambdas(fixtures_csv, div)
            n_up = len(up)
            d = pd.concat([d, up], ignore_index=True)
        age = np.clip((as_of - d["Date"]).dt.days.to_numpy(), 0, None)
        w = 0.5 ** (age / half_life)
        fit = fit_strengths(d, d["lamH"].to_numpy(), d["lamA"].to_numpy(), ridge=RIDGE, weights=w)
        fits[div] = fit
        t = fit["teams"].copy()
        t["div"] = div
        t["matches"] = [((d["HomeTeam"] == x) | (d["AwayTeam"] == x)).sum() for x in t["team"]]
        t["efl_id"] = t["team"].map(resolve)
        t["name"] = t["efl_id"].map(lambda i: squads[i]["name"])
        for n, i in zip(t["team"], t["efl_id"]):
            id2fd[i] = (n, div)
        ratings.append(t)
        print(f"  {DIVS[div]:13s} {len(d) - n_up:3d} played + {n_up:2d} upcoming-odds matches  home x{np.exp(fit['home']):.2f}  base {np.exp(fit['base']):.2f} goals  fit R2 {fit['r2']:.2f}")
    return fits, id2fd, pd.concat(ratings, ignore_index=True)


def bookmaker_probs(fixtures_csv, squads):
    """De-vigged bookmaker probabilities for upcoming games, keyed by (home_efl_id, away_efl_id)."""
    resolve = resolver(squads)
    out = {}
    df = pd.read_csv(fixtures_csv, encoding="utf-8-sig")
    for _, r in df[df["Div"].isin(DIVS)].iterrows():
        try:
            out[(resolve(r["HomeTeam"]), resolve(r["AwayTeam"]))] = devig(r["AvgH"], r["AvgD"], r["AvgA"])
        except Exception:
            pass
    return out


def predict_gameweek(rounds, squads, gw, fits, id2fd, book):
    rnd = [r for r in rounds if r["roundNumber"] == gw and r.get("gameMode", "season") == "season"][0]
    rows = []
    for g in sorted(rnd["games"], key=lambda g: (g["date"], g["competitionId"])):
        hn, div = id2fd[g["homeId"]]
        an, _ = id2fd[g["awayId"]]
        P, lh, la = predict_probs(fits[div], hn, an, RHO)
        M = score_matrix(lh, la, RHO)
        ptsH, sH = expected_points(M, True)
        ptsA, sA = expected_points(M, False)
        b = book.get((g["homeId"], g["awayId"]))
        rows.append(dict(
            date=g["date"][:10], time=g["date"][11:16], league=DIVS[div],
            home_id=g["homeId"], away_id=g["awayId"], home=squads[g["homeId"]]["name"], away=squads[g["awayId"]]["name"],
            pH=P[0], pD=P[1], pA=P[2], pO25=P[3], xgH=lh, xgA=la,
            oddsH=1 / P[0], oddsD=1 / P[1], oddsA=1 / P[2],
            csH=sH["clean_sheet"], csA=sA["clean_sheet"], g2H=sH["two_goals"], g2A=sA["two_goals"], g4H=sH["four_goals"], g4A=sA["four_goals"],
            ptsH=ptsH, ptsA=ptsA,
            source="market" if b is not None else "model",
            bookH=None if b is None else b[0], bookD=None if b is None else b[1], bookA=None if b is None else b[2]))
    return pd.DataFrame(rows)


def club_table(fx):
    """One row per club-fixture, then summed per club for the gameweek."""
    h = fx.assign(club=fx["home"], club_id=fx["home_id"], opp=fx["away"], ha="H", pts=fx["ptsH"], win=fx["pH"], cs=fx["csH"], xg_for=fx["xgH"], xg_ag=fx["xgA"])
    a = fx.assign(club=fx["away"], club_id=fx["away_id"], opp=fx["home"], ha="A", pts=fx["ptsA"], win=fx["pA"], cs=fx["csA"], xg_for=fx["xgA"], xg_ag=fx["xgH"])
    cols = ["club", "club_id", "league", "date", "opp", "ha", "pts", "win", "cs", "xg_for", "xg_ag"]
    games = pd.concat([h[cols], a[cols]], ignore_index=True).sort_values(["club", "date"])
    per = games.groupby(["club", "league"]).agg(games=("opp", "count"), exp_pts=("pts", "sum"), exp_cs=("cs", "sum"), exp_wins=("win", "sum"),
                                              xg_for=("xg_for", "sum")).reset_index()
    per["fixtures"] = per["club"].map(games.groupby("club").apply(
        lambda x: " | ".join(f"{o}({h}) {p:.1f}" for o, h, p in zip(x["opp"], x["ha"], x["pts"]))))
    return games, per.sort_values("exp_pts", ascending=False).reset_index(drop=True)

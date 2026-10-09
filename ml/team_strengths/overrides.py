"""On-demand odds overrides for the weekly run (e.g. Betfair Exchange prices copied from the website).

Workflow when Jack pastes prices:
  1. save the pasted page text to overrides/<name>_paste.txt
  2. python overrides.py import overrides/<name>_paste.txt overrides/<name>.csv      (parses + checks team names)
  3. python run_gameweek.py [gw] --override overrides/<name>.csv                      (swaps them in for that run)

An override replaces the 1X2 odds of the matching football-data fixtures with the mid-price (average of back and lay) of
each outcome; the margin is removed later like for any other odds. Over/under odds still come from football-data.
Nothing is overridden unless --override is passed.
"""
import re
import sys
from pathlib import Path

import pandas as pd

from predict_gw import ALIASES

HERE = Path(__file__).parent
SOURCE = "Betfair Exchange"
COLUMNS = ["home", "away", "h_back", "h_lay", "d_back", "d_lay", "a_back", "a_lay"]
# Betfair short names that need help (others match by unique prefix, e.g. "Charlton" -> "Charlton Athletic")
BETFAIR_ALIASES = {**ALIASES, "Sheff Utd": "Sheffield United", "Sheff Wed": "Sheffield Wednesday", "Notts Co": "Notts County",
                   "Bristol C": "Bristol City", "Bristol R": "Bristol Rovers"}


def resolve_strict(name, squads):
    """EFL squad id for a team name; exact (after aliases) or a unique prefix, otherwise an error naming the candidates."""
    key = BETFAIR_ALIASES.get(name, name).lower()
    by_name = {s["name"].lower(): s["id"] for s in squads.values()}
    if key in by_name:
        return by_name[key]
    hits = [(n, i) for n, i in by_name.items() if n.startswith(key)]
    if len(hits) == 1:
        return hits[0][1]
    raise ValueError(f"Cannot match team '{name}' to one EFL squad (candidates: {[n for n, _ in hits] or 'none'}). Add it to BETFAIR_ALIASES.")


def parse_betfair_paste(text):
    """Parse text copied from a Betfair Exchange 'Match Odds' list.

    Each match is: home name, away name, matched amount (£...), then back/lay for 1, X, 2 (each price followed by its
    £ size). Returns [{home, away, h_back, h_lay, d_back, d_lay, a_back, a_lay}] with Betfair's team names.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    is_price = lambda s: re.fullmatch(r"\d+(\.\d+)?", s) is not None
    is_money = lambda s: re.match(r"^[££]", s) is not None
    is_name = lambda s: not is_price(s) and not is_money(s) and re.search(r"[A-Za-z]", s) and "Coming up" not in s
    out, i = [], 0
    while i < len(lines) - 2:
        if is_name(lines[i]) and is_name(lines[i + 1]) and is_money(lines[i + 2]):
            prices, j = [], i + 3
            while len(prices) < 6 and j < len(lines):
                if is_price(lines[j]):
                    prices.append(float(lines[j]))
                elif not is_money(lines[j]):
                    break
                j += 1
            if len(prices) != 6:
                raise ValueError(f"Expected 6 prices for {lines[i]} v {lines[i + 1]}, found {len(prices)}.")
            row = dict(zip(COLUMNS, [lines[i], lines[i + 1], *prices]))
            for k in ("h", "d", "a"):
                if not row[f"{k}_back"] <= row[f"{k}_lay"]:
                    raise ValueError(f"{row['home']} v {row['away']}: back price {row[k + '_back']} is above lay {row[k + '_lay']}; check the paste.")
            out.append(row)
            i = j
        else:
            i += 1
    if not out:
        raise ValueError("No matches found in the pasted text.")
    return out


def import_paste(paste_path, out_csv, squads):
    rows = parse_betfair_paste(Path(paste_path).read_text(encoding="utf-8"))
    for r in rows:
        r["home"] = squads[resolve_strict(r["home"], squads)]["name"]
        r["away"] = squads[resolve_strict(r["away"], squads)]["name"]
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["source"] = SOURCE
    df.to_csv(out_csv, index=False)
    return df


def apply_overrides(fixtures_csv, override_csvs, squads, out_csv):
    """Write a copy of football-data's fixtures file with the override prices swapped in; returns (path, report rows)."""
    from predict_gw import resolver  # football-data names -> squad ids (same mapping the rest of the pipeline uses)

    from per_season import devig
    resolve = resolver(squads)
    df = pd.read_csv(fixtures_csv, encoding="utf-8-sig")
    df["BookSrc"] = "bookmaker average"
    ids = [(resolve(h), resolve(a)) for h, a in zip(df["HomeTeam"], df["AwayTeam"])]
    report = []
    for f in override_csvs:
        ov = pd.read_csv(f)
        for r in ov.itertuples():
            key = (resolve_strict(r.home, squads), resolve_strict(r.away, squads))
            idx = [k for k, pair in enumerate(ids) if pair == key]
            if not idx:
                report.append(dict(match=f"{r.home} v {r.away}", status="NOT IN football-data fixtures (nothing overridden)"))
                continue
            k = idx[0]
            mid = [(r.h_back + r.h_lay) / 2, (r.d_back + r.d_lay) / 2, (r.a_back + r.a_lay) / 2]
            before = devig(df.at[k, "AvgH"], df.at[k, "AvgD"], df.at[k, "AvgA"])
            after = devig(*mid)
            df.loc[k, ["AvgH", "AvgD", "AvgA"]] = mid
            df.at[k, "BookSrc"] = getattr(r, "source", SOURCE) if isinstance(getattr(r, "source", None), str) else SOURCE
            report.append(dict(match=f"{r.home} v {r.away}", status="overridden", mid_odds=mid, before=before, after=after))
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return out_csv, report


if __name__ == "__main__":
    import json

    import fetch_data
    if len(sys.argv) == 4 and sys.argv[1] == "import":
        squads = {s["id"]: s for s in json.load(open(fetch_data.CACHE / "squads.json", encoding="utf-8"))}
        df = import_paste(sys.argv[2], sys.argv[3], squads)
        print(f"Wrote {len(df)} matches to {sys.argv[3]}")
        print(df.drop(columns="source").to_string(index=False))
    else:
        print(__doc__)

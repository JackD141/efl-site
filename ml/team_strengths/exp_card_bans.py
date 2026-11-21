"""Card-ban inference at pick time (21 Nov 2026): how often does it predict a missed game that he really missed, and how
many real bans does it catch? python exp_card_bans.py -> printed + models/exp_card_bans_results.json

Rules: minutes_model.YELLOW_BANS / RED_BAN_PICK (FA "Essential Information" 2025-26 / 2026-27; see minutes_model.py).
1. Precision, every league game of 2025/26 (GW1-35) and 2026/27 to date: walking each player's games in date order with
   only earlier games known (ban_walk), a player "owes" a match entering a game -> did he play 0 minutes in it?
   By kind of ban, season, days since the card, and the old rule (5th / 10th / 15th yellow with no cut-off) for
   comparison. Base rate: players who played 60+ in their previous game and miss the next.
2. Recall against real bans: the EFL's suspended flags in the weekly snapshots (data/2026_27/status/, from 7 Nov 2026).
   For each snapshot, the rule run on the games before that day: how many EFL-suspended players with a game to come does
   it name, and which players does it name that the EFL has not flagged (and did they miss the game)?
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import minutes_model as mm

HERE = Path(__file__).parent


def precision_rows():
    d = mm.attach_dates(mm.raw_rows()).reset_index(drop=True)
    d = d.sort_values(["player_id", "season", "date", "gameweek"], kind="stable").reset_index(drop=True)
    ban, why = mm.card_ban_events(d)
    ban = np.where(d["red_cards"].to_numpy() > 0, mm.RED_BAN_PICK, ban)
    # the old rule: every 5th yellow (no cut-off), 2 after the 10th
    y = d["yellow_cards"].to_numpy()
    ycum = d.groupby(["player_id", "season"])["yellow_cards"].cumsum().to_numpy()
    mile = ((ycum // 5) > ((ycum - y) // 5)) & (ycum >= 5)
    old = np.where(mile, np.where(ycum >= 10, 2, 1), 0)
    pid, sea, mins, dates = d["player_id"].to_numpy(), d["season"].to_numpy(), d["minutes_played"].to_numpy(), d["date"].to_numpy()
    owe_new, owe_old = np.zeros(len(d), int), np.zeros(len(d), int)
    kind, since = np.full(len(d), "", object), np.full(len(d), np.nan)
    st = {"new": [0, None, None], "old": [0, None, None]}
    prev = None
    for i in range(len(d)):
        if (pid[i], sea[i]) != prev:
            st = {"new": [0, None, None], "old": [0, None, None]}
            prev = (pid[i], sea[i])
        for lab, arr, b in (("new", owe_new, ban), ("old", owe_old, old)):
            s = st[lab]
            arr[i] = s[0]
            if lab == "new" and s[0] > 0:
                kind[i], since[i] = s[1], (pd.Timestamp(dates[i]) - s[2]).days
            if s[0] > 0:
                s[0] = s[0] - 1 if mins[i] == 0 else 0
            if b[i] > 0:
                s[0] += int(b[i])
                s[1], s[2] = (why[i] if lab == "new" else "old"), pd.Timestamp(dates[i])
    d["owe_new"], d["owe_old"], d["kind"], d["since"] = owe_new, owe_old, kind, since
    d["missed"] = d["minutes_played"] == 0
    d["prev_min"] = d.groupby(["player_id", "season"])["minutes_played"].shift(1)
    return d


def precision(d):
    out = {}
    base = d[d["prev_min"] >= 60]
    out["base_rate_60_last_game_misses_next"] = dict(n=int(len(base)), missed=round(float(base["missed"].mean()), 3))
    f = d[d["owe_new"] > 0]
    out["rule"] = dict(n=int(len(f)), missed=int(f["missed"].sum()), precision=round(float(f["missed"].mean()), 3))
    out["rule_by_kind"] = {k: dict(n=int(len(g)), precision=round(float(g["missed"].mean()), 3)) for k, g in f.groupby("kind")}
    out["rule_by_season"] = {s: dict(n=int(len(g)), precision=round(float(g["missed"].mean()), 3)) for s, g in f.groupby("season")}
    bins = pd.cut(f["since"], [-1, 3, 7, 13, 1000], labels=["0-3 days", "4-7", "8-13", "14+"])
    out["rule_by_days_since_card"] = {f"{k} / {b}": dict(n=int(len(g)), precision=round(float(g["missed"].mean()), 3))
                                      for (k, b), g in f.groupby([f["kind"].where(f["kind"] == "red card", "yellow"), bins], observed=True)}
    o = d[d["owe_old"] > 0]
    out["old_rule_no_cutoff"] = dict(n=int(len(o)), missed=int(o["missed"].sum()), precision=round(float(o["missed"].mean()), 3))
    return out


def snapshot_recall():
    snaps = sorted((mm.REPO / "data" / "2026_27" / "status").glob("players_*.json"))
    d = mm.attach_dates(mm.raw_rows((("2026_27", "2627"),))).reset_index(drop=True)
    res = []
    for f in snaps:
        js = json.loads(f.read_text(encoding="utf-8"))
        t = pd.Timestamp(js["savedAt"]).tz_convert("Europe/London").tz_localize(None).normalize()
        before = d[d["date"] < t].reset_index(drop=True)
        after = d[d["date"] >= t]
        _, owed, why, _ = mm.ban_walk(before, mm.RED_BAN_PICK)
        owes = {int(before.at[i, "player_id"]): (int(owed[i]), why[i][0]) for i in np.flatnonzero(owed > 0)}
        # first league game on or after the snapshot day for each player (at his club then), if played yet
        nxt = after.sort_values("date").groupby("player_id").head(1).set_index("player_id")["minutes_played"].to_dict()
        susp = {p["id"] for p in js["players"] if p["suspended"]}
        known = {p["id"] for p in js["players"]}
        caught = sorted(susp & set(owes))
        missed_by_rule = sorted(susp - set(owes))
        extra = sorted(set(owes) & known - susp)
        res.append(dict(snapshot=f.stem[8:], efl_suspended=len(susp), rule_named=len(set(owes) & known), caught=len(caught),
                        efl_only=[dict(id=i, next_minutes=nxt.get(i)) for i in missed_by_rule],
                        rule_only=[dict(id=i, kind=owes[i][1], next_minutes=nxt.get(i)) for i in extra],
                        efl_flagged_next_minutes=[nxt.get(i) for i in sorted(susp)]))
    return res


def main():
    d = precision_rows()
    out = dict(precision=precision(d), snapshots=snapshot_recall())
    p = out["precision"]
    print(f"base: 60+ last game, misses the next {p['base_rate_60_last_game_misses_next']['missed']:.1%} (n {p['base_rate_60_last_game_misses_next']['n']})")
    print(f"rule: {p['rule']['missed']} of {p['rule']['n']} predicted missed games really missed ({p['rule']['precision']:.1%}); "
          f"old rule without cut-offs {p['old_rule_no_cutoff']['missed']} / {p['old_rule_no_cutoff']['n']} ({p['old_rule_no_cutoff']['precision']:.1%})")
    for k in ("rule_by_kind", "rule_by_season", "rule_by_days_since_card"):
        print(f"  {k}: " + "; ".join(f"{a} {b['precision']:.0%} (n {b['n']})" for a, b in p[k].items()))
    for s in out["snapshots"]:
        print(f"snapshot {s['snapshot']}: EFL suspended {s['efl_suspended']}, rule names {s['rule_named']}, both {s['caught']}; "
              f"EFL only {[(x['id'], x['next_minutes']) for x in s['efl_only']]}; rule only {[(x['id'], x['kind'], x['next_minutes']) for x in s['rule_only']]}")
    (HERE / "models" / "exp_card_bans_results.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()

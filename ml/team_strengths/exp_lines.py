"""Experiment (21 Nov 2026): balance start chances by LINE, not just by club.  python exp_lines.py

Reason, written before the numbers: the club total of ten (exp_clubstart.py, live since 9 Nov) shifts every outfield
player's start logit by one common amount. But a club fields a shape: a forward competes with the forwards for a
forward's place, not with the centre-backs. Seen 21 Nov: Y. Salech (Cardiff) benched last game behind two forwards, while
team-mate J. Moylan (a fantasy MID) has been starting up front; the model gave Salech 74% when forwards in his situation
start 58% of the time. If one line is crowded and another short, the club shift moves all lines alike.

Candidates (fixed beforehand), each starting from the same raw start chances (start model fitted on 2025/26 GW1-23):
  B0  club   current: one logit shift per club fixture so the outfield chances add up to 10
  L1  pos6   per FANTASY POSITION (DEF / MID / FWD): shift each group to its target = the club's number of starters in that
             position over its previous 6 club games (shrunk with 2 pseudo-games to the league share), targets scaled
             to add up to 10
  L2  line6  per LINE actually played (back / midfield / forward from FotMob line-up slots, his last 5 starts; no start
             on record -> his fantasy position's line), targets from the club's previous 6 games' starters' slots
  L3  line12 L2 with the previous 12 games
A group that cannot reach its target (too few available players) is filled and the rest goes to the other groups (one
common shift over the uncapped players). Scored on P(start) log-loss, 2025/26 GW24-35 (two halves shown: both must
improve), paired standard error with gameweeks as clusters; 2026/27 for information only (monitor).

RESULT (21 Nov 2026): all three LOSE to the club total, in both halves and on 2026/27. Validation log-loss: B0 0.3150,
L1 0.3217 (-0.0066, se 0.0008), L2 0.3250 (-0.0100, se 0.0012), L3 0.3264 (-0.0113, se 0.0012). Post-hoc (after seeing
that), a partial move towards the line targets: x0.25 +0.0001 (se 0.0002, a tie), x0.5 -0.0010 (se 0.0004). The number
of starters per line varies week to week with the shape; forcing a noisy target moves probability the wrong way.
Rejected; the club total stays. See also exp_absence_line.py.
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit, logit

import minutes_model as mm
from exp_clubstart import FIX, OUTFIELD, ll, rows_with_probs

sys.path.insert(0, str(mm.REPO / "ml" / "fotmob"))
import features as fmf  # noqa: E402

warnings.filterwarnings("ignore")
LINE_OF_JOB = {"back": "B", "back-wide": "B", "deep-wide": "B", "deep": "M", "mid": "M", "mid-wide": "M", "att": "M",
               "att-wide": "M", "fwd": "F", "fwd-wide": "F"}
LINE_OF_POS = {"DEF": "B", "MID": "M", "FWD": "F"}
PSEUDO = 2.0


def shift_group(z, total, a):
    """Common logit shift of z so that sum a * sigmoid(z + d) = total; filled (d = +30) if it cannot be reached."""
    return mm._shift(lambda d: (a * expit(z + d)).sum(), total)


def balance(p, groups, targets, total=10):
    """p: raw start chances of one club fixture's available outfield players; groups: their group labels; targets:
    {group: target count}. Each group shifted to its target; capped groups filled, the shortfall spread by one common
    shift over the players of the uncapped groups."""
    z = logit(np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6))
    out, capped = np.empty(len(z)), np.zeros(len(z), bool)
    for g in set(groups):
        m = groups == g
        t = targets.get(g, 0.0)
        if m.sum() <= t:  # not enough players in this line: all of them start
            out[m], capped[m] = 1.0, True
        else:
            out[m] = expit(z[m] + shift_group(z[m], t, np.ones(m.sum())))
    rest = total - out.sum()
    if abs(rest) > 1e-6 and (~capped).any():
        zz = logit(np.clip(out[~capped], 1e-9, 1 - 1e-9))
        out[~capped] = expit(zz + shift_group(zz, total - out[capped].sum(), np.ones((~capped).sum())))
    return out


def balance_soft(p, groups, targets, lam, total=10):
    """Post-hoc (after L1-L3 all lost): each player's logit moves lam of the way from the club shift to his group's own
    shift, then one common shift restores the club total of ten."""
    z = logit(np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6))
    dc = shift_group(z, total, np.ones(len(z)))
    d = np.full(len(z), dc)
    for g in set(groups):
        m = groups == g
        t = targets.get(g, 0.0)
        if m.sum() > t:
            d[m] = dc + lam * (shift_group(z[m], t, np.ones(m.sum())) - dc)
    z2 = z + d
    return expit(z2 + shift_group(z2, total, np.ones(len(z2))))


def line_history():
    """Per (player, date) of each start: the line of that match (FotMob slot) and line5, the most common line of his
    last 5 starts up to and including it (joined strictly before a later match, it is "his line as of then")."""
    h = fmf.position_history()
    h = h[h["job"].notna() & (h["job"] != "gk")].copy()
    h["line"] = h["job"].map(LINE_OF_JOB)
    for L in "BMF":
        h["n" + L] = (h["line"] == L).astype(float)
    g = h.groupby("player_id", sort=False)
    for L in "BMF":
        h["r" + L] = g["n" + L].transform(lambda s: s.rolling(5, min_periods=1).sum())
    h["line5"] = h[["rB", "rM", "rF"]].idxmax(axis=1).str[1]
    return h[["player_id", "date", "line", "line5"]].drop_duplicates(["player_id", "date"], keep="last")


def main():
    d = rows_with_probs()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    lh = line_history()
    # line of the match (for the club's target history) and his line as of before it (for grouping)
    d = d.merge(lh[["player_id", "date", "line"]], on=["player_id", "date"], how="left")
    asof = pd.merge_asof(d[["player_id", "date"]].reset_index().sort_values("date"), lh[["player_id", "date", "line5"]].rename(columns={"line5": "asof_line"}).sort_values("date"),
                         on="date", by="player_id", allow_exact_matches=False).set_index("index")
    d["grp_line"] = asof["asof_line"].reindex(d.index).fillna(d["pos"].map(LINE_OF_POS))
    d["match_line"] = d["line"].fillna(d["pos"].map(LINE_OF_POS))
    d["grp_pos"] = d["pos"].map(LINE_OF_POS)

    # club target history: number of starters per group in each club fixture, then the previous k fixtures
    fx = d[d["start"] == 1].groupby(FIX + ["date"]).agg(n=("start", "size")).reset_index()
    def targets(col, k):
        c = d[d["start"] == 1].pivot_table(index=FIX + ["date"], columns=col, values="start", aggfunc="sum", fill_value=0)
        c = c.reindex(columns=list("BMF"), fill_value=0)
        c = c.div(c.sum(axis=1), axis=0) * 10  # per fixture, scaled to ten outfield starters
        c = c.reset_index().sort_values(["squad_id", "date"])
        share = c[list("BMF")].mean() / 10  # league shares (constants; tiny leakage-free enough: averages of counts)
        out = []
        for L in "BMF":
            prev = c.groupby("squad_id")[L].transform(lambda s: s.shift(1).rolling(k, min_periods=1).sum())
            n = c.groupby("squad_id")[L].transform(lambda s: s.shift(1).rolling(k, min_periods=1).count())
            out.append(((prev.fillna(0) + PSEUDO * 10 * share[L]) / (n.fillna(0) + PSEUDO)).rename(L))
        t = pd.concat([c[FIX]] + out, axis=1)
        t[list("BMF")] = t[list("BMF")].div(t[list("BMF")].sum(axis=1), axis=0) * 10
        return t.drop_duplicates(FIX)
    cand = {"L1 pos6": ("grp_pos", targets("grp_pos", 6)), "L2 line6": ("grp_line", targets("match_line", 6)),
            "L3 line12": ("grp_line", targets("match_line", 12))}

    d = d.sort_values(FIX + ["player_id"]).reset_index(drop=True)
    d["B0"] = d.groupby(FIX)["ps"].transform(lambda s: mm.shift_to(s.to_numpy(), 10))
    for name, (gcol, t) in cand.items():
        tt = t.set_index(FIX)[list("BMF")]
        vals = np.empty(len(d))
        for key, idx in d.groupby(FIX).indices.items():
            tg = tt.loc[key].to_dict() if key in tt.index else {"B": 4, "M": 4, "F": 2}
            vals[idx] = balance(d["ps"].to_numpy()[idx], d[gcol].to_numpy()[idx], tg)
        d[name] = vals
    for name, gcol, t in (("S1 pos6 x0.25", "grp_pos", cand["L1 pos6"][1]), ("S2 pos6 x0.5", "grp_pos", cand["L1 pos6"][1])):
        lam = 0.25 if "0.25" in name else 0.5
        tt = t.set_index(FIX)[list("BMF")]
        vals = np.empty(len(d))
        for key, idx in d.groupby(FIX).indices.items():
            tg = tt.loc[key].to_dict() if key in tt.index else {"B": 4, "M": 4, "F": 2}
            vals[idx] = balance_soft(d["ps"].to_numpy()[idx], d[gcol].to_numpy()[idx], tg, lam)
        d[name] = vals
    cand["S1 pos6 x0.25"] = cand["S2 pos6 x0.5"] = None

    d["n_start"] = d.groupby(FIX)["start"].transform("sum")
    ok = d["n_start"] == 10
    rowll = lambda y, p: -(y * np.log(np.clip(p, 1e-4, 1 - 1e-4)) + (1 - y) * np.log(1 - np.clip(p, 1e-4, 1 - 1e-4)))
    res = {}
    for lab, m in (("validation 2025/26 GW24-35", (d["season"] == "2526") & (d["gameweek"] > 23)),
                   ("  first half GW24-29", (d["season"] == "2526") & d["gameweek"].between(24, 29)),
                   ("  second half GW30-35", (d["season"] == "2526") & d["gameweek"].between(30, 35)),
                   ("2026/27 (monitor)", d["season"] == "2627")):
        s = d[m & ok]
        r = {"rows": int(len(s)), "raw": ll(s["start"], s["ps"])}
        base = rowll(s["start"].to_numpy(), s["B0"].to_numpy())
        for c in ["B0"] + list(cand):
            x = rowll(s["start"].to_numpy(), s[c].to_numpy())
            diff = pd.Series(base - x).groupby(s["gameweek"].to_numpy())  # gain over B0 per gameweek
            sums, ns = diff.sum(), diff.size()
            mu = sums.sum() / ns.sum()
            G = len(sums)
            se = float(np.sqrt(G / (G - 1) * ((sums - mu * ns) ** 2).sum()) / ns.sum()) if G > 1 else float("nan")
            r[c] = dict(ll=float(x.mean()), gain_vs_B0=float(mu), se=se)
        res[lab] = r
        print(f"\n{lab}: {r['rows']} rows; raw {r['raw']:.4f}")
        for c in ["B0"] + list(cand):
            print(f"   {c:10s} log-loss {r[c]['ll']:.4f}  gain vs club total {r[c]['gain_vs_B0']:+.5f} (se {r[c]['se']:.5f})")
    # the Salech group: started 3 straight, then came off the bench
    d = d.sort_values(["player_id", "date"])
    g = d.groupby("player_id")
    sit = d[(g["start"].transform(lambda s: s.shift(2).rolling(3, min_periods=3).sum()) == 3) & (g["start"].shift(1) == 0) & (g["mins"].shift(1) > 0) & ok]
    print(f"\nbenched after 3 starts (n {len(sit)}): actual {sit['start'].mean():.3f} | raw {sit['ps'].mean():.3f} | "
          + " | ".join(f"{c} {sit[c].mean():.3f}" for c in ["B0"] + list(cand)))
    for pos in OUTFIELD:
        s = d[ok & (d["season"] == "2526") & (d["gameweek"] > 23) & (d["pos"] == pos)]
        print(f"   validation {pos}: " + ", ".join(f"{c} {ll(s['start'], s[c]):.4f}" for c in ["B0"] + list(cand)))
    (mm.MODELS / "exp_lines_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

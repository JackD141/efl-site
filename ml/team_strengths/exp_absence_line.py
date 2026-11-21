"""Experiment (21 Nov 2026): when a player is OUT, his start chance goes to his own line first.  python exp_absence_line.py

Background: exp_lines.py found that balancing every line to a target count loses to the plain club total of ten (line
counts vary week to week). But one case has a clear football logic: a known absence. Cardiff, GW15: two defenders flagged
out, the club's available raw start chances add to 8.9, and the club shift lifts EVERYONE, so Y. Salech (forward,
benched last game) went from 63% to 74%. In practice a defender replaces a defender.

Candidates (fixed beforehand):
  B0  current: absent players removed, one common shift so the available players' chances add up to 10
  A1  absent players' raw start chances first go to the available players of their LINE (fantasy position): that line
      is shifted so its total equals its raw total including the absentees (capped at what the line can field); then
      one common shift over everybody to reach 10 (normally tiny)
Test bed: club fixtures with at least one inferred SUSPENSION (minutes_model.suspended: known before the match, like an
injury flag). Start model fitted on 2025/26 GW1-23; scored on every later fixture with a suspension (2025/26 GW24-35 and
2026/27, shown separately), on the available outfield players of that fixture. Rows of other fixtures are identical under
both candidates.

RESULT (21 Nov 2026): A1 LOSES. 223 fixtures with a suspension (2025/26 GW24-35 + 2026/27): log-loss B0 0.3568, A1 0.3588
(-0.0020, se 0.0011; worse in both periods). The club-wide shift already gets each position's total starts in those
fixtures almost exactly right (DEF 914 predicted vs 912 actual; FWD 547 vs 550); A1 over-fills the absentee's line
(FWD 533). A suspension is covered by moving players across fantasy positions, not only within the line. Rejected.
"""
import json
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit, logit

import minutes_model as mm
from exp_clubstart import FIX, OUTFIELD, ll

warnings.filterwarnings("ignore")


def club_shift(p, total=10):
    return mm.shift_to(p, total)


def absence_line(p, pos, p_out, pos_out, total=10):
    z = logit(np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6))
    adj = z.copy()
    for g in set(pos_out):
        m = pos == g
        if not m.any():
            continue
        target = min(p[m].sum() + p_out[pos_out == g].sum(), m.sum() - 1e-3)
        adj[m] = z[m] + mm._shift(lambda d: expit(z[m] + d).sum(), target)
    q = expit(adj)
    return expit(adj + mm._shift(lambda d: expit(adj + d).sum(), total))


def main():
    parts = []
    for pos in OUTFIELD:
        full = mm.load_rows(pos, drop_suspended=False).sort_values(["player_id", "squad_id", "date", "gameweek"]).reset_index(drop=True)
        susp = mm.suspended(full)
        keep = full[~susp].reset_index(drop=True)
        feats = mm.FEATURES + mm.FM_FEATURES
        d = mm.add_features(keep)
        art = mm.fit_start(d[(d["season"] == "2526") & (d["gameweek"] <= 23)], feats)
        d["ps"] = mm.scenario_probs(art, d)[0]
        # the suspended rows: features from his games before (live: a ban is known, and his raw chance is what he
        # would have had); add_features on the full history, take those rows
        f_all = mm.add_features(full.assign(_s=susp))
        s_rows = f_all[f_all["_s"]].copy()
        s_rows["ps"] = mm.scenario_probs(art, s_rows)[0]
        parts.append((d.assign(pos=pos, out=False), s_rows.assign(pos=pos, out=True)))
    avail = pd.concat([a for a, _ in parts], ignore_index=True)
    out = pd.concat([b for _, b in parts], ignore_index=True)
    out_by_fx = out.groupby(FIX)
    avail["n_start"] = avail.groupby(FIX)["start"].transform("sum")
    keys = set(out_by_fx.groups) & set(avail.groupby(FIX).groups)
    rows = []
    for key in keys:
        a = avail[(avail[FIX] == pd.Series(key, index=FIX)).all(axis=1)] if False else None
    g_av = avail.groupby(FIX)
    for key in keys:
        a = g_av.get_group(key)
        if a["n_start"].iloc[0] != 10:
            continue
        o = out_by_fx.get_group(key)
        p, pos = a["ps"].to_numpy(), a["pos"].to_numpy()
        b0 = club_shift(p)
        a1 = absence_line(p, pos, o["ps"].to_numpy(), o["pos"].to_numpy())
        rows.append(a.assign(B0=b0, A1=a1, out_ps=o["ps"].sum(), n_out=len(o)))
    r = pd.concat(rows, ignore_index=True)
    rowll = lambda y, q: -(y * np.log(np.clip(q, 1e-4, 1 - 1e-4)) + (1 - y) * np.log(1 - np.clip(q, 1e-4, 1 - 1e-4)))
    res = {}
    for lab, m in (("2025/26 GW24-35", (r["season"] == "2526") & (r["gameweek"] > 23)), ("2026/27", r["season"] == "2627"),
                   ("both", ((r["season"] == "2526") & (r["gameweek"] > 23)) | (r["season"] == "2627"))):
        s = r[m]
        x0, x1 = rowll(s["start"].to_numpy(), s["B0"].to_numpy()), rowll(s["start"].to_numpy(), s["A1"].to_numpy())
        gain = pd.Series(x0 - x1).groupby((s["season"] + "-" + s["gameweek"].astype(str)).to_numpy())
        sums, ns = gain.sum(), gain.size()
        mu = sums.sum() / ns.sum()
        G = len(sums)
        se = float(np.sqrt(G / (G - 1) * ((sums - mu * ns) ** 2).sum()) / ns.sum())
        fx = s.drop_duplicates(FIX)
        res[lab] = dict(fixtures=int(len(fx)), rows=int(len(s)), B0=float(x0.mean()), A1=float(x1.mean()), gain=float(mu), se=se)
        print(f"{lab}: {len(fx)} club fixtures with a suspension, {len(s)} player rows | log-loss B0 {x0.mean():.4f} A1 {x1.mean():.4f} | gain {mu:+.5f} (se {se:.5f})")
        for pos in OUTFIELD:
            t = s[s["pos"] == pos]
            print(f"     {pos}: n {len(t)}, actual starts {t['start'].sum():.0f}, B0 {t['B0'].sum():.1f}, A1 {t['A1'].sum():.1f} | log-loss B0 {ll(t['start'], t['B0']):.4f} A1 {ll(t['start'], t['A1']):.4f}")
    # does the replacement come from the same line? share of the absentee's position among the extra starts
    (mm.MODELS / "exp_absence_line_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

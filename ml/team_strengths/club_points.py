"""Expected Fantasy EFL club points for one fixture, from a score-probability matrix.

Scoring (fantasy.efl.com Game Guidelines, verified against all 574 club-games of 2026/27 GW1-8 with 0 mismatches):
  win +5, draw +3, away win +2 (on top of the win), clean sheet +2, 2+ goals scored +2, 4+ goals scored +2 (stacks with 2+).
A club with two fixtures in a gameweek scores both.
"""
import numpy as np

from per_season import MAXG

G = np.arange(MAXG + 1)
HG, AG = np.meshgrid(G, G, indexing="ij")
POINTS = dict(win=5, draw=3, away_win=2, clean_sheet=2, two_goals=2, four_goals=2)


def side_probs(P, home):
    """P[i, j] = P(home scores i, away scores j). Returns the club's outcome probabilities."""
    gf, ga = (HG, AG) if home else (AG, HG)
    return dict(win=P[gf > ga].sum(), draw=P[gf == ga].sum(), loss=P[gf < ga].sum(),
                clean_sheet=P[ga == 0].sum(), two_goals=P[gf >= 2].sum(), four_goals=P[gf >= 4].sum())


def expected_points(P, home):
    s = side_probs(P, home)
    e = (POINTS["win"] * s["win"] + POINTS["draw"] * s["draw"] + (0 if home else POINTS["away_win"] * s["win"])
         + POINTS["clean_sheet"] * s["clean_sheet"] + POINTS["two_goals"] * s["two_goals"] + POINTS["four_goals"] * s["four_goals"])
    return e, s

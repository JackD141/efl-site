"""Download OLDER FotMob seasons into data/fotmob/history/ (21 Nov 2026), kept apart from the live data.

    python fetch_history.py 2023/2024 2022/2023

Same requests and politeness as fetch_fotmob.py (one at a time, a pause between, every match cached once; stop if FotMob
refuses). Why a separate folder: the live feature code reads every data/fotmob/<season>/player_matches.csv
(features.fotmob_history, league priors, job rates), so a new season there would change this week's model inputs
without validation. history/ is only read by build_fantasy_history.py until the models are deliberately retrained
with it. FotMob has full stats (incl. xG / xA) for 2022/23; 2019/20 League Two has counts but no xG / xA.
"""
import sys

import fetch_fotmob as ff

ff.OUT = ff.OUT / "history"
ff.RAW = ff.OUT / "raw"

if __name__ == "__main__":
    ff.run(sys.argv[1:])

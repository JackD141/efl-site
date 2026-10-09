"""Attack vs defence scatter per league from ratings_<season>.csv."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NAMES = {"E1": "Championship", "E2": "League 1", "E3": "League 2"}


def plot(ratings, squads, path, season_label):
    rt = ratings.copy()
    rt["short"] = rt["efl_id"].map(lambda i: squads[i].get("shortName") or squads[i]["name"])
    fig, axes = plt.subplots(1, 3, figsize=(21, 7.4))
    for ax, (div, df) in zip(axes, rt.groupby("div")):
        ax.axhline(0, color="#bbb", lw=1)
        ax.axvline(0, color="#bbb", lw=1)
        ax.scatter(df["attack"], df["defence"], s=36, color="#1f6feb", zorder=3)
        for _, r in df.iterrows():
            ax.annotate(r["short"], (r["attack"], r["defence"]), xytext=(4, 3), textcoords="offset points", fontsize=8)
        ax.set_title(f"{NAMES[div]} ({season_label}, fitted from odds to date)", fontsize=12)
        ax.set_xlabel("Attack strength  (more goals  →)")
        ax.set_ylabel("Defence strength  (fewer goals conceded  →)")
        lim = max(df["attack"].abs().max(), df["defence"].abs().max()) * 1.18
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.grid(alpha=0.25)
    plt.suptitle("Team strengths: attack vs defence (log expected-goals scale, relative to league average)", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)

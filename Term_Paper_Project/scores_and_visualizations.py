"""
02_scores_and_visualizations.py
-------------------------------
Consume the PCA-derived weights from 01_pca_weights.py to compute the
Accessibility score (A), TOD score (T), and the four-quadrant policy
classification for every intersection. Produce the headline visualisations
referenced in the paper.

Inputs:
    DATA_PATH (xlsx)    -- the same feature table.
    WEIGHTS_PATH (json) -- output of 01_pca_weights.py.

Outputs (under OUT_DIR):
    01_weights.png             -- bar charts of PCA weights with bootstrap SD
    02_quadrant_scatter.png    -- A vs T with median split + quadrant colours
    03_score_distributions.png -- histograms of A and T
    04_spatial_map.png         -- intersections coloured by quadrant on lat/lon
    intersection_scores.csv    -- per-intersection A, T, quadrant
    quadrant_summary.csv       -- counts, means, share, policy per quadrant

Methodology (paper stages 3 & 4):
    1. Min-max normalise each feature to [0, 1].
    2. For TOD, invert (1 - x) the semantically inverse features
       (edge_length_avg, circuity_avg) so that higher T = better.
    3. A and T = weighted linear combinations using the PCA weights.
    4. Quadrant = median split on (A, T).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib import rcParams

# ---------------------------------------------------------------------------
# Elsevier journal style defaults
# ---------------------------------------------------------------------------
# Single-column figure width: 3.5 in; double-column: 7.0–7.5 in.
# Font: Times New Roman / Helvetica; base size 8–10 pt for body, 10 for axes.

ELSEVIER_RC = {
    # Font family
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset":   "stix",
    # Sizes  (Elsevier: 8-pt for axis labels/ticks, 9-pt for title/legend)
    "font.size":           9,
    "axes.titlesize":      9,
    "axes.labelsize":      9,
    "xtick.labelsize":     8,
    "ytick.labelsize":     8,
    "legend.fontsize":     8,
    "legend.title_fontsize": 8,
    # Lines / spines
    "axes.linewidth":      0.8,
    "grid.linewidth":      0.5,
    "grid.alpha":          0.35,
    "grid.color":          "#666666",
    "lines.linewidth":     1.0,
    # Ticks
    "xtick.major.width":   0.8,
    "ytick.major.width":   0.8,
    "xtick.major.size":    3.0,
    "ytick.major.size":    3.0,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    # Figure
    "figure.dpi":          300,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.02,
}
mpl.rcParams.update(ELSEVIER_RC)

# --- Configuration -----------------------------------------------------------

DATA_PATH    = "PoI_features.xlsx"
WEIGHTS_PATH = "weights.json"
OUT_DIR      = "outputs"

INVERSE_TOD_FEATURES = {"edge_length_avg", "circuity_avg"}

# ---------------------------------------------------------------------------
# High-contrast quadrant colours (dark, print-safe, distinguishable in grey)
# ---------------------------------------------------------------------------
QUADRANT_COLORS = {
    "Stress / balance":  "#1A237E",   # deep indigo    (high A, high T)
    "Unsustained place": "#B71C1C",   # deep crimson   (high A, low T)
    "Unsustained node":  "#1B5E20",   # deep forest    (low A, high T) ← priority
    "Dependence":        "#4A4A4A",   # dark charcoal  (low A, low T)
}

QUADRANT_MARKERS = {
    "Stress / balance":  "o",
    "Unsustained place": "s",
    "Unsustained node":  "^",
    "Dependence":        "D",
}

QUADRANT_POLICY = {
    "Stress / balance":  "Reinforce existing hub",
    "Unsustained place": "Optimise current services",
    "Unsustained node":  "Priority investment (close access gap)",
    "Dependence":        "Long-term overhaul",
}

# ---------------------------------------------------------------------------
# Approximate NYC borough boundaries as simplified polygons (lon, lat)
# These give a recognisable silhouette for the background shade.
# --- Helpers -----------------------------------------------------------------

def parse_location(s: str):
    s = s.strip().lstrip("(").rstrip(")")
    lat, lon = s.split(",")
    return float(lat), float(lon)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    coords = df["location"].apply(parse_location)
    df["lat"] = coords.apply(lambda t: t[0])
    df["lon"] = coords.apply(lambda t: t[1])
    return df


def minmax(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def compute_score(df: pd.DataFrame, features, weights: dict,
                  invert: set | None = None) -> pd.Series:
    """Weighted linear combination of min-max normalised features."""
    invert = invert or set()
    score = pd.Series(np.zeros(len(df)), index=df.index)
    for f in features:
        v = minmax(df[f])
        if f in invert:
            v = 1.0 - v
        score = score + weights[f] * v
    return score


def assign_quadrant(a: float, t: float, a_med: float, t_med: float) -> str:
    if a >= a_med and t >= t_med:
        return "Stress / balance"
    if a >= a_med and t <  t_med:
        return "Unsustained place"
    if a <  a_med and t >= t_med:
        return "Unsustained node"
    return "Dependence"


def _despine(ax, top=True, right=True):
    """Remove top/right spines (common in Elsevier figures)."""
    if top:   ax.spines["top"].set_visible(False)
    if right: ax.spines["right"].set_visible(False)


# --- Plotting ----------------------------------------------------------------

def plot_weights(wcfg: dict, outpath: Path):
    """Fig. 1 – PCA weight bar charts, Elsevier double-column width."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.8))
    blocks = [
        ("(a) Accessibility weights", wcfg["accessibility"], "#0D47A1"),  # deep blue
        ("(b) TOD weights",           wcfg["tod"],           "#1B5E20"),  # deep green
    ]
    for ax, (label, result, color) in zip(axes, blocks):
        feats = result["features"]
        means = [result["bootstrap"]["mean"][f] for f in feats]
        stds  = [result["bootstrap"]["std"][f]  for f in feats]
        bars = ax.barh(feats, means, xerr=stds,
                       color=color, alpha=0.88,
                       edgecolor="black", linewidth=0.6,
                       error_kw=dict(ecolor="#333333", capsize=2.5,
                                     lw=0.8, capthick=0.8))
        ax.set_xlabel("PCA weight (sum to 1)", fontsize=8)
        ax.set_title(
            f"{label}\n"
            f"(PC1: {result['pc1_explained_variance_ratio']*100:.1f}% variance explained)",
            fontsize=8, loc="left", pad=4
        )
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.30, linewidth=0.5)
        ax.set_xlim(left=0)
        _despine(ax)
    fig.suptitle(
        "PCA-derived feature weights with bootstrap standard deviation"
        r" ($n = 1{,}000$ replicates)",
        fontsize=9, y=1.01
    )
    plt.tight_layout(w_pad=2.5)
    plt.savefig(outpath)
    plt.close()
    print(f"  Saved {outpath.name}")


def plot_quadrant_scatter(df: pd.DataFrame, a_med: float, t_med: float,
                          outpath: Path):
    """Fig. 2 – A × T scatter with quadrant shading.

    Layout strategy to avoid overlap:
    • Legend placed *outside* the axes (right margin via bbox_to_anchor).
    • Quadrant labels rendered in the *margins* outside the data limits,
      anchored to the four corners of the figure using axes-fraction
      coordinates (transform=ax.transAxes), so they never collide with points.
    • Axes expanded with extra top/right padding to accommodate corner labels.
    """
    # Use a wider canvas to accommodate the outside legend
    fig, ax = plt.subplots(figsize=(5.0, 3.8))

    xlo, xhi = df["T"].min(), df["T"].max()
    ylo, yhi = df["A"].min(), df["A"].max()
    # Add breathing room so margin labels don't clip
    x_pad = (xhi - xlo) * 0.04
    y_pad = (yhi - ylo) * 0.04

    # ---- Quadrant background tints (clipped to data range) ----
    ax.axvspan(t_med, xhi + x_pad, ymin=0.5, ymax=1,
               color=QUADRANT_COLORS["Stress / balance"], alpha=0.07, linewidth=0)
    ax.axvspan(xlo - x_pad, t_med, ymin=0.5, ymax=1,
               color=QUADRANT_COLORS["Unsustained place"], alpha=0.07, linewidth=0)
    ax.axvspan(t_med, xhi + x_pad, ymin=0, ymax=0.5,
               color=QUADRANT_COLORS["Unsustained node"], alpha=0.07, linewidth=0)
    ax.axvspan(xlo - x_pad, t_med, ymin=0, ymax=0.5,
               color=QUADRANT_COLORS["Dependence"], alpha=0.07, linewidth=0)

    # ---- Scatter points ----
    for q, color in QUADRANT_COLORS.items():
        sub = df[df["quadrant"] == q]
        ax.scatter(sub["T"], sub["A"],
                   c=color,
                   marker=QUADRANT_MARKERS[q],
                   s=4, alpha=0.70,
                   edgecolors="none",
                   label=f"{q}  ($n={len(sub):,}$)",
                   linewidths=0, rasterized=True)

    # ---- Median split lines ----
    ax.axvline(t_med, color="#222222", linestyle="--", linewidth=0.9, alpha=0.75,
               zorder=5)
    ax.axhline(a_med, color="#222222", linestyle="--", linewidth=0.9, alpha=0.75,
               zorder=5)

    # ---- Quadrant corner labels (axes-fraction coords → never inside data) ----
    # Placed just inside each corner with a semi-transparent white box.
    # corner_labels = [
    #     # (axes x-frac, axes y-frac, text,               ha,      va)
    #     (0.98, 0.98, "Stress /\nbalance",     "right",  "top"),
    #     (0.02, 0.98, "Unsustained\nplace",    "left",   "top"),
    #     (0.98, 0.02, "Unsustained\nnode",     "right",  "bottom"),
    #     (0.02, 0.02, "Dependence",            "left",   "bottom"),
    # ]
    # label_colors = [
    #     QUADRANT_COLORS["Stress / balance"],
    #     QUADRANT_COLORS["Unsustained place"],
    #     QUADRANT_COLORS["Unsustained node"],
    #     QUADRANT_COLORS["Dependence"],
    # ]
    # for (xf, yf, txt, ha, va), fc in zip(corner_labels, label_colors): 
    #     ax.text(xf, yf, txt,
    #             transform=ax.transAxes,
    #             fontsize=7, ha=ha, va=va,
    #             fontstyle="italic", fontweight="bold",
    #             color=fc,
    #             bbox=dict(boxstyle="round,pad=0.25",
    #                       fc="white", ec=fc,
    #                       linewidth=0.6, alpha=0.88),
    #             zorder=10)

    # ---- Median value annotations on the dashed lines ----
    ax.text(t_med, yhi + y_pad * 0.5,
            f"$T_{{med}}={t_med:.3f}$",
            fontsize=6.5, ha="center", va="bottom", color="#333333")
    ax.text(xlo - x_pad * 1, a_med,
            f"$A_{{med}}={a_med:.3f}$",
            fontsize=6.5, ha="right", va="center", color="#333333"
            ) #rotation=90

    ax.set_xlim(xlo - x_pad, xhi + x_pad)
    ax.set_ylim(ylo - y_pad, yhi + y_pad * 3)   # extra top space for T_med label

    ax.set_xlabel("TOD score ($T$)")
    ax.set_ylabel("Accessibility score ($A$)")
    ax.set_title("Policy quadrants — median split ($A \\times T$)",
                 fontsize=9, loc="left")

    # ---- Legend outside the axes (right side) ----
    leg = ax.legend(
        loc="upper left",
        #bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        framealpha=0.95,
        markerscale=2.2,
        handletextpad=0.5,
        borderpad=0.6,
        labelspacing=0.5,
        edgecolor="#aaaaaa",
        fontsize=7.5,
    )
    leg.get_frame().set_linewidth(0.5)

    ax.grid(alpha=0.22, linewidth=0.5)
    _despine(ax)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"  Saved {outpath.name}")


def plot_score_distributions(df: pd.DataFrame, a_med: float, t_med: float,
                             outpath: Path):
    """Fig. 3 – Histograms of A and T scores."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))

    panels = [
        (axes[0], df["A"], a_med, "#0D47A1",  # deep blue
         "Accessibility score ($A$)", "(a) Distribution of accessibility scores"),
        (axes[1], df["T"], t_med, "#1B5E20",  # deep green
         "TOD score ($T$)",           "(b) Distribution of TOD scores"),
    ]
    for ax, data, med, color, xlabel, title in panels:
        ax.hist(data, bins=50, color=color, alpha=0.85,
                edgecolor="black", linewidth=0.25, rasterized=True)
        ax.axvline(med, color="#CC0000", linestyle="--", linewidth=1.0,
                   label=f"Median = {med:.3f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count (intersections)")
        ax.set_title(title, fontsize=9, loc="left")
        leg = ax.legend(framealpha=0.9, edgecolor="#bbbbbb", fontsize=7)
        leg.get_frame().set_linewidth(0.5)
        ax.grid(axis="y", alpha=0.30, linewidth=0.5)
        _despine(ax)

    plt.tight_layout(w_pad=2.5)
    plt.savefig(outpath)
    plt.close()
    print(f"  Saved {outpath.name}")


def plot_spatial_map(df: pd.DataFrame, outpath: Path):
    """Fig. 4 – Clean spatial map, no borough boundaries.

    • No background polygons — plain white canvas keeps focus on points.
    • s=2, alpha=0.60 so dense clusters show density without blobs.
    • Dependence (charcoal) drawn first; vivid colours render on top.
    • Legend inside upper-right white-space with opaque frame.
    • Figure sized at 5.5 × 5.5 in (Elsevier full-column).
    """
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # Draw least-important quadrant first so priority colours are on top
    draw_order = ["Dependence", "Unsustained place", "Unsustained node", "Stress / balance"]
    for q in draw_order:
        color = QUADRANT_COLORS[q]
        sub   = df[df["quadrant"] == q]
        ax.scatter(
            sub["lon"], sub["lat"],
            c=color,
            marker=QUADRANT_MARKERS[q],
            s= 1.5, #2,
            alpha=0.60,
            edgecolors="none",
            linewidths=0.0,
            zorder=2,
            rasterized=True,
            label=f"{q}  ($n={len(sub):,}$)",
        )

    # Legend – upper right, compact
    leg = ax.legend(
        loc="upper left",
        framealpha=0.97,
        edgecolor="#555555",
        fontsize=7.5,
        markerscale=4.0,
        handletextpad=0.4,
        borderpad=0.65,
        labelspacing=0.45,
        title="Quadrant",
        title_fontsize=8,
    )
    leg.get_frame().set_linewidth(0.6)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("NYC intersections by node–place quadrant",
                 fontsize=9, loc="left", pad=5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.20, linewidth=0.4)
    _despine(ax)

    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"  Saved {outpath.name}")


# --- Main --------------------------------------------------------------------

def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {DATA_PATH} and {WEIGHTS_PATH} ...")
    df = load_data(DATA_PATH)
    with open(WEIGHTS_PATH) as f:
        wcfg = json.load(f)

    access_features = wcfg["config"]["accessibility_features"]
    tod_features    = wcfg["config"]["tod_features"]
    access_weights  = wcfg["accessibility"]["weights"]
    tod_weights     = wcfg["tod"]["weights"]

    print("Computing scores ...")
    df["A"] = compute_score(df, access_features, access_weights)
    df["T"] = compute_score(df, tod_features,    tod_weights, INVERSE_TOD_FEATURES)

    a_med, t_med = df["A"].median(), df["T"].median()
    df["quadrant"] = df.apply(
        lambda r: assign_quadrant(r["A"], r["T"], a_med, t_med), axis=1
    )

    print(f"  Medians:  A = {a_med:.4f}    T = {t_med:.4f}")
    print(f"  Quadrant counts:")
    for q, n in df["quadrant"].value_counts().items():
        print(f"    {q:<22}  n = {n:>5}  ({100*n/len(df):.1f}%)")

    print("\nGenerating figures ...")
    plot_weights(wcfg,            out_dir / "01_weights.png")
    plot_quadrant_scatter(df, a_med, t_med, out_dir / "02_quadrant_scatter.png")
    plot_score_distributions(df, a_med, t_med, out_dir / "03_score_distributions.png")
    plot_spatial_map(df,          out_dir / "04_spatial_map.png")

    # Per-intersection scored output
    df[["lat", "lon", "A", "T", "quadrant"]].to_csv(
        out_dir / "intersection_scores.csv", index=False
    )

    # Quadrant summary table
    summary = (df.groupby("quadrant")
                 .agg(n=("A", "size"),
                      A_mean=("A", "mean"),
                      T_mean=("T", "mean"))
                 .round(4))
    summary["share_pct"] = (summary["n"] / len(df) * 100).round(2)
    summary["policy"] = summary.index.map(QUADRANT_POLICY)
    summary.to_csv(out_dir / "quadrant_summary.csv")

    print(f"\nAll outputs written to {out_dir.resolve()}/")


if __name__ == "__main__":
    main()

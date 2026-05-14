"""
01_pca_weights.py
-----------------
Derive PCA-based weights for the Accessibility and TOD scores from a single
feature table, with bootstrap resampling to assess weight stability.

Inputs:
    DATA_PATH (xlsx) -- one row per intersection. Required columns:
        location (str like "(lat, lon)"), and the feature columns named below.

Outputs:
    weights.json -- weights, PC1 loadings, explained variance, and bootstrap
                    statistics for both the accessibility and TOD blocks.
                    Consumed by 02_scores_and_visualizations.py.

Methodology (paper stage 2):
    1. Z-score standardise features within each block.
    2. Run PCA, take the first principal component.
    3. Sign-correct PC1 against a theoretical anchor: hospital must load
       positively for accessibility; k_avg must load positively for TOD.
    4. Convert loadings to weights: take absolute values, normalise to sum
       to 1.
    5. Bootstrap resample n_boot=1000 times to characterise weight stability.
"""

import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- Configuration -----------------------------------------------------------

DATA_PATH = "PoI_features.xlsx"
OUT_PATH  = "weights.json"

N_BOOTSTRAP   = 1000
RANDOM_SEED   = 42

ACCESSIBILITY_FEATURES = ["hospital", "college", "school", "station", "park", "cinema"]
TOD_FEATURES = ["k_avg", "streets_per_node_avg", "edge_length_avg",
                "circuity_avg", "residential", "commercial"]

# Sign-correction anchors. Whichever feature is named here must end up with a
# POSITIVE loading on PC1; if PCA returns it negative, the entire eigenvector
# is flipped. This pins the score's direction to a theoretically meaningful
# axis instead of an arbitrary algebraic sign.
ACCESS_ANCHOR = "station" #"hospital"
TOD_ANCHOR    = "k_avg"


# --- Helpers -----------------------------------------------------------------

def parse_location(s: str):
    """Parse '(lat, lon)' string into (lat, lon) floats."""
    s = s.strip().lstrip("(").rstrip(")")
    lat, lon = s.split(",")
    return float(lat), float(lon)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    coords = df["location"].apply(parse_location)
    df["lat"] = coords.apply(lambda t: t[0])
    df["lon"] = coords.apply(lambda t: t[1])
    return df


def pca_first_component_weights(X: np.ndarray, anchor_idx: int):
    """Run PCA(n=1), sign-correct against anchor, return (weights, loadings, var_ratio)."""
    pca = PCA(n_components=1)
    pca.fit(X)
    loadings = pca.components_[0].copy()  # shape (n_features,)

    if loadings[anchor_idx] < 0:
        loadings = -loadings

    weights = np.abs(loadings)
    weights = weights / weights.sum()
    return weights, loadings, float(pca.explained_variance_ratio_[0])


def derive_weights_with_bootstrap(df: pd.DataFrame, features, anchor: str,
                                  n_boot: int = 1000, seed: int = 42) -> dict:
    """Compute PCA weights on the full sample, then bootstrap for stability."""
    X = df[features].values.astype(float)
    X_std = StandardScaler().fit_transform(X)

    anchor_idx = features.index(anchor)

    # Point estimate from full sample
    weights, loadings, pc1_var = pca_first_component_weights(X_std, anchor_idx)

    # Bootstrap
    rng = np.random.default_rng(seed)
    n = len(X_std)
    boot = np.full((n_boot, len(features)), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            wb, _, _ = pca_first_component_weights(X_std[idx], anchor_idx)
            boot[b] = wb
        except Exception:
            pass  # leave NaN

    valid = ~np.isnan(boot).any(axis=1)
    boot = boot[valid]

    return {
        "features": features,
        "weights":  dict(zip(features, weights.tolist())),
        "loadings": dict(zip(features, loadings.tolist())),
        "pc1_explained_variance_ratio": pc1_var,
        "bootstrap": {
            "mean":    dict(zip(features, boot.mean(axis=0).tolist())),
            "std":     dict(zip(features, boot.std(axis=0).tolist())),
            "ci_low":  dict(zip(features, np.percentile(boot, 2.5,  axis=0).tolist())),
            "ci_high": dict(zip(features, np.percentile(boot, 97.5, axis=0).tolist())),
            "n_iterations": int(valid.sum()),
        },
    }


# --- Main --------------------------------------------------------------------

def main():
    print(f"Loading {DATA_PATH} ...")
    df = load_data(DATA_PATH)
    print(f"  -> {len(df):,} intersections")

    print("\n[1/2] PCA on accessibility features")
    access = derive_weights_with_bootstrap(
        df, ACCESSIBILITY_FEATURES, ACCESS_ANCHOR,
        n_boot=N_BOOTSTRAP, seed=RANDOM_SEED,
    )
    print(f"  PC1 explains {access['pc1_explained_variance_ratio']*100:5.1f}% of variance")
    for f in ACCESSIBILITY_FEATURES:
        w  = access["weights"][f]
        sd = access["bootstrap"]["std"][f]
        print(f"    {f:>22}  w = {w:.4f}   boot SD = {sd:.4f}")

    print("\n[2/2] PCA on TOD features")
    tod = derive_weights_with_bootstrap(
        df, TOD_FEATURES, TOD_ANCHOR,
        n_boot=N_BOOTSTRAP, seed=RANDOM_SEED,
    )
    print(f"  PC1 explains {tod['pc1_explained_variance_ratio']*100:5.1f}% of variance")
    for f in TOD_FEATURES:
        w  = tod["weights"][f]
        sd = tod["bootstrap"]["std"][f]
        print(f"    {f:>22}  w = {w:.4f}   boot SD = {sd:.4f}")

    out = {
        "accessibility": access,
        "tod": tod,
        "config": {
            "accessibility_features": ACCESSIBILITY_FEATURES,
            "tod_features":           TOD_FEATURES,
            "tod_inverse_features":   ["edge_length_avg", "circuity_avg"],
            "access_anchor": ACCESS_ANCHOR,
            "tod_anchor":    TOD_ANCHOR,
            "n_bootstrap":   N_BOOTSTRAP,
            "random_seed":   RANDOM_SEED,
        },
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()

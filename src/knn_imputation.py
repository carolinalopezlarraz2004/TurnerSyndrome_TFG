"""
knn_imputation.py

Stratified KNN imputation for the anthropometric tables, kept fully separate
from the main preprocessing pipeline (it does NOT modify main.py or the
*_preprocessed.csv files).

Idea
----
Each subject is a vector of anthropometric measurements. To fill a missing
value we look at the most similar subjects (nearest neighbours in that vector
space) and take the median of their value for the missing measurement.

Stratification
--------------
Neighbours are searched ONLY within the same site (Colombia / Barcelona /
Brasil are imputed separately) AND within the same population (TYPE = ST for
Turner, CN for control). A control never donates a value to a Turner subject
and vice versa.

Distance
--------
- Every measurement is z-scored within its (site, TYPE) group first, so large
  perimeters do not dominate small distances.
- Distances are NaN-aware (nan_euclidean): they are computed only over the
  measurements that BOTH subjects have, and rescaled by the number of shared
  measurements. This means a subject with a few gaps can still be placed in the
  space without needing to be imputed first.

Two feature modes are compared:
    - "all"  : distances use every numeric measurement.
    - "corr" : redundant measurements (|r| > CORR_THRESHOLD with an already
               kept one) are dropped before computing distances.

Validation
----------
run_validation_experiment() hides one known value at a time (leave-one-value-out),
imputes it, and measures the absolute error. It sweeps several values of k and
both feature modes, so you can pick k from a table instead of guessing.

run_validation_detail() does the same leave-one-value-out test but for ONE chosen
k and feature mode, and records every single attempt (subject, column, true value,
predicted value, error, and which neighbours were used) so the error is fully
auditable.

Output
------
build_complete_tables(k, feature_mode) writes, next to each preprocessed file, a
"*_complete.csv" with the gaps filled, plus a boolean "<col>_was_imputed" flag
per measurement. The original preprocessed file (with real NaN) is never touched.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import nan_euclidean_distances


# ============================================================
# CONFIGURATION
# ============================================================

# k values to sweep in the validation experiment.
K_VALUES = [1, 3, 5, 7, 9]

# Feature modes to compare for the distance computation.
FEATURE_MODES = ["all", "corr"]

# Final choice made from the validation results, used to build the complete
# tables and the detailed validation log. Change here if you change your mind.
CHOSEN_K = 3
CHOSEN_FEATURE_MODE = "all"

# Correlation threshold for the "corr" feature mode: if two measurements are
# correlated above this (in absolute value), one of them is dropped from the
# distance computation.
CORR_THRESHOLD = 0.90

# Columns that are never used at all (metadata / QC bookkeeping). AGE is here
# on purpose: it is metadata, not an anthropometric measurement, so it is never
# imputed, never evaluated, and never used to measure similarity.
NON_FEATURE_COLUMNS = {
    "ID",
    "PSEUDO_ID",
    "TYPE",
    "AGE",
    "QC_FLAGS",
    "QC_COLUMNS",
    "QC_N_FLAGS",
    "IMPUTED_COLUMNS",
    "N_IMPUTED_VALUES",
}

# Columns used ONLY to measure similarity between subjects (they help find
# neighbours of a similar body size) but that are NEVER imputed and NEVER part
# of the validation error. Only the segment anthropometric measurements are
# imputed and evaluated.
DISTANCE_ONLY_COLUMNS = {
    "HEIGHT_cm",
    "WEIGHT_kg",
    "PONDERAL_INDEX_kg_m3",
}

# Population codes used for stratification.
POPULATION_COLUMN = "TYPE"


# ============================================================
# HELPERS
# ============================================================

def get_distance_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the numeric measurements used to compute the distance between
    subjects: every numeric measurement except metadata (AGE is excluded via
    NON_FEATURE_COLUMNS). This includes HEIGHT, WEIGHT and PONDERAL_INDEX plus
    all segment measurements.
    """
    cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    numeric = []
    for c in cols:
        if pd.to_numeric(df[c], errors="coerce").notna().any():
            numeric.append(c)
    return numeric


def get_target_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the measurements that are actually imputed and evaluated: the segment
    anthropometric measurements only (distance-only columns such as HEIGHT,
    WEIGHT and PONDERAL_INDEX are excluded).
    """
    return [
        c for c in get_distance_columns(df)
        if c not in DISTANCE_ONLY_COLUMNS
    ]


def get_id_column(df: pd.DataFrame) -> str | None:
    """Return the best available subject identifier column."""
    if "ID" in df.columns:
        return "ID"
    if "PSEUDO_ID" in df.columns:
        return "PSEUDO_ID"
    return None


def standardize(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Z-score the given columns using this frame's own mean/std (NaN ignored).
    Returns the standardized frame plus the means and stds used.
    """
    numeric = df[cols].apply(pd.to_numeric, errors="coerce")
    means = numeric.mean()
    stds = numeric.std(ddof=0).replace(0, np.nan)  # avoid /0 for constant cols
    z = (numeric - means) / stds
    return z, means, stds


def select_features_by_correlation(
    df: pd.DataFrame,
    cols: list[str],
    threshold: float = CORR_THRESHOLD,
) -> list[str]:
    """
    Greedy redundancy filter: walk the columns in order and drop any column
    that is correlated above `threshold` (absolute) with a column already kept.
    """
    numeric = df[cols].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr().abs()

    kept: list[str] = []
    for c in cols:
        redundant = any(
            (c != k) and pd.notna(corr.loc[c, k]) and corr.loc[c, k] > threshold
            for k in kept
        )
        if not redundant:
            kept.append(c)
    return kept


# ============================================================
# CORE KNN IMPUTATION (for one site+population group)
# ============================================================

def knn_impute_group(
    group: pd.DataFrame,
    feature_cols: list[str],
    distance_cols: list[str],
    k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Impute the missing values of one (site, TYPE) group.

    Input:
        group: rows of a single site and single TYPE.
        feature_cols: measurements that will be imputed if missing.
        distance_cols: measurements used to compute the distance between
                       subjects (may be a reduced set).
        k: number of nearest neighbours to average (median).

    Output:
        (imputed_group, was_imputed_mask)
            imputed_group: copy of `group` with gaps filled where possible.
            was_imputed_mask: boolean DataFrame (same index/feature_cols) marking
                              which cells were filled.
    """
    g = group.copy()
    values = g[feature_cols].apply(pd.to_numeric, errors="coerce")

    # Standardized matrix used ONLY to measure distances.
    z, _, _ = standardize(g, distance_cols)
    z_matrix = z.to_numpy(dtype=float)

    # Pairwise NaN-aware distances between every pair of subjects in the group.
    dist = nan_euclidean_distances(z_matrix, z_matrix)
    np.fill_diagonal(dist, np.inf)  # never pick yourself

    imputed = values.copy()
    was_imputed = pd.DataFrame(
        False, index=g.index, columns=feature_cols
    )

    idx_list = list(g.index)
    pos = {ix: p for p, ix in enumerate(idx_list)}

    for col in feature_cols:
        col_vals = values[col]
        missing_idx = col_vals.index[col_vals.isna()]
        if len(missing_idx) == 0:
            continue

        # Candidate donors: subjects in this group that HAVE this measurement.
        donor_idx = col_vals.index[col_vals.notna()]
        if len(donor_idx) == 0:
            continue  # nobody in this group has it -> cannot impute

        donor_pos = np.array([pos[ix] for ix in donor_idx])

        for ix in missing_idx:
            d = dist[pos[ix], donor_pos]
            valid = np.isfinite(d)
            if not valid.any():
                continue
            d_valid = d[valid]
            donors_valid = donor_idx[valid]

            k_eff = min(k, len(d_valid))
            nearest = np.argsort(d_valid)[:k_eff]
            nearest_ids = donors_valid[nearest]

            imputed.loc[ix, col] = col_vals.loc[nearest_ids].median()
            was_imputed.loc[ix, col] = True

    g[feature_cols] = imputed
    return g, was_imputed


def knn_impute_site(
    df: pd.DataFrame,
    k: int,
    feature_mode: str = "all",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Impute one full site table, stratifying by population (TYPE).
    Returns (imputed_df, was_imputed_mask) aligned to df's index.
    """
    distance_base = get_distance_columns(df)
    target_cols = get_target_columns(df)

    imputed_parts = []
    mask_parts = []

    for _type, group in df.groupby(POPULATION_COLUMN):
        if feature_mode == "corr":
            distance_cols = select_features_by_correlation(group, distance_base)
        else:
            distance_cols = distance_base

        g_imp, g_mask = knn_impute_group(
            group, target_cols, distance_cols, k
        )
        imputed_parts.append(g_imp)
        mask_parts.append(g_mask)

    imputed_df = pd.concat(imputed_parts).loc[df.index]
    mask_df = pd.concat(mask_parts).loc[df.index]
    return imputed_df, mask_df


# ============================================================
# VALIDATION: leave-one-value-out MAE by k and feature mode
# ============================================================

def run_validation_experiment(
    site_tables: dict[str, pd.DataFrame],
    k_values: list[int] = None,
    feature_modes: list[str] = None,
) -> pd.DataFrame:
    """
    Hide one known value at a time (leave-one-value-out), impute it and measure
    the absolute error. All values of k reuse the same sorted-neighbour list, so
    the sweep is cheap.

    Note: the group's z-scoring statistics are computed once from the full group
    (the tiny influence of the hidden value on its own column's mean/std is
    negligible for these sample sizes). The hidden value itself is excluded from
    the distance by nulling that subject's entry for the tested column.

    Returns a tidy DataFrame with columns:
        site, population, feature_mode, k, n_tested, mae, rmse
    """
    k_values = k_values or K_VALUES
    feature_modes = feature_modes or FEATURE_MODES

    rows = []

    for site, df in site_tables.items():
        distance_base = get_distance_columns(df)
        target_cols = get_target_columns(df)

        for _type, group in df.groupby(POPULATION_COLUMN):
            group = group.reset_index(drop=True)

            for mode in feature_modes:
                if mode == "corr":
                    distance_cols = select_features_by_correlation(
                        group, distance_base
                    )
                else:
                    distance_cols = distance_base

                # Standardize the distance columns once for this group.
                z, _, _ = standardize(group, distance_cols)
                z_matrix = z.to_numpy(dtype=float)
                col_pos = {c: i for i, c in enumerate(distance_cols)}

                # errors[k] accumulates absolute errors across all tested cells.
                errors = {k: [] for k in k_values}

                # Only segment measurements are evaluated for error.
                for col in target_cols:
                    col_vals = pd.to_numeric(group[col], errors="coerce")
                    observed = list(col_vals.index[col_vals.notna()])
                    if len(observed) < 3:
                        continue

                    col_in_distance = col in col_pos
                    donor_matrix = z_matrix[observed, :]

                    for ix in observed:
                        true_val = col_vals.loc[ix]

                        # Row for the tested subject; hide the tested column so
                        # its known value does not leak into the distance.
                        row_vec = z_matrix[ix, :].copy()
                        if col_in_distance:
                            row_vec[col_pos[col]] = np.nan

                        d = nan_euclidean_distances(
                            row_vec.reshape(1, -1), donor_matrix
                        )[0]

                        # Exclude the tested subject itself from its donors.
                        donors = [j for j in observed if j != ix]
                        d_donors = np.array([
                            d[observed.index(j)] for j in donors
                        ])
                        finite = np.isfinite(d_donors)
                        if not finite.any():
                            continue
                        d_donors = d_donors[finite]
                        donors = list(np.array(donors)[finite])

                        order = np.argsort(d_donors)
                        sorted_donor_ids = [donors[o] for o in order]

                        for k in k_values:
                            k_eff = min(k, len(sorted_donor_ids))
                            nn = sorted_donor_ids[:k_eff]
                            pred = col_vals.loc[nn].median()
                            if pd.isna(pred):
                                continue
                            errors[k].append(abs(pred - true_val))

                for k in k_values:
                    if errors[k]:
                        arr = np.array(errors[k], dtype=float)
                        rows.append({
                            "site": site,
                            "population": _type,
                            "feature_mode": mode,
                            "k": k,
                            "n_tested": len(arr),
                            "mae": round(float(arr.mean()), 3),
                            "rmse": round(float(np.sqrt((arr ** 2).mean())), 3),
                        })

    return pd.DataFrame(rows)


def run_validation_detail(
    site_tables: dict[str, pd.DataFrame],
    k: int = CHOSEN_K,
    feature_mode: str = CHOSEN_FEATURE_MODE,
) -> pd.DataFrame:
    """
    Same leave-one-value-out test as run_validation_experiment, but for a SINGLE
    k and feature mode, recording every individual attempt so the error is fully
    auditable.

    Returns one row per tested cell with columns:
        site, population, feature_mode, k, subject_id, column,
        true_value, predicted_value, abs_error, n_neighbors_used, neighbor_ids
    """
    rows = []

    for site, df in site_tables.items():
        distance_base = get_distance_columns(df)
        target_cols = get_target_columns(df)
        id_col = get_id_column(df)

        for _type, group in df.groupby(POPULATION_COLUMN):
            group = group.reset_index(drop=True)

            if feature_mode == "corr":
                distance_cols = select_features_by_correlation(group, distance_base)
            else:
                distance_cols = distance_base

            z, _, _ = standardize(group, distance_cols)
            z_matrix = z.to_numpy(dtype=float)
            col_pos = {c: i for i, c in enumerate(distance_cols)}

            # Only segment measurements are evaluated for error.
            for col in target_cols:
                col_vals = pd.to_numeric(group[col], errors="coerce")
                observed = list(col_vals.index[col_vals.notna()])
                if len(observed) < 3:
                    continue

                col_in_distance = col in col_pos
                donor_matrix = z_matrix[observed, :]

                for ix in observed:
                    true_val = col_vals.loc[ix]

                    row_vec = z_matrix[ix, :].copy()
                    if col_in_distance:
                        row_vec[col_pos[col]] = np.nan

                    d = nan_euclidean_distances(
                        row_vec.reshape(1, -1), donor_matrix
                    )[0]

                    donors = [j for j in observed if j != ix]
                    d_donors = np.array([d[observed.index(j)] for j in donors])
                    finite = np.isfinite(d_donors)
                    if not finite.any():
                        continue
                    d_donors = d_donors[finite]
                    donors = list(np.array(donors)[finite])

                    order = np.argsort(d_donors)
                    sorted_donor_ids = [donors[o] for o in order]

                    k_eff = min(k, len(sorted_donor_ids))
                    nn = sorted_donor_ids[:k_eff]
                    pred = col_vals.loc[nn].median()
                    if pd.isna(pred):
                        continue

                    def _label(j):
                        return str(group.loc[j, id_col]) if id_col else str(j)

                    rows.append({
                        "site": site,
                        "population": _type,
                        "feature_mode": feature_mode,
                        "k": k,
                        "subject_id": _label(ix),
                        "column": col,
                        "true_value": round(float(true_val), 3),
                        "predicted_value": round(float(pred), 3),
                        "abs_error": round(float(abs(pred - true_val)), 3),
                        "n_neighbors_used": k_eff,
                        "neighbor_ids": ";".join(_label(j) for j in nn),
                    })

    return pd.DataFrame(rows)


def summarize_validation(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the per-group results into one MAE per (feature_mode, k)."""
    if results.empty:
        return results
    agg = (
        results
        .assign(sq=lambda d: d["rmse"] ** 2 * d["n_tested"],
                abs_=lambda d: d["mae"] * d["n_tested"])
        .groupby(["feature_mode", "k"], as_index=False)
        .agg(n_tested=("n_tested", "sum"),
             abs_sum=("abs_", "sum"),
             sq_sum=("sq", "sum"))
    )
    agg["mae"] = (agg["abs_sum"] / agg["n_tested"]).round(3)
    agg["rmse"] = np.sqrt(agg["sq_sum"] / agg["n_tested"]).round(3)
    return agg[["feature_mode", "k", "n_tested", "mae", "rmse"]]


# ============================================================
# BUILD COMPLETE TABLES  (call this AFTER choosing k)
# ============================================================

def build_complete_tables(
    site_tables: dict[str, pd.DataFrame],
    output_paths: dict[str, Path],
    k: int,
    feature_mode: str = "all",
) -> dict[str, pd.DataFrame]:
    """
    Fill the gaps of each site table and write a *_complete.csv, leaving the
    original preprocessed file untouched. Adds a "<col>_was_imputed" flag per
    measurement so every filled cell is fully traceable.

    Only NaN cells are ever filled. Values flagged as invalid_range or outlier
    are left exactly as they are and still take part in the distance computation.
    """
    completed = {}
    for site, df in site_tables.items():
        imputed_df, mask_df = knn_impute_site(df, k=k, feature_mode=feature_mode)

        out = df.copy()
        target_cols = get_target_columns(df)
        out[target_cols] = imputed_df[target_cols]
        for col in target_cols:
            out[f"{col}_was_imputed"] = mask_df[col].values

        path = output_paths[site]
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)
        completed[site] = out

    return completed


# ============================================================
# CONVENIENCE: load from config / derive output paths
# ============================================================

def _complete_path(preprocessed_path: Path) -> Path:
    """Turn '<site>_preprocessed.csv' into '<site>_complete.csv' (same folder)."""
    return preprocessed_path.with_name(
        preprocessed_path.name.replace("_preprocessed", "_complete")
    )


def load_site_tables() -> tuple[dict[str, pd.DataFrame], dict[str, Path]]:
    """
    Read the three preprocessed tables (which contain the real NaN) and return
    them together with the derived *_complete.csv output paths.

    Config is imported lazily so this module can also be used/tested on its own.
    """
    from src.config import (
        COLOMBIA_PREPROCESSED_TABLE,
        BARCELONA_PREPROCESSED_TABLE,
        BRASIL_PREPROCESSED_TABLE,
    )

    paths_in = {
        "Colombia": COLOMBIA_PREPROCESSED_TABLE,
        "Barcelona": BARCELONA_PREPROCESSED_TABLE,
        "Brasil": BRASIL_PREPROCESSED_TABLE,
    }
    site_tables = {site: pd.read_csv(p) for site, p in paths_in.items()}
    output_paths = {site: _complete_path(p) for site, p in paths_in.items()}
    return site_tables, output_paths


def get_validation_output_dir() -> Path:
    """Folder where the validation CSVs are written (outputs/knn_imputation/)."""
    from src.config import OUTPUT_DIR
    out_dir = OUTPUT_DIR / "knn_imputation"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main():
    """
    1. Run the k-sweep validation and print/save the results.
    2. Save a detailed, cell-by-cell log of every attempt for the chosen k/mode.
    3. Build the *_complete.csv files with the chosen k/mode.
    """
    site_tables, output_paths = load_site_tables()
    out_dir = get_validation_output_dir()

    # -------------------------------------------------------------
    # 1. k-sweep validation
    # -------------------------------------------------------------
    results = run_validation_experiment(site_tables)
    summary = summarize_validation(results)

    print("=== KNN imputation validation (leave-one-value-out) ===\n")
    print("Global MAE / RMSE by feature mode and k:")
    print(summary.to_string(index=False))
    print("\nPer site and population:")
    print(
        results.sort_values(["site", "population", "feature_mode", "k"])
        .to_string(index=False)
    )

    summary.to_csv(out_dir / "knn_validation_summary.csv", index=False)
    results.to_csv(out_dir / "knn_validation_by_group.csv", index=False)

    # -------------------------------------------------------------
    # 2. Detailed, auditable log of every attempt for the chosen k/mode
    # -------------------------------------------------------------
    detail = run_validation_detail(
        site_tables, k=CHOSEN_K, feature_mode=CHOSEN_FEATURE_MODE
    )
    detail_path = out_dir / (
        f"knn_validation_detail_k{CHOSEN_K}_{CHOSEN_FEATURE_MODE}.csv"
    )
    detail.to_csv(detail_path, index=False)

    print(
        f"\n=== Detailed validation log (k={CHOSEN_K}, "
        f"mode='{CHOSEN_FEATURE_MODE}') ==="
    )
    print(f"Rows (cells tested): {len(detail)}")
    print("\nSample of individual attempts:")
    print(detail.head(15).to_string(index=False))
    print("\n10 random attempts (a representative sample of the imputation):")
    print(
        detail.sample(n=min(10, len(detail)), random_state=42)
        .to_string(index=False)
    )
    print(f"\nSaved detailed log to: {detail_path}")

    # -------------------------------------------------------------
    # 3. Build the complete tables with the chosen settings
    # -------------------------------------------------------------
    completed = build_complete_tables(
        site_tables,
        output_paths,
        k=CHOSEN_K,
        feature_mode=CHOSEN_FEATURE_MODE,
    )

    print(
        f"\nCompleted tables written (k={CHOSEN_K}, "
        f"mode='{CHOSEN_FEATURE_MODE}'):"
    )
    for site, path in output_paths.items():
        n_imputed = int(
            completed[site].filter(regex="_was_imputed$").sum().sum()
        )
        print(f"  - {site}: {path}  ({n_imputed} cells imputed)")


if __name__ == "__main__":
    main()
"""
quality_control.py

This module performs quality control on the equalized anthropometric tables.

It detects:
    - missing values
    - physiologically invalid ranges
    - statistical outliers using the IQR rule

The goal is conservative quality control. Values are not removed or corrected
automatically. Instead, quality flags are added to preserve traceability.
"""

import pandas as pd


# ============================================================
# VALID PHYSIOLOGICAL RANGES
# ============================================================

VALID_RANGES = {
    # General variables
    "AGE": (1, 100),
    "HEIGHT_cm": (90, 210),
    "WEIGHT_kg": (12, 200),
    "IMC_kg_m2": (10, 60),

    # Head and neck
    "PER_Skull_cm": (44, 64),
    "PER_Neck_cm": (20, 55),
    "THICK_WingedNeck_cm": (8, 41),

    # Back / torso distances
    "DIST_SupScapularAngToC7_AVG_cm": (3, 16),
    "DIST_SupScapularAngToT10_AVG_cm": (6, 29),
    "DIST_C7ToT10_cm": (7, 35),

    # Arm
    "LONG_Arm_AVG_cm": (25, 129),
    "DIST_UpperArm_AVG_cm": (11, 55),
    "DIST_Forearm_AVG_cm": (8, 43),
    "DIST_Hand_AVG_cm": (6, 32),

    # Leg
    "LONG_Leg_AVG_cm": (28, 144),
    "DIST_Thigh_AVG_cm": (14, 72),
    "DIST_LowerLeg_LATERAL_AVG_cm": (14, 72),
    "DIST_LowerLeg_MEDIAL_AVG_cm": (14, 72),
    "DIST_Foot_AVG_cm": (5, 26),

    # Torso
    "PER_Abdominal_cm": (40, 160),
    "DIST_BetweenSupAntIliacCrest_cm": (11, 56),
    "PER_Hip_cm": (45, 175),
    "DIST_SternalEndClavicleToSternalManubrium_AVG_cm": (0.2, 6),
    "DIST_ManubriumToXiphoidApophysis_cm": (6, 29),
    "DIST_ManubriumToCentralNavel_cm": (14, 71),
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Purpose:
        Return the columns that should be checked as numeric anthropometric
        or general features.

    Input:
        df (pd.DataFrame):
            Equalized or preprocessed dataframe.

    Output:
        list[str]:
            Numeric feature columns excluding metadata and QC columns.
    """

    excluded_columns = [
        "ID",
        "PSEUDO_ID",
        "TYPE",
        "QC_FLAGS",
        "QC_COLUMNS",
        "QC_N_FLAGS",
        "IMPUTED_COLUMNS",
        "N_IMPUTED_VALUES",
    ]

    return [column for column in df.columns if column not in excluded_columns]


def add_qc_issue(
    flags: set[str],
    columns: set[str],
    flag_name: str,
    column_name: str,
) -> None:
    """
    Purpose:
        Add one quality-control issue to the current row containers.

    Input:
        flags (set[str]):
            Set of QC flags for the row.

        columns (set[str]):
            Set of columns affected by QC issues.

        flag_name (str):
            Name of the QC flag.

        column_name (str):
            Name of the affected column.

    Output:
        None.
        The sets are modified in place.
    """

    flags.add(flag_name)
    columns.add(column_name)


# ============================================================
# OUTLIER BOUNDS
# ============================================================

def compute_iqr_bounds(
    df: pd.DataFrame,
    numeric_columns: list[str],
) -> dict[str, tuple[float, float]]:
    """
    Purpose:
        Compute IQR-based outlier limits for each numeric column.

    Input:
        df (pd.DataFrame):
            Equalized dataframe.

        numeric_columns (list[str]):
            Columns for which IQR bounds should be computed.

    Output:
        dict[str, tuple[float, float]]:
            Dictionary where each key is a column name and each value is:
                (lower_bound, upper_bound)
    """

    bounds = {}

    for column in numeric_columns:
        values = pd.to_numeric(df[column], errors="coerce").dropna()

        # IQR is not reliable if there are too few valid observations.
        if len(values) < 5:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        # If IQR is zero, all values are too similar and outlier detection
        # would not be meaningful.
        if iqr == 0:
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        bounds[column] = (lower_bound, upper_bound)

    return bounds


# ============================================================
# MAIN QUALITY CONTROL FUNCTION
# ============================================================

def add_quality_control_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        Add quality-control flags to an equalized dataframe.

    Detected issues:
        - missing_values
        - invalid_range
        - outlier

    Input:
        df (pd.DataFrame):
            Equalized dataframe.

    Output:
        pd.DataFrame:
            Dataframe with QC_FLAGS, QC_COLUMNS and QC_N_FLAGS.
    """

    checked_df = df.copy()

    numeric_columns = get_numeric_feature_columns(checked_df)

    # Ensure numeric columns are numeric before quality control.
    for column in numeric_columns:
        checked_df[column] = pd.to_numeric(checked_df[column], errors="coerce")

    # IQR outlier bounds are computed within the dataframe passed to the function.
    # Since Colombia and Barcelona will be processed separately, outliers are
    # automatically detected within each site.
    iqr_bounds = compute_iqr_bounds(checked_df, numeric_columns)

    qc_flags_all = []
    qc_columns_all = []
    qc_n_flags_all = []

    for _, row in checked_df.iterrows():
        row_flags = set()
        row_columns = set()

        for column in numeric_columns:
            value = row[column]

            # 1. Missing values
            if pd.isna(value):
                add_qc_issue(
                    row_flags,
                    row_columns,
                    "missing_values",
                    column,
                )
                continue

            # 2. Invalid physiological ranges
            if column in VALID_RANGES:
                min_value, max_value = VALID_RANGES[column]

                if value < min_value or value > max_value:
                    add_qc_issue(
                        row_flags,
                        row_columns,
                        "invalid_range",
                        column,
                    )

            # 3. Statistical outliers using IQR
            # AGE is excluded from outlier detection because it is a strong
            # confounder and should be handled during analysis, not flagged here.
            if column != "AGE" and column in iqr_bounds:
                lower_bound, upper_bound = iqr_bounds[column]

                if value < lower_bound or value > upper_bound:
                    add_qc_issue(
                        row_flags,
                        row_columns,
                        "outlier",
                        column,
                    )

        qc_flags_all.append(";".join(sorted(row_flags)))
        qc_columns_all.append(";".join(sorted(row_columns)))
        qc_n_flags_all.append(len(row_columns))

    checked_df["QC_FLAGS"] = qc_flags_all
    checked_df["QC_COLUMNS"] = qc_columns_all
    checked_df["QC_N_FLAGS"] = qc_n_flags_all

    return checked_df

# ============================================================
# QUALITY REPORT
# ============================================================

def get_missingness_status(missing_pct: float) -> str:
    """
    Purpose:
        Classify column completeness based on missing percentage.

    Input:
        missing_pct (float):
            Percentage of missing values in the column.

    Output:
        str:
            Completeness category.
    """

    if missing_pct == 0:
        return "complete"
    elif missing_pct <= 10:
        return "low_missingness"
    elif missing_pct <= 25:
        return "moderate_missingness"
    else:
        return "low_completeness"


def create_quality_report(
    df_qc: pd.DataFrame,
    site: str,
) -> pd.DataFrame:
    """
    Purpose:
        Create a quality-control report for one preprocessed dataframe.

    Input:
        df_qc (pd.DataFrame):
            Dataframe after quality-control flags have been added.

        site (str):
            Dataset/site name. Example: "Colombia" or "Barcelona".

    Output:
        pd.DataFrame:
            Quality report with one row per numeric feature.
    """

    numeric_columns = get_numeric_feature_columns(df_qc)

    report_rows = []

    n_rows = len(df_qc)

    for column in numeric_columns:
        values = pd.to_numeric(df_qc[column], errors="coerce")

        n_missing = values.isna().sum()
        missing_pct = (n_missing / n_rows) * 100 if n_rows > 0 else 0

        n_invalid_range = 0
        n_outliers = 0

        for _, row in df_qc.iterrows():
            qc_flags = str(row.get("QC_FLAGS", ""))
            qc_columns = str(row.get("QC_COLUMNS", ""))

            affected_columns = qc_columns.split(";") if qc_columns else []

            if column in affected_columns and "invalid_range" in qc_flags:
                n_invalid_range += 1

            if column in affected_columns and "outlier" in qc_flags:
                n_outliers += 1

        invalid_range_pct = (n_invalid_range / n_rows) * 100 if n_rows > 0 else 0
        outlier_pct = (n_outliers / n_rows) * 100 if n_rows > 0 else 0

        report_rows.append(
            {
                "site": site,
                "column": column,
                "n_rows": n_rows,
                "n_missing": n_missing,
                "missing_pct": round(missing_pct, 2),
                "n_invalid_range": n_invalid_range,
                "invalid_range_pct": round(invalid_range_pct, 2),
                "n_outliers": n_outliers,
                "outlier_pct": round(outlier_pct, 2),
                "column_status": get_missingness_status(missing_pct),
            }
        )

    return pd.DataFrame(report_rows)


def combine_quality_reports(
    colombia_qc: pd.DataFrame,
    barcelona_qc: pd.DataFrame,
) -> pd.DataFrame:
    """
    Purpose:
        Combine Colombia and Barcelona quality reports into one dataframe.

    Input:
        colombia_qc (pd.DataFrame):
            Colombia dataframe with QC flags.

        barcelona_qc (pd.DataFrame):
            Barcelona dataframe with QC flags.

    Output:
        pd.DataFrame:
            Combined quality report.
    """

    colombia_report = create_quality_report(colombia_qc, site="Colombia")
    barcelona_report = create_quality_report(barcelona_qc, site="Barcelona")

    return pd.concat(
        [colombia_report, barcelona_report],
        ignore_index=True,
    )

# ============================================================
# FLAGGED VALUES REPORT
# ============================================================

def create_flagged_values_report(
    df_qc: pd.DataFrame,
    site: str,
) -> pd.DataFrame:
    """
    Purpose:
        Create a detailed report of the specific values flagged during
        quality control.

    Input:
        df_qc (pd.DataFrame):
            Dataframe after quality-control flags have been added.

        site (str):
            Dataset/site name. Example: "Colombia" or "Barcelona".

    Output:
        pd.DataFrame:
            Detailed report with one row per flagged value.
    """

    report_rows = []

    for _, row in df_qc.iterrows():
        qc_flags = str(row.get("QC_FLAGS", ""))
        qc_columns = str(row.get("QC_COLUMNS", ""))

        if qc_flags == "" or qc_columns == "":
            continue

        affected_columns = qc_columns.split(";")

        # Prefer PSEUDO_ID if the table has already been pseudonymized.
        subject_identifier = row.get("PSEUDO_ID", row.get("ID", ""))

        for column in affected_columns:
            if column not in df_qc.columns:
                continue

            report_rows.append(
                {
                    "site": site,
                    "subject_id": subject_identifier,
                    "TYPE": row.get("TYPE", ""),
                    "column": column,
                    "value": row[column],
                    "QC_FLAGS": qc_flags,
                }
            )

    return pd.DataFrame(report_rows)


def combine_flagged_values_reports(
    colombia_qc: pd.DataFrame,
    barcelona_qc: pd.DataFrame,
) -> pd.DataFrame:
    """
    Purpose:
        Combine detailed flagged-value reports from Colombia and Barcelona.

    Input:
        colombia_qc (pd.DataFrame):
            Colombia dataframe with QC flags.

        barcelona_qc (pd.DataFrame):
            Barcelona dataframe with QC flags.

    Output:
        pd.DataFrame:
            Combined detailed flagged-value report.
    """

    colombia_flagged = create_flagged_values_report(
        colombia_qc,
        site="Colombia",
    )

    barcelona_flagged = create_flagged_values_report(
        barcelona_qc,
        site="Barcelona",
    )

    return pd.concat(
        [colombia_flagged, barcelona_flagged],
        ignore_index=True,
    )

# ============================================================
# MISSING VALUE IMPUTATION
# ============================================================

def get_missingness_level(missing_pct: float) -> str:
    """
    Purpose:
        Classify the missingness level of a column.

    Input:
        missing_pct (float):
            Percentage of missing values.

    Output:
        str:
            Missingness level.
    """

    if missing_pct == 0:
        return "none"
    elif missing_pct <= 10:
        return "low"
    elif missing_pct <= 25:
        return "moderate"
    else:
        return "high"


def add_qc_flag_string(
    current_flags: str,
    new_flag: str,
) -> str:
    """
    Purpose:
        Add a QC flag to an existing semicolon-separated flag string,
        avoiding duplicates.

    Input:
        current_flags (str):
            Existing QC_FLAGS value.

        new_flag (str):
            New flag to add.

    Output:
        str:
            Updated QC_FLAGS string.
    """

    if pd.isna(current_flags) or current_flags == "":
        flags = set()
    else:
        flags = set(str(current_flags).split(";"))

    flags.add(new_flag)

    return ";".join(sorted(flags))


def add_column_string(
    current_columns: str,
    new_column: str,
) -> str:
    """
    Purpose:
        Add a column name to an existing semicolon-separated column string,
        avoiding duplicates.

    Input:
        current_columns (str):
            Existing column list.

        new_column (str):
            New column to add.

    Output:
        str:
            Updated semicolon-separated column string.
    """

    if pd.isna(current_columns) or current_columns == "":
        columns = set()
    else:
        columns = set(str(current_columns).split(";"))

    columns.add(new_column)

    return ";".join(sorted(columns))


def get_imputation_value(
    df: pd.DataFrame,
    column: str,
    subject_type: str,
    min_group_values: int = 5,
) -> float | None:
    """
    Purpose:
        Compute the imputation value for a missing value.

    Strategy:
        1. Use the median of the same TYPE if there are enough valid values.
        2. Otherwise, use the global median within the same dataframe/site.

    Input:
        df (pd.DataFrame):
            Dataframe from one site.

        column (str):
            Column to impute.

        subject_type (str):
            Clinical group of the subject.

        min_group_values (int):
            Minimum number of valid values required to use TYPE-specific median.

    Output:
        float | None:
            Imputation value. None if no valid value exists.
    """

    same_type_values = pd.to_numeric(
        df.loc[df["TYPE"] == subject_type, column],
        errors="coerce",
    ).dropna()

    if len(same_type_values) >= min_group_values:
        return same_type_values.median()

    global_values = pd.to_numeric(df[column], errors="coerce").dropna()

    if len(global_values) == 0:
        return None

    return global_values.median()


def impute_missing_values(
    df_qc: pd.DataFrame,
    max_low_missing_pct: float = 10.0,
    min_group_values: int = 5,
) -> pd.DataFrame:
    """
    Purpose:
        Impute isolated missing values in a conservative and traceable way.

    Strategy:
        - If a column has 0% missing values, nothing is done.
        - If a column has >0% and <=10% missing values, missing values are imputed.
        - If a column has >10% missing values, it is not imputed automatically.

    Imputation method:
        - Median within the same TYPE if there are enough valid values.
        - Otherwise, global median within the same site/dataframe.

    Input:
        df_qc (pd.DataFrame):
            Dataframe after quality-control flags have been added.

        max_low_missing_pct (float):
            Maximum percentage of missingness allowed for automatic imputation.

        min_group_values (int):
            Minimum number of valid values required for TYPE-specific median.

    Output:
        pd.DataFrame:
            Dataframe with imputed values if needed and imputation traceability columns.
    """

    imputed_df = df_qc.copy()

    numeric_columns = get_numeric_feature_columns(imputed_df)
    n_rows = len(imputed_df)

    # Traceability columns.
    imputed_df["IMPUTED_COLUMNS"] = ""
    imputed_df["N_IMPUTED_VALUES"] = 0

    for column in numeric_columns:
        values = pd.to_numeric(imputed_df[column], errors="coerce")
        n_missing = values.isna().sum()

        if n_missing == 0:
            continue

        missing_pct = (n_missing / n_rows) * 100 if n_rows > 0 else 0

        # Conservative rule: only automatically impute low missingness columns.
        if missing_pct > max_low_missing_pct:
            continue

        missing_indices = imputed_df.index[values.isna()]

        for idx in missing_indices:
            subject_type = imputed_df.loc[idx, "TYPE"]

            imputation_value = get_imputation_value(
                imputed_df,
                column=column,
                subject_type=subject_type,
                min_group_values=min_group_values,
            )

            if imputation_value is None:
                continue

            imputed_df.loc[idx, column] = imputation_value

            # Keep QC traceability.
            imputed_df.loc[idx, "QC_FLAGS"] = add_qc_flag_string(
                imputed_df.loc[idx, "QC_FLAGS"],
                "imputed_values",
            )

            imputed_df.loc[idx, "QC_COLUMNS"] = add_column_string(
                imputed_df.loc[idx, "QC_COLUMNS"],
                column,
            )

            imputed_df.loc[idx, "IMPUTED_COLUMNS"] = add_column_string(
                imputed_df.loc[idx, "IMPUTED_COLUMNS"],
                column,
            )

            imputed_df.loc[idx, "N_IMPUTED_VALUES"] += 1

    # Recalculate QC_N_FLAGS after possible imputation updates.
    imputed_df["QC_N_FLAGS"] = imputed_df["QC_COLUMNS"].apply(
        lambda x: 0 if pd.isna(x) or x == "" else len(str(x).split(";"))
    )

    return imputed_df
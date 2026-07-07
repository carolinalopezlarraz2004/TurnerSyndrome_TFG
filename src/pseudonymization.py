"""
pseudonymization.py

This module creates reproducible pseudonymous subject identifiers using SHA-256.

The goal is to remove original subject IDs from analysis-ready tables while
keeping a private internal mapping file for traceability.

This is pseudonymization, not full anonymization, because the private mapping
file allows the original IDs to be recovered internally if needed.
"""

import hashlib
import pandas as pd

from src.config import PSEUDONYMIZATION_SALT


# ============================================================
# HASHING FUNCTIONS
# ============================================================

def hash_subject_id(subject_id: str, site_prefix: str) -> str:
    """
    Purpose:
        Generate a reproducible pseudonymous subject ID using SHA-256.

    Input:
        subject_id (str):
            Original subject identifier.

        site_prefix (str):
            Site prefix used to preserve dataset origin.
            Example: "CO" or "ES".

    Output:
        str:
            Pseudonymous subject identifier.
            Example: "CO_a8f31c9d"
    """

    value_to_hash = PSEUDONYMIZATION_SALT + str(subject_id)
    hash_value = hashlib.sha256(value_to_hash.encode()).hexdigest()[:8]

    return f"{site_prefix}_{hash_value}"


def add_pseudo_id(
    df: pd.DataFrame,
    site_prefix: str,
) -> pd.DataFrame:
    """
    Purpose:
        Add a pseudonymous ID column to a dataframe.

    Input:
        df (pd.DataFrame):
            Dataframe containing an original ID column.

        site_prefix (str):
            Site prefix used in the pseudonymous ID.

    Output:
        pd.DataFrame:
            Dataframe with PSEUDO_ID added as the first column.
    """

    pseudo_df = df.copy()

    pseudo_df["PSEUDO_ID"] = pseudo_df["ID"].apply(
        lambda subject_id: hash_subject_id(
            subject_id=subject_id,
            site_prefix=site_prefix,
        )
    )

    # Move PSEUDO_ID to the first column.
    columns = ["PSEUDO_ID"] + [
        column for column in pseudo_df.columns if column != "PSEUDO_ID"
    ]

    pseudo_df = pseudo_df[columns]

    return pseudo_df


def remove_original_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        Remove the original ID column from an analysis-ready dataframe.

    Input:
        df (pd.DataFrame):
            Dataframe containing ID.

    Output:
        pd.DataFrame:
            Dataframe without original ID.
    """

    return df.drop(columns=["ID"], errors="ignore")


def create_pseudo_id_mapping(
    df: pd.DataFrame,
    site_name: str,
) -> pd.DataFrame:
    """
    Purpose:
        Create a private mapping between original IDs and pseudonymous IDs.

    Input:
        df (pd.DataFrame):
            Dataframe containing ID and PSEUDO_ID.

        site_name (str):
            Site name. Example: "Colombia" or "Barcelona".

    Output:
        pd.DataFrame:
            Mapping dataframe.
    """

    mapping = df[["ID", "PSEUDO_ID"]].copy()
    mapping.insert(0, "site", site_name)

    return mapping


def pseudonymize_preprocessed_table(
    df: pd.DataFrame,
    site_prefix: str,
    site_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Purpose:
        Add PSEUDO_ID, create the private mapping, and remove the original ID
        from the analysis-ready table.

    Input:
        df (pd.DataFrame):
            Preprocessed dataframe with original ID.

        site_prefix (str):
            Site prefix used in PSEUDO_ID.
            Example: "CO" or "ES".

        site_name (str):
            Full site name.
            Example: "Colombia" or "Barcelona".

    Output:
        tuple[pd.DataFrame, pd.DataFrame]:
            - pseudonymized dataframe without original ID
            - private ID mapping dataframe
    """

    df_with_pseudo = add_pseudo_id(
        df=df,
        site_prefix=site_prefix,
    )

    mapping = create_pseudo_id_mapping(
        df=df_with_pseudo,
        site_name=site_name,
    )

    pseudo_df = remove_original_id(df_with_pseudo)

    return pseudo_df, mapping
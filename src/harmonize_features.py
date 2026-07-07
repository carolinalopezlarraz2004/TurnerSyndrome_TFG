"""
harmonize_features.py

This module equalizes the Colombia and Barcelona anthropometric feature tables.

Colombia is used as the master schema.
Barcelona is transformed to match Colombia in column names, units, resolution and order.
"""

import pandas as pd

from src.config import (
    COLOMBIA_TABLE,
    BARCELONA_TABLE,
    COLOMBIA_EQUALIZED_TABLE,
    BARCELONA_EQUALIZED_TABLE,
)


# ============================================================
# FINAL COLUMN ORDER
# ============================================================

FINAL_COLUMNS = [
    "ID",
    "TYPE",
    "AGE",
    "HEIGHT_cm",
    "WEIGHT_kg",
    "IMC_kg_m2",
    "PER_Skull_cm",
    "PER_Neck_cm",
    "THICK_WingedNeck_cm",
    "DIST_SupScapularAngToC7_AVG_cm",
    "DIST_SupScapularAngToT10_AVG_cm",
    "DIST_C7ToT10_cm",
    "LONG_Arm_AVG_cm",
    "DIST_UpperArm_AVG_cm",
    "DIST_Forearm_AVG_cm",
    "DIST_Hand_AVG_cm",
    "LONG_Leg_AVG_cm",
    "DIST_Thigh_AVG_cm",
    "DIST_LowerLeg_LATERAL_AVG_cm",
    "DIST_LowerLeg_MEDIAL_AVG_cm",
    "DIST_Foot_AVG_cm",
    "PER_Abdominal_cm",
    "DIST_BetweenSupAntIliacCrest_cm",
    "PER_Hip_cm",
    "DIST_SternalEndClavicleToSternalManubrium_AVG_cm",
    "DIST_ManubriumToXiphoidApophysis_cm",
    "DIST_ManubriumToCentralNavel_cm",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def mean_columns(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """
    Purpose:
        Compute the row-wise average of several columns.

    Input:
        df (pd.DataFrame):
            Input dataframe.

        columns (list[str]):
            Columns to average.

    Output:
        pd.Series:
            Row-wise average.
    """

    return df[columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)


def force_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        Convert all measurement columns to numeric values.

    Input:
        df (pd.DataFrame):
            Equalized dataframe.

    Output:
        pd.DataFrame:
            Dataframe with numeric feature columns.
    """

    for column in df.columns:
        if column not in ["ID", "TYPE"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


# ============================================================
# COLOMBIA
# ============================================================

def equalize_colombia() -> pd.DataFrame:
    """
    Purpose:
        Read and equalize the Colombia table.

    Output:
        pd.DataFrame:
            Colombia table with the final equalized column structure.
    """

    # Colombia has two header rows. header=1 keeps the real column names.
    df = pd.read_csv(COLOMBIA_TABLE, header=1)

    equalized = df.copy()

    # New BMI column, created to match Barcelona.
    equalized["IMC_kg_m2"] = (
        equalized["WEIGHT_kg"] / (equalized["HEIGHT_cm"] / 100) ** 2
    )

    # Collapse right/left columns into one average column.
    equalized["DIST_SupScapularAngToC7_AVG_cm"] = mean_columns(
        equalized,
        [
            "DIST_SupScapularAngToC7_RIGHT_cm",
            "DIST_SupScapularAngToC7_LEFT_cm",
        ],
    )

    equalized["DIST_SupScapularAngToT10_AVG_cm"] = mean_columns(
        equalized,
        [
            "DIST_SupScapularAngToT10_RIGHT_cm",
            "DIST_SupScapularAngToT10_LEFT_cm",
        ],
    )

    equalized["DIST_SternalEndClavicleToSternalManubrium_AVG_cm"] = mean_columns(
        equalized,
        [
            "DIST_SternalEndClavicleToSternalManubrium_RIGHT_cm",
            "DIST_SternalEndClavicleToSternalManubrium_LEFT_cm",
        ],
    )

    equalized = equalized[FINAL_COLUMNS]
    equalized = force_numeric_features(equalized)

    return equalized


# ============================================================
# BARCELONA
# ============================================================

def equalize_barcelona() -> pd.DataFrame:
    """
    Purpose:
        Read and equalize the Barcelona table to the Colombia schema.

    Output:
        pd.DataFrame:
            Barcelona table with the same columns and order as Colombia.
    """

    # Raw_Match_Anon contains all manual measurements.
    raw = pd.read_excel(BARCELONA_TABLE, sheet_name="Raw_Match_Anon")

    # Remove leading/trailing spaces from column names.
    raw.columns = raw.columns.str.strip()

    equalized = pd.DataFrame()

    # Basic metadata.
    equalized["ID"] = raw["Code"]
    equalized["TYPE"] = raw["Type"]
    equalized["AGE"] = raw["Age"]

    # Units.
    equalized["HEIGHT_cm"] = pd.to_numeric(raw["Altura_m"], errors="coerce") * 100
    equalized["WEIGHT_kg"] = pd.to_numeric(raw["Peso_kg"], errors="coerce")

    # BMI is recalculated, not taken from Excel.
    equalized["IMC_kg_m2"] = (
        equalized["WEIGHT_kg"] / (equalized["HEIGHT_cm"] / 100) ** 2
    )

    # Head and neck.
    equalized["PER_Skull_cm"] = raw["Perimetro_craneal_cm"]
    equalized["PER_Neck_cm"] = raw["Perimetro_cuello_cm"]
    equalized["THICK_WingedNeck_cm"] = raw["Grosor_cuello_alado_cm"]

    # Scapular distances.
    equalized["DIST_SupScapularAngToC7_AVG_cm"] = raw[
        "Distancia de angulo superior escapular a C7"
    ]

    equalized["DIST_SupScapularAngToT10_AVG_cm"] = raw[
        "Distancia de angulo superior escapular a T10"
    ]

    equalized["DIST_C7ToT10_cm"] = raw["Distancia_C7_t10"]

    # Arm measurements.
    equalized["LONG_Arm_AVG_cm"] = mean_columns(
        raw,
        [
            "Longitud_brazo_derecho_cm (Tubérculo mayor a punta dedo 3)",
            "Longitud_brazo_izquierdo_cm (Tubérculo mayor a punta dedo 3)",
        ],
    )

    equalized["DIST_UpperArm_AVG_cm"] = mean_columns(
        raw,
        [
            "Distancia_brazo_derecho_cm (punto medio del tubérculo mayor a fosa del codo)",
            "Distancia_brazo_izquierdo_cm (punto medio del tubérculo mayor a fosa del codo)",
        ],
    )

    equalized["DIST_Forearm_AVG_cm"] = mean_columns(
        raw,
        [
            "Distancia_antebrazo_derecho_cm (fosa del codo a línea muñeca)",
            "Distancia_antebrazo_izquierdo_cm (fosa del codo a línea muñeca)",
        ],
    )

    equalized["DIST_Hand_AVG_cm"] = raw["Distancia_muneca_a_dedo3_cm"]

    # Leg measurements.
    equalized["LONG_Leg_AVG_cm"] = mean_columns(
        raw,
        [
            "Longitud_pierna_derecha_cm (espina ilíaca antero superior al maleolo medial)",
            "Longitud_pierna_izquierda_cm (espina ilíaca antero superior al maleolo medial)",
        ],
    )

    equalized["DIST_Thigh_AVG_cm"] = mean_columns(
        raw,
        [
            "Distancia_muslo_derecho_cm (troncanter mayor a condilo lateral)",
            "Distancia_muslo_izquierdo_cm  (troncanter mayor a condilo lateral)",
        ],
    )

    equalized["DIST_LowerLeg_LATERAL_AVG_cm"] = mean_columns(
        raw,
        [
            "Distancia_pantorrilla_derecha_cm (condilo lateral al maleolo lateral)",
            "Distancia_pantorrilla_izquierda_cm  (condilo lateral al maleolo lateral)",
        ],
    )

    equalized["DIST_LowerLeg_MEDIAL_AVG_cm"] = raw[
        "Distancia_condilo_medial_tibia_a_maleolo_medial_cm"
    ]

    equalized["DIST_Foot_AVG_cm"] = raw["Distancia_astragalo_a_dedo2_pie_cm"]

    # Torso measurements.
    equalized["PER_Abdominal_cm"] = raw[
        "Perimetro_abdominal_cm (debajo de las costillas)"
    ]

    equalized["DIST_BetweenSupAntIliacCrest_cm"] = raw[
        "Distancia de cresta ilíaca antero superior a cresta ilíaca antero superior"
    ]

    equalized["PER_Hip_cm"] = raw["Perimetro_cadera_cm (coxofemoral)"]

    equalized["DIST_SternalEndClavicleToSternalManubrium_AVG_cm"] = raw[
        "Distancia_extremo_esternal_clavicula_a_manubrio_esternal_cm"
    ]

    equalized["DIST_ManubriumToXiphoidApophysis_cm"] = raw[
        "Distancia_manubrio_a_apofisis_xifoides_cm"
    ]

    equalized["DIST_ManubriumToCentralNavel_cm"] = raw[
        "Distancia_manubrio_a_ombligo_cm"
    ]

    equalized = equalized[FINAL_COLUMNS]
    equalized = force_numeric_features(equalized)

    return equalized


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_equalized_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Purpose:
        Create and save the equalized Colombia and Barcelona CSV files.

    Output:
        tuple[pd.DataFrame, pd.DataFrame]:
            Equalized Colombia and Barcelona dataframes.
    """

    colombia_equalized = equalize_colombia()
    barcelona_equalized = equalize_barcelona()

    COLOMBIA_EQUALIZED_TABLE.parent.mkdir(parents=True, exist_ok=True)
    BARCELONA_EQUALIZED_TABLE.parent.mkdir(parents=True, exist_ok=True)

    colombia_equalized.to_csv(COLOMBIA_EQUALIZED_TABLE, index=False)
    barcelona_equalized.to_csv(BARCELONA_EQUALIZED_TABLE, index=False)

    return colombia_equalized, barcelona_equalized
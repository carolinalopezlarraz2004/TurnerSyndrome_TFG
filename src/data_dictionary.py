"""
data_dictionary.py

This module creates the canonical data dictionary for the anthropometric
feature harmonization process.

The dictionary documents the final common variables used to compare Colombia
and Barcelona after equalization. It does not modify the data. Its purpose is
traceability: for each canonical variable, it stores the anatomical meaning,
unit, source column in each dataset and processing rule.
"""

import pandas as pd

from src.config import CANONICAL_DICTIONARY_TABLE


# ============================================================
# CANONICAL DATA DICTIONARY
# ============================================================

CANONICAL_DICTIONARY = [
    {
        "canonical_name": "ID",
        "category": "metadata",
        "unit": "",
        "definition": "Original subject identifier.",
        "colombia_source": "ID",
        "barcelona_source": "Code",
        "processing": "Direct mapping.",
        "notes": "Kept in equalized tables. Replaced by ANON_ID in preprocessed tables.",
    },
    {
        "canonical_name": "TYPE",
        "category": "metadata",
        "unit": "",
        "definition": "Clinical group of the subject: Turner syndrome or control.",
        "colombia_source": "TYPE",
        "barcelona_source": "Type",
        "processing": "Direct mapping.",
        "notes": "Used for group-wise comparison and median imputation if needed.",
    },
    {
        "canonical_name": "AGE",
        "category": "general",
        "unit": "years",
        "definition": "Subject age at acquisition.",
        "colombia_source": "AGE",
        "barcelona_source": "Age",
        "processing": "Direct mapping.",
        "notes": "Age is kept as metadata and possible confounder.",
    },
    {
        "canonical_name": "HEIGHT_cm",
        "category": "general",
        "unit": "cm",
        "definition": "Standing body height.",
        "colombia_source": "HEIGHT_cm",
        "barcelona_source": "Altura_m",
        "processing": "Barcelona height converted from meters to centimeters.",
        "notes": "Used for BMI calculation and later normalized indices.",
    },
    {
        "canonical_name": "WEIGHT_kg",
        "category": "general",
        "unit": "kg",
        "definition": "Body weight.",
        "colombia_source": "WEIGHT_kg",
        "barcelona_source": "Peso_kg",
        "processing": "Direct mapping.",
        "notes": "",
    },
    {
        "canonical_name": "PONDERAL_INDEX_kg_m3",
        "category": "general",
        "unit": "kg/m3",
        "definition": "Ponderal index (Rohrer index).",
        "colombia_source": "computed from HEIGHT_cm and WEIGHT_kg",
        "barcelona_source": "computed from HEIGHT_cm and WEIGHT_kg",
        "processing": "Recomputed as weight divided by cubed height in meters.",
        "notes": "Recomputed for all datasets to ensure consistency.",
    },
    {
        "canonical_name": "PER_Skull_cm",
        "category": "head_neck",
        "unit": "cm",
        "definition": "Skull/head perimeter.",
        "colombia_source": "PER_Skull_cm",
        "barcelona_source": "Perimetro_craneal_cm",
        "processing": "Direct mapping.",
        "notes": "",
    },
    {
        "canonical_name": "PER_Neck_cm",
        "category": "head_neck",
        "unit": "cm",
        "definition": "Neck perimeter.",
        "colombia_source": "PER_Neck_cm",
        "barcelona_source": "Perimetro_cuello_cm",
        "processing": "Direct mapping.",
        "notes": "",
    },
    {
        "canonical_name": "THICK_WingedNeck_cm",
        "category": "head_neck",
        "unit": "cm",
        "definition": "Thickness or width associated with winged neck morphology.",
        "colombia_source": "THICK_WingedNeck_cm",
        "barcelona_source": "Grosor_cuello_alado_cm",
        "processing": "Direct mapping.",
        "notes": "Potentially relevant feature for Turner syndrome phenotype.",
    },
    {
        "canonical_name": "DIST_SupScapularAngToC7_AVG_cm",
        "category": "back_torso",
        "unit": "cm",
        "definition": "Average distance from superior scapular angle to C7.",
        "colombia_source": "DIST_SupScapularAngToC7_RIGHT_cm; DIST_SupScapularAngToC7_LEFT_cm",
        "barcelona_source": "Distancia de angulo superior escapular a C7",
        "processing": "Colombia right/left averaged. Barcelona mapped as available measurement.",
        "notes": "Equalized to a single comparable measurement.",
    },
    {
        "canonical_name": "DIST_SupScapularAngToT10_AVG_cm",
        "category": "back_torso",
        "unit": "cm",
        "definition": "Average distance from superior scapular angle to T10.",
        "colombia_source": "DIST_SupScapularAngToT10_RIGHT_cm; DIST_SupScapularAngToT10_LEFT_cm",
        "barcelona_source": "Distancia de angulo superior escapular a T10",
        "processing": "Colombia right/left averaged. Barcelona mapped as available measurement.",
        "notes": "Equalized to a single comparable measurement.",
    },
    {
        "canonical_name": "DIST_C7ToT10_cm",
        "category": "back_torso",
        "unit": "cm",
        "definition": "Distance from C7 vertebra to T10 vertebra.",
        "colombia_source": "DIST_C7ToT10_cm",
        "barcelona_source": "Distancia_C7_t10",
        "processing": "Direct mapping.",
        "notes": "",
    },
    {
        "canonical_name": "LONG_Arm_AVG_cm",
        "category": "arm",
        "unit": "cm",
        "definition": "Average total arm length.",
        "colombia_source": "LONG_Arm_AVG_cm",
        "barcelona_source": "Longitud_brazo_derecho_cm; Longitud_brazo_izquierdo_cm",
        "processing": "Barcelona right/left averaged to match Colombia.",
        "notes": "Existing equalized rule preserved.",
    },
    {
        "canonical_name": "DIST_UpperArm_AVG_cm",
        "category": "arm",
        "unit": "cm",
        "definition": "Average upper arm distance.",
        "colombia_source": "DIST_UpperArm_AVG_cm",
        "barcelona_source": "Distancia_brazo_derecho_cm; Distancia_brazo_izquierdo_cm",
        "processing": "Barcelona right/left averaged to match Colombia.",
        "notes": "Existing equalized rule preserved.",
    },
    {
        "canonical_name": "DIST_Forearm_AVG_cm",
        "category": "arm",
        "unit": "cm",
        "definition": "Average forearm distance.",
        "colombia_source": "DIST_Forearm_AVG_cm",
        "barcelona_source": "Distancia_antebrazo_derecho_cm; Distancia_antebrazo_izquierdo_cm",
        "processing": "Barcelona right/left averaged to match Colombia.",
        "notes": "Existing equalized rule preserved.",
    },
    {
        "canonical_name": "DIST_Hand_AVG_cm",
        "category": "arm",
        "unit": "cm",
        "definition": "Hand length or wrist-to-third-finger distance.",
        "colombia_source": "DIST_Hand_AVG_cm",
        "barcelona_source": "Distancia_muneca_a_dedo3_cm",
        "processing": "Mapped as available measurement.",
        "notes": "Equalized as a single comparable measurement.",
    },
    {
        "canonical_name": "LONG_Leg_AVG_cm",
        "category": "leg",
        "unit": "cm",
        "definition": "Average total leg length.",
        "colombia_source": "LONG_Leg_AVG_cm",
        "barcelona_source": "Longitud_pierna_derecha_cm; Longitud_pierna_izquierda_cm",
        "processing": "Barcelona right/left averaged to match Colombia.",
        "notes": "Existing equalized rule preserved.",
    },
    {
        "canonical_name": "DIST_Thigh_AVG_cm",
        "category": "leg",
        "unit": "cm",
        "definition": "Average thigh distance.",
        "colombia_source": "DIST_Thigh_AVG_cm",
        "barcelona_source": "Distancia_muslo_derecho_cm; Distancia_muslo_izquierdo_cm",
        "processing": "Barcelona right/left averaged to match Colombia.",
        "notes": "Existing equalized rule preserved.",
    },
    {
        "canonical_name": "DIST_LowerLeg_LATERAL_AVG_cm",
        "category": "leg",
        "unit": "cm",
        "definition": "Average lateral lower leg distance.",
        "colombia_source": "DIST_LowerLeg_LATERAL_AVG_cm",
        "barcelona_source": "Distancia_pantorrilla_derecha_cm; Distancia_pantorrilla_izquierda_cm",
        "processing": "Barcelona right/left averaged to match Colombia.",
        "notes": "Existing equalized rule preserved.",
    },
    {
        "canonical_name": "DIST_LowerLeg_MEDIAL_AVG_cm",
        "category": "leg",
        "unit": "cm",
        "definition": "Medial lower leg distance.",
        "colombia_source": "DIST_LowerLeg_MEDIAL_AVG_cm",
        "barcelona_source": "Distancia_condilo_medial_tibia_a_maleolo_medial_cm",
        "processing": "Mapped as available measurement.",
        "notes": "Equalized as a single comparable measurement because bilateral information is not consistently available.",
    },
    {
        "canonical_name": "DIST_Foot_AVG_cm",
        "category": "leg",
        "unit": "cm",
        "definition": "Foot length.",
        "colombia_source": "DIST_Foot_AVG_cm",
        "barcelona_source": "Distancia_astragalo_a_dedo2_pie_cm",
        "processing": "Mapped as available measurement.",
        "notes": "Equalized as a single comparable measurement.",
    },
    {
        "canonical_name": "PER_Abdominal_cm",
        "category": "torso",
        "unit": "cm",
        "definition": "Abdominal perimeter.",
        "colombia_source": "PER_Abdominal_cm",
        "barcelona_source": "Perimetro_abdominal_cm",
        "processing": "Direct mapping.",
        "notes": "",
    },
    {
        "canonical_name": "DIST_BetweenSupAntIliacCrest_cm",
        "category": "torso",
        "unit": "cm",
        "definition": "Distance between right and left anterior superior iliac crests.",
        "colombia_source": "DIST_BetweenSupAntIliacCrest_cm",
        "barcelona_source": "Distancia de cresta ilíaca antero superior a cresta ilíaca antero superior",
        "processing": "Direct mapping.",
        "notes": "",
    },
    {
        "canonical_name": "PER_Hip_cm",
        "category": "torso",
        "unit": "cm",
        "definition": "Hip perimeter.",
        "colombia_source": "PER_Hip_cm",
        "barcelona_source": "Perimetro_cadera_cm",
        "processing": "Direct mapping.",
        "notes": "",
    },
    {
        "canonical_name": "DIST_SternalEndClavicleToSternalManubrium_AVG_cm",
        "category": "torso",
        "unit": "cm",
        "definition": "Average distance from sternal end of clavicle to sternal manubrium.",
        "colombia_source": "DIST_SternalEndClavicleToSternalManubrium_RIGHT_cm; DIST_SternalEndClavicleToSternalManubrium_LEFT_cm",
        "barcelona_source": "Distancia_extremo_esternal_clavicula_a_manubrio_esternal_cm",
        "processing": "Colombia right/left averaged. Barcelona mapped as available measurement.",
        "notes": "Equalized as a single comparable measurement.",
    },
    {
        "canonical_name": "DIST_ManubriumToXiphoidApophysis_cm",
        "category": "torso",
        "unit": "cm",
        "definition": "Distance from sternal manubrium to xiphoid apophysis.",
        "colombia_source": "DIST_ManubriumToXiphoidApophysis_cm",
        "barcelona_source": "Distancia_manubrio_a_apofisis_xifoides_cm",
        "processing": "Direct mapping.",
        "notes": "",
    },
    {
        "canonical_name": "DIST_ManubriumToCentralNavel_cm",
        "category": "torso",
        "unit": "cm",
        "definition": "Distance from sternal manubrium to central navel.",
        "colombia_source": "DIST_ManubriumToCentralNavel_cm",
        "barcelona_source": "Distancia_manubrio_a_ombligo_cm",
        "processing": "Direct mapping.",
        "notes": "",
    },
]


# ============================================================
# SAVE FUNCTION
# ============================================================

def create_canonical_dictionary() -> pd.DataFrame:
    """
    Purpose:
        Create the canonical data dictionary as a dataframe.

    Input:
        None.

    Output:
        pd.DataFrame:
            Data dictionary with one row per canonical variable.
    """

    return pd.DataFrame(CANONICAL_DICTIONARY)


def save_canonical_dictionary() -> pd.DataFrame:
    """
    Purpose:
        Save the canonical data dictionary to CSV.

    Input:
        None.

    Output:
        pd.DataFrame:
            Saved data dictionary.
    """

    dictionary_df = create_canonical_dictionary()

    CANONICAL_DICTIONARY_TABLE.parent.mkdir(parents=True, exist_ok=True)
    dictionary_df.to_csv(CANONICAL_DICTIONARY_TABLE, index=False)

    return dictionary_df
"""
config.py

Configuration file for the Turner Syndrome TFG project.

This file stores the main paths and constants used by the preprocessing pipeline.
The goal is to keep these values in one place, so if a folder name, ArUco setting
or dataset parameter changes, it only needs to be updated here.
"""

from pathlib import Path
import cv2


# ============================================================
# PROJECT PATHS
# ============================================================

# PROJECT_ROOT points to the main folder of the project.
# Since this file is inside "src", parents[1] goes one level up
# and gives the root folder: TurnerSyndrome_TFG.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Main folder where all input data is stored.
DATA_DIR = PROJECT_ROOT / "data_files"

# Main folder where all generated outputs will be saved.
OUTPUT_DIR = PROJECT_ROOT / "outputs"


# ============================================================
# DATASET FOLDERS
# ============================================================

# Folder containing the Colombian dataset.
COLOMBIA_DIR = DATA_DIR / "data_files_colombia"

# Folder containing the Barcelona dataset.
BARCELONA_DIR = DATA_DIR / "data_files_barcelona"

# Folder reserved for the Brazilian dataset.
BRASIL_DIR = DATA_DIR / "data_files_brasil"


# ============================================================
# MEASUREMENT TABLES
# ============================================================

# Colombia measurement table.
COLOMBIA_TABLE = COLOMBIA_DIR / "data_files_colombia_features" / "Colombia(Clean).csv"

# Barcelona measurement table.
BARCELONA_TABLE = BARCELONA_DIR / "data_files_barcelona_features" / "MesuresSTCarolina.xlsx"

# Brazil measurement table.
# The measurements live in the "Clean" sheet of this workbook, which already
# follows the same schema as the Colombia raw table (RIGHT/LEFT pairs not yet
# averaged and no BMI column).
BRASIL_TABLE = BRASIL_DIR / "data_files_brasil_features" / "Brasil26.xlsx"


# ============================================================
# OUTPUT FOLDERS
# ============================================================

# Folder where ArUco detection debug images will be saved.
DEBUG_ARUCO_DIR = OUTPUT_DIR / "debug_aruco"

# Folder where global table reports will be saved.
TABLES_OUTPUT_DIR = OUTPUT_DIR / "tables"

# Folder where private mapping files will be saved.
PRIVATE_OUTPUT_DIR = OUTPUT_DIR / "private"


# ============================================================
# EQUALIZED TABLE OUTPUTS
# ============================================================

# Equalized tables are saved inside each dataset feature folder.
COLOMBIA_EQUALIZED_TABLE = (
    COLOMBIA_DIR / "data_files_colombia_features" / "Colombia_equalized.csv"
)

BARCELONA_EQUALIZED_TABLE = (
    BARCELONA_DIR / "data_files_barcelona_features" / "Barcelona_equalized.csv"
)

BRASIL_EQUALIZED_TABLE = (
    BRASIL_DIR / "data_files_brasil_features" / "Brasil_equalized.csv"
)


# ============================================================
# PREPROCESSED TABLE OUTPUTS
# ============================================================

# Preprocessed tables are also saved inside each dataset feature folder.
COLOMBIA_PREPROCESSED_TABLE = (
    COLOMBIA_DIR / "data_files_colombia_features" / "Colombia_preprocessed.csv"
)

BARCELONA_PREPROCESSED_TABLE = (
    BARCELONA_DIR / "data_files_barcelona_features" / "Barcelona_preprocessed.csv"
)

BRASIL_PREPROCESSED_TABLE = (
    BRASIL_DIR / "data_files_brasil_features" / "Brasil_preprocessed.csv"
)


# ============================================================
# GLOBAL TABLE OUTPUTS
# ============================================================

# Canonical data dictionary documenting the final common schema.
CANONICAL_DICTIONARY_TABLE = TABLES_OUTPUT_DIR / "canonical_dictionary.csv"

# Quality control report.
QUALITY_REPORT_TABLE = TABLES_OUTPUT_DIR / "quality_report.csv"

# Detailed report of the specific values flagged during quality control.
FLAGGED_VALUES_REPORT_TABLE = TABLES_OUTPUT_DIR / "qc_flagged_values.csv"

# Private mapping between original IDs and pseudonymous IDs.
PSEUDO_ID_MAPPING_PRIVATE_TABLE = PRIVATE_OUTPUT_DIR / "pseudo_id_mapping_private.csv"


# ============================================================
# VERIFICATION OUTPUTS
# ============================================================

VERIFICATION_DIR = OUTPUT_DIR / "verification"

CALIBRATION_REFERENCES_TABLE = VERIFICATION_DIR / "calibration_references.csv"
VERIFICATION_SUMMARY_TABLE = VERIFICATION_DIR / "verification_summary.csv"

# Subcarpetas de verificacion (una por bloque, para tener las imagenes ordenadas).
# - experiment_base: imagenes anotadas del Experimento 1 (persona, cabeza-pies, cubos).
# - horizon:         imagenes anotadas del metodo del horizonte (Experimento 2).
# - cube_detection:  SOLO los cubos con sus aristas marcadas (sin cabeza ni pies).
EXPERIMENT_BASE_DIR = VERIFICATION_DIR / "experiment_base"
HORIZON_DIR = VERIFICATION_DIR / "horizon"
CUBE_DETECTION_DIR = VERIFICATION_DIR / "cube_detection"

# Carpeta de la comparativa final (a nivel de outputs/):
# el CSV que junta los metodos y una carpeta de imagenes por sujeto/perspectiva.
COMPARATIVA_DIR = OUTPUT_DIR / "comparativa_final"
COMPARATIVA_CSV = COMPARATIVA_DIR / "comparativa_final.csv"
COMPARATIVA_MAE_CSV = COMPARATIVA_DIR / "comparativa_metodos_mae.csv"


# ============================================================
# PSEUDONYMIZATION SETTINGS
# ============================================================

# Salt used to generate reproducible pseudonymous identifiers.
# This value should be kept private and will not be included in the thesis document.
PSEUDONYMIZATION_SALT = "TurnerTFG_2026_private_salt"


# ============================================================
# IMAGE METRIC CALIBRATION SETTINGS
# ============================================================

# ArUco dictionary used by the cube markers.
# Used for Colombia and Brazil images.
ARUCO_DICT = cv2.aruco.DICT_4X4_50

# Physical edge of the cube in centimeters.
# Not used in the current calibration because the scale is computed from
# the visible ArUco marker side.
CUBE_EDGE_CM = None

# Physical size of one ArUco marker side in centimeters.
# According to the acquisition protocol, Colombia images include
# two cubes with 10 cm x 10 cm ArUco markers.
# This value is used to convert ArUco side length from pixels to centimeters.
# NOTE: confirm that the Brazil cubes also use 10 cm markers; if the physical
# size differs, this constant (or a per-site override) must be adjusted before
# running the Brazil height experiments.
ARUCO_MARKER_SIZE_CM = 10.0

# Altura de la camara respecto al suelo (cm). La camara estuvo nivelada y fija a
# 1 m para todas las fotos de Colombia. Se usa en el metodo del horizonte.
CAMERA_HEIGHT_CM = 100.0

# Physical diameter of the circular anatomical markers used in Barcelona.
# 14 mm = 1.4 cm.
CIRCULAR_MARKER_DIAMETER_CM = 1.4

# Calibration method assigned to each site.
CALIBRATION_METHOD_BY_SITE = {
    "CO": "aruco_marker",
    "BR": "aruco_marker",
    "ES": "circular_marker",
}

# Minimum number of references expected in a valid image.
# These are initial QC thresholds and can be adjusted later.
MIN_ARUCO_MARKERS = 1
MIN_CIRCULAR_MARKERS = 1

# Output calibration table.
IMAGE_CALIBRATION_TABLE = TABLES_OUTPUT_DIR / "image_calibration.csv"

# Debug folder for calibration visual checks.
DEBUG_IMAGE_CALIBRATION_DIR = OUTPUT_DIR / "debug_image_calibration"


# ============================================================
# IMAGE SETTINGS
# ============================================================

# Expected anatomical views for each subject in each dataset.
VIEWS = {
    "CO": ["front", "back", "left", "right"],
    "ES": ["front", "back", "left", "right"],
    "BR": ["front", "back", "left", "right"],
}

# Image extensions accepted by the pipeline.
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg"]

# Keyword used to identify Barcelona images with anatomical markers.
MARKER_KEYWORD = "markers"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_output_folders():
    """
    Purpose:
        Create the output folders needed by the preprocessing pipeline.

    Input:
        None.

    Output:
        None.
        The folders are created in the project directory if they do not exist.
    """

    OUTPUT_DIR.mkdir(exist_ok=True)
    TABLES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PRIVATE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VERIFICATION_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    HORIZON_DIR.mkdir(parents=True, exist_ok=True)
    CUBE_DETECTION_DIR.mkdir(parents=True, exist_ok=True)
    COMPARATIVA_DIR.mkdir(parents=True, exist_ok=True)
"""
lateral_skeletal_height.py

Experimento 3: estimación de altura en vistas laterales mediante fusión
left/right a nivel de sujeto y regresión Ridge interpretable.

Motivación
----------
Las vistas laterales presentan una escala métrica menos estable que front/back.
Además, left y right son dos observaciones del mismo sujeto y no deben tratarse
como muestras totalmente independientes.

Este experimento sigue una estrategia más robusta:

    1. Extrae, para cada imagen lateral:
       - altura corporal vertical en píxeles;
       - estimación métrica inicial:
             initial_height_cm = body_height_px * scale_cm_per_pixel
       - proporciones anatómicas:
             segmento_px / body_height_px

    2. Fusiona left y right en una única fila por sujeto:
       - media de la altura inicial;
       - diferencia absoluta left-right de la altura inicial;
       - media de ratios anatómicos robustos.

    3. Entrena un modelo Ridge con pocas variables:
       - initial_height_mean_cm
       - initial_height_lr_diff_cm
       - shoulder_hip_ratio_mean
       - hip_knee_ratio_mean
       - knee_ankle_ratio_mean

No se utilizan como variables principales:
    - head_shoulder_ratio, por su sensibilidad al pelo y a la cabeza lateral;
    - ankle_sole_ratio, por ser un segmento corto y muy sensible a pocos píxeles.

Métodos comparados
------------------
A. baseline_mean
   Predice la altura media de los sujetos de entrenamiento.

B. initial_height_subject_mean_raw
   Utiliza directamente la media de left/right de la altura inicial.

C. calibrated_initial_subject_mean
   Aprende:
       altura = a * initial_height_mean_cm + b

D. ridge_subject_fusion
   Combina la media métrica, la discrepancia left/right y tres ratios robustos.

Validación
----------
La validación se realiza a nivel de sujeto mediante KFold, porque tras la fusión
existe exactamente una fila por sujeto.

La selección del parámetro alpha de Ridge se hace mediante validación cruzada
interna únicamente con los sujetos del entrenamiento de cada fold externo.

Este experimento es exploratorio y no debe interpretarse como una herramienta
clínica o diagnóstica.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import VERIFICATION_DIR
from src.image_calibration import (
    estimate_body_height_pixels,
    get_manual_height_for_subject,
    get_pose_keypoints,
)


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

LATERAL_VIEWS = ("left", "right")

MIN_KEYPOINT_CONF = 0.25

N_OUTER_SPLITS = 5

RIDGE_ALPHAS = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]

RANDOM_STATE = 42

LATERAL_OUTPUT_DIR = (
    VERIFICATION_DIR
    / "lateral_skeletal_height"
)


# ============================================================
# ÍNDICES COCO DE YOLO-POSE
# ============================================================

KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6

KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12

KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 14

KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16


BODY_SIDE_INDICES = {
    "left": {
        "shoulder": KP_LEFT_SHOULDER,
        "hip": KP_LEFT_HIP,
        "knee": KP_LEFT_KNEE,
        "ankle": KP_LEFT_ANKLE,
    },
    "right": {
        "shoulder": KP_RIGHT_SHOULDER,
        "hip": KP_RIGHT_HIP,
        "knee": KP_RIGHT_KNEE,
        "ankle": KP_RIGHT_ANKLE,
    },
}


# ============================================================
# VARIABLES UTILIZADAS
# ============================================================

IMAGE_RATIO_FEATURES = [
    "shoulder_hip_ratio",
    "hip_knee_ratio",
    "knee_ankle_ratio",
]

SUBJECT_FUSION_FEATURES = [
    "initial_height_mean_cm",
    "initial_height_lr_diff_cm",
    "shoulder_hip_ratio_mean",
    "hip_knee_ratio_mean",
    "knee_ankle_ratio_mean",
]


# ============================================================
# FUNCIONES BÁSICAS
# ============================================================

def euclidean_distance(
    point_a: np.ndarray,
    point_b: np.ndarray,
) -> float:
    """
    Calcula la distancia euclídea entre dos puntos 2D.
    """

    point_a = np.asarray(
        point_a,
        dtype=float,
    )

    point_b = np.asarray(
        point_b,
        dtype=float,
    )

    return float(
        np.linalg.norm(
            point_a - point_b
        )
    )


def safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    """
    Calcula un cociente evitando divisiones entre cero.
    """

    if (
        not np.isfinite(numerator)
        or not np.isfinite(denominator)
        or abs(denominator) < 1e-9
    ):
        return np.nan

    return float(
        numerator / denominator
    )


def valid_keypoint(
    keypoints: np.ndarray,
    index: int,
    min_confidence: float = MIN_KEYPOINT_CONF,
) -> bool:
    """
    Comprueba que un keypoint exista y tenga confianza suficiente.
    """

    if keypoints is None:
        return False

    if index >= len(keypoints):
        return False

    point = keypoints[index]

    if len(point) < 3:
        return False

    x, y, confidence = point

    return bool(
        np.isfinite(x)
        and np.isfinite(y)
        and np.isfinite(confidence)
        and confidence >= min_confidence
    )


def get_keypoint_xy(
    keypoints: np.ndarray,
    index: int,
) -> np.ndarray:
    """
    Devuelve las coordenadas x, y de un keypoint.
    """

    return np.asarray(
        keypoints[index, :2],
        dtype=float,
    )


def normalise_path(
    path_value,
) -> str:
    """
    Normaliza una ruta para poder compararla con rutas guardadas en CSV.
    """

    try:
        return str(
            Path(path_value)
            .expanduser()
            .resolve()
        )
    except Exception:
        return str(path_value)


# ============================================================
# SELECCIÓN DEL LADO CORPORAL MÁS VISIBLE
# ============================================================

def score_body_side(
    keypoints: np.ndarray,
    side: str,
) -> tuple[int, float, float]:
    """
    Puntúa la cadena hombro-cadera-rodilla-tobillo.

    Se prioriza:
        1. número de puntos válidos;
        2. confianza mínima;
        3. confianza media.
    """

    confidences = []

    for index in BODY_SIDE_INDICES[side].values():
        if valid_keypoint(
            keypoints,
            index,
        ):
            confidences.append(
                float(
                    keypoints[index, 2]
                )
            )

    if not confidences:
        return 0, 0.0, 0.0

    return (
        len(confidences),
        float(np.min(confidences)),
        float(np.mean(confidences)),
    )


def select_visible_body_side(
    keypoints: np.ndarray,
) -> Optional[str]:
    """
    Selecciona la cadena corporal lateral mejor detectada.
    """

    left_score = score_body_side(
        keypoints,
        "left",
    )

    right_score = score_body_side(
        keypoints,
        "right",
    )

    if left_score > right_score:
        selected_side = "left"
        selected_score = left_score
    else:
        selected_side = "right"
        selected_score = right_score

    if selected_score[0] < 4:
        return None

    return selected_side


# ============================================================
# SELECCIÓN DE ESCALA CM/PÍXEL
# ============================================================

def select_scale_from_references(
    references_df: Optional[pd.DataFrame],
    image_path: str,
    subject_id: str,
    view: str,
    feet_y: float,
) -> tuple[float, str]:
    """
    Busca la escala ArUco más cercana verticalmente a los pies.

    Si no existe center_y_px, utiliza la mediana de las referencias válidas.
    """

    if references_df is None or references_df.empty:
        return np.nan, "no_references_table"

    required_columns = {
        "subject_id",
        "view",
        "reference_scale_cm_per_pixel",
    }

    if not required_columns.issubset(
        references_df.columns
    ):
        return np.nan, "missing_reference_columns"

    refs = references_df.copy()

    refs = refs[
        refs["subject_id"].astype(str)
        == str(subject_id)
    ]

    refs = refs[
        refs["view"].astype(str).str.lower()
        == str(view).lower()
    ]

    if (
        not refs.empty
        and "image_path" in refs.columns
    ):
        target_path = normalise_path(
            image_path
        )

        normalised_paths = refs[
            "image_path"
        ].map(normalise_path)

        exact_refs = refs[
            normalised_paths == target_path
        ]

        if not exact_refs.empty:
            refs = exact_refs

    refs = refs.dropna(
        subset=[
            "reference_scale_cm_per_pixel",
        ]
    ).copy()

    refs = refs[
        np.isfinite(
            refs[
                "reference_scale_cm_per_pixel"
            ]
        )
    ]

    if refs.empty:
        return np.nan, "no_valid_reference"

    if (
        "center_y_px" in refs.columns
        and np.isfinite(feet_y)
    ):
        valid_center = refs[
            np.isfinite(
                refs["center_y_px"]
            )
        ].copy()

        if not valid_center.empty:
            distances = (
                valid_center["center_y_px"]
                - feet_y
            ).abs()

            best_index = distances.idxmin()

            scale = float(
                valid_center.loc[
                    best_index,
                    "reference_scale_cm_per_pixel",
                ]
            )

            return (
                scale,
                "closest_reference_to_feet",
            )

    scale = float(
        refs[
            "reference_scale_cm_per_pixel"
        ].median()
    )

    return (
        scale,
        "median_reference_scale",
    )


def select_scale_from_calibration(
    calibration_df: Optional[pd.DataFrame],
    image_path: str,
    subject_id: str,
    view: str,
) -> tuple[float, str]:
    """
    Recupera como respaldo la escala general de image_calibration.csv.
    """

    if calibration_df is None or calibration_df.empty:
        return np.nan, "no_calibration_table"

    required_columns = {
        "subject_id",
        "view",
        "scale_cm_per_pixel",
    }

    if not required_columns.issubset(
        calibration_df.columns
    ):
        return np.nan, "missing_calibration_columns"

    rows = calibration_df.copy()

    rows = rows[
        rows["subject_id"].astype(str)
        == str(subject_id)
    ]

    rows = rows[
        rows["view"].astype(str).str.lower()
        == str(view).lower()
    ]

    if (
        not rows.empty
        and "image_path" in rows.columns
    ):
        target_path = normalise_path(
            image_path
        )

        normalised_paths = rows[
            "image_path"
        ].map(normalise_path)

        exact_rows = rows[
            normalised_paths == target_path
        ]

        if not exact_rows.empty:
            rows = exact_rows

    rows = rows.dropna(
        subset=[
            "scale_cm_per_pixel",
        ]
    ).copy()

    rows = rows[
        np.isfinite(
            rows["scale_cm_per_pixel"]
        )
    ]

    if rows.empty:
        return np.nan, "no_valid_calibration_scale"

    scale = float(
        rows[
            "scale_cm_per_pixel"
        ].median()
    )

    return (
        scale,
        "image_calibration_scale",
    )


def get_image_scale(
    references_df: Optional[pd.DataFrame],
    calibration_df: Optional[pd.DataFrame],
    image_path: str,
    subject_id: str,
    view: str,
    feet_y: float,
) -> tuple[float, str]:
    """
    Obtiene la escala con la misma regla para todas las imágenes.
    """

    scale, source = select_scale_from_references(
        references_df=references_df,
        image_path=image_path,
        subject_id=subject_id,
        view=view,
        feet_y=feet_y,
    )

    if np.isfinite(scale) and scale > 0:
        return scale, source

    return select_scale_from_calibration(
        calibration_df=calibration_df,
        image_path=image_path,
        subject_id=subject_id,
        view=view,
    )


# ============================================================
# EXTRACCIÓN DE FEATURES POR IMAGEN
# ============================================================

def extract_lateral_image_features(
    image: np.ndarray,
    image_path: str,
    subject_id: str,
    view: str,
    calibration_df: Optional[pd.DataFrame],
    references_df: Optional[pd.DataFrame],
) -> dict:
    """
    Extrae la medida métrica inicial y tres ratios robustos de una imagen.
    """

    row = {
        "subject_id": str(subject_id),
        "view": str(view).lower(),
        "image_path": str(image_path),
        "status": "ok",
        "pose_side": "",
        "scale_cm_per_pixel": np.nan,
        "scale_source": "",
        "body_height_px": np.nan,
        "initial_height_cm": np.nan,
        "shoulder_hip_px": np.nan,
        "hip_knee_px": np.nan,
        "knee_ankle_px": np.nan,
        "shoulder_hip_ratio": np.nan,
        "hip_knee_ratio": np.nan,
        "knee_ankle_ratio": np.nan,
        "min_chain_confidence": np.nan,
        "mean_chain_confidence": np.nan,
        "head_x_px": np.nan,
        "head_y_px": np.nan,
        "feet_x_px": np.nan,
        "feet_y_px": np.nan,
        "shoulder_x_px": np.nan,
        "shoulder_y_px": np.nan,
        "hip_x_px": np.nan,
        "hip_y_px": np.nan,
        "knee_x_px": np.nan,
        "knee_y_px": np.nan,
        "ankle_x_px": np.nan,
        "ankle_y_px": np.nan,
    }

    if image is None:
        row["status"] = "image_read_error"
        return row

    view = str(view).lower()

    if view not in LATERAL_VIEWS:
        row["status"] = "not_lateral_view"
        return row

    landmarks, _ = estimate_body_height_pixels(
        image,
        view=view,
    )

    landmark_status = str(
        landmarks.get(
            "landmark_status",
            "",
        )
    )

    if not landmark_status.startswith("ok"):
        row["status"] = (
            "body_landmarks_failed_"
            f"{landmark_status}"
        )
        return row

    head = np.array(
        [
            landmarks["head_x_px"],
            landmarks["head_y_px"],
        ],
        dtype=float,
    )

    feet = np.array(
        [
            landmarks["feet_x_px"],
            landmarks["feet_y_px"],
        ],
        dtype=float,
    )

    row["head_x_px"] = float(
        head[0]
    )
    row["head_y_px"] = float(
        head[1]
    )
    row["feet_x_px"] = float(
        feet[0]
    )
    row["feet_y_px"] = float(
        feet[1]
    )

    body_height_px = float(
        feet[1] - head[1]
    )

    if (
        not np.isfinite(body_height_px)
        or body_height_px <= 0
    ):
        row["status"] = "invalid_body_height_px"
        return row

    row["body_height_px"] = body_height_px

    keypoints = get_pose_keypoints(
        image
    )

    if keypoints is None:
        row["status"] = "pose_not_detected"
        return row

    selected_side = select_visible_body_side(
        keypoints
    )

    if selected_side is None:
        row["status"] = "incomplete_pose_chain"
        return row

    row["pose_side"] = selected_side

    side_indices = BODY_SIDE_INDICES[
        selected_side
    ]

    shoulder_index = side_indices[
        "shoulder"
    ]
    hip_index = side_indices[
        "hip"
    ]
    knee_index = side_indices[
        "knee"
    ]
    ankle_index = side_indices[
        "ankle"
    ]

    shoulder = get_keypoint_xy(
        keypoints,
        shoulder_index,
    )
    hip = get_keypoint_xy(
        keypoints,
        hip_index,
    )
    knee = get_keypoint_xy(
        keypoints,
        knee_index,
    )
    ankle = get_keypoint_xy(
        keypoints,
        ankle_index,
    )

    chain_confidences = np.array(
        [
            keypoints[shoulder_index, 2],
            keypoints[hip_index, 2],
            keypoints[knee_index, 2],
            keypoints[ankle_index, 2],
        ],
        dtype=float,
    )

    row["min_chain_confidence"] = float(
        np.min(chain_confidences)
    )

    row["mean_chain_confidence"] = float(
        np.mean(chain_confidences)
    )

    row["shoulder_x_px"] = float(
        shoulder[0]
    )
    row["shoulder_y_px"] = float(
        shoulder[1]
    )

    row["hip_x_px"] = float(
        hip[0]
    )
    row["hip_y_px"] = float(
        hip[1]
    )

    row["knee_x_px"] = float(
        knee[0]
    )
    row["knee_y_px"] = float(
        knee[1]
    )

    row["ankle_x_px"] = float(
        ankle[0]
    )
    row["ankle_y_px"] = float(
        ankle[1]
    )

    row["shoulder_hip_px"] = (
        euclidean_distance(
            shoulder,
            hip,
        )
    )

    row["hip_knee_px"] = (
        euclidean_distance(
            hip,
            knee,
        )
    )

    row["knee_ankle_px"] = (
        euclidean_distance(
            knee,
            ankle,
        )
    )

    row["shoulder_hip_ratio"] = safe_ratio(
        row["shoulder_hip_px"],
        body_height_px,
    )

    row["hip_knee_ratio"] = safe_ratio(
        row["hip_knee_px"],
        body_height_px,
    )

    row["knee_ankle_ratio"] = safe_ratio(
        row["knee_ankle_px"],
        body_height_px,
    )

    scale, scale_source = get_image_scale(
        references_df=references_df,
        calibration_df=calibration_df,
        image_path=image_path,
        subject_id=subject_id,
        view=view,
        feet_y=float(
            feet[1]
        ),
    )

    row["scale_cm_per_pixel"] = scale
    row["scale_source"] = scale_source

    if (
        not np.isfinite(scale)
        or scale <= 0
    ):
        row["status"] = "invalid_scale"
        return row

    row["initial_height_cm"] = float(
        body_height_px * scale
    )

    required_values = [
        row["initial_height_cm"],
        row["shoulder_hip_ratio"],
        row["hip_knee_ratio"],
        row["knee_ankle_ratio"],
    ]

    if not all(
        np.isfinite(value)
        for value in required_values
    ):
        row["status"] = "missing_features"

    return row


# ============================================================
# TABLA POR IMAGEN
# ============================================================

def build_lateral_image_feature_table(
    assets_by_subject: dict,
    manual_heights_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    references_df: pd.DataFrame,
    max_images: Optional[int] = None,
) -> pd.DataFrame:
    """
    Recorre las imágenes laterales y crea una fila por imagen.
    """

    rows = []
    processed_images = 0

    for _, assets in assets_by_subject.items():
        for asset in assets:
            view = str(
                asset.view
            ).lower()

            if view not in LATERAL_VIEWS:
                continue

            if (
                max_images is not None
                and processed_images >= max_images
            ):
                break

            image = cv2.imread(
                str(asset.path)
            )

            row = extract_lateral_image_features(
                image=image,
                image_path=str(asset.path),
                subject_id=asset.subject_id,
                view=view,
                calibration_df=calibration_df,
                references_df=references_df,
            )

            row["site"] = asset.site

            row["height_manual_cm"] = (
                get_manual_height_for_subject(
                    manual_heights_df,
                    asset.subject_id,
                )
            )

            rows.append(
                row
            )

            processed_images += 1

        if (
            max_images is not None
            and processed_images >= max_images
        ):
            break

    return pd.DataFrame(
        rows
    )


# ============================================================
# FUSIÓN LEFT/RIGHT POR SUJETO
# ============================================================

def absolute_left_right_difference(
    group: pd.DataFrame,
    column: str,
) -> float:
    """
    Calcula |left - right| cuando existen ambas vistas.

    Si falta una de las dos vistas, devuelve NaN para que el pipeline
    pueda imputarlo usando únicamente los datos de entrenamiento.
    """

    left_values = group.loc[
        group["view"] == "left",
        column,
    ].dropna()

    right_values = group.loc[
        group["view"] == "right",
        column,
    ].dropna()

    if (
        left_values.empty
        or right_values.empty
    ):
        return np.nan

    left_value = float(
        left_values.median()
    )

    right_value = float(
        right_values.median()
    )

    return float(
        abs(
            left_value
            - right_value
        )
    )


def build_subject_fusion_table(
    image_feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fusiona left/right en una única fila por sujeto.

    La media reduce ruido aleatorio entre vistas. La diferencia absoluta de
    altura inicial actúa como indicador de inconsistencia de la escala lateral.
    """

    valid = image_feature_df[
        image_feature_df["status"] == "ok"
    ].copy()

    valid = valid.dropna(
        subset=[
            "subject_id",
            "height_manual_cm",
            "initial_height_cm",
        ]
    )

    subject_rows = []

    for subject_id, group in valid.groupby(
        "subject_id"
    ):
        manual_values = group[
            "height_manual_cm"
        ].dropna()

        if manual_values.empty:
            continue

        row = {
            "subject_id": str(subject_id),
            "site": (
                group["site"].iloc[0]
                if "site" in group.columns
                else ""
            ),
            "height_manual_cm": float(
                manual_values.median()
            ),
            "n_lateral_images": int(
                len(group)
            ),
            "has_left": bool(
                (
                    group["view"]
                    == "left"
                ).any()
            ),
            "has_right": bool(
                (
                    group["view"]
                    == "right"
                ).any()
            ),
            "initial_height_mean_cm": float(
                group[
                    "initial_height_cm"
                ].mean()
            ),
            "initial_height_median_cm": float(
                group[
                    "initial_height_cm"
                ].median()
            ),
            "initial_height_std_cm": (
                float(
                    group[
                        "initial_height_cm"
                    ].std(
                        ddof=0
                    )
                )
                if len(group) > 1
                else 0.0
            ),
            "initial_height_lr_diff_cm": (
                absolute_left_right_difference(
                    group,
                    "initial_height_cm",
                )
            ),
            "shoulder_hip_ratio_mean": float(
                group[
                    "shoulder_hip_ratio"
                ].mean()
            ),
            "hip_knee_ratio_mean": float(
                group[
                    "hip_knee_ratio"
                ].mean()
            ),
            "knee_ankle_ratio_mean": float(
                group[
                    "knee_ankle_ratio"
                ].mean()
            ),
            "mean_chain_confidence": float(
                group[
                    "mean_chain_confidence"
                ].mean()
            ),
            "min_chain_confidence": float(
                group[
                    "min_chain_confidence"
                ].min()
            ),
        }

        subject_rows.append(
            row
        )

    return pd.DataFrame(
        subject_rows
    )


# ============================================================
# MÉTRICAS
# ============================================================

def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Calcula n, bias, MAE, RMSE y R².
    """

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    valid = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if len(y_true) == 0:
        return {
            "n": 0,
            "bias_cm": np.nan,
            "mae_cm": np.nan,
            "rmse_cm": np.nan,
            "r2": np.nan,
        }

    errors = (
        y_pred - y_true
    )

    return {
        "n": int(
            len(y_true)
        ),
        "bias_cm": float(
            np.mean(errors)
        ),
        "mae_cm": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "rmse_cm": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        ),
        "r2": (
            float(
                r2_score(
                    y_true,
                    y_pred,
                )
            )
            if len(y_true) >= 2
            else np.nan
        ),
    }


# ============================================================
# MODELOS
# ============================================================

def create_linear_pipeline() -> Pipeline:
    """
    Crea una regresión lineal con imputación por mediana.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "linear_regression",
                LinearRegression(),
            ),
        ]
    )


def create_ridge_pipeline(
    alpha: float = 1.0,
) -> Pipeline:
    """
    Crea un Pipeline con:
        1. imputación por mediana;
        2. estandarización;
        3. Ridge.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "ridge",
                Ridge(
                    alpha=alpha,
                ),
            ),
        ]
    )


def fit_ridge_with_inner_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[Pipeline, float]:
    """
    Selecciona alpha mediante validación cruzada interna.
    """

    n_train_subjects = len(
        X_train
    )

    n_inner_splits = min(
        4,
        n_train_subjects,
    )

    if n_inner_splits < 2:
        fallback_alpha = 1.0

        model = create_ridge_pipeline(
            alpha=fallback_alpha
        )

        model.fit(
            X_train,
            y_train,
        )

        return (
            model,
            fallback_alpha,
        )

    inner_cv = KFold(
        n_splits=n_inner_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = GridSearchCV(
        estimator=create_ridge_pipeline(),
        param_grid={
            "ridge__alpha": RIDGE_ALPHAS,
        },
        scoring="neg_mean_absolute_error",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
    )

    search.fit(
        X_train,
        y_train,
    )

    return (
        search.best_estimator_,
        float(
            search.best_params_[
                "ridge__alpha"
            ]
        ),
    )


# ============================================================
# VALIDACIÓN A NIVEL DE SUJETO
# ============================================================

def cross_validate_subject_models(
    subject_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evalúa los cuatro métodos con una fila por sujeto.
    """

    required_columns = {
        "subject_id",
        "height_manual_cm",
        *SUBJECT_FUSION_FEATURES,
    }

    missing_columns = (
        required_columns
        - set(
            subject_df.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Faltan columnas necesarias: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    data = subject_df.copy()

    data = data.dropna(
        subset=[
            "subject_id",
            "height_manual_cm",
            "initial_height_mean_cm",
        ]
    )

    data = data[
        np.isfinite(
            data["height_manual_cm"]
        )
    ]

    n_subjects = len(
        data
    )

    if n_subjects < 3:
        raise ValueError(
            "No hay suficientes sujetos para realizar validación."
        )

    n_outer_splits = min(
        N_OUTER_SPLITS,
        n_subjects,
    )

    outer_cv = KFold(
        n_splits=n_outer_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    X_initial = data[
        [
            "initial_height_mean_cm",
        ]
    ].copy()

    X_fusion = data[
        SUBJECT_FUSION_FEATURES
    ].copy()

    y = data[
        "height_manual_cm"
    ].astype(float)

    data["cv_fold"] = -1

    data["pred_baseline_mean_cm"] = np.nan
    data["pred_initial_subject_mean_raw_cm"] = np.nan
    data["pred_calibrated_initial_subject_mean_cm"] = np.nan
    data["pred_ridge_subject_fusion_cm"] = np.nan

    data["error_baseline_mean_cm"] = np.nan
    data["error_initial_subject_mean_raw_cm"] = np.nan
    data["error_calibrated_initial_subject_mean_cm"] = np.nan
    data["error_ridge_subject_fusion_cm"] = np.nan

    data["ridge_alpha"] = np.nan

    fold_rows = []

    for fold_number, (
        train_indices,
        test_indices,
    ) in enumerate(
        outer_cv.split(
            X_fusion,
            y,
        ),
        start=1,
    ):
        y_train = y.iloc[
            train_indices
        ]

        y_test = y.iloc[
            test_indices
        ]

        test_dataframe_indices = (
            data.index[
                test_indices
            ]
        )

        # A. Baseline media del train.
        train_mean = float(
            y_train.mean()
        )

        pred_baseline = np.full(
            shape=len(test_indices),
            fill_value=train_mean,
            dtype=float,
        )

        # B. Media left/right sin calibrar.
        pred_initial_raw = (
            X_initial.iloc[
                test_indices
            ][
                "initial_height_mean_cm"
            ]
            .to_numpy(
                dtype=float
            )
        )

        # C. Calibración lineal de la media left/right.
        initial_model = (
            create_linear_pipeline()
        )

        initial_model.fit(
            X_initial.iloc[
                train_indices
            ],
            y_train,
        )

        pred_initial_calibrated = (
            initial_model.predict(
                X_initial.iloc[
                    test_indices
                ]
            )
        )

        # D. Fusión Ridge a nivel de sujeto.
        (
            ridge_model,
            best_alpha,
        ) = fit_ridge_with_inner_cv(
            X_train=X_fusion.iloc[
                train_indices
            ],
            y_train=y_train,
        )

        pred_ridge_fusion = (
            ridge_model.predict(
                X_fusion.iloc[
                    test_indices
                ]
            )
        )

        data.loc[
            test_dataframe_indices,
            "cv_fold",
        ] = fold_number

        predictions = {
            "pred_baseline_mean_cm": pred_baseline,
            "pred_initial_subject_mean_raw_cm": pred_initial_raw,
            "pred_calibrated_initial_subject_mean_cm": pred_initial_calibrated,
            "pred_ridge_subject_fusion_cm": pred_ridge_fusion,
        }

        errors = {
            "error_baseline_mean_cm": (
                pred_baseline
                - y_test.to_numpy()
            ),
            "error_initial_subject_mean_raw_cm": (
                pred_initial_raw
                - y_test.to_numpy()
            ),
            "error_calibrated_initial_subject_mean_cm": (
                pred_initial_calibrated
                - y_test.to_numpy()
            ),
            "error_ridge_subject_fusion_cm": (
                pred_ridge_fusion
                - y_test.to_numpy()
            ),
        }

        for column, values in predictions.items():
            data.loc[
                test_dataframe_indices,
                column,
            ] = values

        for column, values in errors.items():
            data.loc[
                test_dataframe_indices,
                column,
            ] = values

        data.loc[
            test_dataframe_indices,
            "ridge_alpha",
        ] = best_alpha

        method_predictions = {
            "baseline_mean": pred_baseline,
            "initial_height_subject_mean_raw": pred_initial_raw,
            "calibrated_initial_subject_mean": pred_initial_calibrated,
            "ridge_subject_fusion": pred_ridge_fusion,
        }

        for (
            method_name,
            prediction,
        ) in method_predictions.items():
            metrics = (
                calculate_regression_metrics(
                    y_true=y_test.to_numpy(),
                    y_pred=prediction,
                )
            )

            fold_rows.append(
                {
                    "fold": fold_number,
                    "method": method_name,
                    "n_train_subjects": int(
                        len(train_indices)
                    ),
                    "n_test_subjects": int(
                        len(test_indices)
                    ),
                    "ridge_alpha": (
                        best_alpha
                        if method_name
                        == "ridge_subject_fusion"
                        else np.nan
                    ),
                    **metrics,
                }
            )

    return (
        data,
        pd.DataFrame(
            fold_rows
        ),
    )


# ============================================================
# RESUMEN GLOBAL
# ============================================================

def build_subject_model_summary(
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye el resumen final de rendimiento a nivel de sujeto.
    """

    methods = {
        "baseline_mean": (
            "pred_baseline_mean_cm"
        ),
        "initial_height_subject_mean_raw": (
            "pred_initial_subject_mean_raw_cm"
        ),
        "calibrated_initial_subject_mean": (
            "pred_calibrated_initial_subject_mean_cm"
        ),
        "ridge_subject_fusion": (
            "pred_ridge_subject_fusion_cm"
        ),
    }

    y_true = predictions_df[
        "height_manual_cm"
    ].to_numpy(
        dtype=float
    )

    rows = []

    for (
        method_name,
        prediction_column,
    ) in methods.items():
        metrics = (
            calculate_regression_metrics(
                y_true=y_true,
                y_pred=predictions_df[
                    prediction_column
                ].to_numpy(
                    dtype=float
                ),
            )
        )

        rows.append(
            {
                "group": "SUBJECT_LEVEL",
                "method": method_name,
                "n_subjects": int(
                    predictions_df[
                        "subject_id"
                    ].nunique()
                ),
                **metrics,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# COEFICIENTES DEL MODELO FINAL
# ============================================================

def fit_final_subject_fusion_model(
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ajusta Ridge con todos los sujetos solo para inspeccionar coeficientes.

    Las métricas válidas deben tomarse de la validación cruzada.
    """

    valid_alphas = predictions_df[
        "ridge_alpha"
    ].dropna()

    final_alpha = (
        float(
            valid_alphas.median()
        )
        if not valid_alphas.empty
        else 1.0
    )

    X = predictions_df[
        SUBJECT_FUSION_FEATURES
    ]

    y = predictions_df[
        "height_manual_cm"
    ].astype(float)

    model = create_ridge_pipeline(
        alpha=final_alpha
    )

    model.fit(
        X,
        y,
    )

    scaler = model.named_steps[
        "scaler"
    ]

    ridge = model.named_steps[
        "ridge"
    ]

    standardised_coefficients = np.asarray(
        ridge.coef_,
        dtype=float,
    )

    original_coefficients = (
        standardised_coefficients
        / scaler.scale_
    )

    original_intercept = float(
        ridge.intercept_
        - np.sum(
            standardised_coefficients
            * scaler.mean_
            / scaler.scale_
        )
    )

    rows = []

    for (
        feature,
        coefficient_standardised,
        coefficient_original,
    ) in zip(
        SUBJECT_FUSION_FEATURES,
        standardised_coefficients,
        original_coefficients,
    ):
        rows.append(
            {
                "model": "ridge_subject_fusion",
                "feature": feature,
                "alpha": final_alpha,
                "coefficient_standardised": float(
                    coefficient_standardised
                ),
                "coefficient_original_units": float(
                    coefficient_original
                ),
                "intercept_original_units": (
                    original_intercept
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# RUNNER PRINCIPAL
# ============================================================

def run_lateral_skeletal_experiment(
    assets_by_subject: dict,
    manual_heights_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    references_df: pd.DataFrame,
    max_images: Optional[int] = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Ejecuta el experimento completo.

    Genera:
        lateral_image_features.csv
        lateral_subject_features.csv
        lateral_subject_predictions.csv
        lateral_subject_fold_metrics.csv
        lateral_subject_summary.csv
        lateral_subject_ridge_coefficients.csv

    Devuelve:
        subject_predictions_df
        summary_df
    """

    LATERAL_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        "EXPERIMENTO 3: FUSIÓN LATERAL LEFT/RIGHT"
    )

    print(
        "Una fila por sujeto + Ridge reducido"
    )

    print(
        "============================================================"
    )

    # --------------------------------------------------------
    # 1. FEATURES POR IMAGEN
    # --------------------------------------------------------

    image_feature_df = (
        build_lateral_image_feature_table(
            assets_by_subject=assets_by_subject,
            manual_heights_df=manual_heights_df,
            calibration_df=calibration_df,
            references_df=references_df,
            max_images=max_images,
        )
    )

    image_features_path = (
        LATERAL_OUTPUT_DIR
        / "lateral_image_features.csv"
    )

    image_feature_df.to_csv(
        image_features_path,
        index=False,
    )

    print(
        f"\nImágenes laterales procesadas: {len(image_feature_df)}"
    )

    if not image_feature_df.empty:
        print(
            "\nEstados de extracción:"
        )

        print(
            image_feature_df[
                "status"
            ].value_counts(
                dropna=False
            ).to_string()
        )

    # --------------------------------------------------------
    # 2. FUSIÓN POR SUJETO
    # --------------------------------------------------------

    subject_feature_df = (
        build_subject_fusion_table(
            image_feature_df
        )
    )

    subject_features_path = (
        LATERAL_OUTPUT_DIR
        / "lateral_subject_features.csv"
    )

    subject_feature_df.to_csv(
        subject_features_path,
        index=False,
    )

    print(
        "\nDatos fusionados:"
    )

    print(
        f" - sujetos: {len(subject_feature_df)}"
    )

    if not subject_feature_df.empty:
        complete_pairs = (
            subject_feature_df[
                "has_left"
            ]
            & subject_feature_df[
                "has_right"
            ]
        ).sum()

        print(
            f" - sujetos con left y right: {complete_pairs}"
        )

    if len(subject_feature_df) < 3:
        raise ValueError(
            "No hay suficientes sujetos válidos para "
            "entrenar y evaluar los modelos."
        )

    # --------------------------------------------------------
    # 3. VALIDACIÓN CRUZADA
    # --------------------------------------------------------

    (
        subject_predictions_df,
        fold_metrics_df,
    ) = cross_validate_subject_models(
        subject_feature_df
    )

    predictions_path = (
        LATERAL_OUTPUT_DIR
        / "lateral_subject_predictions.csv"
    )

    subject_predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    fold_metrics_path = (
        LATERAL_OUTPUT_DIR
        / "lateral_subject_fold_metrics.csv"
    )

    fold_metrics_df.to_csv(
        fold_metrics_path,
        index=False,
    )

    # --------------------------------------------------------
    # 4. RESUMEN
    # --------------------------------------------------------

    summary_df = build_subject_model_summary(
        subject_predictions_df
    )

    summary_path = (
        LATERAL_OUTPUT_DIR
        / "lateral_subject_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # 5. COEFICIENTES
    # --------------------------------------------------------

    coefficients_df = (
        fit_final_subject_fusion_model(
            subject_predictions_df
        )
    )

    coefficients_path = (
        LATERAL_OUTPUT_DIR
        / "lateral_subject_ridge_coefficients.csv"
    )

    coefficients_df.to_csv(
        coefficients_path,
        index=False,
    )

    # --------------------------------------------------------
    # 6. RESULTADOS POR TERMINAL
    # --------------------------------------------------------

    print(
        "\nResultados a nivel de sujeto:"
    )

    print(
        summary_df.round(
            3
        ).to_string(
            index=False
        )
    )

    best_row = summary_df.loc[
        summary_df[
            "mae_cm"
        ].idxmin()
    ]

    print(
        "\nMejor método:"
    )

    print(
        f" - método: {best_row['method']}"
    )

    print(
        f" - MAE: {best_row['mae_cm']:.2f} cm"
    )

    print(
        f" - bias: {best_row['bias_cm']:+.2f} cm"
    )

    print(
        f" - RMSE: {best_row['rmse_cm']:.2f} cm"
    )

    print(
        f" - R²: {best_row['r2']:.3f}"
    )

    print(
        "\nArchivos guardados:"
    )

    for output_path in [
        image_features_path,
        subject_features_path,
        predictions_path,
        fold_metrics_path,
        summary_path,
        coefficients_path,
    ]:
        print(
            f" - {output_path}"
        )

    return (
        subject_predictions_df,
        summary_df,
    )

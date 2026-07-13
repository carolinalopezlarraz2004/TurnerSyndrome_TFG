"""
lateral_horizon_diagnostic.py

Experimento diagnóstico: estimación de altura lateral utilizando el horizonte
calculado en cada imagen.

Objetivo
--------
Comprobar si el mal resultado de las vistas laterales se debe a utilizar un
horizonte compartido procedente de front/back.

La altura se recalcula mediante:

    H = camera_height_cm * (y_feet - y_head) / (y_feet - y_horizon)

Se comparan:

    1. per_image_cube_horizon
       Horizonte estimado directamente en cada imagen mediante el cubo.

    2. shared_subject_horizon
       Horizonte compartido utilizado anteriormente.

También se fusionan left y right por sujeto para comprobar si el promedio de
ambas vistas reduce el error.

Este módulo parte del CSV horizon_estimates.csv generado por el experimento del
horizonte, porque ese archivo ya contiene los puntos de cabeza, pies y ambos
horizontes. Así se aísla el efecto del horizonte sin repetir la detección de
landmarks ni introducir cambios adicionales.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.config import VERIFICATION_DIR


# ============================================================
# CONFIGURACIÓN
# ============================================================

CAMERA_HEIGHT_CM = 100.0

LATERAL_VIEWS = (
    "left",
    "right",
)

OUTPUT_DIR = (
    VERIFICATION_DIR
    / "lateral_horizon_diagnostic"
)


# ============================================================
# FUNCIONES BÁSICAS
# ============================================================

def estimate_height_from_horizon(
    y_head: float,
    y_feet: float,
    y_horizon: float,
    camera_height_cm: float = CAMERA_HEIGHT_CM,
) -> float:
    """
    Estima la altura de una persona mediante metrología de una sola vista.

    Fórmula:
        H = h_camera * body_height_px / ground_to_horizon_px

    donde:
        body_height_px = y_feet - y_head
        ground_to_horizon_px = y_feet - y_horizon
    """

    values = np.asarray(
        [
            y_head,
            y_feet,
            y_horizon,
            camera_height_cm,
        ],
        dtype=float,
    )

    if not np.all(
        np.isfinite(values)
    ):
        return np.nan

    body_height_px = float(
        y_feet - y_head
    )

    ground_to_horizon_px = float(
        y_feet - y_horizon
    )

    if (
        body_height_px <= 0
        or ground_to_horizon_px <= 0
        or camera_height_cm <= 0
    ):
        return np.nan

    return float(
        camera_height_cm
        * body_height_px
        / ground_to_horizon_px
    )


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Calcula las métricas principales de regresión.
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
# CARGA Y VALIDACIÓN
# ============================================================

def load_horizon_estimates(
    horizon_estimates: Union[
        str,
        Path,
        pd.DataFrame,
    ],
) -> pd.DataFrame:
    """
    Carga horizon_estimates.csv o acepta directamente un DataFrame.
    """

    if isinstance(
        horizon_estimates,
        pd.DataFrame,
    ):
        return horizon_estimates.copy()

    path = Path(
        horizon_estimates
    )

    if not path.exists():
        raise FileNotFoundError(
            f"No se ha encontrado el archivo: {path}"
        )

    return pd.read_csv(
        path
    )


def validate_input_columns(
    data: pd.DataFrame,
) -> None:
    """
    Comprueba que existan las columnas necesarias.
    """

    required_columns = {
        "status",
        "subject_id",
        "view",
        "image_path",
        "height_manual_cm",
        "y_head",
        "y_feet",
        "y_horizon_cube",
        "y_horizon_ref",
    }

    missing_columns = (
        required_columns
        - set(
            data.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Faltan columnas necesarias en horizon_estimates.csv: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )


# ============================================================
# PREDICCIONES POR IMAGEN
# ============================================================

def build_lateral_horizon_predictions(
    horizon_df: pd.DataFrame,
    camera_height_cm: float = CAMERA_HEIGHT_CM,
) -> pd.DataFrame:
    """
    Recalcula la altura lateral utilizando:

        - horizonte propio de cada imagen;
        - horizonte compartido/de referencia.

    Solo se utilizan vistas left y right con status ok.
    """

    validate_input_columns(
        horizon_df
    )

    data = horizon_df.copy()

    data["view"] = (
        data["view"]
        .astype(str)
        .str.lower()
    )

    data = data[
        data["view"].isin(
            LATERAL_VIEWS
        )
    ].copy()

    data = data[
        data["status"].astype(str)
        == "ok"
    ].copy()

    numeric_columns = [
        "height_manual_cm",
        "y_head",
        "y_feet",
        "y_horizon_cube",
        "y_horizon_ref",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    # Altura corporal medida en la imagen.
    data["body_height_px"] = (
        data["y_feet"]
        - data["y_head"]
    )

    # Distancia entre el punto de apoyo y cada horizonte.
    data["feet_to_cube_horizon_px"] = (
        data["y_feet"]
        - data["y_horizon_cube"]
    )

    data["feet_to_shared_horizon_px"] = (
        data["y_feet"]
        - data["y_horizon_ref"]
    )

    # Diferencia entre ambos horizontes.
    data["horizon_delta_px"] = (
        data["y_horizon_cube"]
        - data["y_horizon_ref"]
    )

    # Recalcular explícitamente las dos predicciones.
    data["pred_per_image_horizon_cm"] = data.apply(
        lambda row: estimate_height_from_horizon(
            y_head=row["y_head"],
            y_feet=row["y_feet"],
            y_horizon=row["y_horizon_cube"],
            camera_height_cm=camera_height_cm,
        ),
        axis=1,
    )

    data["pred_shared_horizon_cm"] = data.apply(
        lambda row: estimate_height_from_horizon(
            y_head=row["y_head"],
            y_feet=row["y_feet"],
            y_horizon=row["y_horizon_ref"],
            camera_height_cm=camera_height_cm,
        ),
        axis=1,
    )

    data["error_per_image_horizon_cm"] = (
        data["pred_per_image_horizon_cm"]
        - data["height_manual_cm"]
    )

    data["error_shared_horizon_cm"] = (
        data["pred_shared_horizon_cm"]
        - data["height_manual_cm"]
    )

    data["abs_error_per_image_horizon_cm"] = (
        data[
            "error_per_image_horizon_cm"
        ].abs()
    )

    data["abs_error_shared_horizon_cm"] = (
        data[
            "error_shared_horizon_cm"
        ].abs()
    )

    data["improvement_per_image_vs_shared_cm"] = (
        data[
            "abs_error_shared_horizon_cm"
        ]
        - data[
            "abs_error_per_image_horizon_cm"
        ]
    )

    return data


# ============================================================
# RESUMEN POR VISTA
# ============================================================

def build_view_summary(
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Resume el rendimiento global, left y right.
    """

    methods = {
        "per_image_cube_horizon": (
            "pred_per_image_horizon_cm"
        ),
        "shared_subject_horizon": (
            "pred_shared_horizon_cm"
        ),
    }

    groups = [
        (
            "LATERAL_GLOBAL",
            predictions_df,
        ),
    ]

    for view, group in predictions_df.groupby(
        "view"
    ):
        groups.append(
            (
                str(view),
                group,
            )
        )

    rows = []

    for group_name, group_df in groups:
        y_true = group_df[
            "height_manual_cm"
        ].to_numpy(
            dtype=float
        )

        for method_name, prediction_column in (
            methods.items()
        ):
            metrics = calculate_metrics(
                y_true=y_true,
                y_pred=group_df[
                    prediction_column
                ].to_numpy(
                    dtype=float
                ),
            )

            rows.append(
                {
                    "group": group_name,
                    "method": method_name,
                    "n_subjects": int(
                        group_df[
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
# FUSIÓN LEFT/RIGHT POR SUJETO
# ============================================================

def build_subject_predictions(
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fusiona las predicciones left/right de cada sujeto.

    Se guardan media y mediana. Con dos vistas ambas son equivalentes cuando
    solo existe un valor por vista, pero se conservan las dos para dejar el
    procedimiento explícito.
    """

    rows = []

    for subject_id, group in predictions_df.groupby(
        "subject_id"
    ):
        manual_values = (
            group[
                "height_manual_cm"
            ]
            .dropna()
            .to_numpy(
                dtype=float
            )
        )

        if len(manual_values) == 0:
            continue

        per_image_predictions = (
            group[
                "pred_per_image_horizon_cm"
            ]
            .dropna()
            .to_numpy(
                dtype=float
            )
        )

        shared_predictions = (
            group[
                "pred_shared_horizon_cm"
            ]
            .dropna()
            .to_numpy(
                dtype=float
            )
        )

        row = {
            "subject_id": str(
                subject_id
            ),
            "height_manual_cm": float(
                np.median(
                    manual_values
                )
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
            "pred_per_image_horizon_mean_cm": (
                float(
                    np.mean(
                        per_image_predictions
                    )
                )
                if len(
                    per_image_predictions
                ) > 0
                else np.nan
            ),
            "pred_per_image_horizon_median_cm": (
                float(
                    np.median(
                        per_image_predictions
                    )
                )
                if len(
                    per_image_predictions
                ) > 0
                else np.nan
            ),
            "pred_shared_horizon_mean_cm": (
                float(
                    np.mean(
                        shared_predictions
                    )
                )
                if len(
                    shared_predictions
                ) > 0
                else np.nan
            ),
            "pred_shared_horizon_median_cm": (
                float(
                    np.median(
                        shared_predictions
                    )
                )
                if len(
                    shared_predictions
                ) > 0
                else np.nan
            ),
            "mean_cube_horizon_px": float(
                group[
                    "y_horizon_cube"
                ].mean()
            ),
            "mean_shared_horizon_px": float(
                group[
                    "y_horizon_ref"
                ].mean()
            ),
            "mean_horizon_delta_px": float(
                group[
                    "horizon_delta_px"
                ].mean()
            ),
            "left_right_prediction_difference_cm": (
                float(
                    np.ptp(
                        per_image_predictions
                    )
                )
                if len(
                    per_image_predictions
                ) >= 2
                else np.nan
            ),
        }

        row[
            "error_per_image_horizon_mean_cm"
        ] = (
            row[
                "pred_per_image_horizon_mean_cm"
            ]
            - row[
                "height_manual_cm"
            ]
        )

        row[
            "error_shared_horizon_mean_cm"
        ] = (
            row[
                "pred_shared_horizon_mean_cm"
            ]
            - row[
                "height_manual_cm"
            ]
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def build_subject_summary(
    subject_predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula las métricas después de fusionar left/right.
    """

    methods = {
        "per_image_horizon_left_right_mean": (
            "pred_per_image_horizon_mean_cm"
        ),
        "per_image_horizon_left_right_median": (
            "pred_per_image_horizon_median_cm"
        ),
        "shared_horizon_left_right_mean": (
            "pred_shared_horizon_mean_cm"
        ),
        "shared_horizon_left_right_median": (
            "pred_shared_horizon_median_cm"
        ),
    }

    y_true = subject_predictions_df[
        "height_manual_cm"
    ].to_numpy(
        dtype=float
    )

    rows = []

    for method_name, prediction_column in (
        methods.items()
    ):
        metrics = calculate_metrics(
            y_true=y_true,
            y_pred=subject_predictions_df[
                prediction_column
            ].to_numpy(
                dtype=float
            ),
        )

        rows.append(
            {
                "group": "SUBJECT_LEVEL",
                "method": method_name,
                "n_subjects": int(
                    subject_predictions_df[
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
# DIAGNÓSTICO DEL HORIZONTE
# ============================================================

def build_horizon_diagnostic_summary(
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Resume cuánto cambia el horizonte propio respecto al compartido.
    """

    rows = []

    groups = [
        (
            "LATERAL_GLOBAL",
            predictions_df,
        ),
    ]

    for view, group in predictions_df.groupby(
        "view"
    ):
        groups.append(
            (
                str(view),
                group,
            )
        )

    for group_name, group_df in groups:
        delta = group_df[
            "horizon_delta_px"
        ].dropna()

        improvement = group_df[
            "improvement_per_image_vs_shared_cm"
        ].dropna()

        rows.append(
            {
                "group": group_name,
                "n_images": int(
                    len(group_df)
                ),
                "mean_horizon_delta_px": (
                    float(
                        delta.mean()
                    )
                    if not delta.empty
                    else np.nan
                ),
                "median_horizon_delta_px": (
                    float(
                        delta.median()
                    )
                    if not delta.empty
                    else np.nan
                ),
                "std_horizon_delta_px": (
                    float(
                        delta.std()
                    )
                    if len(delta) >= 2
                    else np.nan
                ),
                "min_horizon_delta_px": (
                    float(
                        delta.min()
                    )
                    if not delta.empty
                    else np.nan
                ),
                "max_horizon_delta_px": (
                    float(
                        delta.max()
                    )
                    if not delta.empty
                    else np.nan
                ),
                "mean_error_improvement_cm": (
                    float(
                        improvement.mean()
                    )
                    if not improvement.empty
                    else np.nan
                ),
                "median_error_improvement_cm": (
                    float(
                        improvement.median()
                    )
                    if not improvement.empty
                    else np.nan
                ),
                "fraction_images_improved": (
                    float(
                        (
                            improvement > 0
                        ).mean()
                    )
                    if not improvement.empty
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# RUNNER PRINCIPAL
# ============================================================

def run_lateral_horizon_diagnostic(
    horizon_estimates: Union[
        str,
        Path,
        pd.DataFrame,
    ],
    camera_height_cm: float = CAMERA_HEIGHT_CM,
    output_dir: Optional[
        Union[
            str,
            Path,
        ]
    ] = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Ejecuta el diagnóstico completo.

    Parámetros
    ----------
    horizon_estimates:
        Ruta a horizon_estimates.csv o DataFrame ya cargado.

    camera_height_cm:
        Altura física de la cámara. En este montaje es 100 cm.

    output_dir:
        Carpeta de salida opcional.

    Genera
    ------
        lateral_horizon_image_predictions.csv
        lateral_horizon_view_summary.csv
        lateral_horizon_subject_predictions.csv
        lateral_horizon_subject_summary.csv
        lateral_horizon_diagnostic_summary.csv

    Devuelve
    --------
        image_predictions_df
        view_summary_df
    """

    if (
        not np.isfinite(
            camera_height_cm
        )
        or camera_height_cm <= 0
    ):
        raise ValueError(
            "camera_height_cm debe ser un valor positivo."
        )

    final_output_dir = (
        Path(
            output_dir
        )
        if output_dir is not None
        else OUTPUT_DIR
    )

    final_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        "DIAGNÓSTICO DEL HORIZONTE LATERAL"
    )

    print(
        f"Altura de cámara: {camera_height_cm:.1f} cm"
    )

    print(
        "============================================================"
    )

    horizon_df = load_horizon_estimates(
        horizon_estimates
    )

    image_predictions_df = (
        build_lateral_horizon_predictions(
            horizon_df=horizon_df,
            camera_height_cm=camera_height_cm,
        )
    )

    if image_predictions_df.empty:
        raise ValueError(
            "No se han encontrado imágenes laterales válidas."
        )

    view_summary_df = build_view_summary(
        image_predictions_df
    )

    subject_predictions_df = (
        build_subject_predictions(
            image_predictions_df
        )
    )

    subject_summary_df = (
        build_subject_summary(
            subject_predictions_df
        )
    )

    diagnostic_summary_df = (
        build_horizon_diagnostic_summary(
            image_predictions_df
        )
    )

    image_predictions_path = (
        final_output_dir
        / "lateral_horizon_image_predictions.csv"
    )

    view_summary_path = (
        final_output_dir
        / "lateral_horizon_view_summary.csv"
    )

    subject_predictions_path = (
        final_output_dir
        / "lateral_horizon_subject_predictions.csv"
    )

    subject_summary_path = (
        final_output_dir
        / "lateral_horizon_subject_summary.csv"
    )

    diagnostic_summary_path = (
        final_output_dir
        / "lateral_horizon_diagnostic_summary.csv"
    )

    image_predictions_df.to_csv(
        image_predictions_path,
        index=False,
    )

    view_summary_df.to_csv(
        view_summary_path,
        index=False,
    )

    subject_predictions_df.to_csv(
        subject_predictions_path,
        index=False,
    )

    subject_summary_df.to_csv(
        subject_summary_path,
        index=False,
    )

    diagnostic_summary_df.to_csv(
        diagnostic_summary_path,
        index=False,
    )

    print(
        "\nResultados por imagen y vista:"
    )

    print(
        view_summary_df.round(
            3
        ).to_string(
            index=False
        )
    )

    print(
        "\nResultados fusionando left/right por sujeto:"
    )

    print(
        subject_summary_df.round(
            3
        ).to_string(
            index=False
        )
    )

    print(
        "\nDiagnóstico de desplazamiento del horizonte:"
    )

    print(
        diagnostic_summary_df.round(
            3
        ).to_string(
            index=False
        )
    )

    print(
        "\nArchivos guardados:"
    )

    for path in [
        image_predictions_path,
        view_summary_path,
        subject_predictions_path,
        subject_summary_path,
        diagnostic_summary_path,
    ]:
        print(
            f" - {path}"
        )

    return (
        image_predictions_df,
        view_summary_df,
    )

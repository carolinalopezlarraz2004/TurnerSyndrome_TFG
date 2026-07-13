"""
comparativa_final.py

Genera un CSV único por imagen que reúne:

    - Experimento 1:
        métodos basados en escala píxel-centímetro.

    - Experimento 2 original:
        método del horizonte con horizonte propio y compartido.

    - Experimento 2 corregido:
        estrategia híbrida por vista con fallbacks y
        leave-one-subject-out para el horizonte global.

Además, genera una tabla resumen de MAE por vista y método.

Salidas
-------
outputs/comparativa_final/comparativa_final.csv

outputs/comparativa_final/comparativa_metodos_mae.csv

Uso
---
Ejecutar después de haber corrido main.py:

    python comparativa_final.py
"""

import numpy as np
import pandas as pd

from src.config import (
    VERIFICATION_DIR,
    HORIZON_DIR,
    COMPARATIVA_DIR,
    COMPARATIVA_CSV,
    COMPARATIVA_MAE_CSV,
)


# ============================================================
# RUTAS DE ENTRADA
# ============================================================

E1 = (
    VERIFICATION_DIR
    / "height_estimates_experiment1.csv"
)

HZ = (
    HORIZON_DIR
    / "horizon_estimates.csv"
)

HZ_CORRECTED = (
    HORIZON_DIR
    / "corrected"
    / "corrected_horizon_estimates.csv"
)


# ============================================================
# UTILIDADES
# ============================================================

def col(
    df: pd.DataFrame,
    name: str,
) -> pd.Series:
    """
    Devuelve una columna si existe.

    Si la columna no existe, devuelve una serie de NaN con el
    mismo índice que el DataFrame.
    """

    if name in df.columns:
        return df[name]

    return pd.Series(
        np.nan,
        index=df.index,
    )


def mae(
    subset: pd.DataFrame,
    column_name: str,
) -> float:
    """
    Calcula el MAE a partir de una columna de errores.

    Los valores no numéricos, infinitos y NaN se eliminan.
    """

    if column_name not in subset.columns:
        return np.nan

    errors = pd.to_numeric(
        subset[column_name],
        errors="coerce",
    )

    errors = errors[
        np.isfinite(
            errors
        )
    ]

    if len(errors) == 0:
        return np.nan

    return float(
        errors.abs().mean()
    )


def bias(
    subset: pd.DataFrame,
    column_name: str,
) -> float:
    """
    Calcula el sesgo medio de una columna de errores.
    """

    if column_name not in subset.columns:
        return np.nan

    errors = pd.to_numeric(
        subset[column_name],
        errors="coerce",
    )

    errors = errors[
        np.isfinite(
            errors
        )
    ]

    if len(errors) == 0:
        return np.nan

    return float(
        errors.mean()
    )


def rmse(
    subset: pd.DataFrame,
    column_name: str,
) -> float:
    """
    Calcula el RMSE de una columna de errores.
    """

    if column_name not in subset.columns:
        return np.nan

    errors = pd.to_numeric(
        subset[column_name],
        errors="coerce",
    )

    errors = errors[
        np.isfinite(
            errors
        )
    ]

    if len(errors) == 0:
        return np.nan

    return float(
        np.sqrt(
            np.mean(
                errors.to_numpy(
                    dtype=float
                )
                ** 2
            )
        )
    )


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    # --------------------------------------------------------
    # COMPROBACIÓN DE ARCHIVOS
    # --------------------------------------------------------

    missing_files = [
        path
        for path in [
            E1,
            HZ,
            HZ_CORRECTED,
        ]
        if not path.exists()
    ]

    if missing_files:
        missing_text = "\n".join(
            f" - {path}"
            for path in missing_files
        )

        raise FileNotFoundError(
            "Faltan archivos necesarios para generar "
            "la comparativa:\n"
            f"{missing_text}\n\n"
            "Ejecuta primero main.py."
        )

    # --------------------------------------------------------
    # LECTURA DE RESULTADOS
    # --------------------------------------------------------

    e1 = pd.read_csv(
        E1
    )

    hz = pd.read_csv(
        HZ
    )

    hz_corrected = pd.read_csv(
        HZ_CORRECTED
    )

    # ========================================================
    # EXPERIMENTO 1: ESCALAS PÍXEL-CENTÍMETRO
    # ========================================================

    e1f = pd.DataFrame()

    e1f[
        "image_path"
    ] = e1[
        "image_path"
    ]

    e1f[
        "site"
    ] = col(
        e1,
        "site",
    )

    e1f[
        "subject_id"
    ] = col(
        e1,
        "subject_id",
    )

    e1f[
        "view"
    ] = col(
        e1,
        "view",
    )

    e1f[
        "height_manual_cm"
    ] = col(
        e1,
        "height_manual_cm",
    )

    e1f[
        "e1_height_px"
    ] = col(
        e1,
        "height_px",
    )

    # --------------------------------------------------------
    # E1: mediana de todas las escalas
    # --------------------------------------------------------

    e1f[
        "e1_scale_median_cmpx"
    ] = col(
        e1,
        "scale_median",
    )

    e1f[
        "e1_H_median"
    ] = col(
        e1,
        "height_cm_median_scale",
    )

    e1f[
        "e1_err_median"
    ] = col(
        e1,
        "error_cm_median_scale",
    )

    # --------------------------------------------------------
    # E1: escala de la referencia de mejor calidad
    # --------------------------------------------------------

    e1f[
        "e1_scale_bestquality_cmpx"
    ] = col(
        e1,
        "scale_best_quality",
    )

    e1f[
        "e1_H_bestquality"
    ] = col(
        e1,
        "height_cm_best_quality_scale",
    )

    e1f[
        "e1_err_bestquality"
    ] = col(
        e1,
        "error_cm_best_quality_scale",
    )

    # --------------------------------------------------------
    # E1: escala de la referencia más próxima a los pies
    # --------------------------------------------------------

    e1f[
        "e1_scale_closestfeet_cmpx"
    ] = col(
        e1,
        "scale_closest_feet",
    )

    e1f[
        "e1_H_closestfeet"
    ] = col(
        e1,
        "height_cm_closest_feet_scale",
    )

    e1f[
        "e1_err_closestfeet"
    ] = col(
        e1,
        "error_cm_closest_feet_scale",
    )

    # --------------------------------------------------------
    # E1: escala de la línea de suelo
    # --------------------------------------------------------

    e1f[
        "e1_scale_groundline_cmpx"
    ] = col(
        e1,
        "scale_ground_line",
    )

    e1f[
        "e1_H_groundline"
    ] = col(
        e1,
        "height_cm_ground_line_scale",
    )

    e1f[
        "e1_err_groundline"
    ] = col(
        e1,
        "error_cm_ground_line_scale",
    )

    # --------------------------------------------------------
    # E1: escala de suelo corregida por vista mediante LOO
    # --------------------------------------------------------

    e1f[
        "e1_H_groundline_corr"
    ] = col(
        e1,
        "height_cm_ground_line_scale_corrected",
    )

    e1f[
        "e1_err_groundline_corr"
    ] = col(
        e1,
        "error_cm_ground_line_scale_corrected",
    )

    # ========================================================
    # EXPERIMENTO 2 ORIGINAL: MÉTODO DEL HORIZONTE
    # ========================================================

    hzf = pd.DataFrame()

    hzf[
        "image_path"
    ] = hz[
        "image_path"
    ]

    hzf[
        "hz_status"
    ] = col(
        hz,
        "status",
    )

    hzf[
        "hz_height_px"
    ] = col(
        hz,
        "height_px",
    )

    hzf[
        "hz_feet_source"
    ] = col(
        hz,
        "feet_source",
    )

    hzf[
        "hz_head_source"
    ] = col(
        hz,
        "head_source",
    )

    hzf[
        "hz_y_horizon_ref"
    ] = col(
        hz,
        "y_horizon_ref",
    )

    hzf[
        "hz_y_horizon_cube"
    ] = col(
        hz,
        "y_horizon_cube",
    )

    hzf[
        "hz_y_horizon_shared"
    ] = col(
        hz,
        "y_horizon_shared",
    )

    hpx = col(
        hz,
        "height_px",
    ).replace(
        0,
        np.nan,
    )

    # --------------------------------------------------------
    # HZ original: horizonte propio calculado desde el cubo
    # --------------------------------------------------------

    hzf[
        "hz_H_cube"
    ] = col(
        hz,
        "H_horizon_cube",
    )

    hzf[
        "hz_scale_cube_cmpx"
    ] = (
        hzf[
            "hz_H_cube"
        ]
        / hpx
    )

    hzf[
        "hz_err_cube"
    ] = col(
        hz,
        "error_horizon_cube",
    )

    # --------------------------------------------------------
    # HZ original: horizonte compartido front/back
    # --------------------------------------------------------

    hzf[
        "hz_H_shared"
    ] = col(
        hz,
        "H_horizon_shared",
    )

    hzf[
        "hz_scale_shared_cmpx"
    ] = (
        hzf[
            "hz_H_shared"
        ]
        / hpx
    )

    hzf[
        "hz_err_shared"
    ] = col(
        hz,
        "error_horizon_shared",
    )

    # --------------------------------------------------------
    # HZ original: comparación de puntos de pie
    # --------------------------------------------------------

    hzf[
        "hz_H_feet_single"
    ] = col(
        hz,
        "H_shared_feet_single",
    )

    hzf[
        "hz_err_feet_single"
    ] = col(
        hz,
        "error_feet_single",
    )

    hzf[
        "hz_H_feet_mask"
    ] = col(
        hz,
        "H_shared_feet_mask",
    )

    hzf[
        "hz_err_feet_mask"
    ] = col(
        hz,
        "error_feet_mask",
    )

    # ========================================================
    # EXPERIMENTO 2 CORREGIDO
    # ========================================================

    hzcf = pd.DataFrame()

    hzcf[
        "image_path"
    ] = hz_corrected[
        "image_path"
    ]

    hzcf[
        "hzc_status_original"
    ] = col(
        hz_corrected,
        "status",
    )

    hzcf[
        "hzc_horizon_source"
    ] = col(
        hz_corrected,
        "horizon_corrected_source",
    )

    hzcf[
        "hzc_reliability"
    ] = col(
        hz_corrected,
        "height_reliability",
    )

    hzcf[
        "hzc_feet_source"
    ] = col(
        hz_corrected,
        "feet_source",
    )

    hzcf[
        "hzc_head_source"
    ] = col(
        hz_corrected,
        "head_source",
    )

    hzcf[
        "hzc_y_horizon_selected"
    ] = col(
        hz_corrected,
        "y_horizon_corrected",
    )

    hzcf[
        "hzc_y_horizon_cube"
    ] = col(
        hz_corrected,
        "y_horizon_cube",
    )

    hzcf[
        "hzc_y_horizon_shared"
    ] = col(
        hz_corrected,
        "y_horizon_shared",
    )

    hzcf[
        "hzc_y_global_lateral_loo"
    ] = col(
        hz_corrected,
        "y_global_lateral_loo",
    )

    hzcf[
        "hzc_y_global_front_back_loo"
    ] = col(
        hz_corrected,
        "y_global_front_back_loo",
    )

    hzcf[
        "hzc_height_px"
    ] = col(
        hz_corrected,
        "height_px",
    )

    hzc_height_px = col(
        hz_corrected,
        "height_px",
    ).replace(
        0,
        np.nan,
    )

    # --------------------------------------------------------
    # HZ corregido: altura final
    # --------------------------------------------------------

    hzcf[
        "hzc_H_corrected"
    ] = col(
        hz_corrected,
        "H_horizon_corrected",
    )

    hzcf[
        "hzc_scale_equivalent_cmpx"
    ] = (
        hzcf[
            "hzc_H_corrected"
        ]
        / hzc_height_px
    )

    hzcf[
        "hzc_err_corrected"
    ] = col(
        hz_corrected,
        "error_horizon_corrected",
    )

    hzcf[
        "hzc_abs_err_corrected"
    ] = col(
        hz_corrected,
        "abs_error_horizon_corrected",
    )

    # --------------------------------------------------------
    # HZ corregido: variantes de pie
    # --------------------------------------------------------

    hzcf[
        "hzc_H_feet_single"
    ] = col(
        hz_corrected,
        "H_corrected_feet_single",
    )

    hzcf[
        "hzc_err_feet_single"
    ] = col(
        hz_corrected,
        "error_corrected_feet_single",
    )

    hzcf[
        "hzc_H_feet_mask"
    ] = col(
        hz_corrected,
        "H_corrected_feet_mask",
    )

    hzcf[
        "hzc_err_feet_mask"
    ] = col(
        hz_corrected,
        "error_corrected_feet_mask",
    )

    # --------------------------------------------------------
    # Imagen de verificación del horizonte seleccionado
    # --------------------------------------------------------

    hzcf[
        "hzc_verification_image_path"
    ] = col(
        hz_corrected,
        "verification_image_path",
    )

    # ========================================================
    # MERGE FINAL POR IMAGEN
    # ========================================================

    final = (
        e1f
        .merge(
            hzf,
            on="image_path",
            how="outer",
            validate="one_to_one",
        )
        .merge(
            hzcf,
            on="image_path",
            how="outer",
            validate="one_to_one",
        )
    )

    # --------------------------------------------------------
    # Orden de columnas principales
    # --------------------------------------------------------

    id_columns = [
        "site",
        "subject_id",
        "view",
        "image_path",
        "height_manual_cm",
    ]

    existing_id_columns = [
        column_name
        for column_name in id_columns
        if column_name in final.columns
    ]

    other_columns = [
        column_name
        for column_name in final.columns
        if column_name
        not in existing_id_columns
    ]

    final = final[
        existing_id_columns
        + other_columns
    ]

    # --------------------------------------------------------
    # Guardado del CSV completo
    # --------------------------------------------------------

    COMPARATIVA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    final.to_csv(
        COMPARATIVA_CSV,
        index=False,
    )

    print(
        "\nCSV final por imagen guardado en:"
    )

    print(
        f" - {COMPARATIVA_CSV}"
    )

    print(
        f"Filas: {len(final)}"
    )

    print(
        f"Columnas: {len(final.columns)}"
    )

    # ========================================================
    # TABLA RESUMEN DE MÉTRICAS
    # ========================================================

    methods = {
        "E1 mediana escalas": (
            "e1_err_median"
        ),
        "E1 escala mejor calidad": (
            "e1_err_bestquality"
        ),
        "E1 escala cercana a pies": (
            "e1_err_closestfeet"
        ),
        "E1 escala suelo": (
            "e1_err_groundline"
        ),
        "E1 suelo corregida LOO*": (
            "e1_err_groundline_corr"
        ),
        "HZ vía cubo": (
            "hz_err_cube"
        ),
        "HZ compartido": (
            "hz_err_shared"
        ),
        "HZ compartido pie single": (
            "hz_err_feet_single"
        ),
        "HZ compartido pie mask": (
            "hz_err_feet_mask"
        ),
        "HZ corregido final LOO": (
            "hzc_err_corrected"
        ),
    }

    views = [
        "front",
        "back",
        "left",
        "right",
    ]

    mae_rows = []
    detailed_rows = []

    for (
        method_name,
        error_column,
    ) in methods.items():
        if error_column not in final.columns:
            continue

        mae_row = {
            "metodo": method_name,
        }

        # Métricas independientes por vista.
        for view in views:
            view_subset = final[
                final[
                    "view"
                ]
                == view
            ]

            mae_row[
                view
            ] = mae(
                view_subset,
                error_column,
            )

            detailed_rows.append(
                {
                    "metodo": method_name,
                    "grupo": view,
                    "n": int(
                        pd.to_numeric(
                            view_subset[
                                error_column
                            ],
                            errors="coerce",
                        )
                        .replace(
                            [
                                np.inf,
                                -np.inf,
                            ],
                            np.nan,
                        )
                        .notna()
                        .sum()
                    ),
                    "bias_cm": bias(
                        view_subset,
                        error_column,
                    ),
                    "mae_cm": mae(
                        view_subset,
                        error_column,
                    ),
                    "rmse_cm": rmse(
                        view_subset,
                        error_column,
                    ),
                }
            )

        front_back_subset = final[
            final[
                "view"
            ].isin(
                [
                    "front",
                    "back",
                ]
            )
        ]

        lateral_subset = final[
            final[
                "view"
            ].isin(
                [
                    "left",
                    "right",
                ]
            )
        ]

        mae_row[
            "FRONT+BACK"
        ] = mae(
            front_back_subset,
            error_column,
        )

        mae_row[
            "LEFT+RIGHT"
        ] = mae(
            lateral_subset,
            error_column,
        )

        mae_row[
            "GLOBAL"
        ] = mae(
            final,
            error_column,
        )

        mae_rows.append(
            mae_row
        )

        for (
            group_name,
            group_subset,
        ) in [
            (
                "FRONT+BACK",
                front_back_subset,
            ),
            (
                "LEFT+RIGHT",
                lateral_subset,
            ),
            (
                "GLOBAL",
                final,
            ),
        ]:
            numeric_errors = pd.to_numeric(
                group_subset[
                    error_column
                ],
                errors="coerce",
            ).replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )

            detailed_rows.append(
                {
                    "metodo": method_name,
                    "grupo": group_name,
                    "n": int(
                        numeric_errors.notna().sum()
                    ),
                    "bias_cm": bias(
                        group_subset,
                        error_column,
                    ),
                    "mae_cm": mae(
                        group_subset,
                        error_column,
                    ),
                    "rmse_cm": rmse(
                        group_subset,
                        error_column,
                    ),
                }
            )

    # --------------------------------------------------------
    # Tabla principal de MAE
    # --------------------------------------------------------

    mae_table = pd.DataFrame(
        mae_rows
    )

    if not mae_table.empty:
        mae_table = mae_table.set_index(
            "metodo"
        )

    print(
        "\n=== MAE (cm) por método y vista ==="
    )

    print(
        mae_table.round(
            2
        ).to_string()
    )

    mae_table.round(
        4
    ).to_csv(
        COMPARATIVA_MAE_CSV
    )

    print(
        "\nTabla de MAE guardada en:"
    )

    print(
        f" - {COMPARATIVA_MAE_CSV}"
    )

    # --------------------------------------------------------
    # Tabla detallada con n, bias, MAE y RMSE
    # --------------------------------------------------------

    detailed_metrics_df = pd.DataFrame(
        detailed_rows
    )

    detailed_metrics_path = (
        COMPARATIVA_DIR
        / "comparativa_metodos_metricas_detalladas.csv"
    )

    detailed_metrics_df.to_csv(
        detailed_metrics_path,
        index=False,
    )

    print(
        "\nTabla detallada guardada en:"
    )

    print(
        f" - {detailed_metrics_path}"
    )

    # ========================================================
    # FUENTES DEL HORIZONTE CORREGIDO
    # ========================================================

    if (
        "hzc_horizon_source"
        in final.columns
    ):
        source_columns = [
            "view",
            "hzc_horizon_source",
            "hzc_reliability",
        ]

        available_source_columns = [
            column_name
            for column_name in source_columns
            if column_name in final.columns
        ]

        source_summary = (
            final
            .groupby(
                available_source_columns,
                dropna=False,
            )
            .size()
            .reset_index(
                name="n"
            )
        )

        source_summary_path = (
            COMPARATIVA_DIR
            / "comparativa_horizonte_corregido_fuentes.csv"
        )

        source_summary.to_csv(
            source_summary_path,
            index=False,
        )

        print(
            "\n=== Fuentes del horizonte corregido ==="
        )

        print(
            source_summary.to_string(
                index=False
            )
        )

        print(
            "\nResumen de fuentes guardado en:"
        )

        print(
            f" - {source_summary_path}"
        )

    # ========================================================
    # MEJOR MÉTODO POR COLUMNA
    # ========================================================

    print(
        "\n=== Mejor método por grupo ==="
    )

    for column_name in mae_table.columns:
        values = pd.to_numeric(
            mae_table[
                column_name
            ],
            errors="coerce",
        ).dropna()

        if len(values) == 0:
            continue

        best_method = values.idxmin()
        best_mae = values.min()

        print(
            f"  {column_name:12s}: "
            f"{best_method} "
            f"({best_mae:.2f} cm)"
        )

    # ========================================================
    # NOTAS METODOLÓGICAS
    # ========================================================

    print(
        "\nNotas metodológicas:"
    )

    print(
        " - E1 suelo corregida LOO utiliza las alturas reales "
        "para estimar la corrección por vista mediante "
        "leave-one-out."
    )

    print(
        " - HZ corregido final LOO no utiliza la altura manual "
        "para corregir la predicción."
    )

    print(
        " - En el horizonte corregido, leave-one-subject-out "
        "se utiliza únicamente para calcular los horizontes "
        "globales de fallback sin incluir al sujeto evaluado."
    )

    print(
        " - La columna hzc_verification_image_path permite "
        "localizar la imagen donde se visualiza el horizonte "
        "finalmente seleccionado."
    )


if __name__ == "__main__":
    main()
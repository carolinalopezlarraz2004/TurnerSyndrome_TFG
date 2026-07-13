"""
comparativa_visual.py

Para cada imagen, genera UNA imagen anotada por MÉTODO, organizada así:

    outputs/comparativa_final/<sujeto>/<perspectiva>/<metodo>.png

Métodos:

    1_experiment_base
        Experimento 1: escala de línea de suelo.

    2_experiment_base_corrected
        Experimento 1: escala de línea de suelo corregida mediante
        leave-one-out.

    3_horizon_cube
        Experimento 2 original: horizonte calculado directamente
        mediante el cubo de referencia.

    4_horizon_shared_single
        Experimento 2 original: horizonte compartido y pies obtenidos
        desde la pose.

    5_horizon_shared_mask
        Experimento 2 original: horizonte compartido y pies obtenidos
        desde la máscara corporal.

    6_horizon_corrected_final
        Experimento 2 corregido: horizonte final seleccionado mediante
        la jerarquía por vista y fallbacks leave-one-subject-out.

Cada imagen muestra:

    - Línea cabeza-pies.
    - Referencia utilizada por el método.
    - Horizonte, cuando corresponde.
    - Altura real.
    - Altura estimada.
    - Error.
    - Escala equivalente píxel-centímetro.
    - Fuente del horizonte y fiabilidad, en el método corregido.

No reprocesa la detección de la persona. Lee los puntos y estimaciones
desde los CSV generados por main.py. Únicamente vuelve a detectar los
marcadores ArUco para dibujar la arista del cubo.

Uso:

    python comparativa_visual.py
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.config import (
    VERIFICATION_DIR,
    HORIZON_DIR,
    COMPARATIVA_DIR,
)

from src.image_calibration import (
    detect_aruco_markers,
    put_label,
)

from src.horizon_method import (
    select_reference_cube,
)


# ============================================================
# RUTAS DE ENTRADA
# ============================================================

E1_CSV = (
    VERIFICATION_DIR
    / "height_estimates_experiment1.csv"
)

HZ_CSV = (
    HORIZON_DIR
    / "horizon_estimates.csv"
)

HZ_CORRECTED_CSV = (
    HORIZON_DIR
    / "corrected"
    / "corrected_horizon_estimates.csv"
)


# ============================================================
# UTILIDADES
# ============================================================

def safe_float(
    value,
) -> float:
    """
    Convierte un valor a float.

    Si no puede convertirse o no es finito, devuelve NaN.
    """

    try:
        value = float(
            value
        )

        if np.isfinite(
            value
        ):
            return value

    except (
        TypeError,
        ValueError,
    ):
        pass

    return np.nan


def get_xy(
    row: pd.Series,
    x_column: str,
    y_column: str,
):
    """
    Obtiene un punto (x, y) desde una fila.

    Devuelve None cuando alguna coordenada no está disponible.
    """

    x_value = safe_float(
        row.get(
            x_column,
            np.nan,
        )
    )

    y_value = safe_float(
        row.get(
            y_column,
            np.nan,
        )
    )

    if not (
        np.isfinite(
            x_value
        )
        and np.isfinite(
            y_value
        )
    ):
        return None

    return (
        x_value,
        y_value,
    )


def point_with_new_y(
    base_point,
    new_y,
):
    """
    Conserva la coordenada x de un punto y sustituye su coordenada y.
    """

    y_value = safe_float(
        new_y
    )

    if (
        base_point is None
        or not np.isfinite(
            y_value
        )
    ):
        return base_point

    return (
        float(
            base_point[0]
        ),
        y_value,
    )


def method_color_from_reliability(
    reliability: str,
):
    """
    Selecciona un color para la etiqueta de fiabilidad.

    Los colores se expresan en formato BGR de OpenCV.
    """

    reliability = str(
        reliability
    ).lower()

    color_map = {
        "high": (
            0,
            220,
            0,
        ),
        "medium": (
            0,
            220,
            220,
        ),
        "low": (
            0,
            140,
            255,
        ),
        "very_low": (
            0,
            0,
            255,
        ),
        "unavailable": (
            180,
            180,
            180,
        ),
    }

    return color_map.get(
        reliability,
        (
            255,
            255,
            255,
        ),
    )


# ============================================================
# DIBUJO DE UN MÉTODO
# ============================================================

def draw_method(
    image,
    head_xy,
    feet_xy,
    title,
    real_cm,
    estimated_cm,
    horizon_y=None,
    cube_edge=None,
    output_path=None,
    horizon_color=(
        0,
        220,
        220,
    ),
    horizon_label="horizonte",
    source=None,
    reliability=None,
):
    """
    Dibuja una imagen anotada para un método concreto.
    """

    canvas = image.copy()

    image_height, image_width = (
        canvas.shape[:2]
    )

    real_cm = safe_float(
        real_cm
    )

    estimated_cm = safe_float(
        estimated_cm
    )

    horizon_y = safe_float(
        horizon_y
    )

    # --------------------------------------------------------
    # Horizonte
    # --------------------------------------------------------

    if np.isfinite(
        horizon_y
    ):
        horizon_position = int(
            np.clip(
                horizon_y,
                0,
                image_height - 1,
            )
        )

        cv2.line(
            canvas,
            (
                0,
                horizon_position,
            ),
            (
                image_width,
                horizon_position,
            ),
            horizon_color,
            4
            if "final" in horizon_label.lower()
            else 2,
            cv2.LINE_AA,
        )

        put_label(
            canvas,
            (
                f"{horizon_label} "
                f"(y={horizon_y:.0f})"
            ),
            (
                30,
                max(
                    horizon_position - 8,
                    20,
                ),
            ),
            color=horizon_color,
        )

    # --------------------------------------------------------
    # Arista del cubo
    # --------------------------------------------------------

    if cube_edge is not None:
        cube_base = tuple(
            np.int32(
                cube_edge[0]
            )
        )

        cube_top = tuple(
            np.int32(
                cube_edge[1]
            )
        )

        cv2.line(
            canvas,
            cube_base,
            cube_top,
            (
                0,
                220,
                0,
            ),
            4,
            cv2.LINE_AA,
        )

        put_label(
            canvas,
            "cubo 10 cm",
            (
                cube_base[0] + 6,
                cube_base[1] + 18,
            ),
            color=(
                0,
                220,
                0,
            ),
        )

    # --------------------------------------------------------
    # Persona
    # --------------------------------------------------------

    if (
        head_xy is not None
        and feet_xy is not None
    ):
        head_point = (
            int(
                head_xy[0]
            ),
            int(
                head_xy[1]
            ),
        )

        feet_point = (
            int(
                feet_xy[0]
            ),
            int(
                feet_xy[1]
            ),
        )

        cv2.line(
            canvas,
            head_point,
            feet_point,
            (
                255,
                0,
                0,
            ),
            3,
            cv2.LINE_AA,
        )

        cv2.circle(
            canvas,
            head_point,
            7,
            (
                0,
                255,
                255,
            ),
            -1,
        )

        put_label(
            canvas,
            "cabeza",
            (
                head_point[0] + 8,
                head_point[1],
            ),
            color=(
                0,
                255,
                255,
            ),
        )

        cv2.circle(
            canvas,
            feet_point,
            7,
            (
                0,
                0,
                255,
            ),
            -1,
        )

        put_label(
            canvas,
            "pies",
            (
                feet_point[0] + 8,
                feet_point[1],
            ),
            color=(
                0,
                0,
                255,
            ),
        )

    # --------------------------------------------------------
    # Cálculo de escala equivalente y error
    # --------------------------------------------------------

    if (
        head_xy is not None
        and feet_xy is not None
    ):
        height_px = float(
            feet_xy[1]
            - head_xy[1]
        )

    else:
        height_px = np.nan

    equivalent_scale = (
        estimated_cm
        / height_px
        if (
            np.isfinite(
                estimated_cm
            )
            and np.isfinite(
                height_px
            )
            and height_px > 0
        )
        else np.nan
    )

    error_cm = (
        estimated_cm
        - real_cm
        if (
            np.isfinite(
                estimated_cm
            )
            and np.isfinite(
                real_cm
            )
        )
        else np.nan
    )

    # --------------------------------------------------------
    # Recuadro informativo
    # --------------------------------------------------------

    information_lines = [
        title,
        (
            f"altura real:      "
            f"{real_cm:.1f} cm"
            if np.isfinite(
                real_cm
            )
            else "altura real:      NA"
        ),
        (
            f"altura estimada:  "
            f"{estimated_cm:.1f} cm"
            if np.isfinite(
                estimated_cm
            )
            else "altura estimada:  NA"
        ),
        (
            f"error:            "
            f"{error_cm:+.1f} cm"
            if np.isfinite(
                error_cm
            )
            else "error:            NA"
        ),
        (
            f"escala px->cm:    "
            f"{equivalent_scale:.5f} cm/px"
            if np.isfinite(
                equivalent_scale
            )
            else "escala px->cm:    NA"
        ),
        (
            f"altura en px:     "
            f"{height_px:.0f}"
            if np.isfinite(
                height_px
            )
            else "altura en px:     NA"
        ),
    ]

    if source is not None:
        information_lines.append(
            f"fuente horizonte: {source}"
        )

    if reliability is not None:
        information_lines.append(
            f"fiabilidad:        {reliability}"
        )

    box_width = min(
        max(
            660,
            int(
                image_width
                * 0.62
            ),
        ),
        image_width - 40,
    )

    box_height = (
        30
        + 30
        * len(
            information_lines
        )
    )

    cv2.rectangle(
        canvas,
        (
            20,
            15,
        ),
        (
            box_width,
            box_height,
        ),
        (
            0,
            0,
            0,
        ),
        -1,
    )

    for line_index, text in enumerate(
        information_lines
    ):
        text_color = (
            0,
            255,
            255,
        )

        if (
            reliability is not None
            and line_index
            == len(
                information_lines
            )
            - 1
        ):
            text_color = (
                method_color_from_reliability(
                    reliability
                )
            )

        put_label(
            canvas,
            text,
            (
                30,
                45
                + line_index
                * 30,
            ),
            color=text_color,
        )

    # --------------------------------------------------------
    # Guardado
    # --------------------------------------------------------

    if output_path is not None:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        cv2.imwrite(
            str(
                output_path
            ),
            canvas,
        )


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    # --------------------------------------------------------
    # Comprobación de archivos
    # --------------------------------------------------------

    required_files = [
        E1_CSV,
        HZ_CSV,
        HZ_CORRECTED_CSV,
    ]

    missing_files = [
        path
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        missing_text = "\n".join(
            f" - {path}"
            for path in missing_files
        )

        raise FileNotFoundError(
            "Faltan archivos necesarios:\n"
            f"{missing_text}\n\n"
            "Ejecuta primero main.py."
        )

    # --------------------------------------------------------
    # Lectura de resultados
    # --------------------------------------------------------

    experiment_1 = pd.read_csv(
        E1_CSV
    )

    horizon_original = pd.read_csv(
        HZ_CSV
    )

    horizon_corrected = pd.read_csv(
        HZ_CORRECTED_CSV
    )

    # --------------------------------------------------------
    # Columnas del Experimento 1
    # --------------------------------------------------------

    experiment_1_columns = [
        "image_path",
        "site",
        "subject_id",
        "view",
        "height_manual_cm",
        "head_x_px",
        "head_y_px",
        "feet_x_px",
        "feet_y_px",
        "height_cm_ground_line_scale",
        "height_cm_ground_line_scale_corrected",
    ]

    experiment_1 = experiment_1[
        [
            column_name
            for column_name
            in experiment_1_columns
            if column_name
            in experiment_1.columns
        ]
    ].copy()

    # --------------------------------------------------------
    # Columnas del horizonte original
    # --------------------------------------------------------

    horizon_original_columns = [
        "image_path",
        "y_head",
        "y_feet",
        "y_feet_single",
        "y_feet_mask",
        "y_horizon_ref",
        "y_horizon_cube",
        "y_horizon_shared",
        "H_horizon_cube",
        "H_horizon_shared",
        "H_shared_feet_single",
        "H_shared_feet_mask",
    ]

    horizon_original = horizon_original[
        [
            column_name
            for column_name
            in horizon_original_columns
            if column_name
            in horizon_original.columns
        ]
    ].copy()

    # --------------------------------------------------------
    # Renombrado del horizonte corregido
    # --------------------------------------------------------

    corrected_columns = [
        "image_path",
        "y_head",
        "y_feet",
        "y_feet_single",
        "y_feet_mask",
        "y_horizon_corrected",
        "y_horizon_cube",
        "y_horizon_shared",
        "H_horizon_corrected",
        "horizon_corrected_source",
        "height_reliability",
        "verification_image_path",
    ]

    horizon_corrected = horizon_corrected[
        [
            column_name
            for column_name
            in corrected_columns
            if column_name
            in horizon_corrected.columns
        ]
    ].copy()

    horizon_corrected = (
        horizon_corrected.rename(
            columns={
                "y_head": (
                    "hzc_y_head"
                ),
                "y_feet": (
                    "hzc_y_feet"
                ),
                "y_feet_single": (
                    "hzc_y_feet_single"
                ),
                "y_feet_mask": (
                    "hzc_y_feet_mask"
                ),
                "y_horizon_corrected": (
                    "hzc_y_horizon_corrected"
                ),
                "y_horizon_cube": (
                    "hzc_y_horizon_cube"
                ),
                "y_horizon_shared": (
                    "hzc_y_horizon_shared"
                ),
                "H_horizon_corrected": (
                    "hzc_H_horizon_corrected"
                ),
                "horizon_corrected_source": (
                    "hzc_horizon_source"
                ),
                "height_reliability": (
                    "hzc_reliability"
                ),
                "verification_image_path": (
                    "hzc_verification_image_path"
                ),
            }
        )
    )

    # --------------------------------------------------------
    # Merge por imagen
    # --------------------------------------------------------

    dataframe = (
        experiment_1
        .merge(
            horizon_original,
            on="image_path",
            how="outer",
            validate="one_to_one",
        )
        .merge(
            horizon_corrected,
            on="image_path",
            how="outer",
            validate="one_to_one",
        )
    )

    generated_images = 0
    failed_images = 0

    # ========================================================
    # GENERACIÓN DE IMÁGENES
    # ========================================================

    for _, row in dataframe.iterrows():
        image_path = row.get(
            "image_path",
            None,
        )

        if not isinstance(
            image_path,
            str,
        ):
            failed_images += 1
            continue

        image = cv2.imread(
            image_path
        )

        if image is None:
            failed_images += 1
            continue

        subject = str(
            row.get(
                "subject_id",
                "NA",
            )
        )

        view = str(
            row.get(
                "view",
                "NA",
            )
        ).lower()

        real_height = safe_float(
            row.get(
                "height_manual_cm",
                np.nan,
            )
        )

        output_directory = (
            COMPARATIVA_DIR
            / subject
            / view
        )

        # ----------------------------------------------------
        # Puntos del Experimento 1
        # ----------------------------------------------------

        head_experiment_1 = get_xy(
            row,
            "head_x_px",
            "head_y_px",
        )

        feet_experiment_1 = get_xy(
            row,
            "feet_x_px",
            "feet_y_px",
        )

        # ----------------------------------------------------
        # Cubo de referencia
        # ----------------------------------------------------

        cube_edge = None

        try:
            gray_image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )

            corners, ids = (
                detect_aruco_markers(
                    gray_image
                )
            )

            markers = (
                [
                    marker.reshape(
                        4,
                        2,
                    )
                    for marker in corners
                ]
                if ids is not None
                else []
            )

            feet_y_reference = safe_float(
                row.get(
                    "y_feet",
                    row.get(
                        "feet_y_px",
                        np.nan,
                    ),
                )
            )

            if (
                markers
                and np.isfinite(
                    feet_y_reference
                )
            ):
                cube_edge = (
                    select_reference_cube(
                        markers,
                        feet_y_reference,
                    )
                )

        except Exception:
            cube_edge = None

        # ====================================================
        # 1. EXPERIMENTO BASE
        # ====================================================

        draw_method(
            image=image,
            head_xy=head_experiment_1,
            feet_xy=feet_experiment_1,
            title=(
                "1 - Exp base: "
                "escala linea de suelo"
            ),
            real_cm=real_height,
            estimated_cm=safe_float(
                row.get(
                    "height_cm_ground_line_scale",
                    np.nan,
                )
            ),
            output_path=(
                output_directory
                / "1_experiment_base.png"
            ),
        )

        # ====================================================
        # 2. EXPERIMENTO BASE CORREGIDO
        # ====================================================

        draw_method(
            image=image,
            head_xy=head_experiment_1,
            feet_xy=feet_experiment_1,
            title=(
                "2 - Exp base: "
                "escala suelo corregida (LOO)"
            ),
            real_cm=real_height,
            estimated_cm=safe_float(
                row.get(
                    "height_cm_ground_line_scale_corrected",
                    np.nan,
                )
            ),
            output_path=(
                output_directory
                / "2_experiment_base_corrected.png"
            ),
        )

        # ----------------------------------------------------
        # Puntos del horizonte original
        # ----------------------------------------------------

        original_head = point_with_new_y(
            head_experiment_1,
            row.get(
                "y_head",
                np.nan,
            ),
        )

        original_feet = point_with_new_y(
            feet_experiment_1,
            row.get(
                "y_feet",
                np.nan,
            ),
        )

        original_feet_single = (
            point_with_new_y(
                feet_experiment_1,
                row.get(
                    "y_feet_single",
                    np.nan,
                ),
            )
        )

        original_feet_mask = (
            point_with_new_y(
                feet_experiment_1,
                row.get(
                    "y_feet_mask",
                    np.nan,
                ),
            )
        )

        # ====================================================
        # 3. HORIZONTE VÍA CUBO
        # ====================================================

        draw_method(
            image=image,
            head_xy=original_head,
            feet_xy=original_feet,
            title=(
                "3 - Horizonte via cubo"
            ),
            real_cm=real_height,
            estimated_cm=safe_float(
                row.get(
                    "H_horizon_cube",
                    np.nan,
                )
            ),
            horizon_y=row.get(
                "y_horizon_cube",
                np.nan,
            ),
            cube_edge=cube_edge,
            output_path=(
                output_directory
                / "3_horizon_cube.png"
            ),
        )

        # ====================================================
        # 4. HORIZONTE COMPARTIDO, PIE POSE
        # ====================================================

        shared_horizon = row.get(
            "y_horizon_ref",
            row.get(
                "y_horizon_shared",
                np.nan,
            ),
        )

        draw_method(
            image=image,
            head_xy=original_head,
            feet_xy=original_feet_single,
            title=(
                "4 - Horizonte compartido "
                "(pie pose)"
            ),
            real_cm=real_height,
            estimated_cm=safe_float(
                row.get(
                    "H_shared_feet_single",
                    np.nan,
                )
            ),
            horizon_y=shared_horizon,
            cube_edge=cube_edge,
            output_path=(
                output_directory
                / "4_horizon_shared_single.png"
            ),
        )

        # ====================================================
        # 5. HORIZONTE COMPARTIDO, PIE MÁSCARA
        # ====================================================

        draw_method(
            image=image,
            head_xy=original_head,
            feet_xy=original_feet_mask,
            title=(
                "5 - Horizonte compartido "
                "(pie mascara)"
            ),
            real_cm=real_height,
            estimated_cm=safe_float(
                row.get(
                    "H_shared_feet_mask",
                    np.nan,
                )
            ),
            horizon_y=shared_horizon,
            cube_edge=cube_edge,
            output_path=(
                output_directory
                / "5_horizon_shared_mask.png"
            ),
        )

        # ----------------------------------------------------
        # Puntos del horizonte corregido
        # ----------------------------------------------------

        corrected_head = point_with_new_y(
            head_experiment_1,
            row.get(
                "hzc_y_head",
                np.nan,
            ),
        )

        corrected_feet = point_with_new_y(
            feet_experiment_1,
            row.get(
                "hzc_y_feet",
                np.nan,
            ),
        )

        corrected_horizon_y = safe_float(
            row.get(
                "hzc_y_horizon_corrected",
                np.nan,
            )
        )

        corrected_height = safe_float(
            row.get(
                "hzc_H_horizon_corrected",
                np.nan,
            )
        )

        corrected_source = str(
            row.get(
                "hzc_horizon_source",
                "NA",
            )
        )

        corrected_reliability = str(
            row.get(
                "hzc_reliability",
                "NA",
            )
        )

        # ====================================================
        # 6. HORIZONTE CORREGIDO FINAL
        # ====================================================

        draw_method(
            image=image,
            head_xy=corrected_head,
            feet_xy=corrected_feet,
            title=(
                "6 - Horizonte corregido final"
            ),
            real_cm=real_height,
            estimated_cm=corrected_height,
            horizon_y=corrected_horizon_y,
            cube_edge=cube_edge,
            output_path=(
                output_directory
                / "6_horizon_corrected_final.png"
            ),
            horizon_color=(
                0,
                255,
                0,
            ),
            horizon_label=(
                "horizonte FINAL"
            ),
            source=corrected_source,
            reliability=(
                corrected_reliability
            ),
        )

        generated_images += 1

    # ========================================================
    # RESULTADOS
    # ========================================================

    print(
        "\nComparativa visual completada."
    )

    print(
        f"Imágenes originales procesadas: "
        f"{generated_images}"
    )

    print(
        f"Imágenes no procesadas: "
        f"{failed_images}"
    )

    print(
        "\nEstructura de salida:"
    )

    print(
        f" - {COMPARATIVA_DIR}/"
        "<sujeto>/<perspectiva>/<metodo>.png"
    )

    print(
        "\nMétodos generados por imagen:"
    )

    print(
        "  1. Experimento base"
    )

    print(
        "  2. Experimento base corregido LOO"
    )

    print(
        "  3. Horizonte vía cubo"
    )

    print(
        "  4. Horizonte compartido con pie pose"
    )

    print(
        "  5. Horizonte compartido con pie máscara"
    )

    print(
        "  6. Horizonte corregido final LOO"
    )


if __name__ == "__main__":
    main()
"""
horizon_corrected.py

Experimento 2 corregido: estimación de altura mediante el método del horizonte.

Fórmula
-------
    H = h_cam * (y_pies - y_cabeza) / (y_pies - y_horizonte)

El eje y de la imagen crece hacia abajo.

Estrategia
----------
FRONT/BACK:
    1. Horizonte compartido front/back del sujeto.
    2. Horizonte propio válido de la imagen.
    3. Horizonte global front/back leave-one-subject-out.
    4. Horizonte global lateral leave-one-subject-out como último recurso.

LEFT/RIGHT:
    1. Horizonte propio válido de la imagen.
    2. Horizonte válido de la otra vista lateral del sujeto.
    3. Horizonte global lateral leave-one-subject-out.
    4. Horizonte front/back del sujeto como último recurso.

El horizonte global usado como fallback excluye todas las imágenes del sujeto
que se está evaluando para evitar fuga de información.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from src.config import (
    HORIZON_DIR,
    ARUCO_MARKER_SIZE_CM,
)

from src.image_calibration import (
    estimate_body_height_pixels,
    detect_aruco_markers,
    get_manual_height_for_subject,
    put_label,
)


try:
    from src.config import CAMERA_HEIGHT_CM
except Exception:
    CAMERA_HEIGHT_CM = 100.0


# ============================================================
# CONFIGURACIÓN
# ============================================================

MIN_EDGE_LEN_PX = 10.0

RELIABLE_VIEWS = (
    "front",
    "back",
)

LATERAL_VIEWS = (
    "left",
    "right",
)

USE_MASK_FEET_LATERAL = True
USE_MASK_HEAD = True

MAX_HEAD_MOVE_FRAC = 0.12
HEAD_WIDTH_FRAC = 0.50

MIN_PLAUSIBLE_HEIGHT_CM = 40.0
MAX_PLAUSIBLE_HEIGHT_CM = 230.0

CORRECTED_HORIZON_DIR = (
    HORIZON_DIR
    / "corrected"
)


# ============================================================
# UTILIDADES GENERALES
# ============================================================

def plausible_height(
    height_cm: float,
) -> bool:
    """
    Comprueba si una altura es numérica y físicamente plausible.
    """

    return bool(
        np.isfinite(height_cm)
        and MIN_PLAUSIBLE_HEIGHT_CM
        < height_cm
        < MAX_PLAUSIBLE_HEIGHT_CM
    )


def calculate_metrics(
    errors: pd.Series,
) -> dict:
    """
    Calcula número de casos, bias, MAE y RMSE.
    """

    values = pd.to_numeric(
        errors,
        errors="coerce",
    ).dropna()

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return {
            "n": 0,
            "bias_cm": np.nan,
            "mae_cm": np.nan,
            "rmse_cm": np.nan,
        }

    values_array = values.to_numpy(
        dtype=float
    )

    return {
        "n": int(
            len(values_array)
        ),
        "bias_cm": float(
            np.mean(values_array)
        ),
        "mae_cm": float(
            np.mean(
                np.abs(values_array)
            )
        ),
        "rmse_cm": float(
            np.sqrt(
                np.mean(
                    values_array ** 2
                )
            )
        ),
    }


# ============================================================
# UTILIDADES DE MÁSCARA
# ============================================================

def _mask_yx(
    body_mask: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    int,
]:
    """
    Devuelve las coordenadas activas de la máscara.
    """

    mask = np.asarray(
        body_mask
    )

    if mask.ndim == 3:
        mask = mask[..., 0]

    ys, xs = np.where(
        mask > 0
    )

    return (
        ys,
        xs,
        mask.shape[0],
    )


def feet_from_mask(
    body_mask: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Obtiene el punto inferior de la máscara corporal.

    Se utiliza la mediana horizontal de las últimas cinco filas
    para reducir el efecto de píxeles aislados.
    """

    ys, xs, _ = _mask_yx(
        body_mask
    )

    if len(ys) == 0:
        return None

    y_max = int(
        ys.max()
    )

    selection = (
        ys >= y_max - 4
    )

    return np.array(
        [
            float(
                np.median(
                    xs[selection]
                )
            ),
            float(
                y_max
            ),
        ],
        dtype=float,
    )


def head_from_mask_trim_hair(
    body_mask: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Estima la corona del cráneo evitando pelo o moños aislados.
    """

    ys, xs, image_height = _mask_yx(
        body_mask
    )

    if len(ys) == 0:
        return None

    y_top = int(
        ys.min()
    )

    y_bottom = int(
        ys.max()
    )

    min_x = np.full(
        image_height,
        1e9,
        dtype=float,
    )

    max_x = np.full(
        image_height,
        -1.0,
        dtype=float,
    )

    np.minimum.at(
        min_x,
        ys,
        xs,
    )

    np.maximum.at(
        max_x,
        ys,
        xs,
    )

    row_width = np.where(
        max_x >= 0,
        max_x - min_x + 1.0,
        0.0,
    )

    head_band_end = int(
        y_top
        + 0.12
        * (
            y_bottom
            - y_top
        )
    )

    head_band = row_width[
        y_top:
        head_band_end + 1
    ]

    head_band = head_band[
        head_band > 0
    ]

    if head_band.size == 0:
        return None

    sorted_band = np.sort(
        head_band
    )

    top_third_size = max(
        1,
        head_band.size // 3,
    )

    skull_width = float(
        np.median(
            sorted_band[
                -top_third_size:
            ]
        )
    )

    threshold = (
        HEAD_WIDTH_FRAC
        * skull_width
    )

    for y_coordinate in range(
        y_top,
        y_bottom + 1,
    ):
        if (
            row_width[
                y_coordinate
            ]
            >= threshold
        ):
            return np.array(
                [
                    float(
                        (
                            min_x[
                                y_coordinate
                            ]
                            + max_x[
                                y_coordinate
                            ]
                        )
                        / 2.0
                    ),
                    float(
                        y_coordinate
                    ),
                ],
                dtype=float,
            )

    return np.array(
        [
            float(
                (
                    min_x[y_top]
                    + max_x[y_top]
                )
                / 2.0
            ),
            float(
                y_top
            ),
        ],
        dtype=float,
    )


def correct_points(
    view: str,
    feet: np.ndarray,
    head: np.ndarray,
    body_mask: Optional[np.ndarray],
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict,
]:
    """
    Corrige los puntos de cabeza y pies usando la máscara corporal.
    """

    info = {
        "feet_source": "pose",
        "head_source": "pose",
        "head_move_px": 0.0,
    }

    if body_mask is None:
        return (
            feet,
            head,
            info,
        )

    if (
        USE_MASK_FEET_LATERAL
        and view in LATERAL_VIEWS
    ):
        mask_feet = feet_from_mask(
            body_mask
        )

        if (
            mask_feet is not None
            and mask_feet[1]
            >= feet[1] - 5
        ):
            feet = mask_feet

            info[
                "feet_source"
            ] = "mask_bottom"

    if USE_MASK_HEAD:
        mask_head = head_from_mask_trim_hair(
            body_mask
        )

        if mask_head is not None:
            maximum_move = (
                MAX_HEAD_MOVE_FRAC
                * (
                    feet[1]
                    - head[1]
                )
            )

            if (
                abs(
                    mask_head[1]
                    - head[1]
                )
                <= maximum_move
            ):
                info[
                    "head_move_px"
                ] = float(
                    head[1]
                    - mask_head[1]
                )

                head = mask_head

                info[
                    "head_source"
                ] = "mask_trim_hair"

    return (
        feet,
        head,
        info,
    )


# ============================================================
# GEOMETRÍA DEL MARCADOR
# ============================================================

def _angle(
    point_1: np.ndarray,
    point_2: np.ndarray,
) -> float:
    """
    Calcula el ángulo absoluto de una arista.
    """

    angle = np.degrees(
        np.arctan2(
            point_2[1]
            - point_1[1],
            point_2[0]
            - point_1[0],
        )
    )

    return float(
        abs(angle)
        % 180.0
    )


def marker_vertical_edge(
    corners_4: np.ndarray,
) -> Optional[
    tuple[
        np.ndarray,
        np.ndarray,
    ]
]:
    """
    Selecciona la arista más vertical del marcador ArUco.
    """

    corners = np.asarray(
        corners_4,
        dtype=float,
    ).reshape(
        4,
        2,
    )

    edges = [
        (
            corners[0],
            corners[1],
        ),
        (
            corners[1],
            corners[2],
        ),
        (
            corners[2],
            corners[3],
        ),
        (
            corners[3],
            corners[0],
        ),
    ]

    best_edge = None
    best_score = -np.inf

    for point_1, point_2 in edges:
        edge_length = float(
            np.linalg.norm(
                point_2
                - point_1
            )
        )

        if edge_length < MIN_EDGE_LEN_PX:
            continue

        verticality = (
            1.0
            - abs(
                _angle(
                    point_1,
                    point_2,
                )
                - 90.0
            )
            / 90.0
        )

        if verticality > best_score:
            if (
                point_1[1]
                >= point_2[1]
            ):
                base = point_1
                top = point_2
            else:
                base = point_2
                top = point_1

            best_edge = (
                base,
                top,
            )

            best_score = verticality

    return best_edge


def select_reference_cube(
    markers: list,
    feet_y: float,
) -> Optional[
    tuple[
        np.ndarray,
        np.ndarray,
    ]
]:
    """
    Selecciona el marcador más próximo verticalmente a los pies.
    """

    best_edge = None
    best_distance = np.inf

    for marker_corners in markers:
        corners = np.asarray(
            marker_corners,
            dtype=float,
        ).reshape(
            4,
            2,
        )

        edge = marker_vertical_edge(
            corners
        )

        if edge is None:
            continue

        center_y = float(
            corners[:, 1].mean()
        )

        distance = abs(
            center_y
            - feet_y
        )

        if distance < best_distance:
            best_edge = edge
            best_distance = distance

    return best_edge


# ============================================================
# MÉTODO DEL HORIZONTE
# ============================================================

def horizon_from_cube(
    y_cube_base: float,
    y_cube_top: float,
    camera_height_cm: float,
    reference_height_cm: float = ARUCO_MARKER_SIZE_CM,
) -> float:
    """
    Calcula el horizonte a partir de una referencia vertical conocida.
    """

    values = np.asarray(
        [
            y_cube_base,
            y_cube_top,
            camera_height_cm,
            reference_height_cm,
        ],
        dtype=float,
    )

    if not np.all(
        np.isfinite(values)
    ):
        return np.nan

    if reference_height_cm <= 0:
        return np.nan

    return float(
        y_cube_base
        - camera_height_cm
        * (
            y_cube_base
            - y_cube_top
        )
        / reference_height_cm
    )


def height_from_horizon(
    camera_height_cm: float,
    y_feet: float,
    y_head: float,
    y_horizon: float,
) -> float:
    """
    Estima la altura de la persona utilizando el horizonte.
    """

    values = np.asarray(
        [
            camera_height_cm,
            y_feet,
            y_head,
            y_horizon,
        ],
        dtype=float,
    )

    if not np.all(
        np.isfinite(values)
    ):
        return np.nan

    denominator = float(
        y_feet
        - y_horizon
    )

    body_height_px = float(
        y_feet
        - y_head
    )

    if (
        abs(
            denominator
        )
        < 1e-6
        or body_height_px <= 0
    ):
        return np.nan

    return float(
        camera_height_cm
        * body_height_px
        / denominator
    )


# ============================================================
# PROCESAMIENTO DE UNA IMAGEN
# ============================================================

def estimate_height_horizon(
    image: np.ndarray,
    view: str = "",
) -> dict:
    """
    Extrae cabeza, pies, cubo, horizonte propio y altura propia.
    """

    view = str(
        view
    ).lower()

    output = {
        "status": "ok",
        "H_horizon_cube": np.nan,
        "y_horizon_cube": np.nan,
        "y_feet": np.nan,
        "y_head": np.nan,
        "height_px": np.nan,
        "y_feet_single": np.nan,
        "y_feet_mask": np.nan,
        "feet": None,
        "head": None,
        "cube_base": None,
        "cube_top": None,
        "feet_source": "pose",
        "head_source": "pose",
        "head_move_px": 0.0,
        "img_h": (
            image.shape[0]
            if image is not None
            else np.nan
        ),
    }

    if image is None:
        output[
            "status"
        ] = "image_read_error"

        return output

    landmarks, body_mask = (
        estimate_body_height_pixels(
            image,
            view=view,
        )
    )

    landmark_status = str(
        landmarks.get(
            "landmark_status",
            "",
        )
    )

    if not landmark_status.startswith(
        "ok"
    ):
        output[
            "status"
        ] = (
            "person_failed_"
            f"{landmark_status}"
        )

        return output

    feet = np.array(
        [
            landmarks[
                "feet_x_px"
            ],
            landmarks[
                "feet_y_px"
            ],
        ],
        dtype=float,
    )

    head = np.array(
        [
            landmarks[
                "head_x_px"
            ],
            landmarks[
                "head_y_px"
            ],
        ],
        dtype=float,
    )

    output[
        "y_feet_single"
    ] = float(
        feet[1]
    )

    mask_feet = (
        feet_from_mask(
            body_mask
        )
        if body_mask is not None
        else None
    )

    if mask_feet is not None:
        output[
            "y_feet_mask"
        ] = float(
            mask_feet[1]
        )

    (
        feet,
        head,
        correction_info,
    ) = correct_points(
        view=view,
        feet=feet,
        head=head,
        body_mask=body_mask,
    )

    output.update(
        correction_info
    )

    output[
        "feet"
    ] = feet

    output[
        "head"
    ] = head

    output[
        "y_feet"
    ] = float(
        feet[1]
    )

    output[
        "y_head"
    ] = float(
        head[1]
    )

    output[
        "height_px"
    ] = float(
        feet[1]
        - head[1]
    )

    gray_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    corners, ids = detect_aruco_markers(
        gray_image
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

    if not markers:
        output[
            "status"
        ] = "no_aruco"

        return output

    selected_edge = select_reference_cube(
        markers=markers,
        feet_y=feet[1],
    )

    if selected_edge is None:
        output[
            "status"
        ] = "no_cube_edge"

        return output

    cube_base, cube_top = selected_edge

    output[
        "cube_base"
    ] = cube_base

    output[
        "cube_top"
    ] = cube_top

    y_horizon_cube = horizon_from_cube(
        y_cube_base=cube_base[1],
        y_cube_top=cube_top[1],
        camera_height_cm=CAMERA_HEIGHT_CM,
    )

    output[
        "y_horizon_cube"
    ] = float(
        y_horizon_cube
    )

    cube_height = height_from_horizon(
        camera_height_cm=CAMERA_HEIGHT_CM,
        y_feet=feet[1],
        y_head=head[1],
        y_horizon=y_horizon_cube,
    )

    output[
        "H_horizon_cube"
    ] = cube_height

    if not plausible_height(
        cube_height
    ):
        output[
            "status"
        ] = "horizon_out_of_range"

    return output


# ============================================================
# DIBUJO DE VERIFICACIÓN
# ============================================================

def draw_corrected_horizon(
    image: np.ndarray,
    record: dict,
    output_path: Optional[
        Path
    ] = None,
) -> np.ndarray:
    """
    Dibuja el horizonte propio, el compartido y el final.
    """

    canvas = image.copy()

    image_height, image_width = (
        canvas.shape[:2]
    )

    horizons = [
        (
            "compartido",
            record.get(
                "y_horizon_shared",
                np.nan,
            ),
            (
                0,
                220,
                220,
            ),
            2,
        ),
        (
            "propio",
            record.get(
                "y_horizon_cube",
                np.nan,
            ),
            (
                255,
                0,
                255,
            ),
            2,
        ),
        (
            "FINAL",
            record.get(
                "y_horizon_corrected",
                np.nan,
            ),
            (
                0,
                255,
                0,
            ),
            4,
        ),
    ]

    for (
        label,
        y_value,
        color,
        thickness,
    ) in horizons:
        if not np.isfinite(
            y_value
        ):
            continue

        y_position = int(
            np.clip(
                y_value,
                0,
                image_height - 1,
            )
        )

        cv2.line(
            canvas,
            (
                0,
                y_position,
            ),
            (
                image_width,
                y_position,
            ),
            color,
            thickness,
            cv2.LINE_AA,
        )

        put_label(
            canvas,
            (
                f"horizonte {label} "
                f"(y={y_value:.0f})"
            ),
            (
                30,
                max(
                    y_position - 8,
                    20,
                ),
            ),
            color=color,
        )

    if record.get(
        "cube_base"
    ) is not None:
        cube_base = tuple(
            np.int32(
                record[
                    "cube_base"
                ]
            )
        )

        cube_top = tuple(
            np.int32(
                record[
                    "cube_top"
                ]
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

    if (
        record.get(
            "feet"
        ) is not None
        and record.get(
            "head"
        ) is not None
    ):
        feet = tuple(
            np.int32(
                record[
                    "feet"
                ]
            )
        )

        head = tuple(
            np.int32(
                record[
                    "head"
                ]
            )
        )

        cv2.line(
            canvas,
            feet,
            head,
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
            head,
            7,
            (
                0,
                255,
                255,
            ),
            -1,
        )

        cv2.circle(
            canvas,
            feet,
            7,
            (
                0,
                0,
                255,
            ),
            -1,
        )

    manual_height = record.get(
        "height_manual_cm",
        np.nan,
    )

    predicted_height = record.get(
        "H_horizon_corrected",
        np.nan,
    )

    error = (
        predicted_height
        - manual_height
        if (
            np.isfinite(
                predicted_height
            )
            and np.isfinite(
                manual_height
            )
        )
        else np.nan
    )

    lines = [
        (
            "Experimento 2 corregido "
            f"(h_cam={CAMERA_HEIGHT_CM:.0f} cm)"
        ),
        (
            "vista: "
            f"{record.get('view', '')}"
        ),
        (
            "fuente: "
            f"{record.get('horizon_corrected_source', '')}"
        ),
        (
            "fiabilidad: "
            f"{record.get('height_reliability', '')}"
        ),
        (
            f"manual: {manual_height:.1f} cm"
            if np.isfinite(
                manual_height
            )
            else "manual: NA"
        ),
        (
            f"estimada: {predicted_height:.1f} cm "
            f"| error {error:+.1f}"
            if np.isfinite(
                predicted_height
            )
            else "estimada: NA"
        ),
    ]

    box_top = (
        image_height
        - 20
        - 30
        * len(
            lines
        )
    )

    cv2.rectangle(
        canvas,
        (
            20,
            box_top,
        ),
        (
            900,
            image_height - 15,
        ),
        (
            0,
            0,
            0,
        ),
        -1,
    )

    for index, text in enumerate(
        lines
    ):
        y_text = (
            image_height
            - 25
            - 30
            * (
                len(lines)
                - 1
                - index
            )
        )

        put_label(
            canvas,
            text,
            (
                30,
                y_text,
            ),
            color=(
                0,
                255,
                255,
            ),
        )

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

    return canvas


# ============================================================
# RUNNER PRINCIPAL
# ============================================================

def run_corrected_horizon_experiment(
    assets_by_subject: dict,
    manual_heights_df: pd.DataFrame,
    max_images: Optional[int] = None,
    save_debug: bool = True,
) -> pd.DataFrame:
    """
    Ejecuta el método del horizonte corregido con fallbacks LOO.
    """

    records = []
    processed_images = 0

    # ========================================================
    # PASADA 1: HORIZONTE PROPIO DE CADA IMAGEN
    # ========================================================

    for _, assets in assets_by_subject.items():
        for asset in assets:
            if (
                max_images is not None
                and processed_images
                >= max_images
            ):
                break

            image = cv2.imread(
                str(
                    asset.path
                )
            )

            if image is None:
                continue

            try:
                result = estimate_height_horizon(
                    image=image,
                    view=asset.view,
                )

            except Exception as exception:
                result = {
                    "status": (
                        "error_"
                        f"{type(exception).__name__}"
                    ),
                    "H_horizon_cube": np.nan,
                    "y_horizon_cube": np.nan,
                    "y_feet": np.nan,
                    "y_head": np.nan,
                    "height_px": np.nan,
                    "y_feet_single": np.nan,
                    "y_feet_mask": np.nan,
                    "feet": None,
                    "head": None,
                    "cube_base": None,
                    "cube_top": None,
                    "feet_source": "-",
                    "head_source": "-",
                    "head_move_px": 0.0,
                    "img_h": image.shape[0],
                }

            result[
                "site"
            ] = asset.site

            result[
                "subject_id"
            ] = asset.subject_id

            result[
                "view"
            ] = str(
                asset.view
            ).lower()

            result[
                "image_path"
            ] = str(
                asset.path
            )

            result[
                "height_manual_cm"
            ] = get_manual_height_for_subject(
                manual_heights_df,
                asset.subject_id,
            )

            records.append(
                result
            )

            processed_images += 1

        if (
            max_images is not None
            and processed_images
            >= max_images
        ):
            break

    if not records:
        raise ValueError(
            "No se ha podido procesar ninguna imagen."
        )

    subject_ids = sorted(
        {
            record[
                "subject_id"
            ]
            for record in records
        },
        key=str,
    )

    # ========================================================
    # FUNCIONES DE VALIDEZ
    # ========================================================

    def valid_front_back_horizon(
        record: dict,
    ) -> bool:
        return bool(
            record.get(
                "view"
            ) in RELIABLE_VIEWS
            and record.get(
                "status"
            ) == "ok"
            and np.isfinite(
                record.get(
                    "y_horizon_cube",
                    np.nan,
                )
            )
            and plausible_height(
                record.get(
                    "H_horizon_cube",
                    np.nan,
                )
            )
        )

    def valid_lateral_horizon(
        record: dict,
    ) -> bool:
        return bool(
            record.get(
                "view"
            ) in LATERAL_VIEWS
            and record.get(
                "status"
            ) == "ok"
            and np.isfinite(
                record.get(
                    "y_horizon_cube",
                    np.nan,
                )
            )
            and plausible_height(
                record.get(
                    "H_horizon_cube",
                    np.nan,
                )
            )
        )

    # ========================================================
    # HORIZONTES FRONT/BACK DEL MISMO SUJETO
    # ========================================================

    front_back_horizon_by_subject = {}

    for subject_id in subject_ids:
        subject_values = [
            record[
                "y_horizon_cube"
            ]
            for record in records
            if (
                record[
                    "subject_id"
                ]
                == subject_id
                and valid_front_back_horizon(
                    record
                )
            )
        ]

        front_back_horizon_by_subject[
            subject_id
        ] = (
            float(
                np.median(
                    subject_values
                )
            )
            if subject_values
            else np.nan
        )

    # ========================================================
    # HORIZONTES LATERALES DEL MISMO SUJETO
    # ========================================================

    lateral_horizon_by_subject = {}

    for subject_id in subject_ids:
        subject_values = [
            record[
                "y_horizon_cube"
            ]
            for record in records
            if (
                record[
                    "subject_id"
                ]
                == subject_id
                and valid_lateral_horizon(
                    record
                )
            )
        ]

        lateral_horizon_by_subject[
            subject_id
        ] = (
            float(
                np.median(
                    subject_values
                )
            )
            if subject_values
            else np.nan
        )

    # ========================================================
    # HORIZONTE GLOBAL FRONT/BACK LEAVE-ONE-SUBJECT-OUT
    # ========================================================

    global_front_back_horizon_by_subject = {}

    for subject_id in subject_ids:
        other_subject_values = [
            record[
                "y_horizon_cube"
            ]
            for record in records
            if (
                record[
                    "subject_id"
                ]
                != subject_id
                and valid_front_back_horizon(
                    record
                )
            )
        ]

        global_front_back_horizon_by_subject[
            subject_id
        ] = (
            float(
                np.median(
                    other_subject_values
                )
            )
            if other_subject_values
            else np.nan
        )

    # ========================================================
    # HORIZONTE GLOBAL LATERAL LEAVE-ONE-SUBJECT-OUT
    # ========================================================

    global_lateral_horizon_by_subject = {}

    for subject_id in subject_ids:
        other_subject_values = [
            record[
                "y_horizon_cube"
            ]
            for record in records
            if (
                record[
                    "subject_id"
                ]
                != subject_id
                and valid_lateral_horizon(
                    record
                )
            )
        ]

        global_lateral_horizon_by_subject[
            subject_id
        ] = (
            float(
                np.median(
                    other_subject_values
                )
            )
            if other_subject_values
            else np.nan
        )

    # ========================================================
    # PASADA 2: SELECCIÓN JERÁRQUICA DEL HORIZONTE
    # ========================================================

    for record in records:
        subject_id = record[
            "subject_id"
        ]

        view = record[
            "view"
        ]

        y_cube = record.get(
            "y_horizon_cube",
            np.nan,
        )

        cube_height = record.get(
            "H_horizon_cube",
            np.nan,
        )

        cube_is_valid = bool(
            record.get(
                "status"
            )
            == "ok"
            and np.isfinite(
                y_cube
            )
            and plausible_height(
                cube_height
            )
        )

        y_front_back_subject = (
            front_back_horizon_by_subject.get(
                subject_id,
                np.nan,
            )
        )

        y_lateral_subject = (
            lateral_horizon_by_subject.get(
                subject_id,
                np.nan,
            )
        )

        y_global_front_back_loo = (
            global_front_back_horizon_by_subject.get(
                subject_id,
                np.nan,
            )
        )

        y_global_lateral_loo = (
            global_lateral_horizon_by_subject.get(
                subject_id,
                np.nan,
            )
        )

        y_shared = (
            y_front_back_subject
            if np.isfinite(
                y_front_back_subject
            )
            else y_global_front_back_loo
        )

        record[
            "y_horizon_shared"
        ] = y_shared

        record[
            "horizon_shared_source"
        ] = (
            "subject_front_back"
            if np.isfinite(
                y_front_back_subject
            )
            else "global_front_back_fallback_loo"
        )

        record[
            "y_global_front_back_loo"
        ] = y_global_front_back_loo

        record[
            "y_global_lateral_loo"
        ] = y_global_lateral_loo

        # ----------------------------------------------------
        # LEFT / RIGHT
        # ----------------------------------------------------

        if view in LATERAL_VIEWS:

            if cube_is_valid:
                selected_horizon = y_cube

                selected_source = (
                    "per_image_cube_lateral"
                )

                reliability = "high"

            elif np.isfinite(
                y_lateral_subject
            ):
                selected_horizon = (
                    y_lateral_subject
                )

                selected_source = (
                    "paired_lateral_subject_fallback"
                )

                reliability = "medium"

            elif np.isfinite(
                y_shared
            ):
                # Sin cubo lateral fiable: el mejor recurso es el horizonte
                # front/back DEL PROPIO SUJETO. La camara no se movio entre
                # vistas, asi que su horizonte fisico es el mismo; es mas fiable
                # que el global lateral (que mezcla otros sujetos) y evita los
                # fallos catastroficos observados cuando el cubo lateral se
                # detecta mal.
                selected_horizon = y_shared

                selected_source = (
                    "subject_front_back_lateral_fallback"
                )

                reliability = "medium"

            elif np.isfinite(
                y_global_lateral_loo
            ):
                selected_horizon = (
                    y_global_lateral_loo
                )

                selected_source = (
                    "global_lateral_fallback_loo"
                )

                reliability = "low"

            else:
                selected_horizon = np.nan

                selected_source = (
                    "no_valid_horizon"
                )

                reliability = "unavailable"

        # ----------------------------------------------------
        # FRONT / BACK
        # ----------------------------------------------------

        elif view in RELIABLE_VIEWS:

            if np.isfinite(
                y_front_back_subject
            ):
                selected_horizon = (
                    y_front_back_subject
                )

                selected_source = (
                    "subject_front_back"
                )

                reliability = "high"

            elif cube_is_valid:
                selected_horizon = y_cube

                selected_source = (
                    "per_image_cube_front_back_fallback"
                )

                reliability = "medium"

            elif np.isfinite(
                y_global_front_back_loo
            ):
                selected_horizon = (
                    y_global_front_back_loo
                )

                selected_source = (
                    "global_front_back_fallback_loo"
                )

                reliability = "low"

            elif np.isfinite(
                y_global_lateral_loo
            ):
                selected_horizon = (
                    y_global_lateral_loo
                )

                selected_source = (
                    "global_lateral_last_resort_loo"
                )

                reliability = "very_low"

            else:
                selected_horizon = np.nan

                selected_source = (
                    "no_valid_horizon"
                )

                reliability = "unavailable"

        # ----------------------------------------------------
        # VISTA DESCONOCIDA
        # ----------------------------------------------------

        else:
            if cube_is_valid:
                selected_horizon = y_cube

                selected_source = (
                    "per_image_cube_unknown_view"
                )

                reliability = "medium"

            elif np.isfinite(
                y_shared
            ):
                selected_horizon = y_shared

                selected_source = (
                    "shared_unknown_view_fallback"
                )

                reliability = "low"

            elif np.isfinite(
                y_global_lateral_loo
            ):
                selected_horizon = (
                    y_global_lateral_loo
                )

                selected_source = (
                    "global_lateral_unknown_view_fallback_loo"
                )

                reliability = "very_low"

            else:
                selected_horizon = np.nan

                selected_source = (
                    "no_valid_horizon"
                )

                reliability = "unavailable"

        record[
            "y_horizon_corrected"
        ] = selected_horizon

        record[
            "horizon_corrected_source"
        ] = selected_source

        record[
            "height_reliability"
        ] = reliability

        y_feet = record.get(
            "y_feet",
            np.nan,
        )

        y_head = record.get(
            "y_head",
            np.nan,
        )

        manual_height = record.get(
            "height_manual_cm",
            np.nan,
        )

        record[
            "H_horizon_shared"
        ] = height_from_horizon(
            camera_height_cm=CAMERA_HEIGHT_CM,
            y_feet=y_feet,
            y_head=y_head,
            y_horizon=y_shared,
        )

        corrected_height = height_from_horizon(
            camera_height_cm=CAMERA_HEIGHT_CM,
            y_feet=y_feet,
            y_head=y_head,
            y_horizon=selected_horizon,
        )

        record[
            "H_horizon_corrected"
        ] = corrected_height

        record[
            "error_horizon_cube"
        ] = (
            cube_height
            - manual_height
            if (
                np.isfinite(
                    cube_height
                )
                and np.isfinite(
                    manual_height
                )
            )
            else np.nan
        )

        record[
            "error_horizon_shared"
        ] = (
            record[
                "H_horizon_shared"
            ]
            - manual_height
            if (
                np.isfinite(
                    record[
                        "H_horizon_shared"
                    ]
                )
                and np.isfinite(
                    manual_height
                )
            )
            else np.nan
        )

        record[
            "error_horizon_corrected"
        ] = (
            corrected_height
            - manual_height
            if (
                np.isfinite(
                    corrected_height
                )
                and np.isfinite(
                    manual_height
                )
            )
            else np.nan
        )

        record[
            "abs_error_horizon_corrected"
        ] = (
            abs(
                record[
                    "error_horizon_corrected"
                ]
            )
            if np.isfinite(
                record[
                    "error_horizon_corrected"
                ]
            )
            else np.nan
        )

        record[
            "delta_horizon_px"
        ] = (
            float(
                y_cube
                - y_shared
            )
            if (
                np.isfinite(
                    y_cube
                )
                and np.isfinite(
                    y_shared
                )
            )
            else np.nan
        )

        image_height = record.get(
            "img_h",
            np.nan,
        )

        record[
            "delta_horizon_frac"
        ] = (
            float(
                record[
                    "delta_horizon_px"
                ]
                / image_height
            )
            if (
                np.isfinite(
                    record[
                        "delta_horizon_px"
                    ]
                )
                and np.isfinite(
                    image_height
                )
                and image_height > 0
            )
            else np.nan
        )

        for foot_variant in [
            "single",
            "mask",
        ]:
            y_variant = record.get(
                f"y_feet_{foot_variant}",
                np.nan,
            )

            variant_height = height_from_horizon(
                camera_height_cm=CAMERA_HEIGHT_CM,
                y_feet=y_variant,
                y_head=y_head,
                y_horizon=selected_horizon,
            )

            record[
                f"H_corrected_feet_{foot_variant}"
            ] = variant_height

            record[
                f"error_corrected_feet_{foot_variant}"
            ] = (
                variant_height
                - manual_height
                if (
                    np.isfinite(
                        variant_height
                    )
                    and np.isfinite(
                        manual_height
                    )
                )
                else np.nan
            )

        if save_debug:
            image = cv2.imread(
                record[
                    "image_path"
                ]
            )

            if image is not None:
                output_path = (
                    CORRECTED_HORIZON_DIR
                    / "debug"
                    / (
                        f"{record['subject_id']}_"
                        f"{record['view']}_"
                        f"{Path(record['image_path']).stem}_"
                        "corrected_horizon.png"
                    )
                )

                draw_corrected_horizon(
                    image=image,
                    record=record,
                    output_path=output_path,
                )

                record[
                    "verification_image_path"
                ] = str(
                    output_path
                )

    # ========================================================
    # DATAFRAME PRINCIPAL
    # ========================================================

    excluded_columns = {
        "feet",
        "head",
        "cube_base",
        "cube_top",
    }

    result_df = pd.DataFrame(
        [
            {
                key: value
                for key, value in record.items()
                if key not in excluded_columns
            }
            for record in records
        ]
    )

    CORRECTED_HORIZON_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    estimates_path = (
        CORRECTED_HORIZON_DIR
        / "corrected_horizon_estimates.csv"
    )

    result_df.to_csv(
        estimates_path,
        index=False,
    )

    # ========================================================
    # RESUMEN POR VISTA Y MÉTODO
    # ========================================================

    summary_rows = []

    methods = [
        (
            "corrected_horizon",
            "error_horizon_corrected",
        ),
        (
            "per_image_cube_horizon",
            "error_horizon_cube",
        ),
        (
            "shared_front_back_horizon",
            "error_horizon_shared",
        ),
    ]

    for view, group in result_df.groupby(
        "view"
    ):
        for (
            method_name,
            error_column,
        ) in methods:
            metrics = calculate_metrics(
                group[
                    error_column
                ]
            )

            summary_rows.append(
                {
                    "group": view,
                    "method": method_name,
                    **metrics,
                }
            )

    for (
        method_name,
        error_column,
    ) in methods:
        metrics = calculate_metrics(
            result_df[
                error_column
            ]
        )

        summary_rows.append(
            {
                "group": "GLOBAL",
                "method": method_name,
                **metrics,
            }
        )

    for group_name, views in [
        (
            "FRONT+BACK",
            RELIABLE_VIEWS,
        ),
        (
            "LEFT+RIGHT",
            LATERAL_VIEWS,
        ),
    ]:
        subset = result_df[
            result_df[
                "view"
            ].isin(
                views
            )
        ]

        metrics = calculate_metrics(
            subset[
                "error_horizon_corrected"
            ]
        )

        summary_rows.append(
            {
                "group": group_name,
                "method": "corrected_horizon",
                **metrics,
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        CORRECTED_HORIZON_DIR
        / "corrected_horizon_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # ========================================================
    # FUENTES Y FIABILIDAD
    # ========================================================

    source_summary_df = (
        result_df
        .groupby(
            [
                "view",
                "horizon_corrected_source",
                "height_reliability",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="n"
        )
    )

    source_summary_path = (
        CORRECTED_HORIZON_DIR
        / "corrected_horizon_source_summary.csv"
    )

    source_summary_df.to_csv(
        source_summary_path,
        index=False,
    )

    # ========================================================
    # ALINEACIÓN DEL HORIZONTE
    # ========================================================

    valid_alignment = result_df[
        np.isfinite(
            result_df[
                "delta_horizon_px"
            ]
        )
    ].copy()

    if not valid_alignment.empty:
        alignment_summary_df = (
            valid_alignment
            .groupby(
                "view"
            )[
                "delta_horizon_px"
            ]
            .agg(
                n="count",
                mean_px="mean",
                median_px="median",
                std_px="std",
                min_px="min",
                max_px="max",
            )
            .reset_index()
        )

    else:
        alignment_summary_df = pd.DataFrame(
            columns=[
                "view",
                "n",
                "mean_px",
                "median_px",
                "std_px",
                "min_px",
                "max_px",
            ]
        )

    alignment_path = (
        CORRECTED_HORIZON_DIR
        / "corrected_horizon_alignment_by_view.csv"
    )

    alignment_summary_df.to_csv(
        alignment_path,
        index=False,
    )

    # ========================================================
    # RESUMEN POR SUJETO
    # ========================================================

    subject_rows = []

    for (
        subject_id,
        subject_group,
    ) in result_df.groupby(
        "subject_id"
    ):
        manual_values = (
            subject_group[
                "height_manual_cm"
            ]
            .dropna()
        )

        predicted_values = (
            subject_group[
                "H_horizon_corrected"
            ]
            .dropna()
        )

        if (
            manual_values.empty
            or predicted_values.empty
        ):
            continue

        manual_height = float(
            manual_values.median()
        )

        predicted_mean = float(
            predicted_values.mean()
        )

        predicted_median = float(
            predicted_values.median()
        )

        subject_rows.append(
            {
                "subject_id": str(
                    subject_id
                ),
                "n_images": int(
                    len(
                        subject_group
                    )
                ),
                "height_manual_cm": (
                    manual_height
                ),
                "pred_corrected_mean_cm": (
                    predicted_mean
                ),
                "pred_corrected_median_cm": (
                    predicted_median
                ),
                "error_corrected_mean_cm": (
                    predicted_mean
                    - manual_height
                ),
                "error_corrected_median_cm": (
                    predicted_median
                    - manual_height
                ),
            }
        )

    subject_df = pd.DataFrame(
        subject_rows
    )

    subject_path = (
        CORRECTED_HORIZON_DIR
        / "corrected_horizon_subject_summary.csv"
    )

    subject_df.to_csv(
        subject_path,
        index=False,
    )

    # ========================================================
    # RESULTADOS POR TERMINAL
    # ========================================================

    print(
        "\n"
        "============================================================"
    )

    print(
        "EXPERIMENTO 2 CORREGIDO"
    )

    print(
        "Jerarquía de horizontes con fallback leave-one-subject-out"
    )

    print(
        "============================================================"
    )

    print(
        "\nEstados originales:"
    )

    print(
        result_df[
            "status"
        ].value_counts(
            dropna=False
        ).to_string()
    )

    print(
        "\nFuentes utilizadas:"
    )

    print(
        source_summary_df.to_string(
            index=False
        )
    )

    print(
        "\nResumen:"
    )

    print(
        summary_df.round(
            3
        ).to_string(
            index=False
        )
    )

    print(
        "\nDesalineación del horizonte:"
    )

    print(
        alignment_summary_df.round(
            3
        ).to_string(
            index=False
        )
    )

    print(
        "\nArchivos guardados:"
    )

    for output_path in [
        estimates_path,
        summary_path,
        source_summary_path,
        alignment_path,
        subject_path,
    ]:
        print(
            f" - {output_path}"
        )

    return result_df
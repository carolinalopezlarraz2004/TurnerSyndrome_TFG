"""
image_calibration.py

Estima una escala pixel-a-centimetro por imagen usando referencias fisicas
visibles y ejecuta el Experimento 1 (estimacion automatica de altura).

Metodos de calibracion soportados:
- aruco_marker: Colombia y (futuro) Brasil.
- circular_marker: imagenes de marcador de Barcelona.

Salidas:
1. image_calibration.csv: una fila por imagen.
2. calibration_references.csv: una fila por referencia detectada.
3. imagenes de verificacion anotadas en outputs/verification.

Experimento 1:
Estimacion automatica de altura. La segmentacion corporal usa un modelo
aprendido de persona (YOLO11-seg, clase COCO 'person'), que sustituye al
GrabCut anterior. GrabCut fallaba cuando el fondo (pared/puerta/suelo) tenia
un color parecido al de la ropa: marcaba como persona casi toda la imagen y la
altura salia absurda. La segmentacion aprendida da una mascara de instancia
limpia y estable.

Instalacion (en tu maquina):
    pip install ultralytics
Los pesos 'yolo11n-seg.pt' se descargan solos la primera vez.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.config import (
    ARUCO_DICT,
    ARUCO_MARKER_SIZE_CM,
    CIRCULAR_MARKER_DIAMETER_CM,
    CALIBRATION_METHOD_BY_SITE,
    IMAGE_CALIBRATION_TABLE,
    CALIBRATION_REFERENCES_TABLE,
    VERIFICATION_DIR,
)

# Configuracion del segmentador de persona. Si defines estos nombres en
# src/config.py se respetan; si no, se usan estos valores por defecto.
try:
    from src.config import PERSON_SEG_MODEL, PERSON_SEG_CONF
except Exception:
    PERSON_SEG_MODEL = "yolo11n-seg.pt"
    PERSON_SEG_CONF = 0.40

# COCO: la clase 'person' es el indice 0.
PERSON_CLASS_ID = 0


# ============================================================
# BASIC GEOMETRY
# ============================================================

def euclidean_distance(point_a, point_b) -> float:
    """Distancia euclidea entre dos puntos 2D."""
    point_a = np.array(point_a, dtype=float)
    point_b = np.array(point_b, dtype=float)
    return float(np.linalg.norm(point_a - point_b))


def coefficient_of_variation(values: list[float]) -> float:
    """
    Coeficiente de variacion de una lista de valores. Se usa como aviso: si dos
    referencias de la misma imagen dan escalas muy distintas, puede haber
    perspectiva o una deteccion inestable.
    """
    values = np.array(values, dtype=float)
    values = values[~np.isnan(values)]

    if len(values) < 2:
        return np.nan

    mean_value = np.mean(values)
    if mean_value == 0:
        return np.nan

    return float(np.std(values) / mean_value)


def summarize_reference_scales(reference_rows: list[dict]) -> dict:
    """Resultado de calibracion a nivel de imagen a partir de las referencias."""
    scales = [
        row["reference_scale_cm_per_pixel"]
        for row in reference_rows
        if not pd.isna(row["reference_scale_cm_per_pixel"])
    ]

    if len(scales) == 0:
        return {
            "scale_cm_per_pixel": np.nan,
            "scale_std_cm_per_pixel": np.nan,
            "scale_cv": np.nan,
            "n_references_detected": 0,
            "calibration_status": "no_marker_detected",
        }

    scale_cv = coefficient_of_variation(scales)

    # Umbral solo como aviso. La imagen no se descarta automaticamente porque la
    # verificacion visual forma parte del proyecto.
    if not np.isnan(scale_cv) and scale_cv > 0.04:
        calibration_status = "valid_low_confidence"
    else:
        calibration_status = "valid"

    return {
        "scale_cm_per_pixel": float(np.median(scales)),
        "scale_std_cm_per_pixel": float(np.std(scales)) if len(scales) > 1 else np.nan,
        "scale_cv": scale_cv,
        "n_references_detected": len(scales),
        "calibration_status": calibration_status,
    }


# ============================================================
# VISUAL VERIFICATION HELPERS
# ============================================================

def put_label(image, text: str, origin: tuple[int, int], color=(0, 255, 255)) -> None:
    """Dibuja texto legible sobre una imagen de verificacion."""
    cv2.putText(
        image, text, origin,
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
    )


def build_verification_filename(asset, image_path: Path, step: str) -> str:
    """Nombre de fichero claro para una imagen de verificacion."""
    return f"{asset.subject_id}_{asset.view}_{image_path.stem}_{step}.png"


# ============================================================
# ARUCO DETECTION
# ============================================================

def detect_aruco_markers(gray_image):
    """Detecta marcadores ArUco de forma compatible entre versiones de OpenCV."""
    aruco_dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)

    try:
        # OpenCV reciente.
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(gray_image)
    except AttributeError:
        # OpenCV antiguo.
        parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray_image, aruco_dictionary, parameters=parameters,
        )

    return corners, ids


def calibrate_with_aruco(image, asset, debug_image=None):
    """
    Estima la escala cm/pixel desde marcadores ArUco.

    1. Detecta marcadores.
    2. Mide los cuatro lados en pixeles.
    3. Escala local por marcador = lado_real_cm / lado_px.
    4. Escala de imagen = mediana de las escalas locales.
    5. Guarda una fila por marcador para verificacion.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids = detect_aruco_markers(gray)

    reference_rows = []

    if ids is None or len(corners) == 0:
        return summarize_reference_scales(reference_rows), reference_rows, debug_image

    ids_flat = ids.flatten()

    for reference_index, marker_corners in enumerate(corners):
        points = marker_corners.reshape(4, 2)

        side_lengths = [
            euclidean_distance(points[0], points[1]),
            euclidean_distance(points[1], points[2]),
            euclidean_distance(points[2], points[3]),
            euclidean_distance(points[3], points[0]),
        ]
        mean_side_px = float(np.mean(side_lengths))

        if max(side_lengths) > 0:
            squareness = float(min(side_lengths) / max(side_lengths))
        else:
            squareness = np.nan

        if mean_side_px > 0:
            scale = ARUCO_MARKER_SIZE_CM / mean_side_px
        else:
            scale = np.nan

        center_x, center_y = points.mean(axis=0)
        marker_id = int(ids_flat[reference_index])

        reference_rows.append({
            "site": asset.site,
            "subject_id": asset.subject_id,
            "view": asset.view,
            "image_path": str(asset.path),
            "reference_index": reference_index,
            "reference_type": "aruco_marker",
            "reference_id": marker_id,
            "center_x_px": float(center_x),
            "center_y_px": float(center_y),
            "reference_size_px": mean_side_px,
            "reference_size_cm": ARUCO_MARKER_SIZE_CM,
            "reference_scale_cm_per_pixel": scale,
            "detection_quality": squareness,
        })

        if debug_image is not None:
            points_int = points.astype(int)
            cv2.polylines(debug_image, [points_int], True, (0, 255, 0), 3)
            cv2.circle(debug_image, (int(center_x), int(center_y)), 4, (0, 0, 255), -1)
            put_label(
                debug_image,
                f"ID {marker_id} | {mean_side_px:.1f}px | {scale:.5f} cm/px",
                (int(center_x) + 10, int(center_y) - 10),
            )

    result = summarize_reference_scales(reference_rows)
    return result, reference_rows, debug_image


# ============================================================
# CIRCULAR MARKER DETECTION
# ============================================================

def calibrate_with_circular_markers(image, asset, debug_image=None):
    """Estima la escala cm/pixel desde los marcadores circulares de Barcelona."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 0
    params.filterByArea = True
    params.minArea = 20
    params.maxArea = 3000
    params.filterByCircularity = True
    params.minCircularity = 0.50
    params.filterByInertia = True
    params.minInertiaRatio = 0.35
    params.filterByConvexity = False

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray_blur)

    reference_rows = []

    for reference_index, keypoint in enumerate(keypoints):
        center_x = float(keypoint.pt[0])
        center_y = float(keypoint.pt[1])
        diameter_px = float(keypoint.size)

        if diameter_px > 0:
            scale = CIRCULAR_MARKER_DIAMETER_CM / diameter_px
        else:
            scale = np.nan

        reference_rows.append({
            "site": asset.site,
            "subject_id": asset.subject_id,
            "view": asset.view,
            "image_path": str(asset.path),
            "reference_index": reference_index,
            "reference_type": "circular_marker",
            "reference_id": reference_index,
            "center_x_px": center_x,
            "center_y_px": center_y,
            "reference_size_px": diameter_px,
            "reference_size_cm": CIRCULAR_MARKER_DIAMETER_CM,
            "reference_scale_cm_per_pixel": scale,
            "detection_quality": float(keypoint.response),
        })

        if debug_image is not None:
            x, y = int(center_x), int(center_y)
            radius = int(diameter_px / 2)
            cv2.circle(debug_image, (x, y), radius, (0, 255, 0), 2)
            cv2.circle(debug_image, (x, y), 3, (0, 0, 255), -1)
            put_label(
                debug_image,
                f"M{reference_index} | {diameter_px:.1f}px | {scale:.5f} cm/px",
                (x + 10, y - 10),
            )

    result = summarize_reference_scales(reference_rows)
    return result, reference_rows, debug_image


# ============================================================
# ASSET LOGIC
# ============================================================

def get_image_type(asset) -> str:
    if asset.site in ["CO", "BR"]:
        return "aruco"
    if asset.site == "ES":
        return "markers" if asset.has_markers else "clean"
    return "unknown"


def should_calibrate_asset(asset) -> bool:
    if asset.site in ["CO", "BR"]:
        return True
    if asset.site == "ES":
        return asset.has_markers
    return False


def calibrate_image_asset(asset, save_debug: bool = True):
    """Calibra un ImageAsset."""
    image_path = Path(asset.path)

    result = {
        "site": asset.site,
        "subject_id": asset.subject_id,
        "view": asset.view,
        "image_type": get_image_type(asset),
        "image_path": str(image_path),
        "calibration_method": CALIBRATION_METHOD_BY_SITE.get(asset.site, "unknown"),
        "scale_cm_per_pixel": np.nan,
        "scale_std_cm_per_pixel": np.nan,
        "scale_cv": np.nan,
        "n_references_detected": 0,
        "calibration_status": "not_processed",
        "verification_image_path": "",
    }

    reference_rows = []
    image = cv2.imread(str(image_path))

    if image is None:
        result["calibration_status"] = "image_read_error"
        return result, reference_rows

    if not should_calibrate_asset(asset):
        result["calibration_status"] = "not_directly_calibrated"
        return result, reference_rows

    debug_image = image.copy() if save_debug else None

    if result["calibration_method"] == "aruco_marker":
        calibration_result, reference_rows, debug_image = calibrate_with_aruco(
            image=image, asset=asset, debug_image=debug_image,
        )
    elif result["calibration_method"] == "circular_marker":
        calibration_result, reference_rows, debug_image = calibrate_with_circular_markers(
            image=image, asset=asset, debug_image=debug_image,
        )
    else:
        calibration_result = {
            "scale_cm_per_pixel": np.nan,
            "scale_std_cm_per_pixel": np.nan,
            "scale_cv": np.nan,
            "n_references_detected": 0,
            "calibration_status": "unknown_calibration_method",
        }

    result.update(calibration_result)

    if save_debug and debug_image is not None:
        VERIFICATION_DIR.mkdir(parents=True, exist_ok=True)
        verification_path = VERIFICATION_DIR / build_verification_filename(
            asset=asset, image_path=image_path, step="calibration",
        )

        scale_text = ("scale=NA" if np.isnan(result["scale_cm_per_pixel"])
                      else f"scale={result['scale_cm_per_pixel']:.5f} cm/px")
        cv_text = "CV=NA" if np.isnan(result["scale_cv"]) else f"CV={result['scale_cv']:.3f}"

        summary_text = (
            f"{result['calibration_method']} | {scale_text} | "
            f"n={result['n_references_detected']} | {cv_text} | {result['calibration_status']}"
        )

        cv2.rectangle(debug_image, (20, 15), (1100, 60), (0, 0, 0), -1)
        put_label(debug_image, summary_text, (30, 45), color=(0, 255, 255))
        cv2.imwrite(str(verification_path), debug_image)
        result["verification_image_path"] = str(verification_path)

    return result, reference_rows


# ============================================================
# BATCH PROCESSING (CALIBRACION)
# ============================================================

def calibrate_image_assets(assets_by_subject: dict, save_debug: bool = True):
    calibration_rows = []
    reference_rows_all = []

    for _, assets in assets_by_subject.items():
        for asset in assets:
            calibration_row, reference_rows = calibrate_image_asset(
                asset=asset, save_debug=save_debug,
            )
            calibration_rows.append(calibration_row)
            reference_rows_all.extend(reference_rows)

    return pd.DataFrame(calibration_rows), pd.DataFrame(reference_rows_all)


def save_image_calibration_table(assets_by_subject: dict, save_debug: bool = True):
    calibration_df, references_df = calibrate_image_assets(
        assets_by_subject=assets_by_subject, save_debug=save_debug,
    )

    IMAGE_CALIBRATION_TABLE.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_REFERENCES_TABLE.parent.mkdir(parents=True, exist_ok=True)

    calibration_df.to_csv(IMAGE_CALIBRATION_TABLE, index=False)
    references_df.to_csv(CALIBRATION_REFERENCES_TABLE, index=False)

    return calibration_df, references_df


# ============================================================
# EXPERIMENT 1: SEGMENTACION DE PERSONA (YOLO11-seg)
# ============================================================

_PERSON_MODEL = None


def _get_person_model():
    """Carga perezosa del modelo YOLO11-seg. Devuelve None si no esta disponible."""
    global _PERSON_MODEL
    if _PERSON_MODEL is not None:
        return _PERSON_MODEL
    try:
        from ultralytics import YOLO
    except Exception:
        return None
    try:
        _PERSON_MODEL = YOLO(PERSON_SEG_MODEL)
    except Exception:
        _PERSON_MODEL = None
    return _PERSON_MODEL


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Conserva la componente conexa mas grande de una mascara binaria."""
    mask_uint8 = (mask > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, 8)
    if n_labels <= 1:
        return np.zeros_like(mask_uint8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Cierra huecos pequenos y se queda con la persona."""
    m = (mask > 0).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return keep_largest_component(m)


def segment_body(image: np.ndarray) -> tuple[np.ndarray, str, dict]:
    """
    Segmenta la persona con YOLO11-seg y devuelve la mascara de la instancia
    mas plausible (mayor confianza * area). Devuelve (mask, status, info).
    """
    h, w = image.shape[:2]
    empty = np.zeros((h, w), np.uint8)

    info = {
        "component_status": "no_component",
        "component_area_px": 0,
        "component_area_ratio": 0.0,
        "component_bbox_x": np.nan,
        "component_bbox_y": np.nan,
        "component_bbox_w": np.nan,
        "component_bbox_h": np.nan,
        "component_aspect_ratio": np.nan,
        "component_score": np.nan,
    }

    model = _get_person_model()
    if model is None:
        info["component_status"] = "model_unavailable"
        return empty, "model_unavailable", info

    try:
        result = model.predict(
            image, classes=[PERSON_CLASS_ID], conf=PERSON_SEG_CONF, verbose=False,
        )[0]
    except Exception:
        info["component_status"] = "inference_error"
        return empty, "inference_error", info

    if result.masks is None or len(result.masks) == 0:
        info["component_status"] = "no_person"
        return empty, "no_person", info

    masks = result.masks.data.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    image_area = float(h * w)

    best_score, best_mask = -1.0, None
    for i in range(masks.shape[0]):
        mask_i = cv2.resize(masks[i], (w, h), interpolation=cv2.INTER_NEAREST)
        mask_i = (mask_i > 0.5).astype(np.uint8)
        area_ratio = mask_i.sum() / image_area
        if area_ratio <= 0:
            continue
        score = float(confs[i]) * min(area_ratio / 0.15, 1.0)
        if score > best_score:
            best_score, best_mask = score, mask_i

    if best_mask is None:
        info["component_status"] = "empty_mask"
        return empty, "empty_mask", info

    body_mask = clean_mask(best_mask)
    ys, xs = np.where(body_mask > 0)
    if len(xs) == 0:
        info["component_status"] = "empty_mask"
        return empty, "empty_mask", info

    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    bw, bh = (x1 - x0 + 1), (y1 - y0 + 1)
    area = int(body_mask.sum())

    info.update({
        "component_status": "ok",
        "component_area_px": area,
        "component_area_ratio": area / image_area,
        "component_bbox_x": x0,
        "component_bbox_y": y0,
        "component_bbox_w": bw,
        "component_bbox_h": bh,
        "component_aspect_ratio": bh / bw if bw > 0 else np.nan,
        "component_score": float(best_score),
    })

    return body_mask, "ok", info


def get_head_feet_from_mask(mask: np.ndarray) -> dict:
    """
    Extrae cabeza (punto superior) y pies (punto de apoyo) de la mascara.

    - Se queda con la componente conexa mayor.
    - x de cabeza y pies = mediana de una banda horizontal (estable frente a
      pelo, sombras o ruido).
    - La x de la cabeza se restringe al eje central del torso, para no confundir
      la corona con un brazo extendido.

    Supuesto: la cabeza es el punto mas alto de la silueta (valido cuando las
    manos no estan por encima de la cabeza; se cumple en este dataset).
    """
    empty = {
        "head_x_px": np.nan, "head_y_px": np.nan,
        "feet_x_px": np.nan, "feet_y_px": np.nan,
        "height_px": np.nan, "body_mask_area_px": 0,
        "landmark_status": "empty_mask",
    }

    mask = keep_largest_component(mask)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return empty

    y_min, y_max = int(ys.min()), int(ys.max())
    height_px = float(y_max - y_min)
    if height_px <= 0:
        empty["body_mask_area_px"] = int(len(xs))
        empty["landmark_status"] = "invalid_height_px"
        return empty

    band = max(int(height_px * 0.03), 4)

    torso = (ys > y_min + 0.30 * height_px) & (ys < y_min + 0.60 * height_px)
    center_x = np.median(xs[torso]) if torso.any() else np.median(xs)
    half_w = 0.18 * (xs.max() - xs.min() + 1)

    # CABEZA
    top = ys <= y_min + band
    top_x = xs[top]
    near_axis = np.abs(top_x - center_x) <= half_w
    if near_axis.sum() >= 3:
        head_x = int(np.median(top_x[near_axis]))
    else:
        head_x = int(np.median(top_x))
    head_y = y_min

    # PIES
    bottom = ys >= y_max - band
    feet_x = int(np.median(xs[bottom]))
    feet_y = y_max

    return {
        "head_x_px": head_x, "head_y_px": head_y,
        "feet_x_px": feet_x, "feet_y_px": feet_y,
        "height_px": height_px, "body_mask_area_px": int(len(xs)),
        "landmark_status": "ok",
    }


def estimate_body_height_pixels(image: np.ndarray) -> tuple[dict, np.ndarray]:
    """
    Segmenta la persona y estima la altura cabeza-pies en pixeles. Devuelve el
    par (landmarks, body_mask) con las mismas claves que la version anterior.
    """
    body_mask, segmentation_status, component_info = segment_body(image)
    landmarks = get_head_feet_from_mask(body_mask)

    landmarks["segmentation_status"] = segmentation_status
    for key, value in component_info.items():
        landmarks[key] = value

    if segmentation_status != "ok" and landmarks["landmark_status"] == "ok":
        landmarks["landmark_status"] = f"ok_with_warning_{segmentation_status}"

    # Sanidad: una mascara que cubre casi toda la imagen o casi nada es
    # sospechosa (era el sintoma del fallo de GrabCut).
    if landmarks["landmark_status"].startswith("ok"):
        img_h = image.shape[0]
        frac = landmarks["height_px"] / img_h if img_h else 0
        if frac > 0.95:
            landmarks["landmark_status"] = "suspect_mask_covers_image"
        elif frac < 0.15:
            landmarks["landmark_status"] = "suspect_mask_too_small"

    return landmarks, body_mask


# ============================================================
# EXPERIMENT 1: ESTRATEGIAS DE ESCALA Y ALTURA
# ============================================================

def get_scale_from_best_quality(references_image_df: pd.DataFrame) -> dict:
    """
    Selecciona la escala de la referencia con mejor calidad geometrica. Para
    ArUco, detection_quality es la squareness (lado_corto/lado_largo); valores
    cercanos a 1 indican una cara mas fronto-paralela.
    """
    required = {"reference_scale_cm_per_pixel", "detection_quality",
                "reference_index", "reference_id"}

    if references_image_df.empty or not required.issubset(references_image_df.columns):
        return {"scale": np.nan, "reference_index": np.nan,
                "reference_id": np.nan, "reason": "no_reference"}

    valid = references_image_df.dropna(
        subset=["reference_scale_cm_per_pixel", "detection_quality"]
    )
    if valid.empty:
        return {"scale": np.nan, "reference_index": np.nan,
                "reference_id": np.nan, "reason": "no_valid_reference"}

    selected = valid.sort_values("detection_quality", ascending=False).iloc[0]
    return {
        "scale": float(selected["reference_scale_cm_per_pixel"]),
        "reference_index": selected["reference_index"],
        "reference_id": selected["reference_id"],
        "reason": "best_quality",
    }


def get_scale_from_closest_to_feet(references_image_df: pd.DataFrame,
                                   feet_x: float, feet_y: float) -> dict:
    """Selecciona la escala de la referencia detectada mas cercana a los pies."""
    required = {"reference_scale_cm_per_pixel", "center_x_px", "center_y_px",
                "reference_index", "reference_id"}

    if (references_image_df.empty
            or not required.issubset(references_image_df.columns)
            or pd.isna(feet_x) or pd.isna(feet_y)):
        return {"scale": np.nan, "reference_index": np.nan, "reference_id": np.nan,
                "distance_to_feet_px": np.nan, "reason": "no_reference_or_feet"}

    valid = references_image_df.dropna(
        subset=["reference_scale_cm_per_pixel", "center_x_px", "center_y_px"]
    ).copy()
    if valid.empty:
        return {"scale": np.nan, "reference_index": np.nan, "reference_id": np.nan,
                "distance_to_feet_px": np.nan, "reason": "no_valid_reference"}

    valid["distance_to_feet_px"] = np.sqrt(
        (valid["center_x_px"] - feet_x) ** 2 + (valid["center_y_px"] - feet_y) ** 2
    )
    selected = valid.sort_values("distance_to_feet_px", ascending=True).iloc[0]
    return {
        "scale": float(selected["reference_scale_cm_per_pixel"]),
        "reference_index": selected["reference_index"],
        "reference_id": selected["reference_id"],
        "distance_to_feet_px": float(selected["distance_to_feet_px"]),
        "reason": "closest_to_feet",
    }


def get_manual_height_for_subject(manual_heights_df: pd.DataFrame,
                                  subject_id: str) -> float:
    """Recupera HEIGHT_cm de la tabla antropometrica equalizada."""
    if manual_heights_df is None or manual_heights_df.empty:
        return np.nan
    if "ID" not in manual_heights_df.columns or "HEIGHT_cm" not in manual_heights_df.columns:
        return np.nan

    heights = manual_heights_df[["ID", "HEIGHT_cm"]].copy()
    heights["ID"] = heights["ID"].astype(str)
    match = heights[heights["ID"] == str(subject_id)]
    if match.empty:
        return np.nan
    return float(match.iloc[0]["HEIGHT_cm"])


def compute_height_estimates_from_scales(height_px: float, scale_median: float,
                                         best_quality_scale: dict,
                                         closest_feet_scale: dict) -> dict:
    """Calcula alturas con las tres estrategias de escala."""
    if pd.isna(height_px):
        height_px = np.nan

    height_cm_median = np.nan if pd.isna(scale_median) else height_px * scale_median
    height_cm_best = (np.nan if pd.isna(best_quality_scale["scale"])
                      else height_px * best_quality_scale["scale"])
    height_cm_feet = (np.nan if pd.isna(closest_feet_scale["scale"])
                      else height_px * closest_feet_scale["scale"])

    return {
        "height_cm_median_scale": height_cm_median,
        "height_cm_best_quality_scale": height_cm_best,
        "height_cm_closest_feet_scale": height_cm_feet,
    }


def add_height_errors(result: dict) -> dict:
    """Anade errores con signo (estimado - manual) a una fila de resultado."""
    manual_height = result.get("height_manual_cm", np.nan)
    for method_name in ["median_scale", "best_quality_scale", "closest_feet_scale"]:
        estimate = result.get(f"height_cm_{method_name}", np.nan)
        if pd.isna(manual_height) or pd.isna(estimate):
            result[f"error_cm_{method_name}"] = np.nan
        else:
            result[f"error_cm_{method_name}"] = estimate - manual_height
    return result


# ============================================================
# EXPERIMENT 1: IMAGEN DE VERIFICACION
# ============================================================

def save_height_experiment_verification_image(image_path: Path, output_path: Path,
                                               body_mask: np.ndarray, landmarks: dict,
                                               references_image_df: pd.DataFrame,
                                               best_quality_scale: dict,
                                               closest_feet_scale: dict,
                                               result: dict) -> None:
    """Guarda una imagen anotada del Experimento 1."""
    image = cv2.imread(str(image_path))
    if image is None:
        return

    overlay = image.copy()
    if body_mask is not None and body_mask.shape[:2] == image.shape[:2]:
        mask_bool = body_mask > 0
        overlay[mask_bool] = (
            0.65 * overlay[mask_bool] + 0.35 * np.array([0, 255, 0])
        ).astype(np.uint8)
    image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)

    # Referencias detectadas.
    if references_image_df is not None and not references_image_df.empty:
        for _, ref in references_image_df.iterrows():
            if pd.isna(ref.get("center_x_px", np.nan)) or pd.isna(ref.get("center_y_px", np.nan)):
                continue
            center = (int(ref["center_x_px"]), int(ref["center_y_px"]))
            cv2.circle(image, center, 5, (255, 255, 0), -1)
            put_label(image, f"ID {int(ref['reference_id'])}",
                      (center[0] + 8, center[1] - 8), color=(255, 255, 0))

    # Referencias seleccionadas por cada estrategia.
    for selected_scale, color, label in [
        (best_quality_scale, (0, 255, 0), "best"),
        (closest_feet_scale, (0, 0, 255), "feet"),
    ]:
        selected_index = selected_scale.get("reference_index", np.nan)
        if pd.isna(selected_index) or references_image_df is None or references_image_df.empty:
            continue
        sel = references_image_df[references_image_df["reference_index"] == selected_index]
        if sel.empty:
            continue
        sel = sel.iloc[0]
        if pd.isna(sel["center_x_px"]) or pd.isna(sel["center_y_px"]):
            continue
        center = (int(sel["center_x_px"]), int(sel["center_y_px"]))
        cv2.circle(image, center, 16, color, 3)
        put_label(image, label, (center[0] + 15, center[1] + 20), color=color)

    # Cabeza y pies.
    if landmarks.get("landmark_status", "") != "empty_mask":
        hx, hy = landmarks.get("head_x_px", np.nan), landmarks.get("head_y_px", np.nan)
        fx, fy = landmarks.get("feet_x_px", np.nan), landmarks.get("feet_y_px", np.nan)
        if not any(pd.isna(v) for v in [hx, hy, fx, fy]):
            head, feet = (int(hx), int(hy)), (int(fx), int(fy))
            cv2.circle(image, head, 8, (0, 255, 255), -1)
            put_label(image, "HEAD", (head[0] + 10, head[1] - 10))
            cv2.circle(image, feet, 8, (0, 0, 255), -1)
            put_label(image, "FEET", (feet[0] + 10, feet[1] - 10), color=(0, 0, 255))
            cv2.line(image, head, feet, (255, 0, 0), 3)

    height_manual = result.get("height_manual_cm", np.nan)
    height_px = result.get("height_px", np.nan)

    lines = [
        "Experiment 1 automatic height estimation",
        f"status: {result.get('status', '')}",
        f"segmentation: {result.get('segmentation_status', '')}",
        f"height_px: {height_px:.1f}" if not pd.isna(height_px) else "height_px: NA",
        f"manual HEIGHT_cm: {height_manual:.1f}" if not pd.isna(height_manual) else "manual HEIGHT_cm: NA",
        f"median: {result.get('height_cm_median_scale', np.nan):.1f} cm | error {result.get('error_cm_median_scale', np.nan):.1f} cm",
        f"best quality: {result.get('height_cm_best_quality_scale', np.nan):.1f} cm | error {result.get('error_cm_best_quality_scale', np.nan):.1f} cm",
        f"closest feet: {result.get('height_cm_closest_feet_scale', np.nan):.1f} cm | error {result.get('error_cm_closest_feet_scale', np.nan):.1f} cm",
    ]

    cv2.rectangle(image, (20, 15), (1030, 275), (0, 0, 0), -1)
    y0 = 45
    for i, line in enumerate(lines):
        put_label(image, line, (30, y0 + i * 30), color=(0, 255, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


# ============================================================
# EXPERIMENT 1: RUNNER
# ============================================================

def run_height_experiment_1_auto(assets_by_subject: dict, calibration_df: pd.DataFrame,
                                 references_df: pd.DataFrame,
                                 manual_heights_df: pd.DataFrame,
                                 max_images: int | None = None,
                                 save_debug: bool = True) -> pd.DataFrame:
    """
    Ejecuta el primer experimento de estimacion automatica de altura.

    Para cada imagen calibrada: segmenta la persona (YOLO11-seg), extrae cabeza
    y pies, calcula altura en pixeles y estima altura con tres estrategias de
    escala (mediana / mejor calidad / mas cercana a los pies), comparando con
    HEIGHT_cm y guardando una imagen de verificacion.
    """
    height_experiment_table = VERIFICATION_DIR / "height_estimates_experiment1.csv"

    results = []
    n_processed = 0

    for _, assets in assets_by_subject.items():
        for asset in assets:
            if max_images is not None and n_processed >= max_images:
                break

            image_path = Path(asset.path)
            image_path_str = str(image_path)

            image_cal = calibration_df[calibration_df["image_path"] == image_path_str]
            if image_cal.empty:
                continue
            image_cal = image_cal.iloc[0]

            if image_cal["calibration_status"] == "no_marker_detected":
                continue
            if pd.isna(image_cal["scale_cm_per_pixel"]):
                continue

            image = cv2.imread(image_path_str)
            if image is None:
                continue

            landmarks, body_mask = estimate_body_height_pixels(image)

            if str(landmarks["landmark_status"]).startswith("ok"):
                status = "ok"
            else:
                status = f"landmark_failed_{landmarks['landmark_status']}"

            references_image_df = references_df[references_df["image_path"] == image_path_str]

            best_quality_scale = get_scale_from_best_quality(references_image_df)
            closest_feet_scale = get_scale_from_closest_to_feet(
                references_image_df=references_image_df,
                feet_x=landmarks["feet_x_px"], feet_y=landmarks["feet_y_px"],
            )

            height_estimates = compute_height_estimates_from_scales(
                height_px=landmarks["height_px"],
                scale_median=float(image_cal["scale_cm_per_pixel"]),
                best_quality_scale=best_quality_scale,
                closest_feet_scale=closest_feet_scale,
            )

            height_manual_cm = get_manual_height_for_subject(
                manual_heights_df=manual_heights_df, subject_id=asset.subject_id,
            )

            result = {
                "site": asset.site,
                "subject_id": asset.subject_id,
                "view": asset.view,
                "image_path": image_path_str,
                "height_manual_cm": height_manual_cm,

                "head_x_px": landmarks.get("head_x_px", np.nan),
                "head_y_px": landmarks.get("head_y_px", np.nan),
                "feet_x_px": landmarks.get("feet_x_px", np.nan),
                "feet_y_px": landmarks.get("feet_y_px", np.nan),
                "height_px": landmarks.get("height_px", np.nan),

                "body_mask_area_px": landmarks.get("body_mask_area_px", np.nan),
                "component_area_ratio": landmarks.get("component_area_ratio", np.nan),
                "component_bbox_x": landmarks.get("component_bbox_x", np.nan),
                "component_bbox_y": landmarks.get("component_bbox_y", np.nan),
                "component_bbox_w": landmarks.get("component_bbox_w", np.nan),
                "component_bbox_h": landmarks.get("component_bbox_h", np.nan),
                "component_aspect_ratio": landmarks.get("component_aspect_ratio", np.nan),
                "component_score": landmarks.get("component_score", np.nan),

                "segmentation_status": landmarks.get("segmentation_status", ""),
                "landmark_status": landmarks.get("landmark_status", ""),

                "scale_median_all": float(image_cal["scale_cm_per_pixel"]),
                "scale_cv_all": image_cal["scale_cv"],
                "n_references_detected": image_cal["n_references_detected"],

                "scale_best_quality": best_quality_scale["scale"],
                "best_quality_reference_index": best_quality_scale["reference_index"],
                "best_quality_reference_id": best_quality_scale["reference_id"],

                "scale_closest_feet": closest_feet_scale["scale"],
                "closest_feet_reference_index": closest_feet_scale["reference_index"],
                "closest_feet_reference_id": closest_feet_scale["reference_id"],
                "closest_feet_distance_px": closest_feet_scale.get("distance_to_feet_px", np.nan),

                **height_estimates,

                "status": status,
                "verification_image_path": "",
            }

            result = add_height_errors(result)

            verification_path = (
                VERIFICATION_DIR
                / f"{asset.subject_id}_{asset.view}_{image_path.stem}_height_experiment1.png"
            )

            if save_debug:
                save_height_experiment_verification_image(
                    image_path=image_path, output_path=verification_path,
                    body_mask=body_mask, landmarks=landmarks,
                    references_image_df=references_image_df,
                    best_quality_scale=best_quality_scale,
                    closest_feet_scale=closest_feet_scale, result=result,
                )
                result["verification_image_path"] = str(verification_path)

            results.append(result)
            n_processed += 1

        if max_images is not None and n_processed >= max_images:
            break

    results_df = pd.DataFrame(results)
    height_experiment_table.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(height_experiment_table, index=False)
    return results_df


def summarize_height_experiment_1(results_df: pd.DataFrame) -> pd.DataFrame:
    """Resume los errores del Experimento 1 para las tres estrategias de escala."""
    if results_df.empty:
        return pd.DataFrame()

    summary_rows = []
    methods = [
        ("median_scale", "error_cm_median_scale"),
        ("best_quality_scale", "error_cm_best_quality_scale"),
        ("closest_feet_scale", "error_cm_closest_feet_scale"),
    ]

    for method_name, error_column in methods:
        if error_column not in results_df.columns:
            continue
        errors = results_df[error_column].dropna()
        if len(errors) == 0:
            summary_rows.append({"method": method_name, "n_valid": 0,
                                 "bias_cm": np.nan, "mae_cm": np.nan, "rmse_cm": np.nan})
            continue
        summary_rows.append({
            "method": method_name,
            "n_valid": int(len(errors)),
            "bias_cm": float(errors.mean()),
            "mae_cm": float(errors.abs().mean()),
            "rmse_cm": float(np.sqrt(np.mean(errors ** 2))),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = VERIFICATION_DIR / "height_estimates_experiment1_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    return summary_df
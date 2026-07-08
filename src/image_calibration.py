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
Segmentacion con YOLO11-seg + refinamiento de cabeza/pies con YOLO11-pose.
- PIES: tobillo para acotar + mascara para el contacto con el suelo.
- CABEZA: corona por offset sobre la linea de ojos (ignora pelo/mono).
Ademas: flag de perspectiva por imagen (cube_scale_ratio) y datos crudos de
pose (pose_eye_y, pose_face_scale, pose_face_mode) para poder calibrar el
offset de corona a posteriori con sweep_crown_offset_factor(), sin re-inferir.

Instalacion (en tu maquina):
    pip install ultralytics
Los pesos 'yolo11n-seg.pt' y 'yolo11n-pose.pt' se descargan solos.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # backend sin ventana, para guardar PNG
import matplotlib.pyplot as plt

from src.config import (
    ARUCO_DICT,
    ARUCO_MARKER_SIZE_CM,
    CIRCULAR_MARKER_DIAMETER_CM,
    CALIBRATION_METHOD_BY_SITE,
    IMAGE_CALIBRATION_TABLE,
    CALIBRATION_REFERENCES_TABLE,
    VERIFICATION_DIR,
)

# Segmentador de persona. Si defines estos nombres en src/config.py se respetan.
try:
    from src.config import PERSON_SEG_MODEL, PERSON_SEG_CONF
except Exception:
    PERSON_SEG_MODEL = "yolo11n-seg.pt"
    PERSON_SEG_CONF = 0.40

# Pose y constantes de refinamiento cabeza/pies + flag de perspectiva.
try:
    from src.config import (
        PERSON_POSE_MODEL, POSE_CONF,
        CROWN_ABOVE_EYES_FACTOR, CROWN_ABOVE_EYES_FACTOR_INTEROCULAR,
        FOOT_WINDOW_FRAC, MAX_FOOT_FRAC, PERSPECTIVE_RATIO_THRESHOLD,
    )
except Exception:
    PERSON_POSE_MODEL = "yolo11n-pose.pt"
    POSE_CONF = 0.30
    # Corona por encima de la linea de ojos. Ajustados a la baja respecto a la
    # primera version (1.6 / 2.2) porque el lote completo mostraba sobreestimacion.
    # Calibralos con sweep_crown_offset_factor() para fijar el optimo por datos.
    CROWN_ABOVE_EYES_FACTOR = 1.6            # relativo a distancia ojo-nariz
    CROWN_ABOVE_EYES_FACTOR_INTEROCULAR = 1.15  # relativo a distancia interocular
    FOOT_WINDOW_FRAC = 0.12                  # ancho ventana de pies (fraccion bbox)
    MAX_FOOT_FRAC = 0.08                     # tope apoyo por debajo del tobillo
    PERSPECTIVE_RATIO_THRESHOLD = 1.35       # umbral flag de perspectiva

# COCO: 'person' es la clase 0.
PERSON_CLASS_ID = 0

# Indices de keypoints COCO (YOLO-pose).
KP_NOSE, KP_LEYE, KP_REYE, KP_LEAR, KP_REAR = 0, 1, 2, 3, 4
KP_LANKLE, KP_RANKLE = 15, 16


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
_POSE_MODEL = None


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


def _get_pose_model():
    """Carga perezosa del modelo YOLO11-pose. Devuelve None si no esta disponible."""
    global _POSE_MODEL
    if _POSE_MODEL is not None:
        return _POSE_MODEL
    try:
        from ultralytics import YOLO
    except Exception:
        return None
    try:
        _POSE_MODEL = YOLO(PERSON_POSE_MODEL)
    except Exception:
        _POSE_MODEL = None
    return _POSE_MODEL


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


# ============================================================
# EXPERIMENT 1: POSE (refinamiento de cabeza y pies)
# ============================================================

def get_pose_keypoints(image: np.ndarray):
    """Devuelve los 17 keypoints COCO (x, y, conf) de la persona principal, o None."""
    model = _get_pose_model()
    if model is None:
        return None
    try:
        result = model.predict(
            image, classes=[PERSON_CLASS_ID], conf=PERSON_SEG_CONF, verbose=False,
        )[0]
    except Exception:
        return None

    if result.keypoints is None or len(result.keypoints) == 0:
        return None

    kdata = result.keypoints.data.cpu().numpy()  # (n, 17, 3)
    if kdata.shape[0] == 0:
        return None

    confs = result.boxes.conf.cpu().numpy() if result.boxes is not None else None
    if confs is not None and len(confs) == kdata.shape[0]:
        idx = int(np.argmax(confs))
    else:
        idx = 0
    return kdata[idx]


def feet_from_pose(mask: np.ndarray, kpts, bbox_w: float):
    """
    Punto de apoyo usando el tobillo para acotar y la mascara para el contacto.
    Devuelve (feet_x, feet_y) o None.
    """
    if kpts is None:
        return None

    ankles = [kpts[KP_LANKLE], kpts[KP_RANKLE]]
    ankles = [k for k in ankles if k[2] >= POSE_CONF]
    if not ankles:
        return None

    ankle_x = float(np.mean([k[0] for k in ankles]))
    ankle_y = float(max(k[1] for k in ankles))
    window = max(FOOT_WINDOW_FRAC * bbox_w, 15.0)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    sel = np.abs(xs - ankle_x) <= window
    if sel.sum() < 5:
        return None

    ground_y = int(ys[sel].max())

    # Tope: el apoyo no puede estar absurdamente por debajo del tobillo
    # (evita coger una sombra o una puntera muy proyectada).
    body_h = ys.max() - ys.min() + 1
    max_foot = MAX_FOOT_FRAC * body_h
    if ground_y > ankle_y + max_foot:
        ground_y = int(ankle_y + max_foot)

    band = max(int(0.02 * (ground_y - ankle_y + 1)), 4)
    bottom = sel & (ys >= ground_y - band)
    feet_x = int(np.median(xs[bottom])) if bottom.sum() > 0 else int(ankle_x)
    return feet_x, ground_y


def head_from_pose(mask: np.ndarray, kpts):
    """
    Corona estimada como offset por encima de la linea de ojos, ignorando pelo.
    Devuelve un dict con head_x, head_y y los datos crudos (eye_y, face_scale,
    face_mode) que permiten recalibrar el offset a posteriori. O None.
    """
    if kpts is None:
        return None

    nose, l_eye, r_eye = kpts[KP_NOSE], kpts[KP_LEYE], kpts[KP_REYE]
    eyes = [k for k in (l_eye, r_eye) if k[2] >= POSE_CONF]
    if len(eyes) == 0:
        return None

    eye_x = float(np.mean([k[0] for k in eyes]))
    eye_y = float(np.mean([k[1] for k in eyes]))

    if len(eyes) == 2:
        face_scale = float(np.hypot(l_eye[0] - r_eye[0], l_eye[1] - r_eye[1]))
        face_mode = "interocular"
        offset = CROWN_ABOVE_EYES_FACTOR_INTEROCULAR * face_scale
    elif nose[2] >= POSE_CONF:
        face_scale = float(max(abs(nose[1] - eye_y), 1.0))
        face_mode = "eye_nose"
        offset = CROWN_ABOVE_EYES_FACTOR * face_scale
    else:
        return None

    crown_y = eye_y - offset

    ys, _ = np.where(mask > 0)
    if len(ys) == 0:
        return None
    mask_top = float(ys.min())

    # La corona no puede estar por encima del pelo (mask_top) ni por debajo de
    # los ojos. El clamp con mask_top hace que el pelo por encima se ignore.
    crown_y = max(crown_y, mask_top)
    crown_y = min(crown_y, eye_y - 1)

    return {
        "head_x": int(round(eye_x)),
        "head_y": int(round(crown_y)),
        "eye_y": eye_y,
        "face_scale": face_scale,
        "face_mode": face_mode,
    }


def get_head_feet_from_mask(mask: np.ndarray) -> dict:
    """
    Extrae cabeza (punto superior) y pies (punto de apoyo) SOLO de la mascara.
    Se usa como fallback cuando pose no esta disponible.
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

    top = ys <= y_min + band
    top_x = xs[top]
    near_axis = np.abs(top_x - center_x) <= half_w
    if near_axis.sum() >= 3:
        head_x = int(np.median(top_x[near_axis]))
    else:
        head_x = int(np.median(top_x))
    head_y = y_min

    bottom = ys >= y_max - band
    feet_x = int(np.median(xs[bottom]))
    feet_y = y_max

    return {
        "head_x_px": head_x, "head_y_px": head_y,
        "feet_x_px": feet_x, "feet_y_px": feet_y,
        "height_px": height_px, "body_mask_area_px": int(len(xs)),
        "landmark_status": "ok",
    }


def get_head_feet(image: np.ndarray, body_mask: np.ndarray) -> dict:
    """
    Cabeza y pies combinando mascara (fallback) con pose (refinamiento).
    Registra la fuente usada (pose/mask) y, para la cabeza por pose, los datos
    crudos (pose_eye_y, pose_face_scale, pose_face_mode).
    """
    landmarks = get_head_feet_from_mask(body_mask)
    landmarks["head_source"] = "mask"
    landmarks["feet_source"] = "mask"
    landmarks["head_y_mask"] = landmarks["head_y_px"]
    landmarks["feet_y_mask"] = landmarks["feet_y_px"]
    landmarks["pose_eye_y"] = np.nan
    landmarks["pose_face_scale"] = np.nan
    landmarks["pose_face_mode"] = "none"

    if landmarks["landmark_status"] != "ok":
        return landmarks

    mask = keep_largest_component(body_mask)
    ys, xs = np.where(mask > 0)
    bbox_w = float(xs.max() - xs.min() + 1)

    kpts = get_pose_keypoints(image)

    feet_pose = feet_from_pose(mask, kpts, bbox_w)
    if feet_pose is not None:
        landmarks["feet_x_px"], landmarks["feet_y_px"] = feet_pose
        landmarks["feet_source"] = "pose"

    head_pose = head_from_pose(mask, kpts)
    if head_pose is not None:
        landmarks["head_x_px"] = head_pose["head_x"]
        landmarks["head_y_px"] = head_pose["head_y"]
        landmarks["head_source"] = "pose"
        landmarks["pose_eye_y"] = head_pose["eye_y"]
        landmarks["pose_face_scale"] = head_pose["face_scale"]
        landmarks["pose_face_mode"] = head_pose["face_mode"]

    landmarks["height_px"] = float(landmarks["feet_y_px"] - landmarks["head_y_px"])
    return landmarks


def estimate_body_height_pixels(image: np.ndarray) -> tuple[dict, np.ndarray]:
    """
    Segmenta la persona (YOLO11-seg) y estima la altura cabeza-pies en pixeles,
    refinando los puntos con pose (YOLO11-pose). Devuelve (landmarks, body_mask).
    """
    body_mask, segmentation_status, component_info = segment_body(image)
    landmarks = get_head_feet(image, body_mask)

    landmarks["segmentation_status"] = segmentation_status
    for key, value in component_info.items():
        landmarks[key] = value

    if segmentation_status != "ok" and landmarks["landmark_status"] == "ok":
        landmarks["landmark_status"] = f"ok_with_warning_{segmentation_status}"

    # Sanidad: una mascara que cubre casi toda la imagen o casi nada es sospechosa.
    if landmarks["landmark_status"].startswith("ok"):
        img_h = image.shape[0]
        frac = landmarks["height_px"] / img_h if img_h else 0
        if frac > 0.95:
            landmarks["landmark_status"] = "suspect_mask_covers_image"
        elif frac < 0.15:
            landmarks["landmark_status"] = "suspect_mask_too_small"

    return landmarks, body_mask


# ============================================================
# EXPERIMENT 1: FLAG DE PERSPECTIVA (discrepancia entre cubos)
# ============================================================

def compute_cube_perspective_flag(references_image_df: pd.DataFrame,
                                  image_width: float) -> dict:
    """
    Mide cuanto difieren las escalas de los DOS cubos del suelo (izq vs der).
    NO cambia la altura estimada: es solo un flag de calidad geometrica.
    """
    out = {"cube_scale_ratio": np.nan, "n_cubes_detected": 0, "perspective_flag": False}

    if references_image_df is None or references_image_df.empty:
        return out

    required = {"reference_scale_cm_per_pixel", "center_x_px", "detection_quality"}
    if not required.issubset(references_image_df.columns):
        return out

    df = references_image_df.dropna(
        subset=["reference_scale_cm_per_pixel", "center_x_px", "detection_quality"]
    )
    if df.empty:
        return out

    mid = image_width / 2.0
    reps = []
    for side_df in [df[df["center_x_px"] < mid], df[df["center_x_px"] >= mid]]:
        if not side_df.empty:
            rep = side_df.sort_values("detection_quality", ascending=False).iloc[0]
            reps.append(float(rep["reference_scale_cm_per_pixel"]))

    out["n_cubes_detected"] = len(reps)
    if len(reps) == 2 and min(reps) > 0:
        ratio = max(reps) / min(reps)
        out["cube_scale_ratio"] = float(ratio)
        out["perspective_flag"] = bool(ratio > PERSPECTIVE_RATIO_THRESHOLD)

    return out


# ============================================================
# EXPERIMENT 1: ESTRATEGIAS DE ESCALA Y ALTURA
# ============================================================

def get_scale_from_best_quality(references_image_df: pd.DataFrame) -> dict:
    """Selecciona la escala de la referencia con mejor calidad geometrica (squareness)."""
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


def get_scale_from_ground_line(references_image_df: pd.DataFrame,
                              feet_y: float) -> dict:
    """
    Selecciona la escala de la referencia cuya BASE esta a la misma linea de
    suelo que los pies (minima diferencia en la coordenada y de imagen).

    Idea: en el plano del suelo, la profundidad se lee en y. La referencia con
    la y mas parecida a la de los pies comparte plano de suelo con la persona,
    asi que su escala cm/px es la mas representativa. Robusto sobre todo en las
    vistas alineadas (front/back); en laterales el error residual es de
    profundidad y no lo resuelve la seleccion.
    """
    required = {"reference_scale_cm_per_pixel", "center_y_px",
                "reference_index", "reference_id"}

    if (references_image_df.empty
            or not required.issubset(references_image_df.columns)
            or pd.isna(feet_y)):
        return {"scale": np.nan, "reference_index": np.nan, "reference_id": np.nan,
                "delta_y_px": np.nan, "reason": "no_reference_or_feet"}

    valid = references_image_df.dropna(
        subset=["reference_scale_cm_per_pixel", "center_y_px"]
    ).copy()
    if valid.empty:
        return {"scale": np.nan, "reference_index": np.nan, "reference_id": np.nan,
                "delta_y_px": np.nan, "reason": "no_valid_reference"}

    valid["delta_y_px"] = (valid["center_y_px"] - feet_y).abs()
    selected = valid.sort_values("delta_y_px", ascending=True).iloc[0]
    return {
        "scale": float(selected["reference_scale_cm_per_pixel"]),
        "reference_index": selected["reference_index"],
        "reference_id": selected["reference_id"],
        "delta_y_px": float(selected["delta_y_px"]),
        "reason": "ground_line_y",
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
                                         closest_feet_scale: dict,
                                         ground_line_scale: dict) -> dict:
    """Calcula alturas con las tres estrategias de escala."""
    if pd.isna(height_px):
        height_px = np.nan

    height_cm_median = np.nan if pd.isna(scale_median) else height_px * scale_median
    height_cm_best = (np.nan if pd.isna(best_quality_scale["scale"])
                      else height_px * best_quality_scale["scale"])
    height_cm_feet = (np.nan if pd.isna(closest_feet_scale["scale"])
                      else height_px * closest_feet_scale["scale"])
    height_cm_ground = (np.nan if pd.isna(ground_line_scale["scale"])
                        else height_px * ground_line_scale["scale"])

    return {
        "height_cm_median_scale": height_cm_median,
        "height_cm_best_quality_scale": height_cm_best,
        "height_cm_closest_feet_scale": height_cm_feet,
        "height_cm_ground_line_scale": height_cm_ground,
    }


def add_height_errors(result: dict) -> dict:
    """Anade errores con signo (estimado - manual) a una fila de resultado."""
    manual_height = result.get("height_manual_cm", np.nan)
    for method_name in ["median_scale", "best_quality_scale", "closest_feet_scale", "ground_line_scale"]:
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

    if references_image_df is not None and not references_image_df.empty:
        for _, ref in references_image_df.iterrows():
            if pd.isna(ref.get("center_x_px", np.nan)) or pd.isna(ref.get("center_y_px", np.nan)):
                continue
            center = (int(ref["center_x_px"]), int(ref["center_y_px"]))
            cv2.circle(image, center, 5, (255, 255, 0), -1)
            put_label(image, f"ID {int(ref['reference_id'])}",
                      (center[0] + 8, center[1] - 8), color=(255, 255, 0))

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
    ratio = result.get("cube_scale_ratio", np.nan)

    lines = [
        "Experiment 1 automatic height estimation",
        f"status: {result.get('status', '')}",
        f"segmentation: {result.get('segmentation_status', '')} | head:{result.get('head_source','')} feet:{result.get('feet_source','')}",
        f"height_px: {height_px:.1f}" if not pd.isna(height_px) else "height_px: NA",
        f"manual HEIGHT_cm: {height_manual:.1f}" if not pd.isna(height_manual) else "manual HEIGHT_cm: NA",
        f"median: {result.get('height_cm_median_scale', np.nan):.1f} cm | error {result.get('error_cm_median_scale', np.nan):.1f} cm",
        f"best quality: {result.get('height_cm_best_quality_scale', np.nan):.1f} cm | error {result.get('error_cm_best_quality_scale', np.nan):.1f} cm",
        f"closest feet: {result.get('height_cm_closest_feet_scale', np.nan):.1f} cm | error {result.get('error_cm_closest_feet_scale', np.nan):.1f} cm",
        f"ground line: {result.get('height_cm_ground_line_scale', np.nan):.1f} cm | error {result.get('error_cm_ground_line_scale', np.nan):.1f} cm",
        (f"cube_scale_ratio: {ratio:.3f} | perspective_flag: {result.get('perspective_flag', False)}"
         if not pd.isna(ratio) else "cube_scale_ratio: NA"),
    ]

    cv2.rectangle(image, (20, 15), (1030, 335), (0, 0, 0), -1)
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
    Ejecuta el Experimento 1 sobre TODAS las imagenes calibradas (max_images=None).
    Segmenta, refina cabeza/pies con pose, estima altura con tres estrategias de
    escala, compara con HEIGHT_cm, anade flag de perspectiva y datos de pose.
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
            ground_line_scale = get_scale_from_ground_line(
                references_image_df=references_image_df,
                feet_y=landmarks["feet_y_px"],
            )

            height_estimates = compute_height_estimates_from_scales(
                height_px=landmarks["height_px"],
                scale_median=float(image_cal["scale_cm_per_pixel"]),
                best_quality_scale=best_quality_scale,
                closest_feet_scale=closest_feet_scale,
                ground_line_scale=ground_line_scale,
            )

            height_manual_cm = get_manual_height_for_subject(
                manual_heights_df=manual_heights_df, subject_id=asset.subject_id,
            )

            perspective = compute_cube_perspective_flag(
                references_image_df=references_image_df,
                image_width=float(image.shape[1]),
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

                "head_source": landmarks.get("head_source", ""),
                "feet_source": landmarks.get("feet_source", ""),
                "head_y_mask": landmarks.get("head_y_mask", np.nan),
                "feet_y_mask": landmarks.get("feet_y_mask", np.nan),

                "pose_eye_y": landmarks.get("pose_eye_y", np.nan),
                "pose_face_scale": landmarks.get("pose_face_scale", np.nan),
                "pose_face_mode": landmarks.get("pose_face_mode", "none"),

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

                "scale_ground_line": ground_line_scale["scale"],
                "ground_line_reference_index": ground_line_scale["reference_index"],
                "ground_line_reference_id": ground_line_scale["reference_id"],
                "ground_line_delta_y_px": ground_line_scale.get("delta_y_px", np.nan),

                "cube_scale_ratio": perspective["cube_scale_ratio"],
                "n_cubes_detected": perspective["n_cubes_detected"],
                "perspective_flag": perspective["perspective_flag"],

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
        ("ground_line_scale", "error_cm_ground_line_scale"),
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


# ============================================================
# CALIBRACION DEL OFFSET DE CORONA (sin re-inferir)
# ============================================================

def sweep_crown_offset_factor(results_df: pd.DataFrame,
                              scale_column: str = "scale_closest_feet",
                              factors=None) -> pd.DataFrame:
    """
    Barre el factor de offset de corona (modo interocular) y devuelve MAE/bias
    para cada valor, recalculando la altura DESDE EL CSV, sin re-ejecutar los
    modelos. Usa las columnas pose_eye_y, pose_face_scale, pose_face_mode,
    head_y_mask, feet_y_px y la escala indicada.

    Solo recalcula las filas con pose_face_mode == 'interocular' (frontales);
    las de 'eye_nose' y 'none' se dejan con su head_y_px actual.

    Devuelve un DataFrame [factor, n, bias_cm, mae_cm, rmse_cm] e imprime el
    factor que minimiza el MAE. Ese valor es el que pondrias en
    CROWN_ABOVE_EYES_FACTOR_INTEROCULAR.
    """
    if factors is None:
        factors = np.round(np.arange(0.6, 2.01, 0.05), 2)

    needed = {"pose_eye_y", "pose_face_scale", "pose_face_mode", "head_y_mask",
              "feet_y_px", "head_y_px", scale_column, "height_manual_cm"}
    if results_df.empty or not needed.issubset(results_df.columns):
        missing = needed - set(results_df.columns)
        raise ValueError(f"Faltan columnas para el sweep: {missing}")

    df = results_df.dropna(subset=["feet_y_px", scale_column, "height_manual_cm"]).copy()

    rows = []
    for factor in factors:
        head_y = df["head_y_px"].astype(float).copy()

        is_inter = df["pose_face_mode"] == "interocular"
        eye_y = df["pose_eye_y"].astype(float)
        face = df["pose_face_scale"].astype(float)
        mask_top = df["head_y_mask"].astype(float)

        crown = eye_y - factor * face
        crown = np.maximum(crown, mask_top)        # no por encima del pelo
        crown = np.minimum(crown, eye_y - 1)       # por encima de los ojos
        head_y = head_y.where(~is_inter, crown)

        height_px = df["feet_y_px"].astype(float) - head_y
        height_cm = height_px * df[scale_column].astype(float)
        err = height_cm - df["height_manual_cm"].astype(float)
        err = err.dropna()

        rows.append({
            "factor": float(factor),
            "n": int(len(err)),
            "bias_cm": float(err.mean()),
            "mae_cm": float(err.abs().mean()),
            "rmse_cm": float(np.sqrt((err ** 2).mean())),
        })

    sweep = pd.DataFrame(rows)
    best = sweep.loc[sweep["mae_cm"].idxmin()]
    print(f"[sweep {scale_column}] mejor factor interocular = {best['factor']:.2f} "
          f"-> MAE {best['mae_cm']:.2f} cm, bias {best['bias_cm']:+.2f} cm "
          f"(n={int(best['n'])})")
    return sweep


# ============================================================
# EXPERIMENT 1: GRAFICO DE RESULTADOS POR VISTA
# ============================================================

def plot_view_results(results_df: pd.DataFrame,
                      view: str = "back",
                      method: str = "ground_line_scale",
                      save_path=None):
    """
    Grafica los resultados de una vista concreta (por defecto 'back') para un
    metodo de escala. Dos paneles:
      (1) altura estimada vs altura manual, con la linea identidad y = x.
      (2) Bland-Altman: diferencia (estimada - real) frente a la media, con el
          sesgo medio y los limites de acuerdo (+-1.96 SD).
    Imprime y anota bias, MAE, RMSE y n. Devuelve un dict de metricas.

    method: 'median_scale' | 'best_quality_scale' | 'closest_feet_scale' |
            'ground_line_scale'.
    """
    est_col = f"height_cm_{method}"
    if "view" not in results_df.columns or est_col not in results_df.columns:
        print(f"Faltan columnas: se necesita 'view' y '{est_col}'")
        return None

    d = results_df[results_df["view"] == view].dropna(
        subset=["height_manual_cm", est_col]
    ).copy()
    if d.empty:
        print(f"No hay filas para view={view} / method={method}")
        return None

    manual = d["height_manual_cm"].to_numpy(float)
    est = d[est_col].to_numpy(float)
    err = est - manual

    bias = float(err.mean())
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    sd = float(err.std(ddof=1)) if len(err) > 1 else 0.0
    loa_hi, loa_lo = bias + 1.96 * sd, bias - 1.96 * sd
    n = len(d)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))

    # Panel 1: estimado vs real
    lo = min(manual.min(), est.min()) - 5
    hi = max(manual.max(), est.max()) + 5
    ax1.plot([lo, hi], [lo, hi], "--", color="gray", lw=1.2, label="y = x (perfecto)")
    ax1.scatter(manual, est, s=45, alpha=0.75, edgecolor="white",
                color="#155e63", zorder=3)
    ax1.set_xlim(lo, hi); ax1.set_ylim(lo, hi)
    ax1.set_xlabel("Altura manual (cm)")
    ax1.set_ylabel("Altura estimada (cm)")
    ax1.set_title(f"Estimado vs real  ·  vista '{view}'  ·  {method}")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.25)
    ax1.text(0.97, 0.05,
             f"n = {n}\nMAE = {mae:.1f} cm\nbias = {bias:+.1f} cm\nRMSE = {rmse:.1f} cm",
             transform=ax1.transAxes, ha="right", va="bottom", fontsize=10,
             bbox=dict(boxstyle="round", fc="#eef4f4", ec="#155e63"))

    # Panel 2: Bland-Altman
    mean_hm = (est + manual) / 2.0
    ax2.scatter(mean_hm, err, s=45, alpha=0.75, edgecolor="white",
                color="#8a6d1f", zorder=3)
    ax2.axhline(bias, color="#155e63", lw=1.6, label=f"sesgo medio {bias:+.1f}")
    ax2.axhline(loa_hi, color="#c0392b", ls="--", lw=1.2,
                label="limites acuerdo (+-1.96 SD)")
    ax2.axhline(loa_lo, color="#c0392b", ls="--", lw=1.2)
    ax2.axhline(0, color="gray", lw=0.8, alpha=0.6)
    ax2.set_xlabel("Media de estimada y manual (cm)")
    ax2.set_ylabel("Diferencia estimada - manual (cm)")
    ax2.set_title(f"Bland-Altman  ·  vista '{view}'")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(alpha=0.25)
    ax2.text(0.03, 0.05, f"LoA: [{loa_lo:.1f}, {loa_hi:.1f}] cm",
             transform=ax2.transAxes, ha="left", va="bottom", fontsize=9,
             bbox=dict(boxstyle="round", fc="#fdf6e3", ec="#8a6d1f"))

    fig.suptitle(f"Experimento 1 - vista '{view}' - {n} sujetos", fontsize=13, y=1.02)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=130, bbox_inches="tight")
        print(f"figura guardada en {save_path}")
    plt.close(fig)

    metrics = {"view": view, "method": method, "n": n,
               "bias_cm": bias, "mae_cm": mae, "rmse_cm": rmse,
               "loa_low": loa_lo, "loa_high": loa_hi}
    print(metrics)
    return metrics


# ============================================================
# EXPERIMENT 1: CORRECCION CALIBRADA (leave-one-out) - antes/despues
# ============================================================

def apply_calibrated_correction(results_df: pd.DataFrame,
                                method: str = "ground_line_scale",
                                by: str = "view",
                                min_n: int = 8) -> pd.DataFrame:
    """
    Corrige el estimador con una recta altura_corregida = a*estimada + b ajustada
    por grupo (por defecto 'view'), usando LEAVE-ONE-OUT: para cada imagen la
    recta se ajusta con todas las demas de su grupo menos ella, y se predice la
    corregida fuera de muestra. Es una curva de calibracion honesta (no
    data-dropping) porque cada valor corregido es out-of-sample.

    Anade las columnas 'height_cm_<method>_corrected' y
    'error_cm_<method>_corrected' SIN tocar las originales, de modo que el CSV
    guarda el antes (columnas base) y el despues (columnas _corrected).

    by='view' corrige por vista; by=None ajusta una sola recta global.
    min_n: si un grupo tiene menos imagenes, no se corrige (se deja igual).
    """
    est_col = f"height_cm_{method}"
    corr_col = f"height_cm_{method}_corrected"
    err_col = f"error_cm_{method}_corrected"

    if est_col not in results_df.columns or "height_manual_cm" not in results_df.columns:
        print(f"Faltan columnas: se necesita '{est_col}' y 'height_manual_cm'")
        return results_df

    df = results_df.copy()
    df[corr_col] = np.nan

    valid = df[est_col].notna() & df["height_manual_cm"].notna()
    if by is not None and by in df.columns:
        groups = df[valid].groupby(by).groups.items()
    else:
        groups = [("all", df.index[valid])]

    for _, idx in groups:
        idx = list(idx)
        x = df.loc[idx, est_col].to_numpy(float)
        y = df.loc[idx, "height_manual_cm"].to_numpy(float)
        n = len(idx)
        if n < min_n:
            df.loc[idx, corr_col] = x            # pocos puntos: sin corregir
            continue
        for j in range(n):
            m = np.ones(n, bool); m[j] = False
            a, b = np.polyfit(x[m], y[m], 1)     # recta sin la j-esima
            df.loc[idx[j], corr_col] = a * x[j] + b

    df[err_col] = np.where(
        df["height_manual_cm"].notna() & df[corr_col].notna(),
        df[corr_col] - df["height_manual_cm"], np.nan)
    return df


def compare_correction(results_df: pd.DataFrame,
                       method: str = "ground_line_scale") -> pd.DataFrame:
    """
    Tabla comparativa antes/despues (MAE, bias, RMSE) global y por vista para el
    metodo indicado. Requiere haber llamado antes a apply_calibrated_correction.
    """
    est_col = f"height_cm_{method}"
    corr_col = f"height_cm_{method}_corrected"
    if corr_col not in results_df.columns:
        print(f"No existe {corr_col}. Llama antes a apply_calibrated_correction().")
        return pd.DataFrame()

    d = results_df.copy()
    d["_raw"] = d[est_col] - d["height_manual_cm"]
    d["_cor"] = d[corr_col] - d["height_manual_cm"]

    def stats(e):
        e = e.dropna()
        if len(e) == 0:
            return (0, np.nan, np.nan, np.nan)
        return (len(e), float(e.mean()), float(e.abs().mean()),
                float(np.sqrt((e ** 2).mean())))

    rows = []
    groups = [("GLOBAL", d)] + [(v, g) for v, g in d.groupby("view")]
    for name, g in groups:
        nr, br, mr, rr = stats(g["_raw"])
        nc, bc, mc, rc = stats(g["_cor"])
        rows.append({"grupo": name, "n": nr,
                     "MAE_antes": mr, "MAE_desp": mc,
                     "bias_antes": br, "bias_desp": bc,
                     "RMSE_antes": rr, "RMSE_desp": rc})

    summary = pd.DataFrame(rows)
    out = VERIFICATION_DIR / f"correction_before_after_{method}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    print(f"tabla antes/despues guardada en {out}")
    return summary
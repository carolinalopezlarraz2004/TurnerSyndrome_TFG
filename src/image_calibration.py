"""
image_calibration.py

This module estimates an image-specific pixel-to-centimeter scale using
visible physical references.

Supported calibration methods:
- aruco_marker: used for Colombia and future Brazil images.
- circular_marker: used for Barcelona marker images.

The module saves:
1. image_calibration.csv: one row per image.
2. calibration_references.csv: one row per detected physical reference.
3. annotated verification images in outputs/verification.

The goal is not only to compute a scale, but also to make the detection
visually verifiable.
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


# ============================================================
# BASIC GEOMETRY
# ============================================================

def euclidean_distance(point_a, point_b) -> float:
    """
    Purpose:
        Compute the Euclidean distance between two 2D points.

    Input:
        point_a:
            First point, usually with coordinates (x, y).

        point_b:
            Second point, usually with coordinates (x, y).

    Output:
        float:
            Distance between both points in pixels.
    """

    point_a = np.array(point_a, dtype=float)
    point_b = np.array(point_b, dtype=float)

    return float(np.linalg.norm(point_a - point_b))


def coefficient_of_variation(values: list[float]) -> float:
    """
    Purpose:
        Compute the coefficient of variation of a list of values.

    Input:
        values (list[float]):
            List of numeric values, in this case local cm/pixel scales.

    Output:
        float:
            Relative dispersion of the values.

    Notes:
        This value is used as a simple warning. If two references in the same
        image give very different scales, the image may have perspective effects
        or one of the detections may be unstable.
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
    """
    Purpose:
        Create the image-level calibration result from the individual detected
        references.

    Input:
        reference_rows (list[dict]):
            List with one dictionary per detected physical reference.

    Output:
        dict:
            Image-level calibration information.
    """

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

    # This threshold is only used as a warning. The image is not discarded
    # automatically because visual verification is part of the project.
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

def put_label(
    image,
    text: str,
    origin: tuple[int, int],
    color=(0, 255, 255),
) -> None:
    """
    Purpose:
        Draw readable text on a verification image.

    Input:
        image:
            Image where the text will be drawn.

        text (str):
            Text to write.

        origin (tuple[int, int]):
            Text position in pixels.

        color:
            Text color in BGR format.

    Output:
        None.
        The image is modified in place.
    """

    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def build_verification_filename(asset, image_path: Path, step: str) -> str:
    """
    Purpose:
        Build a clear filename for a verification image.

    Input:
        asset:
            ImageAsset object.

        image_path (Path):
            Path to the image.

        step (str):
            Processing step. Example: calibration.

    Output:
        str:
            Filename for the verification image.
    """

    return f"{asset.subject_id}_{asset.view}_{image_path.stem}_{step}.png"


# ============================================================
# ARUCO DETECTION
# ============================================================

def detect_aruco_markers(gray_image):
    """
    Purpose:
        Detect ArUco markers using a version-compatible OpenCV approach.

    Input:
        gray_image:
            Grayscale image.

    Output:
        tuple:
            corners, ids
    """

    aruco_dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)

    try:
        # OpenCV newer versions.
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(gray_image)

    except AttributeError:
        # OpenCV older versions.
        parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray_image,
            aruco_dictionary,
            parameters=parameters,
        )

    return corners, ids


def calibrate_with_aruco(
    image,
    asset,
    debug_image=None,
) -> tuple[dict, list[dict], np.ndarray | None]:
    """
    Purpose:
        Estimate the cm/pixel scale from ArUco markers.

    Method:
        1. Detect ArUco markers.
        2. Measure the four sides of each marker in pixels.
        3. Compute one local scale per marker:
               scale = real_marker_side_cm / marker_side_px
        4. Use the median local scale as the image-level scale.
        5. Save one row per detected marker for later verification.

    Input:
        image:
            Original BGR image.

        asset:
            ImageAsset object with subject, site, view and path.

        debug_image:
            Copy of the image used to draw verification annotations.

    Output:
        tuple:
            - image-level calibration result
            - reference-level rows
            - annotated verification image
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

        reference_rows.append(
            {
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
            }
        )

        if debug_image is not None:
            points_int = points.astype(int)

            cv2.polylines(
                debug_image,
                [points_int],
                isClosed=True,
                color=(0, 255, 0),
                thickness=3,
            )

            cv2.circle(
                debug_image,
                (int(center_x), int(center_y)),
                4,
                (0, 0, 255),
                -1,
            )

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

def calibrate_with_circular_markers(
    image,
    asset,
    debug_image=None,
) -> tuple[dict, list[dict], np.ndarray | None]:
    """
    Purpose:
        Estimate the cm/pixel scale from Barcelona circular markers.

    Method:
        1. Detect dark circular blobs.
        2. Measure their apparent diameter in pixels.
        3. Compute one local scale per marker:
               scale = real_marker_diameter_cm / marker_diameter_px
        4. Use the median local scale as the image-level scale.
        5. Save one row per detected marker for later verification.

    Input:
        image:
            Original BGR image.

        asset:
            ImageAsset object with subject, site, view and path.

        debug_image:
            Copy of the image used to draw verification annotations.

    Output:
        tuple:
            - image-level calibration result
            - reference-level rows
            - annotated verification image
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    params = cv2.SimpleBlobDetector_Params()

    # Barcelona markers are expected to be dark circular references.
    params.filterByColor = True
    params.blobColor = 0

    # Broad initial range. These values may need adjustment after visually
    # checking the verification images.
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

        reference_rows.append(
            {
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
            }
        )

        if debug_image is not None:
            x = int(center_x)
            y = int(center_y)
            radius = int(diameter_px / 2)

            cv2.circle(
                debug_image,
                (x, y),
                radius,
                (0, 255, 0),
                2,
            )

            cv2.circle(
                debug_image,
                (x, y),
                3,
                (0, 0, 255),
                -1,
            )

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
    """
    Purpose:
        Assign a simple image type label.

    Input:
        asset:
            ImageAsset object.

    Output:
        str:
            Image type label.
    """

    if asset.site in ["CO", "BR"]:
        return "aruco"

    if asset.site == "ES":
        if asset.has_markers:
            return "markers"
        return "clean"

    return "unknown"


def should_calibrate_asset(asset) -> bool:
    """
    Purpose:
        Decide whether an image should be calibrated directly.

    Rules:
        CO / BR:
            All images are expected to contain ArUco cube markers.

        ES:
            Only marker images are calibrated directly. Clean images are not
            calibrated directly at this stage because they do not contain a
            visible physical reference.

    Input:
        asset:
            ImageAsset object.

    Output:
        bool:
            True if the image should be calibrated directly.
    """

    if asset.site in ["CO", "BR"]:
        return True

    if asset.site == "ES":
        return asset.has_markers

    return False


def calibrate_image_asset(
    asset,
    save_debug: bool = True,
) -> tuple[dict, list[dict]]:
    """
    Purpose:
        Calibrate one ImageAsset.

    Input:
        asset:
            ImageAsset object.

        save_debug (bool):
            If True, save an annotated verification image.

    Output:
        tuple:
            - image-level calibration row
            - reference-level rows
    """

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
            image=image,
            asset=asset,
            debug_image=debug_image,
        )

    elif result["calibration_method"] == "circular_marker":
        calibration_result, reference_rows, debug_image = calibrate_with_circular_markers(
            image=image,
            asset=asset,
            debug_image=debug_image,
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
            asset=asset,
            image_path=image_path,
            step="calibration",
        )

        if np.isnan(result["scale_cm_per_pixel"]):
            scale_text = "scale=NA"
        else:
            scale_text = f"scale={result['scale_cm_per_pixel']:.5f} cm/px"

        if np.isnan(result["scale_cv"]):
            cv_text = "CV=NA"
        else:
            cv_text = f"CV={result['scale_cv']:.3f}"

        summary_text = (
            f"{result['calibration_method']} | "
            f"{scale_text} | "
            f"n={result['n_references_detected']} | "
            f"{cv_text} | "
            f"{result['calibration_status']}"
        )

        # Main summary label.
        cv2.rectangle(
            debug_image,
            (20, 15),
            (1100, 60),
            (0, 0, 0),
            -1,
        )

        put_label(
            debug_image,
            summary_text,
            (30, 45),
            color=(0, 255, 255),
        )

        cv2.imwrite(str(verification_path), debug_image)
        result["verification_image_path"] = str(verification_path)

    return result, reference_rows


# ============================================================
# BATCH PROCESSING
# ============================================================

def calibrate_image_assets(
    assets_by_subject: dict,
    save_debug: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Purpose:
        Calibrate all image assets in a subject dictionary.

    Input:
        assets_by_subject (dict):
            Dictionary where each key is a subject ID and each value is a list
            of ImageAsset objects.

        save_debug (bool):
            If True, save annotated verification images.

    Output:
        tuple:
            - calibration_df: one row per image
            - references_df: one row per detected reference
    """

    calibration_rows = []
    reference_rows_all = []

    for _, assets in assets_by_subject.items():
        for asset in assets:
            calibration_row, reference_rows = calibrate_image_asset(
                asset=asset,
                save_debug=save_debug,
            )

            calibration_rows.append(calibration_row)
            reference_rows_all.extend(reference_rows)

    calibration_df = pd.DataFrame(calibration_rows)
    references_df = pd.DataFrame(reference_rows_all)

    return calibration_df, references_df


def save_image_calibration_table(
    assets_by_subject: dict,
    save_debug: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Purpose:
        Run image calibration and save the output tables.

    Input:
        assets_by_subject (dict):
            Dictionary with all image assets to process.

        save_debug (bool):
            If True, save annotated verification images.

    Output:
        tuple:
            - calibration_df
            - references_df
    """

    calibration_df, references_df = calibrate_image_assets(
        assets_by_subject=assets_by_subject,
        save_debug=save_debug,
    )

    IMAGE_CALIBRATION_TABLE.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_REFERENCES_TABLE.parent.mkdir(parents=True, exist_ok=True)

    calibration_df.to_csv(IMAGE_CALIBRATION_TABLE, index=False)
    references_df.to_csv(CALIBRATION_REFERENCES_TABLE, index=False)

    return calibration_df, references_df
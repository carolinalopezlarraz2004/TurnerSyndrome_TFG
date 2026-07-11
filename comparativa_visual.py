"""
comparativa_visual.py

Para cada imagen, genera UNA imagen anotada por METODO, organizadas asi:

    outputs/comparativa_final/<sujeto>/<perspectiva>/<metodo>.png

Metodos (5):
    1_experiment_base            -> Exp base, escala linea de suelo
    2_experiment_base_corrected  -> Exp base, escala linea de suelo CORREGIDA (LOO)
    3_horizon_cube               -> horizonte via cubo
    4_horizon_shared_single      -> horizonte compartido, pie de la pose
    5_horizon_shared_mask        -> horizonte compartido, pie de la mascara

Cada imagen muestra: la persona (cabeza-pies), la referencia del metodo (cubo u
horizonte) y un recuadro con altura real, altura estimada, error y la escala
pixel->cm (equivalente en el caso del horizonte = altura_estimada / altura_px).

NO reprocesa con YOLO: lee los puntos y estimaciones de los CSV que genera el
main (Exp base + horizonte) y solo re-detecta los cubos (ArUco) para dibujar sus
aristas. Por eso es rapido.

Uso (aparte, DESPUES de correr el main):
    python comparativa_visual.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

from src.config import VERIFICATION_DIR, HORIZON_DIR, COMPARATIVA_DIR, CAMERA_HEIGHT_CM
from src.image_calibration import detect_aruco_markers, put_label
from src.horizon_method import select_reference_cube

E1_CSV = VERIFICATION_DIR / "height_estimates_experiment1.csv"
HZ_CSV = HORIZON_DIR / "horizon_estimates.csv"


# ---------- dibujo de un metodo ----------
def draw_method(image, head_xy, feet_xy, title, real_cm, est_cm,
                horizon_y=None, cube_edge=None, out_path=None):
    """Dibuja una imagen anotada para un metodo y la guarda."""
    img = image.copy()
    hI, wI = img.shape[:2]

    # horizonte (si aplica)
    if horizon_y is not None and np.isfinite(horizon_y):
        yy = int(np.clip(horizon_y, 0, hI - 1))
        cv2.line(img, (0, yy), (wI, yy), (0, 220, 220), 2, cv2.LINE_AA)
        put_label(img, f"horizonte (y={horizon_y:.0f})", (30, max(yy - 8, 20)),
                  color=(0, 220, 220))

    # arista del cubo (si aplica)
    if cube_edge is not None:
        b = tuple(np.int32(cube_edge[0])); t = tuple(np.int32(cube_edge[1]))
        cv2.line(img, b, t, (0, 220, 0), 4, cv2.LINE_AA)
        put_label(img, "cubo 10cm", (b[0] + 6, b[1] + 18), color=(0, 220, 0))

    # persona (cabeza-pies)
    if head_xy is not None and feet_xy is not None:
        h = (int(head_xy[0]), int(head_xy[1]))
        f = (int(feet_xy[0]), int(feet_xy[1]))
        cv2.line(img, h, f, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(img, h, 7, (0, 255, 255), -1); put_label(img, "cabeza", (h[0] + 8, h[1]))
        cv2.circle(img, f, 7, (0, 0, 255), -1); put_label(img, "pies", (f[0] + 8, f[1]),
                                                           color=(0, 0, 255))

    # escala equivalente cm/px y error
    height_px = (feet_xy[1] - head_xy[1]) if (head_xy is not None and feet_xy is not None) else np.nan
    scale = (est_cm / height_px) if (np.isfinite(est_cm) and np.isfinite(height_px) and height_px > 0) else np.nan
    err = (est_cm - real_cm) if (np.isfinite(est_cm) and np.isfinite(real_cm)) else np.nan

    lines = [
        title,
        f"altura real:      {real_cm:.1f} cm" if np.isfinite(real_cm) else "altura real: NA",
        f"altura estimada:  {est_cm:.1f} cm" if np.isfinite(est_cm) else "altura estimada: NA",
        f"error:            {err:+.1f} cm" if np.isfinite(err) else "error: NA",
        f"escala px->cm:    {scale:.5f} cm/px" if np.isfinite(scale) else "escala: NA",
        f"altura en px:     {height_px:.0f}" if np.isfinite(height_px) else "altura px: NA",
    ]
    cv2.rectangle(img, (20, 15), (560, 30 + 30 * len(lines)), (0, 0, 0), -1)
    for i, t in enumerate(lines):
        put_label(img, t, (30, 45 + i * 30), color=(0, 255, 255))

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img)


def _xy(row, xcol, ycol):
    x, y = row.get(xcol, np.nan), row.get(ycol, np.nan)
    return None if (pd.isna(x) or pd.isna(y)) else (float(x), float(y))


def main():
    e1 = pd.read_csv(E1_CSV)
    hz = pd.read_csv(HZ_CSV)

    # columnas de e1 que necesitamos para dibujar y para los numeros
    e1_cols = ["image_path", "site", "subject_id", "view", "height_manual_cm",
               "head_x_px", "head_y_px", "feet_x_px", "feet_y_px",
               "height_cm_ground_line_scale", "height_cm_ground_line_scale_corrected"]
    e1 = e1[[c for c in e1_cols if c in e1.columns]].copy()

    hz_cols = ["image_path", "y_head", "y_feet", "y_feet_single", "y_feet_mask",
               "y_horizon_ref", "y_horizon_cube",
               "H_horizon_cube", "H_horizon_shared",
               "H_shared_feet_single", "H_shared_feet_mask"]
    hz = hz[[c for c in hz_cols if c in hz.columns]].copy()

    df = e1.merge(hz, on="image_path", how="outer")

    n_ok = 0
    for _, row in df.iterrows():
        image_path = row.get("image_path", None)
        if not isinstance(image_path, str):
            continue
        image = cv2.imread(image_path)
        if image is None:
            continue

        subject = str(row.get("subject_id", "NA"))
        view = str(row.get("view", "NA"))
        real = float(row.get("height_manual_cm", np.nan))
        base_dir = COMPARATIVA_DIR / subject / view

        # puntos de la persona: x del Exp base (la persona es casi vertical)
        head_x = row.get("head_x_px", np.nan)
        feet_x = row.get("feet_x_px", np.nan)
        head_e1 = _xy(row, "head_x_px", "head_y_px")
        feet_e1 = _xy(row, "feet_x_px", "feet_y_px")

        # cubo de referencia (re-deteccion ArUco, rapido)
        cube_edge = None
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            corners, ids = detect_aruco_markers(gray)
            markers = [c.reshape(4, 2) for c in corners] if ids is not None else []
            feet_y_ref = row.get("y_feet", row.get("feet_y_px", np.nan))
            if markers and np.isfinite(feet_y_ref):
                cube_edge = select_reference_cube(markers, float(feet_y_ref))
        except Exception:
            cube_edge = None

        # ---- 1) Exp base, escala linea de suelo ----
        draw_method(
            image, head_e1, feet_e1,
            "1 - Exp base: escala linea de suelo",
            real, float(row.get("height_cm_ground_line_scale", np.nan)),
            out_path=base_dir / "1_experiment_base.png",
        )

        # ---- 2) Exp base, escala corregida (LOO) ----
        draw_method(
            image, head_e1, feet_e1,
            "2 - Exp base: escala suelo corregida (LOO)",
            real, float(row.get("height_cm_ground_line_scale_corrected", np.nan)),
            out_path=base_dir / "2_experiment_base_corrected.png",
        )

        # puntos para el horizonte: x del Exp base, y del CSV del horizonte
        yh = row.get("y_head", np.nan)
        head_hz = (float(head_x), float(yh)) if (pd.notna(head_x) and pd.notna(yh)) else head_e1

        def feet_hz(ycol):
            yv = row.get(ycol, np.nan)
            if pd.notna(feet_x) and pd.notna(yv):
                return (float(feet_x), float(yv))
            return feet_e1

        # ---- 3) horizonte via cubo ----
        draw_method(
            image, head_hz, feet_hz("y_feet"),
            "3 - Horizonte via cubo",
            real, float(row.get("H_horizon_cube", np.nan)),
            horizon_y=row.get("y_horizon_cube", np.nan), cube_edge=cube_edge,
            out_path=base_dir / "3_horizon_cube.png",
        )

        # ---- 4) horizonte compartido, pie single ----
        draw_method(
            image, head_hz, feet_hz("y_feet_single"),
            "4 - Horizonte compartido (pie pose)",
            real, float(row.get("H_shared_feet_single", np.nan)),
            horizon_y=row.get("y_horizon_ref", np.nan), cube_edge=cube_edge,
            out_path=base_dir / "4_horizon_shared_single.png",
        )

        # ---- 5) horizonte compartido, pie mask ----
        draw_method(
            image, head_hz, feet_hz("y_feet_mask"),
            "5 - Horizonte compartido (pie mascara)",
            real, float(row.get("H_shared_feet_mask", np.nan)),
            horizon_y=row.get("y_horizon_ref", np.nan), cube_edge=cube_edge,
            out_path=base_dir / "5_horizon_shared_mask.png",
        )

        n_ok += 1

    print(f"Comparativa visual generada para {n_ok} imagenes.")
    print(f"Estructura: {COMPARATIVA_DIR}/<sujeto>/<perspectiva>/<metodo>.png")


if __name__ == "__main__":
    main()
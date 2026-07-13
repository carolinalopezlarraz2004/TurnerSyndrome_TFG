"""
horizon_method.py  -  Experimento 2: altura por el metodo del HORIZONTE

H = h_cam * (y_pies - y_cabeza) / (y_pies - y_horizonte)   (y crece hacia abajo)

Camara nivelada a altura conocida -> invariante a la profundidad, sin calibrar.

Correcciones de puntos (desde la mascara de estimate_body_height_pixels; no
tocan image_calibration):
  * CABEZA (todas las vistas): corona = punto mas alto de la silueta con recorte
    de pelo (salta moños). Se acepta si es coherente con la pose (tope de
    seguridad). Sube la corona donde la estimacion por ojos se quedaba corta y
    la baja donde el pelo sobresalia.
  * PIES EN LATERALES: pie = punto mas bajo de la mascara (suela real), en vez
    del pie unico de la pose que se queda alto.

Ademas, para COMPARAR, calcula la altura con dos variantes de pie:
  - 'single': la y del pie tal como la da la pose (single/midfeet segun vista).
  - 'mask'  : el fondo de la mascara (suela real).
Asi el CSV y el resumen muestran el error de cada variante por vista.

HORIZONTE: (A) via cubo, foto a foto. (B) COMPARTIDO: horizonte estable de las
vistas fiables (front/back) por sujeto (con recurso al global), aplicado a todas.

Uso:
    from src.horizon_method import run_horizon_experiment
    run_horizon_experiment(assets_by_subject, manual_heights_df=colombia_equalized,
                           max_images=None, save_debug=True)
"""

from pathlib import Path
import numpy as np
import cv2

from src.config import HORIZON_DIR, ARUCO_MARKER_SIZE_CM
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

MIN_EDGE_LEN_PX = 10.0
RELIABLE_VIEWS = ("front", "back")

# --- interruptores de las correcciones ---
USE_MASK_FEET_LATERAL = True     # pies = fondo de la mascara en left/right
USE_MASK_HEAD = True             # cabeza = corona de la mascara con recorte de pelo (todas las vistas)
MAX_HEAD_MOVE_FRAC = 0.12        # aceptar la corona de mascara si difiere < este % de la altura
HEAD_WIDTH_FRAC = 0.50           # una fila es "cabeza" si su anchura >= 50% de la anchura del craneo


# ---------- correcciones de puntos usando la mascara ----------
def _mask_yx(body_mask):
    m = np.asarray(body_mask)
    if m.ndim == 3:
        m = m[..., 0]
    ys, xs = np.where(m > 0)
    return ys, xs, m.shape[0]

def feet_from_mask(body_mask):
    """Punto mas bajo de la mascara = suela que toca el suelo. (x, y) o None."""
    ys, xs, _ = _mask_yx(body_mask)
    if len(ys) == 0:
        return None
    ymax = int(ys.max())
    sel = ys >= ymax - 4
    return np.array([float(np.median(xs[sel])), float(ymax)])

def head_from_mask_trim_hair(body_mask):
    """
    Corona del craneo saltando el pelo/moño: de arriba a abajo, primera fila cuya
    anchura ya es >= HEAD_WIDTH_FRAC de la anchura tipica del craneo. (x, y) o None.
    """
    ys, xs, H = _mask_yx(body_mask)
    if len(ys) == 0:
        return None
    ytop, ybot = int(ys.min()), int(ys.max())
    minx = np.full(H, 1e9); maxx = np.full(H, -1.0)
    np.minimum.at(minx, ys, xs); np.maximum.at(maxx, ys, xs)
    width = np.where(maxx >= 0, maxx - minx + 1.0, 0.0)
    head_band = int(ytop + 0.12 * (ybot - ytop))
    band = width[ytop:head_band + 1]
    band = band[band > 0]
    if band.size == 0:
        return None
    skull_w = float(np.median(np.sort(band)[-max(1, band.size // 3):]))
    thr = HEAD_WIDTH_FRAC * skull_w
    for y in range(ytop, ybot + 1):
        if width[y] >= thr:
            return np.array([float((minx[y] + maxx[y]) / 2.0), float(y)])
    return np.array([float((minx[ytop] + maxx[ytop]) / 2.0), float(ytop)])


def correct_points(view, feet, head, body_mask):
    """Pies (laterales) desde la mascara; cabeza (corona) desde la mascara con
    recorte de pelo en TODAS las vistas, si es coherente con la pose."""
    info = {"feet_source": "pose", "head_source": "pose", "head_move_px": 0.0}
    if body_mask is None:
        return feet, head, info

    if USE_MASK_FEET_LATERAL and view in ("left", "right"):
        fm = feet_from_mask(body_mask)
        if fm is not None and fm[1] >= feet[1] - 5:
            feet = fm
            info["feet_source"] = "mask_bottom"

    if USE_MASK_HEAD:
        ht = head_from_mask_trim_hair(body_mask)
        if ht is not None:
            cap = MAX_HEAD_MOVE_FRAC * (feet[1] - head[1])
            if abs(ht[1] - head[1]) <= cap:
                info["head_move_px"] = float(head[1] - ht[1])
                head = ht
                info["head_source"] = "mask_trim_hair"
    return feet, head, info


# ---------- geometria del marcador ----------
def _angle(p1, p2):
    return abs(np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))) % 180.0

def marker_vertical_edge(corners4):
    c = np.asarray(corners4, float).reshape(4, 2)
    edges = [(c[0], c[1]), (c[1], c[2]), (c[2], c[3]), (c[3], c[0])]
    best, best_score = None, -1.0
    for p1, p2 in edges:
        if np.hypot(*(p2 - p1)) < MIN_EDGE_LEN_PX:
            continue
        verticality = 1.0 - abs(_angle(p1, p2) - 90.0) / 90.0
        if verticality > best_score:
            base, top = (p1, p2) if p1[1] >= p2[1] else (p2, p1)
            best, best_score = (base, top), verticality
    return best

def select_reference_cube(markers, feet_y):
    best, best_d = None, np.inf
    for corners in markers:
        c = np.asarray(corners, float).reshape(4, 2)
        edge = marker_vertical_edge(c)
        if edge is None:
            continue
        d = abs(c[:, 1].mean() - feet_y)
        if d < best_d:
            best, best_d = edge, d
    return best


# ---------- metodo del horizonte ----------
def horizon_from_cube(y_cube_base, y_cube_top, h_cam, ref_cm=ARUCO_MARKER_SIZE_CM):
    return y_cube_base - h_cam * (y_cube_base - y_cube_top) / ref_cm

def height_from_horizon(h_cam, y_feet, y_head, y_horizon):
    den = (y_feet - y_horizon)
    return np.nan if abs(den) < 1e-6 else float(h_cam * (y_feet - y_head) / den)


# ---------- una imagen (pasada 1) ----------
def estimate_height_horizon(image, view=""):
    out = {"status": "ok",
           "H_horizon_cube": np.nan, "y_horizon_cube": np.nan,
           "y_feet": np.nan, "y_head": np.nan, "height_px": np.nan,
           "y_feet_single": np.nan, "y_feet_mask": np.nan,
           "feet": None, "head": None, "cube_base": None, "cube_top": None,
           "feet_source": "pose", "head_source": "pose", "head_move_px": 0.0,
           "img_h": image.shape[0]}

    landmarks, body_mask = estimate_body_height_pixels(image, view=view)
    if not str(landmarks.get("landmark_status", "")).startswith("ok"):
        out["status"] = f"person_failed_{landmarks.get('landmark_status','')}"
        return out
    feet = np.array([landmarks["feet_x_px"], landmarks["feet_y_px"]], float)
    head = np.array([landmarks["head_x_px"], landmarks["head_y_px"]], float)

    # --- variantes de pie (para comparar) ---
    out["y_feet_single"] = float(feet[1])                       # pie tal cual da la pose
    fm = feet_from_mask(body_mask) if body_mask is not None else None
    if fm is not None:
        out["y_feet_mask"] = float(fm[1])                       # suela real (mascara)

    # --- correccion "elegida" (la que usa el metodo por defecto) ---
    feet, head, info = correct_points(view, feet, head, body_mask)
    out.update({"feet_source": info["feet_source"], "head_source": info["head_source"],
                "head_move_px": info["head_move_px"]})
    out["feet"], out["head"] = feet, head
    out["y_feet"], out["y_head"] = float(feet[1]), float(head[1])
    out["height_px"] = float(feet[1] - head[1])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids = detect_aruco_markers(gray)
    markers = [c.reshape(4, 2) for c in corners] if ids is not None else []
    if not markers:
        out["status"] = "no_aruco"
        return out
    edge = select_reference_cube(markers, feet[1])
    if edge is None:
        out["status"] = "no_cube_edge"
        return out
    cube_base, cube_top = edge
    out["cube_base"], out["cube_top"] = cube_base, cube_top
    yh = horizon_from_cube(cube_base[1], cube_top[1], CAMERA_HEIGHT_CM)
    out["y_horizon_cube"] = float(yh)
    out["H_horizon_cube"] = height_from_horizon(CAMERA_HEIGHT_CM, feet[1], head[1], yh)
    if not np.isfinite(out["H_horizon_cube"]) or not (40 < out["H_horizon_cube"] < 230):
        out["status"] = "horizon_out_of_range"
    return out


# ---------- dibujo ----------
def draw_horizon(image, rec, out_path=None):
    img = image.copy()
    hI, wI = img.shape[:2]
    ys = rec.get("y_horizon_ref", np.nan)
    if np.isfinite(ys):
        yy = int(np.clip(ys, 0, hI - 1))
        cv2.line(img, (0, yy), (wI, yy), (0, 220, 220), 2, cv2.LINE_AA)
        put_label(img, f"horizonte compartido (y={ys:.0f})", (30, max(yy - 8, 20)), color=(0, 220, 220))
    if rec.get("cube_base") is not None:
        b = tuple(np.int32(rec["cube_base"])); t = tuple(np.int32(rec["cube_top"]))
        cv2.line(img, b, t, (0, 220, 0), 4, cv2.LINE_AA)
        put_label(img, "cubo 10cm", (b[0] + 6, b[1] + 18), color=(0, 220, 0))
    if rec.get("feet") is not None:
        b = tuple(np.int32(rec["feet"])); t = tuple(np.int32(rec["head"]))
        cv2.line(img, b, t, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(img, t, 7, (0, 255, 255), -1)
        put_label(img, f"cabeza [{rec.get('head_source','')}]", (t[0] + 8, t[1]))
        cv2.circle(img, b, 7, (0, 0, 255), -1)
        put_label(img, f"pies [{rec.get('feet_source','')}]", (b[0] + 8, b[1]), color=(0, 0, 255))

    man = rec.get("height_manual_cm", np.nan)
    def em(v): return (v - man) if (np.isfinite(v) and np.isfinite(man)) else np.nan
    hsh, hcu = rec.get("H_horizon_shared", np.nan), rec.get("H_horizon_cube", np.nan)
    lines = [
        f"Experimento 2 - horizonte  (h_cam={CAMERA_HEIGHT_CM:.0f} cm)",
        f"status: {rec['status']}",
        f"manual: {man:.1f} cm" if np.isfinite(man) else "manual: NA",
        f"HORIZONTE compartido: {hsh:.1f} cm | error {em(hsh):+.1f}" if np.isfinite(hsh) else "compartido: NA",
        f"horizonte via cubo:  {hcu:.1f} cm | error {em(hcu):+.1f}" if np.isfinite(hcu) else "via cubo: NA",
    ]
    cv2.rectangle(img, (20, hI - 20 - 30 * len(lines)), (660, hI - 15), (0, 0, 0), -1)
    for i, t in enumerate(lines):
        put_label(img, t, (30, hI - 25 - 30 * (len(lines) - 1 - i)), color=(0, 255, 255))
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img)
    return img


# ---------- runner (dos pasadas) ----------
def run_horizon_experiment(assets_by_subject, manual_heights_df,
                           max_images=None, save_debug=True):
    import pandas as pd

    # -------- PASADA 1 --------
    records, n = [], 0
    for _, assets in assets_by_subject.items():
        for asset in assets:
            if max_images is not None and n >= max_images:
                break
            image = cv2.imread(str(asset.path))
            if image is None:
                continue
            try:
                res = estimate_height_horizon(image, view=asset.view)
            except Exception as e:
                res = {"status": f"error_{type(e).__name__}", "H_horizon_cube": np.nan,
                       "y_horizon_cube": np.nan, "y_feet": np.nan, "y_head": np.nan,
                       "height_px": np.nan, "y_feet_single": np.nan, "y_feet_mask": np.nan,
                       "feet": None, "head": None, "cube_base": None, "cube_top": None,
                       "feet_source": "-", "head_source": "-", "head_move_px": 0.0,
                       "img_h": image.shape[0]}
            res["site"] = asset.site
            res["subject_id"] = asset.subject_id
            res["view"] = asset.view
            res["image_path"] = str(asset.path)
            res["height_manual_cm"] = get_manual_height_for_subject(manual_heights_df, asset.subject_id)
            records.append(res)
            n += 1
        if max_images is not None and n >= max_images:
            break

    # -------- horizonte de referencia (por sujeto, recurso global) --------
    def valid_h(r):
        return (r["view"] in RELIABLE_VIEWS and r["status"] == "ok"
                and np.isfinite(r.get("y_horizon_cube", np.nan)))

    good = [r["y_horizon_cube"] for r in records if valid_h(r)]
    global_ref = float(np.median(good)) if good else np.nan
    per_subject = {}
    for sid in {r["subject_id"] for r in records}:
        vals = [r["y_horizon_cube"] for r in records if r["subject_id"] == sid and valid_h(r)]
        per_subject[sid] = float(np.median(vals)) if vals else global_ref

    # -------- PASADA 2: horizonte compartido + variantes de pie + dibujo --------
    # -------- PASADA 2: horizonte compartido + variantes de pie + dibujo --------
    for r in records:
        yref = per_subject.get(r["subject_id"], global_ref)

        r["y_horizon_ref"] = yref

        has_subject_reference = any(
            rr["subject_id"] == r["subject_id"] and valid_h(rr)
            for rr in records
        )

        r["horizon_ref_source"] = (
            "subject"
            if has_subject_reference
            else "global"
        )

        # --------------------------------------------------------
        # Diagnóstico de alineación del horizonte
        # --------------------------------------------------------
        # Compara el horizonte obtenido mediante el cubo de esta
        # imagen con el horizonte compartido front/back.
        y_cube = r.get("y_horizon_cube", np.nan)
        img_h = r.get("img_h", np.nan)

        if np.isfinite(y_cube) and np.isfinite(yref):
            r["delta_horizon_px"] = float(y_cube - yref)
        else:
            r["delta_horizon_px"] = np.nan

        # Diferencia normalizada por la altura de la imagen.
        if (
                np.isfinite(y_cube)
                and np.isfinite(yref)
                and np.isfinite(img_h)
                and img_h > 0
        ):
            r["delta_horizon_frac"] = float(
                (y_cube - yref) / img_h
            )
        else:
            r["delta_horizon_frac"] = np.nan

        yf = r["y_feet"]
        yhd = r["y_head"]
        man = r["height_manual_cm"]

        # Estimación por defecto usando el pie seleccionado.
        if (
                np.isfinite(yf)
                and np.isfinite(yhd)
                and np.isfinite(yref)
        ):
            r["H_horizon_shared"] = height_from_horizon(
                CAMERA_HEIGHT_CM,
                yf,
                yhd,
                yref,
            )
        else:
            r["H_horizon_shared"] = np.nan

        r["error_horizon_shared"] = (
            r["H_horizon_shared"] - man
            if (
                    np.isfinite(r["H_horizon_shared"])
                    and np.isfinite(man)
            )
            else np.nan
        )

        r["error_horizon_cube"] = (
            r["H_horizon_cube"] - man
            if (
                    np.isfinite(r["H_horizon_cube"])
                    and np.isfinite(man)
            )
            else np.nan
        )

        # Variantes de pie con el mismo horizonte compartido
        # y la misma cabeza.
        for tag in ["single", "mask"]:
            yfv = r.get(f"y_feet_{tag}", np.nan)

            valid_variant = (
                    np.isfinite(yfv)
                    and np.isfinite(yhd)
                    and np.isfinite(yref)
            )

            if valid_variant:
                Hv = height_from_horizon(
                    CAMERA_HEIGHT_CM,
                    yfv,
                    yhd,
                    yref,
                )
            else:
                Hv = np.nan

            r[f"H_shared_feet_{tag}"] = Hv

            r[f"error_feet_{tag}"] = (
                Hv - man
                if np.isfinite(Hv) and np.isfinite(man)
                else np.nan
            )

        if save_debug:
            image = cv2.imread(str(r["image_path"]))

            if image is not None:
                out = (
                        HORIZON_DIR
                        / (
                            f"{r['subject_id']}_{r['view']}_"
                            f"{Path(r['image_path']).stem}_horizon.png"
                        )
                )

                draw_horizon(
                    image,
                    r,
                    out_path=out,
                )

                r["verification_image_path"] = str(out)

        yf, yhd = r["y_feet"], r["y_head"]
        man = r["height_manual_cm"]

        # estimacion por defecto (pie "elegido")
        r["H_horizon_shared"] = (height_from_horizon(CAMERA_HEIGHT_CM, yf, yhd, yref)
                                 if np.isfinite(yf) and np.isfinite(yhd) and np.isfinite(yref) else np.nan)
        r["error_horizon_shared"] = (r["H_horizon_shared"] - man) if np.isfinite(man) else np.nan
        r["error_horizon_cube"] = (r["H_horizon_cube"] - man) if np.isfinite(man) else np.nan

        # variantes de pie con el MISMO horizonte compartido y la misma cabeza
        for tag in ["single", "mask"]:
            yfv = r.get(f"y_feet_{tag}", np.nan)
            ok = np.isfinite(yfv) and np.isfinite(yhd) and np.isfinite(yref)
            Hv = height_from_horizon(CAMERA_HEIGHT_CM, yfv, yhd, yref) if ok else np.nan
            r[f"H_shared_feet_{tag}"] = Hv
            r[f"error_feet_{tag}"] = (Hv - man) if (np.isfinite(Hv) and np.isfinite(man)) else np.nan

        if save_debug:
            image = cv2.imread(str(r["image_path"]))
            if image is not None:
                out = HORIZON_DIR / f"{r['subject_id']}_{r['view']}_{Path(r['image_path']).stem}_horizon.png"
                draw_horizon(image, r, out_path=out)
                r["verification_image_path"] = str(out)

    # ============================================================
    # CREACIÓN Y GUARDADO DEL DATAFRAME PRINCIPAL
    # ============================================================

    drop = {
        "feet",
        "head",
        "cube_base",
        "cube_top",
    }

    df = pd.DataFrame(
        [
            {
                k: v
                for k, v in r.items()
                if k not in drop
            }
            for r in records
        ]
    )

    HORIZON_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        HORIZON_DIR / "horizon_estimates.csv",
        index=False,
    )

    # ============================================================
    # DIAGNÓSTICO DE ALINEACIÓN DEL HORIZONTE
    # ============================================================

    if "delta_horizon_px" in df.columns:
        valid_delta = df[
            np.isfinite(df["delta_horizon_px"])
        ].copy()
    else:
        valid_delta = pd.DataFrame()

    # ------------------------------------------------------------
    # Resumen por vista
    # ------------------------------------------------------------

    if not valid_delta.empty:
        horizon_alignment_summary = (
            valid_delta
            .groupby("view")["delta_horizon_px"]
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

        horizon_alignment_summary.to_csv(
            HORIZON_DIR / "horizon_alignment_by_view.csv",
            index=False,
        )

    else:
        horizon_alignment_summary = pd.DataFrame(
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

        horizon_alignment_summary.to_csv(
            HORIZON_DIR / "horizon_alignment_by_view.csv",
            index=False,
        )

    print("\n=== DESALINEACION DEL HORIZONTE POR VISTA ===")

    if horizon_alignment_summary.empty:
        print(
            "No hay datos validos para analizar "
            "la alineacion del horizonte."
        )
    else:
        print(
            horizon_alignment_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.2f}",
            )
        )

    # ------------------------------------------------------------
    # Tabla detallada: una fila por imagen
    # ------------------------------------------------------------

    alignment_columns = [
        "site",
        "subject_id",
        "view",
        "image_path",
        "status",
        "img_h",
        "y_horizon_cube",
        "y_horizon_ref",
        "horizon_ref_source",
        "delta_horizon_px",
        "delta_horizon_frac",
        "y_feet",
        "y_head",
        "height_px",
        "feet_source",
        "head_source",
        "H_horizon_cube",
        "H_horizon_shared",
        "height_manual_cm",
        "error_horizon_cube",
        "error_horizon_shared",
    ]

    available_alignment_columns = [
        column
        for column in alignment_columns
        if column in df.columns
    ]

    horizon_alignment_subject = df[
        available_alignment_columns
    ].copy()

    if {
        "subject_id",
        "view",
    }.issubset(horizon_alignment_subject.columns):
        horizon_alignment_subject = (
            horizon_alignment_subject
            .sort_values(
                ["subject_id", "view"]
            )
        )

    horizon_alignment_subject.to_csv(
        HORIZON_DIR / "horizon_alignment_by_subject.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Tabla pivotada: una fila por sujeto y una columna por vista
    # ------------------------------------------------------------

    if (
            "delta_horizon_px" in df.columns
            and "subject_id" in df.columns
            and "view" in df.columns
    ):
        delta_pivot = df.pivot_table(
            index="subject_id",
            columns="view",
            values="delta_horizon_px",
            aggfunc="median",
        ).reset_index()

        # Quita el nombre interno del eje de columnas para que
        # el CSV tenga una cabecera más limpia.
        delta_pivot.columns.name = None

    else:
        delta_pivot = pd.DataFrame()

    delta_pivot.to_csv(
        HORIZON_DIR / "horizon_delta_pivot.csv",
        index=False,
    )

    # ============================================================
    # INFORMES DE RENDIMIENTO
    # ============================================================

    def stats(error_series):
        error_series = error_series.dropna()
        error_series = error_series[
            np.isfinite(error_series)
        ]

        if len(error_series) == 0:
            return 0, 0.0, 0.0, 0.0

        n_valid = len(error_series)
        bias = float(error_series.mean())
        mae = float(error_series.abs().mean())
        rmse = float(
            np.sqrt(
                (error_series ** 2).mean()
            )
        )

        return n_valid, bias, mae, rmse

    print(
        f"\nExperimento 2 "
        f"(horizonte, h_cam={CAMERA_HEIGHT_CM:.0f} cm)"
    )

    print("\nEstados:")
    print(
        df["status"]
        .value_counts()
        .to_string()
    )

    n_mask_feet = (
            df["feet_source"] == "mask_bottom"
    ).sum()

    n_mask_head = (
            df["head_source"] == "mask_trim_hair"
    ).sum()

    print(
        "\nCorrecciones:"
        f"\n  pies desde mascara: {n_mask_feet}"
        f"\n  cabeza desde mascara: {n_mask_head}"
    )

    print(
        "\nHorizonte de referencia global "
        f"(mediana front+back): y = {global_ref:.1f}"
    )

    # ------------------------------------------------------------
    # Método vía cubo frente a horizonte compartido
    # ------------------------------------------------------------

    print(
        "\nMAE por vista -> horizonte via CUBO "
        "vs horizonte COMPARTIDO:"
    )

    for view, group in df.groupby("view"):
        _, bias_cube, mae_cube, _ = stats(
            group["error_horizon_cube"]
        )

        _, bias_shared, mae_shared, _ = stats(
            group["error_horizon_shared"]
        )

        print(
            f"  {view:6s}  "
            f"cubo MAE={mae_cube:5.1f} "
            f"(bias {bias_cube:+5.1f})   "
            f"compartido MAE={mae_shared:5.1f} "
            f"(bias {bias_shared:+5.1f})"
        )

    # ------------------------------------------------------------
    # Comparación de variantes de pie
    # ------------------------------------------------------------

    print(
        "\n=== COMPARACION DE PIE "
        "(MAE / bias por vista, horizonte compartido) ==="
    )

    print(
        f"{'vista':6s}  "
        f"{'single MAE':>10s} "
        f"{'single bias':>11s}  "
        f"{'mask MAE':>9s} "
        f"{'mask bias':>9s}"
    )

    for view, group in df.groupby("view"):
        _, bias_single, mae_single, _ = stats(
            group["error_feet_single"]
        )

        _, bias_mask, mae_mask, _ = stats(
            group["error_feet_mask"]
        )

        print(
            f"{view:6s}  "
            f"{mae_single:10.1f} "
            f"{bias_single:+11.1f}  "
            f"{mae_mask:9.1f} "
            f"{bias_mask:+9.1f}"
        )

    _, global_bias_single, global_mae_single, _ = stats(
        df["error_feet_single"]
    )

    _, global_bias_mask, global_mae_mask, _ = stats(
        df["error_feet_mask"]
    )

    print(
        f"{'GLOBAL':6s}  "
        f"{global_mae_single:10.1f} "
        f"{global_bias_single:+11.1f}  "
        f"{global_mae_mask:9.1f} "
        f"{global_bias_mask:+9.1f}"
    )

    # ------------------------------------------------------------
    # Resultado específico de front y back
    # ------------------------------------------------------------

    fb = df[
        df["view"].isin(RELIABLE_VIEWS)
    ].copy()

    n_fb, bias_fb, mae_fb, rmse_fb = stats(
        fb["error_horizon_shared"]
    )

    print(
        "\n>> FRONT+BACK "
        f"(pie por defecto): "
        f"n={n_fb}  "
        f"bias={bias_fb:+.1f}  "
        f"MAE={mae_fb:.1f}  "
        f"RMSE={rmse_fb:.1f}"
    )

    # ============================================================
    # RESUMEN FINAL
    # ============================================================

    summary_rows = []

    for view, group in df.groupby("view"):
        n_view, bias_view, mae_view, rmse_view = stats(
            group["error_horizon_shared"]
        )

        summary_rows.append(
            {
                "grupo": view,
                "n": n_view,
                "bias_cm": round(bias_view, 1),
                "mae_cm": round(mae_view, 1),
                "rmse_cm": round(rmse_view, 1),
            }
        )

    summary_rows.append(
        {
            "grupo": "FRONT+BACK",
            "n": n_fb,
            "bias_cm": round(bias_fb, 1),
            "mae_cm": round(mae_fb, 1),
            "rmse_cm": round(rmse_fb, 1),
        }
    )

    n_global, bias_global, mae_global, rmse_global = stats(
        df["error_horizon_shared"]
    )

    summary_rows.append(
        {
            "grupo": "GLOBAL",
            "n": n_global,
            "bias_cm": round(bias_global, 1),
            "mae_cm": round(mae_global, 1),
            "rmse_cm": round(rmse_global, 1),
        }
    )

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        HORIZON_DIR / "horizon_summary.csv",
        index=False,
    )

    print("\nArchivos guardados:")

    print(
        "  "
        + str(
            HORIZON_DIR / "horizon_estimates.csv"
        )
    )

    print(
        "  "
        + str(
            HORIZON_DIR / "horizon_alignment_by_view.csv"
        )
    )

    print(
        "  "
        + str(
            HORIZON_DIR / "horizon_alignment_by_subject.csv"
        )
    )

    print(
        "  "
        + str(
            HORIZON_DIR / "horizon_delta_pivot.csv"
        )
    )

    print(
        "  "
        + str(
            HORIZON_DIR / "horizon_summary.csv"
        )
    )

    return df
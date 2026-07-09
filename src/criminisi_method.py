"""
criminisi_method.py  (v2 - horizonte desde las lineas del suelo)

Estimacion de altura por metrologia de vista unica (Criminisi). Invariante a la
profundidad: mide la altura por razon doble a lo largo de la vertical.

Ingredientes (todo de la propia imagen):
  1. Linea de fuga del suelo (horizonte)  -> AHORA se extrae de las LINEAS
     LARGAS del suelo (juntas de baldosa, zocalos, cantos), no de los cubos.
  2. Punto de fuga vertical                -> aristas verticales de los cubos +
     segmentos verticales de la escena.
  3. Referencia vertical de 10 cm          -> arista vertical de un cubo ArUco.
  4. Base (pies) y cima (cabeza)           -> pose + mascara (ya validado).

Como el horizonte se saca de "lineas rectas largas" genericas (no de un tipo
concreto de baldosa), el detector no sabe si son baldosas, zocalos o marcos:
solo ve segmentos. Eso lo hace mas robusto y algo mas transferible. La linea de
fuga necesita DOS direcciones distintas del suelo; si no las hay, se marca el
estado y no se reporta altura.

Modo depuracion (debug=True): dibuja las lineas de suelo detectadas de cada
direccion en color y marca los dos puntos de fuga, para comprobar la geometria.

Formula (Criminisi, Reid, Zisserman, IJCV 2000):
    Z / Z_ref = f(base, cima) / f(base_ref, cima_ref),
    f(b, t) = ||b x t|| / ((l . b) * ||v x t||)
Los signos se cancelan en el cociente: no requiere calibrar la camara.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.config import VERIFICATION_DIR, ARUCO_MARKER_SIZE_CM
from src.image_calibration import (
    estimate_body_height_pixels,
    detect_aruco_markers,
    get_manual_height_for_subject,
    put_label,
)

# --------- parametros ajustables ---------
VERTICAL_ANGLE_TOL_DEG = 25.0    # arista "vertical" si su angulo esta a +-esto de 90
MIN_EDGE_LEN_PX = 12.0           # aristas de cubo mas cortas se ignoran
FLOOR_REGION_FRAC = 0.42         # solo lineas cuyo punto medio esta por debajo de este % de alto
FLOOR_MIN_LEN_PX = 45.0          # longitud minima de linea de suelo
RANSAC_ITERS = 700               # iteraciones de RANSAC por punto de fuga
RANSAC_ANG_TOL_DEG = 2.0         # tolerancia angular para considerar una linea inlier de un PF
MIN_FLOOR_INLIERS = 4            # inliers minimos por direccion
MIN_DIR_SEPARATION_DEG = 18.0    # separacion minima entre las dos direcciones del suelo
LSD_MIN_LEN_PX = 40.0
CRIMINISI_DIR = VERIFICATION_DIR / "criminisi"
_RNG = np.random.default_rng(0)


# ============================================================
# GEOMETRIA PROYECTIVA
# ============================================================

def h(p) -> np.ndarray:
    return np.array([float(p[0]), float(p[1]), 1.0])

def line_through(p1, p2) -> np.ndarray:
    return np.cross(h(p1), h(p2))

def intersect(l1, l2) -> np.ndarray:
    return np.cross(l1, l2)

def normalize_pt(p) -> np.ndarray:
    return p / p[2] if abs(p[2]) > 1e-9 else p

def seg_angle_deg(p1, p2) -> float:
    return abs(np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))) % 180.0

def is_vertical(p1, p2) -> bool:
    return abs(seg_angle_deg(p1, p2) - 90.0) <= VERTICAL_ANGLE_TOL_DEG

def vanishing_point(lines: list[np.ndarray]) -> np.ndarray | None:
    if len(lines) < 2:
        return None
    A = np.vstack([l / (np.linalg.norm(l[:2]) + 1e-12) for l in lines])
    try:
        _, _, vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    v = vt[-1]
    return v / v[2] if abs(v[2]) > 1e-9 else v

def project_point_to_line(pt, l) -> np.ndarray:
    a, b, c = l
    x0, y0 = pt[0], pt[1]
    d = a * a + b * b
    if d < 1e-12:
        return np.array([x0, y0])
    x = (b * (b * x0 - a * y0) - a * c) / d
    y = (a * (-b * x0 + a * y0) - b * c) / d
    return np.array([x, y])


# ============================================================
# DETECCION DE SEGMENTOS
# ============================================================

def detect_line_segments(gray: np.ndarray):
    """Segmentos de linea (LSD si esta, si no Hough). Robusto a (N,4) y (N,1,4)."""
    lines = None
    try:
        lsd = cv2.createLineSegmentDetector()
        res = lsd.detect(gray)
        lines = res[0] if res is not None else None
    except Exception:
        lines = None
    segs = []
    if lines is not None and len(lines) > 0:
        arr = np.asarray(lines, dtype=float).reshape(-1, 4)
        for x1, y1, x2, y2 in arr:
            if np.hypot(x2 - x1, y2 - y1) >= LSD_MIN_LEN_PX:
                segs.append((np.array([x1, y1]), np.array([x2, y2])))
        return segs
    edges = cv2.Canny(gray, 60, 180)
    hl = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                         minLineLength=int(LSD_MIN_LEN_PX), maxLineGap=10)
    if hl is not None and len(hl) > 0:
        arr = np.asarray(hl, dtype=float).reshape(-1, 4)
        for x1, y1, x2, y2 in arr:
            segs.append((np.array([x1, y1]), np.array([x2, y2])))
    return segs


def marker_edges(corners4: np.ndarray):
    c = corners4.reshape(4, 2)
    return [(c[0], c[1]), (c[1], c[2]), (c[2], c[3]), (c[3], c[0])]


# ============================================================
# PUNTO DE FUGA VERTICAL (de los cubos + verticales de la escena)
# ============================================================

def estimate_vertical_vp(markers, segments) -> np.ndarray | None:
    lines = []
    for corners in markers:
        for p1, p2 in marker_edges(corners):
            if is_vertical(p1, p2) and np.hypot(*(p2 - p1)) >= MIN_EDGE_LEN_PX:
                lines.append(line_through(p1, p2))
    for p1, p2 in segments:
        if is_vertical(p1, p2):
            lines.append(line_through(p1, p2))
    return vanishing_point(lines)


# ============================================================
# HORIZONTE DESDE LAS LINEAS DEL SUELO (RANSAC secuencial)
# ============================================================

def _seg_info(p1, p2):
    d = p2 - p1
    n = np.linalg.norm(d) + 1e-12
    return {"p1": p1, "p2": p2, "line": line_through(p1, p2),
            "mid": (p1 + p2) / 2.0, "dir": d / n, "len": n}

def _vp_consistency_deg(vp, mid, direction) -> float:
    """Angulo entre la direccion del segmento y la direccion 'hacia el PF'."""
    if abs(vp[2]) < 1e-9:
        pred = vp[:2]
    else:
        pred = vp[:2] / vp[2] - mid
    npred = np.linalg.norm(pred) + 1e-12
    pred = pred / npred
    c = np.clip(abs(np.dot(direction, pred)), 0.0, 1.0)
    return float(np.degrees(np.arccos(c)))

def _ransac_vp(segs, iters=RANSAC_ITERS, tol=RANSAC_ANG_TOL_DEG):
    """Devuelve (vp, indices_inliers) del punto de fuga dominante."""
    n = len(segs)
    if n < 2:
        return None, []
    best_vp, best_inl = None, []
    for _ in range(iters):
        i, j = _RNG.choice(n, size=2, replace=False)
        vp = intersect(segs[i]["line"], segs[j]["line"])
        if np.linalg.norm(vp) < 1e-9:
            continue
        vp = vp / vp[2] if abs(vp[2]) > 1e-9 else vp / (np.linalg.norm(vp[:2]) + 1e-12)
        inl = [k for k in range(n)
               if _vp_consistency_deg(vp, segs[k]["mid"], segs[k]["dir"]) < tol]
        if len(inl) > len(best_inl):
            best_vp, best_inl = vp, inl
    if best_vp is not None and len(best_inl) >= 2:
        refit = vanishing_point([segs[k]["line"] for k in best_inl])
        if refit is not None:
            best_vp = refit
    return best_vp, best_inl

def estimate_ground_vanishing_line(segments, body_mask, image_shape):
    """
    Horizonte a partir de las lineas largas del suelo. Filtra a la zona inferior,
    quita verticales y lineas dentro de la persona, y busca DOS direcciones
    dominantes por RANSAC secuencial. Devuelve:
      (horizonte | None, [vp1, vp2], inliers_dir1, inliers_dir2, status)
    """
    hI, wI = image_shape[:2]
    y_floor = FLOOR_REGION_FRAC * hI

    floor = []
    for p1, p2 in segments:
        if is_vertical(p1, p2):
            continue
        if np.hypot(*(p2 - p1)) < FLOOR_MIN_LEN_PX:
            continue
        mid = (p1 + p2) / 2.0
        if mid[1] < y_floor:
            continue
        # fuera de la persona
        mx, my = int(np.clip(mid[0], 0, wI - 1)), int(np.clip(mid[1], 0, hI - 1))
        if body_mask is not None and body_mask.shape[:2] == (hI, wI) and body_mask[my, mx] > 0:
            continue
        floor.append(_seg_info(p1, p2))

    if len(floor) < 2 * MIN_FLOOR_INLIERS:
        return None, [], [], [], f"few_floor_lines({len(floor)})"

    vp1, in1 = _ransac_vp(floor)
    rest = [floor[k] for k in range(len(floor)) if k not in set(in1)]
    vp2, in2 = _ransac_vp(rest)

    inl1 = [(floor[k]["p1"], floor[k]["p2"]) for k in in1]
    inl2 = [(rest[k]["p1"], rest[k]["p2"]) for k in in2] if rest else []
    vps = [x for x in [vp1, vp2] if x is not None]

    if vp1 is None or vp2 is None or len(in1) < MIN_FLOOR_INLIERS or len(in2) < MIN_FLOOR_INLIERS:
        return None, vps, inl1, inl2, "not_enough_inliers"

    d1 = np.median(np.array([floor[k]["dir"] for k in in1]), axis=0)
    d2 = np.median(np.array([rest[k]["dir"] for k in in2]), axis=0)
    sep = np.degrees(np.arccos(np.clip(abs(np.dot(
        d1 / (np.linalg.norm(d1) + 1e-12), d2 / (np.linalg.norm(d2) + 1e-12))), 0, 1)))
    if sep < MIN_DIR_SEPARATION_DEG:
        return None, vps, inl1, inl2, f"one_direction(sep={sep:.0f})"

    horizon = np.cross(vp1, vp2)
    return horizon, [vp1, vp2], inl1, inl2, "ok"


# ============================================================
# FORMULA DE CRIMINISI
# ============================================================

def _f(base, top, l, v) -> float:
    num = np.linalg.norm(np.cross(base, top))
    den = np.dot(l, base) * np.linalg.norm(np.cross(v, top))
    return np.nan if abs(den) < 1e-9 else num / den

def criminisi_height(l, v, base, top, base_ref, top_ref, z_ref_cm) -> float:
    # la escala de la linea de fuga es arbitraria: la normalizamos para que
    # fo y fr queden bien escalados y la guarda no mate un resultado valido.
    l = np.asarray(l, float) / (np.linalg.norm(np.asarray(l, float)[:2]) + 1e-12)
    base, top, base_ref, top_ref = h(base), h(top), h(base_ref), h(top_ref)
    fo, fr = _f(base, top, l, v), _f(base_ref, top_ref, l, v)
    if not np.isfinite(fo) or not np.isfinite(fr) or abs(fr) < 1e-18:
        return np.nan
    return float(z_ref_cm * abs(fo) / abs(fr))
def pick_reference_vertical(markers):
    best, best_len = None, 0.0
    for corners in markers:
        for p1, p2 in marker_edges(corners):
            if is_vertical(p1, p2):
                L = np.hypot(*(p2 - p1))
                if L > best_len and L >= MIN_EDGE_LEN_PX:
                    base, top = (p1, p2) if p1[1] >= p2[1] else (p2, p1)
                    best, best_len = (base, top), L
    return best


# ============================================================
# UNA IMAGEN
# ============================================================

def estimate_height_criminisi(image: np.ndarray) -> dict:
    out = {"status": "ok", "height_cm": np.nan,
           "person_base": None, "person_top": None, "person_top_corr": None,
           "ref_base": None, "ref_top": None,
           "vanishing_line": None, "vertical_vp": None, "ground_vps": [],
           "floor_inliers_1": [], "floor_inliers_2": [], "height_px": np.nan}

    landmarks, body_mask = estimate_body_height_pixels(image)
    if not str(landmarks.get("landmark_status", "")).startswith("ok"):
        out["status"] = f"person_failed_{landmarks.get('landmark_status','')}"
        return out
    feet = (landmarks["feet_x_px"], landmarks["feet_y_px"])
    head = (landmarks["head_x_px"], landmarks["head_y_px"])
    out["person_base"], out["person_top"] = feet, head
    out["height_px"] = landmarks.get("height_px", np.nan)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids = detect_aruco_markers(gray)
    markers = [c.reshape(4, 2) for c in corners] if ids is not None else []
    if not markers:
        out["status"] = "no_aruco"
        return out

    ref = pick_reference_vertical(markers)
    if ref is None:
        out["status"] = "no_vertical_reference"
        return out
    out["ref_base"], out["ref_top"] = ref

    segments = detect_line_segments(gray)
    v = estimate_vertical_vp(markers, segments)
    if v is None:
        out["status"] = "no_vertical_vp"
        return out
    out["vertical_vp"] = v

    l, vps, in1, in2, gstatus = estimate_ground_vanishing_line(segments, body_mask, image.shape)
    out["ground_vps"], out["floor_inliers_1"], out["floor_inliers_2"] = vps, in1, in2
    if l is None:
        out["status"] = f"no_ground_vanishing_line:{gstatus}"
        return out
    out["vanishing_line"] = l

    vert_line = line_through(feet, (v[0], v[1]))
    head_corr = project_point_to_line(head, vert_line)
    out["person_top_corr"] = head_corr

    z = criminisi_height(l, v, feet, head_corr, out["ref_base"], out["ref_top"],
                         ARUCO_MARKER_SIZE_CM)
    out["height_cm"] = z
    if not np.isfinite(z) or z <= 0 or z > 300:
        out["status"] = "criminisi_out_of_range"
    return out


# ============================================================
# DIBUJO (con modo depuracion)
# ============================================================

def _draw_line_full(img, l, color, thick=2):
    hI, wI = img.shape[:2]
    a, b, c = l
    if abs(b) > 1e-6:
        p1 = (0, int(-c / b)); p2 = (wI, int(-(c + a * wI) / b))
    elif abs(a) > 1e-6:
        p1 = (int(-c / a), 0); p2 = (int(-(c + b * hI) / a), hI)
    else:
        return
    cv2.line(img, p1, p2, color, thick, cv2.LINE_AA)

def draw_criminisi(image, res, manual_cm=np.nan, out_path=None, debug=True):
    img = image.copy()
    hI, wI = img.shape[:2]

    if debug:
        for seg in res.get("floor_inliers_1", []):
            if len(seg) == 2:
                cv2.line(img, tuple(np.int32(seg[0])), tuple(np.int32(seg[1])), (0, 165, 255), 2, cv2.LINE_AA)
        for seg in res.get("floor_inliers_2", []):
            if len(seg) == 2:
                cv2.line(img, tuple(np.int32(seg[0])), tuple(np.int32(seg[1])), (255, 90, 200), 2, cv2.LINE_AA)
        for vp in res.get("ground_vps", []):
            if abs(vp[2]) > 1e-9:
                x, y = int(vp[0] / vp[2]), int(vp[1] / vp[2])
                if -wI < x < 2 * wI and -hI < y < 2 * hI:
                    cv2.circle(img, (int(np.clip(x, 0, wI - 1)), int(np.clip(y, 0, hI - 1))),
                               9, (0, 140, 255), 2)

    if res["vanishing_line"] is not None:
        _draw_line_full(img, res["vanishing_line"], (0, 220, 220), 2)
        put_label(img, "linea de fuga (suelo)", (30, 40), color=(0, 220, 220))

    v = res.get("vertical_vp")
    if v is not None and abs(v[2]) > 1e-9:
        vx, vy = int(v[0] / v[2]), int(v[1] / v[2])
        if 0 <= vx < wI and 0 <= vy < hI:
            cv2.circle(img, (vx, vy), 8, (255, 120, 0), -1)
            put_label(img, "PF vertical", (vx + 8, vy), color=(255, 120, 0))

    if res["ref_base"] is not None:
        b = tuple(np.int32(res["ref_base"])); t = tuple(np.int32(res["ref_top"]))
        cv2.line(img, b, t, (0, 220, 0), 4, cv2.LINE_AA)
        put_label(img, "ref 10 cm", (b[0] + 6, b[1] + 18), color=(0, 220, 0))

    if res["person_base"] is not None:
        b = tuple(np.int32(res["person_base"]))
        top_pt = res.get("person_top_corr")
        if top_pt is None:
            top_pt = res["person_top"]
        t = tuple(np.int32(top_pt))
        cv2.line(img, b, t, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(img, t, 7, (0, 255, 255), -1); put_label(img, "cabeza", (t[0] + 8, t[1]))
        cv2.circle(img, b, 7, (0, 0, 255), -1); put_label(img, "pies", (b[0] + 8, b[1]), color=(0, 0, 255))

    est = res.get("height_cm", np.nan)
    err = (est - manual_cm) if (np.isfinite(est) and np.isfinite(manual_cm)) else np.nan
    n1, n2 = len(res.get("floor_inliers_1", [])), len(res.get("floor_inliers_2", []))
    lines = [
        "Criminisi - metrologia de vista unica",
        f"status: {res['status']}",
        f"lineas suelo: dir1={n1}  dir2={n2}",
        (f"altura estimada: {est:.1f} cm" if np.isfinite(est) else "altura estimada: NA"),
        (f"altura manual: {manual_cm:.1f} cm" if np.isfinite(manual_cm) else "altura manual: NA"),
        (f"error: {err:+.1f} cm" if np.isfinite(err) else "error: NA"),
    ]
    cv2.rectangle(img, (20, hI - 20 - 30 * len(lines)), (620, hI - 15), (0, 0, 0), -1)
    for i, t in enumerate(lines):
        put_label(img, t, (30, hI - 25 - 30 * (len(lines) - 1 - i)), color=(0, 255, 255))

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img)
    return img


# ============================================================
# RUNNER
# ============================================================

def run_criminisi_experiment(assets_by_subject, manual_heights_df,
                             max_images=None, save_debug=True, debug=True):
    rows, n = [], 0
    for _, assets in assets_by_subject.items():
        for asset in assets:
            if max_images is not None and n >= max_images:
                break
            image = cv2.imread(str(asset.path))
            if image is None:
                continue
            try:
                res = estimate_height_criminisi(image)
            except Exception as e:
                res = {"status": f"error_{type(e).__name__}", "height_cm": np.nan,
                       "person_base": None, "person_top": None, "person_top_corr": None,
                       "ref_base": None, "ref_top": None, "vanishing_line": None,
                       "vertical_vp": None, "ground_vps": [],
                       "floor_inliers_1": [], "floor_inliers_2": [], "height_px": np.nan}
            manual = get_manual_height_for_subject(manual_heights_df, asset.subject_id)
            est = res.get("height_cm", np.nan)
            err = (est - manual) if (np.isfinite(est) and np.isfinite(manual)) else np.nan
            rows.append({
                "site": asset.site, "subject_id": asset.subject_id, "view": asset.view,
                "image_path": str(asset.path), "status": res["status"],
                "height_px": res.get("height_px", np.nan),
                "height_cm_criminisi": est, "height_manual_cm": manual, "error_cm": err,
                "n_floor_dir1": len(res.get("floor_inliers_1", [])),
                "n_floor_dir2": len(res.get("floor_inliers_2", [])),
                "has_vanishing_line": res["vanishing_line"] is not None,
            })
            if save_debug:
                out = CRIMINISI_DIR / f"{asset.subject_id}_{asset.view}_{Path(asset.path).stem}_criminisi.png"
                draw_criminisi(image, res, manual_cm=manual, out_path=out, debug=debug)
                rows[-1]["verification_image_path"] = str(out)
            n += 1
        if max_images is not None and n >= max_images:
            break

    df = pd.DataFrame(rows)
    CRIMINISI_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CRIMINISI_DIR / "criminisi_estimates.csv", index=False)

    ok = df[(df["status"] == "ok") & df["error_cm"].notna()]
    summ = [{"grupo": "estados", "detalle": df["status"].value_counts().to_dict()}]
    for name, g in [("GLOBAL", ok)] + list(ok.groupby("view")):
        gg = g if isinstance(g, pd.DataFrame) else g[1]
        e = gg["error_cm"].dropna()
        if len(e):
            summ.append({"grupo": name if isinstance(name, str) else name, "n": int(len(e)),
                         "bias_cm": round(float(e.mean()), 1), "mae_cm": round(float(e.abs().mean()), 1),
                         "rmse_cm": round(float(np.sqrt((e ** 2).mean())), 1)})
    pd.DataFrame(summ).to_csv(CRIMINISI_DIR / "criminisi_summary.csv", index=False)

    print("Criminisi - cobertura por estado:")
    print(df["status"].value_counts().to_string())
    if len(ok):
        print(f"\nError global (status ok, n={len(ok)}): bias {ok['error_cm'].mean():+.1f}  "
              f"MAE {ok['error_cm'].abs().mean():.1f}  RMSE {np.sqrt((ok['error_cm']**2).mean()):.1f} cm")
        for v, g in ok.groupby("view"):
            print(f"  {v:6s} n={len(g):3d}  MAE {g['error_cm'].abs().mean():.1f}  bias {g['error_cm'].mean():+.1f}")
    return df
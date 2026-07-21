"""
sweep_head_width.py

Barrido del umbral HEAD_WIDTH_FRAC (Opcion 1) para la deteccion de la corona de
la cabeza, usando el metodo del HORIZONTE CORREGIDO (el de menor MAE). Reejecuta
run_corrected_horizon_experiment con cada valor del umbral y mide el error, para
elegir el umbral con datos en vez de a ojo.

NO modifica main.py ni ningun modulo: solo cambia en memoria la constante
horizon_corrected.HEAD_WIDTH_FRAC antes de cada corrida (head_from_mask_trim_hair
la lee del modulo en cada llamada, asi que el cambio tiene efecto).

Que mide, por cada valor del umbral (metodo 'corrected_horizon'):
    - FRONT+BACK MAE
    - LEFT MAE
    - RIGHT MAE
todos con la columna error_horizon_corrected.

Uso:
    python sweep_head_width.py            # barre Colombia y Brasil
    python sweep_head_width.py CO         # solo Colombia
    python sweep_head_width.py BR         # solo Brasil
"""

import sys

import pandas as pd

import src.horizon_corrected as hc
from src.config import (
    COLOMBIA_DIR,
    BARCELONA_DIR,
    BRASIL_DIR,
    create_output_folders,
    VERIFICATION_DIR,
)
from src.io_data import discover_subject_images
from src.harmonize_features import save_equalized_tables


# Valores del umbral a probar.
FRAC_VALUES = [0.45, 0.50, 0.55, 0.60]

# Vistas fiables (donde la cabeza manda mas).
RELIABLE_VIEWS = ("front", "back")

# Columna de error del metodo corregido (el de menor MAE).
ERROR_COLUMN = "error_horizon_corrected"


def read_files(site: str):
    site_dirs = {"CO": COLOMBIA_DIR, "ES": BARCELONA_DIR, "BR": BRASIL_DIR}
    return discover_subject_images(site_dirs[site], site=site)


def _mae(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce")
    return float(s.abs().mean())


def sweep_site(site: str, assets_by_subject: dict, manual_heights_df: pd.DataFrame) -> pd.DataFrame:
    """Barre FRAC_VALUES para una sede (horizonte corregido) y devuelve la tabla de MAE."""
    original = hc.HEAD_WIDTH_FRAC
    rows = []
    try:
        for frac in FRAC_VALUES:
            hc.HEAD_WIDTH_FRAC = frac  # cambia el umbral que usa la deteccion de cabeza
            df = hc.run_corrected_horizon_experiment(
                assets_by_subject=assets_by_subject,
                manual_heights_df=manual_heights_df,
                max_images=None,
                save_debug=False,   # sin guardar imagenes: mas rapido
            )

            fb = df[df["view"].isin(RELIABLE_VIEWS)]
            left = df[df["view"] == "left"]
            right = df[df["view"] == "right"]

            rows.append({
                "site": site,
                "HEAD_WIDTH_FRAC": frac,
                "n_fb": int(fb[ERROR_COLUMN].notna().sum()),
                "MAE_front_back": round(_mae(fb[ERROR_COLUMN]), 2),
                "MAE_left": round(_mae(left[ERROR_COLUMN]), 2),
                "MAE_right": round(_mae(right[ERROR_COLUMN]), 2),
            })
    finally:
        hc.HEAD_WIDTH_FRAC = original  # restaura el valor original pase lo que pase

    out = pd.DataFrame(rows)
    if not out.empty:
        best_idx = out["MAE_front_back"].idxmin()
        out["mejor_fb"] = ""
        out.loc[best_idx, "mejor_fb"] = "<== mejor"
    return out


def main():
    create_output_folders()

    which = sys.argv[1].upper() if len(sys.argv) > 1 else "ALL"
    colombia_equalized, _barcelona_equalized, brasil_equalized = save_equalized_tables()

    plan = []
    if which in ("CO", "ALL"):
        plan.append(("Colombia", "CO", colombia_equalized))
    if which in ("BR", "ALL"):
        plan.append(("Brasil", "BR", brasil_equalized))

    all_results = []
    for site_name, site_code, manual in plan:
        print("\n" + "=" * 60)
        print(f"BARRIDO HEAD_WIDTH_FRAC (horizonte corregido) - {site_name}")
        print("=" * 60)
        assets = read_files(site_code)
        res = sweep_site(site_name, assets, manual)
        print("\n" + res.to_string(index=False))
        all_results.append(res)

    if all_results:
        final = pd.concat(all_results, ignore_index=True)
        out_path = VERIFICATION_DIR / "head_width_frac_sweep.csv"
        final.to_csv(out_path, index=False)
        print(f"\nResultados guardados en: {out_path}")
        print(
            "\nElige el HEAD_WIDTH_FRAC con menor MAE_front_back y ponlo en "
            "horizon_corrected.py (y en horizon_method.py si quieres coherencia)."
        )


if __name__ == "__main__":
    main()
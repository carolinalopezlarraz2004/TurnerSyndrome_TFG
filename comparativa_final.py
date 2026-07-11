"""
comparativa_final.py

Genera un CSV unico por imagen que junta el Experimento 1 (escalas) y el
Experimento 2 (metodo del horizonte), con la escala pixel->cm de cada metodo
(la del horizonte es EQUIVALENTE: altura_estimada / altura_px) y todos los
errores, para poder comparar metodos lado a lado.

Salida: outputs/verification/comparativa_final.csv
Ademas imprime una tabla resumen de MAE por vista y metodo.

Uso (archivo aparte, no en el main):
    python comparativa_final.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

V = Path("outputs/verification")
E1 = V / "height_estimates_experiment1.csv"
HZ = V / "horizon" / "horizon_estimates.csv"
OUT = V / "comparativa_final.csv"


def col(df, name):
    """Devuelve la columna si existe, o una de NaN si no."""
    return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)


def main():
    e1 = pd.read_csv(E1)
    hz = pd.read_csv(HZ)

    # ---------- lado EXPERIMENTO 1 ----------
    e1f = pd.DataFrame()
    e1f["image_path"] = e1["image_path"]
    e1f["site"] = col(e1, "site")
    e1f["subject_id"] = col(e1, "subject_id")
    e1f["view"] = col(e1, "view")
    e1f["height_manual_cm"] = col(e1, "height_manual_cm")
    e1f["e1_height_px"] = col(e1, "height_px")
    # escala linea de suelo (la mejor del Exp 1)
    e1f["e1_scale_groundline_cmpx"] = col(e1, "scale_ground_line")
    e1f["e1_H_groundline"] = col(e1, "height_cm_ground_line_scale")
    e1f["e1_err_groundline"] = col(e1, "error_cm_ground_line_scale")
    # escala mejor cara
    e1f["e1_scale_bestquality_cmpx"] = col(e1, "scale_best_quality")
    e1f["e1_H_bestquality"] = col(e1, "height_cm_best_quality_scale")
    e1f["e1_err_bestquality"] = col(e1, "error_cm_best_quality_scale")
    # escala linea de suelo CORREGIDA (leave-one-out), si existe
    e1f["e1_H_groundline_corr"] = col(e1, "height_cm_ground_line_scale_corrected")
    e1f["e1_err_groundline_corr"] = col(e1, "error_cm_ground_line_scale_corrected")

    # ---------- lado HORIZONTE ----------
    hzf = pd.DataFrame()
    hzf["image_path"] = hz["image_path"]
    hzf["hz_height_px"] = col(hz, "height_px")
    hzf["hz_feet_source"] = col(hz, "feet_source")
    hzf["hz_head_source"] = col(hz, "head_source")
    hzf["hz_y_horizon_ref"] = col(hz, "y_horizon_ref")
    hzf["hz_y_horizon_cube"] = col(hz, "y_horizon_cube")

    hpx = col(hz, "height_px").replace(0, np.nan)
    # horizonte via cubo
    hzf["hz_H_cube"] = col(hz, "H_horizon_cube")
    hzf["hz_scale_cube_cmpx"] = hzf["hz_H_cube"] / hpx      # escala EQUIVALENTE
    hzf["hz_err_cube"] = col(hz, "error_horizon_cube")
    # horizonte compartido
    hzf["hz_H_shared"] = col(hz, "H_horizon_shared")
    hzf["hz_scale_shared_cmpx"] = hzf["hz_H_shared"] / hpx  # escala EQUIVALENTE
    hzf["hz_err_shared"] = col(hz, "error_horizon_shared")
    # variantes de pie (con horizonte compartido)
    hzf["hz_H_feet_single"] = col(hz, "H_shared_feet_single")
    hzf["hz_err_feet_single"] = col(hz, "error_feet_single")
    hzf["hz_H_feet_mask"] = col(hz, "H_shared_feet_mask")
    hzf["hz_err_feet_mask"] = col(hz, "error_feet_mask")

    # ---------- merge por imagen ----------
    final = e1f.merge(hzf, on="image_path", how="outer")

    # orden de columnas
    id_cols = ["site", "subject_id", "view", "image_path", "height_manual_cm"]
    other = [c for c in final.columns if c not in id_cols]
    final = final[id_cols + other]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUT, index=False)
    print(f"CSV final por imagen guardado en: {OUT}   ({len(final)} filas)")

    # ---------- resumen: MAE por vista y metodo ----------
    methods = {
        "E1 escala suelo":        "e1_err_groundline",
        "E1 escala mejor cara":   "e1_err_bestquality",
        "E1 suelo corregida*":    "e1_err_groundline_corr",
        "HZ via cubo":            "hz_err_cube",
        "HZ compartido":          "hz_err_shared",
        "HZ compartido pie single":"hz_err_feet_single",
        "HZ compartido pie mask":  "hz_err_feet_mask",
    }

    def mae(sub, c):
        e = sub[c].dropna(); e = e[np.isfinite(e)]
        return e.abs().mean() if len(e) else np.nan

    views = ["front", "back", "left", "right"]
    rows = []
    for name, c in methods.items():
        if c not in final.columns:
            continue
        row = {"metodo": name}
        for v in views:
            row[v] = mae(final[final["view"] == v], c)
        fb = final[final["view"].isin(["front", "back"])]
        row["FRONT+BACK"] = mae(fb, c)
        row["GLOBAL"] = mae(final, c)
        rows.append(row)

    tabla = pd.DataFrame(rows).set_index("metodo").round(1)
    print("\n=== MAE (cm) por metodo y vista ===")
    print(tabla.to_string())
    print("\n(* la corregida APRENDE de las alturas reales con leave-one-out;")
    print("   las demas no usan las alturas reales)")
    tabla.to_csv(V / "comparativa_metodos_mae.csv")

    print("\n=== mejor metodo por columna ===")
    for c in tabla.columns:
        s = tabla[c].dropna()
        if len(s):
            print(f"  {c:11s}: {s.idxmin()}  ({s.min()} cm)")


if __name__ == "__main__":
    main()
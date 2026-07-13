import pandas as pd
from src.config import (
    COLOMBIA_DIR,
    BARCELONA_DIR,
    BRASIL_DIR,
    COLOMBIA_TABLE,
    BARCELONA_TABLE,
    COLOMBIA_PREPROCESSED_TABLE,
    BARCELONA_PREPROCESSED_TABLE,
    QUALITY_REPORT_TABLE,
    FLAGGED_VALUES_REPORT_TABLE,
    PSEUDO_ID_MAPPING_PRIVATE_TABLE,
    IMAGE_CALIBRATION_TABLE,
    CALIBRATION_REFERENCES_TABLE,
    VERIFICATION_DIR,
    create_output_folders,
)
from src.io_data import (
    discover_subject_images,
    load_colombia_table,
    load_barcelona_tables,
)
from src.harmonize_features import save_equalized_tables
from src.data_dictionary import save_canonical_dictionary
from src.quality_control import (
    add_quality_control_flags,
    impute_missing_values,
    combine_quality_reports,
    combine_flagged_values_reports,
)
from src.pseudonymization import pseudonymize_preprocessed_table
from src.image_calibration import (
    save_image_calibration_table,
    run_height_experiment_1_auto,
    summarize_height_experiment_1,
)
from src.image_calibration import (sweep_crown_offset_factor, plot_view_results, apply_calibrated_correction, compare_correction,)
from src.horizon_method import run_horizon_experiment

from src.lateral_skeletal_height import (
    run_lateral_skeletal_experiment,
)

from src.lateral_horizon_diagnostic import (
    run_lateral_horizon_diagnostic,
)

from src.horizon_corrected import (
    run_corrected_horizon_experiment,
)

def read_files(site: str):
    """
    Read and organize image files from one dataset.
    """
    site_dirs = {
        "CO": COLOMBIA_DIR,
        "ES": BARCELONA_DIR,
        "BR": BRASIL_DIR,
    }
    return discover_subject_images(site_dirs[site], site=site)
def read_tables():
    """
    Read the original measurement tables.
    """
    colombia_df = load_colombia_table(COLOMBIA_TABLE)
    barcelona_sheets = load_barcelona_tables(BARCELONA_TABLE)
    return colombia_df, barcelona_sheets
def main():
    # ========================================================
    # CREATE OUTPUT FOLDERS
    # ========================================================
    create_output_folders()
    # ========================================================
    # READ IMAGE FILES AND ORIGINAL TABLES
    # ========================================================
    # For now, we start with Colombia because the ArUco marker size is confirmed:
    # each marker is 10 cm x 10 cm.
    assets_by_subject = read_files("CO")
    colombia_df, barcelona_sheets = read_tables()
    print(f"Subjects found in Colombia image folder: {len(assets_by_subject)}")
    print("\nColombia table:")
    print(colombia_df.shape)
    print("\nBarcelona sheets:")
    for sheet_name, df in barcelona_sheets.items():
        print(f"  - {sheet_name}: {df.shape}")
    print("\nColombia columns:")
    print(colombia_df.columns.tolist())
    print("\nBarcelona Raw columns:")
    print(barcelona_sheets["Raw_Match_Anon"].columns.tolist())
    print("\nBarcelona Recon columns:")
    print(barcelona_sheets["3D_Recon_Clean"].columns.tolist())
    # ========================================================
    # EQUALIZATION
    # ========================================================
    colombia_equalized, barcelona_equalized = save_equalized_tables()
    print("\nColombia equalized:")
    print(colombia_equalized.shape)
    print("\nBarcelona equalized:")
    print(barcelona_equalized.shape)
    print("\nColumns are equal:")
    print(list(colombia_equalized.columns) == list(barcelona_equalized.columns))
    # ========================================================
    # CANONICAL DATA DICTIONARY
    # ========================================================
    canonical_dictionary = save_canonical_dictionary()
    print("\nCanonical dictionary saved:")
    print(canonical_dictionary.shape)
    # ========================================================
    # QUALITY CONTROL + MISSING VALUE IMPUTATION
    # ========================================================
    colombia_qc = add_quality_control_flags(colombia_equalized)
    barcelona_qc = add_quality_control_flags(barcelona_equalized)
    colombia_preprocessed = impute_missing_values(colombia_qc)
    barcelona_preprocessed = impute_missing_values(barcelona_qc)
    # ========================================================
    # PSEUDONYMIZATION
    # ========================================================
    colombia_preprocessed, colombia_mapping = pseudonymize_preprocessed_table(
        df=colombia_preprocessed,
        site_prefix="CO",
        site_name="Colombia",
    )
    barcelona_preprocessed, barcelona_mapping = pseudonymize_preprocessed_table(
        df=barcelona_preprocessed,
        site_prefix="ES",
        site_name="Barcelona",
    )
    pseudo_id_mapping = pd.concat(
        [colombia_mapping, barcelona_mapping],
        ignore_index=True,
    )
    PSEUDO_ID_MAPPING_PRIVATE_TABLE.parent.mkdir(parents=True, exist_ok=True)
    pseudo_id_mapping.to_csv(PSEUDO_ID_MAPPING_PRIVATE_TABLE, index=False)
    colombia_preprocessed.to_csv(COLOMBIA_PREPROCESSED_TABLE, index=False)
    barcelona_preprocessed.to_csv(BARCELONA_PREPROCESSED_TABLE, index=False)
    print("\nPseudonymized preprocessed tables saved:")
    print(f"  - {COLOMBIA_PREPROCESSED_TABLE}")
    print(f"  - {BARCELONA_PREPROCESSED_TABLE}")
    print("\nPrivate pseudo-ID mapping saved:")
    print(f"  - {PSEUDO_ID_MAPPING_PRIVATE_TABLE}")
    print(pseudo_id_mapping.shape)
    # ========================================================
    # QUALITY REPORTS
    # ========================================================
    quality_report = combine_quality_reports(
        colombia_qc=colombia_preprocessed,
        barcelona_qc=barcelona_preprocessed,
    )
    QUALITY_REPORT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    quality_report.to_csv(QUALITY_REPORT_TABLE, index=False)
    print("\nQuality report saved:")
    print(f"  - {QUALITY_REPORT_TABLE}")
    print(quality_report.shape)
    flagged_values_report = combine_flagged_values_reports(
        colombia_qc=colombia_preprocessed,
        barcelona_qc=barcelona_preprocessed,
    )
    FLAGGED_VALUES_REPORT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    flagged_values_report.to_csv(FLAGGED_VALUES_REPORT_TABLE, index=False)
    print("\nFlagged values report saved:")
    print(f"  - {FLAGGED_VALUES_REPORT_TABLE}")
    print(flagged_values_report.shape)
    print("\nColombia QC flag counts:")
    print(colombia_preprocessed["QC_FLAGS"].value_counts(dropna=False))
    print("\nBarcelona QC flag counts:")
    print(barcelona_preprocessed["QC_FLAGS"].value_counts(dropna=False))
    print("\nColombia imputed values:")
    print(colombia_preprocessed["N_IMPUTED_VALUES"].sum())
    print("\nBarcelona imputed values:")
    print(barcelona_preprocessed["N_IMPUTED_VALUES"].sum())
    # ========================================================
    # IMAGE CALIBRATION
    # ========================================================
    # This step estimates the cm/pixel scale for each Colombia image.
    # It also saves verification images so we can visually check whether
    # the ArUco markers were detected correctly.
    calibration_df, calibration_references_df = save_image_calibration_table(
        assets_by_subject=assets_by_subject,
        save_debug=True,
    )
    print("\nImage calibration table saved:")
    print(f"  - {IMAGE_CALIBRATION_TABLE}")
    print(calibration_df.shape)
    print("\nCalibration references table saved:")
    print(f"  - {CALIBRATION_REFERENCES_TABLE}")
    print(calibration_references_df.shape)
    print("\nCalibration status counts:")
    print(calibration_df["calibration_status"].value_counts(dropna=False))
    print("\nNumber of detected references per image:")
    print(calibration_df["n_references_detected"].value_counts(dropna=False).sort_index())
    # ========================================================
    # EXPERIMENT 1: AUTOMATIC HEIGHT ESTIMATION
    # ========================================================
    # This first experiment tests whether the ArUco-derived scales can provide
    # a preliminary height estimate when head and feet points are extracted
    # automatically from the body silhouette.
    #
    # It compares three simple scale strategies:
    #   1. median scale from all detected references
    #   2. scale from the best-quality reference
    #   3. scale from the reference closest to the detected feet point
    #
    # The goal is not to obtain the final method yet, but to evaluate whether
    # a simple pixel-to-centimeter approach is reasonable before moving to
    # more advanced projective geometry methods.
    height_exp1_df = run_height_experiment_1_auto(
        assets_by_subject=assets_by_subject,
        calibration_df=calibration_df,
        references_df=calibration_references_df,
        manual_heights_df=colombia_equalized,
        max_images=None,
        save_debug=True,
    )
    height_exp1_summary = summarize_height_experiment_1(height_exp1_df)
    print("\nExperiment 1 automatic height estimates saved:")
    print(f"  - {VERIFICATION_DIR / 'height_estimates_experiment1.csv'}")
    print(height_exp1_df.shape)
    print("\nExperiment 1 summary saved:")
    print(f"  - {VERIFICATION_DIR / 'height_estimates_experiment1_summary.csv'}")
    print(height_exp1_summary)
    if not height_exp1_df.empty:
        print("\nExperiment 1 mean absolute errors:")
        print(
            height_exp1_df[
                [
                    "error_cm_median_scale",
                    "error_cm_best_quality_scale",
                    "error_cm_closest_feet_scale",
                ]
            ].abs().mean()
        )
    print("\nSweep del offset de corona:")
    sweep_crown_offset_factor(height_exp1_df, scale_column="scale_closest_feet")
    sweep_crown_offset_factor(height_exp1_df, scale_column="scale_best_quality")
    df = pd.read_csv(VERIFICATION_DIR / "height_estimates_experiment1.csv")
    plot_view_results(df, view="back", method="ground_line_scale",
                      save_path=VERIFICATION_DIR / "back_ground_line.png")
    # aplica la correccion calibrada por vista (leave-one-out) y re-guarda el CSV
    height_exp1_df = apply_calibrated_correction(
        height_exp1_df, method="ground_line_scale", by="view"
    )
    height_exp1_df.to_csv(
        VERIFICATION_DIR / "height_estimates_experiment1.csv", index=False
    )
    # tabla antes/despues
    print("\nCorreccion calibrada (antes/despues):")
    print(compare_correction(height_exp1_df, method="ground_line_scale").round(1).to_string(index=False))
    # ========================================================
    # EXPERIMENT 2: MÉTODO DEL HORIZONTE
    # Cámara situada a 1 metro de altura
    # ========================================================

    horizon_df = run_horizon_experiment(
        assets_by_subject=assets_by_subject,
        manual_heights_df=colombia_equalized,
        max_images=None,
        save_debug=True,
    )

    print(
        "\nExperimento 2 "
        "(método del horizonte) guardado:"
    )

    print(
        " - "
        + str(
            VERIFICATION_DIR
            / "horizon"
            / "horizon_estimates.csv"
        )
    )

    print(
        "\nEstados del Experimento 2:"
    )

    print(
        horizon_df[
            "status"
        ].value_counts(
            dropna=False
        )
    )

    # ========================================================
    # EXPERIMENT 2B: DIAGNÓSTICO DEL HORIZONTE LATERAL
    # ========================================================

    # Este análisis utiliza los resultados ya generados por el
    # Experimento 2 y compara dos alternativas en left/right:
    #
    #   1. horizonte propio calculado a partir del cubo de cada imagen;
    #   2. horizonte compartido o transferido desde otra vista.
    #
    # De esta manera podemos comprobar si el elevado error lateral
    # está provocado por una estimación incorrecta del horizonte.

    (
        lateral_horizon_predictions_df,
        lateral_horizon_summary_df,
    ) = run_lateral_horizon_diagnostic(
        horizon_estimates=horizon_df,
        camera_height_cm=100.0,
    )

    print(
        "\nDiagnóstico del horizonte lateral guardado:"
    )

    print(
        " - "
        + str(
            VERIFICATION_DIR
            / "lateral_horizon_diagnostic"
            / "lateral_horizon_image_predictions.csv"
        )
    )

    print(
        "\nResumen del diagnóstico del horizonte lateral:"
    )

    print(
        lateral_horizon_summary_df.round(
            2
        ).to_string(
            index=False
        )
    )

    # ========================================================
    # EXPERIMENT 2 CORRECTED: HORIZONTE SEGÚN LA VISTA
    # ========================================================

    corrected_horizon_df = (
        run_corrected_horizon_experiment(
            assets_by_subject=assets_by_subject,
            manual_heights_df=colombia_equalized,
            max_images=None,
            save_debug=True,
        )
    )

    print(
        "\nExperimento 2 corregido guardado:"
    )

    print(
        " - "
        + str(
            VERIFICATION_DIR
            / "horizon"
            / "corrected"
            / "corrected_horizon_estimates.csv"
        )
    )

    # ========================================================
    # EXPERIMENT 3: ALTURA LATERAL MEDIANTE FEATURES
    # ESQUELÉTICAS Y FUSIÓN LEFT/RIGHT
    # ========================================================

    # Este experimento utiliza las vistas left y right para extraer:
    #
    #   - una estimación métrica inicial;
    #   - proporciones hombro-cadera;
    #   - proporciones cadera-rodilla;
    #   - proporciones rodilla-tobillo.
    #
    # Después fusiona ambas vistas en una única fila por sujeto
    # y compara una calibración lineal con una regresión Ridge.

    (
        lateral_predictions_df,
        lateral_summary_df,
    ) = run_lateral_skeletal_experiment(
        assets_by_subject=assets_by_subject,
        manual_heights_df=colombia_equalized,
        calibration_df=calibration_df,
        references_df=calibration_references_df,
        max_images=None,
    )

    print(
        "\nExperimento 3 "
        "(estimación lateral esquelética) guardado:"
    )

    print(
        " - "
        + str(
            VERIFICATION_DIR
            / "lateral_skeletal_height"
            / "lateral_subject_predictions.csv"
        )
    )

    print(
        "\nResumen del Experimento 3:"
    )

    print(
        lateral_summary_df.round(
            2
        ).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
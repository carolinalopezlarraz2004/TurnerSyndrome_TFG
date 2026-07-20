"""
organize_brasil_data.py

One-shot helper that ingests the raw Brazilian dataset into the project.

The Brazil images arrive as a flat pile of files (not organised into one folder
per subject like Colombia), split across one or more city folders
(e.g. PortoAlegre, Maceio). This script:

    1. Reads the subject IDs from the measurements Excel ("Clean" sheet).
    2. Recursively scans SOURCE_DIR for image files and groups them by subject ID
       (the text before the first underscore, e.g. "BRSTM0487_right_B.jpg"
       -> "BRSTM0487").
    3. For every subject THAT APPEARS IN THE EXCEL, creates a folder inside
       data_files/data_files_brasil/ and copies ALL of that subject's images
       (front/back/left/right) into it, so the layout matches Colombia and works
       with discover_subject_images(). Images whose subject is not in the Excel
       are ignored.
    4. Copies the measurements Excel into data_files_brasil/data_files_brasil_features/.

Nothing outside data_files/data_files_brasil/ is touched, files are COPIED
(never moved) so the originals stay intact, and no CSV report is written
(everything is printed to the console). Running it twice is safe: folders are
reused and files are overwritten.

Usage:
    python organize_brasil_data.py                 # uses SOURCE_DIR below
    python organize_brasil_data.py "/other/path"   # override the source for one run
    python organize_brasil_data.py --dry-run       # show what would happen, copy nothing
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION  (the only machine-specific line is SOURCE_DIR)
# ============================================================

# Folder that contains the raw Brazil download: the city subfolders
# (PortoAlegre, Maceio, ...) AND the Brasil26.xlsx file.
SOURCE_DIR = Path("/Users/carolinalopezlarraz/Downloads/DadesBrasil2026")

# Name of the measurements workbook inside SOURCE_DIR.
EXCEL_NAME = "Brasil26.xlsx"

# Sheet + header settings used to read the subject IDs.
# The "Clean" sheet has a two-row header; row index 1 holds the real names.
EXCEL_SHEET = "Clean"
EXCEL_HEADER_ROW = 1
EXCEL_ID_COLUMN = "ID"

# Destination inside the project (computed relative to this file, so it works
# on any machine without editing). This file is expected to live at the repo root.
PROJECT_ROOT = Path(__file__).resolve().parent
BRASIL_DIR = PROJECT_ROOT / "data_files" / "data_files_brasil"
BRASIL_FEATURES_DIR = BRASIL_DIR / "data_files_brasil_features"

# Accepted image extensions (kept in sync with src/config.IMAGE_EXTENSIONS).
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Anatomical views expected for every subject.
EXPECTED_VIEWS = ["front", "back", "left", "right"]


# ============================================================
# HELPERS
# ============================================================

def subject_id_from_filename(filename: str) -> str | None:
    """Return the subject ID (text before the first underscore), or None."""
    stem = Path(filename).stem
    if "_" not in stem:
        return None
    return stem.split("_")[0]


def infer_view(filename: str) -> str | None:
    """Return the anatomical view found in the filename, or None."""
    low = filename.lower()
    for view in EXPECTED_VIEWS:
        if view in low:
            return view
    return None


def iter_image_files(source_dir: Path):
    """Yield every image file under source_dir (recursively)."""
    for path in source_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def load_excel_ids(excel_path: Path) -> set[str]:
    """Read the subject IDs from the measurements Excel."""
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel not found at {excel_path}")
    df = pd.read_excel(excel_path, sheet_name=EXCEL_SHEET, header=EXCEL_HEADER_ROW)
    if EXCEL_ID_COLUMN not in df.columns:
        raise KeyError(f"Column '{EXCEL_ID_COLUMN}' not found in sheet '{EXCEL_SHEET}'.")
    return set(df[EXCEL_ID_COLUMN].dropna().astype(str).str.strip())


# ============================================================
# MAIN
# ============================================================

def build_brasil_dataset(source_dir: Path, dry_run: bool = False) -> None:
    """Copy the images of the Excel subjects into per-subject folders."""

    if not source_dir.exists():
        raise FileNotFoundError(
            f"SOURCE_DIR does not exist: {source_dir}\n"
            f"Edit SOURCE_DIR at the top of the script or pass the path as an argument."
        )

    print(f"Source : {source_dir}")
    print(f"Target : {BRASIL_DIR}")
    print(f"Mode   : {'DRY-RUN (nothing will be copied)' if dry_run else 'COPY'}")
    print("-" * 60)

    # ---- 1. subject IDs that we accept (only those in the Excel) ----------
    excel_ids = load_excel_ids(source_dir / EXCEL_NAME)

    # ---- 2. group every image file by subject -----------------------------
    # subject_id -> {"city": str, "files": [(view, Path), ...]}
    subjects: dict[str, dict] = {}
    ignored_not_in_excel: set[str] = set()

    for img_path in iter_image_files(source_dir):
        subject_id = subject_id_from_filename(img_path.name)
        if subject_id is None:
            continue

        # Only keep subjects present in the Excel; ignore everything else.
        if subject_id not in excel_ids:
            ignored_not_in_excel.add(subject_id)
            continue

        view = infer_view(img_path.name)
        city = img_path.parent.name  # PortoAlegre / Maceio
        entry = subjects.setdefault(subject_id, {"city": city, "files": []})
        entry["files"].append((view, img_path))

    # ---- 3. copy images into per-subject folders --------------------------
    if not dry_run:
        BRASIL_DIR.mkdir(parents=True, exist_ok=True)
        BRASIL_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    n_copied = 0
    for subject_id, info in sorted(subjects.items()):
        dest_folder = BRASIL_DIR / subject_id
        if not dry_run:
            dest_folder.mkdir(parents=True, exist_ok=True)
        for _view, src in info["files"]:
            if not dry_run:
                shutil.copy2(src, dest_folder / src.name)
            n_copied += 1

    # ---- 4. copy the Excel into the features folder -----------------------
    excel_src = source_dir / EXCEL_NAME
    if not dry_run:
        shutil.copy2(excel_src, BRASIL_FEATURES_DIR / EXCEL_NAME)

    # ---- 5. console report (no files written) -----------------------------
    excel_no_imgs = sorted(excel_ids - set(subjects.keys()))
    incomplete = {
        sid: sorted(v for v in EXPECTED_VIEWS
                    if v not in {vw for vw, _ in info["files"] if vw})
        for sid, info in subjects.items()
        if any(v not in {vw for vw, _ in info["files"] if vw} for v in EXPECTED_VIEWS)
    }

    print(f"Subjects in Excel                 : {len(excel_ids)}")
    print(f"Excel subjects WITH images copied : {len(subjects)}")
    print(f"Image files copied                : {n_copied}")
    print()
    print(f"[!] Excel subjects WITHOUT images ({len(excel_no_imgs)}): "
          f"{', '.join(excel_no_imgs) if excel_no_imgs else 'none'}")
    print(f"[!] Image subjects ignored (not in Excel) ({len(ignored_not_in_excel)}): "
          f"{', '.join(sorted(ignored_not_in_excel)) if ignored_not_in_excel else 'none'}")
    print(f"[!] Copied subjects missing a view ({len(incomplete)}):")
    if incomplete:
        for sid, miss in sorted(incomplete.items()):
            print(f"      {sid}: missing [{', '.join(miss)}]")
    else:
        print("      none")

    print("-" * 60)
    if dry_run:
        print("DRY-RUN complete. Re-run without --dry-run to copy the files.")
    else:
        print("Done.")


def main():
    dry_run = "--dry-run" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    source_dir = Path(args[0]).expanduser() if args else SOURCE_DIR
    build_brasil_dataset(source_dir=source_dir, dry_run=dry_run)


if __name__ == "__main__":
    main()
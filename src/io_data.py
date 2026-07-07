"""
io_data.py

Module for reading subject IDs and discovering image files in the dataset.

This module does not process the image content yet.
It only reads the folder structure and creates ImageAsset objects with the
basic information needed by the rest of the preprocessing pipeline.

Main functions:
    - parse_subject_id()
    - is_image_file()
    - infer_view_from_filename()
    - has_markers_from_filename()
    - discover_subject_images()
"""

import re                                   # Used to search for numbers inside the subject ID
from collections import defaultdict         # Dictionary where each subject can store a list of images
from pathlib import Path                    # Used to work with file and folder paths
import pandas as pd

from src.config import IMAGE_EXTENSIONS, MARKER_KEYWORD
from src.data_models import ImageAsset


# ============================================================
# SUBJECT ID PARSING
# ============================================================

def parse_subject_id(subject_id: str) -> dict:
    """
    Purpose:
        Extract the site, cohort and subject number from a subject ID.

    Input:
        subject_id (str):
            Subject identifier.
            Example: "COSTM0362"

    Output:
        dict:
            Dictionary with:
                - site: dataset/site code
                - cohort: ST or CN
                - number: subject number

            Example:
                {
                    "site": "CO",
                    "cohort": "ST",
                    "number": "0362"
                }
    """

    site = subject_id[:2]                   # Takes the first two letters of the subject ID

    if "ST" in subject_id:
        cohort = "ST"                       # Turner Syndrome subject
    elif "CTC" in subject_id:
        cohort = "CN"                       # Control subject
    else:
        cohort = "unknown"                  # Used if the cohort cannot be identified

    numbers = re.findall(r"\d+", subject_id)
    # r"\d+" is a raw regular expression pattern that matches one or more consecutive digits

    number = numbers[-1] if numbers else "" # Keeps the last number group found in the subject ID

    return {
        "site": site,
        "cohort": cohort,
        "number": number,
    }


# ============================================================
# FILE CHECKING
# ============================================================

def is_image_file(path: Path) -> bool:
    """
    Purpose:
        Check if a file has a valid image extension.

    Input:
        path (Path):
            Path to the file.

    Output:
        bool:
            True if the file extension is accepted.
            False otherwise.
    """

    return path.suffix.lower() in IMAGE_EXTENSIONS
    # Example: accepts .png, .jpg and .jpeg if they are defined in config.py


# ============================================================
# VIEW DETECTION
# ============================================================

def infer_view_from_filename(filename: str) -> str | None:
    """
    Purpose:
        Infer the anatomical view from the image filename.

    Input:
        filename (str):
            Name of the image file.
            Example: "COCTC0377_front.png"

    Output:
        str | None:
            Returns one of:
                - "front"
                - "back"
                - "left"
                - "right"

            Returns None if no valid view is found.
    """

    filename = filename.lower()             # Converts the filename to lowercase to avoid case problems

    for view in ["front", "back", "left", "right"]:
        if view in filename:
            return view                     # Returns the first view found in the filename

    return None                             # Returned when no valid view is found


# ============================================================
# MARKER DETECTION FROM FILENAME
# ============================================================

def has_markers_from_filename(filename: str) -> bool:
    """
    Purpose:
        Check if the filename indicates that the image has anatomical markers.

    Input:
        filename (str):
            Name of the image file.
            Example: "ESCTC0484_front_markers_anon.png"

    Output:
        bool:
            True if the filename contains the marker keyword.
            False otherwise.
    """

    return MARKER_KEYWORD in filename.lower()
    # Example: returns True if the filename contains "markers"


# ============================================================
# DATASET DISCOVERY
# ============================================================

def discover_subject_images(site_dir: Path, site: str) -> dict[str, list[ImageAsset]]:
    """
    Purpose:
        Search all subject folders inside a dataset directory and return their images.

    Input:
        site_dir (Path):
            Path to the dataset folder.
            Example: data_files/data_files_colombia

        site (str):
            Site code.
            Example:
                - "CO" for Colombia
                - "ES" for Barcelona
                - "BR" for Brazil

    Output:
        dict[str, list[ImageAsset]]:
            Dictionary where:
                - the key is the subject ID
                - the value is a list of ImageAsset objects

            Example:
                {
                    "COCTC0377": [
                        ImageAsset(...),
                        ImageAsset(...)
                    ]
                }
    """

    assets_by_subject = defaultdict(list)   # Each subject ID will contain a list of image assets

    if not site_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {site_dir}")

    for subject_folder in site_dir.iterdir():

        if not subject_folder.is_dir():
            continue                        # Skips files that are not subject folders

        subject_id = subject_folder.name    # Uses the folder name as the subject ID

        for img_path in subject_folder.iterdir():

            if not img_path.is_file():
                continue                    # Skips folders or non-file elements

            if not is_image_file(img_path):
                continue                    # Skips files that are not valid images

            view = infer_view_from_filename(img_path.name)

            if view is None:
                print(f"Warning: view not found for image {img_path.name}")
                continue                    # Skips images where the view cannot be detected

            asset = ImageAsset(
                subject_id=subject_id,
                site=site,
                view=view,
                has_markers=has_markers_from_filename(img_path.name),
                path=img_path,
            )

            assets_by_subject[subject_id].append(asset)

    return dict(assets_by_subject)

# ============================================================
# TABLE LOADING
# ============================================================

def load_colombia_table(table_path: Path) -> pd.DataFrame:
    """
    Purpose:
        Read the Colombia measurements table.

    Input:
        table_path (Path):
            Path to the Colombia CSV file.

    Output:
        pd.DataFrame:
            Original Colombia table loaded as a dataframe.
    """

    return pd.read_csv(table_path, header=[0, 1])


def load_barcelona_tables(table_path: Path) -> dict[str, pd.DataFrame]:
    """
    Purpose:
        Read all sheets from the Barcelona Excel file.

    Input:
        table_path (Path):
            Path to the Barcelona XLSX file.

    Output:
        dict[str, pd.DataFrame]:
            Dictionary where each key is a sheet name and each value is a dataframe.
    """

    return pd.read_excel(table_path, sheet_name=None)
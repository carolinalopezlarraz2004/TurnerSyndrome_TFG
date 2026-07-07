"""
data_models.py

Basic data structures used across the preprocessing pipeline.

Each subject can have several images from different views. Barcelona can also
include images with and without anatomical markers. The ImageAsset dataclass
stores this information in a clean and consistent way.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ImageAsset:
    """
    Stores the basic information of one image.
    """

    subject_id: str      # Subject ID, e.g. COCTC0377 or ESSTM0022
    site: str            # Dataset/site: CO, ES or BR
    view: str            # Anatomical view: front, back, left or right
    has_markers: bool    # True if the image contains visible anatomical markers
    path: Path           # Path to the image file
"""
Central configuration for the attendance-sheet extraction pipeline.

Nothing here is a *value* read from a document — these are only layout,
tuning and I/O parameters. Every attendance sheet shares the same layout, so
the table geometry is described declaratively here and consumed by the table
detector and the parser. Adjust these to re-target a different template without
touching the code.
"""

from __future__ import annotations

import os
import re

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Where a per-document debug crop of every detected cell is written when
# DEBUG is enabled (useful when tuning the grid on a new scanner/template).
DEBUG_DIR = os.path.join(OUTPUT_DIR, "_debug")
DEBUG = os.environ.get("ATTENDANCE_DEBUG", "0") == "1"

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

# --------------------------------------------------------------------------- #
# OpenCV preprocessing
# --------------------------------------------------------------------------- #
# Longest edge the working image is scaled to. Big enough for OCR to resolve
# handwriting, small enough to keep morphology fast.
WORK_MAX_EDGE = 2200

# Deskew is only applied when the detected angle exceeds this (degrees), to
# avoid re-sampling (and softening) already-straight scans.
DESKEW_MIN_ANGLE = 0.3
DESKEW_MAX_ANGLE = 15.0

# CLAHE (adaptive contrast) parameters.
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

# --------------------------------------------------------------------------- #
# Table structure detection (morphology-based line extraction)
# --------------------------------------------------------------------------- #
# Kernel length used to isolate horizontal / vertical rules, expressed as a
# fraction of the image width / height. Larger => only long table lines survive.
H_LINE_SCALE = 0.030
V_LINE_SCALE = 0.030

# Minimum gap (px, on the working image) to treat two detected lines as
# distinct grid lines rather than the two edges of one thick rule.
LINE_MERGE_TOLERANCE = 12

# A detected column / row band narrower than this (px) is discarded as noise.
MIN_CELL_WIDTH = 18
MIN_CELL_HEIGHT = 12

# --------------------------------------------------------------------------- #
# Logical table template (same layout for every sheet — NOT data)
# --------------------------------------------------------------------------- #
# The body table has five logical data columns in this order.
TABLE_COLUMNS = [
    "date",
    "matin_entre",
    "matin_sortie",
    "soir_entre",
    "soir_sortie",
]

# Two-level Excel header: top group -> sub columns.
EXCEL_HEADER = {
    "DATE": None,
    "MATIN": ["ENTRE", "SORTIE"],
    "SOIR": ["ENTRE", "SORTIE"],
}

# Header (metadata) fields to read from the top block of the sheet. The key is
# the output field; the list is the set of label variants that may introduce it
# on the page (case-insensitive, accent-insensitive matching is used).
HEADER_FIELDS = {
    "animateur": ["animateur", "nom", "nom et prenom"],
    "magasin": ["magasin"],
    "societe": ["societe", "société"],
    "marque": ["marque"],
    "mois": ["mois"],
}

# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
# An hour cell is valid if it is OFF, empty, or matches an hour pattern. We
# accept the strict spec form plus common real-world variants (lower-case h,
# missing minutes, a trailing "H"), then normalise to HH'H'MM.
HOUR_STRICT = re.compile(r"^(OFF|[0-2]?[0-9]H[0-5][0-9])$", re.IGNORECASE)
HOUR_LOOSE = re.compile(r"^\s*([0-2]?\d)\s*[Hh]\s*([0-5]\d)?\s*$")
OFF_TOKENS = {"off", "0ff", "0f", "of"}

# Date cell like d/m/yyyy, dd/mm/yy, d-m-yy … normalised to dd/mm/yy.
DATE_RE = re.compile(r"^\s*(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\s*$")

# Expected number of day rows, by month number (1-12). Used to pad/trim the
# result to exactly the right number of rows. February is treated as 28/29 and
# resolved from the year when known.
DAYS_IN_MONTH = {
    1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
    7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31,
}

# Map French month names (as they may appear in the "Mois" field) to numbers.
FRENCH_MONTHS = {
    "janvier": 1, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6,
    "juillet": 7, "aout": 8, "septembre": 9, "octobre": 10,
    "novembre": 11, "decembre": 12,
}

# --------------------------------------------------------------------------- #
# OCR
# --------------------------------------------------------------------------- #
# PaddleOCR language pack and toggles.
OCR_LANG = os.environ.get("ATTENDANCE_OCR_LANG", "fr")
OCR_USE_ANGLE_CLS = True

# Below this recognition confidence a cell is re-OCR'd with heavier
# preprocessing before we accept (or blank) its value.
OCR_MIN_CONFIDENCE = 0.55

# Padding (px) added around each detected cell before it is cropped for OCR,
# so glyphs touching the grid lines are not clipped.
CELL_CROP_PADDING = 4

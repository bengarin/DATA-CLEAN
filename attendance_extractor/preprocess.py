"""
OpenCV preprocessing: resize, deskew, denoise and contrast-enhance a scanned
attendance sheet before table detection and OCR.

Two products are returned for every image:
  * a clean *binary* image tuned for morphology / line detection, and
  * a contrast-enhanced *grayscale* image tuned for OCR legibility.

An extra `enhance_for_retry()` applies heavier cleaning for the low-confidence
re-OCR path.
"""

from __future__ import annotations

import cv2
import numpy as np

from config import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID,
    DESKEW_MAX_ANGLE,
    DESKEW_MIN_ANGLE,
    WORK_MAX_EDGE,
)
from utils import get_logger

logger = get_logger()


def load_image(path: str) -> np.ndarray:
    """Read an image from disk as BGR, raising on failure."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def resize_to_work(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale so the longest edge is WORK_MAX_EDGE. Returns (image, scale)."""
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= WORK_MAX_EDGE:
        return img.copy(), 1.0
    scale = WORK_MAX_EDGE / longest
    resized = cv2.resize(
        img, (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _estimate_skew_angle(gray: np.ndarray) -> float:
    """Estimate the page skew (degrees) from the dominant near-horizontal lines.

    Uses the Hough transform on edges and takes the median angle of lines that
    are close to horizontal — table rules make this robust for forms.
    """
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=200,
        minLineLength=gray.shape[1] // 4, maxLineGap=20,
    )
    if lines is None:
        return 0.0

    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 45:  # near-horizontal rules only
            angles.append(angle)
    if not angles:
        return 0.0
    return float(np.median(angles))


def deskew(gray: np.ndarray, color: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Rotate both images to level the page if skew is within a sane range."""
    angle = _estimate_skew_angle(gray)
    if abs(angle) < DESKEW_MIN_ANGLE or abs(angle) > DESKEW_MAX_ANGLE:
        return gray, color, 0.0

    h, w = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    flags = cv2.INTER_CUBIC
    border = cv2.BORDER_REPLICATE
    gray_r = cv2.warpAffine(gray, matrix, (w, h), flags=flags, borderMode=border)
    color_r = cv2.warpAffine(color, matrix, (w, h), flags=flags, borderMode=border)
    logger.info("Deskewed by %.2f degrees", angle)
    return gray_r, color_r, angle


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """Adaptive histogram equalisation (CLAHE) for even, legible strokes."""
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    return clahe.apply(gray)


def binarize(gray: np.ndarray) -> np.ndarray:
    """Adaptive threshold producing black ink on white — good for line finding."""
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 15,
    )
    return binary


def preprocess(path: str) -> dict:
    """Full preprocessing for one sheet.

    Returns a dict with:
      color   : deskewed BGR working image
      gray    : contrast-enhanced grayscale (for OCR)
      binary  : black-on-white adaptive threshold (for table detection)
      scale   : working->original scale factor (to map coords back if needed)
      angle   : applied deskew angle
    """
    original = load_image(path)
    color, scale = resize_to_work(original)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    gray, color, angle = deskew(gray, color)
    gray_eq = enhance_contrast(gray)
    binary = binarize(gray_eq)

    return {
        "color": color,
        "gray": gray_eq,
        "binary": binary,
        "scale": scale,
        "angle": angle,
    }


def enhance_for_retry(cell_gray: np.ndarray) -> np.ndarray:
    """Heavier cleaning for a single low-confidence cell before re-OCR.

    Upscales, sharpens and Otsu-binarises the crop — this often recovers faint
    handwriting that the first pass under-reads.
    """
    if cell_gray.size == 0:
        return cell_gray
    scaled = cv2.resize(cell_gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(scaled, (0, 0), 3)
    sharp = cv2.addWeighted(scaled, 1.5, blur, -0.5, 0)
    _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu

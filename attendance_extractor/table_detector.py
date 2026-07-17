"""
Table structure detection with OpenCV.

The strategy is deliberately geometry-first (not OCR-first): because every
sheet shares the same layout, we recover the physical grid from the printed
rules, then hand well-defined cell rectangles to the OCR stage. This is far more
robust than trying to reconstruct a table from scattered OCR word boxes.

Pipeline:
  1. Isolate long horizontal and vertical lines via morphology.
  2. Collapse each family of lines into distinct grid coordinates.
  3. Intersect them into a matrix of cells (rows x cols).
  4. Pick the largest, densest rectangular block as the attendance body.
"""

from __future__ import annotations

import cv2
import numpy as np

from config import (
    H_LINE_SCALE,
    LINE_MERGE_TOLERANCE,
    MIN_CELL_HEIGHT,
    MIN_CELL_WIDTH,
    V_LINE_SCALE,
)
from utils import get_logger

logger = get_logger()


class Cell:
    """A single grid cell with its pixel rectangle and (row, col) index."""

    __slots__ = ("row", "col", "x0", "y0", "x1", "y1")

    def __init__(self, row: int, col: int, x0: int, y0: int, x1: int, y1: int):
        self.row, self.col = row, col
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def crop(self, image: np.ndarray, pad: int = 0) -> np.ndarray:
        h, w = image.shape[:2]
        x0 = max(0, self.x0 + pad)
        y0 = max(0, self.y0 + pad)
        x1 = min(w, self.x1 - pad)
        y1 = min(h, self.y1 - pad)
        if x1 <= x0 or y1 <= y0:
            return image[self.y0:self.y1, self.x0:self.x1]
        return image[y0:y1, x0:x1]


def _extract_lines(binary: np.ndarray, scale: float, axis: str) -> np.ndarray:
    """Return a mask containing only long lines along the given axis."""
    inverted = cv2.bitwise_not(binary)  # lines become white on black
    h, w = binary.shape[:2]
    if axis == "h":
        length = max(10, int(w * scale))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
    else:
        length = max(10, int(h * scale))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
    eroded = cv2.erode(inverted, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    return dilated


def _line_positions(mask: np.ndarray, axis: str) -> list[int]:
    """Collapse a line mask into a sorted list of distinct grid coordinates."""
    projection = mask.sum(axis=1 if axis == "h" else 0)
    threshold = projection.max() * 0.3 if projection.max() > 0 else 0
    hits = np.where(projection > threshold)[0]
    if len(hits) == 0:
        return []

    # Group consecutive/near indices and take the centre of each group.
    positions: list[int] = []
    group_start = hits[0]
    prev = hits[0]
    for idx in hits[1:]:
        if idx - prev > LINE_MERGE_TOLERANCE:
            positions.append(int((group_start + prev) // 2))
            group_start = idx
        prev = idx
    positions.append(int((group_start + prev) // 2))
    return positions


def _build_cells(rows: list[int], cols: list[int]) -> list[list[Cell]]:
    """Turn row/column grid lines into a matrix of cells."""
    grid: list[list[Cell]] = []
    for r in range(len(rows) - 1):
        y0, y1 = rows[r], rows[r + 1]
        if y1 - y0 < MIN_CELL_HEIGHT:
            continue
        row_cells: list[Cell] = []
        for c in range(len(cols) - 1):
            x0, x1 = cols[c], cols[c + 1]
            if x1 - x0 < MIN_CELL_WIDTH:
                continue
            row_cells.append(Cell(len(grid), len(row_cells), x0, y0, x1, y1))
        if row_cells:
            grid.append(row_cells)
    return grid


def detect_grid(binary: np.ndarray) -> dict:
    """Detect the table grid from a binary image.

    Returns a dict:
      rows    : list of horizontal grid-line y-coordinates
      cols    : list of vertical grid-line x-coordinates
      cells   : matrix (list of rows) of Cell objects
      h_mask  : horizontal line mask (debug)
      v_mask  : vertical line mask (debug)
    """
    h_mask = _extract_lines(binary, H_LINE_SCALE, "h")
    v_mask = _extract_lines(binary, V_LINE_SCALE, "v")

    rows = _line_positions(h_mask, "h")
    cols = _line_positions(v_mask, "v")

    logger.info("Detected %d horizontal and %d vertical grid lines",
                len(rows), len(cols))

    cells = _build_cells(rows, cols) if rows and cols else []
    return {
        "rows": rows,
        "cols": cols,
        "cells": cells,
        "h_mask": h_mask,
        "v_mask": v_mask,
    }


def draw_grid(color: np.ndarray, grid: dict) -> np.ndarray:
    """Render detected cells over the colour image for debugging."""
    canvas = color.copy()
    for row in grid["cells"]:
        for cell in row:
            cv2.rectangle(canvas, (cell.x0, cell.y0), (cell.x1, cell.y1),
                          (0, 0, 255), 1)
    return canvas

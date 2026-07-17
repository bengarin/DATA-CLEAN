"""
Batch entry point for the attendance-sheet extraction pipeline.

Drop scanned sheets into ``input/`` and run::

    python main.py

Every supported image is preprocessed, its table detected, read with
PaddleOCR, validated and exported to ``output/`` as .json, .csv and .xlsx.
The run is unattended and continues past a failing document, reporting a
summary at the end.
"""

from __future__ import annotations

import os
import sys
import traceback

import cv2

from config import DEBUG, DEBUG_DIR, INPUT_DIR, OUTPUT_DIR
from exporter import export_all
from ocr import get_engine
from parser import parse_document
from preprocess import preprocess
from table_detector import detect_grid, draw_grid
from utils import ensure_dir, get_logger, list_images

logger = get_logger()


def process_one(path: str, engine) -> dict:
    """Run the full pipeline for a single image and return the record."""
    stem = os.path.splitext(os.path.basename(path))[0]
    logger.info("Processing %s", os.path.basename(path))

    images = preprocess(path)
    grid = detect_grid(images["binary"])

    if DEBUG:
        ensure_dir(DEBUG_DIR)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{stem}_grid.png"),
                    draw_grid(images["color"], grid))
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{stem}_binary.png"), images["binary"])

    record = parse_document(grid, images, engine)
    export_all(record, OUTPUT_DIR, stem)
    return record


def main(argv: list[str]) -> int:
    ensure_dir(INPUT_DIR)
    ensure_dir(OUTPUT_DIR)

    images = list_images(INPUT_DIR)
    if not images:
        logger.warning("No images found in %s. Add scans and re-run.", INPUT_DIR)
        return 0

    logger.info("Found %d document(s) to process.", len(images))
    engine = get_engine()  # heavy model loads once for the whole batch

    ok, failed = 0, 0
    for path in images:
        try:
            process_one(path, engine)
            ok += 1
        except Exception as exc:  # noqa: BLE001 - keep the batch alive
            failed += 1
            logger.error("Failed on %s: %s", os.path.basename(path), exc)
            if DEBUG:
                traceback.print_exc()

    logger.info("Done. %d succeeded, %d failed, %d total.", ok, failed, len(images))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

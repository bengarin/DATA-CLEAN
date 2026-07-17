"""
One-shot OCR diagnostic.

Runs PaddleOCR directly on an image and prints exactly what it returns, so we
can see whether the engine reads any text at all and in what structure.

Usage:
    python diagnose_ocr.py path\\to\\sheet.jpg
    # or, with no argument, it uses the first image in input/
"""

from __future__ import annotations

import os
import sys

import cv2

from config import INPUT_DIR, OCR_LANG
from utils import list_images


def main() -> int:
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        imgs = list_images(INPUT_DIR)
        if not imgs:
            print("No image given and input/ is empty.")
            return 1
        path = imgs[0]

    print(f"Image: {path}")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print("Could not read the image.")
        return 1
    print(f"Shape: {img.shape}, dtype: {img.dtype}")

    from paddleocr import PaddleOCR

    ocr = None
    for kwargs in (
        {"lang": OCR_LANG, "use_textline_orientation": True,
         "use_doc_orientation_classify": False, "use_doc_unwarping": False},
        {"lang": OCR_LANG},
    ):
        try:
            ocr = PaddleOCR(**kwargs)
            print(f"PaddleOCR initialised with: {list(kwargs)}")
            break
        except Exception as exc:  # noqa: BLE001
            print(f"init failed with {list(kwargs)}: {exc}")
    if ocr is None:
        return 1

    result = ocr.predict(img)
    print(f"\npredict() returned: type={type(result).__name__}, len={len(result) if hasattr(result,'__len__') else '?'}")

    if not result:
        print(">>> Empty result — the detector found no text.")
        return 0

    res0 = result[0]
    print(f"result[0] type: {type(res0).__name__}, is dict: {isinstance(res0, dict)}")
    if isinstance(res0, dict):
        print("keys:", list(res0.keys()))
        texts = res0.get("rec_texts")
        scores = res0.get("rec_scores")
        print(f"\nrec_texts ({0 if texts is None else len(texts)}):")
        if texts is not None:
            for i, t in enumerate(texts):
                s = float(scores[i]) if scores is not None and i < len(scores) else 0.0
                print(f"  [{i:3d}] {s:.2f}  {t!r}")
    else:
        print("repr:", repr(res0)[:500])
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
PaddleOCR wrapper.

Exposes two levels of reading:
  * `read_full(image)`    -> every text line with box + confidence, used to
                             locate and read the header block.
  * `read_cell(image)`    -> the single best text + confidence for one cell
                             crop, with an automatic heavy-preprocessing retry
                             when confidence is low.

The PaddleOCR return format has changed across major versions; `_parse_result`
normalises the common shapes so the rest of the app sees one simple structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from config import OCR_LANG, OCR_MIN_CONFIDENCE, OCR_USE_ANGLE_CLS
from preprocess import enhance_for_retry
from utils import clean_text, get_logger

logger = get_logger()


@dataclass
class OcrLine:
    text: str
    confidence: float
    box: list  # 4 (x, y) points


class OcrEngine:
    """Lazy singleton-ish wrapper around a PaddleOCR instance."""

    def __init__(self, lang: str = OCR_LANG):
        self._lang = lang
        self._ocr = None

    def _engine(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR  # imported lazily (heavy)

            logger.info("Initialising PaddleOCR (lang=%s)…", self._lang)
            # Constructor kwargs differ across major versions; try the newest
            # names first (3.x), then 2.x, then a bare lang-only fallback. On
            # 3.x we also disable the full-page document orientation/unwarping
            # stages: they are meant for whole scans, add failure surface, and
            # would distort the small single-cell crops we feed for reading.
            for kwargs in (
                {"lang": self._lang, "use_textline_orientation": OCR_USE_ANGLE_CLS,
                 "use_doc_orientation_classify": False, "use_doc_unwarping": False},
                {"lang": self._lang, "use_textline_orientation": OCR_USE_ANGLE_CLS},
                {"lang": self._lang, "use_angle_cls": OCR_USE_ANGLE_CLS, "show_log": False},
                {"lang": self._lang},
            ):
                try:
                    self._ocr = PaddleOCR(**kwargs)
                    break
                except (TypeError, ValueError):
                    continue
            if self._ocr is None:
                raise RuntimeError("Could not initialise PaddleOCR with any known signature")
        return self._ocr

    @staticmethod
    def _to_bgr(image: np.ndarray) -> np.ndarray:
        """PaddleOCR expects a 3-channel image; grayscale crops are 2-D.

        Passing a 2-D array makes Paddle read `img.shape[2]` internally and
        raise 'tuple index out of range', so promote grayscale to BGR here.
        """
        if image is None:
            return image
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

    # ------------------------------------------------------------------ #
    # Result normalisation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _first_present(data: dict, keys: list[str]):
        """Return the first key's value that exists (avoids `or` on arrays)."""
        for k in keys:
            if k in data and data[k] is not None:
                return data[k]
        return []

    @classmethod
    def _parse_result(cls, result) -> list[OcrLine]:
        """Normalise PaddleOCR output (2.x list form and 3.x dict form)."""
        lines: list[OcrLine] = []
        if result is None:
            return lines
        # Normalise the container: a single OCRResult (dict) -> [result];
        # a generator/iterator -> list; a list/tuple stays as-is.
        if isinstance(result, dict):
            result = [result]
        elif not isinstance(result, (list, tuple)):
            try:
                result = list(result)
            except TypeError:
                return lines
        if len(result) == 0:
            return lines

        first = result[0]

        # ---- 3.x predict(): list[OCRResult] (dict subclass) with arrays ----
        if isinstance(first, dict):
            data = first
            texts = cls._first_present(data, ["rec_texts", "rec_text"])
            scores = cls._first_present(data, ["rec_scores", "rec_score"])
            boxes = cls._first_present(data, ["rec_polys", "dt_polys", "boxes"])
            n_scores, n_boxes = len(scores), len(boxes)
            for i, text in enumerate(texts):
                conf = float(scores[i]) if i < n_scores else 0.0
                if i < n_boxes:
                    box = boxes[i].tolist() if hasattr(boxes[i], "tolist") else boxes[i]
                else:
                    box = []
                lines.append(OcrLine(clean_text(str(text)), conf, box))
            return lines

        # ---- 2.x ocr(): [[ [box, (text, conf)], ... ]] ----
        page = first if isinstance(first, list) else result
        for item in page:
            try:
                box, (text, conf) = item
                lines.append(OcrLine(clean_text(str(text)), float(conf), box))
            except (ValueError, TypeError):
                continue
        return lines

    def _run(self, image: np.ndarray) -> list[OcrLine]:
        engine = self._engine()
        img = self._to_bgr(image)
        # Prefer the modern predict() API; fall back to ocr().
        if hasattr(engine, "predict"):
            try:
                return self._parse_result(engine.predict(img))
            except TypeError:
                pass  # signature mismatch -> try the ocr() fallback
            except Exception as exc:  # noqa: BLE001
                logger.warning("PaddleOCR predict() failed: %s", exc)
                return []
        for call in (lambda: engine.ocr(img, cls=OCR_USE_ANGLE_CLS),
                     lambda: engine.ocr(img)):
            try:
                return self._parse_result(call())
            except TypeError:
                continue
            except Exception as exc:  # noqa: BLE001 - a bad crop must not kill the batch
                logger.warning("PaddleOCR ocr() failed: %s", exc)
                return []
        return []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def read_full(self, image: np.ndarray) -> list[OcrLine]:
        """Read every text line in an image (for the header block)."""
        return self._run(image)

    def read_cell(self, cell_gray: np.ndarray) -> tuple[str, float]:
        """Read a single cell, retrying with heavy preprocessing if unsure.

        Returns (text, confidence). Empty text for a blank/unreadable cell.
        """
        if cell_gray is None or cell_gray.size == 0:
            return "", 0.0

        best_text, best_conf = self._best_line(self._run(cell_gray))

        if best_conf < OCR_MIN_CONFIDENCE:
            retried = enhance_for_retry(cell_gray)
            text2, conf2 = self._best_line(self._run(retried))
            if conf2 > best_conf:
                best_text, best_conf = text2, conf2

        return best_text, best_conf

    @staticmethod
    def _best_line(lines: list[OcrLine]) -> tuple[str, float]:
        """Merge a cell's lines into one string; report the min confidence."""
        real = [ln for ln in lines if ln.text]
        if not real:
            return "", 0.0
        # A cell rarely holds more than one token; join defensively, and treat
        # the weakest line as the cell confidence (conservative).
        text = " ".join(ln.text for ln in real)
        conf = min(ln.confidence for ln in real)
        return clean_text(text), conf


# Module-level shared engine so the heavy model loads only once per run.
_engine: Optional[OcrEngine] = None


def get_engine() -> OcrEngine:
    global _engine
    if _engine is None:
        _engine = OcrEngine()
    return _engine

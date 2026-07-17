"""
Small shared helpers: logging, filesystem, text normalisation and validation.

Keeping these here avoids circular imports between the pipeline stages and
gives every module one place to reach for common primitives.
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from datetime import date
from typing import Optional

from config import (
    DATE_RE,
    DAYS_IN_MONTH,
    FRENCH_MONTHS,
    HOUR_LOOSE,
    HOUR_STRICT,
    OFF_TOKENS,
    SUPPORTED_EXTENSIONS,
)


def get_logger(name: str = "attendance") -> logging.Logger:
    """Return a module logger with a single stream handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_images(folder: str) -> list[str]:
    """Return sorted absolute paths of every supported image in *folder*."""
    if not os.path.isdir(folder):
        return []
    files = [
        os.path.join(folder, name)
        for name in sorted(os.listdir(folder))
        if name.lower().endswith(SUPPORTED_EXTENSIONS)
    ]
    return files


# --------------------------------------------------------------------------- #
# Text normalisation
# --------------------------------------------------------------------------- #
def strip_accents(text: str) -> str:
    """Remove diacritics so 'Société' matches 'societe'."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize_label(text: str) -> str:
    """Lower-case, de-accent and squeeze whitespace for label comparison."""
    return re.sub(r"\s+", " ", strip_accents(text).lower()).strip(" :\t")


def clean_text(text: str) -> str:
    """Collapse internal whitespace and trim; keep original casing."""
    return re.sub(r"\s+", " ", (text or "").strip())


# --------------------------------------------------------------------------- #
# Cell value validation / normalisation
# --------------------------------------------------------------------------- #
def normalize_hour(raw: str) -> str:
    """Validate and normalise an hour/OFF/empty cell.

    Returns:
        - ""     for an empty / unreadable cell
        - "OFF"  for an off day
        - "HHhMM" (e.g. "08H00") for a valid hour
        - the cleaned raw string if it looks like an hour but cannot be parsed
          strictly (kept so nothing is silently dropped, flagged by the parser)
    """
    text = clean_text(raw)
    if not text:
        return ""

    compact = strip_accents(text).lower().replace(" ", "")
    if compact in OFF_TOKENS or compact == "off":
        return "OFF"

    # Correct the digit/letter confusions handwriting OCR makes (O->0, l->1,
    # S->5 …) around the hour, then re-check. This recovers "l4H00"->14H00,
    # "12HOO"->12H00, "2OH"->20H00 without hard-coding any specific value.
    fixed = _fix_hour_digits(text)
    for candidate in (text, fixed):
        if HOUR_STRICT.match(candidate):
            h, _, m = candidate.upper().partition("H")
            return f"{int(h):02d}H{m if m else '00'}"
        loose = HOUR_LOOSE.match(candidate)
        if loose:
            h = int(loose.group(1))
            m = loose.group(2) or "00"
            if 0 <= h <= 23:
                return f"{h:02d}H{m}"

    # Not OFF and not a valid hour: treat as noise / empty. This also drops any
    # header text ("ENTRE", "SORTIE") that a row-detection overshoot may pass in.
    return ""


# Letter -> digit map for the common OCR confusions on handwritten hours.
_HOUR_DIGIT_FIX = str.maketrans({
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "l": "1", "I": "1", "|": "1", "i": "1",
    "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2", "g": "9",
})


def _fix_hour_digits(token: str) -> str:
    """Map digit-look-alike letters to digits, preserving the H separator."""
    # Keep the hour separator intact, fix everything around it.
    parts = re.split(r"([Hh])", token, maxsplit=1)
    return "".join(
        p if p in ("H", "h") else p.translate(_HOUR_DIGIT_FIX) for p in parts
    )


def normalize_date(raw: str, default_month: Optional[int] = None,
                   default_year: Optional[int] = None) -> str:
    """Validate a date cell and normalise to dd/mm/yy. Empty if invalid."""
    text = clean_text(raw)
    m = DATE_RE.match(text)
    if not m:
        return ""
    day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if year < 100:
        year += 2000
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return ""
    return f"{day:02d}/{month:02d}/{str(year)[-2:]}"


def month_from_text(text: str) -> Optional[int]:
    """Resolve a month number from a French month name or a numeric string."""
    norm = normalize_label(text)
    for name, num in FRENCH_MONTHS.items():
        if name in norm:
            return num
    digits = re.findall(r"\d{1,2}", norm)
    if digits:
        num = int(digits[0])
        if 1 <= num <= 12:
            return num
    return None


def days_in_month(month: int, year: Optional[int] = None) -> int:
    """Number of days in *month*, handling leap Februaries when year is known."""
    if month == 2 and year is not None:
        leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        return 29 if leap else 28
    return DAYS_IN_MONTH.get(month, 31)


def build_date(day: int, month: int, year: int) -> str:
    """Compose a normalised dd/mm/yy string for a known day/month/year."""
    try:
        d = date(year, month, day)
    except ValueError:
        return ""
    return d.strftime("%d/%m/%y")

"""
Turn a detected grid + OCR into the structured attendance record.

Responsibilities:
  * read the header block (metadata fields) from full-page OCR,
  * locate the attendance body inside the detected grid,
  * read each body cell independently and map it to a logical column,
  * validate/normalise dates and hours,
  * guarantee exactly `days_in_month` rows, filling gaps with empty cells.

Nothing about the *content* is hard-coded — only the column order and the
label vocabulary, both of which live in config.py.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Optional

import numpy as np

from config import (
    HEADER_FIELDS,
    TABLE_COLUMNS,
)
from ocr import OcrEngine, OcrLine
from table_detector import Cell
from utils import (
    build_date,
    clean_text,
    days_in_month,
    get_logger,
    month_from_text,
    normalize_hour,
    normalize_label,
)

logger = get_logger()

HOUR_COLUMNS = ["matin_entre", "matin_sortie", "soir_entre", "soir_sortie"]

# Words that identify a printed header/sub-header row so it is not mistaken for
# a data row when the grid detector's body overshoots into the header band.
HEADER_KEYWORDS = {"date", "entre", "sortie", "matin", "soir"}


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
def extract_header(lines: list[OcrLine], body_top_y: Optional[int]) -> dict:
    """Match header labels to the value box on their right.

    Only lines above the table body are considered. For every configured field
    we find the label line, then pick the nearest line to its right sharing the
    same row band as the value.
    """
    header = {key: "" for key in HEADER_FIELDS}

    candidates = [
        ln for ln in lines
        if ln.box and (body_top_y is None or _box_cy(ln.box) < body_top_y)
    ]

    for field, variants in HEADER_FIELDS.items():
        label_line = _find_label(candidates, variants)
        if label_line is None:
            continue
        value = _value_right_of(candidates, label_line)
        header[field] = clean_text(value)

    return header


def _box_cy(box) -> float:
    return float(np.mean([p[1] for p in box]))


def _box_cx(box) -> float:
    return float(np.mean([p[0] for p in box]))


def _find_label(lines: list[OcrLine], variants: list[str]) -> Optional[OcrLine]:
    """Find the line whose text best matches any label variant."""
    for ln in lines:
        norm = normalize_label(ln.text)
        for variant in variants:
            if norm == variant or norm.startswith(variant + " ") or norm == variant + " :":
                return ln
    # looser containment pass
    for ln in lines:
        norm = normalize_label(ln.text)
        if any(variant in norm for variant in variants):
            return ln
    return None


def _value_right_of(lines: list[OcrLine], label: OcrLine) -> str:
    """Return the text of the closest line to the right of *label*'s row."""
    ly = _box_cy(label.box)
    lx_right = max(p[0] for p in label.box)
    row_tol = max(12.0, (max(p[1] for p in label.box) - min(p[1] for p in label.box)))

    best: Optional[OcrLine] = None
    best_dx = None
    for ln in lines:
        if ln is label or not ln.text:
            continue
        if abs(_box_cy(ln.box) - ly) > row_tol:
            continue
        dx = _box_cx(ln.box) - lx_right
        if dx <= 0:
            continue
        if best_dx is None or dx < best_dx:
            best, best_dx = ln, dx
    return best.text if best else ""


# --------------------------------------------------------------------------- #
# Body location
# --------------------------------------------------------------------------- #
def find_body(cells: list[list[Cell]]) -> list[list[Cell]]:
    """Select the block of rows that form the 5-column attendance body.

    The body rows share the same (modal) column count; header/metadata rows
    above have a different structure and are dropped.
    """
    if not cells:
        return []

    counts = Counter(len(row) for row in cells)
    target = len(TABLE_COLUMNS)
    # Prefer rows with exactly the expected column count; otherwise the mode.
    modal = target if counts.get(target) else counts.most_common(1)[0][0]

    body = [row for row in cells if len(row) == modal]
    logger.info("Body has %d rows of %d columns (modal)", len(body), modal)
    return body


def _assign_columns(row: list[Cell]) -> dict[str, Cell]:
    """Map the physical cells of a row to logical columns, left to right."""
    row_sorted = sorted(row, key=lambda c: c.x0)
    mapping: dict[str, Cell] = {}
    if len(row_sorted) == len(TABLE_COLUMNS):
        for name, cell in zip(TABLE_COLUMNS, row_sorted):
            mapping[name] = cell
    else:
        # Merged/extra cells: distribute by relative x fraction of the row span.
        x_start = row_sorted[0].x0
        x_end = row_sorted[-1].x1
        span = max(1, x_end - x_start)
        for cell in row_sorted:
            frac = (cell.x0 - x_start) / span
            idx = min(len(TABLE_COLUMNS) - 1, int(frac * len(TABLE_COLUMNS)))
            mapping.setdefault(TABLE_COLUMNS[idx], cell)
    return mapping


# --------------------------------------------------------------------------- #
# Body reading
# --------------------------------------------------------------------------- #
def _line_center(box) -> tuple[float, float]:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def read_body_from_lines(body: list[list[Cell]], lines: list[OcrLine]) -> list[dict]:
    """Assign each full-page OCR line to the body cell that contains it.

    Reading the whole page once and mapping text boxes onto the detected grid
    is far more reliable than OCR-ing each tiny single-value crop — PaddleOCR's
    detector often finds nothing in an isolated cell-sized image.
    """
    centers = [(_line_center(ln.box), ln.text) for ln in lines if ln.box and ln.text]

    rows: list[dict] = []
    for row in body:
        mapping = _assign_columns(row)
        record = {"date": "", **{c: "" for c in HOUR_COLUMNS}}
        for col, cell in mapping.items():
            texts = [
                text for (cx, cy), text in centers
                if cell.x0 <= cx < cell.x1 and cell.y0 <= cy < cell.y1
            ]
            if texts:
                record[col] = clean_text(" ".join(texts))
        rows.append(record)
    return rows


# --------------------------------------------------------------------------- #
# Validation + canonicalisation
# --------------------------------------------------------------------------- #
def _is_header_row(raw: dict) -> bool:
    """True if a raw row is actually the printed table header, not data."""
    joined = normalize_label(" ".join(
        str(raw.get(c, "")) for c in ["date", *HOUR_COLUMNS]
    ))
    return any(kw in joined.split() for kw in HEADER_KEYWORDS)


def _looks_like_date(text: str) -> bool:
    """True if a cell matches a date pattern in either d/m or m/d order."""
    from config import DATE_RE
    m = DATE_RE.match(clean_text(text))
    if not m:
        return False
    a, b = int(m.group(1)), int(m.group(2))
    return (1 <= a <= 31 and 1 <= b <= 12) or (1 <= b <= 31 and 1 <= a <= 12)


def _looks_like_day(text: str) -> bool:
    """True if the DATE cell anchors a real day: a full date OR a bare day
    number 1..31 (some sheets print only "01", "02", … in the date column)."""
    import re
    t = clean_text(text)
    if _looks_like_date(t):
        return True
    digits = re.sub(r"\D", "", t)
    return bool(digits) and len(digits) <= 2 and 1 <= int(digits) <= 31


def _infer_month_year(raw_rows: list[dict]) -> tuple[Optional[int], Optional[int]]:
    """Infer month/year from the printed date cells, resolving M/D vs D/M.

    Across a month's rows the month digit is constant while the day varies, so
    whichever position is the most constant valid month (1..12) is the month.
    """
    from collections import Counter
    from config import DATE_RE

    firsts, seconds, years = [], [], []
    for r in raw_rows:
        m = DATE_RE.match(clean_text(r.get("date", "")))
        if not m:
            continue
        firsts.append(int(m.group(1)))
        seconds.append(int(m.group(2)))
        y = int(m.group(3))
        years.append(y + 2000 if y < 100 else y)

    if not firsts:
        return None, None

    def dominant(vals):
        val, n = Counter(vals).most_common(1)[0]
        return val, n / len(vals)

    f_val, f_ratio = dominant(firsts)
    s_val, s_ratio = dominant(seconds)
    if 1 <= f_val <= 12 and f_ratio >= s_ratio:
        month = f_val
    elif 1 <= s_val <= 12:
        month = s_val
    else:
        month = f_val if 1 <= f_val <= 12 else (s_val if 1 <= s_val <= 12 else None)

    year = Counter(years).most_common(1)[0][0] if years else None
    return month, year


def canonicalize(raw_rows: list[dict], month: Optional[int],
                 year: Optional[int]) -> list[dict]:
    """Validate values and force exactly days_in_month ordered rows.

    The sheet lists the days of the month in order, so day assignment is driven
    by **row position** (row 1 -> day 1). This is robust to the M/D vs D/M
    ambiguity of the printed dates and to a missing/unreadable date cell. The
    output date is composed canonically from the day, month and year, so it is
    always a valid dd/mm/yy and matches the sheet's own sequence.
    """
    # Prefer the month/year printed in the date cells over the current date.
    inferred_month, inferred_year = _infer_month_year(raw_rows)
    month = month or inferred_month or datetime.now().month
    year = year or inferred_year or datetime.now().year
    n_days = days_in_month(month, year)

    # Keep only real data rows: drop header rows and fully empty grid noise.
    data_rows: list[dict] = []
    for r in raw_rows:
        if _is_header_row(r):
            continue
        hours = {c: normalize_hour(r.get(c, "")) for c in HOUR_COLUMNS}
        # Keep a row if it has any punch/OFF value OR its date cell anchors a
        # real day (so an all-rest or empty-but-dated day stays in sequence).
        has_content = any(hours.values()) or _looks_like_day(r.get("date", ""))
        if not has_content:
            continue
        data_rows.append(hours)

    # Map the ordered data rows onto days 1..N (trim extras, pad missing).
    result: list[dict] = []
    for i in range(n_days):
        day = i + 1
        hours = data_rows[i] if i < len(data_rows) else {c: "" for c in HOUR_COLUMNS}
        result.append({"date": build_date(day, month, year), **hours})

    return result


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def parse_document(grid: dict, images: dict, engine: OcrEngine) -> dict:
    """Full parse: header + table -> the specified JSON structure."""
    body = find_body(grid["cells"])
    body_top_y = body[0][0].y0 if body else None

    # OCR runs on the deskewed *colour* image (PaddleOCR is trained on natural
    # images; the heavily processed binary/CLAHE hurts its detector). The grid
    # cells and this image share the same coordinate space.
    full_lines = engine.read_full(images["color"])
    logger.info("OCR found %d text lines on the page", len(full_lines))
    header = extract_header(full_lines, body_top_y)

    month = month_from_text(header.get("mois", "")) if header.get("mois") else None

    raw_rows = read_body_from_lines(body, full_lines)
    attendance = canonicalize(raw_rows, month=month, year=None)

    return {
        "animateur": header.get("animateur", ""),
        "magasin": header.get("magasin", ""),
        "societe": header.get("societe", ""),
        "marque": header.get("marque", ""),
        "mois": header.get("mois", ""),
        "attendance": attendance,
    }

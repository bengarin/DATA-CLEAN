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


def _table_header_top(lines: list[OcrLine]) -> Optional[float]:
    """Y above which lines are metadata: the top of the table's column-header
    band, located from the printed labels DATE/ENTRE/SORTIE/MATIN/SOIR."""
    ys = []
    for ln in lines:
        if ln.box and normalize_label(ln.text) in ("date", "entre", "sortie",
                                                    "matin", "matain", "soir"):
            ys.append(min(p[1] for p in ln.box))
    return min(ys) if ys else None


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
        # Skip separator/junk tokens (":", ".", a stray single character) so a
        # real value further right is chosen instead.
        cleaned = ln.text.strip(" :.-")
        if len(cleaned) < 2:
            continue
        if abs(_box_cy(ln.box) - ly) > row_tol:
            continue
        dx = _box_cx(ln.box) - lx_right
        if dx <= 0:
            continue
        if best_dx is None or dx < best_dx:
            best, best_dx = ln, dx
    return best.text.strip(" :").strip() if best else ""


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


def extract_table_from_lines(lines: list[OcrLine],
                             row_bounds: Optional[list[int]] = None) -> Optional[list[dict]]:
    """Reconstruct the table from OCR geometry, independent of vertical rules.

    Real photos make vertical grid-line detection unreliable (a missed line
    merges two columns). Instead we anchor:
      * columns  on the printed header labels DATE / ENTRE / SORTIE (which OCR
        reads well), giving five x-anchors, and
      * rows     on the horizontal grid lines (row_bounds) when available —
        each band between two rules is one day, so a handwritten value that
        drifts above/below its printed day number still lands in the right
        row. Falls back to using the DATE column's own lines as y-anchors.
    Every other value is snapped to its nearest column anchor and its row band.
    Returns None if the header labels can't be found (caller falls back).
    """
    geom = _locate_geometry(lines, row_bounds)
    if geom is None:
        return None
    return _assign_lines_to_rows(lines, geom)


class _TableGeometry:
    """Resolved table geometry: column boundaries and row bands (y0, y1)."""

    __slots__ = ("anchors", "col_bounds", "bands", "header_bottom", "positional")

    def __init__(self, anchors, col_bounds, bands, header_bottom, positional):
        self.anchors = anchors            # 5 column-centre x anchors
        self.col_bounds = col_bounds      # 6 x boundaries for the 5 columns
        self.bands = bands                # list of (y0, y1) row bands
        self.header_bottom = header_bottom
        self.positional = positional      # True when bands come from grid rules


def _locate_geometry(lines: list[OcrLine],
                     row_bounds: Optional[list[int]] = None) -> Optional[_TableGeometry]:
    date_label_x = None
    value_label_xs: list[float] = []
    header_bottom = 0.0
    for ln in lines:
        if not ln.box or not ln.text:
            continue
        norm = normalize_label(ln.text)
        cx, _ = _line_center(ln.box)
        bottom = max(p[1] for p in ln.box)
        if norm == "date":
            date_label_x = cx if date_label_x is None else min(date_label_x, cx)
            header_bottom = max(header_bottom, bottom)
        elif norm in ("entre", "sortie"):
            value_label_xs.append(cx)
            header_bottom = max(header_bottom, bottom)

    if date_label_x is None or len(value_label_xs) < 4:
        return None  # not enough anchors -> let the caller use the grid method

    value_label_xs.sort()
    # The four value columns (matin entre/sortie, soir entre/sortie).
    anchors = [date_label_x] + value_label_xs[:4]

    # Column boundaries = midpoints between anchors, edges extended half a gap.
    col_bounds = [anchors[0] - (anchors[1] - anchors[0]) / 2]
    for i in range(len(anchors) - 1):
        col_bounds.append((anchors[i] + anchors[i + 1]) / 2)
    col_bounds.append(anchors[-1] + (anchors[-1] - anchors[-2]) / 2)

    # Row bands. Preferred: the horizontal grid rules (each band between two
    # rules is exactly one day, immune to handwriting drift). Fallback: the y
    # of each line detected in the DATE column.
    bands: list[tuple[float, float]] = []
    positional = False
    if row_bounds:
        rules = [y for y in sorted(row_bounds) if y > header_bottom - 5]
        band_candidates = [
            (float(rules[i]), float(rules[i + 1])) for i in range(len(rules) - 1)
        ]
        heights = [y1 - y0 for y0, y1 in band_candidates]
        if len(band_candidates) >= 10:
            # Discard degenerate bands (double-detected rules).
            median_h = sorted(heights)[len(heights) // 2]
            bands = [b for b in band_candidates if (b[1] - b[0]) > median_h * 0.4]
            positional = True

    if not bands:
        col_gap = min(anchors[i + 1] - anchors[i] for i in range(len(anchors) - 1))
        date_tol = max(30.0, col_gap * 0.6)
        centers = sorted(
            _line_center(ln.box)[1]
            for ln in lines
            if ln.box and ln.text and _line_center(ln.box)[1] > header_bottom + 4
            and abs(_line_center(ln.box)[0] - date_label_x) <= date_tol
        )
        if not centers:
            return None
        # Bands = midpoints between consecutive date centres.
        edges = [header_bottom]
        for i in range(len(centers) - 1):
            edges.append((centers[i] + centers[i + 1]) / 2)
        edges.append(centers[-1] + (centers[-1] - edges[-1]))
        bands = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

    return _TableGeometry(anchors, col_bounds, bands, header_bottom, positional)


def _assign_lines_to_rows(lines: list[OcrLine], geom: _TableGeometry) -> list[dict]:
    """Place every OCR line below the header into its row band and column."""
    rows: list[dict] = [
        {"date": "", **{c: "" for c in HOUR_COLUMNS}} for _ in geom.bands
    ]

    def band_index(cy: float) -> Optional[int]:
        for i, (y0, y1) in enumerate(geom.bands):
            if y0 <= cy < y1:
                return i
        return None

    def col_index(cx: float) -> int:
        return min(range(len(geom.anchors)),
                   key=lambda i: abs(geom.anchors[i] - cx))

    for ln in lines:
        if not ln.box or not ln.text:
            continue
        # The bands themselves start below the header, so they are the only
        # boundary needed; but never let a printed column label (ENTRE…) that
        # a rule sliced into the first band be taken as data.
        if normalize_label(ln.text) in ("date", "entre", "sortie", "matin",
                                        "matain", "soir"):
            continue
        cx, cy = _line_center(ln.box)
        r = band_index(cy)
        if r is None:
            continue
        col = TABLE_COLUMNS[col_index(cx)]
        rows[r][col] = clean_text(f"{rows[r][col]} {ln.text}")
    return rows


def _cell_has_ink(gray: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> bool:
    """True if the cell region contains real ink (not just paper/grid rules)."""
    from config import CELL_MIN_INK_FRACTION

    h, w = gray.shape[:2]
    # Inset to keep the printed rules out of the ink count.
    ix0, iy0 = max(0, x0 + 4), max(0, y0 + 3)
    ix1, iy1 = min(w, x1 - 4), min(h, y1 - 3)
    if ix1 - ix0 < 6 or iy1 - iy0 < 6:
        return False
    region = gray[iy0:iy1, ix0:ix1]
    # Ink = pixels clearly darker than the cell's own paper level.
    paper = float(np.percentile(region, 80))
    ink_fraction = float(np.mean(region < paper - 60))
    return ink_fraction >= CELL_MIN_INK_FRACTION


def refine_rows_with_cells(rows: list[dict], geom: _TableGeometry,
                           images: dict, engine: OcrEngine) -> list[dict]:
    """Second pass: recognition-only re-read of cells the page pass left empty.

    Full-page detection sometimes skips faint handwriting entirely. For every
    empty cell whose region actually contains ink, crop it and run just the
    recogniser (no detection) on the crop. A recovered value is accepted only
    when it is confident AND validates for its column — so a blank cell can
    never gain an invented value.
    """
    from config import OCR_REC_ACCEPT_CONFIDENCE

    color = images["color"]
    gray = images["gray"]
    recovered = 0
    for r, (y0, y1) in enumerate(geom.bands):
        for c, col in enumerate(TABLE_COLUMNS):
            if rows[r][col]:
                continue
            x0 = int(geom.col_bounds[c])
            x1 = int(geom.col_bounds[c + 1])
            iy0, iy1 = int(y0), int(y1)
            if not _cell_has_ink(gray, x0, iy0, x1, iy1):
                continue
            crop = color[max(0, iy0 - 2):iy1 + 2, max(0, x0):x1]
            text, conf = engine.recognize_cell(crop)
            if not text or conf < OCR_REC_ACCEPT_CONFIDENCE:
                continue
            if col == "date":
                if _looks_like_day(text):
                    rows[r][col] = text
                    recovered += 1
            else:
                if normalize_hour(text):
                    rows[r][col] = text
                    recovered += 1
    if recovered:
        logger.info("Cell-recognition pass recovered %d values", recovered)
    return rows


def read_body_from_lines(body: list[list[Cell]], lines: list[OcrLine]) -> list[dict]:
    """Fallback: assign each full-page OCR line to the grid cell containing it.

    Used only when the header labels needed by extract_table_from_lines are not
    found. Reading the whole page once and mapping boxes onto the detected grid
    is still far more reliable than OCR-ing each tiny single-value crop.
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
                 year: Optional[int], positional: bool = False) -> list[dict]:
    """Validate values and force exactly days_in_month ordered rows.

    The sheet lists the days of the month in order, so day assignment is driven
    by **row position** (row 1 -> day 1). This is robust to the M/D vs D/M
    ambiguity of the printed dates and to a missing/unreadable date cell. The
    output date is composed canonically from the day, month and year, so it is
    always a valid dd/mm/yy and matches the sheet's own sequence.

    With positional=True the rows came from the table's own grid bands (one
    band = one day), so empty rows in the middle are REAL empty days and are
    kept in place; only trailing blank bands are trimmed. Without it, empty
    rows are treated as detection noise and dropped.
    """
    # The printed date cells (many of them) are a more reliable source of the
    # month than a single handwritten "Mois" word, which OCR often misreads
    # (e.g. "Juin" -> "8"). Prefer the date-inferred month/year.
    inferred_month, inferred_year = _infer_month_year(raw_rows)
    month = inferred_month or month or datetime.now().month
    year = inferred_year or year or datetime.now().year
    n_days = days_in_month(month, year)

    data_rows: list[dict] = []
    for r in raw_rows:
        if _is_header_row(r):
            continue
        hours = {c: normalize_hour(r.get(c, "")) for c in HOUR_COLUMNS}
        if not positional:
            # Keep a row only if it has a punch/OFF value or an anchoring day.
            has_content = any(hours.values()) or _looks_like_day(r.get("date", ""))
            if not has_content:
                continue
        data_rows.append(hours)

    if positional:
        # Trim trailing blank bands (below the last day of the month).
        while data_rows and not any(data_rows[-1].values()):
            data_rows.pop()

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

    # OCR runs on the deskewed *colour* image (PaddleOCR is trained on natural
    # images; the heavily processed binary/CLAHE hurts its detector). The grid
    # cells and this image share the same coordinate space.
    full_lines = engine.read_full(images["color"])
    logger.info("OCR found %d text lines on the page", len(full_lines))

    # Metadata (header fields) sits above the table's own header band. Find that
    # band from the printed column labels rather than the (possibly wrong) grid.
    metadata_cutoff = _table_header_top(full_lines)
    header = extract_header(full_lines, metadata_cutoff)

    month = month_from_text(header.get("mois", "")) if header.get("mois") else None

    # Preferred method: columns anchored on the printed header labels, rows on
    # the horizontal grid rules (one band = one day, immune to handwriting
    # drift), then a recognition-only pass over inked-but-unread cells.
    geom = _locate_geometry(full_lines, grid.get("rows"))
    positional = False
    if geom is not None:
        raw_rows = _assign_lines_to_rows(full_lines, geom)
        raw_rows = refine_rows_with_cells(raw_rows, geom, images, engine)
        positional = geom.positional
        logger.info("Reconstructed %d table rows (%s bands)",
                    len(raw_rows), "grid-rule" if positional else "date-anchored")
    else:
        logger.info("Header anchors not found; using grid-cell mapping")
        raw_rows = read_body_from_lines(body, full_lines)
    attendance = canonicalize(raw_rows, month=month, year=None,
                              positional=positional)

    return {
        "animateur": header.get("animateur", ""),
        "magasin": header.get("magasin", ""),
        "societe": header.get("societe", ""),
        "marque": header.get("marque", ""),
        "mois": header.get("mois", ""),
        "attendance": attendance,
    }

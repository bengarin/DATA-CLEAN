# Attendance Sheet Extractor (Pointage de Présence)

Batch pipeline that reads scanned monthly attendance sheets and exports the
data to **JSON, CSV and a structured Excel** file. It is geometry-first:
because every sheet shares the same layout, the table grid is recovered with
OpenCV and each cell is read **independently** with PaddleOCR, which is far more
robust than reconstructing a table from scattered OCR words.

```
input/  →  OpenCV preprocess (deskew · denoise · CLAHE)
        →  table detection (morphology line extraction → cell grid)
        →  header OCR + per-cell OCR (PaddleOCR, low-confidence auto-retry)
        →  validation (date & hour regex, forced 30/31 rows)
        →  export JSON + CSV + Excel  →  output/
```

## Project layout

```
attendance_extractor/
├── main.py            # batch entry point (processes every image in input/)
├── config.py          # all tunables: paths, layout template, regex, OCR
├── preprocess.py      # resize, deskew, denoise, contrast; per-cell retry
├── table_detector.py  # morphology line detection → Cell grid
├── ocr.py             # PaddleOCR wrapper (2.x + 3.x), cell/full reading
├── parser.py          # header match + cell→column mapping + canonicalisation
├── exporter.py        # JSON / CSV / structured Excel writers
├── utils.py           # logging, text normalisation, date/hour validation
├── requirements.txt
├── input/             # drop scanned sheets here
└── output/            # generated .json / .csv / .xlsx (+ _debug/ when DEBUG=1)
```

## Install

```bash
pip install -r requirements.txt
```

On the **first run** PaddleOCR downloads its detection/recognition/orientation
models automatically (a few hundred MB) — this needs internet access to one of
PaddleOCR's model hosts (HuggingFace / ModelScope / BOS). After that it runs
offline.

## Run

```bash
# 1. put one or more scanned sheets in input/
# 2. run:
python main.py
```

Each `input/<name>.<ext>` produces `output/<name>.json`, `output/<name>.csv`
and `output/<name>.xlsx`. The batch is unattended — a failing document is logged
and the run continues. Set `ATTENDANCE_DEBUG=1` to also write, per document, the
detected grid overlay and binary image under `output/_debug/`.

## Output shape

```json
{
  "animateur": "WAKRIM HOUDAIFA",
  "magasin": "Electroplanet hay riyad -Rabat",
  "societe": "RICO ACTIF RETAIL",
  "marque": "ASUS",
  "mois": "juin",
  "attendance": [
    { "date": "01/06/26", "matin_entre": "08H00", "matin_sortie": "16H00",
      "soir_entre": "", "soir_sortie": "" }
  ]
}
```

- The `attendance` array always has **exactly the number of days in the month**
  (30 or 31, 28/29 for February) — missing cells stay empty strings.
- Hours are validated against `^(OFF|[0-2]?[0-9]H[0-5][0-9])$` (plus tolerant
  variants) and normalised to `HHhMM`; `OFF` and empty are preserved.
- Dates are validated and normalised to `dd/mm/yy`.

## Excel structure

The workbook reproduces the sheet exactly: a metadata block on top, then the
two-level header — **DATE** (merged over both header rows), **MATIN → ENTRE /
SORTIE**, **SOIR → ENTRE / SORTIE** — with borders and centred text.

## Adapting to another template

Nothing about the content is hard-coded. To re-target a different sheet, edit
`config.py` only:
- `TABLE_COLUMNS` / `EXCEL_HEADER` — logical column order and Excel header.
- `HEADER_FIELDS` — metadata labels to look for.
- `H_LINE_SCALE` / `V_LINE_SCALE` / `*_MERGE_TOLERANCE` — grid line detection.
- `OCR_LANG`, `OCR_MIN_CONFIDENCE` — OCR language and the retry threshold.

## Notes on accuracy

Handwritten hours are the hardest part: PaddleOCR's printed-text models read
clear handwriting well but can miss messy strokes. The pipeline mitigates this
with a per-cell low-confidence retry (upscale + sharpen + Otsu). For very messy
handwriting, a handwriting-specialised recognition model can be plugged into
`ocr.py` without changing the rest of the pipeline.

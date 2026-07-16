# Document Quality Validation API

A minimal PHP 8.3 API that decides whether an uploaded document photo is good
enough (ACCEPTED / REJECTED) before it is sent on for OCR. No OCR, no database,
no image storage, no authentication — the uploaded file is deleted immediately
after the response (or if any error occurs).

The PHP endpoint receives the upload and runs a Python/OpenCV analyzer
(`analyze_image.py`) that scores the image quality and returns the decision.

## Architecture

```
mobile app / ui.html  ──►  index.php  ──►  ImageValidator.php  ──►  analyze_image.py (OpenCV)
                              ▲                                            │
                              └──────────────  JSON  ◄────────────────────┘
```

| File                 | Responsibility                                                                 |
|----------------------|---------------------------------------------------------------------------------|
| `index.php`          | Front controller / REST endpoint. Reads `$_FILES`, returns JSON.               |
| `ImageValidator.php` | Validates the upload (present, real image, size, mime), runs the analyzer, guarantees cleanup of the temp file. |
| `analyze_image.py`   | OpenCV image-quality scoring (the six metrics below).                          |
| `config.php`         | Loads settings (Python path, upload limits) from environment / `.env`.         |
| `ui.html`            | Simple browser tester that posts to `index.php`.                               |

## Scoring rubric

The analyzer returns a 0–100 score built from six sub-metrics. An image is
**accepted** when the total is **≥ 70** *and* no hard failure was detected.

| Metric      | Max | What it measures                                                        |
|-------------|-----|-------------------------------------------------------------------------|
| Sharpness   | 30  | Laplacian variance (blur detection)                                     |
| Brightness  | 20  | Grayscale mean — wide band so white paper is not "overexposed"          |
| Contrast    | 15  | Ink-vs-paper separation **inside content cells**, not global std-dev    |
| Perspective | 15  | Document rectangularity / skew (pure rotation is not penalized)         |
| Resolution  | 10  | Effective megapixels of the original upload                             |
| Detection   | 10  | Is a **complete** document present and filling the frame                |

An image is also **hard-rejected** when the *entire document is not visible* —
i.e. the page is small in the frame, partially covered by another object, or
cut off (measured as `coverage × rectangularity`). A full page photographed
straight-on passes even if it is rotated; a page half-hidden under an envelope
on a table does not.

### Why photos were being falsely rejected — and what changed

Real, perfectly readable documents (white paper, printed tables, photographed
on a table with a torn corner or a slight tilt) were being **rejected**. The
causes and fixes:

- **Contrast** used a global `std(gray)`. A page that is mostly white paper
  with sparse dark text has a low global std-dev, so it was flagged
  *"Low contrast"*. It now measures the ink/paper separation **inside the cells
  that carry text**, which stays high for any legible document.
- **Perspective / Detection** required OpenCV to find a perfect 4-corner
  quadrilateral. Real pages (torn corners, overlapping objects, wrinkles) never
  produced a clean quad, so they scored **0**. Detection now finds the dominant
  bright document region against the background and judges coverage + text
  structure; perspective rewards a roughly rectangular, roughly aligned page.
- **Brightness** band was too narrow for bright paper. It is now wide and only
  penalizes genuine darkness or a fully blown-out frame.
- **Accept threshold** was 80 (a well-lit readable page scoring 71 was
  rejected). It is now 70, gated by "no hard failure".

Genuinely bad photos are still rejected: blurry (low Laplacian variance), too
dark, too low-resolution, or blank / no document.

## Setup

```bash
# PHP 8.3 (extensions: fileinfo, json)
# Python 3 for the analyzer:
pip install -r requirements.txt

cp .env.example .env      # then set PYTHON_PATH if "python3" is not on PATH
```

## Running locally

```bash
php -S localhost:8080
# open http://localhost:8080/ui.html  (or POST to /index.php)
```

## Usage

`POST /index.php` with `multipart/form-data`, field name `image`:

```bash
curl -X POST http://localhost:8080/index.php -F "image=@/path/to/photo.jpg"
```

Accepted response:

```json
{
    "status": "success",
    "score": 92,
    "decision": "ACCEPTED",
    "metrics": {
        "sharpness": "30/30", "brightness": "16/20", "contrast": "15/15",
        "perspective": "13/15", "resolution": "8/10", "detection": "10/10"
    },
    "reasons": []
}
```

Rejected response:

```json
{
    "status": "rejected",
    "score": 60,
    "decision": "REJECTED",
    "metrics": { "sharpness": "0/30", "...": "..." },
    "reasons": ["Image is blurry"]
}
```

## Error responses

| HTTP | Cause                                                        |
|------|--------------------------------------------------------------|
| 400  | No file, invalid file, too large, or unsupported mime type   |
| 405  | Method other than `POST`                                     |
| 500  | Analyzer failed to run or returned invalid JSON              |

## Guarantees

- Nothing is written to a database and no image is stored.
- The uploaded temp file is `unlink()`-ed in a `finally` block right after the
  analyzer runs (success or failure).
- No authentication layer — add one at the infrastructure level (API gateway,
  IP allowlist) if the API is exposed publicly.
```

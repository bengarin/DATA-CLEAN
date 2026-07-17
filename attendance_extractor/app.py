"""
Web interface for the attendance-sheet extractor.

Run:
    python app.py
then open http://localhost:5000 , drop a scanned sheet, and the extracted
header + attendance table are shown in the page with one-click download of the
generated Excel / CSV / JSON.

This reuses the exact same pipeline as the batch `main.py` (preprocess ->
table detect -> PaddleOCR -> parse -> export); the browser is only a front end.
"""

from __future__ import annotations

import os
import time
import uuid

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from config import OUTPUT_DIR
from exporter import export_all
from ocr import get_engine
from parser import parse_document
from preprocess import preprocess
from table_detector import detect_grid
from utils import ensure_dir, get_logger

logger = get_logger()

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_uploads")
ALLOWED = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
MAX_CONTENT_LENGTH = 15 * 1024 * 1024  # 15 MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/extract", methods=["POST"])
def extract():
    """Receive one uploaded image, run the pipeline, return the record + files."""
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded (field 'image')."}), 400

    upload = request.files["image"]
    if not upload.filename:
        return jsonify({"error": "Empty filename."}), 400

    ext = os.path.splitext(upload.filename)[1].lower()
    if ext not in ALLOWED:
        return jsonify({"error": f"Unsupported file type '{ext}'."}), 400

    ensure_dir(UPLOAD_DIR)
    ensure_dir(OUTPUT_DIR)

    stem = f"{secure_filename(os.path.splitext(upload.filename)[0])}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    saved_path = os.path.join(UPLOAD_DIR, stem + ext)
    upload.save(saved_path)

    try:
        images = preprocess(saved_path)
        grid = detect_grid(images["binary"])
        record = parse_document(grid, images, get_engine())
        paths = export_all(record, OUTPUT_DIR, stem)
    except Exception as exc:  # noqa: BLE001 - surface a clean error to the UI
        logger.error("Extraction failed: %s", exc)
        return jsonify({"error": f"Extraction failed: {exc}"}), 500
    finally:
        # The upload is transient; the exports in output/ are the deliverable.
        if os.path.exists(saved_path):
            os.remove(saved_path)

    return jsonify({
        "record": record,
        "downloads": {fmt: os.path.basename(p) for fmt, p in paths.items()},
    })


@app.route("/download/<path:filename>")
def download(filename: str):
    """Serve a generated export from the output folder."""
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    ensure_dir(UPLOAD_DIR)
    ensure_dir(OUTPUT_DIR)
    # Warm the OCR model once at startup so the first request is fast.
    logger.info("Starting web UI on http://localhost:5000 …")
    app.run(host="0.0.0.0", port=5000, debug=False)

import sys
import json
import cv2
import numpy as np


def _grid_local_contrast(gray):
    """Ink-vs-paper separation measured inside the cells that actually carry
    content, instead of a global std-dev.

    A real document is mostly uniform bright paper with sparse dark text, so a
    global std-dev collapses toward zero and falsely reports "low contrast".
    Splitting the image into a grid and looking at the per-cell (max - min)
    range captures how cleanly ink separates from paper regardless of how much
    empty margin surrounds the text.
    """
    h, w = gray.shape[:2]
    rows, cols = 24, 24
    ch, cw = max(1, h // rows), max(1, w // cols)
    ranges = []
    for cy in range(rows):
        y0, y1 = cy * ch, min(h, (cy + 1) * ch)
        for cx in range(cols):
            x0, x1 = cx * cw, min(w, (cx + 1) * cw)
            cell = gray[y0:y1, x0:x1]
            if cell.size == 0:
                continue
            rng = int(cell.max()) - int(cell.min())
            if rng > 40:  # a cell that carries real detail (text/lines)
                ranges.append(rng)
    if not ranges:
        return 0.0, 0.0
    # 75th percentile of content-cell ranges = typical strong ink/paper gap.
    separation = float(np.percentile(ranges, 75))
    fill = len(ranges) / float(rows * cols)  # how much of the frame has content
    return separation, fill


def _document_region(gray):
    """Locate the document as the dominant bright region against a darker
    background, tolerating torn corners, overlapping objects and wrinkles.

    Returns (coverage, rectangularity, skew_deg):
      coverage       fraction of the frame the document occupies (0..1)
      rectangularity how rectangular that region is (0..1, 1 = clean rectangle)
      skew_deg       tilt of the region in degrees (0 = axis aligned)
    """
    h, w = gray.shape[:2]
    frame_area = float(w * h)

    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Make the paper the white side of the mask.
    if np.mean(mask) < 127:
        mask = cv2.bitwise_not(mask)

    bright_fraction = float(np.mean(mask > 0))

    # When the page already fills the frame there is no background to segment;
    # that is a well-framed document, not a missing one.
    if bright_fraction > 0.85:
        return min(1.0, bright_fraction), 1.0, 0.0

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=2
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return bright_fraction, 0.0, 0.0

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    coverage = area / frame_area

    (_, (rw, rh), angle) = cv2.minAreaRect(largest)
    rect_area = rw * rh
    rectangularity = float(area / rect_area) if rect_area > 0 else 0.0
    skew = abs(angle) % 90
    skew = min(skew, 90 - skew)  # fold into 0..45

    return coverage, min(1.0, rectangularity), float(skew)


def analyze_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"status": "error", "message": "Could not read the image."}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = img.shape[:2]

        # 1. Resolution (gradual, not a hard cliff at 500px).
        megapixels = (width * height) / 1_000_000.0
        if megapixels >= 1.5:
            resolution_score = 10
        elif megapixels <= 0.25:
            resolution_score = 0
        else:
            resolution_score = int(round(10 * (megapixels - 0.25) / (1.5 - 0.25)))

        # 2. Sharpness (variance of the Laplacian) — blur detection.
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = int(min(30, max(0, (laplacian_var / 400.0) * 30)))

        # 3. Brightness — grayscale mean with a WIDE band, because a photo of a
        #    document is mostly bright paper. Only genuine darkness or a fully
        #    blown-out frame is penalized (no more "overexposed" white pages).
        brightness = float(np.mean(gray))
        if 105 <= brightness <= 238:
            brightness_score = 20
        elif brightness < 105:
            brightness_score = int(max(0, 20 * (brightness - 45) / 60))
        else:
            brightness_score = int(max(0, 20 * (252 - brightness) / 14))

        # 4. Contrast — local ink/paper separation, not global std-dev.
        separation, content_fill = _grid_local_contrast(gray)
        contrast_score = int(min(15, max(0, (separation / 120.0) * 15)))

        # 5. Document detection + perspective.
        coverage, rectangularity, skew = _document_region(gray)

        #    Completeness = how much of a *whole* document rectangle is visible.
        #    A page that fills the frame as a clean rectangle scores ~1.0; a
        #    page that is small in frame, partially covered by another object,
        #    or torn scores low. This is the "entire document visible" check.
        completeness = coverage * rectangularity

        #    Detection: is a complete, structured document present and framed?
        detection_score = int(round(10 * min(1.0, completeness / 0.55)))

        #    Perspective: reward a rectangular, roughly aligned page. Pure
        #    rotation (skew folded into 0..45) is fine; only real skew loses.
        rect_term = min(1.0, rectangularity / 0.80)
        if skew <= 8:
            skew_term = 1.0
        elif skew >= 30:
            skew_term = 0.0
        else:
            skew_term = (30 - skew) / 22.0
        perspective_score = int(round(15 * (0.5 * rect_term + 0.5 * skew_term)))

        total_score = (
            sharpness_score
            + brightness_score
            + contrast_score
            + perspective_score
            + resolution_score
            + detection_score
        )

        # Hard failure reasons (only genuine, OCR-breaking problems).
        reasons = []
        if sharpness_score < 12:
            reasons.append("Image is blurry")
        if brightness < 80:
            reasons.append("Low brightness")
        elif brightness > 246:
            reasons.append("Image is overexposed")
        if contrast_score < 5:
            reasons.append("Low contrast")
        if resolution_score < 4:
            reasons.append("Low resolution")
        if content_fill < 0.05:
            reasons.append("No document detected")
        elif completeness < 0.45:
            # Document is small in frame, partially covered, or cut off.
            reasons.append("Entire document not visible or partially covered")

        # Accept when the quality is high enough AND nothing is hard-broken.
        decision = "ACCEPTED" if (total_score >= 70 and not reasons) else "REJECTED"

        return {
            "status": "success" if decision == "ACCEPTED" else "rejected",
            "score": int(total_score),
            "decision": decision,
            "metrics": {
                "sharpness": f"{sharpness_score}/30",
                "brightness": f"{brightness_score}/20",
                "contrast": f"{contrast_score}/15",
                "perspective": f"{perspective_score}/15",
                "resolution": f"{resolution_score}/10",
                "detection": f"{detection_score}/10",
            },
            "reasons": reasons,
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "message": "No image path provided."}))
        sys.exit(1)

    image_path = sys.argv[1]
    result = analyze_image(image_path)
    print(json.dumps(result))

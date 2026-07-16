import sys
import json
import cv2
import numpy as np


def _grid_local_contrast(gray):
    """Contrast AND sharpness measured inside the cells that actually carry
    content, instead of over the whole frame.

    A real document is mostly uniform bright paper with sparse dark text. A
    global std-dev collapses toward zero ("low contrast") and a global Laplacian
    variance is dragged down by the empty paper ("blurry") even when the text
    itself is crisp and readable. Looking at each grid cell that has real
    detail — its ink/paper range and its local Laplacian variance — captures how
    legible and how sharp the *text* is, regardless of how much blank margin
    surrounds it.

    Returns (separation, sharpness, fill):
      separation  75th-percentile ink/paper range across content cells
      sharpness   80th-percentile Laplacian variance across content cells
      fill        fraction of the frame that carries content
    """
    h, w = gray.shape[:2]
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    rows, cols = 24, 24
    ch, cw = max(1, h // rows), max(1, w // cols)
    ranges = []
    sharps = []
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
                sharps.append(float(lap[y0:y1, x0:x1].var()))
    if not ranges:
        return 0.0, 0.0, 0.0
    separation = float(np.percentile(ranges, 75))
    sharpness = float(np.percentile(sharps, 80))
    fill = len(ranges) / float(rows * cols)  # how much of the frame has content
    return separation, sharpness, fill


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
    # The paper/document is the BRIGHT cluster (above the Otsu threshold),
    # regardless of whether it is the majority of the frame. Keying off the
    # bright side works whether the page fills the frame or sits small on a
    # dark table.
    thr, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (blurred >= max(1, int(thr))).astype(np.uint8) * 255

    # Close the dark text holes so the paper reads as one solid region.
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=2
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0, 0.0

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    coverage = area / frame_area

    (_, (rw, rh), angle) = cv2.minAreaRect(largest)
    rect_area = rw * rh
    rectangularity = float(area / rect_area) if rect_area > 0 else 0.0
    skew = abs(angle) % 90
    skew = min(skew, 90 - skew)  # fold into 0..45

    return coverage, min(1.0, rectangularity), float(skew)


def _paper_brightness(gray):
    """Mean luminance of the document (paper) itself, not the whole frame.

    A readable page photographed on a dark table must not be judged "too dark"
    just because the background is dark, so brightness is measured on the
    brighter (paper) side of an Otsu split. Falls back to the whole-frame mean
    when there is no distinct bright region.
    """
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thr, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Paper = the bright cluster (above the Otsu threshold), never the darker
    # background, so the paper's own illumination is what gets measured.
    paper = gray[blurred >= max(1, int(thr))]
    if paper.size < 0.03 * gray.size:
        return float(np.mean(gray))
    return float(np.mean(paper))


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

        # Content-cell stats: sharpness and contrast of the TEXT itself, so
        # large blank areas of a sparse page do not drag them down.
        separation, cell_sharpness, content_fill = _grid_local_contrast(gray)

        # 2. Sharpness (Laplacian variance of the text regions) — blur detection.
        sharpness_score = int(min(30, max(0, (cell_sharpness / 900.0) * 30)))

        # 3. Brightness — measured on the paper itself (not the whole frame, so
        #    a dark background/table does not count as "dark"). Paper is meant
        #    to be bright, so only genuine low light loses points; a bright,
        #    well-lit page is never penalized for being bright.
        brightness = _paper_brightness(gray)
        if brightness >= 110:
            brightness_score = 20
        else:
            brightness_score = int(max(0, 20 * (brightness - 45) / 65))  # 45->0,110->20

        # 4. Contrast — local ink/paper separation, not global std-dev.
        contrast_score = int(min(15, max(0, (separation / 120.0) * 15)))

        # 5. Document detection + perspective.
        #    The shape / framing / aspect ratio of the photo does NOT matter —
        #    only whether a complete, readable document is present. A clean
        #    rectangular page is accepted whether it fills the frame or sits in
        #    the middle with a margin, and whether or not it is rotated. What is
        #    rejected is a document that is partially covered, torn, or cut off
        #    (an irregular region), because then part of the data is missing.
        coverage, rectangularity, skew = _document_region(gray)

        coverage_term = min(1.0, coverage / 0.70)       # document fills the frame?
        rect_term = min(1.0, rectangularity / 0.80)     # clean rectangle?
        structure_term = min(1.0, content_fill / 0.20)  # readable text present?
        detection_score = int(round(
            10 * (0.5 * coverage_term + 0.2 * rect_term + 0.3 * structure_term)
        ))

        #    Perspective: reward a rectangular, roughly aligned page. Pure
        #    rotation (skew folded into 0..45) is fine; only real skew loses.
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
        if brightness < 85:
            reasons.append("Low brightness")
        elif brightness > 248 and contrast_score < 6 and sharpness_score >= 12:
            # Blown out only if the paper is white *and* the text washed out
            # with it (contrast collapsed) on an otherwise sharp image — a
            # bright readable page is fine, and blur is reported separately.
            reasons.append("Image is overexposed")
        if contrast_score < 5:
            reasons.append("Low contrast")
        if resolution_score < 4:
            reasons.append("Low resolution")
        if content_fill < 0.05:
            reasons.append("No document detected")
        elif coverage < 0.55:
            # The document does not fill enough of the frame: it is too far
            # away, cut off, or partially covered by another object, so part of
            # the data is missing. Aspect ratio / rotation are NOT judged here —
            # only that the whole page is present and close enough to read.
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

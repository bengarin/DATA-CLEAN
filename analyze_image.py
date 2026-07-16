import sys
import json
import cv2
import numpy as np

def analyze_image(image_path):
    try:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            return {"status": "error", "message": "Could not read the image."}

        # Convert to Grayscale and HSV
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 1. Resolution Check
        height, width = img.shape[:2]
        resolution_score = 10
        if width < 500 or height < 500:
            resolution_score = 0

        # 2. Sharpness (Variance of Laplacian)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(30, max(0, int((laplacian_var / 500.0) * 30)))
        
        # 3. Brightness (Mean of V channel)
        brightness = np.mean(hsv[:, :, 2])
        if 50 <= brightness <= 200:
            brightness_score = 20
        else:
            brightness_score = max(0, int(20 - abs(125 - brightness) / 5))

        # 4. Contrast (Std Dev of Grayscale)
        contrast = np.std(gray)
        contrast_score = min(15, max(0, int((contrast / 80.0) * 15)))

        # 5. Document Detection (Perspective/Contours)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        doc_detected = False
        perspective_score = 0
        detection_score = 0

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            # If our approximated contour has four points, we can assume we found the document
            if len(approx) == 4:
                doc_detected = True
                area_ratio = cv2.contourArea(c) / (width * height)
                
                # Check if document occupies a reasonable portion of the image (> 20%)
                if area_ratio > 0.2:
                    perspective_score = 15
                    detection_score = 10
                else:
                    perspective_score = 5
                    detection_score = 5
                break

        if not doc_detected:
            # Fallback if no perfect 4-point contour found but there are large contours
            if len(contours) > 0 and cv2.contourArea(contours[0]) / (width * height) > 0.3:
                perspective_score = 10
                detection_score = 5

        # Calculate final score (Max: 30 + 20 + 15 + 15 + 10 + 10 = 100)
        total_score = sharpness_score + brightness_score + contrast_score + perspective_score + resolution_score + detection_score

        # Determine reasons if score < 80
        reasons = []
        if sharpness_score < 20: reasons.append("Image is blurry")
        if brightness_score < 15: reasons.append("Low or excessive brightness")
        if contrast_score < 10: reasons.append("Low contrast")
        if perspective_score < 10: reasons.append("Document not properly aligned or cropped")
        if resolution_score == 0: reasons.append("Low resolution")
        if detection_score == 0: reasons.append("Document not detected clearly")

        decision = "ACCEPTED" if total_score >= 80 else "REJECTED"

        result = {
            "status": "success" if decision == "ACCEPTED" else "rejected",
            "score": total_score,
            "decision": decision,
            "metrics": {
                "sharpness": f"{sharpness_score}/30",
                "brightness": f"{brightness_score}/20",
                "contrast": f"{contrast_score}/15",
                "perspective": f"{perspective_score}/15",
                "resolution": f"{resolution_score}/10",
                "detection": f"{detection_score}/10"
            },
            "reasons": reasons
        }
        return result

    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "message": "No image path provided."}))
        sys.exit(1)
        
    image_path = sys.argv[1]
    result = analyze_image(image_path)
    print(json.dumps(result))

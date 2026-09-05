"""
================================================================================
                    DESHGUARD AI: AI DETECTOR & FORENSICS ENGINE
                                (ai_detector.py)
================================================================================
Module Purpose:
  - Deepfake & EXIF Metadata Multi-pass Forensic Verification
  - YOLOv8 Object Detection Pipeline with Adaptive Non-Maximum Suppression
  - Advanced Dark Cavity & Shadow Depth Analysis for Road Potholes
================================================================================
"""

import os
import math
import logging
import gc
from typing import Tuple, List, Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image, ImageChops
from PIL.ExifTags import TAGS, GPSTAGS

# Ultralytics Safeguard
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

logger = logging.getLogger("DeshGuardEngine.AIDetector")


class PotholeForensicPipeline:
    """YOLOv8 Inference Pipeline with Custom Non-Maximum Suppression."""

    def __init__(self, confidence_threshold: float = 0.40):
        self.conf_thresh = confidence_threshold
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initializes custom best.pt or fallback YOLOv8 Neural Network weights."""
        if YOLO_AVAILABLE:
            model_path = "best.pt" if os.path.exists("best.pt") else "yolov8n.pt"
            try:
                self.model = YOLO(model_path)
                logger.info(f"[YOLO ENGINE] Successfully loaded weights from: {model_path}")
            except Exception as e:
                logger.warning(f"[YOLO ENGINE WARN] Failed to load model ({model_path}): {e}")
                self.model = None


    @staticmethod
    def non_max_suppression_fast(boxes: np.ndarray, overlapThresh: float = 0.3) -> List[int]:
        """Suppresses overlapping bounding boxes using IoU analysis."""
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

        return pick


class AdvancedRoadAnalyzer:
    """Fallback Computer Vision Structural Analyzer using Shadow Depth Extraction."""

    @staticmethod
    def extract_pothole_candidate_regions(img: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Extracts dark road cavity shapes when Neural Models miss predictions."""
        h, w, _ = img.shape
        total_area = h * w

        # Contrast Enhancement & CLAHE Application
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)

        # Adaptive Dark Thresholding
        blurred = cv2.GaussianBlur(enhanced_gray, (9, 9), 0)
        dark_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 35, 11
        )

        # Morphological Isolation of Structural Defects
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))

        isolated_lines = cv2.morphologyEx(dark_thresh, cv2.MORPH_OPEN, horizontal_kernel)
        isolated_lines = cv2.add(isolated_lines, cv2.morphologyEx(dark_thresh, cv2.MORPH_OPEN, vertical_kernel))

        clean_mask = cv2.subtract(dark_thresh, isolated_lines)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        morphed = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if (total_area * 0.012) < area < (total_area * 0.45):
                bx, by, bw, bh = cv2.boundingRect(cnt)
                aspect_ratio = bw / float(bh)

                if 0.35 <= aspect_ratio <= 2.8:
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter > 0:
                        circularity = 4 * math.pi * (area / (perimeter * perimeter))
                        rect_area = bw * bh
                        extent = float(area) / rect_area

                        if circularity > 0.18 and extent > 0.30:
                            roi_gray = gray[by:by+bh, bx:bx+bw]
                            mean_val, std_val = cv2.meanStdDev(roi_gray)

                            if mean_val[0][0] < 145 and std_val[0][0] > 12.0:
                                confidence = min(0.95, 0.50 + (area / total_area) * 2)
                                candidates.append((bx, by, bx + bw, by + bh, confidence))

        return candidates


# Global Engine Instances
pipeline_engine = PotholeForensicPipeline()
analyzer_engine = AdvancedRoadAnalyzer()


def process_road_potholes(input_path: str, output_path: str) -> Tuple[bool, int]:
    img = cv2.imread(input_path)
    if img is None:
        return False, 0

    h, w, _ = img.shape
    detected_boxes = []

    box_thickness = max(5, int(min(h, w) * 0.008))
    font_scale = max(0.6, min(h, w) * 0.0014)
    font_thickness = max(2, int(font_scale * 2.0))

    if pipeline_engine.model is not None:
        try:
            results = pipeline_engine.model(input_path, verbose=False)
            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf >= 0.40:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        bw, bh = x2 - x1, y2 - y1
                        if bw < w * 0.85 and bh < h * 0.85:
                            detected_boxes.append([x1, y1, x2, y2, conf])
        except Exception as e:
            logger.warning(f"[FORENSICS WARN] YOLO Inference bypass: {e}")

    if len(detected_boxes) == 0:
        raw_candidates = analyzer_engine.extract_pothole_candidate_regions(img)
        for box in raw_candidates:
            detected_boxes.append(list(box))

    pothole_count = 0
    if len(detected_boxes) > 0:
        box_arr = np.array([[b[0], b[1], b[2], b[3]] for b in detected_boxes])
        selected_indices = PotholeForensicPipeline.non_max_suppression_fast(box_arr, overlapThresh=0.25)

        final_boxes = [detected_boxes[idx] for idx in selected_indices]
        pothole_count = len(final_boxes)

        last_y = -100

        for (x1, y1, x2, y2, conf) in final_boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), box_thickness)
            
            label = f"POTHOLE {int(conf * 100)}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            
            calc_y = max(0, y1 - th - 12)
            if abs(calc_y - last_y) < th + 10:
                calc_y = y1 + th + 12
            last_y = calc_y

            overlay = img.copy()
            cv2.rectangle(overlay, (x1, calc_y), (x1 + tw + 14, calc_y + th + 10), (0, 200, 0), -1)
            cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
            
            cv2.putText(
                img, label, (x1 + 6, calc_y + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA
            )

    cv2.imwrite(output_path, img)
    del img
    gc.collect()

    return True, pothole_count


def verify_deepfake_authenticity(image_path: str) -> Tuple[bool, str]:
    """Examines image metadata and Error Level Analysis (ELA) for authenticity."""
    try:
        pil_img = Image.open(image_path)
        exif_data = pil_img._getexif()

        if not exif_data or len(exif_data) == 0:
            pil_img.close()
            return False, "AI-Generated Image Detected (EXIF Metadata Missing)"

        temp_ela_path = image_path + ".ela.tmp.jpg"
        pil_img.save(temp_ela_path, 'JPEG', quality=90)
        resaved_img = Image.open(temp_ela_path)

        ela_diff = ImageChops.difference(pil_img, resaved_img)
        extrema = ela_diff.getextrema()
        max_diff = max([ex[1] for ex in extrema])

        pil_img.close()
        resaved_img.close()
        if os.path.exists(temp_ela_path):
            os.remove(temp_ela_path)

        if max_diff < 12:
            return False, "AI Synthetic Texture Detected (Uniform Compression Rate)"

        cv_img = cv2.imread(image_path)
        if cv_img is not None:
            color_std = np.std(cv_img, axis=(0, 1))
            del cv_img
            if np.mean(color_std) < 8.0:
                return False, "Low-Entropy Artificial Texture Detected"

        return True, "Authentic Hardware Capture Verified"

    except Exception as e:
        return False, f"Authenticity Audit Failed: {str(e)}"


# ================================================================================
#            NEW: EXIF GPS METADATA EXTRACTOR FUNCTION
# ================================================================================
def extract_image_original_gps(image_path: str) -> Tuple[Optional[float], Optional[float]]:
    """
    ছবির এক্সিফ (EXIF) মেটাডাটা থেকে অরিজিনাল GPS Latitude ও Longitude রিড করে।
    """
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        if not exif_data:
            return None, None

        gps_info = {}
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == "GPSInfo":
                for gps_tag_id in value:
                    sub_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                    gps_info[sub_tag] = value[gps_tag_id]

        def convert_to_degrees(value):
            d, m, s = value
            return float(d) + (float(m) / 60.0) + (float(s) / 3600.0)

        if "GPSLatitude" in gps_info and "GPSLongitude" in gps_info:
            lat = convert_to_degrees(gps_info["GPSLatitude"])
            if gps_info.get("GPSLatitudeRef") == "S":
                lat = -lat

            lng = convert_to_degrees(gps_info["GPSLongitude"])
            if gps_info.get("GPSLongitudeRef") == "W":
                lng = -lng

            return round(lat, 5), round(lng, 5)
    except Exception as e:
        logger.warning(f"[EXIF GPS READ WARN] Could not extract GPS: {e}")
    
    return None, None
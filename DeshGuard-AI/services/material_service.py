# services/material_service.py
import cv2
import numpy as np
from ultralytics import YOLO

material_model = YOLO("material_quality.pt")

BANGLA_LABEL_MAP = {
    "crack": "ফাটল (Crack)",
    "crazing": "সূক্ষ্ম ফাটল (Crazing)",
    "inclusion": "অসংলগ্ন মিশ্রণ (Inclusion)",
    "patches": "প্যাচওয়ার্ক (Patches)",
    "pitted_surface": "ক্ষতযুক্ত তল (Pitted Surface)",
    "rolled-in_scale": "স্কেল রোলিং (Rolled-in Scale)",
    "scratches": "স্ক্র্যাচ (Scratches)",
    "stone_3_4": "পাথর অসামঞ্জস্য (Stone Issue)"
}

def audit_material_image(image_path: str) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        return {"defects_found": 0, "details": [], "quality_grade": "N/A", "status_color": "slate", "verdict": "অকার্যকর ছবি", "processed_image_path": ""}

    h, w, _ = img.shape
    results = material_model(image_path, verbose=False)
    detected_defects = []

    # ইমেজের ডাইমেনশন অনুযায়ী ডায়নামিক স্কেলিং
    box_thickness = max(6, int(min(h, w) * 0.010))
    font_scale = max(0.8, min(h, w) * 0.0018)
    font_thickness = max(2, int(font_scale * 2.5))

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            class_name = material_model.names[cls_id]
            confidence = float(box.conf[0])
            
            bangla_name = BANGLA_LABEL_MAP.get(class_name, class_name)
            
            detected_defects.append({
                "class_name": class_name,
                "bangla_name": bangla_name,
                "confidence": round(confidence * 100, 1)
            })

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # 1. বোল্ড অরেঞ্জ বাউন্ডারি বক্স
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 140, 255), box_thickness)

            # 2. ইমেজের ভেতরের লেবেলে শুধুমাত্র ক্লিন ইংলিশ (Clean English Label)
            clean_english_label = f"{class_name.upper()} ({int(confidence*100)}%)"

            (tw, th), baseline = cv2.getTextSize(clean_english_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            
            # ব্যাকগ্রাউন্ড বক্স পজিশনিং
            bg_y1 = max(0, y1 - th - 16)
            cv2.rectangle(img, (x1, bg_y1), (x1 + tw + 16, y1), (0, 140, 255), -1)
            
            # হাই-ভিজিবিলিটি ক্রিস্প টেক্সট
            text_y = max(y1 - 6, th + 2)
            cv2.putText(
                img, clean_english_label, (x1 + 8, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA
            )

    defects_count = len(detected_defects)
    
    if defects_count == 0:
        quality_grade = "A+ (উৎকৃষ্ট মান)"
        status_color = "emerald"
        verdict = "নির্মাণ সামগ্রীর মান সরকারি স্ট্যান্ডার্ড অনুযায়ী সম্পূর্ণ সঠিক।"
    elif defects_count <= 2:
        quality_grade = "B (সতর্কতামূলক)"
        status_color = "amber"
        verdict = "সামান্য ত্রুটি ধরা পড়েছে। ঠিকাদারকে গুণগত মান বজায় রাখতে নির্দেশ দেওয়া যেতে পারে।"
    else:
        quality_grade = "F (নিম্নমানের / বাতিলযোগ্য)"
        status_color = "rose"
        verdict = "গুরুতর মেটেরিয়াল ত্রুটি সনাক্ত হয়েছে! এই লটের অ্যাসফল্ট/পাথর ব্যবহার নিষিদ্ধ করার পরামর্শ দেওয়া হচ্ছে।"

    output_path = image_path + ".material_proc.jpg"
    cv2.imwrite(output_path, img)

    return {
        "defects_found": defects_count,
        "details": detected_defects,
        "quality_grade": quality_grade,
        "status_color": status_color,
        "verdict": verdict,
        "processed_image_path": output_path
    }
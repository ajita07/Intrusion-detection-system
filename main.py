from ultralytics import YOLO
import cv2
from utils import is_person, is_suspicious, draw_box, draw_zone, is_inside_zone, enhance_low_light
from alert import play_alert
import time
import os

# =========================
# CONFIG
# =========================
MODEL_PATH = "yolov8n.pt"
VIDEO_SOURCE = "videos/test.mp4"

ZONE = (200, 150, 450, 350)

ENABLE_ALERT = True
ENABLE_LOW_LIGHT = True

# =========================
# LOAD MODEL
# =========================
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_SOURCE)

# Track previous state (for entry detection)
previous_inside = False

print("System Running... Press ESC to exit")

# Create outputs folder if not exists
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if ENABLE_LOW_LIGHT:
        frame = enhance_low_light(frame)

    draw_zone(frame, ZONE)

    results = model(frame)

    current_inside = False
    suspicious_detected = False

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])

            # Check person in zone
            if is_person(cls_id) and is_inside_zone(box, ZONE):
                current_inside = True
                draw_box(frame, box, "PERSON IN ZONE")

            # Suspicious detection
            if is_suspicious(cls_id):
                suspicious_detected = True
                draw_box(frame, box, "SUSPICIOUS")

    # =========================
    # ENTRY DETECTION LOGIC
    # =========================
    if current_inside and not previous_inside:
        print("🚨 INTRUSION DETECTED!")

        if ENABLE_ALERT:
            play_alert()

        # Save screenshot
        filename = os.path.join(OUTPUT_DIR, f"intrusion_{int(time.time())}.jpg")
        cv2.imwrite(filename, frame)
        print(f"📸 Saved: {filename}")

    # Suspicious object alert
    if suspicious_detected:
        print("⚠️ Suspicious Activity Detected")

    previous_inside = current_inside

    # Show output
    cv2.imshow("Intrusion Detection System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
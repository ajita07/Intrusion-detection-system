import cv2

# COCO class IDs
PERSON_CLASS_ID = 0

# Suspicious classes (you can expand later)
SUSPICIOUS_CLASSES = [0]  # currently person (later add knife/gun if trained)

def is_person(cls_id):
    return cls_id == PERSON_CLASS_ID


def is_suspicious(cls_id):
    return cls_id in SUSPICIOUS_CLASSES


def draw_box(frame, box, label="ALERT"):
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def draw_zone(frame, zone):
    x1, y1, x2, y2 = zone
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(frame, "RESTRICTED AREA", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


def is_inside_zone(box, zone):
    zx1, zy1, zx2, zy2 = zone
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    return not (x2 < zx1 or x1 > zx2 or y2 < zy1 or y1 > zy2)


def enhance_low_light(frame):
    return cv2.convertScaleAbs(frame, alpha=1.5, beta=30)
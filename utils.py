import cv2


def draw_box(frame, x1, y1, x2, y2, label):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame


def process_frame(frame, model):
    results = model(frame)

    result = results[0]   # IMPORTANT FIX

    boxes = result.boxes
    names = model.names   # correct way in YOLOv8

    counts = {
        'car': 0,
        'truck': 0,
        'bus': 0,
        'motorcycle': 0
    }

    for box in boxes:
        cls_id = int(box.cls[0])
        label = names[cls_id]

        if label in counts:
            counts[label] += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            frame = draw_box(frame, x1, y1, x2, y2, label)

    total = sum(counts.values())

    return frame, counts, total


def get_density(counts):
    total = sum(counts.values())

    if total < 5:
        return "Low"
    elif total < 15:
        return "Medium"
    else:
        return "High"
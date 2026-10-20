import cv2
import torch
from ultralytics import YOLO
import time
from typing import Any, cast

print("Loading YOLO...")

# Load YOLO
yolo = YOLO("yolo11n.pt")

print("Loading MiDaS...")

# Load MiDaS
midas = cast(torch.nn.Module, torch.hub.load("intel-isl/MiDaS", "MiDaS_small"))

device = torch.device("cpu")
midas.to(device)
midas.eval()

# MiDaS preprocessing
midas_transforms = cast(Any, torch.hub.load(
    "intel-isl/MiDaS",
    "transforms"
))

transform = midas_transforms.small_transform

print("Models loaded.")
print("Starting camera...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

previous_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    frame_height, frame_width = frame.shape[:2]

    # ============================================
    # YOLO OBJECT DETECTION
    # ============================================

    results = yolo(frame, verbose=False)

    # ============================================
    # MiDaS DEPTH ESTIMATION
    # ============================================

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

    with torch.no_grad():

        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # ============================================
    # PROCESS YOLO DETECTIONS
    # ============================================

    for result in results:

        boxes = cast(Any, result).boxes

        for box in boxes:

            confidence = float(box.conf[0])

            # Ignore weak detections
            if confidence < 0.5:
                continue

            # Bounding box
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # Object name
            class_id = int(box.cls[0])
            object_name = yolo.names[class_id]

            # ====================================
            # GET DEPTH INSIDE OBJECT BOX
            # ====================================

            # Make sure coordinates stay inside image
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame_width, x2)
            y2 = min(frame_height, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # Use the central part of the bounding box
            # to avoid edges/background.
            box_width = x2 - x1
            box_height = y2 - y1

            cx1 = int(x1 + box_width * 0.25)
            cx2 = int(x1 + box_width * 0.75)

            cy1 = int(y1 + box_height * 0.25)
            cy2 = int(y1 + box_height * 0.75)

            object_depth = depth_map[
                cy1:cy2,
                cx1:cx2
            ]

            if object_depth.size == 0:
                continue

            # Average depth
            depth_value = float(object_depth.mean())

            # ====================================
            # POSITION
            # ====================================

            center_x = (x1 + x2) // 2

            if center_x < frame_width / 3:
                position = "LEFT"

            elif center_x > frame_width * 2 / 3:
                position = "RIGHT"

            else:
                position = "CENTER"

            # ====================================
            # PROXIMITY
            # ====================================

            if depth_value > 800:
                proximity = "VERY CLOSE"

            elif depth_value > 500:
                proximity = "NEAR"

            elif depth_value > 200:
                proximity = "MEDIUM"

            else:
                proximity = "FAR"

            # ====================================
            # DISPLAY
            # ====================================

            label = (
                f"{object_name} | "
                f"{position} | "
                f"{proximity} | "
                f"{depth_value:.0f}"
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

    # ============================================
    # FPS
    # ============================================

    current_time = time.time()

    fps = 1 / (current_time - previous_time)

    previous_time = current_time

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "YOLO + MiDaS Depth Test",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

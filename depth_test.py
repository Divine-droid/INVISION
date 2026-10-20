import cv2
import torch
import time
from typing import Any, cast

print("Loading MiDaS...")

model_type = "MiDaS_small"

midas = cast(torch.nn.Module, torch.hub.load("intel-isl/MiDaS", model_type))

device = torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = cast(
    Any,
    torch.hub.load("intel-isl/MiDaS", "transforms"),
)
transform = midas_transforms.small_transform

print("MiDaS loaded.")
print("Starting camera...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

previous_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    # ------------------------------------------------
    # Resize frame before sending it to MiDaS
    # ------------------------------------------------

    small_frame = cv2.resize(frame, (256, 256))

    depth_start = time.time()

    img = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

    with torch.no_grad():

        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(256, 256),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    depth_time = time.time() - depth_start

    # ------------------------------------------------
    # Normalize depth
    # ------------------------------------------------

    depth_min = depth_map.min()
    depth_max = depth_map.max()

    depth_normalized = (depth_map - depth_min) / (
        depth_max - depth_min + 1e-8
    )

    depth_image = (depth_normalized * 255).astype("uint8")

    # Resize depth image for display
    depth_image = cv2.resize(
        depth_image,
        (frame.shape[1], frame.shape[0])
    )

    # ------------------------------------------------
    # FPS
    # ------------------------------------------------

    current_time = time.time()

    fps = 1 / (current_time - previous_time)

    previous_time = current_time

    # ------------------------------------------------
    # Display information
    # ------------------------------------------------

    cv2.putText(
        depth_image,
        f"Depth FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        255,
        2,
    )

    cv2.putText(
        depth_image,
        f"Depth time: {depth_time:.2f}s",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        255,
        2,
    )
    # Get center region of depth map
    h, w = depth_map.shape

    center_y1 = int(h * 0.4)
    center_y2 = int(h * 0.6)

    center_x1 = int(w * 0.4)
    center_x2 = int(w * 0.6)

    center_region = depth_map[
        center_y1:center_y2,
        center_x1:center_x2
    ]

    center_depth = center_region.mean()

    print(f"Center depth: {center_depth:.3f}")
    
    cv2.imshow("Depth Map - 256x256", depth_image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

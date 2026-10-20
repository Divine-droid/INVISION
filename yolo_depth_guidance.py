import cv2  # pyright: ignore[reportMissingImports]
import torch  # pyright: ignore[reportMissingImports]
import time
import threading
import queue
import itertools
import pyttsx3  # pyright: ignore[reportMissingImports]

from ultralytics import YOLO  # pyright: ignore[reportMissingImports]


# ============================================================
# SPEECH SYSTEM
# ============================================================

speech_queue = queue.PriorityQueue()
speech_sequence = itertools.count()


def speech_worker():

    engine = pyttsx3.init()

    # Slightly slower speech for an assistive system.
    engine.setProperty("rate", 165)

    while True:

        priority, sequence, message = speech_queue.get()

        try:
            engine.say(message)
            engine.runAndWait()

        except Exception as error:

            print(
                f"Speech error: {error}"
            )

        finally:

            speech_queue.task_done()


speech_thread = threading.Thread(
    target=speech_worker,
    daemon=True
)

speech_thread.start()


# ============================================================
# ADD SPEECH TO QUEUE
# ============================================================

def speak(
    message,
    priority
):

    sequence = next(
        speech_sequence
    )

    # PriorityQueue returns the smallest value first.
    # Therefore we use negative priority.
    speech_queue.put(
        (
            -priority,
            sequence,
            message
        )
    )


# ============================================================
# LOAD YOLO
# ============================================================

print("Loading YOLO...")

yolo = YOLO(
    "yolo11n.pt"
)


# ============================================================
# LOAD MiDaS
# ============================================================

print("Loading MiDaS...")

midas = torch.hub.load(
    "intel-isl/MiDaS",
    "MiDaS_small"
)

device = torch.device(
    "cpu"
)

midas.to(device)
midas.eval()


midas_transforms = torch.hub.load(
    "intel-isl/MiDaS",
    "transforms"
)

transform = (
    midas_transforms.small_transform
)


print(
    "Models loaded successfully."
)

print(
    "Starting camera..."
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print(
        "ERROR: Could not open camera."
    )

    exit()


# ============================================================
# SETTINGS
# ============================================================

CONFIDENCE_THRESHOLD = 0.5

WALKING_AREA_START = 0.40


PATH_PROXIMITIES = {
    "MEDIUM",
    "NEAR",
    "VERY CLOSE"
}


# ============================================================
# FREE-SPACE THRESHOLDS
# ============================================================

BLOCKED_THRESHOLD = 0.65

LIMITED_THRESHOLD = 0.30


# ============================================================
# VOICE SETTINGS
# ============================================================

# Minimum time before repeating the same guidance.
NORMAL_SPEECH_COOLDOWN = 4.0

# Critical warnings can repeat sooner.
URGENT_SPEECH_COOLDOWN = 2.5


last_spoken_guidance = None

last_spoken_time = 0


# ============================================================
# OBJECT EFFECT LEVELS
# ============================================================

HIGH_EFFECT = {

    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "train"

}


MEDIUM_EFFECT = {

    "person",
    "chair",
    "bench",
    "table",
    "backpack",
    "suitcase",
    "door"

}


LOW_EFFECT = {

    "bottle",
    "cup",
    "book",
    "laptop",
    "keyboard",
    "mouse",
    "cell phone"

}


# ============================================================
# GET OBJECT EFFECT
# ============================================================

def get_effect(
    object_name
):

    if object_name in HIGH_EFFECT:

        return "HIGH"

    elif object_name in MEDIUM_EFFECT:

        return "MEDIUM"

    elif object_name in LOW_EFFECT:

        return "LOW"

    return "MEDIUM"


# ============================================================
# ANALYZE FREE SPACE
# ============================================================

def analyze_free_space(
    detections,
    frame_width,
    frame_height
):

    walking_y_start = int(
        frame_height *
        WALKING_AREA_START
    )


    walking_height = (
        frame_height -
        walking_y_start
    )


    if walking_height <= 0:

        return {

            "LEFT": 0.0,
            "CENTER": 0.0,
            "RIGHT": 0.0

        }


    zone_width = (
        frame_width / 3
    )


    zones = {

        "LEFT": (
            0,
            zone_width
        ),

        "CENTER": (
            zone_width,
            zone_width * 2
        ),

        "RIGHT": (
            zone_width * 2,
            frame_width
        )

    }


    occupancy = {

        "LEFT": 0.0,
        "CENTER": 0.0,
        "RIGHT": 0.0

    }


    # --------------------------------------------------------
    # Focus on the lower portion of the walking area.
    # --------------------------------------------------------

    strip_height = max(
        int(
            walking_height * 0.20
        ),
        1
    )


    strip_y1 = (
        frame_height -
        strip_height
    )


    strip_y2 = frame_height


    # --------------------------------------------------------
    # Process obstacles
    # --------------------------------------------------------

    for detection in detections:

        if (
            detection["proximity"]
            not in PATH_PROXIMITIES
        ):

            continue


        if (
            detection["effect"]
            == "LOW"
        ):

            continue


        x1, y1, x2, y2 = (
            detection["box"]
        )


        # ----------------------------------------------------
        # Bottom 30% of object
        # ----------------------------------------------------

        object_height = (
            y2 - y1
        )


        footprint_y1 = int(
            y2 -
            object_height * 0.30
        )


        footprint_y2 = y2


        # ----------------------------------------------------
        # Intersection with floor strip
        # ----------------------------------------------------

        overlap_y1 = max(
            footprint_y1,
            strip_y1
        )


        overlap_y2 = min(
            footprint_y2,
            strip_y2
        )


        if (
            overlap_y2 <=
            overlap_y1
        ):

            continue


        # ----------------------------------------------------
        # Reduce horizontal footprint
        # ----------------------------------------------------

        object_width = (
            x2 - x1
        )


        footprint_width = (
            object_width * 0.60
        )


        object_center_x = (
            x1 + x2
        ) / 2


        footprint_x1 = (
            object_center_x -
            footprint_width / 2
        )


        footprint_x2 = (
            object_center_x +
            footprint_width / 2
        )


        footprint_x1 = max(
            0,
            footprint_x1
        )


        footprint_x2 = min(
            frame_width,
            footprint_x2
        )


        # ----------------------------------------------------
        # Compare footprint with zones
        # ----------------------------------------------------

        for zone_name, (
            zone_x1,
            zone_x2
        ) in zones.items():

            overlap_x1 = max(
                footprint_x1,
                zone_x1
            )


            overlap_x2 = min(
                footprint_x2,
                zone_x2
            )


            if (
                overlap_x2 <=
                overlap_x1
            ):

                continue


            overlap_width = (
                overlap_x2 -
                overlap_x1
            )


            zone_width_actual = (
                zone_x2 -
                zone_x1
            )


            occupancy_amount = (
                overlap_width /
                zone_width_actual
            )


            # Use the largest obstacle
            # contribution in a zone.
            occupancy[zone_name] = max(
                occupancy[zone_name],
                occupancy_amount
            )


    return occupancy


# ============================================================
# GET SPACE STATUS
# ============================================================

def get_space_status(
    occupancy
):

    status = {}


    for zone, value in (
        occupancy.items()
    ):

        if (
            value >=
            BLOCKED_THRESHOLD
        ):

            status[zone] = "BLOCKED"

        elif (
            value >=
            LIMITED_THRESHOLD
        ):

            status[zone] = "LIMITED"

        else:

            status[zone] = "OPEN"


    return status


# ============================================================
# NAVIGATION DECISION
# ============================================================

def get_navigation_guidance(
    occupancy,
    status
):

    left = occupancy["LEFT"]

    center = occupancy["CENTER"]

    right = occupancy["RIGHT"]


    left_status = status["LEFT"]

    center_status = status["CENTER"]

    right_status = status["RIGHT"]


    # ========================================================
    # CENTER BLOCKED
    # ========================================================

    if center_status == "BLOCKED":

        # ----------------------------------------------------
        # Both sides blocked
        # ----------------------------------------------------

        if (
            left_status == "BLOCKED"
            and
            right_status == "BLOCKED"
        ):

            return (
                "WARNING: Path blocked. STOP."
            )


        # ----------------------------------------------------
        # Left blocked
        # ----------------------------------------------------

        if (
            left_status == "BLOCKED"
            and
            right_status != "BLOCKED"
        ):

            return (
                "Obstacle ahead. "
                "Move right."
            )


        # ----------------------------------------------------
        # Right blocked
        # ----------------------------------------------------

        if (
            right_status == "BLOCKED"
            and
            left_status != "BLOCKED"
        ):

            return (
                "Obstacle ahead. "
                "Move left."
            )


        # ----------------------------------------------------
        # Both sides available.
        # Choose the side with more space.
        # ----------------------------------------------------

        if left < right:

            return (
                "Obstacle ahead. "
                "Move left."
            )

        elif right < left:

            return (
                "Obstacle ahead. "
                "Move right."
            )

        else:

            return (
                "Obstacle ahead. "
                "Move left or right."
            )


    # ========================================================
    # CENTER NOT BLOCKED
    # ========================================================

    if (
        left_status == "BLOCKED"
        and
        right_status == "BLOCKED"
    ):

        return (
            "Sides blocked. "
            "Continue forward."
        )


    if left_status == "BLOCKED":

        return (
            "Obstacle on your left. "
            "Continue forward."
        )


    if right_status == "BLOCKED":

        return (
            "Obstacle on your right. "
            "Continue forward."
        )


    return (
        "Path clear. Continue forward."
    )


# ============================================================
# DETERMINE SPEECH PRIORITY
# ============================================================

def get_speech_priority(
    guidance,
    detections
):

    # --------------------------------------------------------
    # STOP = highest priority
    # --------------------------------------------------------

    if (
        "STOP" in guidance
        or
        "Path blocked" in guidance
    ):

        return 200


    # --------------------------------------------------------
    # Very close obstacle
    # --------------------------------------------------------

    for detection in detections:

        if (
            detection["proximity"]
            == "VERY CLOSE"
            and
            detection["effect"]
            in {"HIGH", "MEDIUM"}
        ):

            if (
                detection["position"]
                == "CENTER"
            ):

                return 180

            return 160


    # --------------------------------------------------------
    # Center obstacle
    # --------------------------------------------------------

    if (
        "Obstacle ahead"
        in guidance
    ):

        return 150


    # --------------------------------------------------------
    # Side obstacle
    # --------------------------------------------------------

    if (
        "Obstacle on your"
        in guidance
    ):

        return 120


    # --------------------------------------------------------
    # Continue forward
    # --------------------------------------------------------

    if (
        "Continue forward"
        in guidance
    ):

        return 80


    # --------------------------------------------------------
    # Path clear
    # --------------------------------------------------------

    return 40


# ============================================================
# SPEECH CONTROLLER
# ============================================================

def process_speech(
    guidance,
    detections
):

    global last_spoken_guidance
    global last_spoken_time


    current_time = time.time()


    priority = get_speech_priority(
        guidance,
        detections
    )


    # --------------------------------------------------------
    # Determine cooldown
    # --------------------------------------------------------

    if priority >= 160:

        cooldown = (
            URGENT_SPEECH_COOLDOWN
        )

    else:

        cooldown = (
            NORMAL_SPEECH_COOLDOWN
        )


    # --------------------------------------------------------
    # Determine whether guidance changed
    # --------------------------------------------------------

    guidance_changed = (
        guidance !=
        last_spoken_guidance
    )


    time_since_last_speech = (
        current_time -
        last_spoken_time
    )


    # --------------------------------------------------------
    # Speak when:
    #
    # 1. Guidance changed
    # OR
    # 2. Enough time has passed
    #
    # But avoid constantly repeating "Path clear."
    # --------------------------------------------------------

    should_speak = False


    if guidance_changed:

        should_speak = True

    elif (
        priority >= 150
        and
        time_since_last_speech >= cooldown
    ):

        should_speak = True


    # --------------------------------------------------------
    # Speak
    # --------------------------------------------------------

    if should_speak:

        speak(
            guidance,
            priority
        )

        last_spoken_guidance = (
            guidance
        )

        last_spoken_time = (
            current_time
        )


# ============================================================
# MAIN LOOP
# ============================================================

previous_time = time.time()


while True:

    ret, frame = (
        cap.read()
    )


    if not ret:

        print(
            "ERROR: Could not read camera frame."
        )

        break


    frame_height, frame_width = (
        frame.shape[:2]
    )


    # ========================================================
    # YOLO
    # ========================================================

    results = yolo(
        frame,
        verbose=False
    )


    # ========================================================
    # MiDaS
    # ========================================================

    img = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    input_batch = (
        transform(img)
        .to(device)
    )


    with torch.no_grad():

        prediction = (
            midas(input_batch)
        )


        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False
            )
            .squeeze()
        )


    depth_map = (
        prediction.cpu().numpy()
    )


    # ========================================================
    # DETECTIONS
    # ========================================================

    detections = []


    for result in results:

        for box in result.boxes:

            confidence = float(
                box.conf[0]
            )


            if (
                confidence <
                CONFIDENCE_THRESHOLD
            ):

                continue


            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            x1 = max(
                0,
                x1
            )

            y1 = max(
                0,
                y1
            )

            x2 = min(
                frame_width,
                x2
            )

            y2 = min(
                frame_height,
                y2
            )


            if (
                x2 <= x1
                or
                y2 <= y1
            ):

                continue


            # ------------------------------------------------
            # Object
            # ------------------------------------------------

            class_id = int(
                box.cls[0]
            )


            object_name = (
                yolo.names[class_id]
            )


            # ------------------------------------------------
            # Position
            # ------------------------------------------------

            center_x = (
                x1 + x2
            ) // 2


            if (
                center_x <
                frame_width / 3
            ):

                position = "LEFT"

            elif (
                center_x >
                frame_width * 2 / 3
            ):

                position = "RIGHT"

            else:

                position = "CENTER"


            # ------------------------------------------------
            # Depth region
            # ------------------------------------------------

            box_width = (
                x2 - x1
            )

            box_height = (
                y2 - y1
            )


            depth_x1 = int(
                x1 +
                box_width * 0.25
            )


            depth_x2 = int(
                x1 +
                box_width * 0.75
            )


            depth_y1 = int(
                y1 +
                box_height * 0.25
            )


            depth_y2 = int(
                y1 +
                box_height * 0.75
            )


            object_depth = (
                depth_map[
                    depth_y1:depth_y2,
                    depth_x1:depth_x2
                ]
            )


            if (
                object_depth.size == 0
            ):

                continue


            depth_value = float(
                object_depth.mean()
            )


            # ------------------------------------------------
            # Proximity
            # ------------------------------------------------

            if depth_value > 800:

                proximity = (
                    "VERY CLOSE"
                )

            elif depth_value > 500:

                proximity = (
                    "NEAR"
                )

            elif depth_value > 200:

                proximity = (
                    "MEDIUM"
                )

            else:

                proximity = (
                    "FAR"
                )


            # ------------------------------------------------
            # Effect
            # ------------------------------------------------

            effect = get_effect(
                object_name
            )


            # ------------------------------------------------
            # Store detection
            # ------------------------------------------------

            detections.append({

                "name":
                    object_name,

                "confidence":
                    confidence,

                "position":
                    position,

                "depth":
                    depth_value,

                "proximity":
                    proximity,

                "effect":
                    effect,

                "box":
                    (
                        x1,
                        y1,
                        x2,
                        y2
                    )

            })


    # ========================================================
    # FREE SPACE
    # ========================================================

    occupancy = (
        analyze_free_space(
            detections,
            frame_width,
            frame_height
        )
    )


    # ========================================================
    # SPACE STATUS
    # ========================================================

    status = (
        get_space_status(
            occupancy
        )
    )


    # ========================================================
    # NAVIGATION
    # ========================================================

    guidance = (
        get_navigation_guidance(
            occupancy,
            status
        )
    )


    # ========================================================
    # VOICE GUIDANCE
    # ========================================================

    process_speech(
        guidance,
        detections
    )


    # ========================================================
    # DRAW OBJECTS
    # ========================================================

    for detection in detections:

        x1, y1, x2, y2 = (
            detection["box"]
        )


        label = (
            f'{detection["name"]} | '
            f'{detection["position"]} | '
            f'{detection["proximity"]} | '
            f'{detection["depth"]:.0f}'
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
            (
                x1,
                max(
                    y1 - 10,
                    20
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


    # ========================================================
    # WALKING AREA
    # ========================================================

    walking_y = int(
        frame_height *
        WALKING_AREA_START
    )


    cv2.line(
        frame,
        (0, walking_y),
        (
            frame_width,
            walking_y
        ),
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "WALKING AREA",
        (
            10,
            walking_y - 10
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ========================================================
    # FREE SPACE DISPLAY
    # ========================================================

    left_text = (
        f'LEFT: '
        f'{status["LEFT"]} '
        f'({occupancy["LEFT"] * 100:.0f}%)'
    )


    center_text = (
        f'CENTER: '
        f'{status["CENTER"]} '
        f'({occupancy["CENTER"] * 100:.0f}%)'
    )


    right_text = (
        f'RIGHT: '
        f'{status["RIGHT"]} '
        f'({occupancy["RIGHT"] * 100:.0f}%)'
    )


    cv2.putText(
        frame,
        left_text,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        center_text,
        (10, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        right_text,
        (10, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # ========================================================
    # GUIDANCE
    # ========================================================

    cv2.rectangle(
        frame,
        (
            0,
            frame_height - 70
        ),
        (
            frame_width,
            frame_height
        ),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        f"GUIDANCE: {guidance}",
        (
            10,
            frame_height - 25
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ========================================================
    # FPS
    # ========================================================

    current_time = time.time()


    fps = 1 / (
        current_time -
        previous_time
    )


    previous_time = (
        current_time
    )


    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (
            10,
            30
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "YOLO + MiDaS + Voice Guidance",
        frame
    )


    # ========================================================
    # EXIT
    # ========================================================

    if (
        cv2.waitKey(1) & 0xFF
        == ord("q")
    ):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()
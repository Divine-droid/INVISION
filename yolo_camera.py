from ultralytics import YOLO
import cv2
import pyttsx3
import time
import threading
import queue
import itertools
from typing import cast

from ultralytics.engine.results import Results


# ============================================================
# 1. LOAD YOLO MODEL
# ============================================================

model = YOLO("yolo11n.pt")


# ============================================================
# 2. TEXT-TO-SPEECH SYSTEM
# ============================================================

engine = cast(pyttsx3.Engine, pyttsx3.init())

speech_queue = queue.PriorityQueue()
_seq_counter = itertools.count()


def speech_worker():
    while True:
        neg_score, _, message = speech_queue.get()

        try:
            engine.say(message)
            engine.runAndWait()

        finally:
            speech_queue.task_done()


threading.Thread(
    target=speech_worker,
    daemon=True
).start()


def enqueue_speech(message, score):
    speech_queue.put(
        (-score, next(_seq_counter), message)
    )


# ============================================================
# 3. CAMERA
# ============================================================

cap = cv2.VideoCapture(0)


# ============================================================
# 4. OBJECT EFFECT LEVELS
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


def get_effect_level(object_name):

    if object_name in HIGH_EFFECT:
        return "HIGH"

    elif object_name in MEDIUM_EFFECT:
        return "MEDIUM"

    else:
        return "LOW"


# ============================================================
# 5. PROXIMITY DETECTION
# ============================================================

def get_proximity(
    box_area,
    frame_area,
    previous_level=None
):

    ratio = box_area / frame_area

    thresholds = [
        (0.25, "VERY CLOSE"),
        (0.10, "NEAR"),
        (0.03, "MEDIUM")
    ]

    # Hysteresis prevents rapid switching
    margin = 0.02

    for cutoff, level in thresholds:

        adjusted_cutoff = cutoff

        if previous_level == level:
            adjusted_cutoff = cutoff - margin

        elif previous_level is not None:
            adjusted_cutoff = cutoff + margin

        if ratio > adjusted_cutoff:
            return level

    return "FAR"


# ============================================================
# 6. PRIORITY SYSTEM
# ============================================================

EFFECT_BASE = {
    "HIGH": 100,
    "MEDIUM": 50,
    "LOW": 10
}

PROXIMITY_BONUS = {
    "VERY CLOSE": 30,
    "NEAR": 20,
    "MEDIUM": 10,
    "FAR": 0
}


def priority_score(
    effect,
    proximity,
    confidence
):

    return (
        EFFECT_BASE[effect]
        + PROXIMITY_BONUS[proximity]
        + confidence
    )


# ============================================================
# 7. SPEECH MEMORY
# ============================================================

last_spoken_state = {}

cooldown = 4


# ============================================================
# 8. FREE-SPACE ANALYZER SETTINGS
# ============================================================

# The lower portion of the camera is treated as the
# approximate walking area.

WALKING_AREA_START = 0.40


# How much of a zone can be occupied before
# we consider it limited or blocked.

BLOCKED_THRESHOLD = 0.45
LIMITED_THRESHOLD = 0.20


# Objects that should influence free-space analysis.

PATH_EFFECTS = {
    "HIGH",
    "MEDIUM"
}


# ============================================================
# 9. FREE-SPACE ANALYZER
# ============================================================

def analyze_free_space(
    detections,
    frame_width,
    frame_height
):
    """
    Estimate how much of the walking area is occupied
    by detected obstacles.

    Returns:

        left_status
        center_status
        right_status

        left_occupancy
        center_occupancy
        right_occupancy
    """

    walking_start = int(
        frame_height * WALKING_AREA_START
    )

    walking_height = frame_height - walking_start

    zone_width = frame_width // 3

    # Total obstacle area in each zone
    obstacle_area = {
        "LEFT": 0,
        "CENTER": 0,
        "RIGHT": 0
    }

    # Area of each walking zone
    zone_area = {
        "LEFT": zone_width * walking_height,
        "CENTER": zone_width * walking_height,
        "RIGHT": (
            (frame_width - zone_width * 2)
            * walking_height
        )
    }

    for detection in detections:

        # Ignore low-effect objects
        if detection["effect"] not in PATH_EFFECTS:
            continue

        x1 = detection["x1"]
        y1 = detection["y1"]
        x2 = detection["x2"]
        y2 = detection["y2"]

        # Ignore objects that don't reach
        # into the walking region.
        if y2 < walking_start:
            continue

        position = detection["position"]

        # Calculate the part of the bounding box
        # that overlaps the walking area.

        overlap_y1 = max(
            y1,
            walking_start
        )

        overlap_y2 = min(
            y2,
            frame_height
        )

        overlap_height = (
            overlap_y2 - overlap_y1
        )

        if overlap_height <= 0:
            continue

        overlap_width = x2 - x1

        if overlap_width <= 0:
            continue

        # ----------------------------------------------------
        # IMPORTANT:
        # A detection can cross zone boundaries.
        # Calculate its overlap with each zone.
        # ----------------------------------------------------

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

        for zone_name, (zone_x1, zone_x2) in zones.items():

            overlap_x1 = max(
                x1,
                zone_x1
            )

            overlap_x2 = min(
                x2,
                zone_x2
            )

            overlap_width = (
                overlap_x2 - overlap_x1
            )

            if overlap_width <= 0:
                continue

            occupied = (
                overlap_width
                * overlap_height
            )

            obstacle_area[zone_name] += occupied

    # --------------------------------------------------------
    # Calculate occupancy percentage
    # --------------------------------------------------------

    occupancy = {}

    for zone in [
        "LEFT",
        "CENTER",
        "RIGHT"
    ]:

        occupancy[zone] = (
            obstacle_area[zone]
            / zone_area[zone]
        )

        # Prevent values greater than 100%
        occupancy[zone] = min(
            occupancy[zone],
            1.0
        )

    # --------------------------------------------------------
    # Convert occupancy into status
    # --------------------------------------------------------

    status = {}

    for zone in [
        "LEFT",
        "CENTER",
        "RIGHT"
    ]:

        value = occupancy[zone]

        if value >= BLOCKED_THRESHOLD:

            status[zone] = "BLOCKED"

        elif value >= LIMITED_THRESHOLD:

            status[zone] = "LIMITED"

        else:

            status[zone] = "OPEN"

    return (
        status["LEFT"],
        status["CENTER"],
        status["RIGHT"],

        occupancy["LEFT"],
        occupancy["CENTER"],
        occupancy["RIGHT"]
    )


# ============================================================
# 10. FREE-SPACE DIRECTION GUIDANCE
# ============================================================

def get_free_space_guidance(
    left_status,
    center_status,
    right_status,

    left_occupancy,
    center_occupancy,
    right_occupancy
):

    # --------------------------------------------------------
    # CASE 1:
    # Everything is open
    # --------------------------------------------------------

    if (
        left_status == "OPEN"
        and center_status == "OPEN"
        and right_status == "OPEN"
    ):

        return "Path clear. Continue forward."


    # --------------------------------------------------------
    # CASE 2:
    # Center blocked
    # --------------------------------------------------------

    if center_status == "BLOCKED":

        # Both sides blocked
        if (
            left_status == "BLOCKED"
            and right_status == "BLOCKED"
        ):

            return "Warning. Path blocked. Stop."


        # Left blocked, right available
        if left_status == "BLOCKED":

            return "Obstacle ahead. Move right."


        # Right blocked, left available
        if right_status == "BLOCKED":

            return "Obstacle ahead. Move left."


        # Both sides available
        # Choose the side with more free space.

        if left_occupancy < right_occupancy:

            return "Obstacle ahead. Move left."

        else:

            return "Obstacle ahead. Move right."


    # --------------------------------------------------------
    # CASE 3:
    # Center LIMITED
    # --------------------------------------------------------

    if center_status == "LIMITED":

        # If one side is clearly more open,
        # recommend that direction.

        if (
            left_status == "OPEN"
            and right_status != "OPEN"
        ):

            return "Limited space ahead. Move left."


        if (
            right_status == "OPEN"
            and left_status != "OPEN"
        ):

            return "Limited space ahead. Move right."


        # Both sides open.
        # Choose the side with more free space.

        if (
            left_status == "OPEN"
            and right_status == "OPEN"
        ):

            if left_occupancy < right_occupancy:

                return "Limited space ahead. Move left."

            else:

                return "Limited space ahead. Move right."


        return "Limited space ahead. Proceed carefully."


    # --------------------------------------------------------
    # CASE 4:
    # Center is OPEN
    # --------------------------------------------------------

    if center_status == "OPEN":

        # Left blocked
        if (
            left_status == "BLOCKED"
            and right_status != "BLOCKED"
        ):

            return "Obstacle on your left. Continue forward."


        # Right blocked
        if (
            right_status == "BLOCKED"
            and left_status != "BLOCKED"
        ):

            return "Obstacle on your right. Continue forward."


        # Both sides blocked
        if (
            left_status == "BLOCKED"
            and right_status == "BLOCKED"
        ):

            return "Sides blocked. Continue forward."


    return None


# ============================================================
# 11. PATH SPEECH MEMORY
# ============================================================

last_path_state = None
last_path_spoken = 0

path_cooldown = 4


# ============================================================
# 12. MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("Could not access camera")

        break


    # --------------------------------------------------------
    # Frame information
    # --------------------------------------------------------

    height, width, _ = frame.shape

    frame_area = width * height


    # --------------------------------------------------------
    # Zone boundaries
    # --------------------------------------------------------

    left_boundary = width // 3

    right_boundary = (
        width * 2
    ) // 3


    # --------------------------------------------------------
    # Draw zone boundaries
    # --------------------------------------------------------

    cv2.line(
        frame,
        (left_boundary, 0),
        (left_boundary, height),
        (255, 255, 255),
        2
    )

    cv2.line(
        frame,
        (right_boundary, 0),
        (right_boundary, height),
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # Draw walking area boundary
    # --------------------------------------------------------

    walking_start = int(
        height * WALKING_AREA_START
    )

    cv2.line(
        frame,
        (0, walking_start),
        (width, walking_start),
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "WALKING AREA",
        (10, walking_start - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2
    )


    # ========================================================
    # YOLO DETECTION
    # ========================================================

    results = cast(
        list[Results],
        model(
            frame,
            verbose=False
        )
    )


    detections = []


    for result in results:

        boxes = result.boxes

        if boxes is None:
            continue


        for box in boxes:

            confidence = float(
                box.conf[0]
            )


            # Ignore low-confidence detections
            if confidence < 0.5:
                continue


            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # ------------------------------------------------
            # Object class
            # ------------------------------------------------

            class_id = int(
                box.cls[0]
            )

            object_name = model.names[
                class_id
            ]


            # ------------------------------------------------
            # Object center
            # ------------------------------------------------

            center_x = (
                x1 + x2
            ) // 2

            center_y = (
                y1 + y2
            ) // 2


            # ------------------------------------------------
            # Position
            # ------------------------------------------------

            if center_x < left_boundary:

                position = "LEFT"

            elif center_x < right_boundary:

                position = "CENTER"

            else:

                position = "RIGHT"


            # ------------------------------------------------
            # Bounding box area
            # ------------------------------------------------

            box_area = (
                x2 - x1
            ) * (
                y2 - y1
            )


            # ------------------------------------------------
            # Detection key
            # ------------------------------------------------

            detection_key = (
                f"{object_name}_{position}"
            )


            # ------------------------------------------------
            # Previous proximity
            # ------------------------------------------------

            previous_level = (
                last_spoken_state
                .get(
                    detection_key,
                    {}
                )
                .get(
                    "proximity"
                )
            )


            # ------------------------------------------------
            # Proximity
            # ------------------------------------------------

            proximity = get_proximity(
                box_area,
                frame_area,
                previous_level
            )


            # ------------------------------------------------
            # Effect
            # ------------------------------------------------

            effect = get_effect_level(
                object_name
            )


            # ------------------------------------------------
            # Priority
            # ------------------------------------------------

            score = priority_score(
                effect,
                proximity,
                confidence
            )


            # ------------------------------------------------
            # Store detection
            # ------------------------------------------------

            detections.append({

                "object": object_name,

                "position": position,

                "proximity": proximity,

                "effect": effect,

                "confidence": confidence,

                "score": score,

                "detection_key": detection_key,

                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,

                "center_x": center_x,
                "center_y": center_y
            })


            # =================================================
            # DRAW DETECTION
            # =================================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )


            label = (
                f"{object_name} | "
                f"{position} | "
                f"{proximity} | "
                f"{effect}"
            )


            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(y1 - 10, 20)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )


    # ========================================================
    # SORT DETECTIONS BY PRIORITY
    # ========================================================

    current_time = time.time()

    detections.sort(
        key=lambda d: d["score"],
        reverse=True
    )


    # ========================================================
    # OBJECT SPEECH
    # ========================================================

    for detection in detections:

        # Don't speak about low-effect objects
        if detection["effect"] == "LOW":
            continue


        key = detection[
            "detection_key"
        ]

        proximity = detection[
            "proximity"
        ]


        state = last_spoken_state.get(
            key
        )


        level_changed = (
            state is not None
            and state["proximity"]
            != proximity
        )


        cooldown_passed = (
            state is not None
            and current_time
            - state["last_spoken"]
            > cooldown
        )


        is_new = (
            state is None
        )


        if (
            is_new
            or level_changed
            or cooldown_passed
        ):

            object_name = detection[
                "object"
            ]

            position = detection[
                "position"
            ]

            effect = detection[
                "effect"
            ]


            if (
                effect == "HIGH"
                and proximity
                in [
                    "VERY CLOSE",
                    "NEAR"
                ]
            ):

                message = (
                    f"Warning. "
                    f"{object_name} "
                    f"{proximity.lower()} "
                    f"on your "
                    f"{position.lower()}."
                )

            else:

                message = (
                    f"{object_name} "
                    f"{proximity.lower()} "
                    f"on your "
                    f"{position.lower()}."
                )


            enqueue_speech(
                message,
                detection["score"]
            )


            last_spoken_state[key] = {

                "proximity": proximity,

                "last_spoken":
                    current_time
            }


    # ========================================================
    # FREE-SPACE ANALYSIS
    # ========================================================

    (
        left_status,
        center_status,
        right_status,

        left_occupancy,
        center_occupancy,
        right_occupancy

    ) = analyze_free_space(
        detections,
        width,
        height
    )


    # ========================================================
    # DIRECTION GUIDANCE
    # ========================================================

    guidance = get_free_space_guidance(

        left_status,
        center_status,
        right_status,

        left_occupancy,
        center_occupancy,
        right_occupancy
    )


    # ========================================================
    # PATH STATE
    # ========================================================

    path_state = (

        left_status,

        center_status,

        right_status
    )


    # ========================================================
    # PATH STATUS DISPLAY
    # ========================================================

    if center_status == "BLOCKED":

        path_status = "PATH: BLOCKED"

    elif center_status == "LIMITED":

        path_status = "PATH: LIMITED"

    else:

        path_status = "PATH: CLEAR"


    cv2.putText(
        frame,
        path_status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )


    # ========================================================
    # DISPLAY FREE-SPACE STATUS
    # ========================================================

    left_text = (
        f"LEFT: {left_status} "
        f"({left_occupancy * 100:.0f}%)"
    )

    center_text = (
        f"CENTER: {center_status} "
        f"({center_occupancy * 100:.0f}%)"
    )

    right_text = (
        f"RIGHT: {right_status} "
        f"({right_occupancy * 100:.0f}%)"
    )


    # --------------------------------------------------------
    # Left
    # --------------------------------------------------------

    left_display_color = (
        (0, 255, 0)
        if left_status == "OPEN"
        else
        (0, 255, 255)
        if left_status == "LIMITED"
        else
        (0, 0, 255)
    )


    cv2.putText(
        frame,
        left_text,
        (20, height - 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        left_display_color,
        2
    )


    # --------------------------------------------------------
    # Center
    # --------------------------------------------------------

    center_display_color = (
        (0, 255, 0)
        if center_status == "OPEN"
        else
        (0, 255, 255)
        if center_status == "LIMITED"
        else
        (0, 0, 255)
    )


    cv2.putText(
        frame,
        center_text,
        (20, height - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        center_display_color,
        2
    )


    # --------------------------------------------------------
    # Right
    # --------------------------------------------------------

    right_display_color = (
        (0, 255, 0)
        if right_status == "OPEN"
        else
        (0, 255, 255)
        if right_status == "LIMITED"
        else
        (0, 0, 255)
    )


    _ = cv2.putText(
        frame,
        right_text,
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        right_display_color,
        2
    )


    # ========================================================
    # SPEAK PATH GUIDANCE
    # ========================================================

    if guidance is not None:

        if (
            path_state != last_path_state
            or
            current_time
            - last_path_spoken
            > path_cooldown
        ):

            # Path guidance gets very high priority
            enqueue_speech(
                guidance,
                150
            )


            last_path_state = (
                path_state
            )

            last_path_spoken = (
                current_time
            )


    # ========================================================
    # SHOW CAMERA
    # ========================================================

    cv2.imshow(
        "AI Vision Assistant",
        frame
    )


    # ========================================================
    # QUIT WITH Q
    # ========================================================

    if (
        cv2.waitKey(1)
        & 0xFF
        == ord("q")
    ):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()
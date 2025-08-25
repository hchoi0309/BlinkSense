"""
Webcam + Landmark Probe script

What this script does:
- Opens webcam at a configurable resolution (default 640x360).
- Runs MediaPipe Face Mesh (with refine_landmarks=True) to get stable eye & iris points.
- Computes EAR (Eye Aspect Ratio) per frame for left/right eyes and their average.
- Smooths the EAR with an EMA (exponential moving average) to reduce jitter.
- Displays FPS, EAR values, a "closed frames" counter, and an approximate closed% over a short window.
- Press 'b' to set a per-user open-eye EAR baseline (for adaptive thresholds).
- Quit with 'q'.

"""

import cv2
import json
import math
import os
import time
from collections import deque
from typing import List, Tuple

import numpy as np

try:
    import mediapipe as mp
except ImportError:
    raise SystemExit(
        "MediaPipe not installed. Run: pip install mediapipe opencv-python numpy"
    )

# Configuration handling

# Keep internal defaults so the script still runs even if configs/default.json is missing, misnamed, or partially specified
DEFAULT_CFG = {
    "fps": 30,
    "frame_width": 640,
    "frame_height": 360,

    "eye_img_size": 64,  # used later in the project; harmless to keep here

    "mediapipe": {
        "max_num_faces": 1,
        "refine_landmarks": True,           # enables iris/eyelid refinement
        "min_detection_confidence": 0.5,
        "min_tracking_confidence": 0.5
    },

    "ear_thresh": 0.20,     # starting, global threshold (you can adapt this per user below)
    "perclos_thresh": 0.15, # not used for alerts here; just displayed as a sanity check

    "t_window": 64,         # short rolling window length (frames) for quick closed percentage display
    "stride_frames": 8,

    "alert": {              # kept for parity with later stages; not used here
        "prob_threshold": 0.6,
        "hysteresis_on": 3,
        "hysteresis_off": 4,
        "cooldown_sec": 10
    }
}

def load_config(path: str = "configs/default.json") -> dict:
    """
    Load configs/default.json if present and shallow-merge with DEFAULT_CFG.
    Allows overriding specific keys without rewriting everything.
    """
    cfg = DEFAULT_CFG.copy()
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                user_cfg = json.load(f)
            # Shallow merge (good enough for this probe)
            for k, v in user_cfg.items():
                if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
                    cfg[k].update(v)
                else:
                    cfg[k] = v
            print(f"[probe] Loaded config from {path}")
        except Exception as e:
            print(f"[probe] Failed to load {path}: {e}. Using internal defaults.")
    else:
        print("[probe] No configs/default.json found; using internal defaults.")
    return cfg

CFG = load_config()


# MediaPipe setup

mp_face_mesh = mp.solutions.face_mesh

FACE_MESH_KW = dict(
    static_image_mode=False,  # streaming/video mode
    max_num_faces=CFG["mediapipe"]["max_num_faces"],
    refine_landmarks=CFG["mediapipe"]["refine_landmarks"],
    min_detection_confidence=CFG["mediapipe"]["min_detection_confidence"],
    min_tracking_confidence=CFG["mediapipe"]["min_tracking_confidence"]
)

# MediaPipe Face Mesh (468 points) eye indices for EAR (classic 6-point scheme)
# Order: [p1, p2, p3, p4, p5, p6] mapping to the standard EAR formula below.
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# EAR utilities

def _euclidean(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    """Euclidean distance between two (x,y) points."""
    return math.hypot(p[0] - q[0], p[1] - q[1])

def eye_aspect_ratio(eye_pts: List[Tuple[float, float]]) -> float:
    """
    Compute EAR given 6 eye points in order:
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

    Intuition: vertical lid distances shrink when eyes close (numerator),
    while horizontal eye width (denominator) stays roughly constant.
    """
    p1, p2, p3, p4, p5, p6 = eye_pts
    A = _euclidean(p2, p6)
    B = _euclidean(p3, p5)
    C = _euclidean(p1, p4)
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

def landmarks_to_xy(landmarks, w: int, h: int, idxs: List[int]) -> List[Tuple[float, float]]:
    """Convert normalized Face Mesh landmarks to pixel coordinates for given indices."""
    pts = []
    for i in idxs:
        li = landmarks[i]
        pts.append((li.x * w, li.y * h))
    return pts

# Helper functions for: smoothing, FPS, drawing

class EMA:
    """Simple exponential moving average for smoothing jittery signals like EAR."""
    def __init__(self, alpha: float = 0.2, init=None):
        self.alpha = alpha
        self.value = init

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value

class FPSMeter:
    """EMA-smoothed FPS meter."""
    def __init__(self, alpha: float = 0.9):
        self.last_t = None
        self.ema = EMA(alpha=alpha, init=None)

    def tick(self):
        now = time.time()
        if self.last_t is None:
            self.last_t = now
            return None
        dt = now - self.last_t
        self.last_t = now
        if dt <= 0:
            return None
        fps = 1.0 / dt
        return self.ema.update(fps)

def draw_eye_poly(frame, pts, color=(0, 255, 0)):
    """Draw a small polygon around the eye using the six EAR points."""
    pts_int = np.array([[int(x), int(y)] for (x, y) in pts], dtype=np.int32)
    cv2.polylines(frame, [pts_int], isClosed=True, color=color, thickness=1)
    for (x, y) in pts_int:
        cv2.circle(frame, (x, y), 1, color, -1)

def put_text(frame, text, org, color=(0, 255, 255), scale=0.5, thickness=1):
    """Outlined text for readability on any background."""
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

# Main loop

def main(camera_index: int = 0):
    ear_thresh = CFG["ear_thresh"]     # global threshold (can be adapted per user below)
    frame_w = CFG["frame_width"]
    frame_h = CFG["frame_height"]
    roll_len = CFG["t_window"]         # short rolling window for quick closed% display

    # Open camera (DirectShow on Windows reduces startup lag; harmless elsewhere)
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if os.name == "nt" else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)

    fps_meter = FPSMeter(alpha=0.9)
    sm_ear = EMA(alpha=0.2)            # smoothed EAR for nicer visualization
    closed_consec = 0                  # consecutive frames below threshold
    closed_flags = deque(maxlen=roll_len)  # 0/1 flags for "closed" to display quick closed%

    baseline_open_ear = None           # set by pressing 'b' (per-user calibration)

    with mp_face_mesh.FaceMesh(**FACE_MESH_KW) as fm:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[probe] Failed to read frame from camera.")
                break

            # Mirror for "selfie view"
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run the landmark model
            res = fm.process(rgb)
            h, w = frame.shape[:2]

            ear_left = ear_right = None
            face_ok = False

            if res.multi_face_landmarks:
                face_ok = True
                lms = res.multi_face_landmarks[0].landmark

                # Convert normalized landmarks to pixel points
                left_pts = landmarks_to_xy(lms, w, h, LEFT_EYE)
                right_pts = landmarks_to_xy(lms, w, h, RIGHT_EYE)

                # Compute EAR for both eyes and average
                ear_left = eye_aspect_ratio(left_pts)
                ear_right = eye_aspect_ratio(right_pts)
                ear = (ear_left + ear_right) / 2.0

                # Visualize eye polygons
                draw_eye_poly(frame, left_pts, color=(0, 255, 0))
                draw_eye_poly(frame, right_pts, color=(0, 255, 0))

                # Smooth the EAR a bit to remove jitter
                ear_s = sm_ear.update(ear)

                # Classify "closed" by threshold (for visualization only here)
                is_closed = 1 if ear < ear_thresh else 0
                closed_flags.append(is_closed)
                if is_closed:
                    closed_consec += 1
                else:
                    closed_consec = 0

                # Optional: per-user adaptive threshold
                if baseline_open_ear is not None:
                    adapted_thresh = 0.75 * baseline_open_ear
                else:
                    adapted_thresh = ear_thresh

                # On-screen telemetry
                put_text(frame, f"EAR L: {ear_left:.3f}  R: {ear_right:.3f}", (10, 22))
                put_text(frame, f"EAR avg: {ear:.3f}  smoothed: {ear_s:.3f}", (10, 42))
                put_text(frame, f"Thresh: {ear_thresh:.3f} (adapt: {adapted_thresh:.3f})", (10, 62))
                put_text(frame, f"Closed frames (consec): {closed_consec}", (10, 82))
                if len(closed_flags) > 0:
                    closed_pct = 100 * sum(closed_flags) / len(closed_flags)
                    put_text(frame, f"~Closed% (last {len(closed_flags)} fr): {closed_pct:.1f}%", (10, 102))

            else:
                # No face this frame; reset counters gently
                sm_ear.update(0.0)
                closed_consec = 0
                closed_flags.append(0)
                put_text(frame, "No face detected", (10, 22), color=(0, 180, 255))

            # FPS overlay (smoothed)
            fps = fps_meter.tick()
            if fps is not None:
                put_text(frame, f"FPS: {fps:.1f}", (w - 140, 22), color=(0, 255, 0))

            # Help line + controls
            put_text(frame, "Keys: [b]=baseline open EAR  [q]=quit", (10, h - 12), color=(200, 200, 255))

            # Show window and process keys
            cv2.imshow("MediaPipe EAR Probe", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # quit
                break
            elif key == ord('b'):  # set baseline to current smoothed EAR
                if face_ok and sm_ear.value is not None:
                    baseline_open_ear = float(sm_ear.value)
                    print(f"[probe] Baseline open EAR set to {baseline_open_ear:.3f} (press 'b' again to update)")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

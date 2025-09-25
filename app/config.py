"""
Configuration settings for BlinkSense FastAPI application
"""
import os
from pathlib import Path
from typing import List

# Application settings
APP_NAME = "BlinkSense"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Real-time drowsiness detection system using computer vision and machine learning"

# Server settings
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", 8000))
DEBUG = os.environ.get("DEBUG", "True").lower() == "true"

# CORS settings
ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://blinksense.onrender.com",
    "https://*.railway.app",
    "https://*.herokuapp.com",
    "https://*.onrender.com"
]

# File paths
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
MODELS_DIR = BASE_DIR / "models"

# Model settings
MODEL_PATH = MODELS_DIR / "eye" / "0.1.0" / "weights.pt"
MODEL_INPUT_SIZE = 64  # 64x64 input images
MODEL_CLASSES = 2  # Open/Closed
MODEL_EMBEDDING_DIM = 64

# Detection settings
class DetectionConfig:
    # Frame processing
    FPS = 15
    JPEG_QUALITY = 0.7

    # CNN thresholds
    P_OPEN_THRESHOLD = 0.3  # Conservative threshold for CNN

    # Alert thresholds
    CLOSED_STREAK_THRESHOLD = 5.0  # seconds
    PERCLOS_THRESHOLD = 0.20  # 20%

    # Temporal windows
    PERCLOS_WINDOW = 30  # seconds
    COOLDOWN_DURATION = 30.0  # seconds
    MIN_BLINK_DURATION = 2.0  # seconds

    # Calibration
    CALIBRATION_SAMPLES = 15  # seconds worth of samples
    MIN_EAR_THRESHOLD = 0.12  # minimum EAR threshold

    # MediaPipe settings
    MAX_NUM_FACES = 1
    MIN_DETECTION_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5

# Eye landmark indices for MediaPipe Face Mesh
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["default"],
    },
}
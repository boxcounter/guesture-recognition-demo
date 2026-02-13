"""Configuration settings for gesture recognition system.

All configurable parameters and thresholds are centralized here.
Modify these values to tune the system behavior.
"""

# Camera Settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 1920  # Full HD for better long-distance detection (increased from 1280)
CAMERA_HEIGHT = 1080  # Full HD for better long-distance detection (increased from 720)
CAMERA_FPS = 30
MIRROR_CAMERA = True  # Flip horizontally for user-facing cameras

# MediaPipe Hand Tracking Settings
MODEL_PATH = "models/hand_landmarker.task"  # Path to MediaPipe hand landmarker model
MIN_DETECTION_CONFIDENCE = 0.5  # Confidence threshold for initial hand detection (lowered for better long-distance detection)
MIN_TRACKING_CONFIDENCE = (
    0.4  # Confidence threshold for hand tracking between frames (lowered for stability)
)
MAX_NUM_HANDS = 1  # Maximum number of hands to detect (1 for simplicity)

# Gesture Detection Thresholds
FINGER_EXTENSION_THRESHOLD = 0.2  # Ratio for determining if finger is extended (lowered for long-distance stability)
FINGER_CURL_THRESHOLD = 0.4  # Ratio for determining if finger is curled
POINTING_ANGLE_TOLERANCE = (
    30.0  # Degrees tolerance for pointing direction (increased for stability)
)
PALM_FORWARD_NORMAL_THRESHOLD = (
    0.01  # Z-coordinate threshold for palm orientation (reduced from 0.05)
)
PALM_FORWARD_FINGERTIP_Z_THRESHOLD = (
    0.15  # Hand-relative Z threshold for palm forward detection
)
FINGER_TOUCH_THRESHOLD = 0.05  # Reserved for future: finger pinch detection
THUMB_EXTENSION_THRESHOLD = 0.7  # Minimum thumb length ratio for extension

# Gesture Confidence Thresholds (minimum confidence to report gesture)
MIN_CONFIDENCE_PALM = 0.6  # Lowered for long-distance detection
MIN_CONFIDENCE_POINTING = 0.6  # Lowered for long-distance detection
MIN_CONFIDENCE_FIST = 0.6  # Lowered for long-distance detection
MIN_CONFIDENCE_ROBOT_CONTROL = (
    0.7  # Minimum confidence for robot control (higher = safer)
)

# Temporal Smoothing Settings
GESTURE_HISTORY_SIZE = (
    10  # Number of frames to keep in history (increased for stability at distance)
)
MIN_CONSISTENT_FRAMES = (
    5  # Minimum consecutive frames to confirm gesture change (increased for stability)
)
CONFIDENCE_SMOOTHING_ALPHA = (
    0.3  # Reserved for future: alternative EMA smoothing algorithm
)

# Visualization Settings
LANDMARK_COLOR = (0, 255, 0)  # Green for landmarks (BGR format)
CONNECTION_COLOR = (255, 255, 255)  # White for connections
TEXT_COLOR = (255, 255, 0)  # Cyan for text
GESTURE_LABEL_COLOR = (0, 255, 255)  # Yellow for gesture labels
FONT_SCALE = 1.0
FONT_THICKNESS = 2
LANDMARK_RADIUS = 5  # Circle radius for landmarks
CONNECTION_THICKNESS = 2

# Logging Settings
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_GESTURES_TO_CONSOLE = True  # Print detected gestures to console

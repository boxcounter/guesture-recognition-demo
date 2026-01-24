"""Configuration settings for gesture recognition system.

All configurable parameters and thresholds are centralized here.
Modify these values to tune the system behavior.
"""

# Camera Settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
MIRROR_CAMERA = True  # Flip horizontally for user-facing cameras

# MediaPipe Hand Tracking Settings
MODEL_PATH = "models/hand_landmarker.task"  # Path to MediaPipe hand landmarker model
MIN_DETECTION_CONFIDENCE = 0.7  # Confidence threshold for initial hand detection
MIN_TRACKING_CONFIDENCE = 0.5  # Confidence threshold for hand tracking between frames
MAX_NUM_HANDS = 1  # Maximum number of hands to detect (1 for simplicity)

# Gesture Detection Thresholds
FINGER_EXTENSION_THRESHOLD = (
    0.3  # Ratio for determining if finger is extended (reduced from 0.6)
)
FINGER_CURL_THRESHOLD = 0.4  # Ratio for determining if finger is curled
POINTING_ANGLE_TOLERANCE = 22.5  # Degrees tolerance for pointing direction
PALM_FORWARD_NORMAL_THRESHOLD = (
    0.01  # Z-coordinate threshold for palm orientation (reduced from 0.05)
)
FINGER_TOUCH_THRESHOLD = 0.05  # Distance threshold for finger touching detection
THUMB_EXTENSION_THRESHOLD = (
    0.6  # Minimum thumb length ratio for extension (reduced to allow natural gestures)
)
THUMBS_VERTICAL_SEPARATION_THRESHOLD = (
    0.03  # Y-coordinate separation threshold for thumbs up/down direction detection
)

# Gesture Confidence Thresholds (minimum confidence to report gesture)
MIN_CONFIDENCE_PALM = 0.7
MIN_CONFIDENCE_POINTING = 0.75
MIN_CONFIDENCE_FIST = 0.7
MIN_CONFIDENCE_THUMBS = 0.75

# Temporal Smoothing Settings
GESTURE_HISTORY_SIZE = 5  # Number of frames to keep in history
MIN_CONSISTENT_FRAMES = 3  # Minimum consecutive frames to confirm gesture change
CONFIDENCE_SMOOTHING_ALPHA = 0.3  # EMA smoothing factor (0-1, lower = smoother)

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

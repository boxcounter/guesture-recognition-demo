"""Gesture detection algorithms.

Detects specific hand gestures from landmark data using geometric analysis.
Supported gestures: Palm forward (stop), Pointing directions, Fist (go), Thumbs up/down.
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum

from config import settings
from src.hand_tracker import HandData, Landmark

logger = logging.getLogger(__name__)


class GestureType(Enum):
    """Enumeration of recognized gesture types."""

    UNKNOWN = "unknown"
    PALM_FORWARD = "palm_forward"  # Stop gesture
    POINTING_UP = "pointing_up"
    POINTING_DOWN = "pointing_down"
    POINTING_LEFT = "pointing_left"
    POINTING_RIGHT = "pointing_right"
    FIST = "fist"  # Go gesture
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"


@dataclass(slots=True, frozen=True)
class GestureResult:
    """Result of gesture detection with confidence score."""

    gesture: GestureType
    confidence: float  # 0-1 confidence score


class GestureDetector:
    """Detect hand gestures from landmark data using geometric analysis."""

    def detect(self, hand_data: HandData) -> GestureResult:
        """Detect gesture from hand landmark data.

        Args:
            hand_data: Hand data with 21 landmarks.

        Returns:
            Detected gesture with confidence score.
        """
        landmarks = hand_data.landmarks

        # Check gestures in order of distinctiveness
        # Start with palm forward (most distinct)
        if self._is_palm_forward(landmarks):
            logger.info("Detected: PALM_FORWARD")
            return GestureResult(
                gesture=GestureType.PALM_FORWARD,
                confidence=settings.MIN_CONFIDENCE_PALM,
            )

        # Check for thumbs up/down BEFORE fist (both have curled fingers)
        thumbs_result = self._check_thumbs(landmarks)
        if thumbs_result.gesture != GestureType.UNKNOWN:
            logger.info(f"Detected: {thumbs_result.gesture.value.upper()}")
            return thumbs_result

        # Check for fist (all fingers curled, thumb not pointing up/down)
        if self._is_fist(landmarks):
            logger.info("Detected: FIST")
            return GestureResult(
                gesture=GestureType.FIST,
                confidence=settings.MIN_CONFIDENCE_FIST,
            )

        # Check for pointing gestures
        pointing_result = self._check_pointing(landmarks)
        if pointing_result.gesture != GestureType.UNKNOWN:
            logger.info(f"Detected: {pointing_result.gesture.value.upper()}")
            return pointing_result

        # No gesture detected
        return GestureResult(gesture=GestureType.UNKNOWN, confidence=0.0)

    def _is_palm_forward(self, landmarks: list[Landmark]) -> bool:
        """Check if palm is facing forward (toward camera).

        Palm forward is detected when:
        - All fingertips are extended
        - Palm normal vector points toward camera (negative Z)

        Args:
            landmarks: List of 21 hand landmarks.

        Returns:
            True if palm is facing forward.
        """
        # Check if fingers are extended (excluding thumb)
        index_extended = self._is_finger_extended(landmarks, finger_idx=1)
        middle_extended = self._is_finger_extended(landmarks, finger_idx=2)
        ring_extended = self._is_finger_extended(landmarks, finger_idx=3)
        pinky_extended = self._is_finger_extended(landmarks, finger_idx=4)

        fingers_extended = (
            index_extended and middle_extended and ring_extended and pinky_extended
        )

        logger.info(
            f"Palm check - Index:{index_extended} Middle:{middle_extended} "
            f"Ring:{ring_extended} Pinky:{pinky_extended}"
        )

        if not fingers_extended:
            return False

        # Check palm orientation using wrist (0) and middle MCP (9)
        wrist = landmarks[0]
        middle_mcp = landmarks[9]

        # Palm normal approximation: vector from wrist to middle MCP
        # If Z component is negative, palm is facing camera
        palm_z = middle_mcp.z - wrist.z

        logger.info(
            f"Palm Z-axis: {palm_z:.3f} (threshold: {-settings.PALM_FORWARD_NORMAL_THRESHOLD})"
        )

        return palm_z < -settings.PALM_FORWARD_NORMAL_THRESHOLD

    def _is_fist(self, landmarks: list[Landmark]) -> bool:
        """Check if hand is making a fist.

        Fist is detected when all fingers including thumb are curled.

        Args:
            landmarks: List of 21 hand landmarks.

        Returns:
            True if hand is in fist position.
        """
        # All fingers must be curled (including thumb to distinguish from thumbs up)
        return (
            not self._is_thumb_extended(landmarks)  # Thumb must be curled
            and not self._is_finger_extended(landmarks, finger_idx=1)  # Index
            and not self._is_finger_extended(landmarks, finger_idx=2)  # Middle
            and not self._is_finger_extended(landmarks, finger_idx=3)  # Ring
            and not self._is_finger_extended(landmarks, finger_idx=4)  # Pinky
        )

    def _check_thumbs(self, landmarks: list[Landmark]) -> GestureResult:
        """Check for thumbs up or thumbs down gesture.

        Thumbs gesture is detected when:
        - Other four fingers are curled
        - Thumb is extended (to distinguish from fist)
        - Thumb is pointing up or down (based on Y position, not distance)

        Args:
            landmarks: List of 21 hand landmarks.

        Returns:
            GestureResult with thumbs up/down or UNKNOWN.
        """
        # Check if other fingers are curled (not thumb)
        other_fingers_curled = (
            not self._is_finger_extended(landmarks, finger_idx=1)  # Index
            and not self._is_finger_extended(landmarks, finger_idx=2)  # Middle
            and not self._is_finger_extended(landmarks, finger_idx=3)  # Ring
            and not self._is_finger_extended(landmarks, finger_idx=4)  # Pinky
        )

        if not other_fingers_curled:
            return GestureResult(gesture=GestureType.UNKNOWN, confidence=0.0)

        # Check if thumb is extended (to distinguish from fist)
        if not self._is_thumb_extended(landmarks):
            return GestureResult(gesture=GestureType.UNKNOWN, confidence=0.0)

        # Check thumb direction based on Y coordinate (not distance)
        # In image coordinates, Y=0 is top, Y increases downward
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]  # Use IP instead of MCP for better direction detection
        wrist = landmarks[0]

        # Calculate vertical separation
        thumb_vertical_separation = thumb_ip.y - thumb_tip.y

        logger.info(
            f"Thumbs check - tip.y={thumb_tip.y:.3f}, ip.y={thumb_ip.y:.3f}, "
            f"sep={thumb_vertical_separation:.3f}"
        )

        # If thumb tip is significantly above thumb IP, it's pointing up
        # If thumb tip is significantly below thumb IP, it's pointing down
        if thumb_vertical_separation > 0.03:  # Tip is above IP by threshold
            return GestureResult(
                gesture=GestureType.THUMBS_UP,
                confidence=settings.MIN_CONFIDENCE_THUMBS,
            )
        if thumb_vertical_separation < -0.03:  # Tip is below IP by threshold
            return GestureResult(
                gesture=GestureType.THUMBS_DOWN,
                confidence=settings.MIN_CONFIDENCE_THUMBS,
            )

        # Thumb not pointing clearly up or down
        return GestureResult(gesture=GestureType.UNKNOWN, confidence=0.0)

    def _check_pointing(self, landmarks: list[Landmark]) -> GestureResult:
        """Check for pointing gesture in any direction.

        Pointing is detected when:
        - Index finger is extended
        - Other fingers (middle, ring, pinky) are curled
        - Index finger direction determines gesture

        Args:
            landmarks: List of 21 hand landmarks.

        Returns:
            GestureResult with pointing direction or UNKNOWN.
        """
        # Check if index is extended and others are curled
        index_extended = self._is_finger_extended(landmarks, finger_idx=1)
        others_curled = (
            not self._is_finger_extended(landmarks, finger_idx=2)  # Middle
            and not self._is_finger_extended(landmarks, finger_idx=3)  # Ring
            and not self._is_finger_extended(landmarks, finger_idx=4)  # Pinky
        )

        if not (index_extended and others_curled):
            return GestureResult(gesture=GestureType.UNKNOWN, confidence=0.0)

        # Get pointing direction from index finger
        index_mcp = landmarks[5]  # Index MCP (base)
        index_tip = landmarks[8]  # Index tip

        # Calculate direction vector
        dx = index_tip.x - index_mcp.x
        dy = index_tip.y - index_mcp.y

        # Calculate angle in degrees (0° = right, 90° = down, 180° = left, 270° = up)
        angle = math.degrees(math.atan2(dy, dx))

        # Normalize to [0, 360)
        if angle < 0:
            angle += 360

        logger.info(f"Pointing angle: {angle:.1f}° (dx={dx:.3f}, dy={dy:.3f})")

        tolerance = settings.POINTING_ANGLE_TOLERANCE

        # Determine direction based on angle (centered around cardinal directions)
        # Right: 0° (±tolerance around 360°/0°)
        # Down: 90° (±tolerance)
        # Left: 180° (±tolerance)
        # Up: 270° (±tolerance)

        if angle >= 360 - tolerance or angle <= tolerance:  # Right: 337.5-22.5°
            gesture = GestureType.POINTING_RIGHT
        elif 90 - tolerance <= angle <= 90 + tolerance:  # Down: 67.5-112.5°
            gesture = GestureType.POINTING_DOWN
        elif 180 - tolerance <= angle <= 180 + tolerance:  # Left: 157.5-202.5°
            gesture = GestureType.POINTING_LEFT
        elif 270 - tolerance <= angle <= 270 + tolerance:  # Up: 247.5-292.5°
            gesture = GestureType.POINTING_UP
        else:
            # In between directions, not confident enough
            logger.info(f"Pointing angle {angle:.1f}° in between directions")
            return GestureResult(gesture=GestureType.UNKNOWN, confidence=0.0)

        return GestureResult(
            gesture=gesture,
            confidence=settings.MIN_CONFIDENCE_POINTING,
        )

    def _is_finger_extended(self, landmarks: list[Landmark], finger_idx: int) -> bool:
        """Check if a finger is extended based on landmark positions.

        A finger is considered extended if its tip is farther from the wrist
        than its base (MCP joint).

        Args:
            landmarks: List of 21 hand landmarks.
            finger_idx: Finger index (1=index, 2=middle, 3=ring, 4=pinky).

        Returns:
            True if finger is extended.
        """
        # Landmark indices for each finger
        # Index: MCP=5, PIP=6, DIP=7, TIP=8
        # Middle: MCP=9, PIP=10, DIP=11, TIP=12
        # Ring: MCP=13, PIP=14, DIP=15, TIP=16
        # Pinky: MCP=17, PIP=18, DIP=19, TIP=20
        mcp_idx = 5 + (finger_idx - 1) * 4
        tip_idx = mcp_idx + 3

        wrist = landmarks[0]
        mcp = landmarks[mcp_idx]
        tip = landmarks[tip_idx]

        # Calculate distances from wrist
        dist_mcp = self._distance_2d(wrist, mcp)
        dist_tip = self._distance_2d(wrist, tip)

        # Finger is extended if tip is farther than MCP by threshold
        return dist_tip > dist_mcp * (1 + settings.FINGER_EXTENSION_THRESHOLD)

    def _is_thumb_extended(self, landmarks: list[Landmark]) -> bool:
        """Check if thumb is extended.

        Thumb extension is checked differently than other fingers due to
        its different orientation and movement range.

        Args:
            landmarks: List of 21 hand landmarks.

        Returns:
            True if thumb is extended.
        """
        # Thumb landmarks: CMC=1, MCP=2, IP=3, TIP=4
        wrist = landmarks[0]
        thumb_mcp = landmarks[2]
        thumb_tip = landmarks[4]

        # Calculate distances from wrist
        dist_mcp = self._distance_2d(wrist, thumb_mcp)
        dist_tip = self._distance_2d(wrist, thumb_tip)

        # Thumb is extended if tip is farther than MCP
        ratio = dist_tip / dist_mcp if dist_mcp > 0 else 0
        is_extended = dist_tip > dist_mcp * (1 + settings.THUMB_EXTENSION_THRESHOLD)

        logger.info(f"Thumb: dist_mcp={dist_mcp:.3f}, dist_tip={dist_tip:.3f}, ratio={ratio:.3f}, extended={is_extended}")

        return is_extended

    def _distance_2d(self, p1: Landmark, p2: Landmark) -> float:
        """Calculate 2D Euclidean distance between two landmarks.

        Args:
            p1: First landmark.
            p2: Second landmark.

        Returns:
            Distance in normalized coordinate space.
        """
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        return math.sqrt(dx * dx + dy * dy)

"""Gesture detection algorithms.

Detects specific hand gestures from landmark data using geometric analysis.
Supported gestures: Palm forward (stop), Pointing directions, Fist (go).
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum

from config import settings
from src.hand_tracker import HandData, Landmark, transform_to_hand_space

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


@dataclass(slots=True, frozen=True)
class GestureResult:
    """Result of gesture detection with confidence score."""

    gesture: GestureType
    confidence: float  # 0-1 confidence score


class GestureDetector:
    """Detect hand gestures from landmark data using geometric analysis."""

    def __init__(self, use_hand_relative_coords: bool = True) -> None:
        """Initialize gesture detector.

        Args:
            use_hand_relative_coords: If True, transform landmarks to hand-relative
                coordinate system before detection. This makes detection invariant
                to hand rotation and orientation. Defaults to True.
        """
        self.use_hand_relative_coords = use_hand_relative_coords

    def detect(self, hand_data: HandData) -> GestureResult:
        """Detect gesture from hand landmark data.

        Args:
            hand_data: Hand data with 21 landmarks.

        Returns:
            Detected gesture with confidence score.
        """
        landmarks = hand_data.landmarks
        original_landmarks = landmarks  # Keep original for pointing detection

        # Validate landmark count
        if len(landmarks) != 21:
            logger.error(f"Invalid landmark count: {len(landmarks)}, expected 21")
            return GestureResult(gesture=GestureType.UNKNOWN, confidence=0.0)

        # Transform to hand-relative coordinates if enabled
        # (for fist and palm forward detection)
        if self.use_hand_relative_coords:
            landmarks = transform_to_hand_space(landmarks)
            logger.debug("Transformed landmarks to hand-relative coordinate system")

        # Check gestures in order of distinctiveness
        # Start with palm forward (most distinct)
        palm_result = self._check_palm_forward(landmarks)
        if palm_result.gesture != GestureType.UNKNOWN:
            logger.info(
                f"Detected: PALM_FORWARD (confidence: {palm_result.confidence:.2f})"
            )
            return palm_result

        # Check for fist (all fingers curled)
        fist_result = self._check_fist(landmarks)
        if fist_result.gesture != GestureType.UNKNOWN:
            logger.info(f"Detected: FIST (confidence: {fist_result.confidence:.2f})")
            return fist_result

        # Check for pointing gestures (use original image-space landmarks)
        # Pointing direction is inherently image-relative
        pointing_result = self._check_pointing(original_landmarks)
        if pointing_result.gesture != GestureType.UNKNOWN:
            logger.info(
                f"Detected: {pointing_result.gesture.value.upper()} (confidence: {pointing_result.confidence:.2f})"
            )
            return pointing_result

        # No gesture detected
        return GestureResult(gesture=GestureType.UNKNOWN, confidence=0.0)

    def _check_palm_forward(self, landmarks: list[Landmark]) -> GestureResult:
        """Check if palm is facing forward (toward camera).

        In hand-relative coordinates:
        - All fingertips should be extended along X-axis (positive X)
        - Fingertips should have small Z values (close to palm plane)

        Args:
            landmarks: List of 21 hand landmarks (in hand-relative coords if enabled).

        Returns:
            GestureResult with PALM_FORWARD or UNKNOWN and calculated confidence.
        """
        # Check if fingers are extended (excluding thumb)
        index_extended = self._is_finger_extended(landmarks, finger_idx=1)
        middle_extended = self._is_finger_extended(landmarks, finger_idx=2)
        ring_extended = self._is_finger_extended(landmarks, finger_idx=3)
        pinky_extended = self._is_finger_extended(landmarks, finger_idx=4)

        fingers_extended = (
            index_extended and middle_extended and ring_extended and pinky_extended
        )

        logger.debug(
            f"Palm check - Index:{index_extended} Middle:{middle_extended} "
            f"Ring:{ring_extended} Pinky:{pinky_extended}"
        )

        if not fingers_extended:
            return GestureResult(gesture=GestureType.UNKNOWN, confidence=0.0)

        # In hand-relative space, check that fingertips are in palm plane (small Z)
        # In image space, check Z-depth as before
        if self.use_hand_relative_coords:
            # Average Z-coordinate of fingertips (should be close to palm plane)
            fingertip_indices = [8, 12, 16, 20]  # Index, middle, ring, pinky tips
            avg_z = sum(abs(landmarks[i].z) for i in fingertip_indices) / len(
                fingertip_indices
            )

            logger.debug(f"Palm forward - average fingertip |Z|: {avg_z:.3f}")

            # If fingertips are too far from palm plane, not palm forward
            if avg_z > settings.PALM_FORWARD_FINGERTIP_Z_THRESHOLD:
                return GestureResult(gesture=GestureType.UNKNOWN, confidence=0.0)

            # Confidence based on how close fingertips are to palm plane
            confidence_raw = 1.0 - (avg_z / settings.PALM_FORWARD_FINGERTIP_Z_THRESHOLD)
            confidence = max(settings.MIN_CONFIDENCE_PALM, min(1.0, confidence_raw))
        else:
            # Original image-space detection
            wrist = landmarks[0]
            middle_mcp = landmarks[9]

            # Palm normal approximation: vector from wrist to middle MCP
            # If Z component is negative, palm is facing camera
            palm_z = middle_mcp.z - wrist.z

            logger.debug(
                f"Palm Z-axis: {palm_z:.3f} (threshold: {-settings.PALM_FORWARD_NORMAL_THRESHOLD})"
            )

            if palm_z >= -settings.PALM_FORWARD_NORMAL_THRESHOLD:
                return GestureResult(gesture=GestureType.UNKNOWN, confidence=0.0)

            # Calculate confidence
            confidence_raw = abs(palm_z) / (settings.PALM_FORWARD_NORMAL_THRESHOLD * 10)
            confidence = max(settings.MIN_CONFIDENCE_PALM, min(1.0, confidence_raw))

        return GestureResult(gesture=GestureType.PALM_FORWARD, confidence=confidence)

    def _check_fist(self, landmarks: list[Landmark]) -> GestureResult:
        """Check if hand is making a fist.

        Fist is detected when all fingers including thumb are curled.

        Args:
            landmarks: List of 21 hand landmarks.

        Returns:
            GestureResult with FIST or UNKNOWN and calculated confidence.
        """
        # Check all fingers are curled
        thumb_curled = not self._is_thumb_extended(landmarks)
        index_curled = not self._is_finger_extended(landmarks, finger_idx=1)
        middle_curled = not self._is_finger_extended(landmarks, finger_idx=2)
        ring_curled = not self._is_finger_extended(landmarks, finger_idx=3)
        pinky_curled = not self._is_finger_extended(landmarks, finger_idx=4)

        all_curled = (
            thumb_curled
            and index_curled
            and middle_curled
            and ring_curled
            and pinky_curled
        )

        if not all_curled:
            return GestureResult(gesture=GestureType.UNKNOWN, confidence=0.0)

        # Calculate confidence based on curl strength of each finger
        # For simplicity, calculate how much closer tips are to wrist compared to MCPs
        wrist = landmarks[0]

        curl_confidences = []

        # Check each finger (1-4, not thumb which behaves differently)
        for finger_idx in range(1, 5):
            mcp_idx = 5 + (finger_idx - 1) * 4
            tip_idx = mcp_idx + 3
            mcp = landmarks[mcp_idx]
            tip = landmarks[tip_idx]

            dist_mcp = self._distance_2d(wrist, mcp)
            dist_tip = self._distance_2d(wrist, tip)

            # Curl confidence: tip should be closer to wrist than MCP
            # Higher ratio = more curled = higher confidence
            if dist_mcp > 0:
                curl_ratio = 1.0 - (dist_tip / dist_mcp)
                curl_confidences.append(max(0.0, min(1.0, curl_ratio * 2.0)))

        # Check thumb separately
        thumb_mcp = landmarks[2]
        thumb_tip = landmarks[4]
        dist_thumb_mcp = self._distance_2d(wrist, thumb_mcp)
        dist_thumb_tip = self._distance_2d(wrist, thumb_tip)
        if dist_thumb_mcp > 0:
            thumb_curl_ratio = 1.0 - (dist_thumb_tip / dist_thumb_mcp)
            curl_confidences.append(max(0.0, min(1.0, thumb_curl_ratio * 2.0)))

        # Average confidence across all fingers
        if curl_confidences:
            confidence_raw = sum(curl_confidences) / len(curl_confidences)
            confidence = max(settings.MIN_CONFIDENCE_FIST, min(1.0, confidence_raw))
        else:
            confidence = settings.MIN_CONFIDENCE_FIST

        return GestureResult(gesture=GestureType.FIST, confidence=confidence)

    def _check_pointing(self, landmarks: list[Landmark]) -> GestureResult:
        """Check for pointing gesture in any direction.

        Pointing is detected when:
        - Index finger is extended
        - Other fingers (middle, ring, pinky) are curled
        - Index finger direction determines gesture

        Note: Always uses image-space landmarks for direction detection.

        Args:
            landmarks: List of 21 hand landmarks (in image coordinates).

        Returns:
            GestureResult with pointing direction or UNKNOWN with calculated confidence.
        """
        # Check if index is extended and others are curled
        # Use image-space distance (2D) since these are image-space landmarks
        index_extended = self._is_finger_extended(landmarks, finger_idx=1, use_3d=False)
        others_curled = (
            not self._is_finger_extended(
                landmarks, finger_idx=2, use_3d=False
            )  # Middle
            and not self._is_finger_extended(
                landmarks, finger_idx=3, use_3d=False
            )  # Ring
            and not self._is_finger_extended(
                landmarks, finger_idx=4, use_3d=False
            )  # Pinky
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

        logger.debug(f"Pointing angle: {angle:.1f}° (dx={dx:.3f}, dy={dy:.3f})")

        tolerance = settings.POINTING_ANGLE_TOLERANCE

        # Determine direction based on angle (centered around cardinal directions)
        # Right: 0° (±tolerance around 360°/0°)
        # Down: 90° (±tolerance)
        # Left: 180° (±tolerance)
        # Up: 270° (±tolerance)

        gesture = GestureType.UNKNOWN
        angle_diff = 0.0

        if angle >= 360 - tolerance or angle <= tolerance:  # Right: 337.5-22.5°
            gesture = GestureType.POINTING_RIGHT
            angle_diff = min(angle, 360 - angle)  # Distance from 0°/360°
        elif 90 - tolerance <= angle <= 90 + tolerance:  # Down: 67.5-112.5°
            gesture = GestureType.POINTING_DOWN
            angle_diff = abs(angle - 90)
        elif 180 - tolerance <= angle <= 180 + tolerance:  # Left: 157.5-202.5°
            gesture = GestureType.POINTING_LEFT
            angle_diff = abs(angle - 180)
        elif 270 - tolerance <= angle <= 270 + tolerance:  # Up: 247.5-292.5°
            gesture = GestureType.POINTING_UP
            angle_diff = abs(angle - 270)
        else:
            # In between directions, not confident enough
            logger.debug(f"Pointing angle {angle:.1f}° in between directions")
            return GestureResult(gesture=GestureType.UNKNOWN, confidence=0.0)

        # Calculate confidence based on angle proximity to cardinal direction
        # Closer to exact cardinal direction = higher confidence
        confidence_raw = 1.0 - (angle_diff / tolerance)
        confidence = max(settings.MIN_CONFIDENCE_POINTING, min(1.0, confidence_raw))

        return GestureResult(gesture=gesture, confidence=confidence)

    def _is_finger_extended(
        self, landmarks: list[Landmark], finger_idx: int, use_3d: bool | None = None
    ) -> bool:
        """Check if a finger is extended based on landmark positions.

        Args:
            landmarks: List of 21 hand landmarks.
            finger_idx: Finger index (1=index, 2=middle, 3=ring, 4=pinky).
            use_3d: If True, use 3D distance; if False, use 2D distance.
                If None, uses self.use_hand_relative_coords.

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

        # Determine which distance calculation to use
        if use_3d is None:
            use_3d = self.use_hand_relative_coords

        # Use 3D distance in hand-relative space, 2D in image space
        if use_3d:
            dist_mcp = self._distance_3d(wrist, mcp)
            dist_tip = self._distance_3d(wrist, tip)
        else:
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

        # Use 3D distance in hand-relative space, 2D in image space
        if self.use_hand_relative_coords:
            dist_mcp = self._distance_3d(wrist, thumb_mcp)
            dist_tip = self._distance_3d(wrist, thumb_tip)
        else:
            dist_mcp = self._distance_2d(wrist, thumb_mcp)
            dist_tip = self._distance_2d(wrist, thumb_tip)

        # Thumb is extended if tip is farther than MCP by threshold percentage
        is_extended = dist_tip > dist_mcp * (1 + settings.THUMB_EXTENSION_THRESHOLD)

        logger.debug(
            f"Thumb: dist_mcp={dist_mcp:.3f}, dist_tip={dist_tip:.3f}, "
            f"threshold={settings.THUMB_EXTENSION_THRESHOLD}, extended={is_extended}"
        )

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

    def _distance_3d(self, p1: Landmark, p2: Landmark) -> float:
        """Calculate 3D Euclidean distance between two landmarks.

        Args:
            p1: First landmark.
            p2: Second landmark.

        Returns:
            Distance in 3D coordinate space.
        """
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        dz = p2.z - p1.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

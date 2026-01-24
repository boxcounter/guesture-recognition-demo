"""Gesture recognition engine with temporal smoothing.

This module provides the high-level API for gesture recognition, adding temporal
smoothing and confidence aggregation on top of the raw gesture detection algorithms.
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import settings
from src.gesture_detector import GestureDetector, GestureResult, GestureType
from src.hand_tracker import HandData, HandTracker
from src.visualizer import Visualizer

logger = logging.getLogger(__name__)


@dataclass
class Gesture:
    """Detected gesture with metadata.

    This is the smoothed, stable gesture output after temporal filtering.
    """

    name: str  # Gesture name (e.g., "FIST", "PALM_FORWARD")
    gesture_type: GestureType  # Enum value
    confidence: float  # Average confidence over consistent frames (0-1)
    direction: Optional[str] = None  # For pointing gestures: "up", "down", "left", "right"


@dataclass
class RecognitionResult:
    """Complete result from gesture recognition pipeline.

    This is the main output data structure for the API.
    """

    gesture: Optional[Gesture]  # Detected gesture (None if no stable gesture)
    hand_detected: bool  # Whether a hand is visible
    hand_landmarks: Optional[list]  # Raw landmark data (21 points)
    frame_with_annotations: np.ndarray  # Annotated video frame
    raw_confidence: float  # Raw confidence from current frame (0-1)
    is_stable: bool  # Whether gesture has been stable for MIN_CONSISTENT_FRAMES


class GestureSmoother:
    """Temporal smoothing for gesture detection.

    Tracks gesture history and requires consistent detections before reporting
    a gesture change, preventing jitter and false positives.
    """

    def __init__(
        self,
        history_size: int = settings.GESTURE_HISTORY_SIZE,
        min_consistent_frames: int = settings.MIN_CONSISTENT_FRAMES,
    ):
        """Initialize gesture smoother.

        Args:
            history_size: Number of frames to keep in history.
            min_consistent_frames: Minimum frames required to confirm gesture change.
        """
        self.history_size = history_size
        self.min_consistent_frames = min_consistent_frames
        self.gesture_history: deque[GestureType] = deque(maxlen=history_size)
        self.confidence_history: deque[float] = deque(maxlen=history_size)
        self.current_stable_gesture: Optional[GestureType] = None

    def update(self, gesture: GestureType, confidence: float) -> tuple[GestureType, float, bool]:
        """Update history and get smoothed gesture.

        Args:
            gesture: Raw detected gesture from current frame.
            confidence: Raw confidence score from current frame.

        Returns:
            Tuple of (smoothed_gesture, avg_confidence, is_stable)
            - smoothed_gesture: The stable gesture (or UNKNOWN if not stable)
            - avg_confidence: Average confidence over consistent frames
            - is_stable: Whether gesture meets consistency threshold
        """
        # Add to history
        self.gesture_history.append(gesture)
        self.confidence_history.append(confidence)

        # Need enough history first
        if len(self.gesture_history) < self.min_consistent_frames:
            return GestureType.UNKNOWN, 0.0, False

        # Count occurrences of each gesture in history
        gesture_counts: dict[GestureType, int] = {}
        for g in self.gesture_history:
            gesture_counts[g] = gesture_counts.get(g, 0) + 1

        # Find most common gesture
        most_common_gesture = max(gesture_counts.items(), key=lambda x: x[1])
        dominant_gesture, count = most_common_gesture

        # Check if it meets consistency threshold
        is_stable = count >= self.min_consistent_frames

        if is_stable:
            # Calculate average confidence for this gesture
            avg_confidence = self._calculate_avg_confidence(dominant_gesture)
            self.current_stable_gesture = dominant_gesture
            return dominant_gesture, avg_confidence, True
        else:
            # Not stable enough - return UNKNOWN
            return GestureType.UNKNOWN, 0.0, False

    def _calculate_avg_confidence(self, gesture: GestureType) -> float:
        """Calculate average confidence for a specific gesture in history.

        Args:
            gesture: The gesture to calculate confidence for.

        Returns:
            Average confidence score.
        """
        relevant_confidences = [
            conf
            for g, conf in zip(self.gesture_history, self.confidence_history)
            if g == gesture
        ]
        if not relevant_confidences:
            return 0.0
        return sum(relevant_confidences) / len(relevant_confidences)

    def reset(self) -> None:
        """Clear history (useful when hand is lost)."""
        self.gesture_history.clear()
        self.confidence_history.clear()
        self.current_stable_gesture = None


class GestureRecognizer:
    """High-level gesture recognition API with temporal smoothing.

    This is the main class for integrating gesture recognition into applications.
    It orchestrates hand tracking, gesture detection, temporal smoothing, and
    visualization.

    Example usage:
        recognizer = GestureRecognizer()

        while True:
            frame = camera.get_frame()
            result = recognizer.process_frame(frame)

            if result.gesture:
                print(f"Gesture: {result.gesture.name} ({result.gesture.confidence:.2f})")

            cv2.imshow("Gesture Recognition", result.frame_with_annotations)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        recognizer.close()
    """

    def __init__(self):
        """Initialize gesture recognizer with all components."""
        self.hand_tracker = HandTracker()
        self.gesture_detector = GestureDetector()
        self.visualizer = Visualizer()
        self.smoother = GestureSmoother()
        logger.info("GestureRecognizer initialized")

    def process_frame(self, frame: np.ndarray) -> RecognitionResult:
        """Process a single frame and return recognition result.

        This is the main API method. It performs:
        1. Hand tracking (extract landmarks)
        2. Gesture detection (raw, per-frame)
        3. Temporal smoothing (stability over time)
        4. Visualization (annotate frame)

        Args:
            frame: Input video frame (BGR format from OpenCV).

        Returns:
            RecognitionResult with gesture, confidence, and annotated frame.
        """
        # Make a copy for annotations
        annotated_frame = frame.copy()

        # Track hands
        hands = self.hand_tracker.process_frame(frame)

        # No hand detected
        if not hands:
            self.smoother.reset()
            return RecognitionResult(
                gesture=None,
                hand_detected=False,
                hand_landmarks=None,
                frame_with_annotations=annotated_frame,
                raw_confidence=0.0,
                is_stable=False,
            )

        # Process first hand (prioritize right hand if multiple)
        hand_data = self._select_primary_hand(hands)

        # Detect gesture (raw, current frame only)
        raw_result: GestureResult = self.gesture_detector.detect(hand_data)

        # Apply temporal smoothing
        smoothed_gesture, avg_confidence, is_stable = self.smoother.update(
            raw_result.gesture, raw_result.confidence
        )

        # Create Gesture object if stable
        gesture = None
        if is_stable and smoothed_gesture != GestureType.UNKNOWN:
            gesture = self._create_gesture_object(smoothed_gesture, avg_confidence)

        # Visualize
        self.visualizer.draw_landmarks(annotated_frame, hand_data.landmarks)
        if gesture:
            self.visualizer.draw_gesture_label(
                annotated_frame, gesture.name, gesture.confidence
            )

        return RecognitionResult(
            gesture=gesture,
            hand_detected=True,
            hand_landmarks=hand_data.landmarks,
            frame_with_annotations=annotated_frame,
            raw_confidence=raw_result.confidence,
            is_stable=is_stable,
        )

    def _select_primary_hand(self, hands: list[HandData]) -> HandData:
        """Select primary hand from multiple detections.

        Priority: Right hand > Left hand > First detected

        Args:
            hands: List of detected hands.

        Returns:
            Primary hand to process.
        """
        if len(hands) == 1:
            return hands[0]

        # Prefer right hand
        for hand in hands:
            if hand.handedness == "Right":
                return hand

        # Fall back to first hand
        return hands[0]

    def _create_gesture_object(
        self, gesture_type: GestureType, confidence: float
    ) -> Gesture:
        """Create Gesture object from GestureType enum.

        Args:
            gesture_type: The detected gesture type.
            confidence: Confidence score.

        Returns:
            Gesture object with name and metadata.
        """
        name = gesture_type.value.upper()
        direction = None

        # Extract direction for pointing gestures
        if gesture_type in (
            GestureType.POINTING_UP,
            GestureType.POINTING_DOWN,
            GestureType.POINTING_LEFT,
            GestureType.POINTING_RIGHT,
        ):
            direction = gesture_type.value.split("_")[1]  # Extract "up", "down", etc.

        return Gesture(
            name=name,
            gesture_type=gesture_type,
            confidence=confidence,
            direction=direction,
        )

    def get_current_gesture(self) -> Optional[Gesture]:
        """Get the current stable gesture without processing a new frame.

        Returns:
            Current stable gesture, or None if no stable gesture.
        """
        if self.smoother.current_stable_gesture == GestureType.UNKNOWN:
            return None
        if self.smoother.current_stable_gesture is None:
            return None

        # Get average confidence
        avg_confidence = self.smoother._calculate_avg_confidence(
            self.smoother.current_stable_gesture
        )

        return self._create_gesture_object(
            self.smoother.current_stable_gesture, avg_confidence
        )

    def reset(self) -> None:
        """Reset the recognizer state (clear history)."""
        self.smoother.reset()
        logger.info("GestureRecognizer reset")

    def close(self) -> None:
        """Clean up resources."""
        self.hand_tracker.close()
        logger.info("GestureRecognizer closed")

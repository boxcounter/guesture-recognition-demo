"""Hand tracking using MediaPipe HandLandmarker.

This module provides a wrapper around MediaPipe HandLandmarker for detecting and
tracking hand landmarks in video frames.

Note: Uses MediaPipe 0.10+ API (HandLandmarker instead of legacy Hands solution).
"""

import math
from dataclasses import dataclass
from types import TracebackType
from typing import Self

import cv2
import mediapipe as mp  # type: ignore[import-untyped]
import numpy as np
from mediapipe.tasks import python  # type: ignore[import-untyped]
from mediapipe.tasks.python import vision  # type: ignore[import-untyped]

from config import settings


@dataclass(slots=True, frozen=True)
class Landmark:
    """3D landmark position with visibility score.

    Coordinates are normalized to [0, 1] range relative to image dimensions.
    """

    x: float  # Horizontal position (0=left, 1=right)
    y: float  # Vertical position (0=top, 1=bottom)
    z: float  # Depth (negative=toward camera, positive=away)
    visibility: float  # Confidence that landmark is visible (0-1)


@dataclass(slots=True)
class HandData:
    """Detected hand with landmarks and metadata."""

    landmarks: list[Landmark]  # 21 landmarks per hand
    handedness: str  # "Left" or "Right"
    confidence: float  # Detection confidence (0-1)


def normalize_vector(v: tuple[float, float, float]) -> tuple[float, float, float]:
    """Normalize a 3D vector to unit length.

    Args:
        v: Vector as (x, y, z) tuple.

    Returns:
        Normalized vector with length 1.0, or (0, 0, 0) if input is zero vector.
    """
    length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if length < 1e-8:  # Avoid division by zero
        return (0.0, 0.0, 0.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def cross_product(
    v1: tuple[float, float, float], v2: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Calculate cross product of two 3D vectors.

    Args:
        v1: First vector as (x, y, z) tuple.
        v2: Second vector as (x, y, z) tuple.

    Returns:
        Cross product v1 × v2 as (x, y, z) tuple.
    """
    return (
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    )


def dot_product(
    v1: tuple[float, float, float], v2: tuple[float, float, float]
) -> float:
    """Calculate dot product of two 3D vectors.

    Args:
        v1: First vector as (x, y, z) tuple.
        v2: Second vector as (x, y, z) tuple.

    Returns:
        Dot product v1 · v2.
    """
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


def calculate_hand_basis(
    landmarks: list[Landmark],
) -> tuple[
    tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
]:
    """Calculate hand-relative coordinate system basis vectors.

    The hand basis is defined as:
    - Origin: Wrist (landmark 0)
    - X-axis: Direction from wrist to middle finger MCP (landmark 9)
    - Y-axis: Perpendicular to X, in palm plane (using pinky MCP)
    - Z-axis: Palm normal (X × Y)

    This creates a coordinate system that rotates with the hand.

    Args:
        landmarks: List of 21 hand landmarks.

    Returns:
        Tuple of (x_axis, y_axis, z_axis) as unit vectors.
        Returns identity basis if landmarks are degenerate.
    """
    wrist = landmarks[0]
    middle_mcp = landmarks[9]  # Middle finger base
    pinky_mcp = landmarks[17]  # Pinky finger base

    # X-axis: wrist -> middle MCP (forward direction along hand)
    x_vec = (
        middle_mcp.x - wrist.x,
        middle_mcp.y - wrist.y,
        middle_mcp.z - wrist.z,
    )
    x_axis = normalize_vector(x_vec)

    # Temp vector: wrist -> pinky MCP (across palm)
    pinky_vec = (
        pinky_mcp.x - wrist.x,
        pinky_mcp.y - wrist.y,
        pinky_mcp.z - wrist.z,
    )

    # Z-axis: normal to palm plane (X × pinky_vec)
    z_vec = cross_product(x_vec, pinky_vec)
    z_axis = normalize_vector(z_vec)

    # Y-axis: perpendicular to X and Z (Z × X)
    # This ensures right-handed coordinate system
    y_vec = cross_product(z_vec, x_vec)
    y_axis = normalize_vector(y_vec)

    # Check for degenerate case (collinear points)
    if (
        x_axis == (0.0, 0.0, 0.0)
        or y_axis == (0.0, 0.0, 0.0)
        or z_axis == (0.0, 0.0, 0.0)
    ):
        # Return identity basis as fallback
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

    return (x_axis, y_axis, z_axis)


def transform_to_hand_space(
    landmarks: list[Landmark],
) -> list[Landmark]:
    """Transform landmarks from image space to hand-relative coordinate system.

    This makes gesture detection invariant to hand rotation and orientation.

    Args:
        landmarks: List of 21 landmarks in image coordinates.

    Returns:
        List of 21 landmarks in hand-relative coordinates.
    """
    # Calculate hand basis vectors
    x_axis, y_axis, z_axis = calculate_hand_basis(landmarks)

    # Origin is the wrist
    wrist = landmarks[0]
    origin = (wrist.x, wrist.y, wrist.z)

    # Transform each landmark
    transformed = []
    for lm in landmarks:
        # Vector from wrist to landmark
        vec = (lm.x - origin[0], lm.y - origin[1], lm.z - origin[2])

        # Project onto hand basis vectors
        x_new = dot_product(vec, x_axis)
        y_new = dot_product(vec, y_axis)
        z_new = dot_product(vec, z_axis)

        transformed.append(
            Landmark(
                x=x_new,
                y=y_new,
                z=z_new,
                visibility=lm.visibility,
            )
        )

    return transformed


class HandTracker:
    """MediaPipe-based hand tracking and landmark extraction.

    This class wraps MediaPipe HandLandmarker to provide a clean interface for
    detecting hands and extracting their 21 3D landmarks from video frames.

    Landmarks (0-20):
        0: WRIST
        1-4: THUMB (CMC, MCP, IP, TIP)
        5-8: INDEX (MCP, PIP, DIP, TIP)
        9-12: MIDDLE (MCP, PIP, DIP, TIP)
        13-16: RING (MCP, PIP, DIP, TIP)
        17-20: PINKY (MCP, PIP, DIP, TIP)
    """

    def __init__(
        self,
        min_detection_confidence: float | None = None,
        min_tracking_confidence: float | None = None,
        max_num_hands: int | None = None,
        landmark_smoothing: float = 0.5,  # EMA smoothing factor (0=no smoothing, 1=max smoothing)
    ) -> None:
        """Initialize hand tracker with MediaPipe HandLandmarker.

        Args:
            min_detection_confidence: Minimum confidence for initial detection.
                Defaults to settings.MIN_DETECTION_CONFIDENCE.
            min_tracking_confidence: Minimum confidence for tracking between frames.
                Defaults to settings.MIN_TRACKING_CONFIDENCE.
            max_num_hands: Maximum number of hands to detect.
                Defaults to settings.MAX_NUM_HANDS.
            landmark_smoothing: Smoothing factor for landmarks (0-1).
                Higher values = more smoothing = less jitter but more lag.
        """
        self.min_detection_confidence = (
            min_detection_confidence or settings.MIN_DETECTION_CONFIDENCE
        )
        self.min_tracking_confidence = (
            min_tracking_confidence or settings.MIN_TRACKING_CONFIDENCE
        )
        self.max_num_hands = max_num_hands or settings.MAX_NUM_HANDS
        self.landmark_smoothing = max(
            0.0, min(1.0, landmark_smoothing)
        )  # Clamp to [0,1]

        # Store previous landmarks for smoothing
        self._prev_landmarks: list[Landmark] | None = None

        # Configure MediaPipe HandLandmarker
        base_options = python.BaseOptions(model_asset_path=settings.MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.max_num_hands,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_tracking_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._frame_timestamp_ms = 0

    def process_frame(self, frame: np.ndarray) -> list[HandData]:
        """Process a video frame and detect hands.

        Args:
            frame: BGR image from OpenCV (shape: H x W x 3).

        Returns:
            List of detected hands with landmarks. Empty if no hands detected.

        Raises:
            ValueError: If frame is invalid or has wrong shape.
        """
        # Check for None or empty - runtime safety check
        if not isinstance(frame, np.ndarray) or frame.size == 0:  # type: ignore[reportUnnecessaryIsInstance]
            raise ValueError("Frame is None or empty")

        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected BGR image (H, W, 3), got shape {frame.shape}")

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Process with timestamp (required for VIDEO mode)
        self._frame_timestamp_ms += 33  # Approximate 30 FPS
        result = self.landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

        # Extract hand data
        hands_data: list[HandData] = []

        if result.hand_landmarks and result.handedness:
            for hand_landmarks, handedness in zip(
                result.hand_landmarks,
                result.handedness,
                strict=True,
            ):
                # Convert MediaPipe landmarks to our Landmark dataclass
                landmarks = [
                    Landmark(
                        x=lm.x,
                        y=lm.y,
                        z=lm.z,
                        visibility=lm.visibility if hasattr(lm, "visibility") else 1.0,
                    )
                    for lm in hand_landmarks
                ]

                # Apply landmark smoothing if enabled
                if self.landmark_smoothing > 0 and self._prev_landmarks is not None:
                    landmarks = self._smooth_landmarks(landmarks, self._prev_landmarks)

                # Store for next frame
                self._prev_landmarks = landmarks

                # Extract handedness label and confidence
                # handedness is a list of Classification objects
                hand_label = handedness[0].category_name  # "Left" or "Right"
                hand_confidence = handedness[0].score

                hands_data.append(
                    HandData(
                        landmarks=landmarks,
                        handedness=hand_label,
                        confidence=hand_confidence,
                    )
                )

        return hands_data

    def _smooth_landmarks(
        self, current: list[Landmark], previous: list[Landmark]
    ) -> list[Landmark]:
        """Apply exponential moving average smoothing to landmarks.

        This reduces jitter in landmark positions, especially important for
        long-distance detection where landmarks are less stable.

        Args:
            current: Current frame landmarks.
            previous: Previous frame landmarks.

        Returns:
            Smoothed landmarks.
        """
        alpha = 1.0 - self.landmark_smoothing  # Convert to EMA alpha
        smoothed = []

        for curr, prev in zip(current, previous, strict=True):
            smoothed.append(
                Landmark(
                    x=alpha * curr.x + (1 - alpha) * prev.x,
                    y=alpha * curr.y + (1 - alpha) * prev.y,
                    z=alpha * curr.z + (1 - alpha) * prev.z,
                    visibility=curr.visibility,  # Don't smooth visibility
                )
            )

        return smoothed

    def close(self) -> None:
        """Release MediaPipe resources.

        Call this when done with hand tracking to free resources.
        """
        if hasattr(self, "landmarker"):
            self.landmarker.close()

    def __enter__(self) -> Self:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit - cleanup resources."""
        self.close()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.close()

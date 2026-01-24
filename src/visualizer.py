"""Visualization utilities for drawing landmarks and annotations.

Handles rendering hand landmarks, connections, gesture labels, and other
overlays on video frames.
"""

import cv2
import numpy as np

from config import settings
from src.hand_tracker import HandData, Landmark


class Visualizer:
    """Draw hand landmarks and annotations on video frames."""

    def draw_landmarks(
        self,
        frame: np.ndarray,
        hand_data: HandData,
    ) -> None:
        """Draw hand landmarks and connections on frame.

        Args:
            frame: BGR image to draw on (modified in-place).
            hand_data: Hand data with landmarks to visualize.
        """
        # Convert our Landmark dataclass back to MediaPipe format for drawing
        # MediaPipe drawing utils expect the specific landmark format
        height, width = frame.shape[:2]

        # Draw landmark points
        for landmark in hand_data.landmarks:
            # Convert normalized coordinates to pixel coordinates
            x = int(landmark.x * width)
            y = int(landmark.y * height)

            # Draw landmark circle
            cv2.circle(
                frame,
                (x, y),
                settings.LANDMARK_RADIUS,
                settings.LANDMARK_COLOR,
                -1,  # Filled circle
            )

        # Draw connections between landmarks
        self._draw_connections(frame, hand_data.landmarks, width, height)

    def _draw_connections(
        self,
        frame: np.ndarray,
        landmarks: list[Landmark],
        width: int,
        height: int,
    ) -> None:
        """Draw lines connecting hand landmarks.

        Args:
            frame: BGR image to draw on (modified in-place).
            landmarks: List of 21 hand landmarks.
            width: Frame width in pixels.
            height: Frame height in pixels.
        """
        # MediaPipe hand connections (landmark index pairs)
        connections = [
            # Thumb
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            # Index finger
            (0, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            # Middle finger
            (5, 9),
            (9, 10),
            (10, 11),
            (11, 12),
            # Ring finger
            (9, 13),
            (13, 14),
            (14, 15),
            (15, 16),
            # Pinky
            (13, 17),
            (17, 18),
            (18, 19),
            (19, 20),
            # Palm
            (0, 17),
        ]

        for start_idx, end_idx in connections:
            start = landmarks[start_idx]
            end = landmarks[end_idx]

            # Convert to pixel coordinates
            start_point = (int(start.x * width), int(start.y * height))
            end_point = (int(end.x * width), int(end.y * height))

            # Draw connection line
            cv2.line(
                frame,
                start_point,
                end_point,
                settings.CONNECTION_COLOR,
                settings.CONNECTION_THICKNESS,
            )

    def draw_gesture_label(
        self,
        frame: np.ndarray,
        gesture: str,
        confidence: float,
    ) -> None:
        """Draw gesture name and confidence on frame.

        Args:
            frame: BGR image to draw on (modified in-place).
            gesture: Gesture name to display.
            confidence: Confidence score (0-1).
        """
        # Format label text
        label = f"{gesture}: {confidence:.2f}"

        # Draw text with background for better visibility
        text_size = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            settings.FONT_SCALE,
            settings.FONT_THICKNESS,
        )[0]

        # Position at top center
        x = (frame.shape[1] - text_size[0]) // 2
        y = 40

        # Draw background rectangle
        cv2.rectangle(
            frame,
            (x - 10, y - text_size[1] - 10),
            (x + text_size[0] + 10, y + 10),
            (0, 0, 0),  # Black background
            -1,
        )

        # Draw text
        cv2.putText(
            frame,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            settings.FONT_SCALE,
            settings.GESTURE_LABEL_COLOR,
            settings.FONT_THICKNESS,
        )

    def draw_fps(self, frame: np.ndarray, fps: float) -> None:
        """Draw FPS counter on frame.

        Args:
            frame: BGR image to draw on (modified in-place).
            fps: Frames per second to display.
        """
        label = f"FPS: {fps:.1f}"

        # Position at top-right corner
        text_size = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2,
        )[0]

        x = frame.shape[1] - text_size[0] - 10
        y = 30

        # Draw text
        cv2.putText(
            frame,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            settings.TEXT_COLOR,
            2,
        )

    def draw_hand_status(
        self,
        frame: np.ndarray,
        hand_detected: bool,
    ) -> None:
        """Draw hand detection status indicator.

        Args:
            frame: BGR image to draw on (modified in-place).
            hand_detected: Whether hand is currently detected.
        """
        status = "Hand: Detected" if hand_detected else "Hand: Not Found"
        color = (0, 255, 0) if hand_detected else (0, 0, 255)  # Green or Red

        # Position at top-left
        cv2.putText(
            frame,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

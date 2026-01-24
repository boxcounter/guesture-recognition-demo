"""Video capture and frame processing.

This module provides a clean interface for camera/video input using OpenCV,
with automatic resource management and FPS tracking.
"""

import time
from types import TracebackType
from typing import Self

import cv2
import numpy as np

from config import settings


class VideoProcessor:
    """Manage video capture from camera or file with FPS tracking.

    Provides context manager support for automatic resource cleanup.
    """

    def __init__(
        self,
        camera_index: int | None = None,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
        mirror: bool | None = None,
    ) -> None:
        """Initialize video processor.

        Args:
            camera_index: Camera device index. Defaults to settings.CAMERA_INDEX.
            width: Frame width in pixels. Defaults to settings.CAMERA_WIDTH.
            height: Frame height in pixels. Defaults to settings.CAMERA_HEIGHT.
            fps: Target frames per second. Defaults to settings.CAMERA_FPS.
            mirror: Mirror frames horizontally. Defaults to settings.MIRROR_CAMERA.
        """
        self.camera_index = (
            camera_index if camera_index is not None else settings.CAMERA_INDEX
        )
        self.width = width or settings.CAMERA_WIDTH
        self.height = height or settings.CAMERA_HEIGHT
        self.target_fps = fps or settings.CAMERA_FPS
        self.mirror = mirror if mirror is not None else settings.MIRROR_CAMERA

        self.cap: cv2.VideoCapture | None = None
        self._frame_count = 0
        self._start_time: float | None = None
        self._last_fps_calc = 0.0
        self._current_fps = 0.0

    def start(self) -> None:
        """Initialize and configure camera capture.

        Raises:
            RuntimeError: If camera cannot be opened.
        """
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera {self.camera_index}. "
                f"Try different camera index (0, 1, 2...)"
            )

        # Configure camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        self._start_time = time.time()
        self._frame_count = 0

    def get_frame(self) -> np.ndarray | None:
        """Capture and return next frame.

        Returns:
            BGR frame from camera, or None if capture fails.
            Frame is mirrored horizontally if mirror=True.
        """
        if self.cap is None:
            return None

        ret, frame = self.cap.read()

        # OpenCV can return None or empty frame on failure
        if not ret or frame is None or frame.size == 0:  # type: ignore[reportUnnecessaryComparison]
            return None

        # Update FPS tracking
        self._frame_count += 1
        current_time = time.time()

        # Calculate FPS every second
        if self._start_time and current_time - self._last_fps_calc >= 1.0:
            elapsed = current_time - self._start_time
            self._current_fps = self._frame_count / elapsed if elapsed > 0 else 0.0
            self._last_fps_calc = current_time

        # Mirror frame if requested
        if self.mirror:
            frame = cv2.flip(frame, 1)

        return frame

    def get_fps(self) -> float:
        """Get current frames per second.

        Returns:
            Current FPS based on actual frame timing.
        """
        return self._current_fps

    def stop(self) -> None:
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def is_running(self) -> bool:
        """Check if video capture is active.

        Returns:
            True if camera is open and ready.
        """
        return self.cap is not None and self.cap.isOpened()

    def __enter__(self) -> Self:
        """Context manager entry - start capture."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit - cleanup resources."""
        self.stop()
        cv2.destroyAllWindows()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.stop()

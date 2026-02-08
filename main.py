"""Gesture recognition demo application.

This is the main entry point for running gesture recognition with live video feed.
Press 'q' to quit.

Phase 3: Now uses GestureRecognizer with temporal smoothing for stable detection.
"""

import logging

import cv2

from src.gesture_recognizer import GestureRecognizer
from src.video_processor import VideoProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the gesture recognition demo."""
    logger.info("Starting gesture recognition demo (Phase 3 - Temporal Smoothing)...")
    logger.info("Press 'q' to quit")

    # Initialize recognizer (handles tracking, detection, smoothing, visualization)
    recognizer = GestureRecognizer()

    try:
        with VideoProcessor() as video:
            logger.info(f"Camera initialized: {video.width}x{video.height}")

            while True:
                # Capture frame
                frame = video.get_frame()
                if frame is None:
                    logger.warning("Failed to capture frame")
                    break

                # Process frame (all-in-one API call)
                result = recognizer.process_frame(frame)

                # Log gesture changes
                if result.gesture:
                    logger.info(
                        f"Gesture: {result.gesture.name} "
                        f"(confidence: {result.gesture.confidence:.2f}, "
                        f"stable: {result.is_stable})"
                    )

                # Display annotated frame
                cv2.imshow(
                    "Gesture Recognition - Phase 3 (Temporal Smoothing)",
                    result.frame_with_annotations,
                )

                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Quit key pressed")
                    break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
    finally:
        # Cleanup
        recognizer.close()
        cv2.destroyAllWindows()
        logger.info("Demo ended")


if __name__ == "__main__":
    main()

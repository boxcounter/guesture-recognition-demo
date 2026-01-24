"""Gesture recognition demo application.

This is the main entry point for running gesture recognition with live video feed.
Press 'q' to quit.
"""

import logging

import cv2

from src.gesture_detector import GestureDetector
from src.hand_tracker import HandTracker
from src.video_processor import VideoProcessor
from src.visualizer import Visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the gesture recognition demo."""
    logger.info("Starting gesture recognition demo...")
    logger.info("Press 'q' to quit")

    # Initialize components
    tracker = HandTracker()
    detector = GestureDetector()
    visualizer = Visualizer()

    try:
        with VideoProcessor() as video:
            logger.info(f"Camera initialized: {video.width}x{video.height}")

            while True:
                # Capture frame
                frame = video.get_frame()
                if frame is None:
                    logger.warning("Failed to capture frame")
                    break

                # Detect hands
                hands = tracker.process_frame(frame)

                # Process each detected hand
                for hand_data in hands:
                    # Detect gesture
                    gesture_result = detector.detect(hand_data)

                    # Draw landmarks and gesture
                    visualizer.draw_landmarks(frame, hand_data)
                    visualizer.draw_gesture_label(
                        frame,
                        gesture_result.gesture.value,
                        gesture_result.confidence,
                    )

                # Draw status and FPS
                visualizer.draw_hand_status(frame, hand_detected=len(hands) > 0)
                visualizer.draw_fps(frame, video.get_fps())

                # Display frame
                cv2.imshow("Gesture Recognition - Phase 2", frame)

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
        tracker.close()
        cv2.destroyAllWindows()
        logger.info("Demo ended")


if __name__ == "__main__":
    main()

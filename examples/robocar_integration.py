"""Example: Robocar integration with gesture recognition.

This example demonstrates how to integrate the gesture recognition system
with a robocar control system. The API is simple and flexible.

Usage:
    python examples/robocar_integration.py
"""

import logging
import time

import cv2

from src.gesture_recognizer import GestureRecognizer
from src.video_processor import VideoProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Mock robot class (replace with your actual robot API)
class MockRobot:
    """Mock robot for demonstration purposes."""

    def __init__(self):
        self.speed = 0
        self.direction = "stopped"
        logger.info("Robot initialized")

    def stop(self):
        """Stop the robot."""
        self.speed = 0
        self.direction = "stopped"
        logger.info("🛑 ROBOT: STOP")

    def go(self):
        """Start moving forward."""
        self.speed = 50
        self.direction = "forward"
        logger.info("🚗 ROBOT: GO FORWARD (speed: 50)")

    def turn(self, direction: str):
        """Turn in specified direction.

        Args:
            direction: "up", "down", "left", or "right"
        """
        self.direction = direction
        logger.info(f"🔄 ROBOT: TURN {direction.upper()}")

    def increase_speed(self):
        """Increase speed by 10."""
        self.speed = min(100, self.speed + 10)
        logger.info(f"⚡ ROBOT: SPEED UP to {self.speed}")

    def decrease_speed(self):
        """Decrease speed by 10."""
        self.speed = max(0, self.speed - 10)
        logger.info(f"🐌 ROBOT: SLOW DOWN to {self.speed}")

    def status(self) -> str:
        """Get current robot status."""
        return f"Speed: {self.speed}, Direction: {self.direction}"


def main():
    """Run robocar control with gesture recognition."""
    logger.info("=== Robocar Gesture Control Demo ===")
    logger.info("Gestures:")
    logger.info("  🖐️  Palm Forward → STOP")
    logger.info("  ✊ Fist → GO")
    logger.info("  ☝️  Pointing → TURN")
    logger.info("  👍 Thumbs Up → SPEED UP")
    logger.info("  👎 Thumbs Down → SLOW DOWN")
    logger.info("Press 'q' to quit\n")

    # Initialize robot and recognizer
    robot = MockRobot()
    recognizer = GestureRecognizer()
    last_gesture = None

    try:
        with VideoProcessor() as video:
            logger.info(f"Camera ready: {video.width}x{video.height}\n")

            while True:
                # Get frame from camera
                frame = video.get_frame()
                if frame is None:
                    break

                # Process frame with gesture recognition
                result = recognizer.process_frame(frame)

                # Handle gesture commands (only on gesture change for stability)
                if result.gesture and result.gesture.name != last_gesture:
                    gesture_name = result.gesture.name
                    confidence = result.gesture.confidence

                    # Only act on high-confidence gestures
                    if confidence >= 0.7:
                        if gesture_name == "PALM_FORWARD":
                            robot.stop()

                        elif gesture_name == "FIST":
                            robot.go()

                        elif gesture_name.startswith("POINTING_"):
                            direction = result.gesture.direction
                            robot.turn(direction)

                        elif gesture_name == "THUMBS_UP":
                            robot.increase_speed()

                        elif gesture_name == "THUMBS_DOWN":
                            robot.decrease_speed()

                        last_gesture = gesture_name

                # Reset if no stable gesture
                elif not result.gesture:
                    last_gesture = None

                # Display video with annotations
                # Add robot status overlay
                status_text = robot.status()
                cv2.putText(
                    result.frame_with_annotations,
                    status_text,
                    (10, result.frame_with_annotations.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

                cv2.imshow("Robocar Gesture Control", result.frame_with_annotations)

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    robot.stop()
                    logger.info("Shutting down...")
                    break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        robot.stop()
    finally:
        recognizer.close()
        cv2.destroyAllWindows()
        logger.info("Demo ended")


# Alternative pattern: Callback-based
class GestureRobotController:
    """Example of callback-based gesture control.

    This pattern is useful when you want to decouple gesture recognition
    from robot control logic.
    """

    def __init__(self, robot, recognizer):
        self.robot = robot
        self.recognizer = recognizer
        self.last_gesture = None

    def on_gesture_detected(self, gesture_name: str, confidence: float):
        """Callback when gesture is detected.

        Args:
            gesture_name: Name of detected gesture.
            confidence: Confidence score (0-1).
        """
        if confidence < 0.7:
            return

        # Map gestures to robot commands
        gesture_actions = {
            "PALM_FORWARD": self.robot.stop,
            "FIST": self.robot.go,
            "THUMBS_UP": self.robot.increase_speed,
            "THUMBS_DOWN": self.robot.decrease_speed,
        }

        # Handle pointing gestures
        if gesture_name.startswith("POINTING_"):
            direction = gesture_name.split("_")[1].lower()
            self.robot.turn(direction)
        elif gesture_name in gesture_actions:
            gesture_actions[gesture_name]()

    def process_frame(self, frame):
        """Process frame and trigger callbacks.

        Args:
            frame: Video frame to process.

        Returns:
            Recognition result.
        """
        result = self.recognizer.process_frame(frame)

        # Trigger callback on gesture change
        if result.gesture:
            if result.gesture.name != self.last_gesture:
                self.on_gesture_detected(
                    result.gesture.name, result.gesture.confidence
                )
                self.last_gesture = result.gesture.name
        else:
            self.last_gesture = None

        return result


if __name__ == "__main__":
    main()

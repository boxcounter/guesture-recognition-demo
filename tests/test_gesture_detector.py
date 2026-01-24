"""Unit tests for gesture detection algorithms."""

import pytest

from config import settings
from src.gesture_detector import GestureDetector, GestureType
from src.hand_tracker import HandData, Landmark


def create_landmark(x: float, y: float, z: float, visibility: float = 1.0) -> Landmark:
    """Helper to create a landmark for testing."""
    return Landmark(x=x, y=y, z=z, visibility=visibility)


def create_test_hand_data(landmarks: list[Landmark], confidence: float = 1.0) -> HandData:
    """Helper to create HandData for testing."""
    return HandData(landmarks=landmarks, handedness="Right", confidence=confidence)


class TestGestureDetector:
    """Test suite for GestureDetector."""

    @pytest.fixture
    def detector(self):
        """Create a GestureDetector instance for testing."""
        return GestureDetector()

    def test_fist_detection(self, detector):
        """Test that a fist gesture is detected correctly."""
        # Create landmarks for a fist (all fingers curled, thumb not extended)
        # For a fist, fingertips should curl back toward palm, at similar distance to wrist as MCPs
        landmarks = [create_landmark(0.5, 0.5, 0.0)]  # Wrist

        # Thumb (not extended - curled into palm)
        landmarks.extend([
            create_landmark(0.48, 0.48, 0.0),  # Thumb CMC
            create_landmark(0.46, 0.47, 0.0),  # Thumb MCP (distance ~0.057 from wrist)
            create_landmark(0.45, 0.46, 0.0),  # Thumb IP
            create_landmark(0.44, 0.46, 0.0),  # Thumb TIP (distance ~0.063, only 10% more than MCP)
        ])

        # Index finger (curled - tip curls back toward palm)
        landmarks.extend([
            create_landmark(0.55, 0.48, 0.0),  # Index MCP (distance ~0.054 from wrist)
            create_landmark(0.57, 0.47, 0.0),  # Index PIP
            create_landmark(0.58, 0.48, 0.0),  # Index DIP
            create_landmark(0.56, 0.49, 0.0),  # Index TIP (distance ~0.061, only 13% more)
        ])

        # Middle finger (curled)
        landmarks.extend([
            create_landmark(0.56, 0.45, 0.0),  # Middle MCP (distance ~0.078 from wrist)
            create_landmark(0.58, 0.43, 0.0),  # Middle PIP
            create_landmark(0.59, 0.44, 0.0),  # Middle DIP
            create_landmark(0.58, 0.45, 0.0),  # Middle TIP (distance ~0.094, only 20% more)
        ])

        # Ring finger (curled)
        landmarks.extend([
            create_landmark(0.54, 0.43, 0.0),  # Ring MCP (distance ~0.081 from wrist)
            create_landmark(0.55, 0.41, 0.0),  # Ring PIP
            create_landmark(0.56, 0.42, 0.0),  # Ring DIP
            create_landmark(0.55, 0.43, 0.0),  # Ring TIP (distance ~0.094, only 16% more)
        ])

        # Pinky finger (curled)
        landmarks.extend([
            create_landmark(0.51, 0.42, 0.0),  # Pinky MCP (distance ~0.080 from wrist)
            create_landmark(0.52, 0.40, 0.0),  # Pinky PIP
            create_landmark(0.52, 0.41, 0.0),  # Pinky DIP
            create_landmark(0.51, 0.42, 0.0),  # Pinky TIP (distance ~0.080, same as MCP)
        ])

        hand_data = create_test_hand_data(landmarks)
        result = detector.detect(hand_data)

        assert result.gesture == GestureType.FIST
        assert result.confidence >= settings.MIN_CONFIDENCE_FIST

    def test_palm_forward_detection(self, detector):
        """Test that a palm forward gesture is detected correctly."""
        # Create landmarks for palm forward (all fingers extended, palm facing camera)
        landmarks = [create_landmark(0.5, 0.5, 0.0)]  # Wrist
        
        # Thumb
        landmarks.extend([
            create_landmark(0.45, 0.48, 0.02),
            create_landmark(0.42, 0.46, 0.03),
            create_landmark(0.4, 0.44, 0.04),
            create_landmark(0.38, 0.42, 0.05),
        ])
        
        # Index finger (extended upward)
        landmarks.extend([
            create_landmark(0.52, 0.45, -0.03),
            create_landmark(0.52, 0.4, -0.04),
            create_landmark(0.52, 0.35, -0.05),
            create_landmark(0.52, 0.3, -0.06),
        ])
        
        # Middle finger (extended upward)
        landmarks.extend([
            create_landmark(0.5, 0.45, -0.03),
            create_landmark(0.5, 0.4, -0.04),
            create_landmark(0.5, 0.35, -0.05),
            create_landmark(0.5, 0.28, -0.06),
        ])
        
        # Ring finger (extended upward)
        landmarks.extend([
            create_landmark(0.48, 0.45, -0.03),
            create_landmark(0.48, 0.4, -0.04),
            create_landmark(0.48, 0.35, -0.05),
            create_landmark(0.48, 0.3, -0.06),
        ])
        
        # Pinky finger (extended upward)
        landmarks.extend([
            create_landmark(0.46, 0.45, -0.03),
            create_landmark(0.46, 0.4, -0.04),
            create_landmark(0.46, 0.37, -0.05),
            create_landmark(0.46, 0.32, -0.06),
        ])
        
        hand_data = create_test_hand_data(landmarks)
        result = detector.detect(hand_data)
        
        assert result.gesture == GestureType.PALM_FORWARD
        assert result.confidence >= settings.MIN_CONFIDENCE_PALM

    def test_pointing_up_detection(self, detector):
        """Test that pointing up gesture is detected correctly."""
        # Create landmarks for pointing up (index extended upward, others curled)
        landmarks = [create_landmark(0.5, 0.5, 0.0)]  # Wrist

        # Thumb (can be in any position for pointing)
        landmarks.extend([
            create_landmark(0.48, 0.48, 0.0),
            create_landmark(0.46, 0.47, 0.0),
            create_landmark(0.45, 0.46, 0.0),
            create_landmark(0.44, 0.46, 0.0),
        ])

        # Index finger (extended upward - tip much farther from wrist than MCP)
        landmarks.extend([
            create_landmark(0.52, 0.45, 0.0),  # Index MCP (distance ~0.055 from wrist)
            create_landmark(0.52, 0.38, 0.0),  # Index PIP
            create_landmark(0.52, 0.31, 0.0),  # Index DIP
            create_landmark(0.52, 0.22, 0.0),  # Index TIP (distance ~0.28, 5x MCP distance)
        ])

        # Middle finger (curled - tip close to MCP distance)
        landmarks.extend([
            create_landmark(0.5, 0.44, 0.0),   # Middle MCP (distance ~0.061 from wrist)
            create_landmark(0.49, 0.43, 0.0),  # Middle PIP
            create_landmark(0.49, 0.44, 0.0),  # Middle DIP
            create_landmark(0.49, 0.45, 0.0),  # Middle TIP (distance ~0.051, closer than MCP)
        ])

        # Ring finger (curled)
        landmarks.extend([
            create_landmark(0.48, 0.44, 0.0),  # Ring MCP (distance ~0.061 from wrist)
            create_landmark(0.47, 0.44, 0.0),  # Ring PIP
            create_landmark(0.47, 0.45, 0.0),  # Ring DIP
            create_landmark(0.47, 0.46, 0.0),  # Ring TIP (distance ~0.051, closer than MCP)
        ])

        # Pinky finger (curled)
        landmarks.extend([
            create_landmark(0.46, 0.45, 0.0),  # Pinky MCP (distance ~0.057 from wrist)
            create_landmark(0.45, 0.46, 0.0),  # Pinky PIP
            create_landmark(0.45, 0.47, 0.0),  # Pinky DIP
            create_landmark(0.45, 0.48, 0.0),  # Pinky TIP (distance ~0.057, same as MCP)
        ])

        hand_data = create_test_hand_data(landmarks)
        result = detector.detect(hand_data)

        assert result.gesture == GestureType.POINTING_UP
        assert result.confidence >= settings.MIN_CONFIDENCE_POINTING

    def test_thumbs_up_detection(self, detector):
        """Test that thumbs up gesture is detected correctly."""
        # Create landmarks for thumbs up (thumb extended upward, others curled)
        landmarks = [create_landmark(0.5, 0.5, 0.0)]  # Wrist

        # Thumb (extended upward - tip significantly farther from wrist than MCP)
        landmarks.extend([
            create_landmark(0.48, 0.48, 0.0),  # Thumb CMC
            create_landmark(0.46, 0.45, 0.0),  # Thumb MCP (distance ~0.064 from wrist)
            create_landmark(0.44, 0.40, 0.0),  # Thumb IP (y=0.40)
            create_landmark(0.42, 0.35, 0.0),  # Thumb TIP (y=0.35, distance ~0.17, 2.6x MCP)
        ])
        # Vertical separation: IP.y - TIP.y = 0.40 - 0.35 = 0.05 > 0.03 threshold ✓

        # Index through pinky (all curled - tips curl back toward palm)
        # Index finger
        landmarks.extend([
            create_landmark(0.55, 0.45, 0.0),  # Index MCP (distance ~0.071 from wrist)
            create_landmark(0.56, 0.44, 0.0),  # Index PIP
            create_landmark(0.56, 0.45, 0.0),  # Index DIP
            create_landmark(0.55, 0.46, 0.0),  # Index TIP (distance ~0.066, less than MCP)
        ])

        # Middle finger
        landmarks.extend([
            create_landmark(0.57, 0.44, 0.0),  # Middle MCP (distance ~0.083 from wrist)
            create_landmark(0.58, 0.43, 0.0),  # Middle PIP
            create_landmark(0.58, 0.44, 0.0),  # Middle DIP
            create_landmark(0.57, 0.45, 0.0),  # Middle TIP (distance ~0.078, less than MCP)
        ])

        # Ring finger
        landmarks.extend([
            create_landmark(0.56, 0.42, 0.0),  # Ring MCP (distance ~0.089 from wrist)
            create_landmark(0.57, 0.41, 0.0),  # Ring PIP
            create_landmark(0.57, 0.42, 0.0),  # Ring DIP
            create_landmark(0.56, 0.43, 0.0),  # Ring TIP (distance ~0.081, less than MCP)
        ])

        # Pinky finger
        landmarks.extend([
            create_landmark(0.53, 0.41, 0.0),  # Pinky MCP (distance ~0.092 from wrist)
            create_landmark(0.54, 0.40, 0.0),  # Pinky PIP
            create_landmark(0.54, 0.41, 0.0),  # Pinky DIP
            create_landmark(0.53, 0.42, 0.0),  # Pinky TIP (distance ~0.084, less than MCP)
        ])

        hand_data = create_test_hand_data(landmarks)
        result = detector.detect(hand_data)

        assert result.gesture == GestureType.THUMBS_UP
        assert result.confidence >= settings.MIN_CONFIDENCE_THUMBS

    def test_unknown_gesture(self, detector):
        """Test that ambiguous hand positions return UNKNOWN."""
        # Create landmarks for an ambiguous position
        landmarks = [create_landmark(0.5, 0.5, 0.0)]
        
        # Create 20 more landmarks in neutral positions
        for i in range(20):
            landmarks.append(create_landmark(0.5 + i * 0.01, 0.5, 0.0))
        
        hand_data = create_test_hand_data(landmarks)
        result = detector.detect(hand_data)
        
        assert result.gesture == GestureType.UNKNOWN
        assert result.confidence == 0.0

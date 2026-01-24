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
        # Wrist at origin
        landmarks = [create_landmark(0.5, 0.5, 0.0)]  # Wrist
        
        # Thumb (not extended - close to wrist)
        landmarks.extend([
            create_landmark(0.52, 0.48, 0.0),  # Thumb CMC
            create_landmark(0.53, 0.47, 0.0),  # Thumb MCP
            create_landmark(0.54, 0.46, 0.0),  # Thumb IP
            create_landmark(0.55, 0.45, 0.0),  # Thumb TIP
        ])
        
        # Index finger (curled)
        landmarks.extend([
            create_landmark(0.55, 0.5, 0.0),   # Index MCP
            create_landmark(0.56, 0.49, 0.0),  # Index PIP
            create_landmark(0.57, 0.48, 0.0),  # Index DIP
            create_landmark(0.58, 0.47, 0.0),  # Index TIP
        ])
        
        # Middle finger (curled)
        landmarks.extend([
            create_landmark(0.6, 0.5, 0.0),    # Middle MCP
            create_landmark(0.61, 0.49, 0.0),  # Middle PIP
            create_landmark(0.62, 0.48, 0.0),  # Middle DIP
            create_landmark(0.63, 0.47, 0.0),  # Middle TIP
        ])
        
        # Ring finger (curled)
        landmarks.extend([
            create_landmark(0.65, 0.5, 0.0),   # Ring MCP
            create_landmark(0.66, 0.49, 0.0),  # Ring PIP
            create_landmark(0.67, 0.48, 0.0),  # Ring DIP
            create_landmark(0.68, 0.47, 0.0),  # Ring TIP
        ])
        
        # Pinky finger (curled)
        landmarks.extend([
            create_landmark(0.7, 0.5, 0.0),    # Pinky MCP
            create_landmark(0.71, 0.49, 0.0),  # Pinky PIP
            create_landmark(0.72, 0.48, 0.0),  # Pinky DIP
            create_landmark(0.73, 0.47, 0.0),  # Pinky TIP
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
        # Create landmarks for pointing up (index extended, others curled)
        landmarks = [create_landmark(0.5, 0.5, 0.0)]  # Wrist
        
        # Thumb (curled)
        landmarks.extend([
            create_landmark(0.52, 0.48, 0.0),
            create_landmark(0.53, 0.47, 0.0),
            create_landmark(0.54, 0.46, 0.0),
            create_landmark(0.55, 0.45, 0.0),
        ])
        
        # Index finger (extended upward)
        landmarks.extend([
            create_landmark(0.52, 0.45, 0.0),
            create_landmark(0.52, 0.4, 0.0),
            create_landmark(0.52, 0.35, 0.0),
            create_landmark(0.52, 0.25, 0.0),  # Extended far up
        ])
        
        # Middle finger (curled)
        landmarks.extend([
            create_landmark(0.5, 0.45, 0.0),
            create_landmark(0.5, 0.44, 0.0),
            create_landmark(0.5, 0.43, 0.0),
            create_landmark(0.5, 0.42, 0.0),
        ])
        
        # Ring finger (curled)
        landmarks.extend([
            create_landmark(0.48, 0.45, 0.0),
            create_landmark(0.48, 0.44, 0.0),
            create_landmark(0.48, 0.43, 0.0),
            create_landmark(0.48, 0.42, 0.0),
        ])
        
        # Pinky finger (curled)
        landmarks.extend([
            create_landmark(0.46, 0.45, 0.0),
            create_landmark(0.46, 0.44, 0.0),
            create_landmark(0.46, 0.43, 0.0),
            create_landmark(0.46, 0.42, 0.0),
        ])
        
        hand_data = create_test_hand_data(landmarks)
        result = detector.detect(hand_data)
        
        assert result.gesture == GestureType.POINTING_UP
        assert result.confidence >= settings.MIN_CONFIDENCE_POINTING

    def test_thumbs_up_detection(self, detector):
        """Test that thumbs up gesture is detected correctly."""
        # Create landmarks for thumbs up (thumb extended up, others curled)
        landmarks = [create_landmark(0.5, 0.5, 0.0)]  # Wrist
        
        # Thumb (extended upward)
        landmarks.extend([
            create_landmark(0.48, 0.48, 0.0),
            create_landmark(0.46, 0.45, 0.0),
            create_landmark(0.44, 0.4, 0.0),   # Thumb IP
            create_landmark(0.42, 0.35, 0.0),  # Thumb TIP (above IP)
        ])
        
        # Index through pinky (all curled)
        for i in range(4):
            base_x = 0.52 + i * 0.02
            landmarks.extend([
                create_landmark(base_x, 0.45, 0.0),
                create_landmark(base_x, 0.44, 0.0),
                create_landmark(base_x, 0.43, 0.0),
                create_landmark(base_x, 0.42, 0.0),
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

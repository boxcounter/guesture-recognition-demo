"""Unit tests for gesture recognizer and temporal smoothing."""

import pytest

from src.gesture_detector import GestureType
from src.gesture_recognizer import GestureSmoother


class TestGestureSmoother:
    """Test suite for temporal smoothing."""

    @pytest.fixture
    def smoother(self):
        """Create a GestureSmoother with default settings."""
        # history_size=5, min_consistent_frames=3
        return GestureSmoother(history_size=5, min_consistent_frames=3)

    def test_initial_state_returns_unknown(self, smoother):
        """Test that smoother returns UNKNOWN until enough history."""
        # First frame
        gesture, confidence, is_stable = smoother.update(GestureType.FIST, 0.8)
        assert gesture == GestureType.UNKNOWN
        assert confidence == 0.0
        assert is_stable is False

        # Second frame
        gesture, confidence, is_stable = smoother.update(GestureType.FIST, 0.8)
        assert gesture == GestureType.UNKNOWN
        assert is_stable is False

    def test_consistent_gesture_becomes_stable(self, smoother):
        """Test that consistent gestures become stable after 3 frames."""
        # First 2 frames: not stable yet
        smoother.update(GestureType.FIST, 0.8)
        smoother.update(GestureType.FIST, 0.85)

        # Third frame: should become stable
        gesture, confidence, is_stable = smoother.update(GestureType.FIST, 0.9)
        assert gesture == GestureType.FIST
        assert is_stable is True
        # Confidence should be average of 3 frames: (0.8 + 0.85 + 0.9) / 3 = 0.85
        assert abs(confidence - 0.85) < 0.01

    def test_inconsistent_gestures_remain_unstable(self, smoother):
        """Test that inconsistent gestures don't become stable."""
        smoother.update(GestureType.FIST, 0.8)
        smoother.update(GestureType.PALM_FORWARD, 0.7)
        smoother.update(GestureType.POINTING_UP, 0.9)

        # No gesture appears 3 times, should be UNKNOWN
        gesture, confidence, is_stable = smoother.update(GestureType.FIST, 0.8)
        assert gesture == GestureType.UNKNOWN
        assert is_stable is False

    def test_gesture_change_requires_new_consistency(self, smoother):
        """Test that changing gestures requires building new consistency."""
        # Establish FIST as stable
        smoother.update(GestureType.FIST, 0.8)
        smoother.update(GestureType.FIST, 0.8)
        gesture, _, is_stable = smoother.update(GestureType.FIST, 0.8)
        assert gesture == GestureType.FIST
        assert is_stable is True

        # Start showing PALM_FORWARD
        smoother.update(GestureType.PALM_FORWARD, 0.7)
        smoother.update(GestureType.PALM_FORWARD, 0.7)

        # History now: [FIST, FIST, FIST, PALM, PALM]
        # FIST appears 3 times, still stable
        gesture, _, is_stable = smoother.update(GestureType.FIST, 0.8)
        assert gesture == GestureType.FIST  # Still FIST (3/6 frames)

        # Add more PALM_FORWARD
        smoother.update(GestureType.PALM_FORWARD, 0.7)
        smoother.update(GestureType.PALM_FORWARD, 0.7)

        # History now: [FIST, PALM, PALM, FIST, PALM, PALM]
        # PALM appears 4 times, should switch
        gesture, _, is_stable = smoother.update(GestureType.PALM_FORWARD, 0.7)
        assert gesture == GestureType.PALM_FORWARD
        assert is_stable is True

    def test_majority_voting_with_noise(self, smoother):
        """Test that majority voting filters out noise."""
        # 4 FIST, 1 UNKNOWN (noise)
        smoother.update(GestureType.FIST, 0.8)
        smoother.update(GestureType.FIST, 0.8)
        smoother.update(GestureType.UNKNOWN, 0.0)  # Noise
        smoother.update(GestureType.FIST, 0.8)
        gesture, _, is_stable = smoother.update(GestureType.FIST, 0.8)

        # FIST appears 4/5 times (≥3), should be stable
        assert gesture == GestureType.FIST
        assert is_stable is True

    def test_confidence_averaging(self, smoother):
        """Test that confidence is averaged correctly."""
        smoother.update(GestureType.FIST, 0.7)
        smoother.update(GestureType.FIST, 0.8)
        smoother.update(GestureType.FIST, 0.9)
        smoother.update(GestureType.FIST, 1.0)
        gesture, confidence, is_stable = smoother.update(GestureType.FIST, 0.6)

        # All 5 frames are FIST
        # Average: (0.7 + 0.8 + 0.9 + 1.0 + 0.6) / 5 = 0.8
        assert gesture == GestureType.FIST
        assert is_stable is True
        assert abs(confidence - 0.8) < 0.01

    def test_reset_clears_history(self, smoother):
        """Test that reset clears all history."""
        # Build up history
        smoother.update(GestureType.FIST, 0.8)
        smoother.update(GestureType.FIST, 0.8)
        smoother.update(GestureType.FIST, 0.8)

        # Reset
        smoother.reset()

        # Should be back to initial state
        gesture, confidence, is_stable = smoother.update(GestureType.FIST, 0.8)
        assert gesture == GestureType.UNKNOWN
        assert confidence == 0.0
        assert is_stable is False

    def test_unknown_gesture_handling(self, smoother):
        """Test that UNKNOWN gestures are counted in voting."""
        # If most frames are UNKNOWN, it should report UNKNOWN as stable
        smoother.update(GestureType.UNKNOWN, 0.0)
        smoother.update(GestureType.UNKNOWN, 0.0)
        smoother.update(GestureType.UNKNOWN, 0.0)
        smoother.update(GestureType.FIST, 0.8)
        gesture, _, is_stable = smoother.update(GestureType.FIST, 0.8)

        # UNKNOWN appears 3/5 times, so it's "stably unknown"
        assert gesture == GestureType.UNKNOWN
        assert is_stable is True  # Stable at UNKNOWN (no clear gesture)

    def test_custom_thresholds(self):
        """Test smoother with custom threshold values."""
        # Require 4 out of 5 frames for stability
        smoother = GestureSmoother(history_size=5, min_consistent_frames=4)

        # 3 frames is not enough
        smoother.update(GestureType.FIST, 0.8)
        smoother.update(GestureType.FIST, 0.8)
        gesture, _, is_stable = smoother.update(GestureType.FIST, 0.8)
        assert is_stable is False

        # 4 frames should be enough
        gesture, _, is_stable = smoother.update(GestureType.FIST, 0.8)
        assert gesture == GestureType.FIST
        assert is_stable is True

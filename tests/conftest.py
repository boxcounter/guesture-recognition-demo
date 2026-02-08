"""Pytest configuration and shared fixtures."""

import pytest

from src.gesture_detector import GestureDetector
from src.hand_tracker import HandData, Landmark


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")


@pytest.fixture
def detector():
    """Create a GestureDetector instance for testing."""
    return GestureDetector()


def create_landmark(x: float, y: float, z: float, visibility: float = 1.0) -> Landmark:
    """Helper to create a landmark for testing."""
    return Landmark(x=x, y=y, z=z, visibility=visibility)


def create_test_hand_data(
    landmarks: list[Landmark], confidence: float = 1.0
) -> HandData:
    """Helper to create HandData for testing."""
    return HandData(landmarks=landmarks, handedness="Right", confidence=confidence)

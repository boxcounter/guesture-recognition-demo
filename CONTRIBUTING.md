# AI Coding Guide for Gesture Recognition Project

This guide provides detailed coding standards and patterns for AI assistants working on this codebase. Follow these guidelines when implementing features or making changes.

## Core Principles

1. **Keep it simple** - This is a demo project for robocar integration. Avoid over-engineering.
2. **Modern Python 3.12** - Use the latest language features and syntax.
3. **Type safety** - Full type hints with strict Pyright checking.
4. **Performance matters** - Real-time video processing requires efficient code.

## Python 3.12 Type Hints

### Always Use Modern Syntax

```python
# ✅ Use built-in generic types
def process_frames(frames: list[np.ndarray]) -> dict[str, float]:
    pass

# ✅ Use | for unions
def get_gesture(data: str | bytes | None) -> str | None:
    pass

# ✅ Use | None instead of Optional
def find_hand(image: np.ndarray) -> HandData | None:
    pass
```

### Never Use Old Syntax

```python
# ❌ NEVER use typing.List, Dict, Tuple, Optional, Union
from typing import List, Dict, Optional, Union  # ❌ Don't import these

def process_frames(frames: List[np.ndarray]) -> Dict[str, float]:  # ❌ Wrong
    pass

def get_gesture(data: Union[str, bytes, None]) -> Optional[str]:  # ❌ Wrong
    pass
```

### When to Import from typing

Only import from `typing` for advanced features:

```python
from typing import Protocol, TypeVar, Self, TypedDict, Literal, Callable

# Protocol for structural subtyping
class GestureDetector(Protocol):
    def detect(self, landmarks: list[Landmark]) -> str | None: ...

# Self for method chaining
class HandTracker:
    def with_confidence(self, value: float) -> Self:
        return self

# TypedDict for structured dicts
class Config(TypedDict):
    confidence: float
    smoothing_frames: int

# Literal for specific values
def set_mode(mode: Literal["train", "test", "demo"]) -> None:
    pass
```

### Type Aliases (Python 3.12)

```python
# ✅ Use type statement
type LandmarkList = list[tuple[float, float, float]]
type GestureCallback = Callable[[str, float], None]
```

## Modern Python Patterns to Use

### 1. Dataclasses with Slots

Use for all data structures. More efficient than regular classes:

```python
from dataclasses import dataclass

@dataclass(slots=True, frozen=True)  # frozen=True for immutability
class Landmark:
    x: float
    y: float
    z: float
    visibility: float

@dataclass(slots=True)
class GestureResult:
    gesture: str | None
    confidence: float
    hand_detected: bool
    timestamp: float
    landmarks: list[Landmark] | None = None
```

**Benefits:** 40-50% less memory, faster attribute access, prevents typos.

### 2. Enums for Gesture Types

ALWAYS use enums instead of string literals for gestures:

```python
from enum import Enum, auto

class GestureType(Enum):
    FIST = auto()
    PALM_FORWARD = auto()
    POINT_LEFT = auto()
    POINT_RIGHT = auto()
    POINT_UP = auto()
    POINT_DOWN = auto()
    THUMBS_UP = auto()
    THUMBS_DOWN = auto()

    def is_pointing(self) -> bool:
        return self.name.startswith("POINT_")

    def to_direction(self) -> str | None:
        """Get direction for pointing gestures."""
        if not self.is_pointing():
            return None
        return self.name.split("_")[1].lower()

# ✅ Usage
gesture = GestureType.FIST  # Type-safe, autocomplete works
if gesture == GestureType.PALM_FORWARD:
    robot.stop()
```

### 3. Pattern Matching

Use match/case for cleaner branching logic:

```python
# ✅ Use pattern matching
match gesture_type:
    case GestureType.FIST:
        robot.go()
    case GestureType.PALM_FORWARD:
        robot.stop()
    case GestureType.POINT_LEFT | GestureType.POINT_RIGHT as direction:
        robot.turn(direction.to_direction())
    case GestureType.THUMBS_UP:
        robot.increase_speed()
    case _:
        logger.warning(f"Unhandled gesture: {gesture_type}")

# ❌ Avoid long if/elif chains
if gesture_type == GestureType.FIST:
    robot.go()
elif gesture_type == GestureType.PALM_FORWARD:
    robot.stop()
# ... etc
```

### 4. Context Managers for Resources

Essential for camera and MediaPipe lifecycle:

```python
class VideoCapture:
    def __init__(self, index: int) -> None:
        self.index = index
        self.cap: cv2.VideoCapture | None = None

    def __enter__(self) -> Self:
        self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.index}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

# ✅ Usage
with VideoCapture(0) as cap:
    frame = cap.read()
# Automatically cleaned up
```

### 5. Pathlib over os.path

```python
from pathlib import Path

# ✅ Modern
config_path = Path(__file__).parent / "config" / "settings.py"
if config_path.exists():
    content = config_path.read_text()

# ❌ Old
import os.path
config_path = os.path.join(os.path.dirname(__file__), "config", "settings.py")
```

### 6. Walrus Operator

Reduces duplication in conditionals:

```python
# ✅ With walrus
if (result := detector.detect(landmarks)) and result.confidence > 0.8:
    handle_gesture(result)

# ❌ Without (verbose)
result = detector.detect(landmarks)
if result and result.confidence > 0.8:
    handle_gesture(result)
```

## Code Organization

### Module Structure

```python
"""Module docstring describing purpose."""

from __future__ import annotations  # Only if needed for forward refs

# Standard library
import math
from pathlib import Path

# Third-party
import cv2
import numpy as np

# Local
from config.settings import Settings
from src.types import Landmark

# Constants
DEFAULT_CONFIDENCE = 0.7

# Type aliases
type LandmarkList = list[Landmark]

# Implementation...
```

### Class Structure

```python
class GestureDetector:
    """Detect gestures from hand landmarks."""

    # Class constants
    SUPPORTED_GESTURES = [GestureType.FIST, GestureType.PALM_FORWARD]

    def __init__(self, config: Config) -> None:
        """Initialize with configuration."""
        self.config = config
        self._cache: dict[str, float] = {}

    # Public methods
    def detect(self, landmarks: list[Landmark]) -> GestureResult | None:
        """Main public API."""
        pass

    # Private helpers
    def _calculate_confidence(self, data: Any) -> float:
        """Internal calculation."""
        pass

    # Dunder methods last
    def __repr__(self) -> str:
        return f"GestureDetector(config={self.config})"
```

## Naming Conventions

```python
# Classes: PascalCase
class GestureRecognizer:
    pass

# Functions/methods: snake_case
def detect_hand_landmarks() -> list[Landmark]:
    pass

# Constants: UPPER_SNAKE_CASE
MAX_HANDS = 2
DEFAULT_CONFIDENCE = 0.7

# Private: _leading_underscore
def _internal_helper() -> None:
    pass

# Type aliases: PascalCase (legacy) or use 'type' statement
type LandmarkList = list[Landmark]
```

## Configuration Management

ALL magic numbers and thresholds go in `config/settings.py`:

```python
# ❌ Bad - hardcoded
if confidence > 0.8:
    return gesture

if distance < 0.05:
    fingers_touching = True

# ✅ Good - configurable
from config.settings import MIN_GESTURE_CONFIDENCE, FINGER_TOUCH_THRESHOLD

if confidence > MIN_GESTURE_CONFIDENCE:
    return gesture

if distance < FINGER_TOUCH_THRESHOLD:
    fingers_touching = True
```

## Error Handling

```python
# ✅ Specific exceptions
try:
    landmarks = tracker.process_frame(frame)
except MediaPipeError as e:
    logger.error(f"MediaPipe processing failed: {e}")
    return None
except CameraError as e:
    logger.error(f"Camera error: {e}")
    raise

# ✅ Context managers for cleanup
with VideoCapture(0) as cap:
    process(cap)

# ❌ Avoid bare except
try:
    process()
except:  # ❌ Too broad
    pass
```

## Logging

Use standard library logging - keep it simple:

```python
import logging

logger = logging.getLogger(__name__)

# ✅ Good - f-strings with context
logger.info(f"Detected {gesture_type.name} with confidence {confidence:.2f}")
logger.warning(f"Low confidence detection: {confidence:.2f} < {threshold}")

# ✅ Good - use appropriate levels
logger.debug("Frame processing took 23ms")
logger.info("Hand tracking initialized")
logger.warning("Camera framerate dropped to 15 FPS")
logger.error("Failed to initialize MediaPipe", exc_info=True)

# ❌ Don't use print statements
print("Gesture detected")  # ❌ Use logger instead
```

## Docstrings

Use Google-style for public APIs:

```python
def detect_gesture(
    landmarks: list[Landmark],
    threshold: float = 0.8
) -> GestureResult | None:
    """Detect gesture from hand landmarks.

    Args:
        landmarks: List of 21 hand landmarks from MediaPipe.
        threshold: Minimum confidence threshold (0.0 to 1.0).

    Returns:
        GestureResult if detected, None otherwise.

    Raises:
        ValueError: If landmarks list length != 21.
    """
    pass
```

One-liners for simple functions:

```python
def _calculate_distance(p1: Landmark, p2: Landmark) -> float:
    """Calculate Euclidean distance between two landmarks."""
    pass
```

## Performance Guidelines

### Hot Loops

```python
# ✅ Reuse arrays in video processing loops
frame_buffer = np.zeros((720, 1280, 3), dtype=np.uint8)

while running:
    # Reuse buffer instead of allocating new
    cap.read(frame_buffer)
    process(frame_buffer)

# ❌ Allocating in loop
while running:
    frame = cap.read()  # New allocation each iteration
```

### Avoid Premature Optimization

```python
# ✅ Start with clear, readable code
def calculate_angle(p1: Landmark, p2: Landmark, p3: Landmark) -> float:
    """Calculate angle at p2."""
    v1 = np.array([p1.x - p2.x, p1.y - p2.y])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y])
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

# Only optimize if profiling shows it's a bottleneck
```

## Testing

### Test Structure

```python
# tests/test_gesture_detector.py
import pytest
from src.gesture_detector import GestureDetector, GestureType

class TestGestureDetector:
    """Tests for GestureDetector."""

    @pytest.fixture
    def detector(self) -> GestureDetector:
        """Create detector for testing."""
        return GestureDetector()

    def test_detect_fist(self, detector: GestureDetector) -> None:
        """Test fist gesture detection."""
        landmarks = create_fist_landmarks()
        result = detector.detect(landmarks)
        assert result is not None
        assert result.gesture == GestureType.FIST
        assert result.confidence > 0.7

    @pytest.mark.parametrize("gesture_type,landmarks_func", [
        (GestureType.FIST, create_fist_landmarks),
        (GestureType.PALM_FORWARD, create_palm_landmarks),
        (GestureType.THUMBS_UP, create_thumbs_up_landmarks),
    ])
    def test_gesture_detection(
        self,
        detector: GestureDetector,
        gesture_type: GestureType,
        landmarks_func: Callable[[], list[Landmark]]
    ) -> None:
        """Test detection of various gestures."""
        landmarks = landmarks_func()
        result = detector.detect(landmarks)
        assert result is not None
        assert result.gesture == gesture_type
```

### What to Test

- **Core algorithms** - Gesture detection logic
- **Edge cases** - No hand, partial hand, ambiguous gestures
- **Error handling** - Invalid inputs, missing data
- **Integration** - Full pipeline from frame to gesture

### What NOT to Test

- **Third-party libraries** - Don't test MediaPipe or OpenCV
- **Simple getters/setters** - Not worth the boilerplate
- **UI/visualization** - Manual testing is fine

## Architecture

### Module Responsibilities

Each module has ONE clear job:

- `hand_tracker.py` - MediaPipe integration ONLY
- `gesture_detector.py` - Detection algorithms ONLY
- `gesture_recognizer.py` - Orchestration and smoothing
- `video_processor.py` - Camera/video handling ONLY
- `visualizer.py` - Drawing and annotations ONLY

### Dependency Direction

```
main.py
  ↓
gesture_recognizer.py (high-level orchestration)
  ↓                    ↓
hand_tracker.py    gesture_detector.py
  ↓                    ↓
MediaPipe          Pure algorithms
```

**Rule:** Low-level modules never import high-level modules.

## What to Avoid

### Don't Over-Engineer

```python
# ❌ Over-engineered for a demo
class AbstractGestureDetectorFactory:
    def create_detector(self, type: str) -> AbstractGestureDetector:
        ...

# ✅ Keep it simple
class GestureDetector:
    def detect(self, landmarks: list[Landmark]) -> GestureResult | None:
        ...
```

### Don't Add Unused Features

```python
# ❌ Don't add "just in case"
class GestureRecognizer:
    def detect(self, landmarks): ...
    def train_model(self, data): ...  # ❌ Not needed for demo
    def export_to_onnx(self): ...      # ❌ Not needed
    def visualize_3d(self): ...         # ❌ Out of scope

# ✅ Only what's needed
class GestureRecognizer:
    def detect(self, landmarks): ...
```

### Don't Ignore Configuration

```python
# ❌ Hardcoded values scattered everywhere
if confidence > 0.75:  # Why 0.75?
    ...
if distance < 0.1:     # Why 0.1?
    ...

# ✅ Centralized configuration
from config.settings import (
    MIN_GESTURE_CONFIDENCE,
    FINGER_EXTENSION_THRESHOLD
)

if confidence > MIN_GESTURE_CONFIDENCE:
    ...
if distance < FINGER_EXTENSION_THRESHOLD:
    ...
```

## Summary Checklist

When writing code, verify:

- ✅ Modern type hints (| not Union, list not List)
- ✅ All types annotated (Pyright strict mode passes)
- ✅ Using GestureType enum, not strings
- ✅ Using dataclasses with slots for data structures
- ✅ Pattern matching for branching on gestures
- ✅ Context managers for resources (camera, MediaPipe)
- ✅ Configuration in settings.py, not hardcoded
- ✅ Standard logging, not print statements
- ✅ Pathlib for file paths
- ✅ Clear docstrings for public APIs
- ✅ Module does one thing only
- ✅ No premature optimization
- ✅ No over-engineering

When in doubt: **Keep it simple, keep it typed, keep it fast.**

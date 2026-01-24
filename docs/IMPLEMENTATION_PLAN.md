# Gesture Recognition Implementation Plan

## Overview
Build a real-time gesture recognition system using MediaPipe and OpenCV that:
- Runs on MacBook Pro with live camera feed
- Recognizes: Palm forward (Stop), Pointing directions, Fist (Go), Thumbs up/down
- Provides visual feedback with annotated video window
- Exposes clean API for future robocar integration

## Project Structure

```
gesture_recognition/
├── main.py                      # CLI entry point
├── requirements.txt             # Dependencies
├── config/
│   └── settings.py             # Configuration management
├── src/
│   ├── __init__.py
│   ├── hand_tracker.py         # MediaPipe wrapper
│   ├── gesture_detector.py     # Gesture detection algorithms
│   ├── gesture_recognizer.py   # Recognition engine with smoothing
│   ├── video_processor.py      # Camera capture pipeline
│   └── visualizer.py           # Annotation rendering
├── examples/
│   ├── basic_recognition.py
│   └── robocar_integration.py
└── tests/
    ├── test_gesture_detector.py
    └── test_integration.py
```

## Core Modules

### 1. config/settings.py
Centralized configuration for all parameters:
- Camera settings (index, resolution, FPS)
- MediaPipe confidence thresholds (detection: 0.7, tracking: 0.5)
- Gesture detection thresholds
- Visualization settings (colors, fonts)
- Smoothing parameters (history size: 5, min consistent frames: 3)

### 2. src/hand_tracker.py
MediaPipe Hands wrapper providing:
- Hand detection and landmark extraction (21 landmarks per hand)
- Normalized and pixel coordinates
- Left/right hand distinction
- Lifecycle management

**Key class:**
```python
class HandTracker:
    def process_frame(self, frame) -> List[HandData]
    def close()
```

**MediaPipe Landmarks (0-20):**
- 0: WRIST
- 1-4: THUMB (CMC, MCP, IP, TIP)
- 5-8: INDEX (MCP, PIP, DIP, TIP)
- 9-12: MIDDLE, 13-16: RING, 17-20: PINKY

### 3. src/gesture_detector.py
Implements detection algorithms using geometric analysis of landmarks:

**Palm Forward Detection:**
- All fingers extended (fingertips above PIP joints in Y-axis)
- Palm facing camera (Z-coordinate analysis: fingertips closer than palm center)
- Fingers spread apart

**Pointing Detection:**
- Only index finger extended
- Other fingers curled (tips below PIPs)
- Calculate angle from wrist to index tip
- Determine direction: RIGHT (0°±45°), UP (-90°±45°), LEFT (180°±45°), DOWN (90°±45°)

**Fist Detection:**
- All fingertips below PIP joints
- Calculate curl ratios (tip-to-wrist / MCP-to-wrist distance)
- All curl ratios < 1.3 threshold
- Thumb tucked in

**Thumbs Up/Down Detection:**
- Thumb extended (length > threshold)
- Other fingers curled
- Check thumb Y-coordinate relative to wrist:
  - UP: thumb above wrist (dy < -0.1)
  - DOWN: thumb below wrist (dy > 0.1)

**Key class:**
```python
class GestureDetector:
    def detect_palm_forward(self, landmarks) -> Tuple[bool, float]
    def detect_pointing(self, landmarks) -> Tuple[Optional[str], float]
    def detect_fist(self, landmarks) -> Tuple[bool, float]
    def detect_thumbs(self, landmarks) -> Tuple[Optional[str], float]
```

### 4. src/gesture_recognizer.py
High-level engine orchestrating the full pipeline:

**Responsibilities:**
- Coordinate hand tracking and gesture detection
- Temporal smoothing (5-frame history, require 3 consistent frames)
- Confidence aggregation
- Gesture state management
- Priority handling for multiple hands (default: right hand)

**API for robocar integration:**
```python
class GestureRecognizer:
    def __init__(self, config)
    def process_frame(self, frame) -> GestureResult
    def get_current_gesture(self) -> Optional[Gesture]
    def close()

@dataclass
class GestureResult:
    gesture: Optional[Gesture]          # Detected gesture
    confidence: float                   # Confidence score
    hand_detected: bool                 # Hand presence flag
    hand_landmarks: Optional[List]      # Landmark data
    frame_with_annotations: np.ndarray  # Annotated frame
```

**Temporal Smoothing:**
- Maintain deque of last 5 gesture detections
- Report gesture only when it appears in ≥3 of 5 frames
- Average confidence over consistent frames
- Prevents jittery switching between gestures

### 5. src/video_processor.py
Camera capture and frame management:
```python
class VideoProcessor:
    def __init__(self, camera_index, width, height, fps)
    def start()
    def get_frame() -> Optional[np.ndarray]
    def stop()
    def get_fps() -> float
```

### 6. src/visualizer.py
Rendering annotations on video frames:
```python
class Visualizer:
    def draw_landmarks(self, frame, landmarks)
    def draw_gesture_label(self, frame, gesture, confidence)
    def draw_fps(self, frame, fps)
```

### 7. main.py
CLI interface with command-line arguments:
```bash
python main.py [--camera INDEX] [--no-display] [--log-gestures]
```

Main loop:
1. Capture frame
2. Process with GestureRecognizer
3. Render annotations
4. Display window and log to console
5. Handle keyboard input ('q' to quit)

## Implementation Phases

### Phase 1: Foundation ✓
1. Create project structure (directories, __init__.py files)
2. Create requirements.txt:
   ```
   opencv-python==4.9.0.80
   mediapipe==0.10.9
   numpy==1.24.3
   ```
3. Implement config/settings.py with all configuration parameters
4. Implement src/video_processor.py for camera capture
5. Implement src/hand_tracker.py wrapping MediaPipe
6. Create basic main.py displaying camera with hand landmarks

**Verification:** Live video window shows hand landmarks overlaid on camera feed

### Phase 2: Gesture Detection Algorithms ✓
1. Implement src/gesture_detector.py with all 4 gesture algorithms
2. Add helper methods:
   - `calculate_distance(p1, p2)` - Euclidean distance
   - `calculate_angle(p1, p2, p3)` - Angle between three points
   - `is_finger_extended(landmarks, finger)` - Check extension state
3. Test each algorithm individually with debug prints
4. Fine-tune thresholds for each gesture

**Verification:** Each gesture can be detected (may have jitter/false positives at this stage)

### Phase 3: Recognition Engine ✓
1. Implement src/gesture_recognizer.py
2. Add GestureSmoother class for temporal filtering
3. Implement confidence aggregation
4. Add multi-hand prioritization logic
5. Define GestureResult and Gesture dataclasses

**Verification:** Gestures are stable without rapid switching, false positives reduced

### Phase 4: Visualization ✓
1. Implement src/visualizer.py
2. Draw hand landmarks with customizable colors
3. Render gesture labels with confidence percentages
4. Add FPS counter
5. Add status indicators (hand detected, tracking quality)

**Verification:** Professional-looking output with all information clearly displayed

### Phase 5: API & Integration ✓
1. Finalize GestureRecognizer API for external use
2. Create examples/basic_recognition.py showing simple usage
3. Create examples/robocar_integration.py demonstrating:
   - Synchronous frame processing
   - Callback-based gesture events
   - Command mapping (palm→stop, fist→go, point→turn)
4. Add command-line arguments to main.py
5. Update CLAUDE.md with architecture details

**Verification:** Example code runs successfully and is easy to understand

### Phase 6: Testing ✓
1. Write unit tests for gesture_detector.py (synthetic landmark data)
2. Create test video files for each gesture
3. Write integration tests with video files
4. Manual testing with live camera
5. Threshold tuning based on test results

**Verification:** All automated tests pass, manual testing shows reliable recognition

### Phase 7: Documentation ✓
1. Add comprehensive docstrings to all modules
2. Create README.md with:
   - Setup instructions
   - Usage examples
   - Gesture descriptions
   - Configuration guide
3. Update CLAUDE.md with final architecture
4. Create docs/API.md for robocar integration

**Verification:** New developer can set up and use the system from documentation alone

## Critical Files to Create

**Priority 1 (Core functionality):**
1. `src/hand_tracker.py` - Foundation for all hand detection
2. `src/gesture_detector.py` - Core gesture algorithms
3. `src/gesture_recognizer.py` - Main API and orchestration
4. `config/settings.py` - Centralized configuration
5. `main.py` - Entry point and demo

**Priority 2 (Enhanced functionality):**
6. `src/video_processor.py` - Camera management
7. `src/visualizer.py` - Annotations
8. `requirements.txt` - Dependencies

**Priority 3 (Integration & examples):**
9. `examples/robocar_integration.py` - Integration template
10. `examples/basic_recognition.py` - Simple usage example

## Robocar Integration Design

The system provides a clean, standalone module that the robocar can import:

```python
from src.gesture_recognizer import GestureRecognizer
import cv2

# Initialize recognizer
recognizer = GestureRecognizer()

# In robocar's main loop
while True:
    frame = camera.get_frame()
    result = recognizer.process_frame(frame)

    if result.gesture:
        if result.gesture.name == "PALM_FORWARD":
            robot.stop()
        elif result.gesture.name == "FIST":
            robot.go()
        elif result.gesture.name.startswith("POINT_"):
            direction = result.gesture.direction.lower()
            robot.turn(direction)
        elif result.gesture.name == "THUMBS_UP":
            robot.increase_speed()
        elif result.gesture.name == "THUMBS_DOWN":
            robot.decrease_speed()
```

**Key design principles:**
- **Modular:** Can be imported as a library
- **Stateless:** Each process_frame() call is independent
- **Simple API:** Single method call returns all needed info
- **Flexible:** Can use synchronous or callback-based patterns

## Configuration Tuning

**Default settings (balanced):**
- Camera: 1280x720 @ 30 FPS
- MediaPipe detection confidence: 0.7
- MediaPipe tracking confidence: 0.5
- Gesture confidence threshold: 0.8
- Smoothing: 5-frame history, 3 consistent frames required

**For robocar (prioritize stability):**
- Increase gesture confidence threshold to 0.9
- Increase consistent frames to 4-5
- May reduce camera resolution to 640x480 for better performance

**For development (quick iteration):**
- Reduce confidence thresholds to 0.6-0.7
- Reduce consistent frames to 2
- Enable verbose logging

## Edge Cases & Error Handling

1. **No camera access:** Try multiple indices (0, 1, 2), show clear error
2. **No hand detected:** Return None for gesture, don't spam console
3. **Multiple hands:** Prioritize right hand, allow configuration
4. **Partial hand visibility:** Reduce confidence score gracefully
5. **Similar gestures:** Use strict thresholds to differentiate
6. **Fast movements:** MediaPipe tracking handles well with proper confidence settings
7. **Poor lighting:** May need to adjust detection confidence down

## Performance Targets

- **FPS:** 20-30 on MacBook Pro M1/M2
- **Latency:** < 100ms gesture to detection
- **CPU:** 30-50% single core usage
- **Memory:** < 200MB

**Optimization strategies if needed:**
- Reduce camera resolution (640x480)
- Skip every Nth frame
- Use MediaPipe Lite model

## Verification Plan

**End-to-end testing:**
1. Run `python main.py`
2. Show palm forward → Should display "PALM_FORWARD" label
3. Point in each direction → Should detect RIGHT/LEFT/UP/DOWN
4. Make a fist → Should display "FIST"
5. Show thumbs up/down → Should detect "THUMBS_UP"/"THUMBS_DOWN"
6. Check gesture stability (no rapid switching)
7. Verify console output logs gestures
8. Test FPS counter shows 20-30 FPS
9. Run unit tests: `pytest tests/`
10. Review example code in examples/ directory

**Integration testing:**
1. Import GestureRecognizer in a separate script
2. Process frames and verify API returns expected data structure
3. Test callback-based usage pattern
4. Verify clean shutdown with recognizer.close()

## Dependencies

```txt
opencv-python==4.9.0.80    # Camera capture and visualization
mediapipe==0.10.9          # Hand tracking and landmarks
numpy==1.24.3              # Numerical operations

# Development only:
pytest==7.4.3
pytest-cov==4.1.0
black==23.12.1
```

## Next Steps After Approval

1. Create project structure (directories and empty files)
2. Install dependencies: `pip install -r requirements.txt`
3. Implement Phase 1 (Foundation) to get basic hand tracking working
4. Iterate through Phases 2-7, testing at each milestone
5. Fine-tune thresholds based on real testing
6. Create robocar integration example
7. Document the API for seamless integration

## Success Criteria

✓ All 4 gesture types recognized reliably (>90% accuracy)
✓ Stable recognition without jittering
✓ Live video display with clear annotations
✓ Console logging of detected gestures
✓ Clean API ready for robocar integration
✓ Runs smoothly on MacBook Pro (20-30 FPS)
✓ Well-documented and easy to use

# Gesture Recognition System

Real-time hand gesture recognition for robotics control using MediaPipe and OpenCV.

## Features

- Palm forward (stop gesture)
- Fist (go gesture)
- Pointing in 4 directions (up, down, left, right)
- Temporal smoothing for stability
- Landmark smoothing to reduce jitter at distance
- Configurable thresholds
- Example robocar integration

## Installation

1. Install dependencies:

```bash
uv sync
```

2. Install pre-commit hooks (recommended):

```bash
make pre-commit
```

3. Download MediaPipe model (if not included):

The MediaPipe hand landmarker model should be placed at `models/hand_landmarker.task`. If not present, download it from the [MediaPipe documentation](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker).

## Quick Start

Run the main demo application:

```bash
python main.py
```

Supported gestures:
- 🖐️ Palm Forward → Stop
- ✊ Fist → Go
- ☝️ Pointing → Direction control (up, down, left, right)

Press 'q' to quit.

## Configuration

Edit `config/settings.py` to tune:
- Detection thresholds (finger extension, palm orientation, etc.)
- Smoothing parameters (gesture history, consistency requirements)
- Camera settings (resolution, FPS, mirroring)
- Confidence thresholds for gesture detection
- Visualization settings

## Documentation

- [AI Assistant Guide](CLAUDE.md) - For AI-assisted development
- [Architecture](CLAUDE.md#architecture) - System design and data flow
- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md) - Phase-by-phase development guide

## Examples

### Robocar Integration

See `examples/robocar_integration.py` for a complete example of integrating gesture recognition with robot control:

```bash
python examples/robocar_integration.py
```

The example demonstrates two integration patterns:
1. **Imperative pattern** - Direct control flow for simple applications
2. **Callback pattern** - Decoupled design for complex state machines

## Testing

Run tests:

```bash
make test         # Run all tests
make type-check   # Type checking with pyright
make lint         # Linting with ruff
make format       # Format code
```

Or run the pre-commit hooks manually:

```bash
pre-commit run --all-files
```

## Project Structure

```
gesture_recognition/
├── config/                  # Configuration and settings
│   └── settings.py         # All tunable parameters
├── src/                    # Source code
│   ├── hand_tracker.py     # MediaPipe hand tracking wrapper
│   ├── gesture_detector.py # Gesture detection algorithms
│   ├── gesture_recognizer.py # High-level recognition with smoothing
│   ├── video_processor.py  # Camera capture and frame processing
│   └── visualizer.py       # Visualization utilities
├── examples/               # Example integrations
│   └── robocar_integration.py
├── tests/                  # Test suite
└── models/                 # MediaPipe models
```

## How It Works

1. **Capture**: Camera captures frames at 30 FPS
2. **Track**: MediaPipe extracts 21 hand landmarks per frame
3. **Smooth**: Landmark smoothing reduces jitter using exponential moving average
4. **Detect**: Geometric analysis identifies gestures from landmark positions
5. **Stabilize**: Temporal smoothing prevents false detections using majority voting
6. **Visualize**: Annotated frames show landmarks and detected gestures

## License

[Your License Here]

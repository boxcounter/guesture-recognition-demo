# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a gesture recognition project. The codebase is currently in early stages with minimal structure.

## Development Commands

The project uses Python. Standard Python commands apply:

```bash
# Run the main application
python main.py

# Run with Python 3 explicitly
python3 main.py
```

## Code Quality Requirements

**IMPORTANT: This project uses the `pre-commit` framework to enforce code quality automatically.**

### Pre-Commit Hook Setup

The project uses the [pre-commit](https://pre-commit.com) framework to run automated checks before every commit.

**One-time setup**:
```bash
make pre-commit
# Or directly: pre-commit install
```

**What the hooks check**:
1. **Ruff linter**: Checks code style and auto-fixes issues (with `--unsafe-fixes`)
2. **Ruff formatter**: Ensures consistent code formatting
3. **Pyright**: Type checking in strict mode
4. **Pytest**: Runs all tests
5. **File checks**: Trailing whitespace, YAML/JSON/TOML validation, merge conflicts, etc.

**Hook behavior**:
- ✅ If all checks pass: commit proceeds
- 🔧 If auto-fixable issues found: applies fixes automatically, re-stage and commit again
- ❌ If issues need manual fix: blocks commit with error messages
- 💡 To bypass (not recommended): `git commit --no-verify`
- 🔄 To update hooks: `pre-commit autoupdate`

**Run hooks manually** (without committing):
```bash
pre-commit run --all-files        # Run all hooks on all files
pre-commit run ruff                # Run specific hook
pre-commit run --files src/foo.py  # Run on specific files
```

### Manual Code Quality Checks

Quick commands available via Makefile:

```bash
make lint           # Ruff linter
make format         # Format code with black, isort, and ruff
make type-check     # Pyright type checker
make test           # Pytest
```

Or run tools directly:
```bash
uv run ruff check --fix --unsafe-fixes src/
uv run ruff format src/
uv run pyright src/
uv run pytest tests/ -v
```

### Type Hint Requirements

This project uses strict type checking (Python 3.10+ syntax):
- Use `X | None` instead of `Optional[X]`
- Always specify generic type parameters: `list[Landmark]` not `list`
- Add explicit `strict=` parameter to `zip()` calls
- All functions must have type hints for parameters and return values

### Code Review for AI Assistants

When AI assistants make code changes, they should perform a comprehensive review before committing:

1. **CLAUDE.md Compliance**: Check changes comply with all guidance
2. **Bug Detection**: Scan for type mismatches, logic errors, API misuse
3. **Git History Context**: Ensure changes align with historical patterns
4. **Code Comment Compliance**: Verify changes comply with docstrings

The pre-commit hooks will catch most style/type issues automatically, but AI should still review for logical correctness and architectural alignment.

## Architecture

### Core Modules

**Video Processing Pipeline:**
- `main.py` - Application entry point that orchestrates the gesture recognition demo
- `src/video_processor.py` - Camera capture and frame processing using OpenCV
- `src/visualizer.py` - Visualization utilities for drawing landmarks, connections, and gesture labels

**Hand Tracking and Gesture Detection:**
- `src/hand_tracker.py` - MediaPipe HandLandmarker wrapper for 21-point hand tracking with landmark smoothing
- `src/gesture_detector.py` - Gesture detection algorithms using geometric analysis of hand landmarks
- `src/gesture_recognizer.py` - High-level gesture recognition with temporal smoothing for stability
- `config/settings.py` - Centralized configuration for all thresholds and parameters

### Data Flow

1. **Capture**: `VideoProcessor` captures frames from camera at 30 FPS
2. **Track**: `HandTracker` processes frames with MediaPipe to extract 21 hand landmarks
3. **Smooth Landmarks**: `HandTracker` applies exponential moving average to reduce landmark jitter
4. **Detect**: `GestureDetector` analyzes landmarks to identify gestures (palm, fist, pointing)
5. **Smooth Gestures**: `GestureSmoother` applies temporal smoothing using majority voting for stability
6. **Visualize**: `Visualizer` draws landmarks and gesture labels on frames
7. **Display**: OpenCV shows annotated frames in real-time window

### Key Architectural Decisions

- **Geometric Analysis**: Uses landmark positions and distances rather than ML models for gesture detection
- **Configurable Thresholds**: All detection parameters centralized in `config/settings.py` for easy tuning
- **Modular Design**: Separation of concerns (capture, tracking, detection, visualization) for maintainability
- **Real-time Processing**: Optimized for 30 FPS performance with temporal smoothing (Phase 3 complete)
- **Dual Smoothing**: Landmark smoothing reduces jitter, temporal smoothing prevents false detections

### Temporal Smoothing

The system uses a two-layer smoothing approach to ensure stable, reliable gesture detection:

**Landmark Smoothing** (`HandTracker`):
- Applies exponential moving average (EMA) to hand landmarks frame-by-frame
- Reduces tracking jitter, especially important for long-distance detection
- Configurable smoothing factor (default: 0.5) balances responsiveness and stability
- Optional feature that can be enabled/disabled via constructor parameter

**Gesture Smoothing** (`GestureSmoother`):
- Maintains sliding window of recent gestures (GESTURE_HISTORY_SIZE frames)
- Requires MIN_CONSISTENT_FRAMES consecutive frames to confirm gesture change
- Prevents false detections from brief, unstable hand positions
- Averages confidence scores for smoother transitions
- All parameters tunable in `config/settings.py`

**Recommended Settings:**
- `GESTURE_HISTORY_SIZE = 10` - Track last 10 frames (~0.33 seconds at 30 FPS)
- `MIN_CONSISTENT_FRAMES = 5` - Require 5 consistent frames (~0.17 seconds)
- Adjust higher for more stability, lower for faster response

### Integration Points

- **MediaPipe HandLandmarker**: Google's hand tracking solution providing 21 3D landmarks per hand
- **OpenCV (cv2)**: Video capture, frame processing, and visualization
- **NumPy**: Implicit dependency via MediaPipe for landmark data structures

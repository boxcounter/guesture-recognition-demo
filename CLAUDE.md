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

## Architecture

### Core Modules

**Video Processing Pipeline:**
- `main.py` - Application entry point that orchestrates the gesture recognition demo
- `src/video_processor.py` - Camera capture and frame processing using OpenCV
- `src/visualizer.py` - Visualization utilities for drawing landmarks, connections, and gesture labels

**Hand Tracking and Gesture Detection:**
- `src/hand_tracker.py` - MediaPipe HandLandmarker wrapper for 21-point hand tracking
- `src/gesture_detector.py` - Gesture detection algorithms using geometric analysis of hand landmarks
- `config/settings.py` - Centralized configuration for all thresholds and parameters

### Data Flow

1. **Capture**: `VideoProcessor` captures frames from camera at 30 FPS
2. **Track**: `HandTracker` processes frames with MediaPipe to extract 21 hand landmarks
3. **Detect**: `GestureDetector` analyzes landmarks to identify gestures (palm, fist, pointing, thumbs)
4. **Visualize**: `Visualizer` draws landmarks and gesture labels on frames
5. **Display**: OpenCV shows annotated frames in real-time window

### Key Architectural Decisions

- **Geometric Analysis**: Uses landmark positions and distances rather than ML models for gesture detection
- **Configurable Thresholds**: All detection parameters centralized in `config/settings.py` for easy tuning
- **Modular Design**: Separation of concerns (capture, tracking, detection, visualization) for maintainability
- **Real-time Processing**: Optimized for 30 FPS performance without temporal smoothing (planned for Phase 3)

### Integration Points

- **MediaPipe HandLandmarker**: Google's hand tracking solution providing 21 3D landmarks per hand
- **OpenCV (cv2)**: Video capture, frame processing, and visualization
- **NumPy**: Implicit dependency via MediaPipe for landmark data structures

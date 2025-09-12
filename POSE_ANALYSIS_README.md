# SwingFrames Pose Analysis Module

A comprehensive pose detection and swing event analysis module for golf swing videos using MediaPipe Pose.

## 🎯 Features

- **MediaPipe Pose Integration**: Detects 33 body landmarks per frame
- **Configurable Frame Rate**: Process videos at 15-30 FPS for optimal performance
- **Swing Event Detection**: Rule-based detection of key golf swing phases
- **JSON Output**: Structured data with landmarks and detected events
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## 🏌️ Detected Swing Events

1. **Setup**: Frames with minimal movement before major swing motion
2. **Top of Backswing**: When lead wrist/hands reach highest Y coordinate
3. **Impact**: When club/wrists are closest to original setup position
4. **Follow-through**: When hands are high again and weight shifts forward

## 📦 Dependencies

```bash
pip install mediapipe opencv-python numpy
```

## 🚀 Quick Start

### Basic Usage

```python
from app.services.pose_analysis import analyze_swing

# Complete swing analysis
results = analyze_swing("golf_swing.mp4")

# Access detected events
events = results["swing_events"]
print(f"Setup: Frame {events['setup']}")
print(f"Top of Backswing: Frame {events['top_backswing']}")
print(f"Impact: Frame {events['impact']}")
print(f"Follow-through: Frame {events['follow_through']}")
```

### Advanced Usage

```python
from app.services.pose_analysis import PoseAnalyzer

# Initialize with custom parameters
analyzer = PoseAnalyzer(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    target_fps=15
)

# Extract landmarks only
landmarks = analyzer.extract_landmarks("golf_swing.mp4")

# Detect swing events
events = analyzer.detect_swing_events(landmarks)

# Save results to file
analyzer.save_results(results, "analysis_results.json")
```

## 📊 Output Format

### Landmarks Data Structure

```json
{
  "frame_index": 42,
  "original_frame_index": 126,
  "timestamp": 4.2,
  "landmarks": [
    {
      "x": 0.52,
      "y": 0.33,
      "z": -0.1,
      "visibility": 0.98
    }
  ],
  "world_landmarks": [
    {
      "x": 0.15,
      "y": -0.2,
      "z": 0.1,
      "visibility": 0.98
    }
  ]
}
```

### Swing Events Structure

```json
{
  "setup": 10,
  "top_backswing": 87,
  "impact": 142,
  "follow_through": 210
}
```

### Complete Analysis Results

```json
{
  "video_path": "golf_swing.mp4",
  "total_frames": 300,
  "swing_events": {
    "setup": 10,
    "top_backswing": 87,
    "impact": 142,
    "follow_through": 210
  },
  "landmarks_data": [...]
}
```

## 🔧 API Reference

### PoseAnalyzer Class

#### Constructor
```python
PoseAnalyzer(
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    target_fps: int = 30
)
```

#### Methods

- `extract_landmarks(video_path: str) -> List[Dict]`
  - Extract pose landmarks from video file
  - Returns list of frame data with landmarks

- `detect_swing_events(landmarks_data: List[Dict]) -> Dict`
  - Detect swing events from landmarks data
  - Returns dictionary with event frame indices

- `analyze_swing(video_path: str) -> Dict`
  - Complete analysis pipeline
  - Returns full results with landmarks and events

- `save_results(results: Dict, output_path: str)`
  - Save analysis results to JSON file

### Convenience Functions

- `extract_landmarks(video_path: str, target_fps: int = 30) -> List[Dict]`
- `detect_swing_events(landmarks: List[Dict]) -> Dict`
- `analyze_swing(video_path: str, target_fps: int = 30) -> Dict`

## 🎮 Command Line Usage

```bash
# Analyze a golf swing video
python app/services/pose_analysis.py golf_swing.mp4

# Save results to specific file
python app/services/pose_analysis.py golf_swing.mp4 results.json
```

## 🧪 Testing

The module includes comprehensive testing capabilities:

```python
# Test with mock data
from app.services.pose_analysis import PoseAnalyzer

analyzer = PoseAnalyzer()
# Module automatically tests with mock landmark data
```

## 📈 Performance Considerations

- **Frame Rate**: Lower FPS (15-20) for faster processing, higher FPS (30) for accuracy
- **Confidence Thresholds**: Higher thresholds reduce false positives but may miss valid poses
- **Video Length**: Longer videos take proportionally more time to process
- **Hardware**: GPU acceleration available through MediaPipe (if supported)

## 🔍 MediaPipe Landmark Indices

Key landmarks used for swing analysis:

- **Left Wrist**: 15
- **Right Wrist**: 16
- **Left Shoulder**: 11
- **Right Shoulder**: 12
- **Left Hip**: 23
- **Right Hip**: 24

## 🐛 Troubleshooting

### Common Issues

1. **No pose detected**: Lower confidence thresholds or check video quality
2. **Poor event detection**: Ensure golfer is fully visible in frame
3. **Performance issues**: Reduce target FPS or video resolution

### Logging

The module uses structured logging for debugging:

```python
import structlog
logger = structlog.get_logger()

# Logs include:
# - Processing progress
# - Detected events
# - Error conditions
# - Performance metrics
```

## 🔮 Future Enhancements

- Machine learning-based event detection
- Multi-person pose detection
- Real-time analysis capabilities
- Integration with swing comparison algorithms
- 3D pose reconstruction
- Biomechanical analysis

## 📄 License

This module is part of the SwingFrames golf swing analysis system.

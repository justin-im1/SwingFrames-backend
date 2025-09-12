# SwingFrames Snapshot Generator Module

A comprehensive module for extracting key event frames from golf swing videos and overlaying skeleton landmarks for visual comparison and analysis.

## 🎯 Features

- **Event Frame Extraction**: Extract frames at key swing events (Setup, Top, Impact, Follow-through)
- **Skeleton Overlay**: Draw MediaPipe pose landmarks and connections on frames
- **Semi-transparent Rendering**: Overlay skeleton while preserving original video frame visibility
- **Comparison Grid**: Generate side-by-side comparison of all swing events
- **Configurable Styling**: Customizable colors, thickness, and landmark appearance
- **Complete Pipeline**: Integration with pose analysis for end-to-end processing

## 🏌️ Supported Swing Events

1. **Setup**: Initial stance before swing motion
2. **Top of Backswing**: Peak of backswing motion
3. **Impact**: Moment of club-ball contact
4. **Follow-through**: Completion of swing motion

## 📦 Dependencies

```bash
pip install opencv-python mediapipe numpy
```

## 🚀 Quick Start

### Basic Usage

```python
from app.services.snapshot_generator import generate_swing_snapshots

# Complete pipeline: extract landmarks, detect events, generate snapshots
snapshots = generate_swing_snapshots("golf_swing.mp4")

# Access generated snapshots
print(f"Setup: {snapshots['setup']}")
print(f"Impact: {snapshots['impact']}")
```

### Advanced Usage

```python
from app.services.snapshot_generator import SnapshotGenerator
from app.services.pose_analysis import extract_landmarks, detect_swing_events

# Initialize with custom parameters
generator = SnapshotGenerator(
    output_dir="my_outputs",
    skeleton_color=(0, 255, 0),  # Green
    landmark_color=(255, 0, 0),  # Red
    skeleton_thickness=3,
    landmark_radius=4
)

# Extract landmarks and detect events
landmarks = extract_landmarks("golf_swing.mp4")
events = detect_swing_events(landmarks)

# Generate snapshots
snapshots = generator.extract_event_snapshots("golf_swing.mp4", events, landmarks)

# Create comparison grid
grid_path = generator.create_comparison_grid(snapshots)
```

## 📊 Output Format

### Generated Files

```
outputs/
├── swing_setup.jpg              # Setup phase snapshot
├── swing_top_backswing.jpg      # Top of backswing snapshot
├── swing_impact.jpg             # Impact phase snapshot
├── swing_follow_through.jpg     # Follow-through snapshot
└── swing_comparison_grid.jpg    # Side-by-side comparison
```

### Return Value

```python
{
    "setup": "outputs/swing_setup.jpg",
    "top_backswing": "outputs/swing_top_backswing.jpg", 
    "impact": "outputs/swing_impact.jpg",
    "follow_through": "outputs/swing_follow_through.jpg"
}
```

## 🔧 API Reference

### SnapshotGenerator Class

#### Constructor
```python
SnapshotGenerator(
    output_dir: str = "outputs",
    skeleton_color: Tuple[int, int, int] = (0, 255, 0),
    skeleton_thickness: int = 2,
    landmark_color: Tuple[int, int, int] = (255, 0, 0),
    landmark_radius: int = 3
)
```

#### Methods

- `draw_skeleton(frame: np.ndarray, landmarks: List[Dict]) -> np.ndarray`
  - Draw skeleton overlay on a single frame
  - Returns frame with skeleton landmarks and connections

- `extract_event_snapshots(video_path: str, events: Dict[str, int], landmarks_data: List[Dict]) -> Dict[str, str]`
  - Extract snapshots for specific swing events
  - Returns mapping of event names to saved image paths

- `generate_swing_snapshots(video_path: str) -> Dict[str, str]`
  - Complete pipeline: landmarks → events → snapshots
  - Returns mapping of event names to saved image paths

- `create_comparison_grid(snapshot_paths: Dict[str, str], output_path: str = None) -> str`
  - Create side-by-side comparison of all events
  - Returns path to saved comparison grid

### Convenience Functions

- `draw_skeleton(frame: np.ndarray, landmarks: List[Dict]) -> np.ndarray`
- `extract_event_snapshots(video_path: str, events: Dict[str, int], landmarks_data: List[Dict], output_dir: str = "outputs") -> Dict[str, str]`
- `generate_swing_snapshots(video_path: str, output_dir: str = "outputs") -> Dict[str, str]`

## 🎮 Command Line Usage

```bash
# Generate snapshots from video
python app/services/snapshot_generator.py golf_swing.mp4

# Specify output directory
python app/services/snapshot_generator.py golf_swing.mp4 my_outputs/
```

## 🎨 Customization

### Colors and Styling

```python
generator = SnapshotGenerator(
    skeleton_color=(0, 255, 0),      # Green skeleton lines
    landmark_color=(255, 0, 0),      # Red landmark points
    skeleton_thickness=3,            # Thicker lines
    landmark_radius=5                # Larger points
)
```

### Output Directory

```python
generator = SnapshotGenerator(output_dir="custom_outputs")
```

## 🧪 Testing

The module includes comprehensive testing capabilities:

```python
# Test with mock data
from app.services.snapshot_generator import SnapshotGenerator
import numpy as np

generator = SnapshotGenerator()
mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
mock_landmarks = [...]  # 33 MediaPipe landmarks

# Test skeleton drawing
frame_with_skeleton = generator.draw_skeleton(mock_frame, mock_landmarks)
```

## 📈 Performance Considerations

- **Video Processing**: Uses OpenCV for efficient frame extraction
- **Memory Usage**: Processes frames individually to minimize memory footprint
- **File I/O**: Saves images as JPEG for optimal size/quality balance
- **Skeleton Rendering**: Semi-transparent overlay preserves original frame details

## 🔍 Technical Details

### Skeleton Connections

Uses MediaPipe's standard pose connections:
- Head and neck connections
- Shoulder and arm connections  
- Torso and spine connections
- Hip and leg connections

### Semi-transparent Overlay

- **Skeleton Lines**: 60% opacity overlay
- **Landmark Points**: 80% opacity overlay
- **Original Frame**: Preserved at 40% opacity

### Frame Extraction

- Seeks to exact frame indices using `cv2.CAP_PROP_POS_FRAMES`
- Handles frame reading errors gracefully
- Supports various video formats (MP4, AVI, MOV, etc.)

## 🐛 Troubleshooting

### Common Issues

1. **No snapshots generated**: Check video file path and frame indices
2. **Poor skeleton visibility**: Adjust colors or increase thickness
3. **Missing landmarks**: Ensure pose detection confidence is adequate
4. **File permission errors**: Check output directory write permissions

### Logging

The module uses structured logging for debugging:

```python
import structlog
logger = structlog.get_logger()

# Logs include:
# - Snapshot generation progress
# - Frame extraction details
# - File save operations
# - Error conditions
```

## 🔮 Future Enhancements

- Multiple golfer detection and comparison
- 3D skeleton visualization
- Animation between key frames
- Integration with swing analysis metrics
- Real-time snapshot generation
- Custom landmark highlighting

## 📄 Integration with SwingFrames

This module integrates seamlessly with the SwingFrames ecosystem:

```python
# Complete swing analysis pipeline
from app.services.pose_analysis import analyze_swing
from app.services.snapshot_generator import generate_swing_snapshots

# Analyze swing and generate snapshots
analysis = analyze_swing("golf_swing.mp4")
snapshots = generate_swing_snapshots("golf_swing.mp4")

# Results ready for comparison and visualization
```

## 📄 License

This module is part of the SwingFrames golf swing analysis system.

# Advanced Golf Swing Pose Analysis System

A completely redesigned pose analysis system for golf swing videos using MediaPipe Pose. This implementation provides robust landmark tracking, velocity calculations, and event detection with improved accuracy and reliability.

## 🚀 Key Features

### Advanced Pose Analysis
- **High-accuracy MediaPipe integration** with model complexity 2
- **Robust landmark tracking** with confidence-based filtering
- **Coordinate normalization** relative to body dimensions (shoulder width, torso height)
- **Advanced smoothing** with configurable window sizes
- **Velocity and acceleration calculations** for all key body parts

### Enhanced Event Detection
- **Biomechanical heuristics** based on golf swing mechanics
- **Multi-criteria scoring** for each swing phase
- **Velocity-based detection** as backup method
- **Sequence validation** to ensure proper event ordering
- **Confidence scoring** for all detected events

### Comprehensive Metrics
- **Body measurements**: shoulder width, torso height, arm span
- **Angles**: shoulder-hip angle, spine angle, arm angle
- **Velocities**: hand, shoulder, and hip velocities
- **Accelerations**: hand acceleration for impact detection
- **Stability metrics**: movement and posture stability
- **Confidence scores**: overall and per-landmark confidence

## 📁 File Structure

```
app/services/
├── advanced_pose_analysis.py      # Core pose analysis with MediaPipe
├── advanced_swing_detector.py     # Event detection heuristics
├── advanced_golf_processor.py     # Main processor integration
└── test_advanced_processor.py     # Test script
```

## 🎯 Detected Swing Events

### 1. Setup
- **Criteria**: Low hand velocity, minimal shoulder-hip rotation, high stability
- **Detection**: First 30% of frames with minimal movement
- **Confidence**: Based on velocity, rotation, and stability scores

### 2. Top of Backswing
- **Criteria**: Velocity transition point, significant shoulder rotation, high hand position
- **Detection**: After setup, looking for velocity minimum and rotation peak
- **Confidence**: Based on velocity transition and rotation analysis

### 3. Impact
- **Criteria**: High hand velocity, position close to setup, shoulders returning to square
- **Detection**: After top of backswing, maximum velocity or position-based
- **Confidence**: Based on velocity, position, and rotation analysis

### 4. Follow-through
- **Criteria**: Full shoulder rotation, moderate velocity, high hand position
- **Detection**: After impact, looking for full rotation completion
- **Confidence**: Based on rotation, velocity, and position analysis

## 🔧 Usage

### Basic Usage

```python
from app.services.advanced_golf_processor import process_golf_swing_video_advanced

# Process a golf swing video
result = process_golf_swing_video_advanced(
    video_path="path/to/swing_video.mp4",
    output_dir="outputs",
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    target_fps=60
)

if result.success:
    print(f"Detected {len(result.snapshots)} swing events")
    for event_name, snapshot in result.snapshots.items():
        print(f"{event_name}: {snapshot.image_path}")
```

### Advanced Usage

```python
from app.services.advanced_golf_processor import AdvancedGolfProcessor

# Create processor with custom settings
processor = AdvancedGolfProcessor(
    output_dir="custom_outputs",
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
    target_fps=120,
    smoothing_window=7,
    enable_skeleton_overlay=True
)

# Process video
result = processor.process_swing_video("swing_video.mp4")

# Create comparison grid
grid_path = processor.create_comparison_grid(result.snapshots)

# Save detailed results
processor.save_processing_results(result)
```

### Command Line Usage

```bash
# Test with sample videos
python test_advanced_processor.py

# Process specific video
python test_advanced_processor.py path/to/video.mp4

# Direct processor usage
python app/services/advanced_golf_processor.py video.mp4 outputs/
```

## 📊 Output Files

### Generated Snapshots
- `{video_name}_setup.jpg` - Setup position snapshot
- `{video_name}_top_backswing.jpg` - Top of backswing snapshot
- `{video_name}_impact.jpg` - Impact position snapshot
- `{video_name}_follow_through.jpg` - Follow-through snapshot
- `advanced_swing_comparison_grid.jpg` - Side-by-side comparison

### Processing Results
- `advanced_processing_results.json` - Complete analysis results including:
  - Event detection details
  - Frame-by-frame metrics
  - Processing statistics
  - Confidence scores

## 🎨 Visualization Features

### Skeleton Overlay
- **Confidence-based coloring**: Green (high), Yellow (medium), Red (low)
- **Semi-transparent lines** for better visibility
- **Key metrics overlay** showing velocity, angles, and stability

### Comparison Grid
- **2x2 layout** showing all four swing events
- **Enhanced labels** with confidence scores
- **Key metrics display** for each event
- **Professional formatting** for analysis

## 📈 Performance Optimizations

### Processing Efficiency
- **Configurable FPS** for optimal processing speed
- **Frame skipping** for high FPS videos
- **Efficient landmark processing** with numpy operations
- **Memory-optimized** data structures

### Accuracy Improvements
- **Model complexity 2** for highest MediaPipe accuracy
- **Advanced smoothing** with configurable windows
- **Multi-method detection** with fallback strategies
- **Confidence-based filtering** for reliable results

## 🔍 Technical Details

### Coordinate Normalization
- **Shoulder width normalization** for consistent scaling
- **Body-relative coordinates** for different body sizes
- **Robust handling** of partial occlusions

### Velocity Calculations
- **Frame-to-frame differences** for smooth tracking
- **Acceleration analysis** for impact detection
- **Moving averages** for noise reduction

### Event Detection Algorithm
1. **Setup**: Find frames with minimal movement and rotation
2. **Top**: Detect velocity transition and rotation peak
3. **Impact**: Identify maximum velocity or position return
4. **Follow-through**: Locate full rotation completion

## 🧪 Testing

### Test Script
```bash
python test_advanced_processor.py
```

### Sample Videos
The system works with any golf swing video, but sample videos are available:
- `IMG_3845.MOV`
- `IMG_6596.MOV`

### Validation
- **Sequence validation** ensures proper event ordering
- **Confidence thresholds** filter unreliable detections
- **Fallback methods** provide robust detection

## 🚀 Improvements Over Previous Implementation

### Enhanced Accuracy
- **Higher MediaPipe model complexity** (2 vs 1)
- **Improved confidence thresholds** (0.7 vs 0.5)
- **Advanced smoothing algorithms** with configurable windows
- **Multi-criteria scoring** for each event type

### Better Event Detection
- **Biomechanical heuristics** based on golf mechanics
- **Velocity-based backup detection** for reliability
- **Position-based validation** for accuracy
- **Sequence correction** for proper ordering

### Comprehensive Metrics
- **Body dimension normalization** for consistent analysis
- **Velocity and acceleration tracking** for detailed analysis
- **Stability metrics** for movement quality assessment
- **Confidence scoring** for result reliability

### Enhanced Visualization
- **Confidence-based skeleton coloring** for quality indication
- **Metrics overlay** on snapshots for analysis
- **Professional comparison grids** with detailed information
- **Comprehensive result files** with all processing data

## 📝 Requirements

- Python 3.8+
- MediaPipe 0.10.7+
- OpenCV 4.8.1+
- NumPy 1.24.3+
- Structlog for logging

## 🎯 Use Cases

### Golf Instruction
- **Swing analysis** for teaching and improvement
- **Progress tracking** over time
- **Technique comparison** between swings

### Sports Science
- **Biomechanical analysis** of golf swings
- **Research data collection** for swing studies
- **Performance metrics** for athletes

### Personal Training
- **Self-analysis** for amateur golfers
- **Swing comparison** with professionals
- **Improvement tracking** over practice sessions

This advanced implementation provides a robust, accurate, and comprehensive solution for golf swing analysis using modern computer vision techniques.

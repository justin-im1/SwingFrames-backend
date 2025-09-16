#!/usr/bin/env python3
"""
Golf Swing Video Processor

A comprehensive function that processes golf swing videos and extracts snapshots
for four key events: setup, backswing, impact, and follow-through using MediaPipe Pose.

This module integrates all existing pose analysis components to provide a complete
solution for golf swing analysis with robust event detection heuristics.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import structlog
from pathlib import Path
import json
from dataclasses import dataclass
import math

# Import existing services
from .pose_analysis import PoseAnalyzer
from .smart_pose_analysis import SwingFeatureExtractor, SwingFeatures, SwingEvent, SwingPhase
from .smart_swing_detector import SmartSwingDetector
from .biomechanical_swing_detector import BiomechanicalSwingDetector

logger = structlog.get_logger()

@dataclass
class SwingSnapshot:
    """Represents a snapshot of a swing event"""
    event_name: str
    frame_index: int
    timestamp: float
    image_path: str
    confidence: float
    features: Dict[str, float]

@dataclass
class ProcessingResult:
    """Complete processing result for a golf swing video"""
    video_path: str
    total_frames: int
    fps: float
    duration: float
    snapshots: Dict[str, SwingSnapshot]
    events: Dict[str, SwingEvent]
    landmarks_data: List[Dict]
    processing_time: float
    success: bool
    error_message: Optional[str] = None

class GolfSwingProcessor:
    """
    Main processor for golf swing videos that extracts key event snapshots.
    
    This class integrates MediaPipe Pose detection with biomechanical analysis
    to identify and extract frames for four key golf swing events:
    - Setup: Initial address position
    - Backswing: Top of backswing position  
    - Impact: Ball impact moment
    - Follow-through: Post-impact extension
    """
    
    def __init__(self, 
                 output_dir: str = "outputs",
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 target_fps: int = 60,
                 use_biomechanical_detection: bool = True,
                 enable_smoothing: bool = True,
                 smoothing_window: int = 5):
        """
        Initialize the golf swing processor.
        
        Args:
            output_dir: Directory to save extracted snapshots
            min_detection_confidence: MediaPipe pose detection confidence threshold
            min_tracking_confidence: MediaPipe pose tracking confidence threshold
            target_fps: Target frames per second for processing
            use_biomechanical_detection: Whether to use biomechanical analysis
            enable_smoothing: Whether to apply smoothing to landmark data
            smoothing_window: Window size for moving average smoothing
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_biomechanical_detection = use_biomechanical_detection
        self.enable_smoothing = enable_smoothing
        self.smoothing_window = smoothing_window
        
        # Initialize components
        self.pose_analyzer = PoseAnalyzer(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            target_fps=target_fps
        )
        
        self.feature_extractor = SwingFeatureExtractor()
        
        if use_biomechanical_detection:
            self.swing_detector = BiomechanicalSwingDetector()
        else:
            self.swing_detector = SmartSwingDetector()
        
        logger.info(
            "GolfSwingProcessor initialized",
            output_dir=str(self.output_dir),
            use_biomechanical_detection=use_biomechanical_detection,
            enable_smoothing=enable_smoothing,
            target_fps=target_fps
        )
    
    def process_swing_video(self, video_path: str) -> ProcessingResult:
        """
        Process a golf swing video and extract snapshots for key events.
        
        This is the main function that implements the complete pipeline:
        1. Extract pose landmarks using MediaPipe
        2. Apply smoothing to reduce noise
        3. Extract swing features
        4. Detect swing events using biomechanical heuristics
        5. Extract and save snapshot frames
        
        Args:
            video_path: Path to the input golf swing video
            
        Returns:
            ProcessingResult containing all extracted data and snapshots
        """
        import time
        start_time = time.time()
        
        video_path = Path(video_path)
        if not video_path.exists():
            error_msg = f"Video file not found: {video_path}"
            logger.error(error_msg)
            return ProcessingResult(
                video_path=str(video_path),
                total_frames=0,
                fps=0.0,
                duration=0.0,
                snapshots={},
                events={},
                landmarks_data=[],
                processing_time=0.0,
                success=False,
                error_message=error_msg
            )
        
        logger.info("Starting golf swing video processing", video_path=str(video_path))
        
        try:
            # Step 1: Extract pose landmarks using MediaPipe
            logger.info("Step 1: Extracting pose landmarks")
            landmarks_data = self.pose_analyzer.extract_landmarks(str(video_path))
            
            if not landmarks_data:
                error_msg = "No pose landmarks extracted from video"
                logger.error(error_msg)
                return ProcessingResult(
                    video_path=str(video_path),
                    total_frames=0,
                    fps=0.0,
                    duration=0.0,
                    snapshots={},
                    events={},
                    landmarks_data=[],
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message=error_msg
                )
            
            # Step 2: Apply smoothing to landmark data
            if self.enable_smoothing:
                logger.info("Step 2: Applying smoothing to landmark data")
                landmarks_data = self._apply_smoothing(landmarks_data)
            
            # Step 3: Extract swing features
            logger.info("Step 3: Extracting swing features")
            swing_features = self.feature_extractor.extract_features(landmarks_data)
            
            if not swing_features:
                error_msg = "No swing features extracted"
                logger.error(error_msg)
                return ProcessingResult(
                    video_path=str(video_path),
                    total_frames=len(landmarks_data),
                    fps=0.0,
                    duration=0.0,
                    snapshots={},
                    events={},
                    landmarks_data=landmarks_data,
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message=error_msg
                )
            
            # Step 4: Detect swing events using biomechanical heuristics
            logger.info("Step 4: Detecting swing events")
            swing_events = self.swing_detector.detect_swing_events(swing_features)
            
            # Step 5: Extract and save snapshot frames
            logger.info("Step 5: Extracting snapshot frames")
            snapshots = self._extract_event_snapshots(
                str(video_path), 
                swing_events, 
                landmarks_data
            )
            
            # Get video properties
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            cap.release()
            
            processing_time = time.time() - start_time
            
            result = ProcessingResult(
                video_path=str(video_path),
                total_frames=total_frames,
                fps=fps,
                duration=duration,
                snapshots=snapshots,
                events=swing_events,
                landmarks_data=landmarks_data,
                processing_time=processing_time,
                success=True
            )
            
            logger.info(
                "Golf swing video processing completed successfully",
                video_path=str(video_path),
                total_frames=total_frames,
                snapshots_extracted=len(snapshots),
                processing_time=processing_time
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Error processing video: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return ProcessingResult(
                video_path=str(video_path),
                total_frames=0,
                fps=0.0,
                duration=0.0,
                snapshots={},
                events={},
                landmarks_data=[],
                processing_time=time.time() - start_time,
                success=False,
                error_message=error_msg
            )
    
    def _apply_smoothing(self, landmarks_data: List[Dict]) -> List[Dict]:
        """
        Apply moving average smoothing to landmark data to reduce noise.
        
        Args:
            landmarks_data: Raw landmark data from MediaPipe
            
        Returns:
            Smoothed landmark data
        """
        if len(landmarks_data) < self.smoothing_window:
            return landmarks_data
        
        smoothed_data = []
        
        for i, frame_data in enumerate(landmarks_data):
            smoothed_frame = frame_data.copy()
            
            # Apply smoothing to each landmark
            for landmark_idx in range(len(frame_data["landmarks"])):
                # Collect values for smoothing window
                x_values = []
                y_values = []
                z_values = []
                visibility_values = []
                
                # Get values from smoothing window
                start_idx = max(0, i - self.smoothing_window // 2)
                end_idx = min(len(landmarks_data), i + self.smoothing_window // 2 + 1)
                
                for j in range(start_idx, end_idx):
                    if j < len(landmarks_data) and landmark_idx < len(landmarks_data[j]["landmarks"]):
                        landmark = landmarks_data[j]["landmarks"][landmark_idx]
                        x_values.append(landmark["x"])
                        y_values.append(landmark["y"])
                        z_values.append(landmark["z"])
                        visibility_values.append(landmark["visibility"])
                
                # Apply moving average
                if x_values:
                    smoothed_frame["landmarks"][landmark_idx] = {
                        "x": np.mean(x_values),
                        "y": np.mean(y_values),
                        "z": np.mean(z_values),
                        "visibility": np.mean(visibility_values)
                    }
            
            smoothed_data.append(smoothed_frame)
        
        logger.info(
            "Applied smoothing to landmark data",
            original_frames=len(landmarks_data),
            smoothed_frames=len(smoothed_data),
            smoothing_window=self.smoothing_window
        )
        
        return smoothed_data
    
    def _extract_event_snapshots(self, 
                                video_path: str, 
                                events: Dict[str, SwingEvent], 
                                landmarks_data: List[Dict]) -> Dict[str, SwingSnapshot]:
        """
        Extract and save snapshot frames for each detected swing event.
        
        Args:
            video_path: Path to the input video
            events: Dictionary of detected swing events
            landmarks_data: Landmark data for all frames
            
        Returns:
            Dictionary mapping event names to SwingSnapshot objects
        """
        video_name = Path(video_path).stem
        snapshots = {}
        
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Could not open video file for snapshot extraction", video_path=video_path)
            return snapshots
        
        # Create mapping from frame indices to landmarks data
        landmarks_by_frame = {}
        for frame_data in landmarks_data:
            frame_idx = frame_data["frame_index"]
            landmarks_by_frame[frame_idx] = frame_data["landmarks"]
        
        try:
            for event_name, event in events.items():
                frame_index = event.frame_index
                
                # Seek to the specific frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning(
                        "Failed to read frame for event",
                        event_name=event_name,
                        frame_index=frame_index
                    )
                    continue
                
                # Get landmarks for this frame
                frame_landmarks = landmarks_by_frame.get(frame_index, [])
                
                # Draw skeleton overlay on the frame
                frame_with_skeleton = self._draw_skeleton_overlay(frame, frame_landmarks)
                
                # Save the snapshot
                snapshot_filename = f"{video_name}_{event_name}.jpg"
                snapshot_path = self.output_dir / snapshot_filename
                
                success = cv2.imwrite(str(snapshot_path), frame_with_skeleton)
                
                if success:
                    snapshot = SwingSnapshot(
                        event_name=event_name,
                        frame_index=frame_index,
                        timestamp=event.timestamp,
                        image_path=str(snapshot_path),
                        confidence=event.confidence,
                        features=event.features
                    )
                    snapshots[event_name] = snapshot
                    
                    logger.info(
                        "Snapshot saved",
                        event_name=event_name,
                        frame_index=frame_index,
                        path=str(snapshot_path),
                        confidence=event.confidence
                    )
                else:
                    logger.error(
                        "Failed to save snapshot",
                        event_name=event_name,
                        frame_index=frame_index,
                        path=str(snapshot_path)
                    )
        
        finally:
            cap.release()
        
        return snapshots
    
    def _draw_skeleton_overlay(self, frame: np.ndarray, landmarks: List[Dict]) -> np.ndarray:
        """
        Draw skeleton overlay on a frame using MediaPipe pose connections.
        
        Args:
            frame: Input frame (BGR format)
            landmarks: List of landmark dictionaries
            
        Returns:
            Frame with skeleton overlay
        """
        if not landmarks or len(landmarks) < 33:
            return frame
        
        # Create a copy of the frame
        frame_with_skeleton = frame.copy()
        height, width = frame.shape[:2]
        
        # MediaPipe pose connections
        pose_connections = [
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
            (16, 18), (16, 20), (16, 22), (12, 14), (14, 16), (11, 23), (12, 24),
            (23, 24), (23, 25), (25, 27), (27, 29), (29, 31), (24, 26), (26, 28),
            (28, 30), (30, 32)
        ]
        
        # Convert landmarks to pixel coordinates
        landmark_points = []
        for landmark in landmarks:
            if landmark["visibility"] > 0.5:
                x = int(landmark["x"] * width)
                y = int(landmark["y"] * height)
                landmark_points.append((x, y))
            else:
                landmark_points.append(None)
        
        # Draw skeleton connections
        for connection in pose_connections:
            start_idx, end_idx = connection
            
            if (start_idx < len(landmark_points) and 
                end_idx < len(landmark_points) and
                landmark_points[start_idx] is not None and 
                landmark_points[end_idx] is not None):
                
                start_point = landmark_points[start_idx]
                end_point = landmark_points[end_idx]
                
                # Draw line with semi-transparency
                overlay = frame_with_skeleton.copy()
                cv2.line(overlay, start_point, end_point, (0, 255, 0), 2)
                cv2.addWeighted(overlay, 0.6, frame_with_skeleton, 0.4, 0, frame_with_skeleton)
        
        # Draw landmark points
        for point in landmark_points:
            if point is not None:
                overlay = frame_with_skeleton.copy()
                cv2.circle(overlay, point, 3, (255, 0, 0), -1)
                cv2.addWeighted(overlay, 0.8, frame_with_skeleton, 0.2, 0, frame_with_skeleton)
        
        return frame_with_skeleton
    
    def create_comparison_grid(self, snapshots: Dict[str, SwingSnapshot]) -> str:
        """
        Create a comparison grid showing all swing events side by side.
        
        Args:
            snapshots: Dictionary of swing snapshots
            
        Returns:
            Path to the saved comparison grid
        """
        if not snapshots:
            raise ValueError("No snapshots provided")
        
        # Load all images
        images = {}
        max_width = 0
        max_height = 0
        
        for event_name, snapshot in snapshots.items():
            if Path(snapshot.image_path).exists():
                img = cv2.imread(snapshot.image_path)
                if img is not None:
                    images[event_name] = img
                    max_width = max(max_width, img.shape[1])
                    max_height = max(max_height, img.shape[0])
        
        if not images:
            raise ValueError("No valid images found")
        
        # Resize all images to the same size
        resized_images = {}
        for event_name, img in images.items():
            resized = cv2.resize(img, (max_width, max_height))
            resized_images[event_name] = resized
        
        # Create grid layout (2x2 for 4 events)
        event_order = ["setup", "top_backswing", "impact", "follow_through"]
        grid_rows = 2
        grid_cols = 2
        
        # Create grid canvas
        grid_height = max_height * grid_rows
        grid_width = max_width * grid_cols
        grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        # Place images in grid
        for i, event_name in enumerate(event_order):
            if event_name in resized_images:
                row = i // grid_cols
                col = i % grid_cols
                
                y_start = row * max_height
                y_end = y_start + max_height
                x_start = col * max_width
                x_end = x_start + max_width
                
                grid[y_start:y_end, x_start:x_end] = resized_images[event_name]
                
                # Add event label with confidence
                label = event_name.replace("_", " ").title()
                confidence = snapshots[event_name].confidence
                label_with_confidence = f"{label} ({confidence:.2f})"
                
                cv2.putText(grid, label_with_confidence, (x_start + 10, y_start + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Save grid
        grid_path = self.output_dir / "swing_comparison_grid.jpg"
        success = cv2.imwrite(str(grid_path), grid)
        
        if success:
            logger.info("Comparison grid saved", path=str(grid_path))
            return str(grid_path)
        else:
            raise RuntimeError(f"Failed to save comparison grid: {grid_path}")
    
    def save_processing_results(self, result: ProcessingResult, output_path: str = None):
        """
        Save processing results to a JSON file.
        
        Args:
            result: ProcessingResult to save
            output_path: Path to save the results (optional)
        """
        if output_path is None:
            output_path = self.output_dir / "processing_results.json"
        
        # Convert result to dictionary for JSON serialization
        result_dict = {
            "video_path": result.video_path,
            "total_frames": result.total_frames,
            "fps": result.fps,
            "duration": result.duration,
            "processing_time": result.processing_time,
            "success": result.success,
            "error_message": result.error_message,
            "snapshots": {
                event_name: {
                    "event_name": snapshot.event_name,
                    "frame_index": snapshot.frame_index,
                    "timestamp": snapshot.timestamp,
                    "image_path": snapshot.image_path,
                    "confidence": snapshot.confidence,
                    "features": snapshot.features
                }
                for event_name, snapshot in result.snapshots.items()
            },
            "events": {
                event_name: {
                    "phase": event.phase.value,
                    "frame_index": event.frame_index,
                    "timestamp": event.timestamp,
                    "confidence": event.confidence,
                    "detection_method": event.detection_method,
                    "features": event.features
                }
                for event_name, event in result.events.items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info("Processing results saved", path=str(output_path))


# Convenience function for direct use
def process_golf_swing_video(video_path: str, 
                           output_dir: str = "outputs",
                           use_biomechanical_detection: bool = True,
                           enable_smoothing: bool = True) -> ProcessingResult:
    """
    Process a golf swing video and extract snapshots for key events.
    
    This is the main convenience function that implements the complete pipeline:
    1. Extract pose landmarks using MediaPipe
    2. Apply smoothing to reduce noise  
    3. Extract swing features
    4. Detect swing events using biomechanical heuristics
    5. Extract and save snapshot frames
    
    Args:
        video_path: Path to the input golf swing video
        output_dir: Directory to save extracted snapshots
        use_biomechanical_detection: Whether to use biomechanical analysis
        enable_smoothing: Whether to apply smoothing to landmark data
        
    Returns:
        ProcessingResult containing all extracted data and snapshots
    """
    processor = GolfSwingProcessor(
        output_dir=output_dir,
        use_biomechanical_detection=use_biomechanical_detection,
        enable_smoothing=enable_smoothing
    )
    
    return processor.process_swing_video(video_path)


if __name__ == "__main__":
    # Test the golf swing processor
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python golf_swing_processor.py <video_path> [output_dir]")
        print("Example: python golf_swing_processor.py test_swing.mp4 outputs/")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs"
    
    try:
        print(f"🎬 Processing golf swing video: {video_path}")
        print(f"📁 Output directory: {output_dir}")
        
        # Process the video
        result = process_golf_swing_video(video_path, output_dir)
        
        if result.success:
            print(f"\n✅ Processing completed successfully!")
            print(f"📊 Video Info:")
            print(f"  Total frames: {result.total_frames}")
            print(f"  FPS: {result.fps:.2f}")
            print(f"  Duration: {result.duration:.2f} seconds")
            print(f"  Processing time: {result.processing_time:.2f} seconds")
            
            print(f"\n🎯 Extracted Snapshots:")
            for event_name, snapshot in result.snapshots.items():
                print(f"  {event_name}: Frame {snapshot.frame_index} (confidence: {snapshot.confidence:.2f})")
                print(f"    Saved to: {snapshot.image_path}")
            
            # Create comparison grid
            if result.snapshots:
                try:
                    processor = GolfSwingProcessor(output_dir=output_dir)
                    grid_path = processor.create_comparison_grid(result.snapshots)
                    print(f"\n📊 Comparison grid: {grid_path}")
                except Exception as e:
                    print(f"\n⚠️  Could not create comparison grid: {e}")
            
            # Save results
            processor = GolfSwingProcessor(output_dir=output_dir)
            processor.save_processing_results(result)
            print(f"\n💾 Results saved to: {output_dir}/processing_results.json")
            
        else:
            print(f"\n❌ Processing failed: {result.error_message}")
            sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        sys.exit(1)

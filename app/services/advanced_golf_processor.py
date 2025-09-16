#!/usr/bin/env python3
"""
Advanced Golf Swing Video Processor

A completely redesigned golf swing processing system that integrates advanced
pose analysis, robust event detection, and snapshot extraction with improved
accuracy and reliability.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import structlog
from pathlib import Path
import json
from dataclasses import dataclass
import math
import time

# Import the new advanced modules
from .advanced_pose_analysis import AdvancedPoseAnalyzer, FrameData, SwingEvent, SwingPhase
from .advanced_swing_detector import AdvancedSwingDetector

logger = structlog.get_logger()

@dataclass
class SwingSnapshot:
    """Represents a snapshot of a swing event with enhanced metadata"""
    event_name: str
    frame_index: int
    timestamp: float
    image_path: str
    confidence: float
    features: Dict[str, float]
    normalized_position: Tuple[float, float]
    body_metrics: Dict[str, float]

@dataclass
class ProcessingResult:
    """Complete processing result with enhanced information"""
    video_path: str
    total_frames: int
    fps: float
    duration: float
    snapshots: Dict[str, SwingSnapshot]
    events: Dict[str, SwingEvent]
    frames_data: List[FrameData]
    processing_time: float
    success: bool
    error_message: Optional[str] = None
    processing_stats: Optional[Dict[str, Any]] = None

class AdvancedGolfProcessor:
    """
    Advanced golf swing video processor with comprehensive analysis.
    
    This processor implements a complete pipeline for golf swing analysis:
    1. Advanced pose landmark extraction with MediaPipe
    2. Robust landmark tracking with velocity/acceleration calculations
    3. Coordinate normalization relative to body dimensions
    4. Advanced event detection using biomechanical heuristics
    5. High-quality snapshot extraction with skeleton overlays
    """
    
    def __init__(self, 
                 output_dir: str = "outputs",
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.7,
                 target_fps: int = 60,
                 smoothing_window: int = 5,
                 enable_skeleton_overlay: bool = True,
                 skeleton_color: Tuple[int, int, int] = (0, 255, 0),
                 skeleton_thickness: int = 2):
        """
        Initialize the advanced golf processor.
        
        Args:
            output_dir: Directory to save extracted snapshots
            min_detection_confidence: MediaPipe detection confidence threshold
            min_tracking_confidence: MediaPipe tracking confidence threshold
            target_fps: Target frames per second for processing
            smoothing_window: Window size for smoothing
            enable_skeleton_overlay: Whether to draw skeleton overlays
            skeleton_color: BGR color for skeleton lines
            skeleton_thickness: Thickness of skeleton lines
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_skeleton_overlay = enable_skeleton_overlay
        self.skeleton_color = skeleton_color
        self.skeleton_thickness = skeleton_thickness
        
        # Initialize components
        self.pose_analyzer = AdvancedPoseAnalyzer(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            target_fps=target_fps,
            smoothing_window=smoothing_window
        )
        
        self.swing_detector = AdvancedSwingDetector()
        
        # MediaPipe pose connections for skeleton drawing
        import mediapipe as mp
        self.mp_pose = mp.solutions.pose
        self.pose_connections = self.mp_pose.POSE_CONNECTIONS
        
        logger.info(
            "AdvancedGolfProcessor initialized",
            output_dir=str(self.output_dir),
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            target_fps=target_fps,
            smoothing_window=smoothing_window
        )

    def process_swing_video(self, video_path: str) -> ProcessingResult:
        """
        Process a golf swing video with advanced analysis.
        
        This is the main function that implements the complete pipeline:
        1. Extract pose landmarks using advanced MediaPipe analysis
        2. Apply smoothing and normalization
        3. Detect swing events using robust heuristics
        4. Extract and save high-quality snapshot frames
        
        Args:
            video_path: Path to the input golf swing video
            
        Returns:
            ProcessingResult containing all extracted data and snapshots
        """
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
                frames_data=[],
                processing_time=0.0,
                success=False,
                error_message=error_msg
            )
        
        logger.info("Starting advanced golf swing video processing", video_path=str(video_path))
        
        try:
            # Step 1: Extract pose landmarks with advanced analysis
            logger.info("Step 1: Extracting pose landmarks with advanced analysis")
            frames_data = self.pose_analyzer.extract_landmarks(str(video_path))
            
            if not frames_data:
                error_msg = "No pose landmarks extracted from video"
                logger.error(error_msg)
                return ProcessingResult(
                    video_path=str(video_path),
                    total_frames=0,
                    fps=0.0,
                    duration=0.0,
                    snapshots={},
                    events={},
                    frames_data=[],
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message=error_msg
                )
            
            # Step 2: Detect swing events using advanced heuristics
            logger.info("Step 2: Detecting swing events with advanced heuristics")
            swing_events = self.swing_detector.detect_swing_events(frames_data)
            
            # Step 3: Extract and save snapshot frames
            logger.info("Step 3: Extracting snapshot frames")
            snapshots = self._extract_event_snapshots(
                str(video_path), 
                swing_events, 
                frames_data
            )
            
            # Get video properties
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            cap.release()
            
            processing_time = time.time() - start_time
            
            # Calculate processing statistics
            processing_stats = self._calculate_processing_stats(frames_data, swing_events)
            
            result = ProcessingResult(
                video_path=str(video_path),
                total_frames=total_frames,
                fps=fps,
                duration=duration,
                snapshots=snapshots,
                events=swing_events,
                frames_data=frames_data,
                processing_time=processing_time,
                success=True,
                processing_stats=processing_stats
            )
            
            logger.info(
                "Advanced golf swing video processing completed successfully",
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
                frames_data=[],
                processing_time=time.time() - start_time,
                success=False,
                error_message=error_msg
            )

    def _extract_event_snapshots(self, 
                                video_path: str, 
                                events: Dict[str, SwingEvent], 
                                frames_data: List[FrameData]) -> Dict[str, SwingSnapshot]:
        """
        Extract and save snapshot frames for each detected swing event.
        
        Args:
            video_path: Path to the input video
            events: Dictionary of detected swing events
            frames_data: Processed frame data
            
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
        
        # Create mapping from frame indices to frame data
        frames_by_index = {}
        for frame_data in frames_data:
            frames_by_index[frame_data.frame_index] = frame_data
        
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
                
                # Get frame data for this frame
                frame_data = frames_by_index.get(frame_index)
                
                # Draw skeleton overlay if enabled
                if self.enable_skeleton_overlay and frame_data:
                    frame_with_skeleton = self._draw_advanced_skeleton_overlay(frame, frame_data)
                else:
                    frame_with_skeleton = frame
                
                # Save the snapshot
                snapshot_filename = f"{video_name}_{event_name}.jpg"
                snapshot_path = self.output_dir / snapshot_filename
                
                success = cv2.imwrite(str(snapshot_path), frame_with_skeleton)
                
                if success:
                    # Extract body metrics for this frame
                    body_metrics = {}
                    if frame_data:
                        body_metrics = {
                            "shoulder_width": frame_data.shoulder_width,
                            "torso_height": frame_data.torso_height,
                            "arm_span": frame_data.arm_span,
                            "hand_velocity": frame_data.hand_velocity,
                            "shoulder_hip_angle": frame_data.shoulder_hip_angle,
                            "spine_angle": frame_data.spine_angle,
                            "movement_stability": frame_data.movement_stability,
                            "posture_stability": frame_data.posture_stability,
                            "overall_confidence": frame_data.overall_confidence
                        }
                    
                    snapshot = SwingSnapshot(
                        event_name=event_name,
                        frame_index=frame_index,
                        timestamp=event.timestamp,
                        image_path=str(snapshot_path),
                        confidence=event.confidence,
                        features=event.features,
                        normalized_position=event.normalized_position,
                        body_metrics=body_metrics
                    )
                    snapshots[event_name] = snapshot
                    
                    logger.info(
                        "Advanced snapshot saved",
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

    def _draw_advanced_skeleton_overlay(self, frame: np.ndarray, frame_data: FrameData) -> np.ndarray:
        """
        Draw advanced skeleton overlay with enhanced visualization.
        
        Args:
            frame: Input frame (BGR format)
            frame_data: Processed frame data with landmarks
            
        Returns:
            Frame with advanced skeleton overlay
        """
        if not frame_data.landmarks or len(frame_data.landmarks) < 33:
            return frame
        
        # Create a copy of the frame
        frame_with_skeleton = frame.copy()
        height, width = frame.shape[:2]
        
        # Convert landmarks to pixel coordinates
        landmark_points = []
        for landmark in frame_data.landmarks:
            if landmark.visibility > 0.5:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                landmark_points.append((x, y))
            else:
                landmark_points.append(None)
        
        # Draw skeleton connections with enhanced visualization
        for connection in self.pose_connections:
            start_idx, end_idx = connection
            
            if (start_idx < len(landmark_points) and 
                end_idx < len(landmark_points) and
                landmark_points[start_idx] is not None and 
                landmark_points[end_idx] is not None):
                
                start_point = landmark_points[start_idx]
                end_point = landmark_points[end_idx]
                
                # Draw line with semi-transparency
                overlay = frame_with_skeleton.copy()
                cv2.line(overlay, start_point, end_point, 
                        self.skeleton_color, self.skeleton_thickness)
                cv2.addWeighted(overlay, 0.7, frame_with_skeleton, 0.3, 0, frame_with_skeleton)
        
        # Draw landmark points with confidence-based coloring
        for i, point in enumerate(landmark_points):
            if point is not None:
                # Color based on confidence
                confidence = frame_data.landmarks[i].visibility
                if confidence > 0.8:
                    color = (0, 255, 0)  # Green for high confidence
                elif confidence > 0.6:
                    color = (0, 255, 255)  # Yellow for medium confidence
                else:
                    color = (0, 0, 255)  # Red for low confidence
                
                overlay = frame_with_skeleton.copy()
                cv2.circle(overlay, point, 3, color, -1)
                cv2.addWeighted(overlay, 0.8, frame_with_skeleton, 0.2, 0, frame_with_skeleton)
        
        # Add key metrics as text overlay
        self._add_metrics_overlay(frame_with_skeleton, frame_data)
        
        return frame_with_skeleton

    def _add_metrics_overlay(self, frame: np.ndarray, frame_data: FrameData):
        """Add key metrics as text overlay on the frame"""
        # Prepare metrics text
        metrics_text = [
            f"Hand Velocity: {frame_data.hand_velocity:.3f}",
            f"Shoulder-Hip Angle: {frame_data.shoulder_hip_angle:.1f}°",
            f"Spine Angle: {frame_data.spine_angle:.1f}°",
            f"Stability: {frame_data.movement_stability:.2f}",
            f"Confidence: {frame_data.overall_confidence:.2f}"
        ]
        
        # Draw text overlay
        y_offset = 30
        for text in metrics_text:
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25

    def _calculate_processing_stats(self, frames_data: List[FrameData], events: Dict[str, SwingEvent]) -> Dict[str, Any]:
        """Calculate processing statistics"""
        if not frames_data:
            return {}
        
        # Calculate average confidence
        avg_confidence = np.mean([frame.overall_confidence for frame in frames_data])
        
        # Calculate velocity statistics
        hand_velocities = [frame.hand_velocity for frame in frames_data]
        max_velocity = max(hand_velocities) if hand_velocities else 0
        avg_velocity = np.mean(hand_velocities) if hand_velocities else 0
        
        # Calculate angle statistics
        shoulder_hip_angles = [frame.shoulder_hip_angle for frame in frames_data]
        max_rotation = max([abs(angle) for angle in shoulder_hip_angles]) if shoulder_hip_angles else 0
        
        # Calculate stability statistics
        stabilities = [frame.movement_stability for frame in frames_data]
        avg_stability = np.mean(stabilities) if stabilities else 0
        
        # Event confidence statistics
        event_confidences = [event.confidence for event in events.values()]
        avg_event_confidence = np.mean(event_confidences) if event_confidences else 0
        
        return {
            "average_landmark_confidence": avg_confidence,
            "maximum_hand_velocity": max_velocity,
            "average_hand_velocity": avg_velocity,
            "maximum_shoulder_rotation": max_rotation,
            "average_movement_stability": avg_stability,
            "average_event_confidence": avg_event_confidence,
            "total_processed_frames": len(frames_data),
            "events_detected": len(events)
        }

    def create_comparison_grid(self, snapshots: Dict[str, SwingSnapshot]) -> str:
        """
        Create an enhanced comparison grid showing all swing events.
        
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
                
                # Add enhanced event label with confidence and metrics
                snapshot = snapshots[event_name]
                label = event_name.replace("_", " ").title()
                confidence = snapshot.confidence
                
                # Add main label
                cv2.putText(grid, label, (x_start + 10, y_start + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Add confidence
                cv2.putText(grid, f"Conf: {confidence:.2f}", (x_start + 10, y_start + 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Add key metrics
                if snapshot.body_metrics:
                    velocity = snapshot.body_metrics.get("hand_velocity", 0)
                    angle = snapshot.body_metrics.get("shoulder_hip_angle", 0)
                    cv2.putText(grid, f"Vel: {velocity:.3f}", (x_start + 10, y_start + 85),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(grid, f"Angle: {angle:.1f}°", (x_start + 10, y_start + 105),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save grid
        grid_path = self.output_dir / "advanced_swing_comparison_grid.jpg"
        success = cv2.imwrite(str(grid_path), grid)
        
        if success:
            logger.info("Advanced comparison grid saved", path=str(grid_path))
            return str(grid_path)
        else:
            raise RuntimeError(f"Failed to save comparison grid: {grid_path}")

    def save_processing_results(self, result: ProcessingResult, output_path: str = None):
        """
        Save comprehensive processing results to JSON file.
        
        Args:
            result: ProcessingResult to save
            output_path: Path to save the results (optional)
        """
        if output_path is None:
            output_path = self.output_dir / "advanced_processing_results.json"
        
        # Convert result to dictionary for JSON serialization
        result_dict = {
            "video_path": result.video_path,
            "total_frames": result.total_frames,
            "fps": result.fps,
            "duration": result.duration,
            "processing_time": result.processing_time,
            "success": result.success,
            "error_message": result.error_message,
            "processing_stats": result.processing_stats,
            "snapshots": {
                event_name: {
                    "event_name": snapshot.event_name,
                    "frame_index": snapshot.frame_index,
                    "timestamp": snapshot.timestamp,
                    "image_path": snapshot.image_path,
                    "confidence": snapshot.confidence,
                    "features": snapshot.features,
                    "normalized_position": snapshot.normalized_position,
                    "body_metrics": snapshot.body_metrics
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
                    "features": event.features,
                    "normalized_position": event.normalized_position
                }
                for event_name, event in result.events.items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info("Advanced processing results saved", path=str(output_path))


# Convenience function for direct use
def process_golf_swing_video_advanced(video_path: str, 
                                    output_dir: str = "outputs",
                                    min_detection_confidence: float = 0.7,
                                    min_tracking_confidence: float = 0.7,
                                    target_fps: int = 60) -> ProcessingResult:
    """
    Process a golf swing video with advanced analysis.
    
    This is the main convenience function that implements the complete advanced pipeline:
    1. Extract pose landmarks using advanced MediaPipe analysis
    2. Apply smoothing and normalization
    3. Detect swing events using robust heuristics
    4. Extract and save high-quality snapshot frames
    
    Args:
        video_path: Path to the input golf swing video
        output_dir: Directory to save extracted snapshots
        min_detection_confidence: MediaPipe detection confidence threshold
        min_tracking_confidence: MediaPipe tracking confidence threshold
        target_fps: Target frames per second for processing
        
    Returns:
        ProcessingResult containing all extracted data and snapshots
    """
    processor = AdvancedGolfProcessor(
        output_dir=output_dir,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        target_fps=target_fps
    )
    
    return processor.process_swing_video(video_path)


if __name__ == "__main__":
    # Test the advanced golf processor
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python advanced_golf_processor.py <video_path> [output_dir]")
        print("Example: python advanced_golf_processor.py test_swing.mp4 outputs/")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs"
    
    try:
        print(f"🎬 Processing golf swing video with advanced analysis: {video_path}")
        print(f"📁 Output directory: {output_dir}")
        
        # Process the video
        result = process_golf_swing_video_advanced(video_path, output_dir)
        
        if result.success:
            print(f"\n✅ Advanced processing completed successfully!")
            print(f"📊 Video Info:")
            print(f"  Total frames: {result.total_frames}")
            print(f"  FPS: {result.fps:.2f}")
            print(f"  Duration: {result.duration:.2f} seconds")
            print(f"  Processing time: {result.processing_time:.2f} seconds")
            
            if result.processing_stats:
                print(f"\n📈 Processing Statistics:")
                for key, value in result.processing_stats.items():
                    print(f"  {key}: {value}")
            
            print(f"\n🎯 Extracted Snapshots:")
            for event_name, snapshot in result.snapshots.items():
                print(f"  {event_name}: Frame {snapshot.frame_index} (confidence: {snapshot.confidence:.2f})")
                print(f"    Saved to: {snapshot.image_path}")
                if snapshot.body_metrics:
                    print(f"    Hand velocity: {snapshot.body_metrics.get('hand_velocity', 0):.3f}")
                    print(f"    Shoulder-hip angle: {snapshot.body_metrics.get('shoulder_hip_angle', 0):.1f}°")
            
            # Create comparison grid
            if result.snapshots:
                try:
                    processor = AdvancedGolfProcessor(output_dir=output_dir)
                    grid_path = processor.create_comparison_grid(result.snapshots)
                    print(f"\n📊 Advanced comparison grid: {grid_path}")
                except Exception as e:
                    print(f"\n⚠️  Could not create comparison grid: {e}")
            
            # Save results
            processor = AdvancedGolfProcessor(output_dir=output_dir)
            processor.save_processing_results(result)
            print(f"\n💾 Advanced results saved to: {output_dir}/advanced_processing_results.json")
            
        else:
            print(f"\n❌ Processing failed: {result.error_message}")
            sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        sys.exit(1)

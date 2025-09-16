#!/usr/bin/env python3
"""
SwingFrames Snapshot Generator Module

This module extracts key event frames from golf swing videos and overlays
skeleton landmarks for visual comparison and analysis.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import structlog
from pathlib import Path
import mediapipe as mp
from .pose_analysis import extract_landmarks, detect_swing_events, analyze_swing

logger = structlog.get_logger()

class SnapshotGenerator:
    """Generates skeleton overlay snapshots from golf swing videos."""
    
    def __init__(self, 
                 output_dir: str = "outputs",
                 skeleton_color: Tuple[int, int, int] = (0, 255, 0),
                 skeleton_thickness: int = 2,
                 landmark_color: Tuple[int, int, int] = (255, 0, 0),
                 landmark_radius: int = 3,
                 use_smart_detection: bool = True):
        """
        Initialize the snapshot generator.
        
        Args:
            output_dir: Directory to save snapshot images
            skeleton_color: BGR color for skeleton lines (default: green)
            skeleton_thickness: Thickness of skeleton lines
            landmark_color: BGR color for landmark points (default: red)
            landmark_radius: Radius of landmark circles
            use_smart_detection: Whether to use smart detection methods
        """
        self.use_smart_detection = use_smart_detection
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.skeleton_color = skeleton_color
        self.skeleton_thickness = skeleton_thickness
        self.landmark_color = landmark_color
        self.landmark_radius = landmark_radius
        
        # MediaPipe pose connections for skeleton drawing
        self.mp_pose = mp.solutions.pose
        self.pose_connections = self.mp_pose.POSE_CONNECTIONS
        
        logger.info(
            "SnapshotGenerator initialized",
            output_dir=str(self.output_dir),
            skeleton_color=skeleton_color,
            landmark_color=landmark_color,
            use_smart_detection=use_smart_detection
        )

    def draw_skeleton(self, frame: np.ndarray, landmarks: List[Dict]) -> np.ndarray:
        """
        Draw skeleton landmarks and connections on a frame.
        
        Args:
            frame: Input frame (BGR format)
            landmarks: List of landmark dictionaries with x, y, z, visibility
            
        Returns:
            Frame with skeleton overlay
        """
        if not landmarks or len(landmarks) < 33:
            logger.warning("Insufficient landmarks for skeleton drawing", 
                         landmark_count=len(landmarks) if landmarks else 0)
            return frame
        
        # Create a copy of the frame to avoid modifying the original
        frame_with_skeleton = frame.copy()
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Convert landmarks to pixel coordinates
        landmark_points = []
        for landmark in landmarks:
            if landmark["visibility"] > 0.5:  # Only draw visible landmarks
                x = int(landmark["x"] * width)
                y = int(landmark["y"] * height)
                landmark_points.append((x, y))
            else:
                landmark_points.append(None)
        
        # Draw skeleton connections
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
                cv2.addWeighted(overlay, 0.6, frame_with_skeleton, 0.4, 0, frame_with_skeleton)
        
        # Draw landmark points
        for i, point in enumerate(landmark_points):
            if point is not None:
                # Draw landmark with semi-transparency
                overlay = frame_with_skeleton.copy()
                cv2.circle(overlay, point, self.landmark_radius, 
                          self.landmark_color, -1)
                cv2.addWeighted(overlay, 0.8, frame_with_skeleton, 0.2, 0, frame_with_skeleton)
        
        return frame_with_skeleton

    def extract_event_snapshots(self, 
                              video_path: str, 
                              events: Dict[str, int], 
                              landmarks_data: List[Dict]) -> Dict[str, str]:
        """
        Extract and save skeleton overlay snapshots for each swing event.
        
        Args:
            video_path: Path to the input video file
            events: Dictionary mapping event names to frame indices
            landmarks_data: List of frame data with landmarks
            
        Returns:
            Dictionary mapping event names to saved image paths
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        logger.info(
            "Starting snapshot extraction",
            video_path=str(video_path),
            swing_events=events
        )
        
        # Open video capture
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(
            "Video properties",
            total_frames=total_frames,
            fps=fps
        )
        
        # Create a mapping from frame indices to landmarks data
        landmarks_by_frame = {}
        for frame_data in landmarks_data:
            frame_idx = frame_data["frame_index"]
            landmarks_by_frame[frame_idx] = frame_data["landmarks"]
        
        saved_snapshots = {}
        
        try:
            for event_name, frame_index in events.items():
                if frame_index >= total_frames:
                    logger.warning(
                        "Event frame index exceeds video length",
                        event_name=event_name,
                        frame_index=frame_index,
                        total_frames=total_frames
                    )
                    continue
                
                # Seek to the specific frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                
                if not ret:
                    logger.error(
                        "Failed to read frame",
                        event_name=event_name,
                        frame_index=frame_index
                    )
                    continue
                
                # Get landmarks for this frame
                frame_landmarks = landmarks_by_frame.get(frame_index, [])
                
                if not frame_landmarks:
                    logger.warning(
                        "No landmarks found for frame",
                        event_name=event_name,
                        frame_index=frame_index
                    )
                    # Still save the frame without skeleton
                    frame_with_skeleton = frame
                else:
                    # Draw skeleton overlay
                    frame_with_skeleton = self.draw_skeleton(frame, frame_landmarks)
                
                # Save the snapshot
                snapshot_filename = f"swing_{event_name}.jpg"
                snapshot_path = self.output_dir / snapshot_filename
                
                # Convert BGR to RGB for saving
                frame_rgb = cv2.cvtColor(frame_with_skeleton, cv2.COLOR_BGR2RGB)
                
                # Save as JPEG
                success = cv2.imwrite(str(snapshot_path), frame_with_skeleton)
                
                if success:
                    saved_snapshots[event_name] = str(snapshot_path)
                    logger.info(
                        "Snapshot saved",
                        event_name=event_name,
                        frame_index=frame_index,
                        path=str(snapshot_path)
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
        
        logger.info(
            "Snapshot extraction completed",
            saved_count=len(saved_snapshots),
            total_events=len(events)
        )
        
        return saved_snapshots

    def generate_swing_snapshots(self, video_path: str) -> Dict[str, str]:
        """
        Complete pipeline: extract landmarks, detect events, generate snapshots.
        
        Args:
            video_path: Path to the input video file
            
        Returns:
            Dictionary mapping event names to saved image paths
        """
        logger.info("Starting complete swing snapshot generation", video_path=video_path, use_smart_detection=self.use_smart_detection)
        
        if self.use_smart_detection:
            # Use new advanced pose analysis for better event detection
            try:
                from .advanced_golf_processor import process_golf_swing_video_advanced
                
                # Process video with advanced analysis
                result = process_golf_swing_video_advanced(
                    video_path=video_path,
                    output_dir=str(self.output_dir),
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7,
                    target_fps=60
                )
                
                if result.success:
                    # Convert to the expected format
                    swing_events = {
                        "setup": result.events["setup"].frame_index,
                        "top_backswing": result.events["top_backswing"].frame_index,
                        "impact": result.events["impact"].frame_index,
                        "follow_through": result.events["follow_through"].frame_index
                    }
                    
                    # Convert frames_data to landmarks_data format for compatibility
                    landmarks_data = []
                    for frame_data in result.frames_data:
                        landmarks_data.append({
                            "frame_index": frame_data.frame_index,
                            "timestamp": frame_data.timestamp,
                            "landmarks": [
                                {
                                    "x": landmark.x,
                                    "y": landmark.y,
                                    "z": landmark.z,
                                    "visibility": landmark.visibility
                                }
                                for landmark in frame_data.landmarks
                            ]
                        })
                    
                    logger.info("Advanced pose analysis completed", events=swing_events)
                    
                else:
                    raise Exception(f"Advanced processing failed: {result.error_message}")
                
            except Exception as e:
                logger.warning("Advanced pose analysis failed, falling back to original method", error=str(e))
                # Fall back to original method
                landmarks_data = extract_landmarks(video_path)
                swing_events = detect_swing_events(landmarks_data)
        else:
            # Use original method
            landmarks_data = extract_landmarks(video_path)
            swing_events = detect_swing_events(landmarks_data)
        
        # Generate snapshots
        snapshots = self.extract_event_snapshots(video_path, swing_events, landmarks_data)
        
        logger.info(
            "Complete swing snapshot generation finished",
            snapshots_generated=len(snapshots),
            use_smart_detection=self.use_smart_detection
        )
        
        return snapshots

    def create_comparison_grid(self, 
                             snapshot_paths: Dict[str, str], 
                             output_path: str = None) -> str:
        """
        Create a comparison grid showing all swing events side by side.
        
        Args:
            snapshot_paths: Dictionary mapping event names to image paths
            output_path: Path to save the comparison grid
            
        Returns:
            Path to the saved comparison grid
        """
        if not snapshot_paths:
            raise ValueError("No snapshot paths provided")
        
        # Load all images
        images = {}
        max_width = 0
        max_height = 0
        
        for event_name, path in snapshot_paths.items():
            if Path(path).exists():
                img = cv2.imread(path)
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
                
                # Add event label
                label = event_name.replace("_", " ").title()
                cv2.putText(grid, label, (x_start + 10, y_start + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save grid
        if output_path is None:
            output_path = self.output_dir / "swing_comparison_grid.jpg"
        
        success = cv2.imwrite(str(output_path), grid)
        
        if success:
            logger.info("Comparison grid saved", path=str(output_path))
            return str(output_path)
        else:
            raise RuntimeError(f"Failed to save comparison grid: {output_path}")


# Convenience functions for direct use
def draw_skeleton(frame: np.ndarray, landmarks: List[Dict]) -> np.ndarray:
    """
    Draw skeleton landmarks and connections on a frame.
    
    Args:
        frame: Input frame (BGR format)
        landmarks: List of landmark dictionaries with x, y, z, visibility
        
    Returns:
        Frame with skeleton overlay
    """
    generator = SnapshotGenerator()
    return generator.draw_skeleton(frame, landmarks)


def extract_event_snapshots(video_path: str, 
                          events: Dict[str, int], 
                          landmarks_data: List[Dict],
                          output_dir: str = "outputs") -> Dict[str, str]:
    """
    Extract and save skeleton overlay snapshots for each swing event.
    
    Args:
        video_path: Path to the input video file
        events: Dictionary mapping event names to frame indices
        landmarks_data: List of frame data with landmarks
        output_dir: Directory to save snapshots
        
    Returns:
        Dictionary mapping event names to saved image paths
    """
    generator = SnapshotGenerator(output_dir=output_dir)
    return generator.extract_event_snapshots(video_path, events, landmarks_data)


def generate_swing_snapshots(video_path: str, output_dir: str = "outputs", use_smart_detection: bool = False) -> Dict[str, str]:
    """
    Complete pipeline: extract landmarks, detect events, generate snapshots.
    
    Args:
        video_path: Path to the input video file
        output_dir: Directory to save snapshots
        use_smart_detection: Whether to use smart detection methods
        
    Returns:
        Dictionary mapping event names to saved image paths
    """
    generator = SnapshotGenerator(output_dir=output_dir, use_smart_detection=use_smart_detection)
    return generator.generate_swing_snapshots(video_path)


if __name__ == "__main__":
    # Test the snapshot generator module
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python snapshot_generator.py <video_path> [output_dir]")
        print("Example: python snapshot_generator.py test_swing.mp4 outputs/")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs"
    
    try:
        print(f"🎬 Generating swing snapshots from: {video_path}")
        print(f"📁 Output directory: {output_dir}")
        
        # Create snapshot generator
        generator = SnapshotGenerator(output_dir=output_dir)
        
        # Run complete pipeline
        snapshots = generator.generate_swing_snapshots(video_path)
        
        # Print results
        print("\n🎯 Generated Snapshots:")
        for event_name, path in snapshots.items():
            print(f"  {event_name}: {path}")
        
        # Create comparison grid
        if snapshots:
            try:
                grid_path = generator.create_comparison_grid(snapshots)
                print(f"\n📊 Comparison grid: {grid_path}")
            except Exception as e:
                print(f"\n⚠️  Could not create comparison grid: {e}")
        
        print(f"\n✅ Successfully generated {len(snapshots)} swing snapshots!")
        
    except Exception as e:
        print(f"❌ Error during snapshot generation: {e}")
        sys.exit(1)

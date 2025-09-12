#!/usr/bin/env python3
"""
SwingFrames Pose Analysis Module

This module provides pose detection and swing event analysis using MediaPipe Pose.
It extracts body landmarks from golf swing videos and identifies key swing events.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Optional, Tuple
import structlog
import json
from pathlib import Path

logger = structlog.get_logger()

class PoseAnalyzer:
    """MediaPipe-based pose analyzer for golf swing videos."""
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 target_fps: int = 60):  # Increased for better impact detection
        """
        Initialize the pose analyzer.
        
        Args:
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
            target_fps: Target frames per second for processing
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.target_fps = target_fps
        
        logger.info(
            "PoseAnalyzer initialized",
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            target_fps=target_fps
        )

    def extract_landmarks(self, video_path: str) -> List[Dict]:
        """
        Extract pose landmarks from a video file.
        
        Args:
            video_path: Path to the input video file
            
        Returns:
            List of frames with landmarks data
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        logger.info("Starting landmark extraction", video_path=str(video_path))
        
        # Open video capture
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps if original_fps > 0 else 0
        
        logger.info(
            "Video properties",
            original_fps=original_fps,
            total_frames=total_frames,
            duration=duration
        )
        
        # Calculate frame skip for target FPS
        # For impact detection, we want higher temporal resolution
        frame_skip = max(1, int(original_fps / self.target_fps)) if original_fps > 0 else 1
        
        # If original FPS is low (< 60), process all frames for better impact detection
        if original_fps < 60:
            frame_skip = 1
            logger.info(
                "Low framerate detected - processing all frames for better impact detection",
                original_fps=original_fps,
                frame_skip=frame_skip
            )
        
        frames_data = []
        frame_index = 0
        processed_frames = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames to achieve target FPS
                if frame_index % frame_skip != 0:
                    frame_index += 1
                    continue
                
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame with MediaPipe
                results = self.pose.process(rgb_frame)
                
                frame_data = {
                    "frame_index": processed_frames,
                    "original_frame_index": frame_index,
                    "timestamp": frame_index / original_fps if original_fps > 0 else 0,
                    "landmarks": []
                }
                
                if results.pose_landmarks:
                    # Extract landmarks
                    for landmark in results.pose_landmarks.landmark:
                        frame_data["landmarks"].append({
                            "x": float(landmark.x),
                            "y": float(landmark.y),
                            "z": float(landmark.z),
                            "visibility": float(landmark.visibility)
                        })
                    
                    # Add pose world landmarks if available
                    if results.pose_world_landmarks:
                        frame_data["world_landmarks"] = []
                        for landmark in results.pose_world_landmarks.landmark:
                            frame_data["world_landmarks"].append({
                                "x": float(landmark.x),
                                "y": float(landmark.y),
                                "z": float(landmark.z),
                                "visibility": float(landmark.visibility)
                            })
                else:
                    logger.warning("No pose detected in frame", frame_index=frame_index)
                
                frames_data.append(frame_data)
                processed_frames += 1
                frame_index += 1
                
                # Log progress every 100 frames
                if processed_frames % 100 == 0:
                    logger.info(
                        "Processing progress",
                        processed_frames=processed_frames,
                        total_frames=total_frames,
                        progress_percent=(processed_frames / (total_frames // frame_skip)) * 100
                    )
        
        finally:
            cap.release()
        
        logger.info(
            "Landmark extraction completed",
            total_processed_frames=len(frames_data),
            original_frames=total_frames,
            frame_skip=frame_skip
        )
        
        return frames_data

    def detect_swing_events(self, landmarks_data: List[Dict]) -> Dict:
        """
        Detect key swing events from pose landmarks.
        
        Args:
            landmarks_data: List of frame data with landmarks
            
        Returns:
            Dictionary with detected event frame indices
        """
        if not landmarks_data:
            raise ValueError("No landmarks data provided")
        
        logger.info("Starting swing event detection", total_frames=len(landmarks_data))
        
        # MediaPipe pose landmark indices
        # Left and right wrist indices
        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_HIP = 23
        RIGHT_HIP = 24
        
        # Extract relevant landmark positions
        wrist_positions = []
        shoulder_positions = []
        hip_positions = []
        head_positions = []
        
        for frame_data in landmarks_data:
            if len(frame_data["landmarks"]) >= 33:  # Ensure we have all landmarks
                frame_idx = frame_data["frame_index"]
                
                # Get wrist positions (average of left and right)
                left_wrist = frame_data["landmarks"][LEFT_WRIST]
                right_wrist = frame_data["landmarks"][RIGHT_WRIST]
                
                if left_wrist["visibility"] > 0.5 and right_wrist["visibility"] > 0.5:
                    avg_wrist_y = (left_wrist["y"] + right_wrist["y"]) / 2
                    avg_wrist_x = (left_wrist["x"] + right_wrist["x"]) / 2
                    wrist_positions.append({
                        "frame": frame_idx,
                        "x": avg_wrist_x,
                        "y": avg_wrist_y,
                        "left_wrist": left_wrist,
                        "right_wrist": right_wrist
                    })
                
                # Get shoulder positions
                left_shoulder = frame_data["landmarks"][LEFT_SHOULDER]
                right_shoulder = frame_data["landmarks"][RIGHT_SHOULDER]
                
                if left_shoulder["visibility"] > 0.5 and right_shoulder["visibility"] > 0.5:
                    avg_shoulder_y = (left_shoulder["y"] + right_shoulder["y"]) / 2
                    shoulder_positions.append({
                        "frame": frame_idx,
                        "y": avg_shoulder_y,
                        "left_shoulder": left_shoulder,
                        "right_shoulder": right_shoulder
                    })
                
                # Get hip positions
                left_hip = frame_data["landmarks"][LEFT_HIP]
                right_hip = frame_data["landmarks"][RIGHT_HIP]
                
                if left_hip["visibility"] > 0.5 and right_hip["visibility"] > 0.5:
                    avg_hip_y = (left_hip["y"] + right_hip["y"]) / 2
                    hip_positions.append({
                        "frame": frame_idx,
                        "y": avg_hip_y,
                        "left_hip": left_hip,
                        "right_hip": right_hip
                    })
                
                # Get head position (nose landmark)
                NOSE = 0
                nose = frame_data["landmarks"][NOSE]
                if nose["visibility"] > 0.5:
                    head_positions.append({
                        "frame": frame_idx,
                        "x": nose["x"],
                        "y": nose["y"],
                        "nose": nose
                    })
        
        if not wrist_positions:
            logger.warning("No valid wrist positions found for swing detection")
            return {"setup": 0, "top_backswing": 0, "impact": 0, "follow_through": 0}
        
        # Detect swing events
        events = self._detect_swing_phases(wrist_positions, shoulder_positions, hip_positions, head_positions, landmarks_data)
        
        logger.info(
            "Swing events detected",
            setup=events["setup"],
            top_backswing=events["top_backswing"],
            impact=events["impact"],
            follow_through=events["follow_through"]
        )
        
        return events

    def _detect_swing_phases(self, 
                           wrist_positions: List[Dict], 
                           shoulder_positions: List[Dict],
                           hip_positions: List[Dict],
                           head_positions: List[Dict],
                           landmarks_data: List[Dict]) -> Dict:
        """Detect individual swing phases using rule-based approach."""
        
        if len(wrist_positions) < 10:
            return {"setup": 0, "top_backswing": 0, "impact": 0, "follow_through": 0}
        
        # 1. Setup: Use lenient detection to find true setup position
        setup_frame = self._find_setup_with_lenient_detection(landmarks_data)
        
        # 2. Top of Backswing: Find highest Y position of wrists (use strict detection)
        top_backswing_frame = self._find_top_backswing(wrist_positions)
        
        # 3. Impact: Find when wrists return closest to setup position
        impact_frame = self._find_impact(wrist_positions, setup_frame)
        
        # 4. Follow-through: Find when hands are high again after impact
        follow_through_frame = self._find_follow_through(wrist_positions, impact_frame)
        
        return {
            "setup": setup_frame,
            "top_backswing": top_backswing_frame,
            "impact": impact_frame,
            "follow_through": follow_through_frame
        }

    def _find_setup_phase(self, wrist_positions: List[Dict], shoulder_positions: List[Dict]) -> int:
        """Find setup phase - true setup position when golfer addresses the ball with club on ground."""
        if not wrist_positions:
            return 0
        
        # For setup detection, we need to look at frames before the main swing
        # Use a more lenient approach to find the true setup position
        setup_frame = self._find_setup_with_lenient_detection([])
        
        logger.info(
            "Setup phase detected",
            setup_frame=setup_frame,
            method="lenient_detection"
        )
        
        return setup_frame
    
    def _find_setup_with_lenient_detection(self, landmarks_data: List[Dict]) -> int:
        """Find setup using lenient detection to capture early frames."""
        if not landmarks_data:
            return 0
        
        # Use lenient visibility thresholds to find the true setup position
        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        
        # Look for frames where at least one wrist is visible with lenient threshold
        lenient_wrist_positions = []
        
        for frame_data in landmarks_data:
            if len(frame_data["landmarks"]) >= 33:
                frame_idx = frame_data["frame_index"]
                left_wrist = frame_data["landmarks"][LEFT_WRIST]
                right_wrist = frame_data["landmarks"][RIGHT_WRIST]
                
                # Use lenient threshold: either wrist > 0.3
                if left_wrist["visibility"] > 0.3 or right_wrist["visibility"] > 0.3:
                    # Use the wrist with higher visibility
                    if right_wrist["visibility"] > left_wrist["visibility"]:
                        wrist_y = right_wrist["y"]
                    else:
                        wrist_y = left_wrist["y"]
                    
                    lenient_wrist_positions.append({
                        "frame": frame_idx,
                        "y": wrist_y
                    })
        
        if not lenient_wrist_positions:
            return 0
        
        # Ultra-conservative setup detection:
        # 1. Look for the earliest frame with minimal movement (true setup)
        # 2. Use the smallest possible window (first 5% of frames) to avoid backswing
        # 3. Find the frame with highest Y position in that window
        
        early_frames = lenient_wrist_positions[:max(1, len(lenient_wrist_positions)//20)]  # First 5%
        if early_frames:
            # Find the frame with highest Y position (most upright stance)
            setup_frame = max(early_frames, key=lambda p: p["y"])
            
            # Additional validation: ensure this frame is truly early
            if setup_frame["frame"] < len(landmarks_data) // 10:  # Within first 10% of video
                return setup_frame["frame"]
        
        # Fallback: use the very first frame
        return lenient_wrist_positions[0]["frame"]
    
    def _find_earliest_stable_frame(self, wrist_positions: List[Dict]) -> int:
        """Find the earliest frame with minimal movement (true setup)."""
        if len(wrist_positions) < 10:
            return 0
        
        # Look at first 30% of frames for true setup
        setup_end = max(5, len(wrist_positions) // 3)
        setup_frames = wrist_positions[:setup_end]
        
        # Find the earliest frame where movement is minimal
        for i in range(len(setup_frames) - 4):
            # Check stability in a 5-frame window
            window = setup_frames[i:i+5]
            
            # Calculate movement variance
            y_values = [frame["y"] for frame in window]
            x_values = [frame["x"] for frame in window]
            
            y_variance = np.var(y_values)
            x_variance = np.var(x_values)
            
            # If movement is very low, this is likely setup
            if y_variance < 0.001 and x_variance < 0.001:  # Very low movement threshold
                return window[0]["frame"]
        
        return 0  # No stable frame found
    
    def _find_early_setup_frame(self, wrist_positions: List[Dict]) -> int:
        """Fallback: Find setup in the first few frames."""
        if not wrist_positions:
            return 0
        
        # Use first 10% of frames, or first 5 frames, whichever is smaller
        early_frames = min(5, max(1, len(wrist_positions) // 10))
        setup_frames = wrist_positions[:early_frames]
        
        # Find frame with lowest movement variance
        min_variance = float('inf')
        setup_frame = setup_frames[0]["frame"]
        
        for i, frame in enumerate(setup_frames):
            # Calculate variance in a small window around this frame
            window_size = min(3, len(setup_frames) - i)
            if window_size < 2:
                continue
                
            y_values = [setup_frames[j]["y"] for j in range(i, i + window_size)]
            variance = np.var(y_values)
            
            if variance < min_variance:
                min_variance = variance
                setup_frame = frame["frame"]
        
        return setup_frame

    def _find_top_backswing(self, wrist_positions: List[Dict]) -> int:
        """Find top of backswing - highest Y position of wrists after setup."""
        if not wrist_positions:
            return 0
        
        # Find setup frame first to ensure we're looking after setup
        setup_frame = self._find_setup_phase(wrist_positions, [])
        
        # Look for frames after setup for the highest position
        backswing_candidates = [pos for pos in wrist_positions if pos["frame"] > setup_frame]
        
        if not backswing_candidates:
            # Fallback to original method if no candidates
            min_y = float('inf')
            top_frame = 0
            
            for pos in wrist_positions:
                if pos["y"] < min_y:  # Lower Y = higher in image
                    min_y = pos["y"]
                    top_frame = pos["frame"]
            
            return top_frame
        
        # Find the frame with highest Y coordinate (lowest Y value in image coordinates)
        min_y = float('inf')
        top_frame = backswing_candidates[0]["frame"]
        
        for pos in backswing_candidates:
            if pos["y"] < min_y:  # Lower Y = higher in image
                min_y = pos["y"]
                top_frame = pos["frame"]
        
        logger.info(
            "Top of backswing detected",
            top_frame=top_frame,
            setup_frame=setup_frame,
            y_position=min_y
        )
        
        return top_frame

    def _find_impact(self, wrist_positions: List[Dict], setup_frame: int) -> int:
        """Find impact using multiple detection methods for better accuracy."""
        if not wrist_positions:
            return 0
        
        # Find setup position
        setup_pos = None
        for pos in wrist_positions:
            if pos["frame"] == setup_frame:
                setup_pos = pos
                break
        
        if not setup_pos:
            setup_pos = wrist_positions[0]
        
        top_backswing_frame = self._find_top_backswing(wrist_positions)
        
        # Use position-based detection (closest to setup position after top of backswing)
        impact_frame = self._find_impact_by_position(wrist_positions, setup_pos, top_backswing_frame)
        
        return impact_frame
    
    def _find_impact_by_velocity(self, wrist_positions: List[Dict], top_backswing_frame: int) -> int:
        """Find impact by detecting maximum downward velocity of wrists."""
        if len(wrist_positions) < 3:
            return top_backswing_frame
        
        # Calculate velocities for frames after top of backswing
        velocities = []
        for i in range(1, len(wrist_positions)):
            if wrist_positions[i]["frame"] <= top_backswing_frame:
                continue
                
            prev_pos = wrist_positions[i-1]
            curr_pos = wrist_positions[i]
            
            # Calculate velocity (change in Y position per frame)
            # Positive velocity = downward movement (impact zone)
            velocity = curr_pos["y"] - prev_pos["y"]
            
            velocities.append({
                "frame": curr_pos["frame"],
                "velocity": velocity,
                "index": i
            })
        
        if not velocities:
            return top_backswing_frame
        
        # Find frame with maximum downward velocity (impact)
        max_velocity = max(velocities, key=lambda v: v["velocity"])
        
        logger.info(
            "Impact detected by velocity",
            impact_frame=max_velocity["frame"],
            max_velocity=max_velocity["velocity"]
        )
        
        return max_velocity["frame"]
    
    def _find_impact_by_position(self, wrist_positions: List[Dict], setup_pos: Dict, top_backswing_frame: int) -> int:
        """Fallback method: Find impact by position proximity to setup."""
        impact_candidates = [pos for pos in wrist_positions if pos["frame"] > top_backswing_frame]
        
        if not impact_candidates:
            return top_backswing_frame
        
        # Find closest to setup position
        min_distance = float('inf')
        impact_frame = top_backswing_frame
        
        for pos in impact_candidates:
            distance = np.sqrt(
                (pos["x"] - setup_pos["x"])**2 + 
                (pos["y"] - setup_pos["y"])**2
            )
            
            if distance < min_distance:
                min_distance = distance
                impact_frame = pos["frame"]
        
        return impact_frame

    def _find_follow_through(self, wrist_positions: List[Dict], impact_frame: int) -> int:
        """Find follow-through - when hands are high again after impact."""
        if not wrist_positions:
            return 0
        
        # Look for frames after impact
        follow_through_candidates = [pos for pos in wrist_positions if pos["frame"] > impact_frame]
        
        if not follow_through_candidates:
            return impact_frame
        
        # Find frame where wrists are high again (low Y value)
        # Look for local minimum in Y coordinate after impact
        min_y = float('inf')
        follow_through_frame = impact_frame
        
        for pos in follow_through_candidates:
            if pos["y"] < min_y:  # Lower Y = higher in image
                min_y = pos["y"]
                follow_through_frame = pos["frame"]
        
        return follow_through_frame

    def analyze_swing(self, video_path: str) -> Dict:
        """
        Complete swing analysis pipeline.
        
        Args:
            video_path: Path to the input video file
            
        Returns:
            Complete analysis results with landmarks and events
        """
        logger.info("Starting complete swing analysis", video_path=video_path)
        
        # Extract landmarks
        landmarks_data = self.extract_landmarks(video_path)
        
        # Detect swing events
        swing_events = self.detect_swing_events(landmarks_data)
        
        # Compile results
        results = {
            "video_path": str(video_path),
            "total_frames": len(landmarks_data),
            "swing_events": swing_events,
            "landmarks_data": landmarks_data
        }
        
        logger.info(
            "Swing analysis completed",
            total_frames=len(landmarks_data),
            events=swing_events
        )
        
        return results

    def save_results(self, results: Dict, output_path: str):
        """Save analysis results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("Results saved", output_path=str(output_path))


# Convenience functions for direct use
def extract_landmarks(video_path: str, target_fps: int = 30) -> List[Dict]:
    """
    Extract pose landmarks from a video file.
    
    Args:
        video_path: Path to the input video file
        target_fps: Target frames per second for processing
        
    Returns:
        List of frames with landmarks data
    """
    analyzer = PoseAnalyzer(target_fps=target_fps)
    return analyzer.extract_landmarks(video_path)


def detect_swing_events(landmarks: List[Dict]) -> Dict:
    """
    Detect key swing events from pose landmarks.
    
    Args:
        landmarks: List of frame data with landmarks
        
    Returns:
        Dictionary with detected event frame indices
    """
    analyzer = PoseAnalyzer()
    return analyzer.detect_swing_events(landmarks)


def analyze_swing(video_path: str, target_fps: int = 30) -> Dict:
    """
    Complete swing analysis pipeline.
    
    Args:
        video_path: Path to the input video file
        target_fps: Target frames per second for processing
        
    Returns:
        Complete analysis results with landmarks and events
    """
    analyzer = PoseAnalyzer(target_fps=target_fps)
    return analyzer.analyze_swing(video_path)


if __name__ == "__main__":
    # Test the pose analysis module
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pose_analysis.py <video_path> [output_path]")
        print("Example: python pose_analysis.py test_swing.mp4 results.json")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "swing_analysis_results.json"
    
    try:
        print(f"Analyzing swing video: {video_path}")
        
        # Run complete analysis
        results = analyze_swing(video_path)
        
        # Print detected events
        events = results["swing_events"]
        print("\n🎯 Detected Swing Events:")
        print(f"  Setup: Frame {events['setup']}")
        print(f"  Top of Backswing: Frame {events['top_backswing']}")
        print(f"  Impact: Frame {events['impact']}")
        print(f"  Follow-through: Frame {events['follow_through']}")
        
        print(f"\n📊 Analysis Summary:")
        print(f"  Total frames processed: {results['total_frames']}")
        print(f"  Video path: {results['video_path']}")
        
        # Save results
        analyzer = PoseAnalyzer()
        analyzer.save_results(results, output_path)
        print(f"\n💾 Results saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        sys.exit(1)

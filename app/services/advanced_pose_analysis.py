#!/usr/bin/env python3
"""
Advanced Golf Swing Pose Analysis

A completely redesigned pose analysis system for golf swing videos using MediaPipe Pose.
This module provides robust landmark tracking, velocity calculations, and event detection
with improved accuracy and noise reduction.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import structlog
import json
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import math
from collections import deque

logger = structlog.get_logger()

class SwingPhase(Enum):
    """Golf swing phases"""
    SETUP = "setup"
    BACKSWING = "backswing"
    TOP = "top_backswing"
    DOWNSWING = "downswing"
    IMPACT = "impact"
    FOLLOW_THROUGH = "follow_through"
    FINISH = "finish"

@dataclass
class LandmarkData:
    """Normalized landmark data with confidence"""
    x: float
    y: float
    z: float
    visibility: float
    normalized_x: float
    normalized_y: float

@dataclass
class FrameData:
    """Complete frame data with landmarks and derived metrics"""
    frame_index: int
    timestamp: float
    landmarks: List[LandmarkData]
    
    # Body measurements (normalized)
    shoulder_width: float
    torso_height: float
    arm_span: float
    
    # Key positions (normalized coordinates)
    head_center: Tuple[float, float]
    shoulder_center: Tuple[float, float]
    hip_center: Tuple[float, float]
    hand_center: Tuple[float, float]
    foot_center: Tuple[float, float]
    
    # Velocities (pixels/frame)
    hand_velocity: float
    shoulder_velocity: float
    hip_velocity: float
    
    # Accelerations (pixels/frame²)
    hand_acceleration: float
    
    # Angles (degrees)
    shoulder_hip_angle: float
    spine_angle: float
    arm_angle: float
    
    # Stability metrics
    movement_stability: float
    posture_stability: float
    
    # Confidence scores
    overall_confidence: float

@dataclass
class SwingEvent:
    """Detected swing event with detailed information"""
    phase: SwingPhase
    frame_index: int
    timestamp: float
    confidence: float
    detection_method: str
    features: Dict[str, Any]
    normalized_position: Tuple[float, float]

class AdvancedPoseAnalyzer:
    """Advanced pose analyzer with improved tracking and analysis"""
    
    def __init__(self, 
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.7,
                 target_fps: int = 60,
                 smoothing_window: int = 5):
        """
        Initialize the advanced pose analyzer.
        
        Args:
            min_detection_confidence: MediaPipe detection confidence threshold
            min_tracking_confidence: MediaPipe tracking confidence threshold
            target_fps: Target processing FPS
            smoothing_window: Window size for smoothing
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Use complexity 1 to avoid model download issues
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.target_fps = target_fps
        self.smoothing_window = smoothing_window
        
        # Landmark indices for key body parts
        self.LANDMARK_INDICES = {
            'nose': 0,
            'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
            'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
            'left_ear': 7, 'right_ear': 8,
            'mouth_left': 9, 'mouth_right': 10,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_pinky': 17, 'right_pinky': 18,
            'left_index': 19, 'right_index': 20,
            'left_thumb': 21, 'right_thumb': 22,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28,
            'left_heel': 29, 'right_heel': 30,
            'left_foot_index': 31, 'right_foot_index': 32
        }
        
        # Smoothing buffers
        self.landmark_buffer = deque(maxlen=smoothing_window)
        self.velocity_buffer = deque(maxlen=smoothing_window)
        
        logger.info(
            "AdvancedPoseAnalyzer initialized",
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            target_fps=target_fps,
            smoothing_window=smoothing_window
        )

    def extract_landmarks(self, video_path: str) -> List[FrameData]:
        """
        Extract and process landmarks from video with advanced analysis.
        
        Args:
            video_path: Path to input video
            
        Returns:
            List of FrameData with processed landmarks and metrics
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        logger.info("Starting advanced landmark extraction", video_path=str(video_path))
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame skip for target FPS
        frame_skip = max(1, int(original_fps / self.target_fps)) if original_fps > 0 else 1
        
        logger.info(
            "Video properties",
            original_fps=original_fps,
            total_frames=total_frames,
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
                
                # Process frame
                frame_data = self._process_frame(frame, processed_frames, frame_index, original_fps)
                if frame_data:
                    frames_data.append(frame_data)
                
                processed_frames += 1
                frame_index += 1
                
                # Log progress
                if processed_frames % 100 == 0:
                    logger.info(
                        "Processing progress",
                        processed_frames=processed_frames,
                        total_frames=total_frames
                    )
        
        finally:
            cap.release()
        
        logger.info(
            "Advanced landmark extraction completed",
            total_processed_frames=len(frames_data),
            original_frames=total_frames
        )
        
        return frames_data

    def _process_frame(self, frame: np.ndarray, frame_index: int, original_frame_index: int, fps: float) -> Optional[FrameData]:
        """Process a single frame and extract all metrics"""
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None
        
        # Extract landmarks
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmark_data = LandmarkData(
                x=float(landmark.x),
                y=float(landmark.y),
                z=float(landmark.z),
                visibility=float(landmark.visibility),
                normalized_x=0.0,  # Will be calculated later
                normalized_y=0.0   # Will be calculated later
            )
            landmarks.append(landmark_data)
        
        # Calculate body measurements for normalization
        body_metrics = self._calculate_body_metrics(landmarks)
        
        # Normalize landmarks
        normalized_landmarks = self._normalize_landmarks(landmarks, body_metrics)
        
        # Calculate derived metrics
        frame_data = FrameData(
            frame_index=frame_index,
            timestamp=original_frame_index / fps if fps > 0 else 0,
            landmarks=normalized_landmarks,
            
            # Body measurements
            shoulder_width=body_metrics['shoulder_width'],
            torso_height=body_metrics['torso_height'],
            arm_span=body_metrics['arm_span'],
            
            # Key positions
            head_center=body_metrics['head_center'],
            shoulder_center=body_metrics['shoulder_center'],
            hip_center=body_metrics['hip_center'],
            hand_center=body_metrics['hand_center'],
            foot_center=body_metrics['foot_center'],
            
            # Velocities and accelerations (calculated from previous frames)
            hand_velocity=0.0,
            shoulder_velocity=0.0,
            hip_velocity=0.0,
            hand_acceleration=0.0,
            
            # Angles
            shoulder_hip_angle=body_metrics['shoulder_hip_angle'],
            spine_angle=body_metrics['spine_angle'],
            arm_angle=body_metrics['arm_angle'],
            
            # Stability metrics
            movement_stability=0.0,
            posture_stability=body_metrics['posture_stability'],
            
            # Confidence
            overall_confidence=body_metrics['overall_confidence']
        )
        
        # Calculate velocities and accelerations from previous frames
        self._calculate_velocities_and_accelerations(frame_data)
        
        # Add to smoothing buffer
        self.landmark_buffer.append(frame_data)
        
        return frame_data

    def _calculate_body_metrics(self, landmarks: List[LandmarkData]) -> Dict[str, Any]:
        """Calculate body measurements for normalization"""
        metrics = {}
        
        # Get key landmarks
        left_shoulder = landmarks[self.LANDMARK_INDICES['left_shoulder']]
        right_shoulder = landmarks[self.LANDMARK_INDICES['right_shoulder']]
        left_hip = landmarks[self.LANDMARK_INDICES['left_hip']]
        right_hip = landmarks[self.LANDMARK_INDICES['right_hip']]
        left_wrist = landmarks[self.LANDMARK_INDICES['left_wrist']]
        right_wrist = landmarks[self.LANDMARK_INDICES['right_wrist']]
        left_ankle = landmarks[self.LANDMARK_INDICES['left_ankle']]
        right_ankle = landmarks[self.LANDMARK_INDICES['right_ankle']]
        nose = landmarks[self.LANDMARK_INDICES['nose']]
        
        # Calculate shoulder width
        metrics['shoulder_width'] = math.sqrt(
            (right_shoulder.x - left_shoulder.x)**2 + 
            (right_shoulder.y - left_shoulder.y)**2
        )
        
        # Calculate torso height
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_center_y = (left_hip.y + right_hip.y) / 2
        metrics['torso_height'] = abs(shoulder_center_y - hip_center_y)
        
        # Calculate arm span
        metrics['arm_span'] = math.sqrt(
            (right_wrist.x - left_wrist.x)**2 + 
            (right_wrist.y - left_wrist.y)**2
        )
        
        # Calculate center points
        metrics['head_center'] = (nose.x, nose.y)
        metrics['shoulder_center'] = (
            (left_shoulder.x + right_shoulder.x) / 2,
            (left_shoulder.y + right_shoulder.y) / 2
        )
        metrics['hip_center'] = (
            (left_hip.x + right_hip.x) / 2,
            (left_hip.y + right_hip.y) / 2
        )
        metrics['hand_center'] = (
            (left_wrist.x + right_wrist.x) / 2,
            (left_wrist.y + right_wrist.y) / 2
        )
        metrics['foot_center'] = (
            (left_ankle.x + right_ankle.x) / 2,
            (left_ankle.y + right_ankle.y) / 2
        )
        
        # Calculate angles
        metrics['shoulder_hip_angle'] = self._calculate_shoulder_hip_angle(landmarks)
        metrics['spine_angle'] = self._calculate_spine_angle(landmarks)
        metrics['arm_angle'] = self._calculate_arm_angle(landmarks)
        
        # Calculate stability
        metrics['posture_stability'] = self._calculate_posture_stability(landmarks)
        
        # Calculate overall confidence
        metrics['overall_confidence'] = self._calculate_overall_confidence(landmarks)
        
        return metrics

    def _normalize_landmarks(self, landmarks: List[LandmarkData], body_metrics: Dict[str, Any]) -> List[LandmarkData]:
        """Normalize landmarks relative to body dimensions"""
        normalized_landmarks = []
        shoulder_width = body_metrics['shoulder_width']
        torso_height = body_metrics['torso_height']
        
        # Use shoulder width as primary normalization factor
        normalization_factor = max(shoulder_width, 0.01)  # Avoid division by zero
        
        for landmark in landmarks:
            # Normalize relative to shoulder width
            normalized_x = landmark.x / normalization_factor
            normalized_y = landmark.y / normalization_factor
            
            normalized_landmark = LandmarkData(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z,
                visibility=landmark.visibility,
                normalized_x=normalized_x,
                normalized_y=normalized_y
            )
            normalized_landmarks.append(normalized_landmark)
        
        return normalized_landmarks

    def _calculate_shoulder_hip_angle(self, landmarks: List[LandmarkData]) -> float:
        """Calculate angle between shoulder and hip lines"""
        left_shoulder = landmarks[self.LANDMARK_INDICES['left_shoulder']]
        right_shoulder = landmarks[self.LANDMARK_INDICES['right_shoulder']]
        left_hip = landmarks[self.LANDMARK_INDICES['left_hip']]
        right_hip = landmarks[self.LANDMARK_INDICES['right_hip']]
        
        if (left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5 or
            left_hip.visibility < 0.5 or right_hip.visibility < 0.5):
            return 0.0
        
        # Calculate shoulder line angle
        shoulder_angle = math.atan2(
            right_shoulder.y - left_shoulder.y,
            right_shoulder.x - left_shoulder.x
        )
        
        # Calculate hip line angle
        hip_angle = math.atan2(
            right_hip.y - left_hip.y,
            right_hip.x - left_hip.x
        )
        
        # Calculate difference
        angle_diff = math.degrees(shoulder_angle - hip_angle)
        
        # Normalize to -180 to 180 degrees
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360
        
        return angle_diff

    def _calculate_spine_angle(self, landmarks: List[LandmarkData]) -> float:
        """Calculate spine angle (forward/backward lean)"""
        nose = landmarks[self.LANDMARK_INDICES['nose']]
        left_shoulder = landmarks[self.LANDMARK_INDICES['left_shoulder']]
        right_shoulder = landmarks[self.LANDMARK_INDICES['right_shoulder']]
        left_hip = landmarks[self.LANDMARK_INDICES['left_hip']]
        right_hip = landmarks[self.LANDMARK_INDICES['right_hip']]
        
        if (nose.visibility < 0.5 or left_shoulder.visibility < 0.5 or
            right_shoulder.visibility < 0.5 or left_hip.visibility < 0.5 or
            right_hip.visibility < 0.5):
            return 0.0
        
        # Calculate shoulder and hip centers
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_center_y = (left_hip.y + right_hip.y) / 2
        
        # Calculate spine angle
        spine_angle = math.degrees(math.atan2(
            shoulder_center_y - hip_center_y,
            0.1  # Small value to avoid division by zero
        )) - 90
        
        return spine_angle

    def _calculate_arm_angle(self, landmarks: List[LandmarkData]) -> float:
        """Calculate average arm angle"""
        left_shoulder = landmarks[self.LANDMARK_INDICES['left_shoulder']]
        left_wrist = landmarks[self.LANDMARK_INDICES['left_wrist']]
        right_shoulder = landmarks[self.LANDMARK_INDICES['right_shoulder']]
        right_wrist = landmarks[self.LANDMARK_INDICES['right_wrist']]
        
        if (left_shoulder.visibility < 0.5 or left_wrist.visibility < 0.5 or
            right_shoulder.visibility < 0.5 or right_wrist.visibility < 0.5):
            return 0.0
        
        # Calculate left arm angle
        left_arm_angle = math.degrees(math.atan2(
            left_wrist.y - left_shoulder.y,
            left_wrist.x - left_shoulder.x
        ))
        
        # Calculate right arm angle
        right_arm_angle = math.degrees(math.atan2(
            right_wrist.y - right_shoulder.y,
            right_wrist.x - right_shoulder.x
        ))
        
        # Return average
        return (left_arm_angle + right_arm_angle) / 2

    def _calculate_posture_stability(self, landmarks: List[LandmarkData]) -> float:
        """Calculate posture stability based on body alignment"""
        left_shoulder = landmarks[self.LANDMARK_INDICES['left_shoulder']]
        right_shoulder = landmarks[self.LANDMARK_INDICES['right_shoulder']]
        left_hip = landmarks[self.LANDMARK_INDICES['left_hip']]
        right_hip = landmarks[self.LANDMARK_INDICES['right_hip']]
        left_ankle = landmarks[self.LANDMARK_INDICES['left_ankle']]
        right_ankle = landmarks[self.LANDMARK_INDICES['right_ankle']]
        
        if (left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5 or
            left_hip.visibility < 0.5 or right_hip.visibility < 0.5 or
            left_ankle.visibility < 0.5 or right_ankle.visibility < 0.5):
            return 0.0
        
        # Calculate alignment differences
        shoulder_alignment = abs(left_shoulder.y - right_shoulder.y)
        hip_alignment = abs(left_hip.y - right_hip.y)
        ankle_alignment = abs(left_ankle.y - right_ankle.y)
        
        # Lower alignment differences = higher stability
        alignment_score = 1.0 - (shoulder_alignment + hip_alignment + ankle_alignment) / 3
        
        return max(0.0, min(1.0, alignment_score))

    def _calculate_overall_confidence(self, landmarks: List[LandmarkData]) -> float:
        """Calculate overall confidence based on landmark visibility"""
        key_landmarks = [
            self.LANDMARK_INDICES['left_shoulder'], self.LANDMARK_INDICES['right_shoulder'],
            self.LANDMARK_INDICES['left_hip'], self.LANDMARK_INDICES['right_hip'],
            self.LANDMARK_INDICES['left_wrist'], self.LANDMARK_INDICES['right_wrist'],
            self.LANDMARK_INDICES['left_elbow'], self.LANDMARK_INDICES['right_elbow']
        ]
        
        total_visibility = 0.0
        valid_landmarks = 0
        
        for landmark_idx in key_landmarks:
            if landmark_idx < len(landmarks):
                visibility = landmarks[landmark_idx].visibility
                total_visibility += visibility
                valid_landmarks += 1
        
        if valid_landmarks == 0:
            return 0.0
        
        return total_visibility / valid_landmarks

    def _calculate_velocities_and_accelerations(self, frame_data: FrameData):
        """Calculate velocities and accelerations from previous frames"""
        if len(self.landmark_buffer) < 2:
            return
        
        # Get previous frame
        prev_frame = self.landmark_buffer[-1]
        
        # Calculate hand velocity
        current_hand = frame_data.hand_center
        prev_hand = prev_frame.hand_center
        
        frame_data.hand_velocity = math.sqrt(
            (current_hand[0] - prev_hand[0])**2 + 
            (current_hand[1] - prev_hand[1])**2
        )
        
        # Calculate shoulder velocity
        current_shoulder = frame_data.shoulder_center
        prev_shoulder = prev_frame.shoulder_center
        
        frame_data.shoulder_velocity = math.sqrt(
            (current_shoulder[0] - prev_shoulder[0])**2 + 
            (current_shoulder[1] - prev_shoulder[1])**2
        )
        
        # Calculate hip velocity
        current_hip = frame_data.hip_center
        prev_hip = prev_frame.hip_center
        
        frame_data.hip_velocity = math.sqrt(
            (current_hip[0] - prev_hip[0])**2 + 
            (current_hip[1] - prev_hip[1])**2
        )
        
        # Calculate hand acceleration
        if len(self.landmark_buffer) >= 2:
            prev_velocity = prev_frame.hand_velocity
            frame_data.hand_acceleration = frame_data.hand_velocity - prev_velocity
        
        # Calculate movement stability
        frame_data.movement_stability = self._calculate_movement_stability(frame_data)

    def _calculate_movement_stability(self, frame_data: FrameData) -> float:
        """Calculate movement stability over recent frames"""
        if len(self.landmark_buffer) < 3:
            return 1.0
        
        # Calculate variance in hand positions over recent frames
        hand_positions = []
        
        # Add current frame
        hand_positions.append(frame_data.hand_center)
        
        # Add previous frames
        for prev_frame in list(self.landmark_buffer)[-3:]:
            hand_positions.append(prev_frame.hand_center)
        
        if len(hand_positions) < 2:
            return 1.0
        
        # Calculate variance
        x_positions = [pos[0] for pos in hand_positions]
        y_positions = [pos[1] for pos in hand_positions]
        
        x_variance = np.var(x_positions)
        y_variance = np.var(y_positions)
        
        # Stability is inverse of variance
        stability = 1.0 / (1.0 + x_variance + y_variance)
        
        return min(1.0, max(0.0, stability))

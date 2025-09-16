#!/usr/bin/env python3
"""
Smart Golf Swing Pose Analysis Module

Enhanced pose detection system specifically designed for golf swing analysis
from behind-the-golfer videos. Uses biomechanical analysis and multi-landmark
fusion for improved accuracy.
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
class SwingFeatures:
    """Comprehensive swing features for a single frame"""
    frame_index: int
    timestamp: float
    
    # Basic landmark positions
    landmarks: List[Dict]
    
    # Calculated angles (degrees)
    shoulder_turn_angle: float
    spine_angle: float
    arm_plane_angle: float
    hip_sway: float
    knee_flexion: float
    
    # Velocities (pixels/frame)
    hand_velocity: float
    shoulder_velocity: float
    hip_velocity: float
    
    # Accelerations (pixels/frame²)
    hand_acceleration: float
    
    # Distances and positions
    shoulder_hip_separation: float
    weight_distribution: float
    club_angle_estimate: float
    
    # Stability metrics
    movement_stability: float
    posture_stability: float
    
    # Confidence scores
    landmark_confidence: float
    feature_confidence: float

@dataclass
class SwingEvent:
    """Detected swing event with confidence"""
    phase: SwingPhase
    frame_index: int
    timestamp: float
    confidence: float
    detection_method: str
    features: Dict[str, Any]

class SwingFeatureExtractor:
    """Extract golf-specific features from pose landmarks"""
    
    def __init__(self):
        # MediaPipe pose landmark indices
        self.LANDMARKS = {
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
        
        logger.info("SwingFeatureExtractor initialized")
    
    def extract_features(self, landmarks_data: List[Dict]) -> List[SwingFeatures]:
        """Extract comprehensive swing features from landmarks data"""
        features = []
        
        for i, frame_data in enumerate(landmarks_data):
            if len(frame_data["landmarks"]) < 33:
                continue
                
            landmarks = frame_data["landmarks"]
            
            # Calculate all features for this frame
            frame_features = SwingFeatures(
                frame_index=frame_data["frame_index"],
                timestamp=frame_data["timestamp"],
                landmarks=landmarks,
                
                # Angles
                shoulder_turn_angle=self._calculate_shoulder_turn_angle(landmarks),
                spine_angle=self._calculate_spine_angle(landmarks),
                arm_plane_angle=self._calculate_arm_plane_angle(landmarks),
                hip_sway=self._calculate_hip_sway(landmarks),
                knee_flexion=self._calculate_knee_flexion(landmarks),
                
                # Velocities (calculated from previous frames)
                hand_velocity=self._calculate_hand_velocity(landmarks, features[-1] if features else None),
                shoulder_velocity=self._calculate_shoulder_velocity(landmarks, features[-1] if features else None),
                hip_velocity=self._calculate_hip_velocity(landmarks, features[-1] if features else None),
                
                # Accelerations
                hand_acceleration=self._calculate_hand_acceleration(landmarks, features[-2:] if len(features) >= 2 else []),
                
                # Distances and positions
                shoulder_hip_separation=self._calculate_shoulder_hip_separation(landmarks),
                weight_distribution=self._calculate_weight_distribution(landmarks),
                club_angle_estimate=self._estimate_club_angle(landmarks),
                
                # Stability metrics
                movement_stability=self._calculate_movement_stability(landmarks, features[-4:] if len(features) >= 4 else []),
                posture_stability=self._calculate_posture_stability(landmarks),
                
                # Confidence scores
                landmark_confidence=self._calculate_landmark_confidence(landmarks),
                feature_confidence=0.0  # Will be calculated after all features are extracted
            )
            
            features.append(frame_features)
        
        # Calculate feature confidence scores
        self._calculate_feature_confidence(features)
        
        logger.info(
            "Extracted swing features",
            total_frames=len(features),
            features_per_frame=len(vars(features[0])) if features else 0
        )
        
        return features
    
    def _calculate_shoulder_turn_angle(self, landmarks: List[Dict]) -> float:
        """Calculate shoulder turn angle relative to hips"""
        try:
            left_shoulder = landmarks[self.LANDMARKS['left_shoulder']]
            right_shoulder = landmarks[self.LANDMARKS['right_shoulder']]
            left_hip = landmarks[self.LANDMARKS['left_hip']]
            right_hip = landmarks[self.LANDMARKS['right_hip']]
            
            # Check visibility
            if (left_shoulder['visibility'] < 0.5 or right_shoulder['visibility'] < 0.5 or
                left_hip['visibility'] < 0.5 or right_hip['visibility'] < 0.5):
                return 0.0
            
            # Calculate shoulder line vector
            shoulder_center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
            shoulder_center_y = (left_shoulder['y'] + right_shoulder['y']) / 2
            
            # Calculate hip line vector
            hip_center_x = (left_hip['x'] + right_hip['x']) / 2
            hip_center_y = (left_hip['y'] + right_hip['y']) / 2
            
            # Calculate angle between shoulder line and hip line
            shoulder_line_angle = math.atan2(
                right_shoulder['y'] - left_shoulder['y'],
                right_shoulder['x'] - left_shoulder['x']
            )
            
            hip_line_angle = math.atan2(
                right_hip['y'] - left_hip['y'],
                right_hip['x'] - left_hip['x']
            )
            
            # Calculate turn angle (difference between shoulder and hip angles)
            turn_angle = math.degrees(shoulder_line_angle - hip_line_angle)
            
            # Normalize to -180 to 180 degrees
            while turn_angle > 180:
                turn_angle -= 360
            while turn_angle < -180:
                turn_angle += 360
                
            return turn_angle
            
        except (IndexError, KeyError, ZeroDivisionError):
            return 0.0
    
    def _calculate_spine_angle(self, landmarks: List[Dict]) -> float:
        """Calculate spine angle (forward bend)"""
        try:
            nose = landmarks[self.LANDMARKS['nose']]
            left_shoulder = landmarks[self.LANDMARKS['left_shoulder']]
            right_shoulder = landmarks[self.LANDMARKS['right_shoulder']]
            left_hip = landmarks[self.LANDMARKS['left_hip']]
            right_hip = landmarks[self.LANDMARKS['right_hip']]
            
            # Check visibility
            if (nose['visibility'] < 0.5 or left_shoulder['visibility'] < 0.5 or
                right_shoulder['visibility'] < 0.5 or left_hip['visibility'] < 0.5 or
                right_hip['visibility'] < 0.5):
                return 0.0
            
            # Calculate shoulder and hip centers
            shoulder_center_y = (left_shoulder['y'] + right_shoulder['y']) / 2
            hip_center_y = (left_hip['y'] + right_hip['y']) / 2
            
            # Calculate spine angle (forward bend)
            # Positive angle = forward bend, negative = backward lean
            spine_angle = math.degrees(math.atan2(
                shoulder_center_y - hip_center_y,
                0.1  # Small value to avoid division by zero
            )) - 90
            
            return spine_angle
            
        except (IndexError, KeyError, ZeroDivisionError):
            return 0.0
    
    def _calculate_arm_plane_angle(self, landmarks: List[Dict]) -> float:
        """Calculate arm plane angle relative to spine"""
        try:
            left_shoulder = landmarks[self.LANDMARKS['left_shoulder']]
            right_shoulder = landmarks[self.LANDMARKS['right_shoulder']]
            left_elbow = landmarks[self.LANDMARKS['left_elbow']]
            right_elbow = landmarks[self.LANDMARKS['right_elbow']]
            left_wrist = landmarks[self.LANDMARKS['left_wrist']]
            right_wrist = landmarks[self.LANDMARKS['right_wrist']]
            
            # Check visibility
            if (left_shoulder['visibility'] < 0.5 or right_shoulder['visibility'] < 0.5 or
                left_elbow['visibility'] < 0.5 or right_elbow['visibility'] < 0.5 or
                left_wrist['visibility'] < 0.5 or right_wrist['visibility'] < 0.5):
                return 0.0
            
            # Calculate average arm position
            arm_center_y = (left_elbow['y'] + right_elbow['y'] + left_wrist['y'] + right_wrist['y']) / 4
            shoulder_center_y = (left_shoulder['y'] + right_shoulder['y']) / 2
            
            # Calculate arm plane angle
            arm_plane_angle = math.degrees(math.atan2(
                arm_center_y - shoulder_center_y,
                0.1
            ))
            
            return arm_plane_angle
            
        except (IndexError, KeyError, ZeroDivisionError):
            return 0.0
    
    def _calculate_hip_sway(self, landmarks: List[Dict]) -> float:
        """Calculate hip sway (lateral movement)"""
        try:
            left_hip = landmarks[self.LANDMARKS['left_hip']]
            right_hip = landmarks[self.LANDMARKS['right_hip']]
            
            # Check visibility
            if left_hip['visibility'] < 0.5 or right_hip['visibility'] < 0.5:
                return 0.0
            
            # Calculate hip center position
            hip_center_x = (left_hip['x'] + right_hip['x']) / 2
            
            # Hip sway is the deviation from center (0.5)
            hip_sway = (hip_center_x - 0.5) * 100  # Convert to percentage
            
            return hip_sway
            
        except (IndexError, KeyError):
            return 0.0
    
    def _calculate_knee_flexion(self, landmarks: List[Dict]) -> float:
        """Calculate knee flexion angle"""
        try:
            left_hip = landmarks[self.LANDMARKS['left_hip']]
            left_knee = landmarks[self.LANDMARKS['left_knee']]
            left_ankle = landmarks[self.LANDMARKS['left_ankle']]
            right_hip = landmarks[self.LANDMARKS['right_hip']]
            right_knee = landmarks[self.LANDMARKS['right_knee']]
            right_ankle = landmarks[self.LANDMARKS['right_ankle']]
            
            # Check visibility
            if (left_hip['visibility'] < 0.5 or left_knee['visibility'] < 0.5 or
                left_ankle['visibility'] < 0.5 or right_hip['visibility'] < 0.5 or
                right_knee['visibility'] < 0.5 or right_ankle['visibility'] < 0.5):
                return 0.0
            
            # Calculate knee angles for both legs
            left_angle = self._calculate_angle_between_points(
                left_hip, left_knee, left_ankle
            )
            right_angle = self._calculate_angle_between_points(
                right_hip, right_knee, right_ankle
            )
            
            # Return average knee flexion
            return (left_angle + right_angle) / 2
            
        except (IndexError, KeyError, ZeroDivisionError):
            return 0.0
    
    def _calculate_angle_between_points(self, p1: Dict, p2: Dict, p3: Dict) -> float:
        """Calculate angle between three points (p2 is the vertex)"""
        try:
            # Vector from p2 to p1
            v1_x = p1['x'] - p2['x']
            v1_y = p1['y'] - p2['y']
            
            # Vector from p2 to p3
            v2_x = p3['x'] - p2['x']
            v2_y = p3['y'] - p2['y']
            
            # Calculate angle using dot product
            dot_product = v1_x * v2_x + v1_y * v2_y
            magnitude1 = math.sqrt(v1_x**2 + v1_y**2)
            magnitude2 = math.sqrt(v2_x**2 + v2_y**2)
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            cos_angle = dot_product / (magnitude1 * magnitude2)
            cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
            
            angle = math.degrees(math.acos(cos_angle))
            return angle
            
        except (ValueError, ZeroDivisionError):
            return 0.0
    
    def _calculate_hand_velocity(self, landmarks: List[Dict], prev_features: Optional[SwingFeatures]) -> float:
        """Calculate hand velocity"""
        if prev_features is None:
            return 0.0
        
        try:
            left_wrist = landmarks[self.LANDMARKS['left_wrist']]
            right_wrist = landmarks[self.LANDMARKS['right_wrist']]
            
            # Check visibility
            if left_wrist['visibility'] < 0.5 or right_wrist['visibility'] < 0.5:
                return 0.0
            
            # Calculate current hand center
            current_hand_x = (left_wrist['x'] + right_wrist['x']) / 2
            current_hand_y = (left_wrist['y'] + right_wrist['y']) / 2
            
            # Get previous hand position from landmarks
            prev_landmarks = prev_features.landmarks
            prev_left_wrist = prev_landmarks[self.LANDMARKS['left_wrist']]
            prev_right_wrist = prev_landmarks[self.LANDMARKS['right_wrist']]
            
            if prev_left_wrist['visibility'] < 0.5 or prev_right_wrist['visibility'] < 0.5:
                return 0.0
            
            prev_hand_x = (prev_left_wrist['x'] + prev_right_wrist['x']) / 2
            prev_hand_y = (prev_left_wrist['y'] + prev_right_wrist['y']) / 2
            
            # Calculate velocity (pixels per frame)
            velocity = math.sqrt(
                (current_hand_x - prev_hand_x)**2 + 
                (current_hand_y - prev_hand_y)**2
            )
            
            return velocity
            
        except (IndexError, KeyError):
            return 0.0
    
    def _calculate_shoulder_velocity(self, landmarks: List[Dict], prev_features: Optional[SwingFeatures]) -> float:
        """Calculate shoulder velocity"""
        if prev_features is None:
            return 0.0
        
        try:
            left_shoulder = landmarks[self.LANDMARKS['left_shoulder']]
            right_shoulder = landmarks[self.LANDMARKS['right_shoulder']]
            
            # Check visibility
            if left_shoulder['visibility'] < 0.5 or right_shoulder['visibility'] < 0.5:
                return 0.0
            
            # Calculate current shoulder center
            current_shoulder_x = (left_shoulder['x'] + right_shoulder['x']) / 2
            current_shoulder_y = (left_shoulder['y'] + right_shoulder['y']) / 2
            
            # Get previous shoulder position
            prev_landmarks = prev_features.landmarks
            prev_left_shoulder = prev_landmarks[self.LANDMARKS['left_shoulder']]
            prev_right_shoulder = prev_landmarks[self.LANDMARKS['right_shoulder']]
            
            if prev_left_shoulder['visibility'] < 0.5 or prev_right_shoulder['visibility'] < 0.5:
                return 0.0
            
            prev_shoulder_x = (prev_left_shoulder['x'] + prev_right_shoulder['x']) / 2
            prev_shoulder_y = (prev_left_shoulder['y'] + prev_right_shoulder['y']) / 2
            
            # Calculate velocity
            velocity = math.sqrt(
                (current_shoulder_x - prev_shoulder_x)**2 + 
                (current_shoulder_y - prev_shoulder_y)**2
            )
            
            return velocity
            
        except (IndexError, KeyError):
            return 0.0
    
    def _calculate_hip_velocity(self, landmarks: List[Dict], prev_features: Optional[SwingFeatures]) -> float:
        """Calculate hip velocity"""
        if prev_features is None:
            return 0.0
        
        try:
            left_hip = landmarks[self.LANDMARKS['left_hip']]
            right_hip = landmarks[self.LANDMARKS['right_hip']]
            
            # Check visibility
            if left_hip['visibility'] < 0.5 or right_hip['visibility'] < 0.5:
                return 0.0
            
            # Calculate current hip center
            current_hip_x = (left_hip['x'] + right_hip['x']) / 2
            current_hip_y = (left_hip['y'] + right_hip['y']) / 2
            
            # Get previous hip position
            prev_landmarks = prev_features.landmarks
            prev_left_hip = prev_landmarks[self.LANDMARKS['left_hip']]
            prev_right_hip = prev_landmarks[self.LANDMARKS['right_hip']]
            
            if prev_left_hip['visibility'] < 0.5 or prev_right_hip['visibility'] < 0.5:
                return 0.0
            
            prev_hip_x = (prev_left_hip['x'] + prev_right_hip['x']) / 2
            prev_hip_y = (prev_left_hip['y'] + prev_right_hip['y']) / 2
            
            # Calculate velocity
            velocity = math.sqrt(
                (current_hip_x - prev_hip_x)**2 + 
                (current_hip_y - prev_hip_y)**2
            )
            
            return velocity
            
        except (IndexError, KeyError):
            return 0.0
    
    def _calculate_hand_acceleration(self, landmarks: List[Dict], prev_features: List[SwingFeatures]) -> float:
        """Calculate hand acceleration"""
        if len(prev_features) < 2:
            return 0.0
        
        try:
            # Get current velocity
            current_velocity = self._calculate_hand_velocity(landmarks, prev_features[-1])
            
            # Get previous velocity
            prev_velocity = prev_features[-1].hand_velocity
            
            # Calculate acceleration (change in velocity per frame)
            acceleration = current_velocity - prev_velocity
            
            return acceleration
            
        except (IndexError, KeyError):
            return 0.0
    
    def _calculate_shoulder_hip_separation(self, landmarks: List[Dict]) -> float:
        """Calculate shoulder-hip separation (X-axis difference)"""
        try:
            left_shoulder = landmarks[self.LANDMARKS['left_shoulder']]
            right_shoulder = landmarks[self.LANDMARKS['right_shoulder']]
            left_hip = landmarks[self.LANDMARKS['left_hip']]
            right_hip = landmarks[self.LANDMARKS['right_hip']]
            
            # Check visibility
            if (left_shoulder['visibility'] < 0.5 or right_shoulder['visibility'] < 0.5 or
                left_hip['visibility'] < 0.5 or right_hip['visibility'] < 0.5):
                return 0.0
            
            # Calculate centers
            shoulder_center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
            hip_center_x = (left_hip['x'] + right_hip['x']) / 2
            
            # Calculate separation
            separation = abs(shoulder_center_x - hip_center_x) * 100  # Convert to percentage
            
            return separation
            
        except (IndexError, KeyError):
            return 0.0
    
    def _calculate_weight_distribution(self, landmarks: List[Dict]) -> float:
        """Calculate weight distribution between feet"""
        try:
            left_ankle = landmarks[self.LANDMARKS['left_ankle']]
            right_ankle = landmarks[self.LANDMARKS['right_ankle']]
            left_hip = landmarks[self.LANDMARKS['left_hip']]
            right_hip = landmarks[self.LANDMARKS['right_hip']]
            
            # Check visibility
            if (left_ankle['visibility'] < 0.5 or right_ankle['visibility'] < 0.5 or
                left_hip['visibility'] < 0.5 or right_hip['visibility'] < 0.5):
                return 50.0  # Default to balanced
            
            # Calculate hip center
            hip_center_x = (left_hip['x'] + right_hip['x']) / 2
            
            # Calculate foot center
            foot_center_x = (left_ankle['x'] + right_ankle['x']) / 2
            
            # Weight distribution based on hip position relative to feet
            # Positive = weight on right foot, negative = weight on left foot
            weight_distribution = (hip_center_x - foot_center_x) * 100
            
            return weight_distribution
            
        except (IndexError, KeyError):
            return 50.0
    
    def _estimate_club_angle(self, landmarks: List[Dict]) -> float:
        """Estimate club angle from arm positions"""
        try:
            left_shoulder = landmarks[self.LANDMARKS['left_shoulder']]
            left_wrist = landmarks[self.LANDMARKS['left_wrist']]
            right_shoulder = landmarks[self.LANDMARKS['right_shoulder']]
            right_wrist = landmarks[self.LANDMARKS['right_wrist']]
            
            # Check visibility
            if (left_shoulder['visibility'] < 0.5 or left_wrist['visibility'] < 0.5 or
                right_shoulder['visibility'] < 0.5 or right_wrist['visibility'] < 0.5):
                return 0.0
            
            # Calculate arm vectors
            left_arm_angle = math.degrees(math.atan2(
                left_wrist['y'] - left_shoulder['y'],
                left_wrist['x'] - left_shoulder['x']
            ))
            
            right_arm_angle = math.degrees(math.atan2(
                right_wrist['y'] - right_shoulder['y'],
                right_wrist['x'] - right_shoulder['x']
            ))
            
            # Estimate club angle as average of arm angles
            club_angle = (left_arm_angle + right_arm_angle) / 2
            
            return club_angle
            
        except (IndexError, KeyError, ZeroDivisionError):
            return 0.0
    
    def _calculate_movement_stability(self, landmarks: List[Dict], prev_features: List[SwingFeatures]) -> float:
        """Calculate movement stability over recent frames"""
        if len(prev_features) < 3:
            return 1.0  # Assume stable if not enough history
        
        try:
            # Calculate variance in hand positions over recent frames
            hand_positions = []
            
            # Add current frame
            left_wrist = landmarks[self.LANDMARKS['left_wrist']]
            right_wrist = landmarks[self.LANDMARKS['right_wrist']]
            
            if left_wrist['visibility'] > 0.5 and right_wrist['visibility'] > 0.5:
                hand_x = (left_wrist['x'] + right_wrist['x']) / 2
                hand_y = (left_wrist['y'] + right_wrist['y']) / 2
                hand_positions.append((hand_x, hand_y))
            
            # Add previous frames
            for prev_feat in prev_features[-3:]:  # Last 3 frames
                prev_landmarks = prev_feat.landmarks
                prev_left_wrist = prev_landmarks[self.LANDMARKS['left_wrist']]
                prev_right_wrist = prev_landmarks[self.LANDMARKS['right_wrist']]
                
                if prev_left_wrist['visibility'] > 0.5 and prev_right_wrist['visibility'] > 0.5:
                    prev_hand_x = (prev_left_wrist['x'] + prev_right_wrist['x']) / 2
                    prev_hand_y = (prev_left_wrist['y'] + prev_right_wrist['y']) / 2
                    hand_positions.append((prev_hand_x, prev_hand_y))
            
            if len(hand_positions) < 2:
                return 1.0
            
            # Calculate variance
            x_positions = [pos[0] for pos in hand_positions]
            y_positions = [pos[1] for pos in hand_positions]
            
            x_variance = np.var(x_positions)
            y_variance = np.var(y_positions)
            
            # Stability is inverse of variance (higher variance = lower stability)
            stability = 1.0 / (1.0 + x_variance + y_variance)
            
            return min(1.0, max(0.0, stability))
            
        except (IndexError, KeyError, ZeroDivisionError):
            return 1.0
    
    def _calculate_posture_stability(self, landmarks: List[Dict]) -> float:
        """Calculate posture stability based on body alignment"""
        try:
            # Check if key landmarks are visible and aligned
            left_shoulder = landmarks[self.LANDMARKS['left_shoulder']]
            right_shoulder = landmarks[self.LANDMARKS['right_shoulder']]
            left_hip = landmarks[self.LANDMARKS['left_hip']]
            right_hip = landmarks[self.LANDMARKS['right_hip']]
            left_ankle = landmarks[self.LANDMARKS['left_ankle']]
            right_ankle = landmarks[self.LANDMARKS['right_ankle']]
            
            # Check visibility
            if (left_shoulder['visibility'] < 0.5 or right_shoulder['visibility'] < 0.5 or
                left_hip['visibility'] < 0.5 or right_hip['visibility'] < 0.5 or
                left_ankle['visibility'] < 0.5 or right_ankle['visibility'] < 0.5):
                return 0.0
            
            # Calculate alignment scores
            shoulder_alignment = abs(left_shoulder['y'] - right_shoulder['y'])
            hip_alignment = abs(left_hip['y'] - right_hip['y'])
            ankle_alignment = abs(left_ankle['y'] - right_ankle['y'])
            
            # Lower alignment differences = higher stability
            alignment_score = 1.0 - (shoulder_alignment + hip_alignment + ankle_alignment) / 3
            
            return max(0.0, min(1.0, alignment_score))
            
        except (IndexError, KeyError):
            return 0.0
    
    def _calculate_landmark_confidence(self, landmarks: List[Dict]) -> float:
        """Calculate overall landmark confidence"""
        try:
            # Calculate average visibility of key landmarks
            key_landmarks = [
                self.LANDMARKS['left_shoulder'], self.LANDMARKS['right_shoulder'],
                self.LANDMARKS['left_hip'], self.LANDMARKS['right_hip'],
                self.LANDMARKS['left_wrist'], self.LANDMARKS['right_wrist'],
                self.LANDMARKS['left_elbow'], self.LANDMARKS['right_elbow']
            ]
            
            total_visibility = 0.0
            valid_landmarks = 0
            
            for landmark_idx in key_landmarks:
                if landmark_idx < len(landmarks):
                    visibility = landmarks[landmark_idx]['visibility']
                    total_visibility += visibility
                    valid_landmarks += 1
            
            if valid_landmarks == 0:
                return 0.0
            
            return total_visibility / valid_landmarks
            
        except (IndexError, KeyError):
            return 0.0
    
    def _calculate_feature_confidence(self, features: List[SwingFeatures]):
        """Calculate feature confidence scores"""
        if not features:
            return
        
        # Calculate confidence based on feature consistency and landmark quality
        for i, feature in enumerate(features):
            confidence_factors = []
            
            # Landmark confidence
            confidence_factors.append(feature.landmark_confidence)
            
            # Feature consistency (compare with neighbors)
            if i > 0 and i < len(features) - 1:
                prev_feature = features[i-1]
                next_feature = features[i+1]
                
                # Check consistency of key features
                angle_consistency = 1.0 - abs(feature.shoulder_turn_angle - 
                    (prev_feature.shoulder_turn_angle + next_feature.shoulder_turn_angle) / 2) / 180.0
                confidence_factors.append(max(0.0, angle_consistency))
            
            # Movement stability
            confidence_factors.append(feature.movement_stability)
            
            # Posture stability
            confidence_factors.append(feature.posture_stability)
            
            # Calculate overall feature confidence
            feature.feature_confidence = sum(confidence_factors) / len(confidence_factors)

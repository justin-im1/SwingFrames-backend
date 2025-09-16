#!/usr/bin/env python3
"""
Biomechanical Golf Swing Detector

A new approach to golf swing analysis based on biomechanical principles.
Instead of complex velocity calculations and finger tracking, this detector
identifies swing positions based on how the human body should look at each phase.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import structlog
from dataclasses import dataclass
from enum import Enum
import math

from app.services.smart_pose_analysis import SwingFeatures, SwingPhase, SwingEvent

logger = structlog.get_logger()

@dataclass
class BodyPosition:
    """Represents the body position at a specific frame"""
    frame_index: int
    timestamp: float
    
    # Key body landmarks (normalized coordinates)
    nose: Tuple[float, float]
    left_shoulder: Tuple[float, float]
    right_shoulder: Tuple[float, float]
    left_elbow: Tuple[float, float]
    right_elbow: Tuple[float, float]
    left_wrist: Tuple[float, float]
    right_wrist: Tuple[float, float]
    left_hip: Tuple[float, float]
    right_hip: Tuple[float, float]
    left_knee: Tuple[float, float]
    right_knee: Tuple[float, float]
    left_ankle: Tuple[float, float]
    right_ankle: Tuple[float, float]
    
    # Calculated body metrics
    shoulder_center: Tuple[float, float]
    hip_center: Tuple[float, float]
    spine_angle: float  # Forward/backward lean
    shoulder_hip_angle: float  # Rotation between shoulders and hips
    arm_extension: float  # How extended the arms are
    stance_width: float  # Distance between feet
    weight_balance: float  # Left/right weight distribution

class BiomechanicalSwingDetector:
    """Detects golf swing positions based on biomechanical body positions"""
    
    def __init__(self):
        # Biomechanical thresholds based on golf swing mechanics
        # Adjusted based on actual data analysis
        self.SETUP_SPINE_ANGLE_RANGE = (-20, 20)  # degrees - slight forward lean
        self.SETUP_SHOULDER_HIP_ANGLE_RANGE = (-30, 30)  # degrees - minimal rotation
        self.SETUP_ARM_EXTENSION_RANGE = (0.05, 0.20)  # normalized distance
        
        self.TOP_SHOULDER_HIP_ANGLE_MIN = 30  # degrees - significant shoulder turn
        self.TOP_ARM_EXTENSION_MIN = 0.10  # arms should be extended
        self.TOP_SPINE_ANGLE_RANGE = (-25, 25)  # maintain spine angle
        
        self.IMPACT_SHOULDER_HIP_ANGLE_RANGE = (-20, 20)  # degrees - returning to square
        self.IMPACT_ARM_EXTENSION_RANGE = (0.08, 0.18)  # arms returning to impact position
        self.IMPACT_WEIGHT_BALANCE_RANGE = (-30, 30)  # weight transferring
        
        self.FOLLOW_THROUGH_SHOULDER_HIP_ANGLE_MIN = 40  # degrees - full rotation
        self.FOLLOW_THROUGH_ARM_EXTENSION_MIN = 0.12  # arms extended behind body
        
        logger.info("BiomechanicalSwingDetector initialized")
    
    def detect_swing_events(self, features: List[SwingFeatures]) -> Dict[str, SwingEvent]:
        """Detect swing events using biomechanical analysis"""
        if not features:
            return self._create_empty_events()
        
        logger.info("Starting biomechanical swing detection", total_frames=len(features))
        
        # Convert features to body positions
        body_positions = self._extract_body_positions(features)
        
        if not body_positions:
            logger.warning("No valid body positions extracted")
            return self._create_empty_events()
        
        # Detect each swing phase
        setup_event = self._detect_setup_biomechanical(body_positions)
        top_event = self._detect_top_biomechanical(body_positions, setup_event)
        impact_event = self._detect_impact_biomechanical(body_positions, setup_event, top_event)
        follow_through_event = self._detect_follow_through_biomechanical(body_positions, impact_event)
        
        events = {
            "setup": setup_event,
            "top_backswing": top_event,
            "impact": impact_event,
            "follow_through": follow_through_event
        }
        
        # Validate sequence
        validated_events = self._validate_sequence(events, body_positions)
        
        logger.info(
            "Biomechanical swing events detected",
            setup=validated_events["setup"].frame_index,
            top_backswing=validated_events["top_backswing"].frame_index,
            impact=validated_events["impact"].frame_index,
            follow_through=validated_events["follow_through"].frame_index
        )
        
        return validated_events
    
    def _extract_body_positions(self, features: List[SwingFeatures]) -> List[BodyPosition]:
        """Extract body positions from swing features"""
        body_positions = []
        
        for feature in features:
            if len(feature.landmarks) < 33:
                continue
            
            landmarks = feature.landmarks
            
            # Extract key landmarks
            try:
                body_pos = BodyPosition(
                    frame_index=feature.frame_index,
                    timestamp=feature.timestamp,
                    
                    # Extract landmark coordinates
                    nose=self._get_landmark_coords(landmarks, 0),
                    left_shoulder=self._get_landmark_coords(landmarks, 11),
                    right_shoulder=self._get_landmark_coords(landmarks, 12),
                    left_elbow=self._get_landmark_coords(landmarks, 13),
                    right_elbow=self._get_landmark_coords(landmarks, 14),
                    left_wrist=self._get_landmark_coords(landmarks, 15),
                    right_wrist=self._get_landmark_coords(landmarks, 16),
                    left_hip=self._get_landmark_coords(landmarks, 23),
                    right_hip=self._get_landmark_coords(landmarks, 24),
                    left_knee=self._get_landmark_coords(landmarks, 25),
                    right_knee=self._get_landmark_coords(landmarks, 26),
                    left_ankle=self._get_landmark_coords(landmarks, 27),
                    right_ankle=self._get_landmark_coords(landmarks, 28),
                    
                    # Calculate derived metrics
                    shoulder_center=self._calculate_center(landmarks, 11, 12),
                    hip_center=self._calculate_center(landmarks, 23, 24),
                    spine_angle=self._calculate_spine_angle(landmarks),
                    shoulder_hip_angle=self._calculate_shoulder_hip_angle(landmarks),
                    arm_extension=self._calculate_arm_extension(landmarks),
                    stance_width=self._calculate_stance_width(landmarks),
                    weight_balance=self._calculate_weight_balance(landmarks)
                )
                
                body_positions.append(body_pos)
                
            except (IndexError, KeyError, ValueError) as e:
                logger.warning("Failed to extract body position", frame=feature.frame_index, error=str(e))
                continue
        
        logger.info("Extracted body positions", count=len(body_positions))
        return body_positions
    
    def _get_landmark_coords(self, landmarks: List[Dict], index: int) -> Tuple[float, float]:
        """Get landmark coordinates with visibility check"""
        if index >= len(landmarks):
            return (0.0, 0.0)
        
        landmark = landmarks[index]
        if landmark['visibility'] < 0.5:
            return (0.0, 0.0)
        
        return (landmark['x'], landmark['y'])
    
    def _calculate_center(self, landmarks: List[Dict], left_idx: int, right_idx: int) -> Tuple[float, float]:
        """Calculate center point between two landmarks"""
        left = self._get_landmark_coords(landmarks, left_idx)
        right = self._get_landmark_coords(landmarks, right_idx)
        
        if left == (0.0, 0.0) or right == (0.0, 0.0):
            return (0.0, 0.0)
        
        return ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
    
    def _calculate_spine_angle(self, landmarks: List[Dict]) -> float:
        """Calculate spine angle (forward/backward lean)"""
        shoulder_center = self._calculate_center(landmarks, 11, 12)
        hip_center = self._calculate_center(landmarks, 23, 24)
        
        if shoulder_center == (0.0, 0.0) or hip_center == (0.0, 0.0):
            return 0.0
        
        # Calculate spine angle as forward/backward lean
        # Positive = forward lean, negative = backward lean
        spine_vector_x = shoulder_center[0] - hip_center[0]
        spine_vector_y = shoulder_center[1] - hip_center[1]
        
        # Calculate angle from vertical (0 degrees = straight up)
        if spine_vector_y == 0:
            return 0.0
        
        angle = math.degrees(math.atan2(spine_vector_x, spine_vector_y))
        
        return angle
    
    def _calculate_shoulder_hip_angle(self, landmarks: List[Dict]) -> float:
        """Calculate rotation angle between shoulders and hips"""
        left_shoulder = self._get_landmark_coords(landmarks, 11)
        right_shoulder = self._get_landmark_coords(landmarks, 12)
        left_hip = self._get_landmark_coords(landmarks, 23)
        right_hip = self._get_landmark_coords(landmarks, 24)
        
        if (left_shoulder == (0.0, 0.0) or right_shoulder == (0.0, 0.0) or
            left_hip == (0.0, 0.0) or right_hip == (0.0, 0.0)):
            return 0.0
        
        # Calculate shoulder line vector
        shoulder_vector = (right_shoulder[0] - left_shoulder[0], right_shoulder[1] - left_shoulder[1])
        
        # Calculate hip line vector
        hip_vector = (right_hip[0] - left_hip[0], right_hip[1] - left_hip[1])
        
        # Calculate angle between vectors using dot product
        dot_product = shoulder_vector[0] * hip_vector[0] + shoulder_vector[1] * hip_vector[1]
        shoulder_magnitude = math.sqrt(shoulder_vector[0]**2 + shoulder_vector[1]**2)
        hip_magnitude = math.sqrt(hip_vector[0]**2 + hip_vector[1]**2)
        
        if shoulder_magnitude == 0 or hip_magnitude == 0:
            return 0.0
        
        cos_angle = dot_product / (shoulder_magnitude * hip_magnitude)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
        
        angle = math.degrees(math.acos(cos_angle))
        
        # Determine direction of rotation using cross product
        cross_product = shoulder_vector[0] * hip_vector[1] - shoulder_vector[1] * hip_vector[0]
        if cross_product < 0:
            angle = -angle
        
        return angle
    
    def _calculate_arm_extension(self, landmarks: List[Dict]) -> float:
        """Calculate how extended the arms are"""
        left_shoulder = self._get_landmark_coords(landmarks, 11)
        left_wrist = self._get_landmark_coords(landmarks, 15)
        right_shoulder = self._get_landmark_coords(landmarks, 12)
        right_wrist = self._get_landmark_coords(landmarks, 16)
        
        # Check if we have valid landmarks
        valid_arms = []
        
        if left_shoulder != (0.0, 0.0) and left_wrist != (0.0, 0.0):
            left_distance = math.sqrt(
                (left_wrist[0] - left_shoulder[0])**2 + 
                (left_wrist[1] - left_shoulder[1])**2
            )
            valid_arms.append(left_distance)
        
        if right_shoulder != (0.0, 0.0) and right_wrist != (0.0, 0.0):
            right_distance = math.sqrt(
                (right_wrist[0] - right_shoulder[0])**2 + 
                (right_wrist[1] - right_shoulder[1])**2
            )
            valid_arms.append(right_distance)
        
        if not valid_arms:
            return 0.0
        
        # Return average extension
        return sum(valid_arms) / len(valid_arms)
    
    def _calculate_stance_width(self, landmarks: List[Dict]) -> float:
        """Calculate stance width (distance between feet)"""
        left_ankle = self._get_landmark_coords(landmarks, 27)
        right_ankle = self._get_landmark_coords(landmarks, 28)
        
        if left_ankle == (0.0, 0.0) or right_ankle == (0.0, 0.0):
            return 0.0
        
        return math.sqrt(
            (right_ankle[0] - left_ankle[0])**2 + 
            (right_ankle[1] - left_ankle[1])**2
        )
    
    def _calculate_weight_balance(self, landmarks: List[Dict]) -> float:
        """Calculate weight distribution between feet"""
        hip_center = self._calculate_center(landmarks, 23, 24)
        ankle_center = self._calculate_center(landmarks, 27, 28)
        
        if hip_center == (0.0, 0.0) or ankle_center == (0.0, 0.0):
            return 0.0
        
        # Weight balance based on hip position relative to feet
        # Positive = weight on right foot, negative = weight on left foot
        balance = (hip_center[0] - ankle_center[0]) * 100
        
        return balance
    
    def _detect_setup_biomechanical(self, body_positions: List[BodyPosition]) -> SwingEvent:
        """Detect setup position based on biomechanical criteria"""
        logger.info("Detecting setup position biomechanically")
        
        # Look in first 30% of frames for setup
        setup_window = body_positions[:max(1, len(body_positions) // 3)]
        
        if not setup_window:
            return self._create_default_event(SwingPhase.SETUP, 0)
        
        best_setup_score = 0.0
        best_frame = 0
        
        for i, pos in enumerate(setup_window):
            setup_score = self._calculate_setup_score(pos)
            
            if setup_score > best_setup_score:
                best_setup_score = setup_score
                best_frame = i
        
        confidence = min(1.0, best_setup_score)
        
        return SwingEvent(
            phase=SwingPhase.SETUP,
            frame_index=setup_window[best_frame].frame_index,
            timestamp=setup_window[best_frame].timestamp,
            confidence=confidence,
            detection_method="biomechanical_setup",
            features={
                "spine_angle": setup_window[best_frame].spine_angle,
                "shoulder_hip_angle": setup_window[best_frame].shoulder_hip_angle,
                "arm_extension": setup_window[best_frame].arm_extension,
                "setup_score": best_setup_score
            }
        )
    
    def _calculate_setup_score(self, pos: BodyPosition) -> float:
        """Calculate how well this position matches setup criteria"""
        score = 0.0
        
        # Criterion 1: Minimal shoulder-hip rotation (50% weight) - most important for setup
        if self.SETUP_SHOULDER_HIP_ANGLE_RANGE[0] <= pos.shoulder_hip_angle <= self.SETUP_SHOULDER_HIP_ANGLE_RANGE[1]:
            rotation_score = 1.0 - abs(pos.shoulder_hip_angle) / 30.0
            score += rotation_score * 0.5
        
        # Criterion 2: Arms should be moderately extended (30% weight)
        if self.SETUP_ARM_EXTENSION_RANGE[0] <= pos.arm_extension <= self.SETUP_ARM_EXTENSION_RANGE[1]:
            extension_score = 1.0 - abs(pos.arm_extension - 0.125) / 0.075
            score += extension_score * 0.3
        
        # Criterion 3: Spine angle should be reasonable (20% weight)
        if self.SETUP_SPINE_ANGLE_RANGE[0] <= pos.spine_angle <= self.SETUP_SPINE_ANGLE_RANGE[1]:
            spine_score = 1.0 - abs(pos.spine_angle) / 20.0
            score += spine_score * 0.2
        
        return score
    
    def _detect_top_biomechanical(self, body_positions: List[BodyPosition], setup_event: SwingEvent) -> SwingEvent:
        """Detect top of backswing based on biomechanical criteria"""
        logger.info("Detecting top of backswing biomechanically")
        
        # Look after setup for top of backswing, but constrain to backswing portion only
        start_idx = setup_event.frame_index
        # Limit search to first 50% of remaining frames to capture the true peak
        remaining_frames = [pos for pos in body_positions if pos.frame_index > start_idx]
        max_search_frames = max(10, int(len(remaining_frames) * 0.5))  # At least 10 frames, max 50% of remaining
        search_window = remaining_frames[:max_search_frames]
        
        if not search_window:
            return self._create_default_event(SwingPhase.TOP, start_idx)
        
        # Try multiple approaches to find the true top of backswing
        
        # Approach 1: Use the improved scoring method
        best_score = 0.0
        best_frame = 0
        
        for i, pos in enumerate(search_window):
            score = self._calculate_top_of_backswing_score(pos, search_window, i)
            
            if score > best_score:
                best_score = score
                best_frame = i
        
        # Approach 2: Always try wrist height analysis as it's more reliable
        logger.info("Trying wrist height analysis for more precise detection")
        wrist_based_frame = self._find_top_by_wrist_height(search_window)
        if wrist_based_frame is not None:
            # Use wrist height analysis if it's different from scoring method
            if wrist_based_frame != best_frame:
                logger.info(f"Wrist height analysis found different frame: {search_window[wrist_based_frame].frame_index} vs {search_window[best_frame].frame_index}")
                # Prefer wrist height analysis as it's more precise
                best_frame = wrist_based_frame
                best_score = 0.8  # High confidence for wrist-based detection
                logger.info(f"Using wrist height analysis: frame {search_window[best_frame].frame_index}")
            else:
                logger.info(f"Wrist height analysis confirms scoring method: frame {search_window[best_frame].frame_index}")
                best_score = max(best_score, 0.8)  # Boost confidence
        
        # Approach 3: If still low confidence, try shoulder position analysis
        if best_score < 0.4:
            logger.info("Low confidence, trying shoulder position analysis")
            shoulder_based_frame = self._find_top_by_shoulder_position(search_window)
            if shoulder_based_frame is not None:
                best_frame = shoulder_based_frame
                best_score = 0.6  # Higher confidence for shoulder-based detection
                logger.info(f"Using shoulder position analysis: frame {search_window[best_frame].frame_index}")
        
        # Calculate confidence based on the score
        confidence = min(1.0, best_score)
        
        return SwingEvent(
            phase=SwingPhase.TOP,
            frame_index=search_window[best_frame].frame_index,
            timestamp=search_window[best_frame].timestamp,
            confidence=confidence,
            detection_method="biomechanical_top",
            features={
                "shoulder_hip_angle": search_window[best_frame].shoulder_hip_angle,
                "arm_extension": search_window[best_frame].arm_extension,
                "spine_angle": search_window[best_frame].spine_angle,
                "top_score": best_score
            }
        )
    
    def _calculate_top_of_backswing_score(self, pos: BodyPosition, search_window: List[BodyPosition], current_idx: int) -> float:
        """Calculate how well this position matches top of backswing criteria using multiple indicators"""
        score = 0.0
        
        # Indicator 1: Look for the transition point where shoulder rotation starts decreasing (40% weight)
        if current_idx > 0 and current_idx < len(search_window) - 1:
            prev_rotation = abs(search_window[current_idx - 1].shoulder_hip_angle)
            curr_rotation = abs(pos.shoulder_hip_angle)
            next_rotation = abs(search_window[current_idx + 1].shoulder_hip_angle)
            
            # Check if this is a local maximum (peak of rotation)
            if curr_rotation > prev_rotation and curr_rotation > next_rotation:
                # This is a peak - calculate how significant it is
                peak_score = min(1.0, curr_rotation / 30.0)  # Normalize to 30 degrees
                score += peak_score * 0.4
        
        # Indicator 2: Arm extension should be reasonable (30% weight)
        if 0.08 <= pos.arm_extension <= 0.18:
            extension_score = 1.0 - abs(pos.arm_extension - 0.13) / 0.05
            score += extension_score * 0.3
        
        # Indicator 3: Look for when the golfer's body is most rotated (30% weight)
        # This is the frame where the shoulders are most turned relative to hips
        if abs(pos.shoulder_hip_angle) > 10:  # Must have some rotation
            rotation_score = min(1.0, abs(pos.shoulder_hip_angle) / 25.0)
            score += rotation_score * 0.3
        
        return score
    
    def _find_top_by_wrist_height(self, search_window: List[BodyPosition]) -> Optional[int]:
        """Find top of backswing by looking for the frame just after the maximum peak"""
        if len(search_window) < 5:
            return None
        
        # Find the maximum angle and then look for the frame just after it
        # This represents the true top of backswing - just after the peak
        
        max_angle = 0.0
        max_frame = None
        
        # First, find the maximum angle and its frame
        for i, pos in enumerate(search_window):
            angle = abs(pos.shoulder_hip_angle)
            if angle > max_angle:
                max_angle = angle
                max_frame = i
        
        # Use adaptive threshold based on the maximum angle
        # For higher angles, use a larger threshold
        if max_angle > 30:
            threshold = max_angle * 0.85  # 15% drop for high angles
        elif max_angle > 15:
            threshold = max_angle * 0.90  # 10% drop for medium angles
        else:
            threshold = max_angle - 1.5   # Fixed 1.5° drop for low angles
        
        # Look for the frame just after the maximum
        best_frame = max_frame
        
        # Check if there's a frame 1-4 frames after the max that's still very close
        for offset in [1, 2, 3, 4]:
            if max_frame + offset < len(search_window):
                next_pos = search_window[max_frame + offset]
                next_angle = abs(next_pos.shoulder_hip_angle)
                
                # If the next frame is within the adaptive threshold, use it
                if next_angle >= threshold:
                    best_frame = max_frame + offset
                else:
                    break
        
        return best_frame
    
    def _calculate_upper_body_height_score(self, pos: BodyPosition) -> float:
        """Calculate a composite score for upper body height using multiple landmarks"""
        # For now, let's focus on finding the highest point using the most reliable landmarks
        # We'll use a simpler approach that prioritizes the highest visible landmark
        
        max_y = 0.0
        
        # Check wrists
        wrist_y = self._get_avg_wrist_y(pos)
        if wrist_y is not None:
            max_y = max(max_y, wrist_y)
        
        # Check elbows
        elbow_y = self._get_avg_elbow_y(pos)
        if elbow_y is not None:
            max_y = max(max_y, elbow_y)
        
        # Check head
        head_y = self._get_head_y(pos)
        if head_y is not None:
            max_y = max(max_y, head_y)
        
        return max_y
    
    def _get_avg_wrist_y(self, pos: BodyPosition) -> Optional[float]:
        """Get average wrist Y position for a body position"""
        left_wrist_y = pos.left_wrist[1] if pos.left_wrist != (0.0, 0.0) else None
        right_wrist_y = pos.right_wrist[1] if pos.right_wrist != (0.0, 0.0) else None
        
        if left_wrist_y is not None and right_wrist_y is not None:
            return (left_wrist_y + right_wrist_y) / 2
        return None
    
    def _get_avg_hand_y(self, pos: BodyPosition) -> Optional[float]:
        """Get average hand Y position for a body position (using wrists as hand proxy)"""
        # Since we don't have hand landmarks, use wrists as a proxy for hands
        return self._get_avg_wrist_y(pos)
    
    def _get_avg_elbow_y(self, pos: BodyPosition) -> Optional[float]:
        """Get average elbow Y position for a body position"""
        left_elbow_y = pos.left_elbow[1] if pos.left_elbow != (0.0, 0.0) else None
        right_elbow_y = pos.right_elbow[1] if pos.right_elbow != (0.0, 0.0) else None
        
        if left_elbow_y is not None and right_elbow_y is not None:
            return (left_elbow_y + right_elbow_y) / 2
        return None
    
    def _get_head_y(self, pos: BodyPosition) -> Optional[float]:
        """Get head Y position for a body position"""
        return pos.nose[1] if pos.nose != (0.0, 0.0) else None
    
    def _find_top_by_shoulder_position(self, search_window: List[BodyPosition]) -> Optional[int]:
        """Find top of backswing by looking for when shoulders are most turned"""
        if len(search_window) < 3:
            return None
        
        # Look for the frame where the shoulders are most turned relative to the camera
        # This is when the golfer's back is most visible to the camera
        best_frame = None
        best_shoulder_turn_score = 0.0
        
        for i, pos in enumerate(search_window):
            # Calculate how turned the shoulders are by looking at the separation
            # between left and right shoulders
            left_shoulder = pos.left_shoulder
            right_shoulder = pos.right_shoulder
            
            if left_shoulder != (0.0, 0.0) and right_shoulder != (0.0, 0.0):
                # Calculate shoulder separation (how turned the body is)
                shoulder_separation = abs(left_shoulder[0] - right_shoulder[0])
                
                # At the top of backswing, shoulders should be well separated
                # (golfer is turned away from camera)
                if shoulder_separation > 0.1:  # Minimum separation threshold
                    # Also check that the shoulders are at a reasonable height
                    avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
                    
                    # Shoulders should be at a reasonable height (not too high or low)
                    if 0.2 <= avg_shoulder_y <= 0.6:
                        # Calculate turn score based on separation and height
                        separation_score = min(1.0, shoulder_separation / 0.3)
                        height_score = 1.0 - abs(avg_shoulder_y - 0.4) / 0.2
                        turn_score = (separation_score * 0.7) + (height_score * 0.3)
                        
                        if turn_score > best_shoulder_turn_score:
                            best_shoulder_turn_score = turn_score
                            best_frame = i
        
        return best_frame
    
    def _calculate_top_score(self, pos: BodyPosition) -> float:
        """Calculate how well this position matches top of backswing criteria"""
        score = 0.0
        
        # Criterion 1: Significant shoulder turn (50% weight)
        if pos.shoulder_hip_angle >= self.TOP_SHOULDER_HIP_ANGLE_MIN:
            turn_score = min(1.0, pos.shoulder_hip_angle / 90.0)
            score += turn_score * 0.5
        
        # Criterion 2: Arms should be extended (30% weight)
        if pos.arm_extension >= self.TOP_ARM_EXTENSION_MIN:
            extension_score = min(1.0, pos.arm_extension / 0.4)
            score += extension_score * 0.3
        
        # Criterion 3: Maintain spine angle (20% weight)
        if self.TOP_SPINE_ANGLE_RANGE[0] <= pos.spine_angle <= self.TOP_SPINE_ANGLE_RANGE[1]:
            spine_score = 1.0 - abs(pos.spine_angle) / 10.0
            score += spine_score * 0.2
        
        return score
    
    def _detect_impact_biomechanical(self, body_positions: List[BodyPosition], setup_event: SwingEvent, top_event: SwingEvent) -> SwingEvent:
        """Detect impact position based on biomechanical criteria"""
        logger.info("Detecting impact position biomechanically")
        
        # Look between top and end of swing
        start_idx = top_event.frame_index
        search_window = [pos for pos in body_positions if pos.frame_index > start_idx]
        
        if not search_window:
            return self._create_default_event(SwingPhase.IMPACT, start_idx)
        
        best_impact_score = 0.0
        best_frame = 0
        
        for i, pos in enumerate(search_window):
            impact_score = self._calculate_impact_score(pos, setup_event)
            
            if impact_score > best_impact_score:
                best_impact_score = impact_score
                best_frame = i
        
        confidence = min(1.0, best_impact_score)
        
        return SwingEvent(
            phase=SwingPhase.IMPACT,
            frame_index=search_window[best_frame].frame_index,
            timestamp=search_window[best_frame].timestamp,
            confidence=confidence,
            detection_method="biomechanical_impact",
            features={
                "shoulder_hip_angle": search_window[best_frame].shoulder_hip_angle,
                "arm_extension": search_window[best_frame].arm_extension,
                "weight_balance": search_window[best_frame].weight_balance,
                "impact_score": best_impact_score
            }
        )
    
    def _calculate_impact_score(self, pos: BodyPosition, setup_event: SwingEvent) -> float:
        """Calculate how well this position matches impact criteria"""
        score = 0.0
        
        # Criterion 1: Shoulders returning to square (40% weight)
        if self.IMPACT_SHOULDER_HIP_ANGLE_RANGE[0] <= pos.shoulder_hip_angle <= self.IMPACT_SHOULDER_HIP_ANGLE_RANGE[1]:
            rotation_score = 1.0 - abs(pos.shoulder_hip_angle) / 30.0
            score += rotation_score * 0.4
        
        # Criterion 2: Arms in impact position (30% weight)
        if self.IMPACT_ARM_EXTENSION_RANGE[0] <= pos.arm_extension <= self.IMPACT_ARM_EXTENSION_RANGE[1]:
            extension_score = 1.0 - abs(pos.arm_extension - 0.30) / 0.1
            score += extension_score * 0.3
        
        # Criterion 3: Weight transfer happening (30% weight)
        if self.IMPACT_WEIGHT_BALANCE_RANGE[0] <= pos.weight_balance <= self.IMPACT_WEIGHT_BALANCE_RANGE[1]:
            weight_score = 1.0 - abs(pos.weight_balance) / 20.0
            score += weight_score * 0.3
        
        return score
    
    def _detect_follow_through_biomechanical(self, body_positions: List[BodyPosition], impact_event: SwingEvent) -> SwingEvent:
        """Detect follow-through position based on biomechanical criteria"""
        logger.info("Detecting follow-through position biomechanically")
        
        # Look after impact
        start_idx = impact_event.frame_index
        search_window = [pos for pos in body_positions if pos.frame_index > start_idx]
        
        if not search_window:
            return self._create_default_event(SwingPhase.FOLLOW_THROUGH, start_idx)
        
        best_follow_score = 0.0
        best_frame = 0
        
        for i, pos in enumerate(search_window):
            follow_score = self._calculate_follow_through_score(pos)
            
            if follow_score > best_follow_score:
                best_follow_score = follow_score
                best_frame = i
        
        confidence = min(1.0, best_follow_score)
        
        return SwingEvent(
            phase=SwingPhase.FOLLOW_THROUGH,
            frame_index=search_window[best_frame].frame_index,
            timestamp=search_window[best_frame].timestamp,
            confidence=confidence,
            detection_method="biomechanical_follow_through",
            features={
                "shoulder_hip_angle": search_window[best_frame].shoulder_hip_angle,
                "arm_extension": search_window[best_frame].arm_extension,
                "follow_through_score": best_follow_score
            }
        )
    
    def _calculate_follow_through_score(self, pos: BodyPosition) -> float:
        """Calculate how well this position matches follow-through criteria"""
        score = 0.0
        
        # Criterion 1: Full shoulder rotation (60% weight)
        if pos.shoulder_hip_angle >= self.FOLLOW_THROUGH_SHOULDER_HIP_ANGLE_MIN:
            turn_score = min(1.0, pos.shoulder_hip_angle / 120.0)
            score += turn_score * 0.6
        
        # Criterion 2: Arms extended behind body (40% weight)
        if pos.arm_extension >= self.FOLLOW_THROUGH_ARM_EXTENSION_MIN:
            extension_score = min(1.0, pos.arm_extension / 0.5)
            score += extension_score * 0.4
        
        return score
    
    def _validate_sequence(self, events: Dict[str, SwingEvent], body_positions: List[BodyPosition]) -> Dict[str, SwingEvent]:
        """Validate that events occur in correct sequence"""
        # Check sequence order
        if (events["setup"].frame_index >= events["top_backswing"].frame_index or
            events["top_backswing"].frame_index >= events["impact"].frame_index or
            events["impact"].frame_index >= events["follow_through"].frame_index):
            
            logger.warning("Invalid event sequence detected, applying corrections")
            events = self._correct_sequence(events, body_positions)
        
        return events
    
    def _correct_sequence(self, events: Dict[str, SwingEvent], body_positions: List[BodyPosition]) -> Dict[str, SwingEvent]:
        """Correct invalid event sequence"""
        total_frames = len(body_positions)
        
        # Ensure proper ordering with reasonable spacing
        setup_frame = min(events["setup"].frame_index, total_frames // 4)
        top_frame = max(setup_frame + 5, min(events["top_backswing"].frame_index, total_frames // 2))
        impact_frame = max(top_frame + 5, min(events["impact"].frame_index, total_frames * 3 // 4))
        follow_through_frame = max(impact_frame + 5, min(events["follow_through"].frame_index, total_frames - 1))
        
        # Update events
        events["setup"].frame_index = setup_frame
        events["top_backswing"].frame_index = top_frame
        events["impact"].frame_index = impact_frame
        events["follow_through"].frame_index = follow_through_frame
        
        return events
    
    def _create_default_event(self, phase: SwingPhase, frame_index: int) -> SwingEvent:
        """Create a default event with low confidence"""
        return SwingEvent(
            phase=phase,
            frame_index=frame_index,
            timestamp=0.0,
            confidence=0.1,
            detection_method="fallback",
            features={}
        )
    
    def _create_empty_events(self) -> Dict[str, SwingEvent]:
        """Create empty events dictionary"""
        return {
            "setup": self._create_default_event(SwingPhase.SETUP, 0),
            "top_backswing": self._create_default_event(SwingPhase.TOP, 0),
            "impact": self._create_default_event(SwingPhase.IMPACT, 0),
            "follow_through": self._create_default_event(SwingPhase.FOLLOW_THROUGH, 0)
        }

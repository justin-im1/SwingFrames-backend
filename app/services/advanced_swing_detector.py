#!/usr/bin/env python3
"""
Advanced Golf Swing Event Detector

A completely redesigned swing event detection system that uses robust heuristics
and biomechanical analysis to identify key golf swing phases with high accuracy.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import structlog
from dataclasses import dataclass
import math
from collections import deque

from .advanced_pose_analysis import FrameData, SwingEvent, SwingPhase

logger = structlog.get_logger()

@dataclass
class DetectionCriteria:
    """Criteria for detecting swing events"""
    min_confidence: float
    max_velocity_threshold: float
    min_velocity_threshold: float
    angle_threshold: float
    stability_threshold: float
    position_tolerance: float

class AdvancedSwingDetector:
    """Advanced swing event detector with robust heuristics"""
    
    def __init__(self):
        """Initialize the advanced swing detector"""
        
        # Detection criteria for each swing phase
        self.criteria = {
            SwingPhase.SETUP: DetectionCriteria(
                min_confidence=0.6,
                max_velocity_threshold=0.01,  # Very low movement
                min_velocity_threshold=0.0,
                angle_threshold=15.0,  # Minimal rotation
                stability_threshold=0.8,  # High stability
                position_tolerance=0.05
            ),
            SwingPhase.TOP: DetectionCriteria(
                min_confidence=0.7,
                max_velocity_threshold=0.02,  # Low movement at top
                min_velocity_threshold=0.0,
                angle_threshold=45.0,  # Significant rotation
                stability_threshold=0.6,
                position_tolerance=0.1
            ),
            SwingPhase.IMPACT: DetectionCriteria(
                min_confidence=0.8,
                max_velocity_threshold=0.1,  # High velocity
                min_velocity_threshold=0.05,  # Must have movement
                angle_threshold=20.0,  # Returning to square
                stability_threshold=0.4,
                position_tolerance=0.08
            ),
            SwingPhase.FOLLOW_THROUGH: DetectionCriteria(
                min_confidence=0.7,
                max_velocity_threshold=0.08,  # Moderate velocity
                min_velocity_threshold=0.02,
                angle_threshold=60.0,  # Full rotation
                stability_threshold=0.5,
                position_tolerance=0.1
            )
        }
        
        # Buffers for tracking trends
        self.velocity_buffer = deque(maxlen=10)
        self.angle_buffer = deque(maxlen=10)
        self.position_buffer = deque(maxlen=10)
        
        logger.info("AdvancedSwingDetector initialized")

    def detect_swing_events(self, frames_data: List[FrameData]) -> Dict[str, SwingEvent]:
        """
        Detect all swing events using advanced heuristics.
        
        Args:
            frames_data: List of processed frame data
            
        Returns:
            Dictionary mapping event names to SwingEvent objects
        """
        if not frames_data:
            return self._create_empty_events()
        
        logger.info("Starting advanced swing event detection", total_frames=len(frames_data))
        
        # Detect each event in sequence
        setup_event = self._detect_setup(frames_data)
        top_event = self._detect_top_backswing(frames_data, setup_event)
        impact_event = self._detect_impact(frames_data, setup_event, top_event)
        follow_through_event = self._detect_follow_through(frames_data, impact_event)
        
        events = {
            "setup": setup_event,
            "top_backswing": top_event,
            "impact": impact_event,
            "follow_through": follow_through_event
        }
        
        # Validate and correct sequence
        validated_events = self._validate_sequence(events, frames_data)
        
        logger.info(
            "Advanced swing events detected",
            setup=validated_events["setup"].frame_index,
            top_backswing=validated_events["top_backswing"].frame_index,
            impact=validated_events["impact"].frame_index,
            follow_through=validated_events["follow_through"].frame_index
        )
        
        return validated_events

    def _detect_setup(self, frames_data: List[FrameData]) -> SwingEvent:
        """Detect setup position using multiple criteria with improved logic"""
        logger.info("Detecting setup position")
        
        # Look in first 40% of frames for setup, but skip very early frames
        setup_window = frames_data[max(5, len(frames_data) // 10):max(1, len(frames_data) // 2)]
        
        if not setup_window:
            # Fallback to first 30% if no suitable window
            setup_window = frames_data[:max(1, len(frames_data) // 3)]
        
        if not setup_window:
            return self._create_default_event(SwingPhase.SETUP, 0)
        
        best_score = 0.0
        best_frame = 0
        
        for i, frame in enumerate(setup_window):
            score = self._calculate_setup_score(frame, setup_window, i)
            
            if score > best_score:
                best_score = score
                best_frame = i
        
        confidence = min(1.0, best_score)
        
        return SwingEvent(
            phase=SwingPhase.SETUP,
            frame_index=setup_window[best_frame].frame_index,
            timestamp=setup_window[best_frame].timestamp,
            confidence=confidence,
            detection_method="advanced_setup",
            features={
                "hand_velocity": setup_window[best_frame].hand_velocity,
                "shoulder_hip_angle": setup_window[best_frame].shoulder_hip_angle,
                "movement_stability": setup_window[best_frame].movement_stability,
                "posture_stability": setup_window[best_frame].posture_stability,
                "setup_score": best_score
            },
            normalized_position=setup_window[best_frame].hand_center
        )

    def _calculate_setup_score(self, frame: FrameData, setup_window: List[FrameData], index: int) -> float:
        """Calculate setup score based on multiple criteria with improved logic"""
        score = 0.0
        criteria = self.criteria[SwingPhase.SETUP]
        
        # Criterion 1: Low hand velocity (35% weight)
        if frame.hand_velocity <= criteria.max_velocity_threshold:
            velocity_score = 1.0 - (frame.hand_velocity / criteria.max_velocity_threshold)
            score += velocity_score * 0.35
        
        # Criterion 2: Shoulder-hip rotation (30% weight) - more lenient for setup
        # Accept moderate rotation as setup (golfer might be addressing ball)
        if abs(frame.shoulder_hip_angle) <= 45.0:  # More lenient threshold
            rotation_score = 1.0 - (abs(frame.shoulder_hip_angle) / 45.0)
            score += rotation_score * 0.3
        
        # Criterion 3: High movement stability (20% weight)
        if frame.movement_stability >= criteria.stability_threshold:
            stability_score = frame.movement_stability
            score += stability_score * 0.2
        
        # Criterion 4: High posture stability (15% weight)
        if frame.posture_stability >= criteria.stability_threshold:
            posture_score = frame.posture_stability
            score += posture_score * 0.15
        
        return score

    def _detect_top_backswing(self, frames_data: List[FrameData], setup_event: SwingEvent) -> SwingEvent:
        """Detect top of backswing using velocity and position analysis"""
        logger.info("Detecting top of backswing")
        
        # Look after setup for top of backswing
        start_idx = setup_event.frame_index
        search_window = [frame for frame in frames_data if frame.frame_index > start_idx]
        
        # Limit search to first 50% of remaining frames
        max_search_frames = max(10, int(len(search_window) * 0.5))
        search_window = search_window[:max_search_frames]
        
        if not search_window:
            return self._create_default_event(SwingPhase.TOP, start_idx)
        
        # Use multiple detection methods
        best_score = 0.0
        best_frame = 0
        
        for i, frame in enumerate(search_window):
            score = self._calculate_top_score(frame, search_window, i)
            
            if score > best_score:
                best_score = score
                best_frame = i
        
        # Try velocity-based detection as backup
        velocity_based_frame = self._find_top_by_velocity(search_window)
        if velocity_based_frame is not None:
            velocity_score = 0.8  # High confidence for velocity-based detection
            if velocity_score > best_score:
                best_frame = velocity_based_frame
                best_score = velocity_score
        
        confidence = min(1.0, best_score)
        
        return SwingEvent(
            phase=SwingPhase.TOP,
            frame_index=search_window[best_frame].frame_index,
            timestamp=search_window[best_frame].timestamp,
            confidence=confidence,
            detection_method="advanced_top",
            features={
                "hand_velocity": search_window[best_frame].hand_velocity,
                "shoulder_hip_angle": search_window[best_frame].shoulder_hip_angle,
                "arm_angle": search_window[best_frame].arm_angle,
                "top_score": best_score
            },
            normalized_position=search_window[best_frame].hand_center
        )

    def _calculate_top_score(self, frame: FrameData, search_window: List[FrameData], index: int) -> float:
        """Calculate top of backswing score"""
        score = 0.0
        criteria = self.criteria[SwingPhase.TOP]
        
        # Criterion 1: Look for velocity transition (40% weight)
        if index > 0 and index < len(search_window) - 1:
            prev_velocity = search_window[index - 1].hand_velocity
            curr_velocity = frame.hand_velocity
            next_velocity = search_window[index + 1].hand_velocity
            
            # Check if this is a local minimum in velocity (transition point)
            if curr_velocity < prev_velocity and curr_velocity < next_velocity:
                transition_score = 1.0 - (curr_velocity / max(prev_velocity, next_velocity, 0.01))
                score += transition_score * 0.4
        
        # Criterion 2: Significant shoulder rotation (30% weight)
        if abs(frame.shoulder_hip_angle) >= criteria.angle_threshold:
            rotation_score = min(1.0, abs(frame.shoulder_hip_angle) / 90.0)
            score += rotation_score * 0.3
        
        # Criterion 3: Hand position analysis (20% weight)
        # At top of backswing, hands should be high and away from body
        hand_y = frame.hand_center[1]
        shoulder_y = frame.shoulder_center[1]
        
        if hand_y < shoulder_y:  # Hands above shoulders
            position_score = (shoulder_y - hand_y) / 0.2  # Normalize
            score += min(1.0, position_score) * 0.2
        
        # Criterion 4: Arm angle (10% weight)
        if abs(frame.arm_angle) > 45:  # Arms should be extended
            arm_score = min(1.0, abs(frame.arm_angle) / 90.0)
            score += arm_score * 0.1
        
        return score

    def _find_top_by_velocity(self, search_window: List[FrameData]) -> Optional[int]:
        """Find top of backswing by analyzing velocity patterns"""
        if len(search_window) < 5:
            return None
        
        # Find the frame with minimum velocity after initial movement
        min_velocity = float('inf')
        min_frame = None
        
        for i, frame in enumerate(search_window):
            if frame.hand_velocity < min_velocity:
                min_velocity = frame.hand_velocity
                min_frame = i
        
        return min_frame

    def _detect_impact(self, frames_data: List[FrameData], setup_event: SwingEvent, top_event: SwingEvent) -> SwingEvent:
        """Detect pre-impact position - the frame just before impact where club approaches ball"""
        logger.info("Detecting pre-impact position (club approaching ball)")
        
        # Look between top and end of swing
        start_idx = top_event.frame_index
        search_window = [frame for frame in frames_data if frame.frame_index > start_idx]
        
        if not search_window:
            return self._create_default_event(SwingPhase.IMPACT, start_idx)
        
        best_score = 0.0
        best_frame = 0
        
        for i, frame in enumerate(search_window):
            score = self._calculate_pre_impact_score(frame, setup_event, search_window, i)
            
            if score > best_score:
                best_score = score
                best_frame = i
        
        # Disable velocity-based detection - rely only on scoring-based detection
        # This ensures we get frames 3-4 after top (mid-downswing) as requested
        
        confidence = min(1.0, best_score)
        
        return SwingEvent(
            phase=SwingPhase.IMPACT,
            frame_index=search_window[best_frame].frame_index,
            timestamp=search_window[best_frame].timestamp,
            confidence=confidence,
            detection_method="advanced_pre_impact",
            features={
                "hand_velocity": search_window[best_frame].hand_velocity,
                "hand_acceleration": search_window[best_frame].hand_acceleration,
                "shoulder_hip_angle": search_window[best_frame].shoulder_hip_angle,
                "pre_impact_score": best_score
            },
            normalized_position=search_window[best_frame].hand_center
        )

    def _calculate_pre_impact_score(self, frame: FrameData, setup_event: SwingEvent, search_window: List[FrameData], index: int) -> float:
        """Calculate pre-impact score - detect when hands are halfway down the downswing"""
        score = 0.0
        criteria = self.criteria[SwingPhase.IMPACT]
        
        # Get top position (first frame in search window)
        if not search_window:
            return 0.0
        
        top_frame = search_window[0]
        setup_hand = setup_event.normalized_position
        current_hand = frame.hand_center
        top_hand = top_frame.hand_center
        
        # Criterion 1: Simple early frame detection (60% weight)
        # Just look for the first few frames after top with some movement
        if index <= 3:  # First 4 frames after top (index 0, 1, 2, 3)
            if index == 1 or index == 2:  # Frames 2-3 after top
                position_score = 1.0  # Perfect early downswing
            elif index == 0:
                position_score = 0.8  # First frame might be too early
            elif index == 3:
                position_score = 0.7  # Fourth frame might be too late
            else:
                position_score = 0.5
        else:
            position_score = 0.0  # Too late
        
        score += position_score * 0.6
        
        # Criterion 2: Moderate velocity (25% weight)
        # Hands should be moving but not at maximum speed yet
        if frame.hand_velocity >= criteria.min_velocity_threshold * 0.3:
            max_velocity_in_window = max(f.hand_velocity for f in search_window)
            if max_velocity_in_window > 0:
                velocity_ratio = frame.hand_velocity / max_velocity_in_window
                # Prefer moderate velocity (30-70% of max)
                if 0.3 <= velocity_ratio <= 0.7:
                    velocity_score = 1.0
                elif velocity_ratio < 0.3:
                    velocity_score = velocity_ratio * 0.8  # Too slow
                else:
                    velocity_score = 0.7  # Too fast
                score += velocity_score * 0.25
        
        # Criterion 3: Shoulders partially rotated (15% weight)
        # Shoulders should be starting to square up but not fully there
        if abs(frame.shoulder_hip_angle) >= criteria.angle_threshold * 0.3:
            rotation_score = min(1.0, abs(frame.shoulder_hip_angle) / (criteria.angle_threshold * 1.5))
            score += rotation_score * 0.15
        
        return score

    def _find_pre_impact_by_velocity(self, search_window: List[FrameData]) -> Optional[int]:
        """Find pre-impact by detecting early velocity frame (2-4 frames after top)"""
        if len(search_window) < 3:
            return None
        
        # Look for frames in the first 4 positions (early in downswing)
        best_pre_impact_frame = None
        best_score = 0.0
        
        for i in range(min(5, len(search_window))):
            frame = search_window[i]
            
            # Look for moderate velocity in mid-downswing frames
            if frame.hand_velocity > 0.01:  # Some movement
                # Strongly prefer frames 3-4 after top (index 2-3) - halfway down downswing
                if i == 2 or i == 3:
                    score = 1.0  # Perfect - 3-4 frames after top (mid-downswing)
                elif i == 1:
                    score = 0.6  # Second frame is too early
                elif i == 4:
                    score = 0.5  # Fifth frame is too late
                elif i == 0:
                    score = 0.3  # First frame is way too early
                else:
                    score = 0.2  # Any other frame is not ideal
                
                if score > best_score:
                    best_score = score
                    best_pre_impact_frame = i
        
        return best_pre_impact_frame

    def _detect_follow_through(self, frames_data: List[FrameData], impact_event: SwingEvent) -> SwingEvent:
        """Detect follow-through position"""
        logger.info("Detecting follow-through position")
        
        # Look after impact
        start_idx = impact_event.frame_index
        search_window = [frame for frame in frames_data if frame.frame_index > start_idx]
        
        if not search_window:
            return self._create_default_event(SwingPhase.FOLLOW_THROUGH, start_idx)
        
        best_score = 0.0
        best_frame = 0
        
        for i, frame in enumerate(search_window):
            score = self._calculate_follow_through_score(frame, search_window, i)
            
            if score > best_score:
                best_score = score
                best_frame = i
        
        confidence = min(1.0, best_score)
        
        return SwingEvent(
            phase=SwingPhase.FOLLOW_THROUGH,
            frame_index=search_window[best_frame].frame_index,
            timestamp=search_window[best_frame].timestamp,
            confidence=confidence,
            detection_method="advanced_follow_through",
            features={
                "hand_velocity": search_window[best_frame].hand_velocity,
                "shoulder_hip_angle": search_window[best_frame].shoulder_hip_angle,
                "arm_angle": search_window[best_frame].arm_angle,
                "follow_through_score": best_score
            },
            normalized_position=search_window[best_frame].hand_center
        )

    def _calculate_follow_through_score(self, frame: FrameData, search_window: List[FrameData], index: int) -> float:
        """Calculate follow-through score with improved logic"""
        score = 0.0
        criteria = self.criteria[SwingPhase.FOLLOW_THROUGH]
        
        # Criterion 1: Shoulder rotation (40% weight) - more lenient
        # Accept moderate rotation as follow-through
        if abs(frame.shoulder_hip_angle) >= 20.0:  # Lower threshold
            rotation_score = min(1.0, abs(frame.shoulder_hip_angle) / 60.0)  # More lenient scaling
            score += rotation_score * 0.4
        
        # Criterion 2: Velocity analysis (30% weight)
        # Look for decreasing velocity after impact
        if frame.hand_velocity >= criteria.min_velocity_threshold:
            velocity_score = min(1.0, frame.hand_velocity / criteria.max_velocity_threshold)
            score += velocity_score * 0.3
        
        # Criterion 3: Position relative to impact (20% weight)
        # Look for hands moving away from impact position
        if index > 0:
            prev_frame = search_window[index - 1]
            hand_movement = abs(frame.hand_center[0] - prev_frame.hand_center[0]) + abs(frame.hand_center[1] - prev_frame.hand_center[1])
            if hand_movement > 0.01:  # Some movement
                position_score = min(1.0, hand_movement / 0.05)
                score += position_score * 0.2
        
        # Criterion 4: Stability (10% weight)
        # Follow-through should have some stability
        if frame.movement_stability >= 0.8:
            stability_score = frame.movement_stability
            score += stability_score * 0.1
        
        return score

    def _validate_sequence(self, events: Dict[str, SwingEvent], frames_data: List[FrameData]) -> Dict[str, SwingEvent]:
        """Validate that events occur in correct sequence"""
        # Check sequence order
        if (events["setup"].frame_index >= events["top_backswing"].frame_index or
            events["top_backswing"].frame_index >= events["impact"].frame_index or
            events["impact"].frame_index >= events["follow_through"].frame_index):
            
            logger.warning("Invalid event sequence detected, applying corrections")
            events = self._correct_sequence(events, frames_data)
        
        return events

    def _correct_sequence(self, events: Dict[str, SwingEvent], frames_data: List[FrameData]) -> Dict[str, SwingEvent]:
        """Correct invalid event sequence"""
        total_frames = len(frames_data)
        
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
            features={},
            normalized_position=(0.0, 0.0)
        )

    def _create_empty_events(self) -> Dict[str, SwingEvent]:
        """Create empty events dictionary"""
        return {
            "setup": self._create_default_event(SwingPhase.SETUP, 0),
            "top_backswing": self._create_default_event(SwingPhase.TOP, 0),
            "impact": self._create_default_event(SwingPhase.IMPACT, 0),
            "follow_through": self._create_default_event(SwingPhase.FOLLOW_THROUGH, 0)
        }

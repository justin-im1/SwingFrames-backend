#!/usr/bin/env python3
"""
Smart Pose Analyzer - Main Integration Module

Integrates the smart feature extraction and swing detection systems
for improved golf swing analysis from behind-the-golfer videos.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import structlog
import json
from pathlib import Path

from .pose_analysis import PoseAnalyzer
from .smart_pose_analysis import SwingFeatureExtractor, SwingFeatures
from .smart_swing_detector import SmartSwingDetector, SwingEvent

logger = structlog.get_logger()

class SmartPoseAnalyzer(PoseAnalyzer):
    """Enhanced pose analyzer with smart detection capabilities"""
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 target_fps: int = 60,
                 use_smart_detection: bool = True):
        """
        Initialize the smart pose analyzer.
        
        Args:
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
            target_fps: Target frames per second for processing
            use_smart_detection: Whether to use smart detection methods
        """
        super().__init__(min_detection_confidence, min_tracking_confidence, target_fps)
        
        self.use_smart_detection = use_smart_detection
        
        if self.use_smart_detection:
            self.feature_extractor = SwingFeatureExtractor()
            self.swing_detector = SmartSwingDetector()
        
        logger.info(
            "SmartPoseAnalyzer initialized",
            use_smart_detection=use_smart_detection,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            target_fps=target_fps
        )
    
    def detect_swing_events(self, landmarks_data: List[Dict]) -> Dict:
        """
        Detect key swing events using smart detection methods.
        
        Args:
            landmarks_data: List of frame data with landmarks
            
        Returns:
            Dictionary with detected event frame indices and confidence scores
        """
        if not landmarks_data:
            raise ValueError("No landmarks data provided")
        
        logger.info("Starting smart swing event detection", total_frames=len(landmarks_data))
        
        if not self.use_smart_detection:
            # Fallback to original detection method
            return super().detect_swing_events(landmarks_data)
        
        try:
            # Extract comprehensive features
            features = self.feature_extractor.extract_features(landmarks_data)
            
            if not features:
                logger.warning("No features extracted, falling back to original detection")
                return super().detect_swing_events(landmarks_data)
            
            # Detect swing events using smart methods
            swing_events = self.swing_detector.detect_swing_events(features)
            
            # Convert to original format for compatibility
            events_dict = {
                "setup": swing_events["setup"].frame_index,
                "top_backswing": swing_events["top_backswing"].frame_index,
                "impact": swing_events["impact"].frame_index,
                "follow_through": swing_events["follow_through"].frame_index
            }
            
            # Add confidence scores
            events_dict["confidence_scores"] = {
                "setup": swing_events["setup"].confidence,
                "top_backswing": swing_events["top_backswing"].confidence,
                "impact": swing_events["impact"].confidence,
                "follow_through": swing_events["follow_through"].confidence
            }
            
            # Add detection methods used
            events_dict["detection_methods"] = {
                "setup": swing_events["setup"].detection_method,
                "top_backswing": swing_events["top_backswing"].detection_method,
                "impact": swing_events["impact"].detection_method,
                "follow_through": swing_events["follow_through"].detection_method
            }
            
            logger.info(
                "Smart swing events detected",
                setup=events_dict["setup"],
                top_backswing=events_dict["top_backswing"],
                impact=events_dict["impact"],
                follow_through=events_dict["follow_through"],
                avg_confidence=np.mean(list(events_dict["confidence_scores"].values()))
            )
            
            return events_dict
            
        except Exception as e:
            logger.error(
                "Smart detection failed, falling back to original method",
                error=str(e)
            )
            return super().detect_swing_events(landmarks_data)
    
    def analyze_swing_smart(self, video_path: str) -> Dict:
        """
        Complete smart swing analysis pipeline.
        
        Args:
            video_path: Path to the input video file
            
        Returns:
            Complete analysis results with features and smart events
        """
        logger.info("Starting complete smart swing analysis", video_path=video_path)
        
        # Extract landmarks using parent class
        landmarks_data = self.extract_landmarks(video_path)
        
        if not self.use_smart_detection:
            # Fallback to original analysis
            return super().analyze_swing(video_path)
        
        try:
            # Extract comprehensive features
            features = self.feature_extractor.extract_features(landmarks_data)
            
            # Detect swing events using smart methods
            swing_events = self.swing_detector.detect_swing_events(features)
            
            # Compile comprehensive results
            results = {
                "video_path": str(video_path),
                "total_frames": len(landmarks_data),
                "swing_events": {
                    "setup": swing_events["setup"].frame_index,
                    "top_backswing": swing_events["top_backswing"].frame_index,
                    "impact": swing_events["impact"].frame_index,
                    "follow_through": swing_events["follow_through"].frame_index
                },
                "confidence_scores": {
                    "setup": swing_events["setup"].confidence,
                    "top_backswing": swing_events["top_backswing"].confidence,
                    "impact": swing_events["impact"].confidence,
                    "follow_through": swing_events["follow_through"].confidence
                },
                "detection_methods": {
                    "setup": swing_events["setup"].detection_method,
                    "top_backswing": swing_events["top_backswing"].detection_method,
                    "impact": swing_events["impact"].detection_method,
                    "follow_through": swing_events["follow_through"].detection_method
                },
                "landmarks_data": landmarks_data,
                "swing_features": self._serialize_features(features),
                "analysis_metadata": {
                    "smart_detection_enabled": True,
                    "feature_extraction_successful": len(features) > 0,
                    "average_feature_confidence": np.mean([f.feature_confidence for f in features]) if features else 0.0,
                    "average_landmark_confidence": np.mean([f.landmark_confidence for f in features]) if features else 0.0
                }
            }
            
            logger.info(
                "Smart swing analysis completed",
                total_frames=len(landmarks_data),
                features_extracted=len(features),
                events=results["swing_events"],
                avg_confidence=np.mean(list(results["confidence_scores"].values()))
            )
            
            return results
            
        except Exception as e:
            logger.error(
                "Smart analysis failed, falling back to original method",
                error=str(e)
            )
            return super().analyze_swing(video_path)
    
    def get_swing_features(self, landmarks_data: List[Dict]) -> List[SwingFeatures]:
        """
        Extract swing features from landmarks data.
        
        Args:
            landmarks_data: List of frame data with landmarks
            
        Returns:
            List of swing features
        """
        if not self.use_smart_detection:
            raise ValueError("Smart detection must be enabled to extract features")
        
        return self.feature_extractor.extract_features(landmarks_data)
    
    def get_detailed_swing_events(self, landmarks_data: List[Dict]) -> Dict[str, SwingEvent]:
        """
        Get detailed swing events with full metadata.
        
        Args:
            landmarks_data: List of frame data with landmarks
            
        Returns:
            Dictionary of detailed swing events
        """
        if not self.use_smart_detection:
            raise ValueError("Smart detection must be enabled to get detailed events")
        
        features = self.feature_extractor.extract_features(landmarks_data)
        return self.swing_detector.detect_swing_events(features)
    
    def _serialize_features(self, features: List[SwingFeatures]) -> List[Dict]:
        """Serialize features for JSON storage"""
        serialized = []
        
        for feature in features:
            feature_dict = {
                "frame_index": feature.frame_index,
                "timestamp": feature.timestamp,
                "landmarks": feature.landmarks,
                "angles": {
                    "shoulder_turn_angle": feature.shoulder_turn_angle,
                    "spine_angle": feature.spine_angle,
                    "arm_plane_angle": feature.arm_plane_angle,
                    "hip_sway": feature.hip_sway,
                    "knee_flexion": feature.knee_flexion
                },
                "velocities": {
                    "hand_velocity": feature.hand_velocity,
                    "shoulder_velocity": feature.shoulder_velocity,
                    "hip_velocity": feature.hip_velocity
                },
                "accelerations": {
                    "hand_acceleration": feature.hand_acceleration
                },
                "distances": {
                    "shoulder_hip_separation": feature.shoulder_hip_separation,
                    "weight_distribution": feature.weight_distribution,
                    "club_angle_estimate": feature.club_angle_estimate
                },
                "stability": {
                    "movement_stability": feature.movement_stability,
                    "posture_stability": feature.posture_stability
                },
                "confidence": {
                    "landmark_confidence": feature.landmark_confidence,
                    "feature_confidence": feature.feature_confidence
                }
            }
            serialized.append(feature_dict)
        
        return serialized
    
    def save_smart_results(self, results: Dict, output_path: str):
        """Save smart analysis results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("Smart results saved", output_path=str(output_path))
    
    def compare_detection_methods(self, video_path: str) -> Dict:
        """
        Compare smart detection with original detection methods.
        
        Args:
            video_path: Path to the input video file
            
        Returns:
            Comparison results
        """
        logger.info("Comparing detection methods", video_path=video_path)
        
        # Extract landmarks once
        landmarks_data = self.extract_landmarks(video_path)
        
        # Original detection
        original_events = super().detect_swing_events(landmarks_data)
        
        # Smart detection
        smart_events = self.detect_swing_events(landmarks_data)
        
        # Compare results
        comparison = {
            "video_path": str(video_path),
            "total_frames": len(landmarks_data),
            "original_detection": original_events,
            "smart_detection": smart_events,
            "differences": {
                "setup": smart_events["setup"] - original_events["setup"],
                "top_backswing": smart_events["top_backswing"] - original_events["top_backswing"],
                "impact": smart_events["impact"] - original_events["impact"],
                "follow_through": smart_events["follow_through"] - original_events["follow_through"]
            },
            "confidence_available": "confidence_scores" in smart_events,
            "smart_detection_enabled": self.use_smart_detection
        }
        
        if "confidence_scores" in smart_events:
            comparison["average_confidence"] = np.mean(list(smart_events["confidence_scores"].values()))
        
        logger.info(
            "Detection method comparison completed",
            differences=comparison["differences"],
            avg_confidence=comparison.get("average_confidence", 0.0)
        )
        
        return comparison


# Convenience functions for direct use
def analyze_swing_smart(video_path: str, target_fps: int = 60) -> Dict:
    """
    Complete smart swing analysis pipeline.
    
    Args:
        video_path: Path to the input video file
        target_fps: Target frames per second for processing
        
    Returns:
        Complete analysis results with smart detection
    """
    analyzer = SmartPoseAnalyzer(target_fps=target_fps, use_smart_detection=True)
    return analyzer.analyze_swing_smart(video_path)


def detect_swing_events_smart(landmarks: List[Dict]) -> Dict:
    """
    Detect key swing events using smart detection methods.
    
    Args:
        landmarks: List of frame data with landmarks
        
    Returns:
        Dictionary with detected event frame indices and confidence scores
    """
    analyzer = SmartPoseAnalyzer(use_smart_detection=True)
    return analyzer.detect_swing_events(landmarks)


def compare_detection_methods(video_path: str, target_fps: int = 60) -> Dict:
    """
    Compare smart detection with original detection methods.
    
    Args:
        video_path: Path to the input video file
        target_fps: Target frames per second for processing
        
    Returns:
        Comparison results
    """
    analyzer = SmartPoseAnalyzer(target_fps=target_fps, use_smart_detection=True)
    return analyzer.compare_detection_methods(video_path)


if __name__ == "__main__":
    # Test the smart pose analysis module
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python smart_pose_analyzer.py <video_path> [output_path] [--compare]")
        print("Example: python smart_pose_analyzer.py test_swing.mp4 results.json --compare")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "smart_swing_analysis_results.json"
    compare_mode = "--compare" in sys.argv
    
    try:
        print(f"🎯 Analyzing swing video with smart detection: {video_path}")
        
        if compare_mode:
            # Compare detection methods
            print("📊 Comparing detection methods...")
            comparison = compare_detection_methods(video_path)
            
            print("\n🔍 Detection Comparison Results:")
            print(f"  Original vs Smart Detection:")
            for event, diff in comparison["differences"].items():
                print(f"    {event}: {diff:+d} frames")
            
            if comparison["confidence_available"]:
                print(f"  Average Smart Detection Confidence: {comparison['average_confidence']:.2f}")
            
            # Save comparison results
            analyzer = SmartPoseAnalyzer(use_smart_detection=True)
            analyzer.save_smart_results(comparison, output_path.replace('.json', '_comparison.json'))
            print(f"\n💾 Comparison results saved to: {output_path.replace('.json', '_comparison.json')}")
        
        else:
            # Run smart analysis
            results = analyze_swing_smart(video_path)
            
            # Print detected events
            events = results["swing_events"]
            confidence_scores = results["confidence_scores"]
            detection_methods = results["detection_methods"]
            
            print("\n🎯 Smart Detection Results:")
            for event_name in ["setup", "top_backswing", "impact", "follow_through"]:
                frame = events[event_name]
                confidence = confidence_scores[event_name]
                method = detection_methods[event_name]
                print(f"  {event_name.replace('_', ' ').title()}: Frame {frame} (Confidence: {confidence:.2f}, Method: {method})")
            
            print(f"\n📊 Analysis Summary:")
            print(f"  Total frames processed: {results['total_frames']}")
            print(f"  Features extracted: {len(results['swing_features'])}")
            print(f"  Average feature confidence: {results['analysis_metadata']['average_feature_confidence']:.2f}")
            print(f"  Average landmark confidence: {results['analysis_metadata']['average_landmark_confidence']:.2f}")
            
            # Save results
            analyzer = SmartPoseAnalyzer(use_smart_detection=True)
            analyzer.save_smart_results(results, output_path)
            print(f"\n💾 Smart results saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error during smart analysis: {e}")
        sys.exit(1)

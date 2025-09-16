#!/usr/bin/env python3
"""
Debug script for IMG_3845 analysis
"""

import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from services.advanced_golf_processor import AdvancedGolfProcessor
import json

def debug_img3845():
    """Debug the IMG_3845 processing"""
    
    print("🔍 Debugging IMG_3845 Processing")
    print("=" * 50)
    
    # Create processor with more detailed logging
    processor = AdvancedGolfProcessor(
        output_dir="debug_outputs",
        min_detection_confidence=0.6,  # Lower threshold for debugging
        min_tracking_confidence=0.6,
        target_fps=60
    )
    
    # Process the video
    result = processor.process_swing_video("IMG_3845.MOV")
    
    if result.success:
        print(f"✅ Processing completed")
        print(f"📊 Total frames: {result.total_frames}")
        print(f"📊 Processed frames: {len(result.frames_data)}")
        
        # Analyze frame-by-frame data
        print(f"\n🔍 Frame Analysis:")
        print(f"{'Frame':<6} {'Time':<8} {'Hand Vel':<10} {'Shoulder-Hip':<12} {'Confidence':<10}")
        print("-" * 60)
        
        for i, frame in enumerate(result.frames_data[::5]):  # Every 5th frame
            print(f"{frame.frame_index:<6} {frame.timestamp:<8.2f} {frame.hand_velocity:<10.3f} {frame.shoulder_hip_angle:<12.1f} {frame.overall_confidence:<10.3f}")
        
        # Analyze events
        print(f"\n🎯 Event Analysis:")
        for event_name, event in result.events.items():
            print(f"\n{event_name}:")
            print(f"  Frame: {event.frame_index}")
            print(f"  Confidence: {event.confidence:.3f}")
            print(f"  Method: {event.detection_method}")
            print(f"  Features: {event.features}")
            
            # Find the corresponding frame data
            frame_data = None
            for frame in result.frames_data:
                if frame.frame_index == event.frame_index:
                    frame_data = frame
                    break
            
            if frame_data:
                print(f"  Frame Data:")
                print(f"    Hand velocity: {frame_data.hand_velocity:.3f}")
                print(f"    Shoulder-hip angle: {frame_data.shoulder_hip_angle:.1f}°")
                print(f"    Movement stability: {frame_data.movement_stability:.3f}")
                print(f"    Posture stability: {frame_data.posture_stability:.3f}")
        
        # Check for potential issues
        print(f"\n⚠️  Potential Issues:")
        
        # Check if setup and top are too close
        setup_frame = result.events["setup"].frame_index
        top_frame = result.events["top_backswing"].frame_index
        if top_frame - setup_frame < 5:
            print(f"  - Setup and top backswing are too close: {top_frame - setup_frame} frames apart")
        
        # Check confidence scores
        for event_name, event in result.events.items():
            if event.confidence < 0.6:
                print(f"  - {event_name} has low confidence: {event.confidence:.3f}")
        
        # Check velocity patterns
        hand_velocities = [frame.hand_velocity for frame in result.frames_data]
        max_velocity = max(hand_velocities)
        if max_velocity < 0.05:
            print(f"  - Maximum hand velocity is very low: {max_velocity:.3f}")
        
        # Save detailed results
        with open("debug_outputs/img3845_debug_results.json", "w") as f:
            debug_data = {
                "video_info": {
                    "total_frames": result.total_frames,
                    "processed_frames": len(result.frames_data),
                    "fps": result.fps,
                    "duration": result.duration
                },
                "events": {
                    name: {
                        "frame_index": event.frame_index,
                        "timestamp": event.timestamp,
                        "confidence": event.confidence,
                        "detection_method": event.detection_method,
                        "features": event.features
                    }
                    for name, event in result.events.items()
                },
                "frame_data": [
                    {
                        "frame_index": frame.frame_index,
                        "timestamp": frame.timestamp,
                        "hand_velocity": frame.hand_velocity,
                        "shoulder_hip_angle": frame.shoulder_hip_angle,
                        "movement_stability": frame.movement_stability,
                        "overall_confidence": frame.overall_confidence
                    }
                    for frame in result.frames_data
                ]
            }
            json.dump(debug_data, f, indent=2)
        
        print(f"\n💾 Debug results saved to: debug_outputs/img3845_debug_results.json")
        
    else:
        print(f"❌ Processing failed: {result.error_message}")

if __name__ == "__main__":
    debug_img3845()

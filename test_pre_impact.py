#!/usr/bin/env python3
"""
Test script for the new pre-impact detection
"""

import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from services.advanced_golf_processor import process_golf_swing_video_advanced

def test_pre_impact_detection():
    """Test the new pre-impact detection logic"""
    
    print("🎯 Testing Pre-Impact Detection")
    print("=" * 50)
    
    # Test with both sample videos
    test_videos = ["IMG_3845.MOV", "IMG_6596.MOV"]
    
    for video_path in test_videos:
        if not Path(video_path).exists():
            print(f"❌ Test video not found: {video_path}")
            continue
            
        print(f"\n🎬 Testing: {video_path}")
        print("-" * 30)
        
        try:
            # Process with advanced system
            result = process_golf_swing_video_advanced(
                video_path=video_path,
                output_dir="test_pre_impact_outputs",
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7,
                target_fps=60
            )
            
            if result.success:
                impact_event = result.events["impact"]
                
                print(f"✅ Pre-Impact Detection Results:")
                print(f"   Frame: {impact_event.frame_index}")
                print(f"   Timestamp: {impact_event.timestamp:.2f}s")
                print(f"   Confidence: {impact_event.confidence:.3f}")
                print(f"   Method: {impact_event.detection_method}")
                print(f"   Hand Velocity: {impact_event.features['hand_velocity']:.4f}")
                print(f"   Hand Acceleration: {impact_event.features['hand_acceleration']:.4f}")
                print(f"   Shoulder-Hip Angle: {impact_event.features['shoulder_hip_angle']:.1f}°")
                print(f"   Pre-Impact Score: {impact_event.features['pre_impact_score']:.3f}")
                
                # Check if snapshot was generated
                snapshot_path = f"test_pre_impact_outputs/{Path(video_path).stem}_impact.jpg"
                if Path(snapshot_path).exists():
                    print(f"   ✅ Snapshot saved: {snapshot_path}")
                else:
                    print(f"   ❌ Snapshot missing: {snapshot_path}")
                    
            else:
                print(f"❌ Processing failed: {result.error_message}")
                
        except Exception as e:
            print(f"❌ Error processing {video_path}: {e}")
            import traceback
            traceback.print_exc()

def compare_with_old_system():
    """Compare pre-impact detection with old impact detection"""
    
    print(f"\n🔄 Comparing Pre-Impact vs Old Impact Detection")
    print("=" * 50)
    
    # Test with IMG_3845 (the problematic one)
    video_path = "IMG_3845.MOV"
    
    if not Path(video_path).exists():
        print(f"❌ Test video not found: {video_path}")
        return
    
    print(f"🎬 Comparing: {video_path}")
    
    try:
        # Test new pre-impact system
        print("\n📊 New Pre-Impact System:")
        result_new = process_golf_swing_video_advanced(
            video_path=video_path,
            output_dir="test_pre_impact_outputs",
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            target_fps=60
        )
        
        if result_new.success:
            impact_event = result_new.events["impact"]
            print(f"   Frame: {impact_event.frame_index}")
            print(f"   Confidence: {impact_event.confidence:.3f}")
            print(f"   Method: {impact_event.detection_method}")
            print(f"   Hand Velocity: {impact_event.features['hand_velocity']:.4f}")
            print(f"   Pre-Impact Score: {impact_event.features['pre_impact_score']:.3f}")
        
        # Test old system for comparison
        print("\n📊 Old Impact System (for comparison):")
        from services.snapshot_generator import SnapshotGenerator
        
        generator_old = SnapshotGenerator(
            output_dir="test_old_impact_outputs",
            use_smart_detection=False  # Use old system
        )
        
        snapshots_old = generator_old.generate_swing_snapshots(video_path)
        
        if "impact" in snapshots_old:
            print(f"   Old system generated impact snapshot")
            print(f"   Path: {snapshots_old['impact']}")
        
        print(f"\n🎯 Analysis:")
        print(f"   The new pre-impact system should capture a frame just before")
        print(f"   the actual impact moment, making it more visually clear and")
        print(f"   useful for swing analysis.")
        
    except Exception as e:
        print(f"❌ Error in comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pre_impact_detection()
    compare_with_old_system()

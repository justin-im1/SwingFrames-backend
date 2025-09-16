#!/usr/bin/env python3
"""
Test script for the Advanced Golf Swing Processor

This script demonstrates the new advanced pose analysis system with improved
accuracy and robust event detection for golf swing videos.
"""

import sys
import os
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from services.advanced_golf_processor import process_golf_swing_video_advanced

def test_advanced_processor():
    """Test the advanced golf swing processor with sample videos"""
    
    # Check for sample videos in the project directory
    sample_videos = [
        "IMG_3845.MOV",
        "IMG_6596.MOV"
    ]
    
    print("🎬 Advanced Golf Swing Processor Test")
    print("=" * 50)
    
    # Look for sample videos
    found_videos = []
    for video_name in sample_videos:
        video_path = Path(video_name)
        if video_path.exists():
            found_videos.append(str(video_path))
            print(f"✅ Found sample video: {video_name}")
        else:
            print(f"❌ Sample video not found: {video_name}")
    
    if not found_videos:
        print("\n⚠️  No sample videos found. Please provide a video path as an argument.")
        if len(sys.argv) > 1:
            video_path = sys.argv[1]
            if Path(video_path).exists():
                found_videos.append(video_path)
                print(f"✅ Using provided video: {video_path}")
            else:
                print(f"❌ Provided video not found: {video_path}")
                return
        else:
            print("Usage: python test_advanced_processor.py [video_path]")
            return
    
    # Process each found video
    for video_path in found_videos:
        print(f"\n🎯 Processing: {video_path}")
        print("-" * 30)
        
        try:
            # Process the video with advanced analysis
            result = process_golf_swing_video_advanced(
                video_path=video_path,
                output_dir="outputs",
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7,
                target_fps=60
            )
            
            if result.success:
                print(f"✅ Processing completed successfully!")
                print(f"📊 Video Info:")
                print(f"  Total frames: {result.total_frames}")
                print(f"  FPS: {result.fps:.2f}")
                print(f"  Duration: {result.duration:.2f} seconds")
                print(f"  Processing time: {result.processing_time:.2f} seconds")
                
                if result.processing_stats:
                    print(f"\n📈 Processing Statistics:")
                    for key, value in result.processing_stats.items():
                        if isinstance(value, float):
                            print(f"  {key}: {value:.3f}")
                        else:
                            print(f"  {key}: {value}")
                
                print(f"\n🎯 Detected Swing Events:")
                for event_name, event in result.events.items():
                    print(f"  {event_name}:")
                    print(f"    Frame: {event.frame_index}")
                    print(f"    Timestamp: {event.timestamp:.2f}s")
                    print(f"    Confidence: {event.confidence:.3f}")
                    print(f"    Method: {event.detection_method}")
                
                print(f"\n📸 Generated Snapshots:")
                for event_name, snapshot in result.snapshots.items():
                    print(f"  {event_name}:")
                    print(f"    Image: {snapshot.image_path}")
                    print(f"    Confidence: {snapshot.confidence:.3f}")
                    if snapshot.body_metrics:
                        print(f"    Hand velocity: {snapshot.body_metrics.get('hand_velocity', 0):.3f}")
                        print(f"    Shoulder-hip angle: {snapshot.body_metrics.get('shoulder_hip_angle', 0):.1f}°")
                        print(f"    Movement stability: {snapshot.body_metrics.get('movement_stability', 0):.3f}")
                
                # Create comparison grid
                if result.snapshots:
                    try:
                        from services.advanced_golf_processor import AdvancedGolfProcessor
                        processor = AdvancedGolfProcessor(output_dir="outputs")
                        grid_path = processor.create_comparison_grid(result.snapshots)
                        print(f"\n📊 Comparison grid: {grid_path}")
                    except Exception as e:
                        print(f"\n⚠️  Could not create comparison grid: {e}")
                
                print(f"\n💾 Results saved to: outputs/advanced_processing_results.json")
                
            else:
                print(f"❌ Processing failed: {result.error_message}")
                
        except Exception as e:
            print(f"❌ Error processing video: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main function"""
    test_advanced_processor()

if __name__ == "__main__":
    main()

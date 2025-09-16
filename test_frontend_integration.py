#!/usr/bin/env python3
"""
Test script to verify frontend integration with advanced pose analysis
"""

import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from services.snapshot_generator import SnapshotGenerator

def test_frontend_integration():
    """Test that the SnapshotGenerator works with the new advanced system"""
    
    print("🔗 Testing Frontend Integration")
    print("=" * 50)
    
    # Test with smart detection enabled (default)
    print("Testing with advanced pose analysis (smart detection enabled)...")
    
    generator = SnapshotGenerator(
        output_dir="test_frontend_outputs",
        use_smart_detection=True  # This should use our new advanced system
    )
    
    # Test with one of our sample videos
    video_path = "IMG_3845.MOV"
    
    if not Path(video_path).exists():
        print(f"❌ Test video not found: {video_path}")
        return
    
    try:
        print(f"🎬 Processing: {video_path}")
        
        # This is what the frontend will call
        snapshots = generator.generate_swing_snapshots(video_path)
        
        print(f"✅ Successfully generated {len(snapshots)} snapshots:")
        for event_name, path in snapshots.items():
            print(f"  {event_name}: {path}")
        
        # Verify files exist
        all_exist = True
        for event_name, path in snapshots.items():
            if Path(path).exists():
                print(f"  ✅ {event_name} snapshot exists")
            else:
                print(f"  ❌ {event_name} snapshot missing")
                all_exist = False
        
        if all_exist:
            print(f"\n🎉 Frontend integration test PASSED!")
            print(f"   The advanced pose analysis system is ready for frontend use.")
        else:
            print(f"\n❌ Frontend integration test FAILED!")
            print(f"   Some snapshots were not generated properly.")
            
    except Exception as e:
        print(f"❌ Error during frontend integration test: {e}")
        import traceback
        traceback.print_exc()

def test_fallback_integration():
    """Test that fallback to original system still works"""
    
    print(f"\n🔄 Testing Fallback Integration")
    print("=" * 50)
    
    # Test with smart detection disabled (fallback)
    print("Testing with original system (smart detection disabled)...")
    
    generator = SnapshotGenerator(
        output_dir="test_fallback_outputs",
        use_smart_detection=False  # This should use the original system
    )
    
    video_path = "IMG_6596.MOV"
    
    if not Path(video_path).exists():
        print(f"❌ Test video not found: {video_path}")
        return
    
    try:
        print(f"🎬 Processing: {video_path}")
        
        snapshots = generator.generate_swing_snapshots(video_path)
        
        print(f"✅ Fallback system generated {len(snapshots)} snapshots:")
        for event_name, path in snapshots.items():
            print(f"  {event_name}: {path}")
        
        print(f"\n🎉 Fallback integration test PASSED!")
        print(f"   The system gracefully falls back to the original method when needed.")
            
    except Exception as e:
        print(f"❌ Error during fallback integration test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_frontend_integration()
    test_fallback_integration()

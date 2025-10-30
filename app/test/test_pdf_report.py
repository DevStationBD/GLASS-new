#!/usr/bin/env python3
"""
Test script for PDF report generation
"""

import json
import tempfile
import os
from app.pdf_report_generator import GLASSReportGenerator

# Sample JSON data from your example
sample_data = {
    "session_id": "20251022_180721",
    "timestamp": "2025-10-22T18:07:21.287127",
    "video_file": "grid_combined.mp4",
    "output_file": "mvtec_grid_tracked_inference.mp4",
    "frames_processed": 297,
    "processing_time_seconds": 17.587074518203735,
    "fps_processing": 16.887402148241666,
    "unique_defects": 2,
    "peak_concurrent_defects": 1,
    "tracking_statistics": {
        "frame_count": 297,
        "total_detections": 930,
        "total_tracks_created": 2,
        "active_tracks": 0,
        "total_tracks": 0,
        "avg_detections_per_frame": 3.1313131313131315,
        "current_fabric_speed": 30.295948820614885,
        "fabric_state": "variable_speed",
        "speed_confidence": 0.0028845617113160728,
        "is_speed_bootstrapped": True,
        "unique_defects_tracked": 2,
        "completed_tracks": 2,
        "defect_types_distribution": {
            "large_defect": 2
        },
        "total_defect_area_mm2": 359.55000000000007,
        "avg_defect_duration": 1.0,
        "max_defect_duration": 1
    },
    "tracked_defects": [
        {
            "track_id": 1,
            "defect_type": "large_defect",
            "first_frame": 2,
            "last_frame": 2,
            "duration_frames": 1,
            "max_confidence": 0.7137468457221985,
            "max_area_mm2": 89.33000000000001,
            "trajectory_length": 1
        },
        {
            "track_id": 2,
            "defect_type": "large_defect",
            "first_frame": 16,
            "last_frame": 16,
            "duration_frames": 1,
            "max_confidence": 0.7325787544250488,
            "max_area_mm2": 270.22,
            "trajectory_length": 1
        }
    ]
}

def main():
    """Test PDF report generation"""
    print("🧪 Testing PDF Report Generation")
    print("=" * 50)
    
    # Create temporary JSON file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_data, f, indent=2)
        temp_json_path = f.name
    
    try:
        # Generate PDF report
        generator = GLASSReportGenerator()
        pdf_path = generator.generate_report(
            json_path=temp_json_path,
            open_after=True
        )
        
        print(f"✅ Test completed successfully!")
        print(f"📄 PDF report generated: {pdf_path}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_json_path):
            os.unlink(temp_json_path)

if __name__ == '__main__':
    main()

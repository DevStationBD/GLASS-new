# GLASS Defect Tracking System

This module provides temporal defect tracking capabilities for continuous fabric inspection, enabling tracking of defects across multiple frames as fabric moves through the system.

## Overview

The defect tracking system extends GLASS anomaly detection with temporal tracking to:
- Track the same defect across multiple consecutive frames
- Compensate for fabric movement using motion estimation
- Generate comprehensive defect reports with physical measurements
- Provide 4-point inspection compatibility

## Components

### 1. FabricByteTracker (`fabric_bytetrack.py`)
- Core ByteTrack implementation adapted for fabric defects
- Handles defect association across frames using IoU + motion consistency
- Manages track lifecycle (active, lost, completed)
- Kalman filter prediction with fabric motion bias

### 2. GLASSDefectTracker (`glass_integration.py`)  
- Integration layer between GLASS detection and ByteTrack tracking
- Converts GLASS anomaly masks to trackable defect detections
- Provides comprehensive defect summaries and statistics
- Handles temporal defect lifecycle management

### 3. FabricMotionEstimator (`motion_estimation.py`)
- Multiple motion estimation methods (optical flow, template matching, etc.)
- Fabric-specific motion patterns and temporal smoothing
- Confidence scoring for motion estimates
- ROI-based motion estimation for fabric area

## Quick Start

```python
from defect_tracking import GLASSDefectTracker, FabricMotionEstimator

# Initialize tracking system
tracker = GLASSDefectTracker(
    pixel_size_mm=0.1,  # 0.1mm per pixel
    fabric_speed_pixels_per_frame=5.0
)

motion_estimator = FabricMotionEstimator()

# Process video frames
for frame in video_frames:
    # Estimate fabric motion
    motion = motion_estimator.estimate_motion(frame)
    
    # Run GLASS inference (your existing code)
    anomaly_mask = your_glass_model.predict(frame)
    
    # Track defects
    active_tracks = tracker.process_frame(
        frame, anomaly_mask, motion.displacement
    )
    
    print(f"Frame has {len(active_tracks)} active defects")

# Get final report
summaries = tracker.get_tracked_defect_summaries()
stats = tracker.get_tracking_statistics()
```

## Usage with Existing GLASS Pipeline

### Enhanced Video Inference
```bash
python inference/video_inference_with_tracking.py \
    --model_path results/models/backbone_0/ \
    --class_name custom_grid \
    --video_path test-video/fabric_inspection.mp4 \
    --output_path output/tracked_results.mp4 \
    --fabric_length 2.0 \
    --pixel_size 0.1
```

### Testing
```bash
# Test all components
python test_defect_tracking.py --all

# Test with existing videos
python test_defect_tracking.py --test_videos

# Test individual components  
python test_defect_tracking.py --test_components
```

## Key Features

### ✅ Temporal Tracking
- Tracks defects across multiple frames
- Handles defect appearance/disappearance at frame edges
- Maintains defect identity throughout fabric movement

### ✅ Motion Compensation
- Multiple motion estimation algorithms
- Fabric-specific motion patterns
- Temporal smoothing for stable tracking

### ✅ Physical Measurements
- Convert pixel measurements to physical units (mm, cm, inches)
- Track defect area changes over time
- Calculate fabric length affected by each defect

### ✅ Comprehensive Reporting
- Individual defect lifecycle summaries
- Tracking statistics and performance metrics
- JSON reports with detailed tracking data
- 4-point inspection compatibility (future)

## Configuration

### Tracker Parameters
```python
tracker = GLASSDefectTracker(
    pixel_size_mm=0.1,                    # Physical calibration
    fabric_speed_pixels_per_frame=5.0,    # Expected fabric speed
    high_conf_threshold=0.7,              # High confidence for new tracks
    low_conf_threshold=0.3,               # Low confidence for track recovery
    min_defect_area=10,                   # Minimum defect size (pixels)
    detection_threshold=0.5               # GLASS anomaly threshold
)
```

### Motion Estimator Parameters
```python
motion_estimator = FabricMotionEstimator(
    method=MotionEstimationMethod.OPTICAL_FLOW_SPARSE,  # Fastest method
    fabric_roi=(50, 50, 500, 400)  # Focus on fabric area (optional)
)
```

## Output Format

### Tracking Report (JSON)
```json
{
  "session_id": "20240101_123456",
  "unique_defects": 5,
  "defect_density_per_meter": 2.5,
  "total_defect_area_mm2": 156.7,
  "tracked_defects": [
    {
      "track_id": 1,
      "defect_type": "hole",
      "duration_frames": 25,
      "max_confidence": 0.95,
      "max_area_mm2": 45.2,
      "fabric_length_affected_mm": 12.5
    }
  ]
}
```

## Performance

- **Real-time capable**: ~30 FPS on modern GPUs
- **Memory efficient**: Sliding window approach for long videos
- **Scalable**: Handles videos with hundreds of defects
- **Robust**: Handles partial occlusions and detection noise

## Requirements

- OpenCV 4.x
- NumPy
- SciPy
- Existing GLASS dependencies

## Integration Notes

- **Drop-in compatible**: Works with existing GLASS models without retraining
- **Configurable**: All parameters tunable for different fabric types
- **Extensible**: Easy to add new motion estimation methods or tracking algorithms
- **Production ready**: Comprehensive error handling and logging
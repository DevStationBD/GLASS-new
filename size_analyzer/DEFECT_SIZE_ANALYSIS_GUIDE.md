# GLASS Defect Size Analysis Guide

## Overview

The GLASS defect size analysis system provides comprehensive measurement capabilities for anomaly detection, including individual defect sizing, spatial analysis, and physical measurements. This guide covers all aspects of using the defect size analysis features.

## Architecture

### Core Components

1. **`defect_size_analyzer.py`**: Main analysis module with comprehensive measurement algorithms
2. **`video_inference_with_size.py`**: Enhanced video inference with integrated size analysis
3. **`test_defect_size_analysis.py`**: Test suite and examples

### Analysis Approaches

- **Pixel-Based**: Count pixels above threshold for total defect coverage
- **Connected Components**: Identify and measure individual defects
- **Contour Analysis**: Precise boundary detection and shape analysis
- **Multi-Threshold**: Defect severity assessment at different confidence levels
- **Physical Measurements**: Real-world size calculations with pixel-to-physical conversion

## Quick Start

### 1. Test the System

Run the test script to verify functionality:

```bash
python test_defect_size_analysis.py
```

This generates:
- `test_defect_visualization.png`: Annotated defect visualization
- `test_defect_measurements.json`: Detailed measurements
- `test_size_distribution.png`: Size distribution histogram

### 2. Video Analysis with Size Measurements

```bash
python video_inference_with_size.py \
    --model_path results/models/backbone_0/ \
    --class_name wfdd_yellow_cloth \
    --video_path input/fabric_video.mp4 \
    --output_path output/fabric_analysis_with_size.mp4 \
    --pixel_size 0.1 \
    --physical_unit mm \
    --export_results \
    --results_dir results/fabric_analysis
```

### 3. Single Image Analysis

```python
from defect_size_analyzer import DefectSizeAnalyzer
import cv2
import numpy as np

# Initialize analyzer with physical measurements
analyzer = DefectSizeAnalyzer(pixel_size=0.1, physical_unit="mm")

# Load image and mask (from GLASS inference)
image = cv2.imread("image.jpg")
mask = np.load("glass_anomaly_mask.npy")  # GLASS output mask

# Analyze defects
metrics = analyzer.analyze_defects(mask, threshold=0.5)

# Print results
print(f"Found {metrics.num_defects} defects")
print(f"Total area: {metrics.total_defect_area_physical:.2f} mm²")
```

## Detailed Usage

### DefectSizeAnalyzer Class

#### Initialization Parameters

```python
DefectSizeAnalyzer(
    pixel_size=None,        # Physical size per pixel (e.g., 0.1 for 0.1mm/pixel)
    physical_unit="mm",     # Unit for physical measurements
    min_defect_area=10      # Minimum defect size in pixels
)
```

#### Key Methods

**`analyze_defects(mask, threshold=0.5, use_morphology=True)`**
- **Input**: GLASS anomaly probability mask (0-1 float values)
- **Output**: DefectMetrics object with comprehensive measurements
- **Parameters**:
  - `threshold`: Binary threshold for defect detection (0.0-1.0)
  - `use_morphology`: Apply morphological operations to clean noise

**`create_defect_visualization(image, mask, metrics, threshold=0.5)`**
- Creates annotated image with defect boundaries and size labels
- **Returns**: Annotated image with measurements overlay

**`export_measurements(metrics, filename, include_coordinates=True)`**
- Exports detailed measurements to JSON format
- **Options**: Include/exclude spatial coordinates and bounding boxes

### DefectMetrics Output

The analysis returns a comprehensive `DefectMetrics` object containing:

#### Overall Metrics
- `total_defect_pixels`: Total pixels classified as defective
- `total_image_pixels`: Total image size
- `defect_percentage`: Percentage of image covered by defects

#### Individual Defect Data
- `num_defects`: Number of separate defect regions
- `defect_areas`: List of individual defect areas (pixels)
- `defect_centroids`: Center coordinates of each defect
- `defect_bounding_boxes`: Bounding rectangles for each defect

#### Statistical Analysis
- `largest_defect_area`: Size of biggest defect
- `average_defect_area`: Mean defect size
- `median_defect_area`: Median defect size
- `severity_levels`: Pixel counts at different threshold levels

#### Physical Measurements (if pixel_size provided)
- `physical_unit`: Unit of measurement (mm, cm, etc.)
- `total_defect_area_physical`: Total defect area in physical units
- `defect_areas_physical`: Individual defect areas in physical units

## Video Analysis Features

### Enhanced Video Processing

The `video_inference_with_size.py` script provides:

#### Real-Time Size Analysis
- Frame-by-frame defect measurement
- Live display with size annotations
- Performance monitoring with FPS tracking

#### Interactive Controls
- **'q'**: Quit processing
- **'p'**: Pause/Resume
- **'s'**: Save current frame analysis

#### Comprehensive Output
- Side-by-side video with size annotations
- Individual defect labeling with measurements
- Real-time statistics overlay
- Trend analysis across frames

#### Export Capabilities
- JSON summary with video-level statistics
- Frame-by-frame size history
- Sampled detailed measurements
- Trend analysis data

### Example Output Structure

```
results/fabric_analysis/
├── video_analysis_summary.json     # Overall video statistics
├── frame_size_history.json         # Frame-by-frame measurements
├── frame_000010_metrics.json       # Detailed frame samples
├── frame_000020_metrics.json
└── detailed_frame_samples.json     # List of sampled frames
```

## Physical Measurements Setup

### Calibration Process

1. **Determine Pixel Size**: Measure known object in image
   ```
   pixel_size = real_size_mm / size_in_pixels
   ```

2. **Example**: If 10mm object appears as 100 pixels:
   ```
   pixel_size = 10.0 / 100.0 = 0.1 mm/pixel
   ```

3. **Use in Analysis**:
   ```bash
   --pixel_size 0.1 --physical_unit mm
   ```

### Supported Units
- `mm`: Millimeters
- `cm`: Centimeters  
- `inch`: Inches
- `um`: Micrometers
- Custom units supported

## Analysis Parameters

### Threshold Selection

| Threshold | Sensitivity | Use Case |
|-----------|-------------|----------|
| 0.3-0.4 | High | Detect subtle defects, may include noise |
| 0.5-0.6 | Moderate | Balanced detection for most applications |
| 0.7-0.8 | Conservative | High-confidence defects only |
| 0.9+ | Very strict | Critical defects with high certainty |

### Morphological Processing

**Enabled (default)**: Cleans noise and fills holes
```python
use_morphology=True  # Recommended for most cases
```

**Disabled**: Raw threshold without cleanup
```python
use_morphology=False  # For very precise analysis
```

### Minimum Defect Area

Filter small regions that may be noise:
```python
min_defect_area=10   # Default: 10 pixels minimum
min_defect_area=5    # High sensitivity
min_defect_area=20   # Filter small defects
```

## Advanced Features

### Multi-Threshold Severity Analysis

Automatically analyzes defects at multiple confidence levels:

```python
severity_thresholds = {
    'low': 0.3,      # Possible defects
    'medium': 0.5,   # Likely defects  
    'high': 0.7,     # Confident defects
    'critical': 0.9  # Definite defects
}
```

### Batch Processing

Process multiple frames efficiently:

```python
from defect_size_analyzer import batch_analyze_video_frames

# List of (image, mask) tuples
video_frames = [(frame1, mask1), (frame2, mask2), ...]

# Batch analyze
results = batch_analyze_video_frames(
    video_frames, 
    pixel_size=0.1, 
    threshold=0.5
)
```

### Size Distribution Analysis

Generate histograms and statistics:

```python
analyzer.create_size_distribution_plot(metrics, 'size_histogram.png')
```

## Integration Examples

### Custom Analysis Pipeline

```python
import cv2
import numpy as np
from defect_size_analyzer import DefectSizeAnalyzer

def analyze_production_samples(image_dir, model, pixel_size=0.1):
    """Analyze production samples with size measurements"""
    analyzer = DefectSizeAnalyzer(pixel_size=pixel_size, physical_unit="mm")
    results = []
    
    for image_path in glob.glob(os.path.join(image_dir, "*.jpg")):
        # Load and preprocess image
        image = cv2.imread(image_path)
        
        # Get GLASS prediction (your existing inference code)
        mask = your_glass_inference_function(image, model)
        
        # Analyze defects
        metrics = analyzer.analyze_defects(mask, threshold=0.6)
        
        # Create visualization
        vis = analyzer.create_defect_visualization(image, mask, metrics)
        
        # Save results
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        cv2.imwrite(f"results/{base_name}_analysis.jpg", vis)
        analyzer.export_measurements(metrics, f"results/{base_name}_metrics.json")
        
        results.append({
            'image': image_path,
            'num_defects': metrics.num_defects,
            'total_area_mm2': metrics.total_defect_area_physical,
            'defect_percentage': metrics.defect_percentage
        })
    
    return results
```

### Quality Control Integration

```python
def quality_control_check(mask, analyzer, thresholds):
    """Automated quality control with size-based decisions"""
    metrics = analyzer.analyze_defects(mask, threshold=0.5)
    
    # Decision logic based on size measurements
    if metrics.total_defect_area_physical > thresholds['max_total_area']:
        return "REJECT", "Excessive defect area"
    
    if metrics.largest_defect_area > thresholds['max_single_defect']:
        return "REJECT", "Large defect detected"
    
    if metrics.num_defects > thresholds['max_defect_count']:
        return "REJECT", "Too many defects"
    
    return "ACCEPT", "Within quality limits"

# Usage
thresholds = {
    'max_total_area': 2.0,      # 2 mm² total
    'max_single_defect': 0.5,   # 0.5 mm² largest
    'max_defect_count': 5       # 5 defects max
}

decision, reason = quality_control_check(mask, analyzer, thresholds)
```

## Performance Considerations

### Processing Speed
- **Video analysis**: ~10-25 FPS on modern GPUs
- **Single image**: ~50-100ms per analysis
- **Batch processing**: Linear scaling with frame count

### Memory Usage
- **Analyzer**: Minimal memory footprint
- **Video processing**: Scales with video resolution
- **Export data**: JSON files typically <1MB per frame

### Optimization Tips

1. **Adjust minimum defect area** to filter noise
2. **Use appropriate thresholds** to avoid over-detection
3. **Disable morphology** if speed is critical
4. **Batch process** multiple frames for efficiency
5. **Export selectively** (e.g., every 10th frame) for large videos

## Output Examples

### Console Output
```
GLASS VIDEO ANALYSIS WITH SIZE MEASUREMENTS - COMPLETE
============================================================
Output saved to: output/fabric_analysis.mp4
Total frames processed: 1500
Processing time: 75.32 seconds
Average processing FPS: 19.92
Anomalous frames detected: 245
Anomaly percentage: 16.33%

DEFECT SIZE ANALYSIS:
Total defects detected: 1234
Frames with defects: 245
Average defects per frame: 0.82
Average defects per anomalous frame: 5.04

DEFECT SIZE STATISTICS (pixels):
Size range: 12 - 1847 pixels
Mean defect size: 156.3 ± 287.4 pixels
Median defect size: 67.0 pixels

PHYSICAL MEASUREMENTS (mm²):
Pixel size: 0.1 mm/pixel
Total defect area: 123.4 mm²
Mean defect size: 1.56 mm²
Largest defect: 18.47 mm²
```

### JSON Export Example
```json
{
  "summary": {
    "total_defect_pixels": 1234,
    "defect_percentage": 2.34,
    "num_defects": 8,
    "largest_defect_area": 187,
    "average_defect_area": 154.25
  },
  "physical_measurements": {
    "unit": "mm²",
    "total_defect_area": 12.34,
    "defect_areas": [1.87, 1.54, 0.67, ...]
  },
  "individual_defects": [
    {
      "id": 1,
      "area_pixels": 187,
      "area_physical": 1.87,
      "centroid": {"x": 123.4, "y": 567.8},
      "bounding_box": {"x": 100, "y": 550, "width": 47, "height": 36}
    }
  ],
  "severity_levels": {
    "low": 1456,
    "medium": 1234,
    "high": 892,
    "critical": 234
  }
}
```

## Troubleshooting

### Common Issues

**No defects detected**
- Lower threshold value
- Check mask format (should be 0-1 float)
- Verify min_defect_area setting

**Too many small detections**
- Increase min_defect_area
- Enable morphological processing
- Raise threshold

**Inaccurate size measurements**
- Verify pixel_size calibration
- Check image resolution consistency
- Ensure proper physical unit

**Performance issues**
- Reduce video resolution
- Increase min_defect_area
- Disable live display for batch processing

## Support and Extensions

### Customization Options
- Modify severity thresholds in analyzer
- Add custom morphological operations
- Implement domain-specific quality rules
- Create custom visualization styles

### Future Enhancements
- Shape analysis (circularity, aspect ratio)
- Defect classification by size/shape
- Real-time streaming analysis
- Database integration for production monitoring

## Best Practices

1. **Calibrate pixel size** carefully for accurate physical measurements
2. **Test threshold values** on representative samples
3. **Use morphological processing** unless precision is critical
4. **Export results systematically** for quality tracking
5. **Monitor trends** across production runs
6. **Validate measurements** against manual inspection initially
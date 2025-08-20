# Size Analyzer Module

This module provides comprehensive defect size analysis capabilities for the GLASS framework, enabling precise measurement and characterization of detected anomalies.

## Overview

The Size Analyzer module extends GLASS anomaly detection with quantitative defect analysis, including:
- Physical size measurements in multiple units (mm, cm, inch)
- Statistical analysis of defect distributions
- Severity classification and trend analysis
- Batch processing and export capabilities
- Advanced visualization with detailed annotations

## Module Structure

```
size_analyzer/
├── __init__.py                      # Module initialization
├── defect_size_analyzer.py          # Main analyzer class
├── test_defect_size_analysis.py     # Comprehensive testing script
├── test_defect_measurements.json    # Sample test data
├── DEFECT_SIZE_ANALYSIS_GUIDE.md   # Detailed usage guide
└── README.md                       # This file
```

## Core Components

### DefectSizeAnalyzer Class
The main analyzer class providing comprehensive defect measurement capabilities.

```python
from size_analyzer import DefectSizeAnalyzer, DefectMetrics

# Initialize analyzer
analyzer = DefectSizeAnalyzer(
    pixel_size=0.1,           # Physical size per pixel (mm/pixel)
    physical_unit="mm",       # Unit for measurements
    min_defect_area=10       # Minimum defect size threshold
)

# Analyze defects in a mask
metrics = analyzer.analyze_defects(
    anomaly_mask,             # 2D numpy array (0-1 values)
    threshold=0.5,            # Threshold for defect detection
    use_morphology=True       # Use morphological operations
)
```

### DefectMetrics Class
Comprehensive metrics container for defect analysis results.

**Key Attributes**:
- `num_defects`: Total number of defects detected
- `total_defect_pixels`: Total defect area in pixels
- `defect_percentage`: Percentage of image covered by defects
- `defect_areas`: List of individual defect sizes
- `largest_defect_area`: Size of largest defect
- `average_defect_area`: Mean defect size
- `median_defect_area`: Median defect size
- `defect_areas_physical`: Physical sizes (if pixel_size provided)
- `severity_levels`: Classification by severity (low/medium/high/critical)

## Key Features

### 1. Physical Measurements
Convert pixel measurements to real-world units:

```python
# Enable physical measurements
analyzer = DefectSizeAnalyzer(pixel_size=0.05, physical_unit="mm")
metrics = analyzer.analyze_defects(mask, threshold=0.5)

print(f"Total defect area: {metrics.total_defect_area_physical:.2f} mm²")
print(f"Largest defect: {max(metrics.defect_areas_physical):.2f} mm²")
```

### 2. Severity Classification
Automatic defect severity classification based on size:

```python
# Access severity levels
for level, pixel_count in metrics.severity_levels.items():
    print(f"{level}: {pixel_count} pixels")
```

**Severity Thresholds**:
- **Low**: < 50 pixels
- **Medium**: 50-200 pixels  
- **High**: 200-1000 pixels
- **Critical**: > 1000 pixels

### 3. Advanced Visualization
Create annotated visualizations with size information:

```python
# Generate visualization with size annotations
vis_image = analyzer.create_defect_visualization(
    original_image, mask, metrics, threshold=0.5
)

# Save visualization
cv2.imwrite('defect_analysis.png', vis_image)
```

**Visualization Features**:
- Color-coded defects by size
- Individual defect numbering
- Size annotations with physical measurements
- Statistical overlays
- Customizable color schemes

### 4. Statistical Analysis
Comprehensive statistical analysis of defect distributions:

```python
# Generate size distribution plot
analyzer.create_size_distribution_plot(metrics, 'size_distribution.png')

# Access statistical measures
print(f"Mean size: {metrics.average_defect_area:.1f} pixels")
print(f"Size std dev: {np.std(metrics.defect_areas):.1f} pixels")
```

### 5. Export Capabilities
Export detailed measurements and analysis results:

```python
# Export comprehensive measurements
analyzer.export_measurements(
    metrics, 
    'analysis_results.json',
    include_coordinates=True    # Include defect centroids and bounding boxes
)
```

**Export Formats**:
- **JSON**: Structured data with all measurements
- **CSV**: Tabular format for spreadsheet analysis
- **Images**: Annotated visualizations
- **Plots**: Statistical distribution charts

## Usage Examples

### Basic Defect Analysis
```python
from size_analyzer import DefectSizeAnalyzer
import cv2
import numpy as np

# Load image and anomaly mask
image = cv2.imread('sample.png')
mask = np.load('anomaly_mask.npy')  # GLASS output

# Initialize analyzer
analyzer = DefectSizeAnalyzer()

# Analyze defects
metrics = analyzer.analyze_defects(mask, threshold=0.5)

# Print results
print(f"Found {metrics.num_defects} defects")
print(f"Total coverage: {metrics.defect_percentage:.2f}%")
print(f"Largest defect: {metrics.largest_defect_area} pixels")
```

### Physical Measurements
```python
# Initialize with physical calibration
analyzer = DefectSizeAnalyzer(pixel_size=0.1, physical_unit="mm")

# Analyze with physical measurements
metrics = analyzer.analyze_defects(mask, threshold=0.5)

# Print physical measurements
print(f"Total defect area: {metrics.total_defect_area_physical:.2f} mm²")
for i, area in enumerate(metrics.defect_areas_physical):
    print(f"Defect {i+1}: {area:.2f} mm²")
```

### Batch Video Analysis
```python
from size_analyzer.defect_size_analyzer import batch_analyze_video_frames

# Prepare frame data (image, mask) pairs
frames_data = [(image1, mask1), (image2, mask2), ...]

# Batch analyze
results = batch_analyze_video_frames(
    frames_data, 
    pixel_size=0.1, 
    threshold=0.5
)

# Process results
for i, metrics in enumerate(results):
    print(f"Frame {i}: {metrics.num_defects} defects")
```

### Custom Threshold Analysis
```python
# Test multiple thresholds
thresholds = [0.3, 0.5, 0.7, 0.9]
results = {}

for thresh in thresholds:
    metrics = analyzer.analyze_defects(mask, threshold=thresh)
    results[thresh] = {
        'defects': metrics.num_defects,
        'coverage': metrics.defect_percentage,
        'avg_size': metrics.average_defect_area
    }

# Find optimal threshold
optimal_thresh = max(results.keys(), key=lambda t: results[t]['defects'])
print(f"Optimal threshold: {optimal_thresh}")
```

## Integration with GLASS Inference

The Size Analyzer integrates seamlessly with GLASS video inference:

```python
# In video inference scripts
from size_analyzer import DefectSizeAnalyzer

# Initialize during setup
self.size_analyzer = DefectSizeAnalyzer(
    pixel_size=pixel_size,
    physical_unit=physical_unit,
    min_defect_area=5
)

# Analyze each frame
def predict_frame(self, frame_tensor, threshold=0.5):
    # GLASS inference
    scores, masks = self.glass_model._predict(frame_tensor)
    
    # Size analysis
    metrics = self.size_analyzer.analyze_defects(
        masks[0], threshold=threshold
    )
    
    return scores[0], masks[0], metrics
```

## Testing

### Run Comprehensive Tests
```bash
cd size_analyzer
python test_defect_size_analysis.py
```

**Test Coverage**:
- Synthetic anomaly generation
- Multiple analyzer configurations
- Physical measurement accuracy
- Visualization generation
- Export functionality
- Batch processing

**Generated Test Files**:
- `test_defect_visualization.png`: Annotated defect visualization
- `test_defect_measurements.json`: Comprehensive measurements
- `test_size_distribution.png`: Size distribution histogram

### Custom Testing
```python
# Create custom test data
def create_test_mask():
    mask = np.zeros((384, 384), dtype=np.float32)
    cv2.circle(mask, (100, 100), 20, 0.8, -1)  # Large defect
    cv2.circle(mask, (200, 200), 10, 0.6, -1)  # Medium defect
    return mask

# Test analyzer
mask = create_test_mask()
analyzer = DefectSizeAnalyzer(pixel_size=0.1, physical_unit="mm")
metrics = analyzer.analyze_defects(mask, threshold=0.5)

assert metrics.num_defects == 2
assert metrics.total_defect_area_physical > 0
```

## Performance Optimization

### For Real-time Applications
- Use `min_defect_area` threshold to filter noise
- Disable morphological operations for speed: `use_morphology=False`
- Cache analyzer instances to avoid re-initialization

### For High Accuracy Applications
- Use lower `min_defect_area` values
- Enable morphological operations: `use_morphology=True`
- Use multiple threshold values for sensitivity analysis

## Advanced Configuration

### Morphological Operations
Control noise filtering and defect separation:

```python
analyzer = DefectSizeAnalyzer(
    morphology_kernel_size=3,    # Kernel size for operations
    closing_iterations=2,        # Fill small holes
    opening_iterations=1         # Remove noise
)
```

### Custom Severity Thresholds
Modify severity classification:

```python
# Custom severity levels (in pixels)
analyzer.severity_thresholds = {
    'low': (0, 25),
    'medium': (25, 100),
    'high': (100, 500),
    'critical': (500, float('inf'))
}
```

## Troubleshooting

### Common Issues

1. **No defects detected**: Check threshold value and mask range
2. **Too many small defects**: Increase `min_defect_area`
3. **Import errors**: Ensure proper path setup in inference scripts
4. **Physical measurements missing**: Verify `pixel_size` parameter

### Debug Mode
Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

analyzer = DefectSizeAnalyzer()
# Detailed analysis info will be logged
```

## Future Enhancements

- Support for non-uniform pixel sizes
- 3D defect analysis for multi-layer materials
- Machine learning-based severity classification
- Real-time defect tracking across video frames
- Advanced shape analysis (circularity, elongation, etc.)

## References

- See `DEFECT_SIZE_ANALYSIS_GUIDE.md` for detailed usage examples
- Integration examples in `inference/` directory
- Test cases in `test_defect_size_analysis.py`
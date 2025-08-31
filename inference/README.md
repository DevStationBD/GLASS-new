# GLASS Inference Scripts

This directory contains all inference-related scripts for the GLASS (Global and Local Anomaly co-Synthesis Strategy) framework.

## Scripts Overview

### Core Inference Scripts

**`inference_orchestrator.py`** - 🤖 **NEW** Automatic model selection with inference
- **Smart Model Selection**: Automatically tests all available models and selects the best one
- **Lowest Anomaly Score**: Chooses model with lowest average anomaly score (best fit)
- **Organized Output**: Creates structured output with timestamps and reports
- **One-Command Solution**: No need to specify model or class manually

**`video_inference_with_tracking.py`** - 🎯 **NEW** Advanced tracking with defect persistence
- **Temporal Defect Tracking**: Tracks same defects across multiple frames
- **Motion Compensation**: Fabric movement estimation using optical flow
- **Physical Measurements**: Real-world defect size calculations
- **Individual Frame Extraction**: Saves one representative frame per tracked defect
- **Organized Output**: Automatic directory structure with timestamps

**`video_inference_sidebyside.py`** - Main video inference script with side-by-side display
- Creates side-by-side videos showing original and anomaly detection results
- Includes defect size analysis and real-time metrics
- Supports live display during processing
- Most comprehensive inference script

**`video_inference_clean.py`** - Clean inference without text overlays
- Clean output videos without annotations for presentations
- Adjustable intensity settings for defect highlighting
- Professional output for demonstrations

**`video_inference.py`** - Basic video inference script
- Simple video processing with anomaly detection
- Outputs annotated video with anomaly overlays
- Lightweight version without side-by-side display

**`video_inference_with_size.py`** - Video inference with detailed defect measurements
- Advanced defect size analysis and physical measurements
- Detailed metrics output and defect characterization
- Physical unit support (mm, cm, inch)

**`run_video_inference.py`** - Easy-to-use wrapper script
- Simplified interface for common inference tasks
- Pre-configured settings for WFDD models
- Minimal command-line arguments

## Usage Examples

### 🤖 Automatic Model Selection (NEW - Recommended)
```bash
# Orchestrator automatically selects best model and runs inference
python inference/inference_orchestrator.py \
    --video_path input_video.mp4

# List all available trained models
python inference/inference_orchestrator.py --list_models

# Custom settings
python inference/inference_orchestrator.py \
    --video_path input_video.mp4 \
    --sample_frames 50 \
    --device cuda:0
```

### 🎯 Advanced Tracking with Defect Persistence (NEW)
```bash
# Defect tracking with organized output (all defaults enabled)
python inference/video_inference_with_tracking.py \
    --model_path results/models/backbone_0/ \
    --class_name mvtec_grid-test \
    --video_path input_video.mp4

# Disable organized output (use manual paths)
python inference/video_inference_with_tracking.py \
    --model_path results/models/backbone_0/ \
    --class_name mvtec_grid-test \
    --video_path input_video.mp4 \
    --output_path output.mp4 \
    --no_organized_output \
    --no_save_defect_frames
```

### Side-by-Side Video Inference
```bash
python inference/video_inference_sidebyside.py \
    --model_path results/models/backbone_0/ \
    --class_name mvtec_grid \
    --video_path test-video/grid_test_video.mp4 \
    --output_path output/grid_inference_sidebyside.mp4 \
    --threshold 0.8 \
    --no_display
```

### Basic Video Inference
```bash
python inference/video_inference.py \
    --model_path results/models/backbone_0/ \
    --class_name wfdd_yellow_cloth \
    --video_path input.mp4 \
    --output_path output.mp4
```

### Video Inference with Size Analysis
```bash
python inference/video_inference_with_size.py \
    --model_path results/models/backbone_0/ \
    --class_name wfdd_pink_flower \
    --video_path input.mp4 \
    --output_path output.mp4 \
    --pixel_size 0.1 \
    --physical_unit mm
```

### Easy Inference Wrapper
```bash
python inference/run_video_inference.py \
    input.mp4 output.mp4 \
    --class_name wfdd_yellow_cloth \
    --threshold 0.8
```

## Common Parameters

### Required Parameters
- `--model_path`: Path to model directory (e.g., `results/models/backbone_0/`)
- `--class_name`: Model class name (e.g., `mvtec_grid`, `wfdd_yellow_cloth`)
- `--video_path`: Input video file path
- `--output_path`: Output video file path

### Optional Parameters
- `--threshold`: Anomaly detection threshold (default: 0.8-0.9 depending on script)
- `--image_size`: Input image size for model (default: 384)
- `--device`: GPU device (default: `cuda:0`)
- `--no_display`: Disable live preview window
- `--pixel_size`: Physical size per pixel for measurements
- `--physical_unit`: Unit for physical measurements (mm, cm, inch)

## Supported Models

The inference scripts work with any trained GLASS model. Common model classes:

### MVTec AD Dataset
- `mvtec_grid`
- `mvtec_cable`
- `mvtec_capsule`
- `mvtec_hazelnut`
- And others...

### WFDD Dataset
- `wfdd_grey_cloth`
- `wfdd_grid_cloth`
- `wfdd_pink_flower`
- `wfdd_yellow_cloth`

### Custom Datasets
- Use the exact class name from your training configuration

## Output Formats

### Side-by-Side Video
- Left: Original video frames
- Right: Anomaly detection results with overlays
- Annotations include: anomaly scores, defect counts, area percentages, FPS metrics

### Regular Video
- Original frames with anomaly overlays
- Color-coded anomaly maps (red = anomalous regions)
- Text annotations with scores and status

### Defect Analysis Output
- Detailed defect metrics in console
- Physical measurements when pixel size is provided
- JSON output options for batch processing

## Performance Notes

- **GPU Recommended**: CUDA-compatible GPU significantly improves processing speed
- **Memory Usage**: ~4-8GB GPU memory depending on model and video resolution
- **Processing Speed**: Typically 20-60 FPS depending on hardware
- **Real-time Capability**: Most setups process faster than real-time video playback

## Troubleshooting

### Common Issues
1. **Import Errors**: Scripts automatically add parent directory to Python path
2. **Model Not Found**: Ensure correct model_path and class_name
3. **CUDA Errors**: Use `--device cpu` for CPU-only inference
4. **Memory Errors**: Reduce batch size or video resolution

### File Paths
All scripts should be run from the project root directory:
```bash
cd /path/to/GLASS-new
python inference/script_name.py [arguments]
```

## Integration with Preprocessing

The inference scripts work seamlessly with preprocessing tools:
- Use `preprocessing/create_video.py` to create test videos from image datasets
- Process the generated videos with any inference script
- Output videos are saved to specified paths

## Advanced Usage

### Batch Processing
```bash
# Process multiple videos
for video in test-videos/*.mp4; do
    python inference/video_inference_sidebyside.py \
        --model_path results/models/backbone_0/ \
        --class_name mvtec_grid \
        --video_path "$video" \
        --output_path "output/$(basename "$video" .mp4)_inference.mp4" \
        --no_display
done
```

### Custom Thresholds
Different models may require different thresholds:
- MVTec AD: 0.8-0.9
- WFDD: 0.7-0.8  
- Custom datasets: Experiment with 0.5-0.9 range

### Physical Measurements
For accurate size measurements:
1. Calibrate pixel size using known reference objects
2. Use consistent units across measurements
3. Validate measurements with manual verification

## 🆕 New Features Overview

### Automatic Model Selection
The **Inference Orchestrator** (`inference_orchestrator.py`) eliminates the need to manually specify models:
- Tests all available trained models on sample frames
- Selects model with **lowest average anomaly score** (best fit for the video content)
- Automatically runs full inference with selected model
- Provides comprehensive selection reports

### Advanced Defect Tracking
The **Tracking Inference** (`video_inference_with_tracking.py`) provides enterprise-grade defect monitoring:
- **Temporal Tracking**: Follows defects across frames using ByteTrack algorithm
- **Motion Compensation**: Accounts for fabric movement during inspection
- **Smart Frame Extraction**: Saves one representative frame per tracked defect (middle frame)
- **Physical Measurements**: Real-world defect size calculations
- **Organized Output**: Automatic directory structure with timestamps

### Organized Output Structure
Both new scripts create structured output directories:
```
output/[class-name]/[timestamp]/
├── output-video/          # Inference videos
├── defects/              # Individual defect frames with annotations
└── report/               # JSON tracking and selection reports
```

### Enhanced Annotations
- **Semi-transparent overlays**: Info panels with 30% opacity to preserve fabric visibility
- **Defect-specific annotations**: Track IDs displayed directly on detected defects
- **Comprehensive tracking data**: Confidence scores, physical measurements, temporal statistics
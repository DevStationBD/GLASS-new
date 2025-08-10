# GLASS Video Inference Manual

## Overview

The `video_inference_sidebyside.py` script provides real-time video anomaly detection using the GLASS model. It processes video files frame-by-frame and creates side-by-side output videos showing the original frames alongside anomaly detection results with heatmaps and scores.

## Features

- **Real-time Inference**: Processes video frames with live FPS monitoring
- **Side-by-Side Output**: Creates comparison videos with original and annotated frames
- **Live Display**: Optional real-time preview with pause/resume controls
- **Anomaly Visualization**: Color-coded heatmaps highlighting anomalous regions
- **Performance Metrics**: Detailed timing and detection statistics
- **Interactive Controls**: Keyboard controls for live viewing

## Requirements

- Trained GLASS model checkpoint
- Python 3.9+ with required dependencies
- CUDA-compatible GPU (recommended)
- OpenCV for video processing

## Usage

### Basic Command Structure

```bash
python video_inference_sidebyside.py \
    --model_path <path_to_models> \
    --class_name <class_name> \
    --video_path <input_video> \
    --output_path <output_video> \
    [options]
```

### Required Parameters

- `--model_path`: Path to model directory containing checkpoints (e.g., `results/models/backbone_0/`)
- `--class_name`: Class name matching your trained model (e.g., `wfdd_yellow_cloth`)
- `--video_path`: Path to input video file (supports common formats: .mp4, .avi, .mov)
- `--output_path`: Path for output side-by-side video file

### Optional Parameters

- `--threshold`: Anomaly detection threshold (default: 0.9, range: 0.0-1.0)
- `--image_size`: Image size used during training (default: 384)
- `--device`: Processing device (default: 'cuda:0', fallback: 'cpu')
- `--no_display`: Disable live preview window (for headless operation)

## Example Usage

### Basic Video Processing
```bash
python video_inference_sidebyside.py \
    --model_path results/models/backbone_0/ \
    --class_name wfdd_yellow_cloth \
    --video_path input/fabric_video.mp4 \
    --output_path output/fabric_analysis.mp4
```

### Custom Threshold and Device
```bash
python video_inference_sidebyside.py \
    --model_path results/models/backbone_0/ \
    --class_name mvtec_bottle \
    --video_path videos/bottle_inspection.mp4 \
    --output_path results/bottle_inspection_analyzed.mp4 \
    --threshold 0.8 \
    --device cuda:1
```

### Headless Processing (No Live Display)
```bash
python video_inference_sidebyside.py \
    --model_path results/models/backbone_0/ \
    --class_name visa_candle \
    --video_path input/candle_sequence.mp4 \
    --output_path output/candle_results.mp4 \
    --no_display
```

## Output Visualization

### Side-by-Side Layout
- **Left Panel**: Original video frames with "Original" label
- **Right Panel**: Annotated frames with anomaly detection overlay

### Annotation Elements
- **Anomaly Heatmap**: Color-coded overlay (blue=normal, red=anomalous)
- **Anomaly Score**: Numerical confidence score (0.0-1.0)
- **Threshold Line**: Reference threshold value
- **Status Indicator**: "NORMAL" (green) or "ANOMALY" (red)
- **Performance Metrics**: Real-time FPS information
- **Frame Labels**: "Original" and "GLASS Detection"

### Color Coding
- **Green Text**: Normal status, general information
- **Red Text**: Anomaly detected status
- **Yellow Text**: Performance metrics (FPS)
- **White Text**: Labels and threshold information

## Interactive Controls

### Live Display Window
When live display is enabled (default), use these keyboard controls:

- **'q'**: Quit processing and save current results
- **'p'**: Pause/Resume processing
- **ESC**: Close preview window (processing continues)

### Window Features
- Automatic window resizing for optimal display
- Real-time progress updates
- Performance monitoring overlay

## Model Configuration

### Automatic Model Loading
The script automatically detects and loads:
- Best checkpoint (`ckpt_best_*.pth`) or fallback to latest (`ckpt.pth`)
- WideResNet50 backbone with layer2+layer3 feature extraction
- Discriminator and pre-projection networks
- Training parameters (dimensions, patch size, etc.)

### Supported Model Types
- MVTec AD trained models
- VisA trained models
- WFDD trained models
- MPDD trained models
- Custom dataset models

## Performance Optimization

### GPU Memory Management
- Automatic device detection (CUDA → CPU fallback)
- Efficient batch processing (single frame inference)
- Memory-optimized preprocessing pipeline

### Processing Speed
- Typical performance: 10-30 FPS on modern GPUs
- CPU fallback: 1-5 FPS depending on hardware
- Real-time processing possible for most use cases

### Optimization Tips
- Use appropriate `--image_size` matching training resolution
- Enable GPU acceleration with `--device cuda:0`
- Disable live display (`--no_display`) for maximum throughput
- Process shorter video segments for testing

## Output Statistics

### Processing Summary
The script provides comprehensive statistics:

```
Processing complete!
Output saved to: output/results.mp4
Total frames processed: 1500
Processing time: 75.32 seconds
Average processing FPS: 19.92
Original video FPS: 30.0
Speed ratio: 0.66x slower than real-time
Anomalous frames detected: 245
Anomaly percentage: 16.33%
Score range: 0.1234 - 0.9876
Mean score: 0.4567 ± 0.2345
```

### Key Metrics
- **Total Frames**: Number of frames processed
- **Processing Time**: Total execution time
- **Processing FPS**: Actual processing speed
- **Speed Ratio**: Comparison to real-time playback
- **Anomaly Count**: Frames exceeding threshold
- **Anomaly Percentage**: Detection rate
- **Score Statistics**: Distribution of anomaly scores

## Troubleshooting

### Common Issues

#### Model Loading Errors
```
Warning: No checkpoint found for <class_name>
```
**Solution**: Verify model path and class name match your trained model directory structure.

#### Video Loading Errors
```
Could not open video: <video_path>
```
**Solution**: Check video file exists and format is supported (MP4, AVI, MOV recommended).

#### GPU Memory Issues
```
CUDA out of memory
```
**Solution**: Use smaller `--image_size` or switch to CPU with `--device cpu`.

#### Performance Issues
- **Slow Processing**: Enable GPU acceleration, reduce image size, disable live display
- **High Memory Usage**: Process shorter video segments, reduce batch size
- **Poor Detection**: Adjust threshold, verify model compatibility with input content

### Model Compatibility
- Ensure video content matches training dataset domain
- Use appropriate threshold for your specific use case
- Consider model fine-tuning for new video types

## File Structure

### Input Requirements
```
video_file.mp4                    # Input video
results/models/backbone_0/        # Model directory
├── class_name/
│   ├── ckpt_best_*.pth          # Best checkpoint (preferred)
│   └── ckpt.pth                 # Latest checkpoint (fallback)
```

### Output Generation
```
output_video.mp4                  # Side-by-side result video
```

## Advanced Usage

### Batch Processing Script Example
```bash
#!/bin/bash
MODEL_PATH="results/models/backbone_0/"
INPUT_DIR="input_videos/"
OUTPUT_DIR="output_videos/"

for video in "$INPUT_DIR"*.mp4; do
    filename=$(basename "$video" .mp4)
    python video_inference_sidebyside.py \
        --model_path "$MODEL_PATH" \
        --class_name wfdd_yellow_cloth \
        --video_path "$video" \
        --output_path "$OUTPUT_DIR/${filename}_analyzed.mp4" \
        --threshold 0.85 \
        --no_display
done
```

### Integration with Other Tools
- **FFmpeg**: Pre/post-process videos for format conversion
- **OpenCV**: Additional video analysis and processing
- **Matplotlib**: Plot score distributions and statistics
- **TensorBoard**: Log detection results for analysis

## Performance Benchmarks

### Typical Processing Speeds
- **RTX 3080**: ~25-30 FPS (384x384 input)
- **RTX 2060**: ~15-20 FPS (384x384 input)
- **CPU (i7-8700K)**: ~2-4 FPS (384x384 input)

### Memory Requirements
- **GPU Memory**: 4-8GB VRAM recommended
- **System RAM**: 8GB minimum, 16GB recommended
- **Storage**: ~2x original video size for output

## Best Practices

1. **Model Selection**: Use models trained on similar data domains
2. **Threshold Tuning**: Start with default 0.9, adjust based on results
3. **Quality Assessment**: Monitor FPS and score distributions
4. **Batch Processing**: Use `--no_display` for automated workflows
5. **Result Validation**: Review output videos for detection accuracy

## Support and Extensions

### Customization Options
- Modify colormap and visualization styles in `create_annotated_frame()`
- Adjust preprocessing pipeline in `preprocess_frame()`
- Add custom metrics and logging
- Implement additional output formats

### Integration Points
- Export detection results to JSON/CSV
- Real-time streaming capabilities
- Multi-camera processing support
- Web-based visualization interface
# Frame Extraction Guide

This guide covers how to extract frames from videos using the `extract_frames.py` script for GLASS dataset preparation and analysis.

## Overview

The frame extraction script (`extract_frames.py`) provides a simple and efficient way to extract frames from video files at specified intervals. This is useful for:

- Creating datasets from video sources
- Sampling video content for analysis
- Preparing frames for GLASS model training
- Quality inspection workflows

## Quick Start

```bash
# Single video - extract every 30th frame
python preprocessing/extract_frames.py input_video.mp4 output_frames/

# Multiple videos - explicit paths
python preprocessing/extract_frames.py video1.mp4 video2.mp4 video3.mp4 output_frames/

# Multiple videos - glob pattern
python preprocessing/extract_frames.py "videos/*.mp4" output_frames/

# Multiple videos with separate folders
python preprocessing/extract_frames.py video1.mp4 video2.mp4 output_frames/ --separate-folders
```

## Command Line Options

### Required Arguments
- `video_paths`: Path(s) to input video file(s) - supports multiple files and glob patterns
- `output_dir`: Directory where extracted frames will be saved

### Optional Arguments
- `--frequency`, `-f`: Extract every Nth frame (default: 30)
- `--start-frame`, `-s`: Frame number to start extraction from (default: 0)
- `--max-frames`, `-m`: Maximum number of frames to extract per video (default: no limit)
- `--separate-folders`: Create separate output folder for each video

## Usage Examples

### Basic Frame Extraction
```bash
# Extract every 30th frame from entire video
python preprocessing/extract_frames.py my_video.mp4 extracted_frames/
```

### High-Frequency Sampling
```bash
# Extract every 5th frame for detailed analysis
python preprocessing/extract_frames.py inspection_video.mp4 detailed_frames/ --frequency 5
```

### Targeted Extraction
```bash
# Start from 10 seconds into video (assuming 30fps = frame 300)
python preprocessing/extract_frames.py fabric_video.mp4 analysis_frames/ --start-frame 300 --frequency 15
```

### Limited Extraction
```bash
# Extract maximum 50 frames for quick preview
python preprocessing/extract_frames.py test_video.mp4 preview_frames/ --frequency 20 --max-frames 50
```

## Multiple Video Processing

The script now supports processing multiple videos in a single command, with options for organization and naming.

### Multiple Video Examples
```bash
# Process multiple videos explicitly
python preprocessing/extract_frames.py video1.mp4 video2.mp4 video3.mp4 output_frames/

# Use glob patterns to process all MP4 files
python preprocessing/extract_frames.py "*.mp4" output_frames/

# Process all videos in a directory
python preprocessing/extract_frames.py "videos/*.mp4" output_frames/

# Mixed patterns and explicit files
python preprocessing/extract_frames.py video1.mp4 "batch/*.mp4" video_final.mp4 output_frames/
```

### Output Organization Options

#### Single Output Directory (Default)
All frames from all videos go into one directory with video name prefixes:
```bash
python preprocessing/extract_frames.py video1.mp4 video2.mp4 output_frames/
```
Output:
```
output_frames/
├── video1_frame_000000.jpg
├── video1_frame_000030.jpg
├── video2_frame_000000.jpg
├── video2_frame_000030.jpg
└── ...
```

#### Separate Folders per Video
Each video gets its own subdirectory:
```bash
python preprocessing/extract_frames.py video1.mp4 video2.mp4 output_frames/ --separate-folders
```
Output:
```
output_frames/
├── video1/
│   ├── frame_000000.jpg
│   └── frame_000030.jpg
├── video2/
│   ├── frame_000000.jpg
│   └── frame_000030.jpg
└── ...
```

### Multiple Video Processing Summary
When processing multiple videos, you'll see a comprehensive summary:
```
🎭 Processing 3 videos...
📁 Base output directory: output_frames/
📂 Separate folders: false

============================================================
🎬 Processing video 1/3: video1.mp4
============================================================
📹 Video Info: video1.mp4
   Resolution: 1920x1080
   FPS: 30.00
   [... processing ...]

============================================================
📊 PROCESSING SUMMARY
============================================================
✅ Successful: 3/3 videos
❌ Failed: 0/3 videos
📸 Total frames extracted: 180
```

### Batch Processing Legacy Example
```bash
# Traditional bash loop approach (still works)
for video in *.mp4; do
    echo "Processing $video..."
    python preprocessing/extract_frames.py "$video" "frames_${video%.*}/" --frequency 30
done
```

## Output Format

### File Naming Convention
Extracted frames are saved with zero-padded frame numbers:
- `frame_000000.jpg` (frame 0)
- `frame_000030.jpg` (frame 30, if frequency=30)
- `frame_000060.jpg` (frame 60, if frequency=30)

### Directory Structure
```
output_frames/
├── frame_000000.jpg
├── frame_000030.jpg
├── frame_000060.jpg
├── frame_000090.jpg
└── ...
```

## Video Information Display

The script displays comprehensive video information before extraction:

```
📹 Video Info:
   Resolution: 1920x1080
   FPS: 30.00
   Total frames: 1800
   Duration: 60.00 seconds
   Save frequency: every 30 frames
   Start from frame: 0

🎬 Extracting frames to: output_frames/
   Extracted 10 frames...
   Extracted 20 frames...

✅ Frame extraction completed!
   Total frames processed: 1800
   Frames saved: 60
   Output directory: output_frames/
   Time span covered: 60.00 seconds
   Extraction ratio: 3.33%
```

## Integration with GLASS Workflow

### For Dataset Creation
```bash
# Extract frames for training data
python preprocessing/extract_frames.py fabric_video.mp4 datasets/custom/my_fabric/train/good/ --frequency 30

# Extract frames for testing
python preprocessing/extract_frames.py test_fabric_video.mp4 datasets/custom/my_fabric/test/good/ --frequency 15
```

### For Inference Preparation
```bash
# Extract frames for batch inference
python preprocessing/extract_frames.py inspection_video.mp4 inference_frames/ --frequency 10

# Then run GLASS inference on extracted frames
python main.py --test test net -b wideresnet50 -le layer2 -le layer3 dataset -d my_fabric custom datasets/custom datasets/dtd/images
```

### Quality Control Sampling
```bash
# Extract sparse samples for quality review
python preprocessing/extract_frames.py production_video.mp4 quality_samples/ --frequency 100 --max-frames 20
```

## Performance Considerations

### Frame Rate vs Extraction Frequency
- **High FPS videos (60fps)**: Use higher frequency (60-120) to avoid redundant frames
- **Standard FPS videos (30fps)**: Default frequency (30) extracts ~1 frame per second
- **Low FPS videos (15fps)**: Use lower frequency (5-15) for better coverage

### Storage Requirements
```bash
# Estimate storage needs
# Formula: (video_frames / frequency) * ~500KB per frame
# Example: 3600 frames / 30 = 120 frames × 500KB = ~60MB
```

### Memory Usage
The script processes frames one at a time, so memory usage is minimal regardless of video size.

## Error Handling

### Common Issues and Solutions

**Video Not Found**
```bash
❌ File Error: Video file not found: nonexistent_video.mp4
```
→ Check file path and ensure video file exists

**Video Cannot Be Opened**
```bash
❌ Video Error: Could not open video: corrupted_video.mp4
```
→ Verify video file integrity and format compatibility

**No Frames Extracted**
```bash
⚠️ No frames were extracted. Check your input parameters.
```
→ Check start-frame and frequency settings against video length

### Supported Video Formats
The script supports all formats that OpenCV can read:
- MP4, AVI, MOV, WMV, FLV
- H.264, H.265, VP9 codecs
- Most common video formats

## Advanced Usage

### Custom Frame Selection
```python
# For programmatic use
from extract_frames import extract_frames

# Extract specific frame ranges
saved_count = extract_frames(
    video_path="my_video.mp4",
    output_dir="custom_frames/",
    save_frequency=15,
    start_frame=1000,
    max_frames=50
)
```

### Integration with Other Scripts
```bash
# Chain with video creation
python preprocessing/create_enhanced_video.py --dataset custom --class_name grid
python preprocessing/extract_frames.py test-video/custom/grid/grid.mp4 verification_frames/ --frequency 20

# Chain with inference
python preprocessing/extract_frames.py input.mp4 temp_frames/ --frequency 30
python inference/video_inference.py --model_path results/models/backbone_0/ --class_name grid --video_path input.mp4
```

## Tips and Best Practices

### Choosing Frame Frequency
- **Analysis/Training**: 15-30 frames (1-2 seconds apart)
- **Quality Control**: 60-120 frames (2-4 seconds apart)  
- **Preview/Thumbnails**: 300+ frames (10+ seconds apart)
- **Detailed Inspection**: 5-10 frames (0.2-0.5 seconds apart)

### Organizing Output
```bash
# Create organized structure
mkdir -p frames/{train,test,validation}
python preprocessing/extract_frames.py train_video.mp4 frames/train/ --frequency 30
python preprocessing/extract_frames.py test_video.mp4 frames/test/ --frequency 20
python preprocessing/extract_frames.py val_video.mp4 frames/validation/ --frequency 25
```

### Batch Processing
```bash
# Process all videos in directory
find . -name "*.mp4" -exec python preprocessing/extract_frames.py {} frames_{} --frequency 30 \;
```

## Related Tools

- **Video Creation**: `create_enhanced_video.py` - Create videos from image datasets
- **Video Inference**: `inference/video_inference.py` - Run GLASS inference on videos
- **Training Frame Extraction**: `test-video/extract_training_frames.py` - Specialized extraction with ground truth

## Troubleshooting

### Performance Issues
- Large videos: Use higher frequency values to reduce output
- Slow extraction: Check available disk space and I/O performance
- Memory issues: Script uses minimal memory, check system resources

### Quality Issues
- Blurry frames: Check if video has motion blur or low resolution
- Missing frames: Verify start-frame and frequency parameters
- Format issues: Convert video to MP4 H.264 for best compatibility
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GLASS (Global and Local Anomaly co-Synthesis Strategy) is a unified framework for unsupervised industrial anomaly detection and localization presented at ECCV 2024. It addresses limitations in existing anomaly synthesis strategies, particularly for weak defects that resemble normal regions, using gradient ascent guided Gaussian noise and truncated projection.

**Key Innovation**: Synthesizes near-in-distribution anomalies in a controllable way using Global Anomaly Synthesis (GAS) at feature level and Local Anomaly Synthesis (LAS) at image level.

**Performance Results**:
- MVTec AD: I-AUROC 99.9%, P-AUROC 99.3%
- VisA: I-AUROC 98.8%, P-AUROC 98.8%  
- MPDD: I-AUROC 99.6%, P-AUROC 99.4%
- WFDD: I-AUROC 100%, P-AUROC 98.9%

## Environment Setup

```bash
conda create -n glass_env python=3.9.15
conda activate glass_env
pip install -r requirements.txt
```

## Common Commands

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Check environment compatibility
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

### Training and Testing
Use shell scripts in the `shell/` directory for different datasets:

```bash
# MVTec AD dataset
bash shell/run-mvtec.sh

# VisA dataset  
bash shell/run-visa.sh

# MPDD dataset
bash shell/run-mpdd.sh

# WFDD dataset
bash shell/run-wfdd.sh

# MAD-man dataset (weak defects)
bash shell/run-mad-man.sh

# MAD-sys dataset (synthetic defects)  
bash shell/run-mad-sys.sh

# Custom dataset
bash shell/run-custom.sh

# Custom training (single class)
bash shell/run-custom-training.sh
```

### Manual Execution
Run the main script directly with Click CLI:

```bash
python main.py \
    --gpu 0 \
    --seed 0 \
    --test ckpt \
  net \
    -b wideresnet50 \
    -le layer2 \
    -le layer3 \
    --pretrain_embed_dimension 1536 \
    --target_embed_dimension 1536 \
    --meta_epochs 640 \
  dataset \
    --batch_size 8 \
    --resize 288 \
    --imagesize 288 \
    -d classname \
    mvtec /path/to/data /path/to/dtd
```

### Key Parameters
- `--test`: Set to 'ckpt' for training+testing, 'test' for testing only
- `--gpu`: GPU device ID
- `--meta_epochs`: Training epochs (default: 640)
- `--batch_size`: Batch size for training/testing
- `-b`: Backbone network (e.g., wideresnet50)
- `-le`: Layers to extract features from
- `-d`: Dataset class names

### ONNX Export
Convert PyTorch models to ONNX format:

```bash
python onnx/pth2onnx.py
python onnx/ort.py  # Run ONNX inference
```

### Video Inference

#### 🤖 Automatic Model Selection (NEW - Recommended)
The **Inference Orchestrator** automatically selects the best model for your video:

```bash
# Automatic model selection and inference (simplest approach)
python inference/inference_orchestrator.py \
    --video_path input_video.mp4

# List all available trained models  
python inference/inference_orchestrator.py --list_models

# Custom evaluation settings
python inference/inference_orchestrator.py \
    --video_path input_video.mp4 \
    --sample_frames 50 \
    --device cuda:0
```

**How it works**: Tests all available models on sample frames, selects the one with lowest average anomaly score (best fit), then runs full inference automatically.

#### 🎯 Advanced Defect Tracking (NEW)
Enterprise-grade defect tracking with temporal persistence:

```bash
# Defect tracking with organized output (recommended)
python inference/video_inference_with_tracking.py \
    --model_path results/models/backbone_0/ \
    --class_name mvtec_grid-test \
    --video_path input_video.mp4

# Manual output paths (if needed)
python inference/video_inference_with_tracking.py \
    --model_path results/models/backbone_0/ \
    --class_name mvtec_grid-test \
    --video_path input_video.mp4 \
    --output_path output.mp4 \
    --no_organized_output
```

**Features**: Tracks defects across frames, saves one representative frame per defect, provides comprehensive tracking reports with physical measurements.

**Organized Output Structure**:
```
output/[class-name]/[timestamp]/
├── output-video/          # Inference videos
├── defects/              # Individual defect frames with track IDs
└── report/               # JSON tracking and selection reports
```

#### Traditional Video Inference
Run inference on video files using trained GLASS models:

```bash
# Side-by-side video inference (recommended)
python inference/video_inference_sidebyside.py \
    --model_path results/models/backbone_0/ \
    --class_name mvtec_grid \
    --video_path input.mp4 \
    --output_path output.mp4 \
    --threshold 0.8

# Basic video inference
python inference/video_inference.py \
    --model_path results/models/backbone_0/ \
    --class_name wfdd_yellow_cloth \
    --video_path input.mp4 \
    --output_path output.mp4

# Video inference with defect size analysis
python inference/video_inference_with_size.py \
    --model_path results/models/backbone_0/ \
    --class_name wfdd_pink_flower \
    --video_path input.mp4 \
    --output_path output.mp4 \
    --pixel_size 0.1 \
    --physical_unit mm

# Clean inference without text overlays
python inference/video_inference_clean.py \
    --model_path results/models/backbone_0/ \
    --class_name mvtec_grid \
    --video_path input.mp4 \
    --output_path output.mp4 \
    --intensity 0.6

# Easy wrapper script
python inference/run_video_inference.py \
    input.mp4 output.mp4 \
    --class_name wfdd_yellow_cloth
```

### Video Creation from Datasets

#### **Enhanced Video Creation with Defect Distribution (New)**
Create videos with defects distributed equally throughout duration and save good frames as images:

```bash
# Create video with 30% defects distributed throughout entire duration
python preprocessing/create_enhanced_video.py --dataset custom --class_name grid --video_name grid_test

# Create video with 50% defects and enhanced images
python preprocessing/create_enhanced_video.py --dataset custom --class_name grid --defect_ratio 0.5 --enhance

# Create video with specific frame count and save good frames
python preprocessing/create_enhanced_video.py --dataset wfdd --class_name yellow_cloth --total_frames 300 --save_good_frames

# Create video without saving individual frames
python preprocessing/create_enhanced_video.py --dataset custom --class_name grid --no_save_frames
```

**Enhanced Features**:
- **Defect Distribution**: Defects spread equally throughout video duration (not just at beginning)
- **Good Frame Extraction**: Saves only good/normal frames as individual images in `test-video/[video-name]/good_frames/`
- **Controllable Ratios**: Specify exact defect-to-good frame ratios (default: 30% defects)
- **Reproducible**: Uses seed for consistent defect distribution patterns

**Output Structure**: `test-video/{video-name}/{video-name}.mp4` + `good_frames/frame_*.jpg`

#### **Enhanced Dataset Video Creation (Standard)**
Create organized videos from test datasets with preprocessing options:

```bash
# Create combined video for all test types (organized storage)
python preprocessing/create_dataset_video.py --dataset custom --class_name grid

# Create separate videos for each defect type
python preprocessing/create_dataset_video.py --dataset custom --class_name grid --separate

# Create enhanced videos with image improvements
python preprocessing/create_dataset_video.py --dataset wfdd --class_name yellow_cloth --enhance

# Create videos from specific test types only
python preprocessing/create_dataset_video.py --dataset custom --class_name grid --test_types good hole spot

# Create videos with largest dimensions (no padding)
python preprocessing/create_dataset_video.py --dataset custom --class_name grid --resize_method largest

# List available test structure
python preprocessing/create_dataset_video.py --dataset custom --class_name grid --list_structure
```

**Output Organization**: `test-video/{dataset-name}/{class-name}/`

#### **Basic Video Creation (Legacy)**
```bash
# Create video from all test images (includes all defect types)
python preprocessing/create_video.py grid \
    --input_path datasets/custom/grid/test \
    --output preprocessing/grid_test_video.mp4

# Create video from specific defect type
python preprocessing/create_video.py grid \
    --defect hole \
    --output preprocessing/grid_hole_video.mp4
```

**Video Inference Features**:
- Real-time anomaly detection and scoring
- Side-by-side comparison with original frames
- Defect area analysis and physical measurements
- Live preview during processing
- Comprehensive performance metrics
- Support for all trained model classes

### Defect Tracking with Temporal Analysis
Advanced video inference with defect tracking across multiple frames for continuous fabric inspection:

```bash
# Defect tracking inference (recommended for continuous inspection)
python inference/video_inference_with_tracking.py \
    --model_path results/models/backbone_0/ \
    --class_name mvtec_grid-visible \
    --video_path input_video.mp4 \
    --output_path output/tracked_results.mp4 \
    --pixel_size 0.1 \
    --fabric_speed 5.0 \
    --device cuda:0

# Defect tracking with individual frame saving (NEW)
python inference/video_inference_with_tracking.py \
    --model_path results/models/backbone_0/ \
    --class_name mvtec_grid-visible \
    --video_path input_video.mp4 \
    --output_path output/tracked_results.mp4 \
    --pixel_size 0.1 \
    --fabric_speed 5.0 \
    --save_defect_frames \
    --defect_frames_dir output/defect_frames \
    --device cuda:0

# Test defect tracking system
python test_defect_tracking.py --all
```

**Defect Tracking Features**:
- **Temporal Tracking**: Track the same defect across multiple consecutive frames
- **Motion Compensation**: Fabric movement estimation and correction using optical flow
- **Physical Measurements**: Convert pixel measurements to physical units (mm, cm, inch)
- **ByteTrack Integration**: Advanced multi-object tracking adapted for fabric defects
- **Individual Frame Extraction**: Save annotated frames for each tracked defect with track ID, confidence, and area information
- **Comprehensive Reporting**: JSON reports with defect lifecycle and statistics
- **Real-time Performance**: ~17-18 FPS processing speed on modern GPUs
- **Production Ready**: Handles long videos (1000+ frames) with robust error handling

**Tracking System Architecture**:
- **FabricByteTracker**: Core tracking algorithm with Kalman filtering and motion prediction
- **GLASSDefectTracker**: Integration layer between GLASS detection and tracking
- **FabricMotionEstimator**: Multiple motion estimation methods (optical flow, template matching)
- **TrackedDefectSummary**: Comprehensive defect lifecycle analysis

**Example Tracking Results**:
- Successfully tracked defects across 50+ consecutive frames
- Motion compensation for fabric speeds up to 79 pixels/frame
- Physical defect area measurements (tested: 45.9 mm² to 11,494 mm²)
- Comprehensive tracking reports with temporal statistics

### Defect Size Analysis
Quantitative analysis of detected anomalies with physical measurements:

```bash
# Test size analyzer functionality
python size_analyzer/test_defect_size_analysis.py

# Direct usage in Python
from size_analyzer import DefectSizeAnalyzer, DefectMetrics

# Initialize with physical calibration
analyzer = DefectSizeAnalyzer(
    pixel_size=0.1,          # 0.1mm per pixel
    physical_unit="mm",      # Units: mm, cm, inch
    min_defect_area=10       # Minimum size threshold
)

# Analyze defects in anomaly mask
metrics = analyzer.analyze_defects(
    anomaly_mask,            # GLASS output mask
    threshold=0.5,           # Detection threshold
    use_morphology=True      # Noise filtering
)

# Access measurements
print(f"Found {metrics.num_defects} defects")
print(f"Total area: {metrics.total_defect_area_physical:.2f} {metrics.physical_unit}²")
print(f"Largest defect: {max(metrics.defect_areas_physical):.2f} {metrics.physical_unit}²")
```

**Size Analysis Features**:
- Physical size measurements (mm, cm, inch)
- Statistical analysis (mean, median, std deviation)
- Severity classification (low, medium, high, critical)
- Batch processing for video analysis
- Comprehensive visualizations with annotations
- Export capabilities (JSON, CSV, images)

## Architecture Overview

### Core Components

1. **main.py**: Entry point with Click CLI interface
   - `net` command: Configures GLASS model parameters
   - `dataset` command: Configures data loading and preprocessing

2. **glass.py**: Main GLASS class implementing the anomaly detection pipeline
   - `GLASS.trainer()`: Training loop with discriminator and projection training
   - `GLASS._train_discriminator()`: Discriminator training with gradient ascent
   - `GLASS.predict()`: Inference for anomaly scoring and segmentation
   - `GLASS._embed()`: Feature embedding extraction from backbone

3. **model.py**: Neural network components
   - `Discriminator`: Binary classifier for real/fake features
   - `Projection`: Feature projection layers
   - `PatchMaker`: Patch extraction and scoring utilities

4. **backbones.py**: Feature extraction backbones (ResNet variants, etc.)

5. **common.py**: Shared utilities for feature aggregation and preprocessing

### Data Flow

1. **Feature Extraction**: Backbone networks extract multi-scale features
2. **Patch Processing**: Features converted to patches using PatchMaker
3. **Anomaly Synthesis**: Gradient ascent generates synthetic anomalies from normal features
4. **Discriminator Training**: Binary classifier trained on real vs synthetic features
5. **Inference**: Discriminator scores patches for anomaly detection/localization

### Key Algorithms

- **Gradient Ascent Synthesis**: Iteratively optimizes features to maximize anomaly likelihood
- **Distribution-Aware Training**: Adapts to manifold vs hypersphere data distributions
- **Mining Strategy**: Hard negative mining for improved discriminator training
- **Projection Networks**: Optional feature space transformation

## Dataset Structure

Datasets should follow this structure:
```
dataset/
├── classname/
│   ├── train/
│   │   └── good/
│   └── test/
│       ├── good/
│       └── defect_type/
└── ground_truth/
    └── defect_type/
```

### Supported Datasets
- **MVTec AD**: Industrial anomaly detection benchmark (15 categories)
- **VisA**: Vision Anomaly dataset  
- **MPDD**: Magnetic Particle Defect Detection
- **WFDD**: Woven Fabric Defect Detection (4,101 images, 4 categories with pixel-level annotations)
- **DTD**: Describable Textures Dataset (for augmentation)

### Custom Datasets Released by GLASS Authors
1. **WFDD**: 4,101 woven fabric images across 4 categories (grey cloth, grid cloth, yellow cloth, pink flower)
2. **MAD-man**: MVTec AD-manual test set for weak defect evaluation (samples selected by 5 individuals)
3. **MAD-sys**: MVTec AD-synthesis test set with varying defect transparency levels (4 subsets with 320 normal + 946 anomaly samples each)
4. **Foreground Masks**: Binary masks for normal samples from various datasets

## Project Structure

### Directory Organization
```
GLASS-new/
├── datasets/            # Training and test data
│   ├── custom/         # Custom datasets (grid, etc.)
│   ├── WFDD/          # WFDD dataset
│   └── dtd/           # DTD texture dataset for augmentation
├── inference/          # Video inference scripts
│   ├── inference_orchestrator.py           # 🤖 NEW: Automatic model selection (recommended)
│   ├── video_inference_with_tracking.py    # 🎯 NEW: Defect tracking inference
│   ├── video_inference_sidebyside.py       # Side-by-side inference  
│   ├── video_inference_clean.py            # Clean inference without overlays
│   ├── video_inference.py                  # Basic video inference
│   ├── video_inference_with_size.py        # Inference with size analysis
│   ├── run_video_inference.py              # Easy wrapper script
│   └── README.md                           # Inference documentation
├── defect_tracking/    # Temporal defect tracking system
│   ├── __init__.py                    # Module exports
│   ├── fabric_bytetrack.py            # ByteTrack tracking implementation
│   ├── glass_integration.py           # GLASS-tracking integration layer
│   ├── motion_estimation.py           # Fabric motion estimation
│   └── README.md                      # Tracking system documentation
├── preprocessing/      # Data preprocessing and video creation tools
│   ├── create_dataset_video.py      # Enhanced dataset video creation (recommended)
│   ├── create_video.py               # Basic video creation from datasets
│   ├── preprocess_images.py         # Image preprocessing for dataset preparation
│   ├── defect_simulator.py          # Synthetic defect generation
│   └── README.md                    # Preprocessing documentation
├── size_analyzer/      # Defect size analysis and measurement tools
│   ├── defect_size_analyzer.py      # Main analyzer class
│   ├── test_defect_size_analysis.py # Testing and validation
│   ├── DEFECT_SIZE_ANALYSIS_GUIDE.md # Detailed usage guide
│   └── README.md                    # Size analyzer documentation
├── results/            # Training outputs and model checkpoints
│   ├── training/classname/  # Training visualizations
│   ├── eval/classname/      # Evaluation results
│   ├── judge/avg/           # Distribution analysis
│   └── models/              # Saved checkpoints
├── test-video/         # Organized dataset videos by dataset/class structure
│   ├── custom/        # Custom dataset videos
│   ├── wfdd/          # WFDD dataset videos
│   └── mvtec/         # MVTec dataset videos
├── shell/              # Training scripts for different datasets
├── onnx/              # ONNX export utilities
└── output/            # Video inference outputs
```

### Results and Outputs

### Checkpoints
- `ckpt.pth`: Latest training checkpoint
- `ckpt_best_X.pth`: Best performing checkpoint at epoch X

### Performance Tracking
- TensorBoard logs in `tb/` directories
- CSV results with AUROC, AP, and PRO metrics
- Excel files tracking distribution analysis

## Configuration Notes

- **GPU Memory**: Requires ~8GB VRAM for default settings
- **Backbone Networks**: wideresnet50 recommended for best performance
- **Feature Layers**: layer2+layer3 combination works well for most datasets
- **Hyperparameters**: Default values optimized for MVTec AD
- **Distribution Detection**: Automatic analysis determines manifold vs hypersphere training

## Development Workflow

### Quick Development Setup
```bash
# 1. Environment setup and verification
conda create -n glass_env python=3.9.15
conda activate glass_env
pip install -r requirements.txt
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# 2. Quick model test (single epoch)
python main.py --test ckpt --meta_epochs 1 net -b wideresnet50 -le layer2 -le layer3 dataset -d grid custom datasets/custom datasets/dtd/images

# 3. Full training pipeline
bash shell/run-custom.sh
```

### Debugging Common Issues

**CUDA/GPU Issues**:
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"

# Monitor GPU memory during training
nvidia-smi -l 1
```

**Dataset Path Issues**:
- Verify dataset structure matches expected format in `Dataset Structure` section
- Check that `datapath` and `augpath` in shell scripts point to correct directories
- For custom datasets, ensure `dataset_info.json` exists in class directories

**Training Convergence Issues**:
- Monitor TensorBoard logs in `results/training/[class]/tb/` 
- Check distribution analysis in `results/judge/avg/` Excel files
- Adjust learning rate (`--lr`) and meta epochs (`--meta_epochs`) for difficult datasets

**Memory Issues**:
- Reduce batch size (`--batch_size`) from 8 to 4 or 2
- Use gradient checkpointing if available
- Monitor system memory usage alongside GPU memory

### Performance Optimization

**Training Speed**:
- Use multiple GPUs: `--gpu 0 1 2 3`
- Increase batch size up to memory limits
- Pre-process datasets to standard sizes to reduce runtime resizing

**Inference Speed**:
- Use ONNX export for deployment: `python onnx/pth2onnx.py`
- Batch process multiple videos
- Use lower resolution inputs for real-time applications

## Testing and Validation

### Running Tests
```bash
# Basic model training test (single epoch)
python main.py --test ckpt --meta_epochs 1 net -b wideresnet50 -le layer2 -le layer3 dataset -d grid custom /path/to/data /path/to/dtd

# Quick inference test
python main.py --test test net -b wideresnet50 -le layer2 -le layer3 dataset -d grid custom /path/to/data /path/to/dtd
```

This codebase includes several validation approaches:
- Cross-validation during training (`eval_epochs` parameter)
- Benchmark evaluation against standard datasets
- Visual inspection of generated anomaly maps
- Size analyzer validation: `python size_analyzer/test_defect_size_analysis.py`
- Defect tracking validation: `python test_defect_tracking.py --all`

### Enhanced Features Testing
Test the new enhanced video creation and clean inference features:

```bash
# Test enhanced video creation and clean inference
python test_enhanced_features.py

# Test individual components
python preprocessing/create_enhanced_video.py --dataset custom --class_name grid --video_name test_grid --total_frames 20
python inference/video_inference_clean.py --model_path results/models/backbone_0 --class_name grid --video_path test-video/test_grid/test_grid.mp4 --output_path output/clean_test.mp4 --no_display
```

**Enhanced Features Validated**:
- Defect distribution throughout entire video duration (not clustered at beginning)
- Good frame extraction as individual images for reference
- Clean inference output without text overlays
- Controllable defect-to-good ratios with reproducible seeded distribution

### Defect Tracking System Tests
```bash
# Test all tracking components
python test_defect_tracking.py --all

# Test individual components  
python test_defect_tracking.py --test_components

# Test with existing videos
python test_defect_tracking.py --test_videos

# Create synthetic test data
python test_defect_tracking.py --create_synthetic
```

**Validated Performance**:
- Processed 2,798 frames in 157 seconds (17.8 FPS)
- Successfully tracked 4 unique defects with temporal consistency
- Motion compensation tested up to 79 pixels/frame fabric speed
- Physical measurements validated: 45.9 mm² to 11,494 mm² defect areas

### Single Class Training/Testing
For testing individual classes or during development:
```bash
# Train/test single class (modify classes array in shell script)
bash shell/run-custom-training.sh  # For custom datasets
# Or run directly:
python main.py --test ckpt net -b wideresnet50 -le layer2 -le layer3 dataset -d classname mvtec /path/to/data /path/to/dtd
```

## File Modifications

When modifying shell scripts, update dataset paths:
- `datapath`: Path to main dataset
- `augpath`: Path to DTD texture dataset for augmentation
- `classes`: Array of dataset class names to process

## Citation and References

**Paper**: "A Unified Anomaly Synthesis Strategy with Gradient Ascent for Industrial Anomaly Detection and Localization"
- **Conference**: ECCV 2024 (European Conference on Computer Vision)
- **Authors**: Qiyu Chen, Huiyuan Luo, Chengkan Lv, Zhengtao Zhang
- **ArXiv**: https://arxiv.org/abs/2407.09359
- **DOI**: https://link.springer.com/chapter/10.1007/978-3-031-72855-6_3

```bibtex
@inproceedings{chen2024unified,
  title={A unified anomaly synthesis strategy with gradient ascent for industrial anomaly detection and localization},
  author={Chen, Qiyu and Luo, Huiyuan and Lv, Chengkan and Zhang, Zhengtao},
  booktitle={European Conference on Computer Vision},
  pages={37--54},
  year={2024},
  organization={Springer}
}
```

**License**: MIT License

**Acknowledgments**: Based on inspiration from SimpleNet (https://github.com/DonaldRR/SimpleNet/)
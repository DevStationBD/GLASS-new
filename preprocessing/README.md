# Fabric Defect Dataset Creation Pipeline

This directory contains scripts for creating WFDD-compatible fabric defect datasets from good fabric images.

## Overview

The pipeline transforms raw fabric images into a complete anomaly detection dataset with:
- **Good images** split into train/test sets
- **Defective images** with simulated fabric defects
- **Ground truth masks** for pixel-level defect localization

## Supported Defect Types

1. **Hole** - Circular/irregular missing fabric areas
2. **Foreign Yarn** - Different colored threads/yarns
3. **Missing Yarn** - Linear gaps where threads should be
4. **Slab** - Thick yarn sections with uneven texture  
5. **Spot** - Stains, contamination marks

## Scripts

### 1. `orchestrator.py` - Main Pipeline Controller
**Primary interface for dataset creation and video generation**

```bash
# Complete pipeline with separate dataset and class names (recommended)
# Full pipeline and video creation are now enabled by default
python preprocessing/orchestrator.py \
    --source raw-data/custom-grid/good-images \
    --dataset custom \
    --class_name grid

# Video creation only from existing dataset (disable pipeline)
python preprocessing/orchestrator.py \
    --source raw-data/custom-grid/good-images \
    --dataset custom \
    --class_name grid \
    --no_pipeline

# Dataset creation only (disable video creation)
python preprocessing/orchestrator.py \
    --source raw-data/custom-grid/good-images \
    --dataset custom \
    --class_name grid \
    --no_videos

# Specific steps only (disables defaults)
python preprocessing/orchestrator.py \
    --source raw-data/custom-grid/good-images \
    --dataset custom \
    --class_name grid \
    --steps structure preprocess defects videos

# Backward compatible (legacy format still works)
python preprocessing/orchestrator.py \
    --source raw-data/custom-grid/good-images \
    --dataset custom_grid
```

**Key Options:**
- `--source`: Directory with good fabric images
- `--dataset`: Dataset name (e.g., custom, wfdd, mvtec)
- `--class_name`: Class name (e.g., grid, yellow_cloth, bottle). If not provided, will be extracted from dataset name for backward compatibility
- `--full_pipeline`: Run complete dataset creation pipeline (default: True)
- `--create_videos`: Create videos from test images (default: True)
- `--no_pipeline`: Disable full pipeline (use with --steps or video-only mode)
- `--no_videos`: Disable video creation (dataset creation only)
- `--steps`: Run specific steps (structure, preprocess, defects, validate, videos)
- `--images_per_defect`: Number of defective images per type (default: 50)
- `--image_size`: Target image size (default: 384)

**Video Configuration Options:**
- `--video_fps`: Video FPS (default: 1 = 1 second per image)
- `--video_resize_method`: Resize method (none, largest, preserve_aspect, crop)
- `--video_enhance`: Apply image enhancement to video frames
- `--video_combined`: Create combined single video (default behavior)
- `--video_separate`: Create separate videos for each defect type instead of combined
- `--video_output`: Video output directory (default: test-video)

### 2. `prepare_dataset_structure.py` - Directory Setup
**Creates WFDD-compatible directory structure**

```bash
python preprocessing/prepare_dataset_structure.py --dataset_name custom_grid
```

Creates:
```
datasets/custom/grid/
├── train/good/
├── test/good/
├── test/{defect_type}/
└── ground_truth/{defect_type}/
```

### 3. `preprocess_images.py` - Image Processing & Splitting
**Preprocesses and splits good images into train/test**

```bash
python preprocessing/preprocess_images.py \
    --source raw-data/custom-grid/good-images \
    --target datasets/custom/grid \
    --image_size 288 \
    --train_ratio 0.8
```

Features:
- Preserves original image dimensions (no resizing or padding)
- Enhances contrast, brightness, sharpness
- Splits into train/test sets
- Saves as PNG format

### 4. `defect_simulator.py` - Defect Generation Engine
**Simulates realistic fabric defects on good images**

```bash
python preprocessing/defect_simulator.py \
    --input datasets/custom/grid/test/good \
    --output datasets/custom/grid/test \
    --defect_types hole foreign_yarn missing_yarn slab spot \
    --images_per_defect 50
```

Defect Details:
- **Hole**: Random circular/irregular black regions
- **Foreign Yarn**: Colored thread overlays with random paths
- **Missing Yarn**: Linear gaps (horizontal/vertical) 
- **Slab**: Thick yarn sections with texture variation
- **Spot**: Irregular stain-like contamination marks

### 5. `mask_utils.py` - Mask Validation & Processing
**Validates and processes defect masks**

```bash
# Validate all masks
python preprocessing/mask_utils.py --validate datasets/custom/grid/ground_truth --plot_stats

# Clean/process masks
python preprocessing/mask_utils.py --clean datasets/custom/grid/ground_truth/hole
```

Validation checks:
- Binary mask format (0 and 255 values)
- Defect area ratios
- Connected components analysis
- Generates statistics plots

## Quick Start

### Step 1: Prepare Your Data
Ensure good fabric images are in the source directory:
```
raw-data/custom-grid/good-images/
├── image001.jpg
├── image002.jpg
└── ...
```

### Step 2: Run Complete Pipeline
```bash
# New simplified format with separate dataset and class names (recommended)
# Full pipeline and video creation are enabled by default
python preprocessing/orchestrator.py \
    --source raw-data/custom-grid/good-images \
    --dataset custom \
    --class_name grid \
    --images_per_defect 50

# Or use legacy format (still supported)
python preprocessing/orchestrator.py \
    --source raw-data/custom-grid/good-images \
    --dataset custom_grid \
    --images_per_defect 50
```

### Step 3: Verify Results
Check the created dataset:
```
datasets/custom/grid/
├── train/good/           # ~200 training images
├── test/good/            # ~50 test images
├── test/hole/            # 50 hole defect images
├── test/foreign_yarn/    # 50 foreign yarn defect images
├── test/missing_yarn/    # 50 missing yarn defect images
├── test/slab/            # 50 slab defect images
├── test/spot/            # 50 spot defect images
└── ground_truth/         # Corresponding masks
    ├── hole/
    ├── foreign_yarn/
    ├── missing_yarn/
    ├── slab/
    └── spot/
```

### Step 4: Use with GLASS
```bash
python main.py \
    --gpu 0 \
    --test ckpt \
  net \
    -b wideresnet50 \
    -le layer2 layer3 \
    --meta_epochs 640 \
  dataset \
    --batch_size 8 \
    --resize 288 \
    --imagesize 288 \
    -d grid \
    datasets/custom/grid /path/to/dtd
```

## Configuration

### Default Parameters
- **Image Processing**: Original dimensions preserved (no resizing)
- **Train/Test Split**: 80/20
- **Images per Defect**: 50
- **Defect Types**: hole, foreign_yarn, missing_yarn, slab, spot

### Customization
Modify parameters in `orchestrator.py`:
```python
self.config = {
    "image_size": 288,
    "train_ratio": 0.8, 
    "images_per_defect": 50,
    "defect_types": ["hole", "foreign_yarn", "missing_yarn", "slab", "spot"],
    "seed": 42
}
```

## Dependencies

Required Python packages:
```
opencv-python
Pillow
numpy
matplotlib
```

Install with:
```bash
pip install opencv-python Pillow numpy matplotlib
```

## Video Creation

The orchestrator can create videos from test datasets for visual inspection and demonstration purposes.

### Video Creation Examples
```bash
# Create videos from existing dataset (no pipeline, videos enabled by default)
python preprocessing/orchestrator.py \
    --source datasets/custom/grid/test/good \
    --dataset custom \
    --class_name grid \
    --no_pipeline

# Create videos with enhanced frames
python preprocessing/orchestrator.py \
    --source datasets/wfdd/yellow_cloth/test/good \
    --dataset wfdd \
    --class_name yellow_cloth \
    --no_pipeline \
    --video_enhance

# Create combined video instead of separate per defect type
python preprocessing/orchestrator.py \
    --source datasets/custom/grid/test/good \
    --dataset custom \
    --class_name grid \
    --no_pipeline \
    --video_combined
```

### Video Output Structure
Videos are organized by dataset and class:
```
test-video/
├── custom/
│   └── grid/
│       ├── good.mp4                 # Normal images
│       ├── hole.mp4                # Hole defect images
│       ├── foreign_yarn.mp4        # Foreign yarn defect images
│       ├── missing_yarn.mp4        # Missing yarn defect images
│       ├── slab.mp4                # Slab defect images
│       ├── spot.mp4                # Spot defect images
│       └── grid_combined.mp4       # All images combined (if --video_combined)
├── wfdd/
│   └── yellow_cloth/
│       └── ...
└── mvtec/
    └── bottle/
        └── ...
```

## Output Files

### Dataset Structure
- **Training Images**: `train/good/*.png`
- **Test Good Images**: `test/good/*.png`
- **Test Defect Images**: `test/{defect_type}/*.png`
- **Ground Truth Masks**: `ground_truth/{defect_type}/*_mask.png`

### Video Files
- **Test Videos**: `test-video/{dataset}/{class}/{defect_type}.mp4`
- **Combined Videos**: `test-video/{dataset}/{class}/{class}_combined.mp4` (if enabled)

### Metadata
- **dataset_info.json**: Complete dataset statistics and configuration
- **mask_statistics.png**: Defect analysis plots (if generated)

## Troubleshooting

### Common Issues

1. **No images found**: Check source directory path and file extensions
2. **Script not found**: Ensure all scripts are in preprocessing/ directory
3. **Permission errors**: Check write permissions for output directory
4. **Memory issues**: Reduce `images_per_defect` or `image_size`

### Validation
Run validation separately:
```bash
python preprocessing/mask_utils.py --validate datasets/custom/grid/ground_truth
```

### Manual Steps
Run pipeline steps individually for debugging:
```bash
# Step 1: Structure
python preprocessing/prepare_dataset_structure.py --dataset_name custom_grid

# Step 2: Preprocessing  
python preprocessing/preprocess_images.py --source raw-data/custom-grid/good-images --target datasets/custom/grid

# Step 3: Defect generation
python preprocessing/defect_simulator.py --input datasets/custom/grid/test/good --output datasets/custom/grid/test

# Step 4: Validation
python preprocessing/mask_utils.py --validate datasets/custom/grid/ground_truth
```

## Advanced Usage

### Custom Defect Types
Modify `defect_simulator.py` to add new defect types:
```python
def simulate_custom_defect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Implement custom defect logic
    defect_image = image.copy()
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    # ... defect simulation code ...
    return defect_image, mask
```

### Batch Processing
Process multiple datasets with separate names:
```bash
# Process multiple fabric types with separate dataset and class names
# Full pipeline and video creation are enabled by default
declare -A datasets=(
    ["custom"]="grid"
    ["wfdd"]="yellow_cloth"
    ["mvtec"]="bottle"
)

for dataset in "${!datasets[@]}"; do
    class_name="${datasets[$dataset]}"
    echo "Processing $dataset/$class_name..."
    python preprocessing/orchestrator.py \
        --source raw-data/${dataset}-${class_name}/good-images \
        --dataset "$dataset" \
        --class_name "$class_name"
done

# Or use legacy format for backward compatibility
for dataset in custom_grid wfdd_yellow_cloth mvtec_bottle; do
    python preprocessing/orchestrator.py \
        --source raw-data/$dataset/good-images \
        --dataset $dataset
done
```

---

**Created for GLASS Fabric Defect Detection Framework**  
Compatible with ECCV 2024 GLASS implementation
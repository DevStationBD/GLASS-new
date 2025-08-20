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
**Primary interface for dataset creation**

```bash
# Complete pipeline
python orchestrator.py \
    --source raw-data/custom-grid/good-images \
    --dataset custom_grid \
    --full_pipeline

# Specific steps only
python orchestrator.py \
    --source raw-data/custom-grid/good-images \
    --dataset custom_grid \
    --steps structure preprocess defects
```

**Key Options:**
- `--source`: Directory with good fabric images
- `--dataset`: Dataset name (creates datasets/{name}/)
- `--full_pipeline`: Run complete pipeline
- `--steps`: Run specific steps (structure, preprocess, defects, validate)
- `--images_per_defect`: Number of defective images per type (default: 50)
- `--image_size`: Target image size (default: 288)

### 2. `prepare_dataset_structure.py` - Directory Setup
**Creates WFDD-compatible directory structure**

```bash
python prepare_dataset_structure.py --dataset_name custom_grid
```

Creates:
```
datasets/custom_grid/
├── train/good/
├── test/good/
├── test/{defect_type}/
└── ground_truth/{defect_type}/
```

### 3. `preprocess_images.py` - Image Processing & Splitting
**Preprocesses and splits good images into train/test**

```bash
python preprocess_images.py \
    --source raw-data/custom-grid/good-images \
    --target datasets/custom_grid \
    --image_size 288 \
    --train_ratio 0.8
```

Features:
- Resizes images to target size with padding
- Enhances contrast, brightness, sharpness
- Splits into train/test sets
- Saves as PNG format

### 4. `defect_simulator.py` - Defect Generation Engine
**Simulates realistic fabric defects on good images**

```bash
python defect_simulator.py \
    --input datasets/custom_grid/test/good \
    --output datasets/custom_grid/test \
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
python mask_utils.py --validate datasets/custom_grid/ground_truth --plot_stats

# Clean/process masks
python mask_utils.py --clean datasets/custom_grid/ground_truth/hole
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
cd preprocessing
python orchestrator.py \
    --source ../raw-data/custom-grid/good-images \
    --dataset custom_grid \
    --full_pipeline \
    --images_per_defect 50
```

### Step 3: Verify Results
Check the created dataset:
```
datasets/custom_grid/
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
cd ..
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
    -d custom_grid \
    datasets/custom_grid /path/to/dtd
```

## Configuration

### Default Parameters
- **Image Size**: 288x288 pixels
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

## Output Files

### Dataset Structure
- **Training Images**: `train/good/*.png`
- **Test Good Images**: `test/good/*.png`
- **Test Defect Images**: `test/{defect_type}/*.png`
- **Ground Truth Masks**: `ground_truth/{defect_type}/*_mask.png`

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
python mask_utils.py --validate datasets/custom_grid/ground_truth
```

### Manual Steps
Run pipeline steps individually for debugging:
```bash
# Step 1: Structure
python prepare_dataset_structure.py --dataset_name custom_grid

# Step 2: Preprocessing  
python preprocess_images.py --source raw-data/custom-grid/good-images --target datasets/custom_grid

# Step 3: Defect generation
python defect_simulator.py --input datasets/custom_grid/test/good --output datasets/custom_grid/test

# Step 4: Validation
python mask_utils.py --validate datasets/custom_grid/ground_truth
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
Process multiple datasets:
```bash
for dataset in grid_fabric woven_fabric knit_fabric; do
    python orchestrator.py \
        --source raw-data/$dataset/good-images \
        --dataset $dataset \
        --full_pipeline
done
```

---

**Created for GLASS Fabric Defect Detection Framework**  
Compatible with ECCV 2024 GLASS implementation
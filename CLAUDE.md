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

## Results and Outputs

### Directory Structure
```
results/
├── training/classname/  # Training visualizations
├── eval/classname/      # Evaluation results
├── judge/avg/           # Distribution analysis
└── models/              # Saved checkpoints
```

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

## Testing Notes

This codebase has no explicit test framework. Validation occurs through:
- Cross-validation during training (`eval_epochs` parameter)
- Benchmark evaluation against standard datasets
- Visual inspection of generated anomaly maps

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
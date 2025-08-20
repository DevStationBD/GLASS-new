#!/bin/bash

# GLASS Training Script for Custom Grid Fabric Dataset
# Trains the GLASS anomaly detection model on custom grid fabric with simulated defects

set -e

echo "🚀 Training GLASS on Custom Grid Fabric Dataset..."
echo "================================================"

# Dataset configuration
datapath=/home/arif/Projects/GLASS-new/datasets/custom
augpath=/home/arif/Projects/GLASS-new/datasets/dtd/images
classes=('grid')
flags=($(for class in "${classes[@]}"; do echo '-d '"${class}"; done))

# Check if dataset exists
if [ ! -d "$datapath" ]; then
    echo "❌ Error: Dataset directory '$datapath' does not exist!"
    echo "Please run the dataset creation script first:"
    echo "  bash shell/create-custom-grid-dataset.sh"
    exit 1
fi

# Check if DTD augmentation dataset exists
if [ ! -d "$augpath" ]; then
    echo "❌ Error: DTD augmentation directory '$augpath' does not exist!"
    echo "Please download DTD dataset from: https://www.robots.ox.ac.uk/~vgg/data/dtd/"
    exit 1
fi

# Validate dataset structure (GLASS expects nested structure: dataset_name/class_name/)
required_dirs=("grid/train/good" "grid/test/good" "grid/test/hole" "grid/test/foreign_yarn" "grid/test/missing_yarn" "grid/test/slab" "grid/test/spot" "grid/ground_truth")
for dir in "${required_dirs[@]}"; do
    if [ ! -d "$datapath/$dir" ]; then
        echo "❌ Error: Required directory '$datapath/$dir' not found!"
        echo "Dataset structure is incomplete. Please regenerate the dataset."
        exit 1
    fi
done

# Count dataset files
train_count=$(find "$datapath/grid/train/good" -name "*.png" | wc -l)
test_good_count=$(find "$datapath/grid/test/good" -name "*.png" | wc -l)
total_defect_count=0
for defect_type in hole foreign_yarn missing_yarn slab spot; do
    defect_count=$(find "$datapath/grid/test/$defect_type" -name "*.png" | wc -l)
    total_defect_count=$((total_defect_count + defect_count))
done

echo "📊 Dataset Statistics:"
echo "   Training images: $train_count"
echo "   Test good images: $test_good_count"  
echo "   Total defect images: $total_defect_count"
echo "   Defect types: hole, foreign_yarn, missing_yarn, slab, spot"
echo ""

# Ask for confirmation
read -p "Continue with GLASS training? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Training cancelled."
    exit 1
fi

echo "🔧 Starting GLASS training..."
echo "⚙️  Configuration:"
echo "   Backbone: WideResNet50"
echo "   Image size: 384x384"
echo "   Batch size: 8"
echo "   Meta epochs: 640"
echo "   Patch size: 3"
echo ""

# Change to project directory
cd /home/arif/Projects/GLASS-new

# Run GLASS training with optimized parameters for custom grid fabric
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
    --patchsize 3 \
    --meta_epochs 640 \
    --eval_epochs 10 \
    --dsc_layers 2 \
    --dsc_hidden 1024 \
    --pre_proj 1 \
    --mining 1 \
    --noise 0.015 \
    --radius 0.75 \
    --p 0.5 \
    --step 20 \
    --limit 392 \
  dataset \
    --distribution 0 \
    --mean 0.5 \
    --std 0.1 \
    --fg 0 \
    --rand_aug 1 \
    --batch_size 8 \
    --resize 384 \
    --imagesize 384 "${flags[@]}" mvtec $datapath $augpath

# Check training success
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ GLASS training completed successfully!"
    echo ""
    echo "📁 Results saved to:"
    echo "   - Models: results/models/"
    echo "   - Evaluation: results/eval/grid/"
    echo "   - Training logs: results/training/grid/"
    echo ""
    echo "📊 Performance metrics:"
    echo "   Check results.csv for detailed metrics"
    echo ""
    echo "🎯 Next steps:"
    echo "   1. Review training results:"
    echo "      ls -la results/"
    echo ""
    echo "   2. Analyze performance:"
    echo "      python -c \"import pandas as pd; print(pd.read_csv('results/results.csv').tail())\" 2>/dev/null || echo \"   Check results/results.csv manually\""
    echo ""
    echo "   3. Run inference on new images:"
    echo "      python video_inference.py --model_path results/models/backbone_0/ckpt.pth --input_path /path/to/test/image.jpg"
    echo ""
    echo "   4. Visualize results:"
    echo "      Check results/eval/grid/ for anomaly maps"
else
    echo "❌ GLASS training failed! Check the error messages above."
    echo ""
    echo "🔍 Troubleshooting tips:"
    echo "   - Ensure sufficient GPU memory (recommend 8GB+)"
    echo "   - Check dataset integrity with: python preprocessing/mask_utils.py --validate datasets/custom/grid/ground_truth"
    echo "   - Reduce batch_size if out of memory errors occur"
    echo "   - Verify CUDA/PyTorch installation"
    exit 1
fi
#!/bin/bash

# Custom Grid Fabric Dataset Creation Script
# Creates a complete WFDD-like dataset from good fabric images

set -e

echo "🚀 Creating Custom Grid Fabric Dataset..."
echo "======================================"

# Configuration
DATASET_NAME="custom_grid"
SOURCE_DIR="raw-data/custom-grid/good-images"
IMAGE_SIZE=384
TRAIN_RATIO=0.8
IMAGES_PER_DEFECT=50
SEED=42

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "❌ Error: Source directory '$SOURCE_DIR' does not exist!"
    echo "Please ensure your good fabric images are in: $SOURCE_DIR"
    exit 1
fi

# Count source images
IMAGE_COUNT=$(find "$SOURCE_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" \) | wc -l)
echo "📂 Found $IMAGE_COUNT source images in $SOURCE_DIR"

if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo "❌ Error: No image files found in source directory!"
    exit 1
fi

# Display configuration
echo ""
echo "⚙️  Configuration:"
echo "   Dataset name: $DATASET_NAME"
echo "   Source directory: $SOURCE_DIR"
echo "   Image size: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "   Train ratio: $TRAIN_RATIO"
echo "   Images per defect: $IMAGES_PER_DEFECT"
echo "   Random seed: $SEED"
echo ""

# Ask for confirmation
read -p "Continue with dataset creation? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Dataset creation cancelled."
    exit 1
fi

# Run the complete pipeline
echo "🔧 Starting dataset creation pipeline..."
python preprocessing/orchestrator.py \
    --source "$SOURCE_DIR" \
    --dataset "$DATASET_NAME" \
    --full_pipeline \
    --image_size $IMAGE_SIZE \
    --train_ratio $TRAIN_RATIO \
    --images_per_defect $IMAGES_PER_DEFECT \
    --seed $SEED

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Dataset creation completed successfully!"
    echo ""
    echo "📁 Dataset location: datasets/$DATASET_NAME/"
    echo "📊 Dataset statistics:"
    
    # Count files in each directory
    TRAIN_COUNT=$(find "datasets/$DATASET_NAME/train/good" -name "*.png" 2>/dev/null | wc -l)
    TEST_GOOD_COUNT=$(find "datasets/$DATASET_NAME/test/good" -name "*.png" 2>/dev/null | wc -l)
    
    echo "   Training images: $TRAIN_COUNT"
    echo "   Test good images: $TEST_GOOD_COUNT"
    echo "   Defect types:"
    
    for defect_type in hole foreign_yarn missing_yarn slab spot; do
        DEFECT_COUNT=$(find "datasets/$DATASET_NAME/test/$defect_type" -name "*.png" 2>/dev/null | wc -l)
        MASK_COUNT=$(find "datasets/$DATASET_NAME/ground_truth/$defect_type" -name "*_mask.png" 2>/dev/null | wc -l)
        echo "      $defect_type: $DEFECT_COUNT images, $MASK_COUNT masks"
    done
    
    echo ""
    echo "🎯 Next steps:"
    echo "   1. Review the generated dataset:"
    echo "      ls -la datasets/$DATASET_NAME/"
    echo ""
    echo "   2. Train GLASS model:"
    echo "      python main.py \\"
    echo "        --gpu 0 --test ckpt \\"
    echo "        net -b wideresnet50 -le layer2 layer3 --meta_epochs 640 \\"
    echo "        dataset --batch_size 8 --resize 384 --imagesize 384 \\"
    echo "          -d $DATASET_NAME datasets/$DATASET_NAME datasets/dtd"
    echo ""
    echo "   3. Validate defects visually:"
    echo "      python preprocessing/mask_utils.py --validate datasets/$DATASET_NAME/ground_truth"
    echo ""
else
    echo "❌ Dataset creation failed! Check the error messages above."
    exit 1
fi
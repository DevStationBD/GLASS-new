#!/bin/bash

# GLASS Training Script for Multiple Custom Fabric Classes
# Runs GLASS training separately for each class in the list

set -e

# Get class name from parameter or use default
CLASS_NAME="${1:-mvt2}"

echo "🚀 Training GLASS on Custom Fabric Dataset (per class)..."
echo "========================================================"

# Dataset configuration
datapath=~/GLASS-new/datasets/custom
augpath=~/GLASS-new/datasets/dtd/images
classes=("$CLASS_NAME")   # Use parameter as class name

# Check dataset root
if [ ! -d "$datapath" ]; then
    echo "❌ Error: Dataset directory '$datapath' does not exist!"
    exit 1
fi

# Check augmentation dataset
if [ ! -d "$augpath" ]; then
    echo "❌ Error: DTD augmentation directory '$augpath' does not exist!"
    exit 1
fi

# Validate datasets
for cls in "${classes[@]}"; do
    echo ""
    echo "📂 Validating dataset for class: $cls"
    echo "-------------------------------------"

    required_dirs=(
        "$cls/train/good"
        "$cls/test/good"
        "$cls/test/hole"
        "$cls/test/foreign_yarn"
        "$cls/test/missing_yarn"
        "$cls/test/slab"
        "$cls/test/spot"
        "$cls/ground_truth"
    )

    for dir in "${required_dirs[@]}"; do
        if [ ! -d "$datapath/$dir" ]; then
            echo "❌ Error: Required directory '$datapath/$dir' not found!"
            exit 1
        fi
    done

    train_count=$(find "$datapath/$cls/train/good" -name "*.png" | wc -l)
    test_good_count=$(find "$datapath/$cls/test/good" -name "*.png" | wc -l)
    total_defect_count=0
    for defect_type in hole foreign_yarn missing_yarn slab spot; do
        defect_count=$(find "$datapath/$cls/test/$defect_type" -name "*.png" | wc -l)
        total_defect_count=$((total_defect_count + defect_count))
    done

    echo "📊 Dataset Statistics for $cls:"
    echo "   Training images: $train_count"
    echo "   Test good images: $test_good_count"
    echo "   Total defect images: $total_defect_count"
done

# Auto-proceed for automated training (no user interaction needed)
echo ""
echo "✅ Validation complete. Proceeding with GLASS training..."
echo ""

# Training loop
cd ~/GLASS-new
for cls in "${classes[@]}"; do
    echo ""
    echo "🚀 Training GLASS on class: $cls"
    echo "--------------------------------"

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
        --batch_size 32 \
        --resize 384 \
        --imagesize 384 -d "$cls" mvtec $datapath $augpath
        # Image size options (preserves aspect ratio):
        # --resize 384 --imagesize 384   # Standard (8GB GPU, faster training)
        # --resize 448 --imagesize 448   # Balanced (10GB GPU, good quality)
        # --resize 512 --imagesize 512   # High quality (12GB GPU, better defects)
        # --resize 768 --imagesize 768   # Maximum (20GB GPU, finest details)
        # Note: Larger sizes need smaller batch_size if GPU memory limited

    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ GLASS training for class '$cls' completed successfully!"
        echo ""
        echo "📁 Results saved to:"
        echo "   - Models: results/models/"
        echo "   - Evaluation: results/eval/$cls/"
        echo "   - Training logs: results/training/$cls/"
        echo ""
        echo "📊 Performance metrics:"
        echo "   Check results/results.csv (look for '$cls' rows)"
        echo ""
    else
        echo "❌ Training failed for class '$cls'."
        exit 1
    fi
done

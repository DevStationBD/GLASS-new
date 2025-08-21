#!/usr/bin/env python3
"""
Defect Overlay Visualization Tool for GLASS Dataset
Creates visible overlay images showing defects highlighted on original images
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path
import json

def create_overlay_visualization(image_path, mask_path, output_path, alpha=0.4):
    """Create overlay visualization with defect highlighted in red"""
    # Read image and mask
    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Warning: Could not read image {image_path}")
        return False
    
    if mask is None:
        print(f"Warning: Could not read mask {mask_path}")
        return False
    
    # Ensure mask and image have compatible dimensions
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
    # Normalize mask to binary (0 or 255)
    mask_binary = (mask > 127).astype(np.uint8) * 255
    
    # Create colored overlay (red for defects)
    overlay = image.copy()
    
    # Apply red color to defect areas
    overlay[mask_binary > 0] = [0, 0, 255]  # Red in BGR format
    
    # Blend original image with overlay
    result = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)
    
    # Add green contours for better visibility
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)  # Green contours
    
    # Add text showing defect area
    defect_area = np.sum(mask_binary > 0)
    cv2.putText(result, f'Defect Area: {defect_area} pixels', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Save result
    cv2.imwrite(str(output_path), result)
    return True

def create_pattern_grid(image_paths, mask_paths, output_path, defect_type, grid_size=(3, 3)):
    """Create a grid showing multiple defect patterns of the same type"""
    max_samples = grid_size[0] * grid_size[1]
    
    if len(image_paths) < max_samples:
        # Adjust grid size based on available samples
        rows = min(3, (len(image_paths) + 2) // 3)
        cols = min(3, len(image_paths))
        grid_size = (rows, cols)
        max_samples = rows * cols
    
    # Select evenly spaced samples
    if len(image_paths) > max_samples:
        indices = np.linspace(0, len(image_paths)-1, max_samples, dtype=int)
    else:
        indices = list(range(len(image_paths)))
    
    # Read first image to get dimensions
    first_img = cv2.imread(str(image_paths[0]))
    if first_img is None:
        return False
        
    target_size = (256, 256)  # Standard size for grid display
    
    # Create grid
    grid_height = grid_size[0] * target_size[1]
    grid_width = grid_size[1] * target_size[0]
    grid_img = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    for idx, sample_idx in enumerate(indices):
        row = idx // grid_size[1]
        col = idx % grid_size[1]
        
        if sample_idx < len(image_paths) and sample_idx < len(mask_paths):
            # Create overlay for this sample
            image = cv2.imread(str(image_paths[sample_idx]))
            mask = cv2.imread(str(mask_paths[sample_idx]), cv2.IMREAD_GRAYSCALE)
            
            if image is not None and mask is not None:
                # Resize to target size
                image = cv2.resize(image, target_size)
                mask = cv2.resize(mask, target_size)
                
                # Create overlay
                mask_binary = (mask > 127).astype(np.uint8) * 255
                
                if mask_binary.max() > 0:
                    # Apply defect highlighting
                    overlay = image.copy()
                    overlay[mask_binary > 0] = [0, 0, 255]  # Red defects
                    result = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
                    
                    # Add contours
                    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(result, contours, -1, (0, 255, 0), 1)
                else:
                    result = image
                
                # Calculate position in grid
                y_start = row * target_size[1]
                y_end = (row + 1) * target_size[1]
                x_start = col * target_size[0]
                x_end = (col + 1) * target_size[0]
                
                # Place in grid
                grid_img[y_start:y_end, x_start:x_end] = result
                
                # Add sample number label
                label = f"#{Path(image_paths[sample_idx]).stem}"
                cv2.putText(grid_img, label, (x_start + 5, y_start + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Add defect area info
                defect_area = np.sum(mask_binary > 0)
                area_label = f"{defect_area}px"
                cv2.putText(grid_img, area_label, (x_start + 5, y_end - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Add title
    title = f"{defect_type.upper()} Defect Patterns"
    cv2.putText(grid_img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    cv2.imwrite(str(output_path), grid_img)
    return True

def create_side_by_side_comparison(image_path, mask_path, output_path):
    """Create side-by-side comparison: original | mask | overlay"""
    # Read image and mask
    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    if image is None or mask is None:
        return False
    
    # Resize if needed
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
    # Create mask visualization (colorized)
    mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
    
    # Create overlay
    mask_binary = (mask > 127).astype(np.uint8) * 255
    overlay = image.copy()
    overlay[mask_binary > 0] = [0, 0, 255]
    result_overlay = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    
    # Add contours to overlay
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result_overlay, contours, -1, (0, 255, 0), 2)
    
    # Combine images side by side
    combined = np.hstack([image, mask_colored, result_overlay])
    
    # Add labels
    h, w = image.shape[:2]
    cv2.putText(combined, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, 'Ground Truth', (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, 'Overlay', (2*w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imwrite(str(output_path), combined)
    return True

def process_dataset_class(dataset_path, class_name, output_base):
    """Process a single dataset class and create all visualizations"""
    class_path = Path(dataset_path) / class_name
    test_path = class_path / "test"
    gt_path = class_path / "ground_truth"
    
    if not test_path.exists():
        print(f"❌ Test path not found: {test_path}")
        return None
    
    if not gt_path.exists():
        print(f"❌ Ground truth path not found: {gt_path}")
        return None
    
    # Create output directories
    patterns_dir = Path(output_base) / "patterns" / class_name
    patterns_dir.mkdir(parents=True, exist_ok=True)
    
    overlays_dir = patterns_dir / "overlays"
    overlays_dir.mkdir(exist_ok=True)
    
    grids_dir = patterns_dir / "grids" 
    grids_dir.mkdir(exist_ok=True)
    
    comparisons_dir = patterns_dir / "comparisons"
    comparisons_dir.mkdir(exist_ok=True)
    
    # Get defect types (exclude 'good' folder)
    defect_types = [d.name for d in test_path.iterdir() 
                   if d.is_dir() and d.name != "good"]
    
    if not defect_types:
        print(f"❌ No defect types found in {test_path}")
        return None
    
    stats = {
        "class": class_name, 
        "defect_types": {},
        "total_samples": 0,
        "processing_status": "success"
    }
    
    print(f"📊 Found defect types: {defect_types}")
    
    for defect_type in defect_types:
        print(f"\n🔍 Processing defect type: {defect_type}")
        
        test_defect_path = test_path / defect_type
        gt_defect_path = gt_path / defect_type
        
        if not gt_defect_path.exists():
            print(f"⚠️  No ground truth folder for {defect_type}, skipping...")
            continue
            
        # Get matching image and mask files
        image_files = sorted([f for f in test_defect_path.glob("*.png")])
        mask_files = sorted([f for f in gt_defect_path.glob("*_mask.png")])
        
        if not image_files:
            print(f"⚠️  No test images found for {defect_type}")
            continue
            
        if not mask_files:
            print(f"⚠️  No mask files found for {defect_type}")
            continue
        
        # Match images with masks based on filename
        matched_pairs = []
        for img_file in image_files:
            img_stem = img_file.stem  # e.g., "001"
            mask_file = gt_defect_path / f"{img_stem}_mask.png"
            if mask_file.exists():
                matched_pairs.append((img_file, mask_file))
        
        if not matched_pairs:
            print(f"⚠️  No matching image-mask pairs found for {defect_type}")
            continue
        
        num_pairs = len(matched_pairs)
        print(f"✅ Found {num_pairs} image-mask pairs for {defect_type}")
        stats["defect_types"][defect_type] = num_pairs
        stats["total_samples"] += num_pairs
        
        # Create individual overlay images (first 5)
        overlay_type_dir = overlays_dir / defect_type
        overlay_type_dir.mkdir(exist_ok=True)
        
        comparison_type_dir = comparisons_dir / defect_type
        comparison_type_dir.mkdir(exist_ok=True)
        
        created_overlays = 0
        created_comparisons = 0
        
        for i, (img_path, mask_path) in enumerate(matched_pairs[:5]):
            # Create overlay
            overlay_output = overlay_type_dir / f"{img_path.stem}_overlay.png"
            if create_overlay_visualization(img_path, mask_path, overlay_output):
                created_overlays += 1
            
            # Create side-by-side comparison
            comparison_output = comparison_type_dir / f"{img_path.stem}_comparison.png"
            if create_side_by_side_comparison(img_path, mask_path, comparison_output):
                created_comparisons += 1
        
        print(f"   📸 Created {created_overlays} overlay images")
        print(f"   📸 Created {created_comparisons} comparison images")
        
        # Create pattern grid
        grid_output = grids_dir / f"{defect_type}_pattern_grid.png"
        image_paths = [pair[0] for pair in matched_pairs]
        mask_paths = [pair[1] for pair in matched_pairs]
        
        if create_pattern_grid(image_paths, mask_paths, grid_output, defect_type):
            print(f"   🎨 Created pattern grid")
    
    # Save statistics
    stats_file = patterns_dir / "statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ Completed processing {class_name}")
    print(f"📁 Results saved to: {patterns_dir}")
    print(f"📊 Total samples processed: {stats['total_samples']}")
    
    return patterns_dir

def main():
    parser = argparse.ArgumentParser(description="Create defect overlay visualizations for GLASS dataset")
    parser.add_argument("--dataset", default="/home/arif/Projects/GLASS-new/datasets/custom", 
                       help="Dataset root path")
    parser.add_argument("--class_name", required=True, help="Class name to process")
    parser.add_argument("--output", default="/home/arif/Projects/GLASS-new", help="Output base directory")
    parser.add_argument("--all", action="store_true", help="Process all classes in dataset")
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ Dataset path {dataset_path} does not exist")
        return
    
    print(f"🚀 Starting defect overlay visualization...")
    print(f"📂 Dataset: {dataset_path}")
    print(f"📤 Output: {args.output}/patterns/")
    
    if args.all:
        # Process all classes
        class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
        for class_dir in class_dirs:
            print(f"\n{'='*60}")
            print(f"Processing class: {class_dir.name}")
            print(f"{'='*60}")
            process_dataset_class(dataset_path, class_dir.name, args.output)
    else:
        # Process single class
        print(f"\n{'='*60}")
        print(f"Processing class: {args.class_name}")
        print(f"{'='*60}")
        result_dir = process_dataset_class(dataset_path, args.class_name, args.output)
        
        if result_dir and result_dir.exists():
            print(f"\n{'='*60}")
            print("🎉 VISUALIZATION COMPLETE!")
            print(f"{'='*60}")
            print(f"Results saved to: {result_dir}")
            print(f"")
            print("📁 Generated files:")
            print(f"   📸 overlays/[defect_type]/: Defects highlighted in red")
            print(f"   📊 grids/: Pattern summary grids")
            print(f"   🔍 comparisons/[defect_type]/: Side-by-side comparisons")
            print(f"   📋 statistics.json: Processing summary")
            print(f"")
            print("💡 Open any PNG file to see the defect visualizations!")

if __name__ == "__main__":
    main()
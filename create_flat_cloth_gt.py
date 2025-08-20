#!/usr/bin/env python3
"""
Create ground truth masks for flat_cloth dataset defects.
Automatically detects anomalous regions in defect images and creates binary masks.
"""

import os
import cv2
import numpy as np
from PIL import Image
import glob
from pathlib import Path

def create_ground_truth_mask(image_path, output_path, debug=False):
    """
    Create ground truth mask for a single defect image.
    
    Args:
        image_path: Path to defect image
        output_path: Path to save ground truth mask
        debug: If True, saves intermediate processing steps
    """
    # Read image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Get image dimensions
    height, width = gray.shape
    
    # Create binary mask (same size as image)
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Method 1: Detect very bright spots (threads, debris)
    # Find pixels significantly brighter than median
    median_val = np.median(gray)
    std_val = np.std(gray)
    bright_threshold = median_val + 2.5 * std_val
    bright_mask = gray > bright_threshold
    
    # Method 2: Detect very dark spots (shadows, stains)
    # Find pixels significantly darker than median
    dark_threshold = median_val - 2.5 * std_val
    dark_mask = gray < dark_threshold
    
    # Method 3: Edge detection for linear defects
    # Use Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges to make them more prominent
    kernel = np.ones((3,3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Method 4: Local contrast detection
    # Use Laplacian to find areas with high local variation
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_abs = np.abs(laplacian)
    laplacian_threshold = np.mean(laplacian_abs) + 2 * np.std(laplacian_abs)
    contrast_mask = laplacian_abs > laplacian_threshold
    
    # Combine all detection methods
    combined_mask = bright_mask | dark_mask | (edges_dilated > 0) | contrast_mask
    
    # Clean up the mask
    # Remove very small regions (noise)
    kernel_clean = np.ones((2,2), np.uint8)
    cleaned_mask = cv2.morphologyEx(combined_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel_clean)
    
    # Dilate remaining regions slightly to ensure full defect coverage
    kernel_dilate = np.ones((3,3), np.uint8)
    final_mask = cv2.dilate(cleaned_mask, kernel_dilate, iterations=1)
    
    # Convert to 0-255 range for proper mask format
    final_mask = (final_mask * 255).astype(np.uint8)
    
    # Save the mask
    cv2.imwrite(output_path, final_mask)
    
    if debug:
        # Save intermediate steps for debugging
        debug_dir = os.path.dirname(output_path) + "_debug"
        os.makedirs(debug_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        cv2.imwrite(f"{debug_dir}/{base_name}_bright.png", (bright_mask * 255).astype(np.uint8))
        cv2.imwrite(f"{debug_dir}/{base_name}_dark.png", (dark_mask * 255).astype(np.uint8))
        cv2.imwrite(f"{debug_dir}/{base_name}_edges.png", edges_dilated)
        cv2.imwrite(f"{debug_dir}/{base_name}_contrast.png", (contrast_mask * 255).astype(np.uint8))
        cv2.imwrite(f"{debug_dir}/{base_name}_combined.png", (combined_mask * 255).astype(np.uint8))
    
    return final_mask

def main():
    """Main function to process all defect images."""
    
    # Paths
    defect_dir = "/home/arif/Projects/GLASS-new/datasets/gray/flat_cloth/test/defect"
    gt_dir = "/home/arif/Projects/GLASS-new/datasets/gray/flat_cloth/ground_truth/defect"
    
    # Ensure output directory exists
    os.makedirs(gt_dir, exist_ok=True)
    
    # Get all defect images
    defect_images = glob.glob(os.path.join(defect_dir, "*.jpg"))
    defect_images.sort()
    
    print(f"Processing {len(defect_images)} defect images...")
    
    # Process each image
    for i, img_path in enumerate(defect_images):
        img_name = os.path.basename(img_path)
        mask_name = os.path.splitext(img_name)[0] + "_mask.png"
        mask_path = os.path.join(gt_dir, mask_name)
        
        print(f"Processing {i+1}/{len(defect_images)}: {img_name}")
        
        try:
            mask = create_ground_truth_mask(img_path, mask_path, debug=False)
            
            # Check if mask has any defects detected
            if np.sum(mask) > 0:
                print(f"  ✓ Defects detected, mask saved: {mask_name}")
            else:
                print(f"  ⚠ No defects detected for: {img_name}")
                
        except Exception as e:
            print(f"  ✗ Error processing {img_name}: {e}")
    
    print(f"\nGround truth creation completed!")
    print(f"Masks saved in: {gt_dir}")
    
    # Count successful masks
    mask_files = glob.glob(os.path.join(gt_dir, "*.png"))
    print(f"Total masks created: {len(mask_files)}")

if __name__ == "__main__":
    main()
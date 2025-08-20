#!/usr/bin/env python3
"""
Refined ground truth mask creation for flat_cloth dataset.
Creates cleaner, more focused masks by reducing noise and improving defect detection.
"""

import os
import cv2
import numpy as np
from PIL import Image
import glob
from pathlib import Path

def create_refined_mask(image_path, output_path, debug=False):
    """
    Create refined ground truth mask with less noise and better defect detection.
    """
    # Read image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Get image dimensions
    height, width = gray.shape
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Create binary mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Method 1: Statistical outlier detection (more conservative)
    mean_val = np.mean(blurred)
    std_val = np.std(blurred)
    
    # Detect significant bright anomalies
    bright_threshold = mean_val + 3.0 * std_val
    bright_mask = blurred > bright_threshold
    
    # Detect significant dark anomalies
    dark_threshold = mean_val - 3.0 * std_val
    dark_mask = blurred < dark_threshold
    
    # Method 2: Morphological operations for structure detection
    # Use opening and closing to detect blob-like defects
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Detect bright blobs
    bright_blobs = cv2.morphologyEx((blurred > mean_val + 2.5 * std_val).astype(np.uint8), 
                                    cv2.MORPH_CLOSE, kernel_small)
    
    # Detect dark blobs  
    dark_blobs = cv2.morphologyEx((blurred < mean_val - 2.5 * std_val).astype(np.uint8), 
                                  cv2.MORPH_CLOSE, kernel_small)
    
    # Method 3: Edge-based detection for linear defects
    # Use adaptive thresholding for better edge detection
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)
    
    # Find contours and filter by size
    contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edge_mask = np.zeros_like(gray)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        # Keep only medium-sized contours (filter out noise and very large background features)
        if 50 < area < 5000:
            cv2.drawContours(edge_mask, [contour], -1, 255, -1)
    
    # Combine all detection methods
    combined_mask = (bright_mask.astype(np.uint8) | 
                     dark_mask.astype(np.uint8) | 
                     bright_blobs | 
                     dark_blobs | 
                     (edge_mask > 0).astype(np.uint8))
    
    # Clean up the mask
    # Remove very small regions (noise)
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_clean)
    
    # Close small gaps in defects
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    final_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Remove regions that are too small to be meaningful defects
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(final_mask)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        # Keep only defects larger than 30 pixels
        if area > 30:
            cv2.drawContours(filtered_mask, [contour], -1, 255, -1)
    
    # Convert to 0-255 range
    final_mask = filtered_mask
    
    # Save the mask
    cv2.imwrite(output_path, final_mask)
    
    if debug:
        # Save intermediate steps for debugging
        debug_dir = os.path.dirname(output_path) + "_debug_refined"
        os.makedirs(debug_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        cv2.imwrite(f"{debug_dir}/{base_name}_bright.png", (bright_mask * 255).astype(np.uint8))
        cv2.imwrite(f"{debug_dir}/{base_name}_dark.png", (dark_mask * 255).astype(np.uint8))
        cv2.imwrite(f"{debug_dir}/{base_name}_edges.png", edge_mask)
        cv2.imwrite(f"{debug_dir}/{base_name}_combined.png", (combined_mask * 255).astype(np.uint8))
        cv2.imwrite(f"{debug_dir}/{base_name}_cleaned.png", (cleaned_mask * 255).astype(np.uint8))
    
    return final_mask

def main():
    """Main function to refine all ground truth masks."""
    
    # Paths
    defect_dir = "/home/arif/Projects/GLASS-new/datasets/gray/flat_cloth/test/defect"
    gt_dir = "/home/arif/Projects/GLASS-new/datasets/gray/flat_cloth/ground_truth/defect"
    
    # Get all defect images
    defect_images = glob.glob(os.path.join(defect_dir, "*.jpg"))
    defect_images.sort()
    
    print(f"Refining {len(defect_images)} ground truth masks...")
    
    # Process each image
    for i, img_path in enumerate(defect_images):
        img_name = os.path.basename(img_path)
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(gt_dir, mask_name)
        
        print(f"Refining {i+1}/{len(defect_images)}: {img_name}")
        
        try:
            mask = create_refined_mask(img_path, mask_path, debug=False)
            
            # Check if mask has any defects detected
            if np.sum(mask) > 0:
                print(f"  ✓ Refined mask saved: {mask_name}")
            else:
                print(f"  ⚠ No defects detected after refinement: {img_name}")
                
        except Exception as e:
            print(f"  ✗ Error processing {img_name}: {e}")
    
    print(f"\nGround truth refinement completed!")
    print(f"Refined masks saved in: {gt_dir}")

if __name__ == "__main__":
    main()
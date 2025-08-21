#!/usr/bin/env python3
"""
Pattern Visualization Tool
Exports extracted defect patterns as PNG images for inspection.
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.append(str(Path(__file__).parent))
from real_defect_extractor import DefectPatternLibrary

def visualize_pattern(pattern, save_path: Path):
    """Visualize a single defect pattern."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original defect region
    axes[0].imshow(pattern.defect_region)
    axes[0].set_title(f'Defect Region\n{pattern.fabric_type} - {pattern.defect_type}')
    axes[0].axis('off')
    
    # Defect mask
    axes[1].imshow(pattern.mask, cmap='gray')
    axes[1].set_title(f'Defect Mask\nArea: {pattern.area} pixels')
    axes[1].axis('off')
    
    # Overlay (defect region with mask outline)
    axes[2].imshow(pattern.defect_region)
    
    # Add mask contour
    mask_binary = pattern.mask > 128
    contours = []
    if mask_binary.any():
        import cv2
        mask_uint8 = (mask_binary).astype(np.uint8) * 255
        contours_cv, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours_cv:
            contour = contour.squeeze()
            if len(contour.shape) == 2 and contour.shape[0] > 2:
                axes[2].plot(contour[:, 0], contour[:, 1], 'r-', linewidth=2)
    
    axes[2].set_title(f'Overlay\nCompactness: {pattern.compactness:.3f}')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def export_patterns_as_png(patterns_path: str, output_dir: str, limit_per_type: int = 5):
    """Export defect patterns as PNG images for visualization."""
    
    # Load pattern library
    library_path = Path(patterns_path) / "wfdd_patterns.pkl"
    if not library_path.exists():
        print(f"❌ Pattern library not found: {library_path}")
        return
    
    print(f"📦 Loading pattern library from {library_path}")
    library = DefectPatternLibrary.load(str(library_path))
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    stats = library.get_stats()
    print(f"📊 Found {stats['total_patterns']} patterns")
    print(f"   Fabric types: {list(stats['fabric_types'].keys())}")
    print(f"   Defect types: {list(stats['defect_types'].keys())}")
    
    # Export patterns by fabric and defect type
    for fabric_type in stats['fabric_types'].keys():
        fabric_dir = output_path / fabric_type
        fabric_dir.mkdir(exist_ok=True)
        
        for defect_type in stats['defect_types'].keys():
            patterns = library.search_patterns(fabric_type=fabric_type, defect_type=defect_type)
            
            if not patterns:
                continue
                
            defect_dir = fabric_dir / defect_type
            defect_dir.mkdir(exist_ok=True)
            
            print(f"🎨 Exporting {fabric_type}/{defect_type}: {len(patterns)} patterns")
            
            # Limit patterns to avoid too many files
            patterns_to_export = patterns[:limit_per_type]
            
            for i, pattern in enumerate(patterns_to_export):
                # Export defect region as PNG
                defect_img_path = defect_dir / f"defect_{i+1:03d}.png"
                Image.fromarray(pattern.defect_region).save(defect_img_path)
                
                # Export mask as PNG
                mask_img_path = defect_dir / f"mask_{i+1:03d}.png"
                Image.fromarray(pattern.mask).save(mask_img_path)
                
                # Export visualization
                viz_path = defect_dir / f"viz_{i+1:03d}.png"
                visualize_pattern(pattern, viz_path)
    
    # Create summary report
    summary_path = output_path / "pattern_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ Pattern visualization completed!")
    print(f"   Output directory: {output_path}")
    print(f"   Summary: {summary_path}")

def create_pattern_grid(patterns_path: str, output_file: str, grid_size: tuple = (5, 5)):
    """Create a grid view of patterns for quick overview."""
    
    library_path = Path(patterns_path) / "wfdd_patterns.pkl"
    if not library_path.exists():
        print(f"❌ Pattern library not found: {library_path}")
        return
    
    library = DefectPatternLibrary.load(str(library_path))
    
    # Get a sample of patterns
    max_patterns = grid_size[0] * grid_size[1]
    sample_patterns = library.patterns[:max_patterns]
    
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(15, 15))
    axes = axes.flatten()
    
    for i, pattern in enumerate(sample_patterns):
        if i < len(axes):
            axes[i].imshow(pattern.defect_region)
            axes[i].set_title(f'{pattern.fabric_type}\n{pattern.defect_type}\nArea: {pattern.area}', 
                            fontsize=8)
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(sample_patterns), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Pattern grid created: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Visualize extracted defect patterns")
    parser.add_argument("--patterns_path", default="patterns/",
                       help="Path to patterns directory")
    parser.add_argument("--output_dir", default="patterns/visualizations/",
                       help="Output directory for PNG exports")
    parser.add_argument("--limit_per_type", type=int, default=5,
                       help="Max patterns to export per defect type")
    parser.add_argument("--grid_only", action="store_true",
                       help="Create only grid overview")
    parser.add_argument("--grid_file", default="patterns/pattern_grid.png",
                       help="Grid overview output file")
    
    args = parser.parse_args()
    
    if args.grid_only:
        create_pattern_grid(args.patterns_path, args.grid_file)
    else:
        export_patterns_as_png(args.patterns_path, args.output_dir, args.limit_per_type)
        create_pattern_grid(args.patterns_path, args.grid_file)

if __name__ == "__main__":
    main()
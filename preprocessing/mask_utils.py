#!/usr/bin/env python3
"""
Mask Utilities for Fabric Defect Detection
Provides utilities for mask validation, post-processing, and quality checks.

Usage:
    python mask_utils.py --validate datasets/custom_grid/ground_truth
"""

import os
import argparse
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
import matplotlib.pyplot as plt
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MaskValidator:
    """Validates and analyzes defect masks for quality assurance."""
    
    def __init__(self):
        self.validation_results = defaultdict(list)
        self.statistics = defaultdict(dict)
    
    def validate_mask(self, mask_path: Path) -> Dict[str, any]:
        """Validate a single mask file."""
        try:
            # Load mask
            mask = np.array(Image.open(mask_path))
            
            # Convert to grayscale if needed
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            
            validation_result = {
                'path': str(mask_path),
                'valid': True,
                'errors': [],
                'warnings': [],
                'statistics': {}
            }
            
            # Check 1: Binary mask (only 0 and 255 values)
            unique_values = np.unique(mask)
            if not np.array_equal(unique_values, [0]) and not np.array_equal(unique_values, [0, 255]):
                if len(unique_values) > 2:
                    validation_result['warnings'].append(
                        f"Non-binary mask detected. Values: {unique_values}"
                    )
                    # Auto-fix: binarize
                    mask = (mask > 127).astype(np.uint8) * 255
            
            # Check 2: Mask size
            h, w = mask.shape
            if h != w:
                validation_result['warnings'].append(f"Non-square mask: {w}x{h}")
            
            # Check 3: Defect area ratio
            defect_pixels = np.sum(mask > 0)
            total_pixels = h * w
            defect_ratio = defect_pixels / total_pixels
            
            if defect_ratio == 0:
                validation_result['errors'].append("Empty mask (no defect pixels)")
                validation_result['valid'] = False
            elif defect_ratio > 0.5:
                validation_result['warnings'].append(
                    f"Large defect area: {defect_ratio:.3f} of total image"
                )
            
            # Check 4: Connected components analysis
            num_labels, labels = cv2.connectedComponents(mask)
            num_components = num_labels - 1  # Subtract background
            
            if num_components > 10:
                validation_result['warnings'].append(
                    f"Many separate defect regions: {num_components}"
                )
            
            # Calculate statistics
            validation_result['statistics'] = {
                'width': w,
                'height': h,
                'defect_pixels': int(defect_pixels),
                'defect_ratio': float(defect_ratio),
                'num_components': int(num_components),
                'unique_values': unique_values.tolist()
            }
            
            return validation_result
            
        except Exception as e:
            return {
                'path': str(mask_path),
                'valid': False,
                'errors': [f"Failed to load mask: {str(e)}"],
                'warnings': [],
                'statistics': {}
            }
    
    def validate_dataset_masks(self, ground_truth_dir: Path) -> Dict[str, List[Dict]]:
        """Validate all masks in the dataset."""
        results = defaultdict(list)
        
        for defect_type_dir in ground_truth_dir.iterdir():
            if defect_type_dir.is_dir():
                defect_type = defect_type_dir.name
                logger.info(f"Validating {defect_type} masks...")
                
                mask_files = list(defect_type_dir.glob("*_mask.png"))
                
                for mask_file in mask_files:
                    result = self.validate_mask(mask_file)
                    results[defect_type].append(result)
                
                logger.info(f"Validated {len(mask_files)} {defect_type} masks")
        
        return dict(results)
    
    def generate_validation_report(self, validation_results: Dict[str, List[Dict]]) -> Dict:
        """Generate a comprehensive validation report."""
        report = {
            'summary': {},
            'defect_types': {},
            'issues': {
                'errors': [],
                'warnings': []
            }
        }
        
        total_masks = 0
        total_valid = 0
        total_errors = 0
        total_warnings = 0
        
        for defect_type, results in validation_results.items():
            type_stats = {
                'total_masks': len(results),
                'valid_masks': sum(1 for r in results if r['valid']),
                'error_count': sum(len(r['errors']) for r in results),
                'warning_count': sum(len(r['warnings']) for r in results),
                'avg_defect_ratio': np.mean([r['statistics'].get('defect_ratio', 0) for r in results]),
                'avg_components': np.mean([r['statistics'].get('num_components', 0) for r in results])
            }
            
            report['defect_types'][defect_type] = type_stats
            
            total_masks += type_stats['total_masks']
            total_valid += type_stats['valid_masks']
            total_errors += type_stats['error_count']
            total_warnings += type_stats['warning_count']
            
            # Collect issues
            for result in results:
                for error in result['errors']:
                    report['issues']['errors'].append({
                        'defect_type': defect_type,
                        'file': result['path'],
                        'error': error
                    })
                for warning in result['warnings']:
                    report['issues']['warnings'].append({
                        'defect_type': defect_type,
                        'file': result['path'],
                        'warning': warning
                    })
        
        report['summary'] = {
            'total_masks': total_masks,
            'valid_masks': total_valid,
            'validity_rate': total_valid / total_masks if total_masks > 0 else 0,
            'total_errors': total_errors,
            'total_warnings': total_warnings
        }
        
        return report
    
    def print_validation_report(self, report: Dict):
        """Print a formatted validation report."""
        print("\n" + "="*60)
        print("MASK VALIDATION REPORT")
        print("="*60)
        
        # Summary
        summary = report['summary']
        print(f"\n📊 SUMMARY:")
        print(f"   Total masks: {summary['total_masks']}")
        print(f"   Valid masks: {summary['valid_masks']}")
        print(f"   Validity rate: {summary['validity_rate']:.2%}")
        print(f"   Total errors: {summary['total_errors']}")
        print(f"   Total warnings: {summary['total_warnings']}")
        
        # Per defect type
        print(f"\n📋 BY DEFECT TYPE:")
        for defect_type, stats in report['defect_types'].items():
            print(f"   {defect_type}:")
            print(f"      Masks: {stats['total_masks']}")
            print(f"      Valid: {stats['valid_masks']}")
            print(f"      Avg defect ratio: {stats['avg_defect_ratio']:.3f}")
            print(f"      Avg components: {stats['avg_components']:.1f}")
        
        # Issues
        if report['issues']['errors']:
            print(f"\n❌ ERRORS ({len(report['issues']['errors'])}):")
            for issue in report['issues']['errors'][:5]:  # Show first 5
                print(f"   {issue['defect_type']}: {issue['error']}")
            if len(report['issues']['errors']) > 5:
                print(f"   ... and {len(report['issues']['errors']) - 5} more")
        
        if report['issues']['warnings']:
            print(f"\n⚠️  WARNINGS ({len(report['issues']['warnings'])}):")
            for issue in report['issues']['warnings'][:5]:  # Show first 5
                print(f"   {issue['defect_type']}: {issue['warning']}")
            if len(report['issues']['warnings']) > 5:
                print(f"   ... and {len(report['issues']['warnings']) - 5} more")
        
        print("\n" + "="*60)


class MaskProcessor:
    """Post-processes and enhances defect masks."""
    
    def __init__(self):
        pass
    
    def clean_mask(self, mask: np.ndarray, min_area: int = 5) -> np.ndarray:
        """Clean mask by removing small noise and filling holes."""
        # Ensure binary mask
        binary_mask = (mask > 127).astype(np.uint8)
        
        # Remove small objects
        num_labels, labels = cv2.connectedComponents(binary_mask)
        cleaned_mask = np.zeros_like(binary_mask)
        
        for label in range(1, num_labels):
            component_mask = (labels == label)
            if np.sum(component_mask) >= min_area:
                cleaned_mask[component_mask] = 1
        
        # Fill small holes
        kernel = np.ones((3,3), np.uint8)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
        
        return (cleaned_mask * 255).astype(np.uint8)
    
    def smooth_mask_edges(self, mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Smooth mask edges to reduce artifacts."""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Apply morphological operations
        smoothed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, kernel)
        
        return smoothed
    
    def process_mask_directory(self, input_dir: Path, output_dir: Path = None):
        """Process all masks in a directory."""
        if output_dir is None:
            output_dir = input_dir
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        mask_files = list(input_dir.glob("*_mask.png"))
        
        for mask_file in mask_files:
            # Load mask
            mask = np.array(Image.open(mask_file))
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            
            # Process mask
            cleaned_mask = self.clean_mask(mask)
            smoothed_mask = self.smooth_mask_edges(cleaned_mask)
            
            # Save processed mask
            output_path = output_dir / mask_file.name
            Image.fromarray(smoothed_mask).save(output_path)
        
        logger.info(f"Processed {len(mask_files)} masks")


class MaskVisualizer:
    """Visualizes masks and defect statistics."""
    
    def __init__(self):
        pass
    
    def visualize_mask_overlay(self, image_path: Path, mask_path: Path, 
                              output_path: Path = None, alpha: float = 0.5):
        """Create overlay visualization of image and mask."""
        # Load image and mask
        image = np.array(Image.open(image_path))
        mask = np.array(Image.open(mask_path))
        
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        
        # Create colored overlay
        overlay = image.copy()
        overlay[mask > 0] = [255, 0, 0]  # Red for defects
        
        # Blend with original image
        result = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)
        
        if output_path:
            Image.fromarray(result).save(output_path)
        
        return result
    
    def plot_defect_statistics(self, validation_results: Dict[str, List[Dict]], 
                             output_path: Path = None):
        """Plot statistics about defect masks."""
        # Collect statistics
        defect_ratios = defaultdict(list)
        component_counts = defaultdict(list)
        
        for defect_type, results in validation_results.items():
            for result in results:
                stats = result['statistics']
                if stats:
                    defect_ratios[defect_type].append(stats.get('defect_ratio', 0))
                    component_counts[defect_type].append(stats.get('num_components', 0))
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Defect ratio distribution
        for defect_type, ratios in defect_ratios.items():
            ax1.hist(ratios, alpha=0.7, label=defect_type, bins=20)
        ax1.set_xlabel('Defect Ratio')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Defect Area Ratio Distribution')
        ax1.legend()
        
        # Component count distribution
        for defect_type, counts in component_counts.items():
            ax2.hist(counts, alpha=0.7, label=defect_type, bins=10)
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Connected Components Distribution')
        ax2.legend()
        
        # Average defect ratio by type
        avg_ratios = [np.mean(ratios) for ratios in defect_ratios.values()]
        defect_types = list(defect_ratios.keys())
        ax3.bar(defect_types, avg_ratios)
        ax3.set_ylabel('Average Defect Ratio')
        ax3.set_title('Average Defect Area by Type')
        ax3.tick_params(axis='x', rotation=45)
        
        # Average components by type
        avg_components = [np.mean(counts) for counts in component_counts.values()]
        ax4.bar(defect_types, avg_components)
        ax4.set_ylabel('Average Components')
        ax4.set_title('Average Components by Type')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Statistics plot saved to {output_path}")
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Mask utilities for fabric defect detection")
    parser.add_argument("--validate", 
                       help="Validate masks in ground truth directory")
    parser.add_argument("--clean", 
                       help="Clean masks in specified directory")
    parser.add_argument("--visualize", 
                       help="Create visualization for image and mask pairs")
    parser.add_argument("--output", 
                       help="Output directory for processed results")
    parser.add_argument("--plot_stats", action="store_true",
                       help="Generate statistics plots")
    
    args = parser.parse_args()
    
    if args.validate:
        validator = MaskValidator()
        ground_truth_dir = Path(args.validate)
        
        # Validate all masks
        results = validator.validate_dataset_masks(ground_truth_dir)
        
        # Generate and print report
        report = validator.generate_validation_report(results)
        validator.print_validation_report(report)
        
        # Generate statistics plot if requested
        if args.plot_stats:
            visualizer = MaskVisualizer()
            output_path = Path(args.output) / "mask_statistics.png" if args.output else None
            visualizer.plot_defect_statistics(results, output_path)
    
    elif args.clean:
        processor = MaskProcessor()
        input_dir = Path(args.clean)
        output_dir = Path(args.output) if args.output else input_dir
        
        processor.process_mask_directory(input_dir, output_dir)
        logger.info("Mask cleaning completed!")
    
    else:
        logger.error("Please specify an operation: --validate, --clean, or --visualize")


if __name__ == "__main__":
    main()
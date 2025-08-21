#!/usr/bin/env python3
"""
Real Defect Pattern Extraction and Augmentation System
Extracts real defect patterns from WFDD dataset and applies them to good fabric images.

Usage:
    python real_defect_extractor.py --wfdd_path datasets/WFDD --output_path patterns/ --extract
    python real_defect_extractor.py --patterns_path patterns/ --target_images datasets/custom/grid/test/good --output datasets/custom/grid/test --augment
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance
import random
from dataclasses import dataclass
from collections import defaultdict
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DefectPattern:
    """Represents an extracted defect pattern with metadata."""
    id: str
    fabric_type: str
    defect_type: str
    defect_region: np.ndarray
    mask: np.ndarray
    background_region: np.ndarray
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Calculate derived properties after initialization."""
        self.size = self.mask.shape[:2]
        self.area = np.sum(self.mask > 128)
        self.area_ratio = self.area / (self.size[0] * self.size[1])
        self.bbox = self._calculate_bbox()
        self.compactness = self._calculate_compactness()
    
    def _calculate_bbox(self) -> Tuple[int, int, int, int]:
        """Calculate bounding box of defect area."""
        coords = np.where(self.mask > 128)
        if len(coords[0]) == 0:
            return (0, 0, 0, 0)
        y1, x1 = coords[0].min(), coords[1].min()
        y2, x2 = coords[0].max(), coords[1].max()
        return (x1, y1, x2, y2)
    
    def _calculate_compactness(self) -> float:
        """Calculate shape compactness (circularity measure)."""
        if self.area == 0:
            return 0.0
        
        # Find contours
        mask_uint8 = (self.mask > 128).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Calculate perimeter of largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest_contour, True)
        
        if perimeter == 0:
            return 0.0
        
        # Compactness = 4π * area / perimeter²
        compactness = (4 * np.pi * self.area) / (perimeter * perimeter)
        return min(compactness, 1.0)  # Cap at 1.0


class DefectPatternLibrary:
    """Manages a collection of defect patterns with search capabilities."""
    
    def __init__(self):
        self.patterns: List[DefectPattern] = []
        self.fabric_index: Dict[str, List[DefectPattern]] = defaultdict(list)
        self.defect_type_index: Dict[str, List[DefectPattern]] = defaultdict(list)
        self.size_index: Dict[str, List[DefectPattern]] = defaultdict(list)
    
    def add_pattern(self, pattern: DefectPattern):
        """Add a defect pattern to the library."""
        self.patterns.append(pattern)
        self.fabric_index[pattern.fabric_type].append(pattern)
        self.defect_type_index[pattern.defect_type].append(pattern)
        
        # Index by size category
        if pattern.area < 1000:
            size_category = "small"
        elif pattern.area < 5000:
            size_category = "medium"
        else:
            size_category = "large"
        self.size_index[size_category].append(pattern)
    
    def search_patterns(self, fabric_type: str = None, defect_type: str = None, 
                       size_category: str = None, min_area: int = None, 
                       max_area: int = None) -> List[DefectPattern]:
        """Search patterns by various criteria."""
        candidates = self.patterns.copy()
        
        if fabric_type:
            candidates = [p for p in candidates if p.fabric_type == fabric_type]
        
        if defect_type:
            candidates = [p for p in candidates if p.defect_type == defect_type]
        
        if size_category:
            size_patterns = self.size_index[size_category]
            candidates = [p for p in candidates if p in size_patterns]
        
        if min_area is not None:
            candidates = [p for p in candidates if p.area >= min_area]
        
        if max_area is not None:
            candidates = [p for p in candidates if p.area <= max_area]
        
        return candidates
    
    def get_compatible_patterns(self, target_fabric: str, target_size: Tuple[int, int]) -> List[DefectPattern]:
        """Get patterns compatible with target fabric and image size."""
        max_pattern_size = min(target_size) // 3  # Pattern shouldn't exceed 1/3 of image
        max_area = max_pattern_size * max_pattern_size
        
        # First try same fabric type
        patterns = self.search_patterns(fabric_type=target_fabric, max_area=max_area)
        
        # If not enough patterns, try other fabric types
        if len(patterns) < 10:
            all_patterns = self.search_patterns(max_area=max_area)
            patterns.extend([p for p in all_patterns if p not in patterns])
        
        return patterns
    
    def save(self, filepath: str):
        """Save pattern library to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Pattern library saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'DefectPatternLibrary':
        """Load pattern library from disk."""
        with open(filepath, 'rb') as f:
            library = pickle.load(f)
        logger.info(f"Pattern library loaded from {filepath}")
        return library
    
    def get_stats(self) -> Dict:
        """Get library statistics."""
        fabric_stats = {fabric: len(patterns) for fabric, patterns in self.fabric_index.items()}
        defect_stats = {defect: len(patterns) for defect, patterns in self.defect_type_index.items()}
        size_stats = {size: len(patterns) for size, patterns in self.size_index.items()}
        
        return {
            "total_patterns": len(self.patterns),
            "fabric_types": fabric_stats,
            "defect_types": defect_stats,
            "size_distribution": size_stats
        }


class RealDefectExtractor:
    """Extracts real defect patterns from WFDD dataset."""
    
    def __init__(self, wfdd_path: str, patterns_cache_dir: str = "patterns"):
        self.wfdd_path = Path(wfdd_path)
        self.patterns_cache_dir = Path(patterns_cache_dir)
        self.patterns_cache_dir.mkdir(parents=True, exist_ok=True)
        self.library = DefectPatternLibrary()
        
        # Cache file path
        self.cache_file = self.patterns_cache_dir / "wfdd_patterns.pkl"
        
        # Fabric type mapping
        self.fabric_types = {
            "grey_cloth": "grey_cloth",
            "grid_cloth": "grid_cloth", 
            "pink_flower": "pink_flower",
            "yellow_cloth": "yellow_cloth"
        }
    
    def get_or_extract_patterns(self) -> DefectPatternLibrary:
        """Get patterns from cache or extract from WFDD if not cached."""
        # Check if cache exists
        if self.cache_file.exists():
            logger.info(f"📁 Loading cached patterns from {self.cache_file}")
            try:
                self.library = DefectPatternLibrary.load(str(self.cache_file))
                stats = self.library.get_stats()
                logger.info(f"✅ Loaded {stats['total_patterns']} cached patterns")
                logger.info(f"   Fabric types: {list(stats['fabric_types'].keys())}")
                logger.info(f"   Defect types: {list(stats['defect_types'].keys())}")
                return self.library
            except Exception as e:
                logger.warning(f"Failed to load cached patterns: {e}")
                logger.info("Proceeding with fresh extraction...")
        
        # Extract patterns if not cached
        logger.info("🔍 No cached patterns found. Starting fresh extraction from WFDD dataset...")
        library = self.extract_all_patterns()
        
        # Save to cache
        logger.info(f"💾 Saving patterns to cache: {self.cache_file}")
        library.save(str(self.cache_file))
        
        return library
    
    def extract_all_patterns(self) -> DefectPatternLibrary:
        """Extract all defect patterns from WFDD dataset."""
        logger.info("🔍 Starting defect pattern extraction from WFDD dataset...")
        
        total_patterns = 0
        
        for fabric_name, fabric_type in self.fabric_types.items():
            fabric_path = self.wfdd_path / fabric_name
            
            if not fabric_path.exists():
                logger.warning(f"Fabric directory not found: {fabric_path}")
                continue
            
            logger.info(f"Processing {fabric_name}...")
            patterns = self._extract_fabric_patterns(fabric_path, fabric_type)
            total_patterns += len(patterns)
            
            for pattern in patterns:
                self.library.add_pattern(pattern)
        
        logger.info(f"✅ Extracted {total_patterns} defect patterns from {len(self.fabric_types)} fabric types")
        return self.library
    
    def _extract_fabric_patterns(self, fabric_path: Path, fabric_type: str) -> List[DefectPattern]:
        """Extract defect patterns from a single fabric type."""
        patterns = []
        
        test_defect_path = fabric_path / "test"
        gt_path = fabric_path / "ground_truth"
        
        if not (test_defect_path.exists() and gt_path.exists()):
            logger.warning(f"Missing test or ground_truth directories for {fabric_type}")
            return patterns
        
        # Get all defect types for this fabric
        defect_types = [d.name for d in gt_path.iterdir() if d.is_dir()]
        
        for defect_type in defect_types:
            defect_dir = test_defect_path / defect_type
            mask_dir = gt_path / defect_type
            
            if defect_type == "good":  # Skip good images
                continue
                
            if not (defect_dir.exists() and mask_dir.exists()):
                logger.warning(f"Missing directories for {fabric_type}/{defect_type}")
                continue
            
            # Extract patterns for this defect type
            defect_patterns = self._extract_defect_type_patterns(
                defect_dir, mask_dir, fabric_type, defect_type
            )
            patterns.extend(defect_patterns)
            
            logger.info(f"  {defect_type}: {len(defect_patterns)} patterns")
        
        return patterns
    
    def _extract_defect_type_patterns(self, defect_dir: Path, mask_dir: Path, 
                                    fabric_type: str, defect_type: str) -> List[DefectPattern]:
        """Extract patterns for a specific defect type."""
        patterns = []
        
        # Get all defect images and corresponding masks
        defect_images = sorted(defect_dir.glob("*.png"))
        
        for defect_img_path in defect_images:
            # Find corresponding mask
            mask_path = mask_dir / f"{defect_img_path.stem}_mask.png"
            
            if not mask_path.exists():
                logger.warning(f"Missing mask for {defect_img_path}")
                continue
            
            try:
                # Load images
                defect_image = np.array(Image.open(defect_img_path))
                mask_image = np.array(Image.open(mask_path))
                
                # Extract pattern
                pattern = self._extract_single_pattern(
                    defect_image, mask_image, fabric_type, defect_type, defect_img_path.stem
                )
                
                if pattern:
                    patterns.append(pattern)
                    
            except Exception as e:
                logger.error(f"Error processing {defect_img_path}: {e}")
                continue
        
        return patterns
    
    def _extract_single_pattern(self, defect_image: np.ndarray, mask_image: np.ndarray,
                               fabric_type: str, defect_type: str, image_id: str) -> Optional[DefectPattern]:
        """Extract a single defect pattern from defect image and mask."""
        try:
            # Ensure mask is binary
            if len(mask_image.shape) == 3:
                mask_image = cv2.cvtColor(mask_image, cv2.COLOR_RGB2GRAY)
            
            # Threshold mask
            _, mask_binary = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)
            
            # Find defect area
            coords = np.where(mask_binary > 128)
            if len(coords[0]) == 0:
                return None
            
            # Calculate bounding box with padding
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            # Add padding (20% of defect size)
            padding_y = max(5, (y_max - y_min) // 5)
            padding_x = max(5, (x_max - x_min) // 5)
            
            y1 = max(0, y_min - padding_y)
            y2 = min(defect_image.shape[0], y_max + padding_y)
            x1 = max(0, x_min - padding_x)
            x2 = min(defect_image.shape[1], x_max + padding_x)
            
            # Extract regions
            defect_region = defect_image[y1:y2, x1:x2]
            mask_region = mask_binary[y1:y2, x1:x2]
            
            # Create background region (good fabric around defect)
            background_mask = (mask_region == 0).astype(np.uint8) * 255
            background_region = defect_image[y1:y2, x1:x2].copy()
            background_region[mask_region > 0] = 0  # Zero out defect areas
            
            # Calculate metadata
            metadata = {
                "original_size": defect_image.shape[:2],
                "extraction_bbox": (x1, y1, x2, y2),
                "source_image": image_id,
                "extraction_date": "2025-08-21"
            }
            
            # Create pattern
            pattern_id = f"{fabric_type}_{defect_type}_{image_id}"
            pattern = DefectPattern(
                id=pattern_id,
                fabric_type=fabric_type,
                defect_type=defect_type,
                defect_region=defect_region,
                mask=mask_region,
                background_region=background_region,
                metadata=metadata
            )
            
            return pattern
            
        except Exception as e:
            logger.error(f"Error extracting pattern from {image_id}: {e}")
            return None


class RealDefectAugmentor:
    """Applies real defect patterns to good fabric images."""
    
    def __init__(self, pattern_library: DefectPatternLibrary):
        self.library = pattern_library
        self.blend_methods = ["normal", "overlay", "multiply", "screen"]
    
    def augment_image(self, good_image: np.ndarray, target_fabric: str, 
                     num_defects: int = 1, defect_types: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random defect patterns to a good fabric image."""
        augmented_image = good_image.copy()
        combined_mask = np.zeros(good_image.shape[:2], dtype=np.uint8)
        
        # Get compatible patterns
        compatible_patterns = self.library.get_compatible_patterns(target_fabric, good_image.shape[:2])
        
        if not compatible_patterns:
            logger.warning(f"No compatible patterns found for {target_fabric}")
            return augmented_image, combined_mask
        
        # Filter by defect types if specified
        if defect_types:
            compatible_patterns = [p for p in compatible_patterns if p.defect_type in defect_types]
        
        if not compatible_patterns:
            logger.warning(f"No patterns found for specified defect types: {defect_types}")
            return augmented_image, combined_mask
        
        applied_regions = []  # Track applied defect regions to avoid overlap
        
        for _ in range(num_defects):
            # Select random pattern
            pattern = random.choice(compatible_patterns)
            
            # Find valid position
            position = self._find_valid_position(
                good_image.shape[:2], pattern.size, applied_regions
            )
            
            if position is None:
                logger.warning("Could not find valid position for defect pattern")
                continue
            
            # Apply pattern
            success = self._apply_pattern_at_position(
                augmented_image, combined_mask, pattern, position
            )
            
            if success:
                # Record applied region
                y, x = position
                h, w = pattern.size
                applied_regions.append((x, y, x + w, y + h))
        
        return augmented_image, combined_mask
    
    def _find_valid_position(self, image_size: Tuple[int, int], pattern_size: Tuple[int, int],
                           existing_regions: List[Tuple[int, int, int, int]], 
                           max_attempts: int = 50) -> Optional[Tuple[int, int]]:
        """Find a valid position for placing defect pattern."""
        img_h, img_w = image_size
        pattern_h, pattern_w = pattern_size
        
        # Ensure pattern fits in image
        if pattern_h >= img_h or pattern_w >= img_w:
            return None
        
        for _ in range(max_attempts):
            # Random position with margin from edges
            margin = 20
            y = random.randint(margin, img_h - pattern_h - margin)
            x = random.randint(margin, img_w - pattern_w - margin)
            
            # Check for overlap with existing defects
            new_region = (x, y, x + pattern_w, y + pattern_h)
            
            overlaps = False
            for existing in existing_regions:
                if self._regions_overlap(new_region, existing):
                    overlaps = True
                    break
            
            if not overlaps:
                return (y, x)
        
        return None
    
    def _regions_overlap(self, region1: Tuple[int, int, int, int], 
                        region2: Tuple[int, int, int, int]) -> bool:
        """Check if two regions overlap."""
        x1_1, y1_1, x2_1, y2_1 = region1
        x1_2, y1_2, x2_2, y2_2 = region2
        
        return not (x2_1 <= x1_2 or x2_2 <= x1_1 or y2_1 <= y1_2 or y2_2 <= y1_1)
    
    def _apply_pattern_at_position(self, image: np.ndarray, mask: np.ndarray,
                                 pattern: DefectPattern, position: Tuple[int, int]) -> bool:
        """Apply defect pattern at specified position."""
        try:
            y, x = position
            pattern_h, pattern_w = pattern.size
            
            # Extract target region
            target_region = image[y:y+pattern_h, x:x+pattern_w]
            
            # Apply defect using intelligent blending
            blended_region = self._blend_pattern(target_region, pattern)
            
            # Update image
            image[y:y+pattern_h, x:x+pattern_w] = blended_region
            
            # Update mask
            mask[y:y+pattern_h, x:x+pattern_w] = np.maximum(
                mask[y:y+pattern_h, x:x+pattern_w], pattern.mask
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying pattern at position {position}: {e}")
            return False
    
    def _blend_pattern(self, target_region: np.ndarray, pattern: DefectPattern) -> np.ndarray:
        """Intelligently blend defect pattern with target region."""
        # Use defect mask to blend only defect areas
        mask_norm = pattern.mask.astype(np.float32) / 255.0
        
        # Expand mask to match image channels
        if len(target_region.shape) == 3:
            mask_norm = np.expand_dims(mask_norm, axis=2)
            mask_norm = np.repeat(mask_norm, target_region.shape[2], axis=2)
        
        # Blend: target * (1 - mask) + defect * mask
        result = target_region.astype(np.float32) * (1.0 - mask_norm)
        result += pattern.defect_region.astype(np.float32) * mask_norm
        
        return result.astype(np.uint8)


def extract_patterns_from_wfdd(wfdd_path: str, output_path: str):
    """Extract all defect patterns from WFDD dataset with intelligent caching."""
    logger.info("🚀 Starting WFDD defect pattern extraction...")
    
    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use caching extractor - will load from cache if available
    extractor = RealDefectExtractor(wfdd_path, patterns_cache_dir=str(output_dir))
    library = extractor.get_or_extract_patterns()
    
    # Ensure library is saved to expected location
    library_path = output_dir / "defect_patterns.pkl"
    if not library_path.exists():
        library.save(str(library_path))
    
    # Save/update statistics
    stats = library.get_stats()
    stats_path = output_dir / "pattern_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    logger.info("✅ Pattern extraction completed!")
    logger.info(f"📊 Statistics:")
    logger.info(f"   Total patterns: {stats['total_patterns']}")
    logger.info(f"   Fabric types: {list(stats['fabric_types'].keys())}")
    logger.info(f"   Defect types: {list(stats['defect_types'].keys())}")
    logger.info(f"   Patterns saved to: {library_path}")


def augment_images_with_real_patterns(patterns_path: str, target_images_dir: str, 
                                    output_dir: str, target_fabric: str,
                                    images_per_defect: int = 50):
    """Augment good images with real defect patterns."""
    logger.info("🎨 Starting real defect augmentation...")
    
    # Load pattern library
    library_path = Path(patterns_path) / "defect_patterns.pkl"
    if not library_path.exists():
        logger.error(f"Pattern library not found: {library_path}")
        return
    
    library = DefectPatternLibrary.load(str(library_path))
    augmentor = RealDefectAugmentor(library)
    
    # Get target images
    target_dir = Path(target_images_dir)
    if not target_dir.exists():
        logger.error(f"Target images directory not found: {target_dir}")
        return
    
    image_files = list(target_dir.glob("*.png")) + list(target_dir.glob("*.jpg"))
    if not image_files:
        logger.error(f"No images found in {target_dir}")
        return
    
    logger.info(f"Found {len(image_files)} target images")
    
    # Get available defect types from library
    available_defects = list(library.defect_type_index.keys())
    logger.info(f"Available defect types: {available_defects}")
    
    # Create output directories
    output_path = Path(output_dir)
    gt_path = output_path.parent / "ground_truth"
    
    total_generated = 0
    
    for defect_type in available_defects:
        defect_output_dir = output_path / defect_type
        defect_gt_dir = gt_path / defect_type
        
        defect_output_dir.mkdir(parents=True, exist_ok=True)
        defect_gt_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating {images_per_defect} images for defect type: {defect_type}")
        
        for i in range(images_per_defect):
            # Select random good image
            good_image_path = random.choice(image_files)
            good_image = np.array(Image.open(good_image_path))
            
            # Apply defect pattern
            augmented_image, defect_mask = augmentor.augment_image(
                good_image, target_fabric, num_defects=1, defect_types=[defect_type]
            )
            
            # Save augmented image
            output_filename = f"{i+1:03d}.png"
            augmented_path = defect_output_dir / output_filename
            Image.fromarray(augmented_image).save(augmented_path)
            
            # Save ground truth mask
            mask_filename = f"{i+1:03d}_mask.png"
            mask_path = defect_gt_dir / mask_filename
            Image.fromarray(defect_mask).save(mask_path)
            
            total_generated += 1
    
    logger.info(f"✅ Real defect augmentation completed!")
    logger.info(f"   Generated {total_generated} augmented images")
    logger.info(f"   Output directory: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Real Defect Pattern Extraction and Augmentation")
    parser.add_argument("--mode", choices=["extract", "augment"], required=True,
                       help="Operation mode: extract patterns or augment images")
    
    # Extraction arguments
    parser.add_argument("--wfdd_path", 
                       help="Path to WFDD dataset (required for extract mode)")
    parser.add_argument("--patterns_output", default="patterns/",
                       help="Output directory for extracted patterns (default: patterns/)")
    
    # Augmentation arguments
    parser.add_argument("--patterns_path",
                       help="Path to extracted patterns directory (required for augment mode)")
    parser.add_argument("--target_images",
                       help="Directory containing good images to augment (required for augment mode)")
    parser.add_argument("--output_dir",
                       help="Output directory for augmented images (required for augment mode)")
    parser.add_argument("--target_fabric", default="grid_cloth",
                       help="Target fabric type for pattern matching (default: grid_cloth)")
    parser.add_argument("--images_per_defect", type=int, default=50,
                       help="Number of images to generate per defect type (default: 50)")
    
    args = parser.parse_args()
    
    if args.mode == "extract":
        if not args.wfdd_path:
            logger.error("--wfdd_path is required for extract mode")
            sys.exit(1)
        
        extract_patterns_from_wfdd(args.wfdd_path, args.patterns_output)
        
    elif args.mode == "augment":
        if not all([args.patterns_path, args.target_images, args.output_dir]):
            logger.error("--patterns_path, --target_images, and --output_dir are required for augment mode")
            sys.exit(1)
        
        augment_images_with_real_patterns(
            args.patterns_path, args.target_images, args.output_dir, 
            args.target_fabric, args.images_per_defect
        )


if __name__ == "__main__":
    main()
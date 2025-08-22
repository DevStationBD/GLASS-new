#!/usr/bin/env python3
"""
Fabric Defect Simulation Engine
Generates realistic fabric defects on normal fabric images.

Usage:
    python defect_simulator.py --input datasets/custom_grid/test/good --output datasets/custom_grid/test
"""

import os
import argparse
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import logging
from dataclasses import dataclass
from sklearn.cluster import KMeans
import colorsys
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DefectParams:
    """Parameters for defect generation."""
    min_size: int
    max_size: int
    min_count: int
    max_count: int
    intensity: float
    opacity: float
    color_variation: float


class FabricDefectSimulator:
    """Simulates various types of fabric defects on normal images."""
    
    def __init__(self, seed: int = 42, min_contrast_threshold: float = 50.0, 
                 min_visibility_score: float = 0.3, config_file: Optional[str] = None):
        # Set random seeds first
        random.seed(seed)
        np.random.seed(seed)
        
        # Load configuration if provided
        if config_file:
            config = self.load_config(config_file)
            
            # Update visibility parameters from config
            self.min_contrast_threshold = config.get('visibility', {}).get('min_contrast_threshold', min_contrast_threshold)
            self.min_visibility_score = config.get('visibility', {}).get('min_visibility_score', min_visibility_score)
            
            # Update seed from config if specified
            config_seed = config.get('seed', seed)
            if config_seed != seed:
                random.seed(config_seed)
                np.random.seed(config_seed)
            
            # Load defect parameters from config
            self.defect_params = self.load_defect_params_from_config(config)
            
            # Load color palettes from config
            self.color_palettes = config.get('color_palettes', {})
            
            logger.info(f"✅ Loaded configuration from: {config_file}")
            logger.info(f"   Contrast threshold: {self.min_contrast_threshold}")
            logger.info(f"   Visibility score: {self.min_visibility_score}")
        else:
            # Use default parameters
            self.min_contrast_threshold = min_contrast_threshold
            self.min_visibility_score = min_visibility_score
            self.defect_params = self.get_default_defect_params()
            self.color_palettes = self.get_default_color_palettes()
    
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_path = Path(config_file)
        
        if not config_path.exists():
            # Try relative to script directory
            script_dir = Path(__file__).parent
            config_path = script_dir / config_file
            
        if not config_path.exists():
            # Try in configs directory
            config_path = script_dir / "configs" / config_file
            
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
    
    def load_defect_params_from_config(self, config: Dict[str, Any]) -> Dict[str, DefectParams]:
        """Load defect parameters from configuration."""
        defect_params = {}
        defects_config = config.get('defects', {})
        
        # Default parameters in case config is incomplete
        defaults = self.get_default_defect_params()
        
        for defect_type in ['hole', 'foreign_yarn', 'missing_yarn', 'slab', 'spot']:
            if defect_type in defects_config:
                defect_config = defects_config[defect_type]
                defect_params[defect_type] = DefectParams(
                    min_size=defect_config.get('min_size', defaults[defect_type].min_size),
                    max_size=defect_config.get('max_size', defaults[defect_type].max_size),
                    min_count=defect_config.get('min_count', defaults[defect_type].min_count),
                    max_count=defect_config.get('max_count', defaults[defect_type].max_count),
                    intensity=defect_config.get('intensity', defaults[defect_type].intensity),
                    opacity=defect_config.get('opacity', defaults[defect_type].opacity),
                    color_variation=defect_config.get('color_variation', defaults[defect_type].color_variation)
                )
            else:
                # Use default parameters if not specified in config
                defect_params[defect_type] = defaults[defect_type]
        
        return defect_params
    
    def get_default_defect_params(self) -> Dict[str, DefectParams]:
        """Get default defect parameters."""
        return {
            "hole": DefectParams(
                min_size=2, max_size=8, min_count=1, max_count=3,
                intensity=0.6, opacity=0.8, color_variation=0.1
            ),
            "foreign_yarn": DefectParams(
                min_size=1, max_size=2, min_count=3, max_count=8,
                intensity=0.5, opacity=0.6, color_variation=0.8
            ),
            "missing_yarn": DefectParams(
                min_size=1, max_size=3, min_count=1, max_count=3,
                intensity=0.4, opacity=0.7, color_variation=0.2
            ),
            "slab": DefectParams(
                min_size=2, max_size=6, min_count=2, max_count=5,
                intensity=0.3, opacity=0.5, color_variation=0.4
            ),
            "spot": DefectParams(
                min_size=3, max_size=10, min_count=1, max_count=2,
                intensity=0.4, opacity=0.6, color_variation=0.6
            )
        }
    
    def get_default_color_palettes(self) -> Dict[str, List[List[int]]]:
        """Get default color palettes."""
        return {
            "foreign_yarn": [
                [255, 0, 0],    # Red
                [0, 255, 0],    # Green
                [0, 0, 255],    # Blue
                [255, 255, 0],  # Yellow
                [255, 0, 255],  # Magenta
                [0, 255, 255],  # Cyan
                [255, 128, 0],  # Orange
                [128, 0, 255],  # Purple
            ],
            "spot": [
                [139, 69, 19],   # Brown (dirt)
                [85, 85, 85],    # Dark gray (oil)
                [160, 82, 45],   # Saddle brown
                [105, 105, 105], # Dim gray
                [128, 128, 0],   # Olive (stain)
                [75, 0, 130],    # Indigo (ink)
            ]
        }
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration parameters."""
        try:
            # Validate visibility parameters
            visibility = config.get('visibility', {})
            if 'min_contrast_threshold' in visibility:
                threshold = visibility['min_contrast_threshold']
                if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 200:
                    logger.warning(f"Invalid contrast threshold: {threshold}. Should be 0-200.")
                    return False
            
            if 'min_visibility_score' in visibility:
                score = visibility['min_visibility_score']
                if not isinstance(score, (int, float)) or score < 0 or score > 1:
                    logger.warning(f"Invalid visibility score: {score}. Should be 0-1.")
                    return False
            
            # Validate defect parameters
            defects = config.get('defects', {})
            for defect_type, params in defects.items():
                if not isinstance(params, dict):
                    logger.warning(f"Invalid defect parameters for {defect_type}")
                    return False
                
                # Check required numeric parameters
                for param in ['min_size', 'max_size', 'min_count', 'max_count']:
                    if param in params:
                        value = params[param]
                        if not isinstance(value, int) or value < 0:
                            logger.warning(f"Invalid {param} for {defect_type}: {value}")
                            return False
                
                # Check float parameters
                for param in ['intensity', 'opacity', 'color_variation']:
                    if param in params:
                        value = params[param]
                        if not isinstance(value, (int, float)) or value < 0 or value > 1:
                            logger.warning(f"Invalid {param} for {defect_type}: {value}")
                            return False
            
            logger.info("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False
    
    def analyze_fabric_colors(self, image: np.ndarray, region_mask: np.ndarray = None) -> Dict[str, Any]:
        """Analyze fabric colors in the specified region to determine optimal defect colors."""
        if region_mask is not None:
            # Analyze only the defect region
            masked_pixels = image[region_mask > 0].reshape(-1, 3)
        else:
            # Analyze entire image
            masked_pixels = image.reshape(-1, 3)
        
        if len(masked_pixels) == 0:
            return {"mean_color": [128, 128, 128], "brightness": 128, "contrast": 0}
        
        # Calculate basic statistics
        mean_color = np.mean(masked_pixels, axis=0).astype(int)
        brightness = np.mean(cv2.cvtColor(masked_pixels.reshape(1, -1, 3), cv2.COLOR_RGB2GRAY))
        
        # Calculate local contrast (standard deviation of brightness)
        gray_pixels = cv2.cvtColor(masked_pixels.reshape(1, -1, 3), cv2.COLOR_RGB2GRAY)
        contrast = np.std(gray_pixels)
        
        return {
            "mean_color": mean_color,
            "brightness": brightness,
            "contrast": contrast,
            "pixel_count": len(masked_pixels)
        }
    
    def calculate_color_contrast(self, color1: np.ndarray, color2: np.ndarray) -> float:
        """Calculate perceptual color contrast between two RGB colors."""
        # Convert to LAB color space for perceptual distance
        lab1 = cv2.cvtColor(np.uint8([[color1]]), cv2.COLOR_RGB2LAB)[0, 0]
        lab2 = cv2.cvtColor(np.uint8([[color2]]), cv2.COLOR_RGB2LAB)[0, 0]
        
        # Calculate Delta E (CIE76)
        delta_e = np.sqrt(np.sum((lab1.astype(float) - lab2.astype(float)) ** 2))
        return delta_e
    
    def generate_contrasting_color(self, fabric_color: np.ndarray, defect_type: str) -> np.ndarray:
        """Generate a color that contrasts well with the fabric color."""
        # Convert to HSV for easier manipulation
        hsv_fabric = cv2.cvtColor(np.uint8([[fabric_color]]), cv2.COLOR_RGB2HSV)[0, 0]
        h, s, v = hsv_fabric
        
        if defect_type == "foreign_yarn":
            # For foreign yarn, use complementary or highly contrasting colors
            contrasting_colors = [
                [255, 0, 0],    # Red
                [0, 255, 0],    # Green
                [0, 0, 255],    # Blue
                [255, 255, 0],  # Yellow
                [255, 0, 255],  # Magenta
                [0, 255, 255],  # Cyan
                [255, 128, 0],  # Orange
                [128, 0, 255],  # Purple
            ]
            
            # Find the color with maximum contrast
            best_color = contrasting_colors[0]
            max_contrast = 0
            
            for color in contrasting_colors:
                contrast = self.calculate_color_contrast(fabric_color, np.array(color))
                if contrast > max_contrast:
                    max_contrast = contrast
                    best_color = color
            
            # Ensure minimum contrast threshold
            if max_contrast < self.min_contrast_threshold:
                # Force high contrast by using very bright or very dark color
                if v > 128:  # If fabric is bright, use dark color
                    best_color = [50, 50, 50]
                else:  # If fabric is dark, use bright color
                    best_color = [255, 255, 255]
            
            return np.array(best_color)
            
        elif defect_type == "spot":
            # For spots, use realistic stain colors that contrast with fabric
            realistic_spot_colors = [
                [139, 69, 19],   # Brown (dirt)
                [85, 85, 85],    # Dark gray (oil)
                [160, 82, 45],   # Saddle brown
                [105, 105, 105], # Dim gray
                [128, 128, 0],   # Olive (stain)
                [75, 0, 130],    # Indigo (ink)
            ]
            
            # Find the most contrasting realistic color
            best_color = realistic_spot_colors[0]
            max_contrast = 0
            
            for color in realistic_spot_colors:
                contrast = self.calculate_color_contrast(fabric_color, np.array(color))
                if contrast > max_contrast:
                    max_contrast = contrast
                    best_color = color
            
            # If no realistic color provides enough contrast, create a custom one
            if max_contrast < self.min_contrast_threshold:
                if v > 128:  # Bright fabric - use dark stain
                    best_color = [60, 40, 30]  # Dark brown
                else:  # Dark fabric - use lighter stain
                    best_color = [180, 140, 100]  # Light brown
            
            return np.array(best_color)
            
        elif defect_type == "hole":
            # For holes, always use very dark colors that contrast with fabric
            if v > 128:  # Bright fabric - use very dark hole
                return np.array([20, 15, 10])  # Very dark brownish
            else:  # Dark fabric - use darker hole but not pure black
                return np.array([10, 8, 5])   # Even darker
            
        elif defect_type in ["missing_yarn", "slab"]:
            # For missing yarn and slab, use fabric-based contrast
            if v > 128:  # Bright fabric - darken
                contrast_color = np.clip(fabric_color * 0.4, 0, 255).astype(int)
            else:  # Dark fabric - lighten or darken more
                contrast_color = np.clip(fabric_color * 1.6, 0, 255).astype(int)
            
            return contrast_color
        
        return fabric_color  # Fallback
    
    def calculate_adaptive_opacity(self, defect_color: np.ndarray, fabric_color: np.ndarray, 
                                 base_opacity: float) -> float:
        """Calculate adaptive opacity based on color contrast."""
        contrast = self.calculate_color_contrast(defect_color, fabric_color)
        
        # If contrast is already high, we can use lower opacity
        if contrast > self.min_contrast_threshold * 1.5:
            return max(base_opacity * 0.7, 0.4)
        # If contrast is low, increase opacity
        elif contrast < self.min_contrast_threshold:
            return min(base_opacity * 1.5, 0.9)
        else:
            return base_opacity
    
    def ensure_defect_visibility(self, original_image: np.ndarray, defect_image: np.ndarray, 
                               mask: np.ndarray, defect_type: str) -> np.ndarray:
        """Post-process defect to ensure it's visible."""
        if np.sum(mask) == 0:  # No defect area
            return defect_image
        
        # Analyze the defect region
        defect_indices = mask > 0
        original_region = original_image[defect_indices]
        defect_region = defect_image[defect_indices]
        
        if len(original_region) == 0:
            return defect_image
        
        # Calculate visibility score (difference between original and defected regions)
        visibility_score = np.mean(np.abs(defect_region.astype(float) - original_region.astype(float))) / 255.0
        
        # If visibility is too low, enhance the defect
        if visibility_score < self.min_visibility_score:
            enhancement_factor = self.min_visibility_score / max(visibility_score, 0.01)
            enhancement_factor = min(enhancement_factor, 2.0)  # Cap enhancement
            
            # Calculate fabric color
            fabric_analysis = self.analyze_fabric_colors(original_image, mask)
            fabric_color = fabric_analysis["mean_color"]
            
            if defect_type in ["foreign_yarn", "spot"]:
                # Enhance color contrast
                optimal_color = self.generate_contrasting_color(fabric_color, defect_type)
                enhanced_region = (0.6 * optimal_color + 0.4 * defect_region).astype(np.uint8)
                defect_image[defect_indices] = enhanced_region
                
            elif defect_type in ["hole", "missing_yarn"]:
                # Enhance darkness for holes and missing yarns
                darkening_factor = 0.8 - (visibility_score * 2)  # More darkening for less visible defects
                darkening_factor = max(darkening_factor, 0.1)  # Don't make completely black
                defect_image[defect_indices] = (defect_region * darkening_factor).astype(np.uint8)
                
            elif defect_type == "slab":
                # Enhance texture and brightness differences
                brightness_shift = 40 if fabric_analysis["brightness"] > 128 else -40
                enhanced_region = np.clip(defect_region.astype(float) + brightness_shift, 0, 255)
                defect_image[defect_indices] = enhanced_region.astype(np.uint8)
        
        return defect_image
    
    def simulate_hole_defect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate hole defects - circular/irregular missing fabric areas."""
        h, w = image.shape[:2]
        defect_image = image.copy()
        mask = np.zeros((h, w), dtype=np.uint8)
        
        params = self.defect_params["hole"]
        num_holes = random.randint(params.min_count, params.max_count)
        
        for _ in range(num_holes):
            # Random position avoiding edges
            x = random.randint(params.max_size, w - params.max_size)
            y = random.randint(params.max_size, h - params.max_size)
            
            # Random hole size
            radius = random.randint(params.min_size, params.max_size)
            
            # Create irregular hole shape
            hole_mask = np.zeros((h, w), dtype=np.uint8)
            
            # Base circle
            cv2.circle(hole_mask, (x, y), radius, 255, -1)
            
            # Add irregularity with random distortion
            if random.random() > 0.5:
                # Add some irregular edges
                for i in range(8):
                    angle = i * 45
                    dx = int(radius * 0.3 * np.cos(np.radians(angle)) * random.uniform(0.5, 1.5))
                    dy = int(radius * 0.3 * np.sin(np.radians(angle)) * random.uniform(0.5, 1.5))
                    small_radius = max(1, radius//3)
                    cv2.circle(hole_mask, (x + dx, y + dy), small_radius, 255, -1)
            
            # Apply hole effect with enhanced visibility
            hole_indices = hole_mask > 0
            
            # Analyze fabric colors in the hole region
            fabric_analysis = self.analyze_fabric_colors(image, hole_mask)
            fabric_color = fabric_analysis["mean_color"]
            
            # Generate optimal contrasting hole color (should be very dark)
            optimal_color = self.generate_contrasting_color(fabric_color, "hole")
            
            # Calculate adaptive opacity based on contrast
            adaptive_opacity = self.calculate_adaptive_opacity(optimal_color, fabric_color, params.opacity)
            
            # Apply hole with enhanced visibility
            defect_image[hole_indices] = (adaptive_opacity * optimal_color + 
                                       (1 - adaptive_opacity) * defect_image[hole_indices]).astype(np.uint8)
            
            # Update mask
            mask[hole_indices] = 255
        
        return defect_image, mask
    
    def simulate_foreign_yarn_defect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate foreign yarn defects - different colored threads."""
        h, w = image.shape[:2]
        defect_image = image.copy()
        mask = np.zeros((h, w), dtype=np.uint8)
        
        params = self.defect_params["foreign_yarn"]
        num_yarns = random.randint(params.min_count, params.max_count)
        
        # Define foreign yarn colors (different from typical fabric colors)
        foreign_colors = [
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green
            [0, 0, 255],    # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
            [255, 128, 0],  # Orange
            [128, 0, 255],  # Purple
        ]
        
        for _ in range(num_yarns):
            # Random yarn path
            start_x = random.randint(0, w-1)
            start_y = random.randint(0, h-1)
            
            # Random direction and length
            angle = random.uniform(0, 2 * np.pi)
            length = random.randint(20, min(w, h) // 2)
            
            end_x = int(start_x + length * np.cos(angle))
            end_y = int(start_y + length * np.sin(angle))
            
            # Clamp to image boundaries
            end_x = max(0, min(w-1, end_x))
            end_y = max(0, min(h-1, end_y))
            
            # Random yarn thickness
            thickness = random.randint(params.min_size, params.max_size)
            
            # Draw yarn line first to get the mask
            yarn_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.line(yarn_mask, (start_x, start_y), (end_x, end_y), 255, thickness)
            yarn_indices = yarn_mask > 0
            
            # Analyze fabric colors in the yarn region
            fabric_analysis = self.analyze_fabric_colors(image, yarn_mask)
            fabric_color = fabric_analysis["mean_color"]
            
            # Generate optimal contrasting color for this fabric
            optimal_color = self.generate_contrasting_color(fabric_color, "foreign_yarn")
            
            # Calculate adaptive opacity based on contrast
            adaptive_opacity = self.calculate_adaptive_opacity(optimal_color, fabric_color, params.opacity)
            
            # Apply foreign yarn with adaptive opacity
            defect_image[yarn_indices] = (adaptive_opacity * optimal_color + 
                                       (1 - adaptive_opacity) * defect_image[yarn_indices]).astype(np.uint8)
            
            # Update mask
            mask[yarn_indices] = 255
        
        return defect_image, mask
    
    def simulate_missing_yarn_defect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate missing yarn defects - linear gaps where threads should be."""
        h, w = image.shape[:2]
        defect_image = image.copy()
        mask = np.zeros((h, w), dtype=np.uint8)
        
        params = self.defect_params["missing_yarn"]
        num_gaps = random.randint(params.min_count, params.max_count)
        
        for _ in range(num_gaps):
            # Choose direction (horizontal or vertical)
            if random.random() > 0.5:
                # Horizontal missing yarn
                y = random.randint(0, h-1)
                start_x = random.randint(0, w//4)
                end_x = random.randint(3*w//4, w-1)
                thickness = random.randint(params.min_size, params.max_size)
                
                gap_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.line(gap_mask, (start_x, y), (end_x, y), 255, thickness)
            else:
                # Vertical missing yarn
                x = random.randint(0, w-1)
                start_y = random.randint(0, h//4)
                end_y = random.randint(3*h//4, h-1)
                thickness = random.randint(params.min_size, params.max_size)
                
                gap_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.line(gap_mask, (x, start_y), (x, end_y), 255, thickness)
            
            # Apply missing yarn effect with enhanced visibility
            gap_indices = gap_mask > 0
            
            # Analyze fabric colors in the gap region
            fabric_analysis = self.analyze_fabric_colors(image, gap_mask)
            fabric_color = fabric_analysis["mean_color"]
            
            # Generate optimal contrasting color for missing yarn
            optimal_color = self.generate_contrasting_color(fabric_color, "missing_yarn")
            
            # Calculate adaptive opacity based on contrast
            adaptive_opacity = self.calculate_adaptive_opacity(optimal_color, fabric_color, params.opacity)
            
            # Apply missing yarn with enhanced visibility
            defect_image[gap_indices] = (adaptive_opacity * optimal_color + 
                                       (1 - adaptive_opacity) * defect_image[gap_indices]).astype(np.uint8)
            
            # Update mask
            mask[gap_indices] = 255
        
        return defect_image, mask
    
    def simulate_slab_defect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate slab defects - thick yarn sections with uneven texture."""
        h, w = image.shape[:2]
        defect_image = image.copy()
        mask = np.zeros((h, w), dtype=np.uint8)
        
        params = self.defect_params["slab"]
        num_slabs = random.randint(params.min_count, params.max_count)
        
        for _ in range(num_slabs):
            # Random slab position and size
            x = random.randint(params.max_size, w - params.max_size)
            y = random.randint(params.max_size, h - params.max_size)
            
            # Slab dimensions (elongated shape)
            if random.random() > 0.5:
                # Horizontal slab
                slab_w = random.randint(20, 60)
                slab_h = random.randint(params.min_size, params.max_size)
            else:
                # Vertical slab
                slab_w = random.randint(params.min_size, params.max_size)
                slab_h = random.randint(20, 60)
            
            # Create slab mask
            slab_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(slab_mask, (x, y), (x + slab_w, y + slab_h), 255, -1)
            
            # Apply slab effect with enhanced visibility
            slab_indices = slab_mask > 0
            
            # Analyze fabric colors in the slab region
            fabric_analysis = self.analyze_fabric_colors(image, slab_mask)
            fabric_color = fabric_analysis["mean_color"]
            
            # Generate optimal contrasting slab color
            optimal_color = self.generate_contrasting_color(fabric_color, "slab")
            
            # Calculate adaptive opacity based on contrast
            adaptive_opacity = self.calculate_adaptive_opacity(optimal_color, fabric_color, params.opacity)
            
            # Apply slab with enhanced visibility
            defect_image[slab_indices] = (adaptive_opacity * optimal_color + 
                                       (1 - adaptive_opacity) * defect_image[slab_indices]).astype(np.uint8)
            
            # Add texture variation to make it look more realistic
            texture_noise = np.random.normal(0, 10, defect_image[slab_indices].shape)
            defect_image[slab_indices] = np.clip(
                defect_image[slab_indices].astype(float) + texture_noise,
                0, 255
            ).astype(np.uint8)
            
            # Update mask
            mask[slab_indices] = 255
        
        return defect_image, mask
    
    def simulate_spot_defect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate spot defects - stains and contamination marks."""
        h, w = image.shape[:2]
        defect_image = image.copy()
        mask = np.zeros((h, w), dtype=np.uint8)
        
        params = self.defect_params["spot"]
        num_spots = random.randint(params.min_count, params.max_count)
        
        # Define spot colors (stains, oil, dirt, etc.)
        spot_colors = [
            [139, 69, 19],   # Brown (dirt)
            [85, 85, 85],    # Dark gray (oil)
            [160, 82, 45],   # Saddle brown
            [105, 105, 105], # Dim gray
            [128, 128, 0],   # Olive (stain)
            [75, 0, 130],    # Indigo (ink)
        ]
        
        for _ in range(num_spots):
            # Random spot position
            x = random.randint(params.max_size, w - params.max_size)
            y = random.randint(params.max_size, h - params.max_size)
            
            # Random spot size
            radius = random.randint(params.min_size, params.max_size)
            
            # Create irregular spot shape
            spot_mask = np.zeros((h, w), dtype=np.uint8)
            
            # Base ellipse with random orientation
            angle = random.randint(0, 180)
            axes = (radius, int(radius * random.uniform(0.6, 1.4)))
            cv2.ellipse(spot_mask, (x, y), axes, angle, 0, 360, 255, -1)
            
            # Add irregular edges
            for i in range(random.randint(2, 5)):
                offset_x = random.randint(-radius//2, radius//2)
                offset_y = random.randint(-radius//2, radius//2)
                small_radius = max(1, random.randint(radius//4, max(1, radius//2)))
                cv2.circle(spot_mask, (x + offset_x, y + offset_y), 
                          small_radius, 255, -1)
            
            # Apply spot color with smart blending
            spot_indices = spot_mask > 0
            
            # Analyze fabric colors in the spot region
            fabric_analysis = self.analyze_fabric_colors(image, spot_mask)
            fabric_color = fabric_analysis["mean_color"]
            
            # Generate optimal contrasting spot color
            optimal_color = self.generate_contrasting_color(fabric_color, "spot")
            
            # Calculate adaptive opacity based on contrast
            adaptive_opacity = self.calculate_adaptive_opacity(optimal_color, fabric_color, params.opacity)
            
            defect_image[spot_indices] = (adaptive_opacity * optimal_color + 
                                       (1 - adaptive_opacity) * defect_image[spot_indices]).astype(np.uint8)
            
            # Add slight blur to make it look more natural
            spot_region = defect_image[spot_indices]
            if len(spot_region) > 0:
                # Apply gaussian blur locally
                temp_image = defect_image.copy()
                temp_image = cv2.GaussianBlur(temp_image, (3, 3), 0.5)
                defect_image[spot_indices] = temp_image[spot_indices]
            
            # Update mask
            mask[spot_indices] = 255
        
        return defect_image, mask
    
    def generate_defect(self, image: np.ndarray, defect_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Generate specified defect type on the image."""
        if defect_type == "hole":
            return self.simulate_hole_defect(image)
        elif defect_type == "foreign_yarn":
            return self.simulate_foreign_yarn_defect(image)
        elif defect_type == "missing_yarn":
            return self.simulate_missing_yarn_defect(image)
        elif defect_type == "slab":
            return self.simulate_slab_defect(image)
        elif defect_type == "spot":
            return self.simulate_spot_defect(image)
        else:
            raise ValueError(f"Unknown defect type: {defect_type}")
    
    def process_dataset(self, input_dir: Path, output_base_dir: Path, 
                       defect_types: List[str], images_per_defect: int = 50):
        """Process entire dataset to generate defects."""
        input_images = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
        
        if not input_images:
            logger.error(f"No images found in {input_dir}")
            return False
        
        logger.info(f"Found {len(input_images)} input images")
        
        for defect_type in defect_types:
            logger.info(f"Generating {defect_type} defects...")
            
            defect_output_dir = output_base_dir / defect_type
            defect_output_dir.mkdir(parents=True, exist_ok=True)
            
            mask_output_dir = output_base_dir.parent / "ground_truth" / defect_type
            mask_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate specified number of defective images
            for i in range(images_per_defect):
                # Select random input image
                input_image_path = random.choice(input_images)
                
                # Load image
                image = cv2.imread(str(input_image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Generate defect
                defect_image, mask = self.generate_defect(image, defect_type)
                
                # Save defective image
                output_filename = f"{i+1:03d}.png"
                defect_output_path = defect_output_dir / output_filename
                
                defect_image_pil = Image.fromarray(defect_image)
                defect_image_pil.save(defect_output_path)
                
                # Save mask
                mask_filename = f"{i+1:03d}_mask.png"
                mask_output_path = mask_output_dir / mask_filename
                
                mask_image = Image.fromarray(mask)
                mask_image.save(mask_output_path)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Generated {i + 1}/{images_per_defect} {defect_type} defects")
        
        logger.info("✅ Defect generation completed!")
        return True


def main():
    parser = argparse.ArgumentParser(description="Generate fabric defects on normal images")
    parser.add_argument("--input", required=True,
                       help="Input directory containing good images")
    parser.add_argument("--output", required=True,
                       help="Output base directory for defective images")
    parser.add_argument("--defect_types", nargs="+",
                       default=["hole", "foreign_yarn", "missing_yarn", "slab", "spot"],
                       help="Types of defects to generate")
    parser.add_argument("--images_per_defect", type=int, default=50,
                       help="Number of images to generate per defect type")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--min_contrast_threshold", type=float, default=50.0,
                       help="Minimum perceptual contrast threshold for defect visibility (default: 50.0)")
    parser.add_argument("--min_visibility_score", type=float, default=0.3,
                       help="Minimum visibility score for defects (0-1, default: 0.3)")
    parser.add_argument("--config", type=str, 
                       help="Configuration file path (YAML format). Overrides other parameters if specified.")
    
    args = parser.parse_args()
    
    # Create defect simulator with visibility parameters or config file
    if args.config:
        # Use configuration file
        simulator = FabricDefectSimulator(
            seed=args.seed,
            min_contrast_threshold=args.min_contrast_threshold,
            min_visibility_score=args.min_visibility_score,
            config_file=args.config
        )
        
        # Override images_per_defect and defect_types from config if available
        try:
            config = simulator.load_config(args.config)
            dataset_config = config.get('dataset', {})
            
            # Use config values if not explicitly provided via command line
            if hasattr(args, 'images_per_defect') and not args.images_per_defect != 50:  # Default value
                args.images_per_defect = dataset_config.get('images_per_defect', args.images_per_defect)
            
            if hasattr(args, 'defect_types') and args.defect_types == ["hole", "foreign_yarn", "missing_yarn", "slab", "spot"]:  # Default value
                args.defect_types = dataset_config.get('defect_types', args.defect_types)
                
        except Exception as e:
            logger.warning(f"Could not load dataset config: {e}. Using command line arguments.")
    else:
        # Use command line parameters
        simulator = FabricDefectSimulator(
            seed=args.seed,
            min_contrast_threshold=args.min_contrast_threshold,
            min_visibility_score=args.min_visibility_score
        )
    
    # Process dataset
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    success = simulator.process_dataset(
        input_dir=input_dir,
        output_base_dir=output_dir,
        defect_types=args.defect_types,
        images_per_defect=args.images_per_defect
    )
    
    if success:
        logger.info(f"Defect simulation completed successfully!")
        logger.info(f"Defective images saved to: {output_dir}")
        logger.info(f"Masks saved to: {output_dir.parent / 'ground_truth'}")


if __name__ == "__main__":
    main()
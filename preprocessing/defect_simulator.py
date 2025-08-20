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
from typing import Tuple, List, Dict, Any
import logging
from dataclasses import dataclass

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
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
        # Defect parameters for different types - smaller, more realistic defects
        self.defect_params = {
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
            
            # Apply hole effect (darken significantly but not pure black)
            hole_indices = hole_mask > 0
            # Darken the area while keeping some fabric texture
            defect_image[hole_indices] = (defect_image[hole_indices] * 0.2).astype(np.uint8)
            
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
            
            # Random foreign color
            foreign_color = random.choice(foreign_colors)
            
            # Draw yarn line
            yarn_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.line(yarn_mask, (start_x, start_y), (end_x, end_y), 255, thickness)
            
            # Apply foreign yarn with some transparency
            yarn_indices = yarn_mask > 0
            alpha = params.opacity
            defect_image[yarn_indices] = (alpha * np.array(foreign_color) + 
                                       (1 - alpha) * defect_image[yarn_indices]).astype(np.uint8)
            
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
            
            # Apply missing yarn effect (darker/shadowed area)
            gap_indices = gap_mask > 0
            defect_image[gap_indices] = (defect_image[gap_indices] * 
                                       (1 - params.intensity)).astype(np.uint8)
            
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
            
            # Add texture variation to slab
            slab_indices = slab_mask > 0
            
            # Create thicker appearance by adding brightness variation
            texture_noise = np.random.normal(0, 20, defect_image[slab_indices].shape)
            defect_image[slab_indices] = np.clip(
                defect_image[slab_indices].astype(float) + texture_noise,
                0, 255
            ).astype(np.uint8)
            
            # Add slight color shift to simulate thick yarn
            color_shift = np.array([10, 5, -5]) * random.uniform(0.5, 1.5)
            defect_image[slab_indices] = np.clip(
                defect_image[slab_indices].astype(float) + color_shift,
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
            
            # Apply spot color with blending
            spot_indices = spot_mask > 0
            spot_color = random.choice(spot_colors)
            alpha = params.opacity
            
            defect_image[spot_indices] = (alpha * np.array(spot_color) + 
                                       (1 - alpha) * defect_image[spot_indices]).astype(np.uint8)
            
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
    
    args = parser.parse_args()
    
    # Create defect simulator
    simulator = FabricDefectSimulator(seed=args.seed)
    
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
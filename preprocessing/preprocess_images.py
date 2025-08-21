#!/usr/bin/env python3
"""
Image Preprocessing and Train/Test Split Script
Processes raw fabric images and splits them into training and testing sets.

Usage:
    python preprocess_images.py --source raw-data/custom-grid/good-images --target datasets/custom_grid
"""

import os
import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Handles image preprocessing and train/test splitting for fabric datasets."""
    
    def __init__(self, source_path: str, target_path: str, 
                 image_size: int = 288, train_ratio: float = 0.8):
        self.source_path = Path(source_path)
        self.target_path = Path(target_path)
        self.image_size = image_size
        self.train_ratio = train_ratio
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Quality enhancement settings
        self.enhancement_params = {
            'contrast': 1.1,
            'brightness': 1.0,
            'sharpness': 1.05,
            'color': 1.0
        }
    
    def get_image_files(self) -> List[Path]:
        """Get all supported image files from source directory."""
        image_files = []
        for file_path in self.source_path.iterdir():
            if file_path.suffix.lower() in self.supported_formats:
                image_files.append(file_path)
        
        logger.info(f"Found {len(image_files)} image files in {self.source_path}")
        return sorted(image_files)
    
    def preprocess_image(self, image_path: Path) -> np.ndarray:
        """
        Preprocess a single image preserving original dimensions.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Load image
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Keep original image dimensions (no resizing or padding)
            # Only apply image enhancements
            image = self._enhance_image(image)
            
            return np.array(image)
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return None
    
    def _resize_with_padding(self, image: Image.Image, target_size: int) -> Image.Image:
        """Resize image to target size with padding to maintain aspect ratio."""
        # Calculate scaling factor
        width, height = image.size
        scale = min(target_size / width, target_size / height)
        
        # Resize image
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create new image with padding
        new_image = Image.new('RGB', (target_size, target_size), (0, 0, 0))
        
        # Calculate position to center the image
        x = (target_size - new_width) // 2
        y = (target_size - new_height) // 2
        
        # Paste resized image onto the padded image
        new_image.paste(image, (x, y))
        
        return new_image
    
    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Apply image enhancement to improve quality."""
        # Contrast enhancement
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(self.enhancement_params['contrast'])
        
        # Brightness enhancement
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(self.enhancement_params['brightness'])
        
        # Sharpness enhancement
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(self.enhancement_params['sharpness'])
        
        # Color enhancement
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(self.enhancement_params['color'])
        
        return image
    
    def split_train_test(self, image_files: List[Path]) -> Tuple[List[Path], List[Path]]:
        """Split image files into training and testing sets."""
        # Shuffle files for random split
        shuffled_files = image_files.copy()
        random.shuffle(shuffled_files)
        
        # Calculate split point
        split_point = int(len(shuffled_files) * self.train_ratio)
        
        train_files = shuffled_files[:split_point]
        test_files = shuffled_files[split_point:]
        
        logger.info(f"Split: {len(train_files)} training, {len(test_files)} testing images")
        return train_files, test_files
    
    def process_and_save_images(self, image_files: List[Path], output_dir: Path, 
                               prefix: str = "") -> int:
        """Process and save images to output directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        processed_count = 0
        
        for i, image_path in enumerate(image_files):
            # Process image
            processed_image = self.preprocess_image(image_path)
            
            if processed_image is not None:
                # Generate output filename
                output_filename = f"{prefix}{i+1:03d}.png"
                output_path = output_dir / output_filename
                
                # Save processed image
                Image.fromarray(processed_image).save(output_path, 'PNG')
                processed_count += 1
                
                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1}/{len(image_files)} images")
        
        logger.info(f"Successfully processed {processed_count}/{len(image_files)} images to {output_dir}")
        return processed_count
    
    def create_dataset_splits(self, seed: int = 42):
        """Create train/test splits and process all images."""
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        # Get all image files
        image_files = self.get_image_files()
        
        if not image_files:
            logger.error("No image files found in source directory!")
            return False
        
        # Split into train/test
        train_files, test_files = self.split_train_test(image_files)
        
        # Process training images
        train_output_dir = self.target_path / "train" / "good"
        train_count = self.process_and_save_images(train_files, train_output_dir)
        
        # Process testing images
        test_output_dir = self.target_path / "test" / "good"
        test_count = self.process_and_save_images(test_files, test_output_dir)
        
        # Print summary
        total_processed = train_count + test_count
        logger.info(f"✅ Dataset creation completed:")
        logger.info(f"   Training images: {train_count}")
        logger.info(f"   Testing images: {test_count}")
        logger.info(f"   Total processed: {total_processed}/{len(image_files)}")
        logger.info(f"   Original image dimensions preserved")
        
        return True
    
    def validate_preprocessing(self) -> bool:
        """Validate that preprocessing was successful."""
        train_dir = self.target_path / "train" / "good"
        test_dir = self.target_path / "test" / "good"
        
        if not train_dir.exists() or not test_dir.exists():
            logger.error("Training or testing directories do not exist!")
            return False
        
        train_count = len(list(train_dir.glob("*.png")))
        test_count = len(list(test_dir.glob("*.png")))
        
        if train_count == 0 or test_count == 0:
            logger.error("No processed images found in train/test directories!")
            return False
        
        # Check that images exist and are valid (keep original dimensions)
        sample_image_path = next(train_dir.glob("*.png"))
        sample_image = Image.open(sample_image_path)
        
        logger.info(f"✅ Preprocessing validation passed!")
        logger.info(f"   Sample image dimensions: {sample_image.size}")
        logger.info(f"   Original dimensions preserved")
        return True


def main():
    parser = argparse.ArgumentParser(description="Preprocess fabric images for dataset creation")
    parser.add_argument("--source", required=True,
                       help="Source directory containing raw images")
    parser.add_argument("--target", required=True,
                       help="Target directory for processed dataset")
    parser.add_argument("--image_size", type=int, default=288,
                       help="Target image size (default: 288)")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="Ratio of images for training (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--validate", action="store_true",
                       help="Validate existing preprocessing instead of creating new")
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = ImagePreprocessor(
        source_path=args.source,
        target_path=args.target,
        image_size=args.image_size,
        train_ratio=args.train_ratio
    )
    
    if args.validate:
        preprocessor.validate_preprocessing()
    else:
        success = preprocessor.create_dataset_splits(seed=args.seed)
        if success:
            preprocessor.validate_preprocessing()


if __name__ == "__main__":
    main()
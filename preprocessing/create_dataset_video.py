#!/usr/bin/env python3
"""
Combined Dataset Video Creation Script
Creates videos from test images with optional preprocessing and organized storage.

Combines functionality from:
- create_video.py: Video creation capabilities
- preprocess_images.py: Image enhancement features

Usage:
    python preprocessing/create_dataset_video.py --dataset custom --class_name grid
    python preprocessing/create_dataset_video.py --dataset wfdd --class_name yellow_cloth --enhance
"""

import os
import argparse
import cv2
import glob
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetVideoCreator:
    """Creates videos from dataset test images with optional preprocessing."""
    
    def __init__(self, dataset_base: str = "datasets", output_base: str = "test-video"):
        self.dataset_base = Path(dataset_base)
        self.output_base = Path(output_base)
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Default enhancement parameters (from preprocess_images.py)
        self.enhancement_params = {
            'contrast': 1.1,
            'brightness': 1.0,
            'sharpness': 1.05,
            'color': 1.0
        }
    
    def get_dataset_structure(self, dataset_name: str, class_name: str) -> dict:
        """
        Analyze dataset structure and return available test directories.
        
        Returns:
            dict: Structure with available test directories and image counts
        """
        dataset_path = self.dataset_base / dataset_name / class_name / "test"
        
        if not dataset_path.exists():
            logger.error(f"Dataset path does not exist: {dataset_path}")
            return {}
        
        structure = {}
        
        # Scan all subdirectories in test folder
        for subdir in dataset_path.iterdir():
            if subdir.is_dir():
                # Count images in this subdirectory
                image_files = []
                for ext in [f"*{fmt}" for fmt in self.supported_formats]:
                    image_files.extend(glob.glob(str(subdir / ext)))
                    image_files.extend(glob.glob(str(subdir / f"{ext.upper()}")))
                
                if image_files:
                    structure[subdir.name] = {
                        'path': subdir,
                        'count': len(image_files),
                        'files': sorted(image_files)
                    }
        
        logger.info(f"Found test structure for {dataset_name}/{class_name}: {list(structure.keys())}")
        return structure
    
    def enhance_image(self, image: Image.Image) -> Image.Image:
        """Apply image enhancement similar to preprocess_images.py."""
        try:
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
        except Exception as e:
            logger.warning(f"Enhancement failed: {e}")
            return image
    
    def resize_with_aspect_ratio(self, image, target_width, target_height):
        """Resize image while preserving aspect ratio with padding."""
        h, w = image.shape[:2]
        
        # Calculate scaling factor
        scale = min(target_width / w, target_height / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create padded image with black background
        padded = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Calculate padding offsets
        y_offset = (target_height - new_h) // 2
        x_offset = (target_width - new_w) // 2
        
        # Place resized image in center
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return padded

    def center_crop(self, image, target_width, target_height):
        """Center crop image to target dimensions."""
        h, w = image.shape[:2]
        
        # Calculate crop coordinates
        start_y = max(0, (h - target_height) // 2)
        start_x = max(0, (w - target_width) // 2)
        
        # Crop image
        cropped = image[start_y:start_y + target_height, start_x:start_x + target_width]
        
        # If image is smaller than target, pad with black
        if cropped.shape[0] < target_height or cropped.shape[1] < target_width:
            padded = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            y_offset = (target_height - cropped.shape[0]) // 2
            x_offset = (target_width - cropped.shape[1]) // 2
            padded[y_offset:y_offset + cropped.shape[0], x_offset:x_offset + cropped.shape[1]] = cropped
            return padded
        
        return cropped
    
    def create_video_from_test_data(self, dataset_name: str, class_name: str, 
                                  test_types: list = None, fps: int = 1, 
                                  resize_method: str = 'none', enhance: bool = False,
                                  separate_videos: bool = False) -> dict:
        """
        Create videos from test data with specified parameters.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'custom', 'wfdd')
            class_name: Name of the class (e.g., 'grid', 'yellow_cloth')
            test_types: List of test types to include (e.g., ['good', 'hole']). If None, includes all.
            fps: Frames per second for video
            resize_method: How to handle image resizing
            enhance: Whether to apply image enhancement
            separate_videos: Create separate videos for each test type
            
        Returns:
            dict: Results with created video information
        """
        # Get dataset structure
        structure = self.get_dataset_structure(dataset_name, class_name)
        if not structure:
            return {"success": False, "error": "No test data found"}
        
        # Filter test types if specified
        if test_types:
            structure = {k: v for k, v in structure.items() if k in test_types}
            if not structure:
                return {"success": False, "error": f"Specified test types {test_types} not found"}
        
        # Create output directory
        output_dir = self.output_base / dataset_name / class_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {"success": True, "videos_created": [], "total_images": 0}
        
        if separate_videos:
            # Create separate video for each test type
            for test_type, info in structure.items():
                video_result = self._create_single_video(
                    image_files=info['files'],
                    output_path=output_dir / f"{test_type}.mp4",
                    video_name=f"{dataset_name}_{class_name}_{test_type}",
                    fps=fps,
                    resize_method=resize_method,
                    enhance=enhance
                )
                
                if video_result["success"]:
                    results["videos_created"].append(video_result)
                    results["total_images"] += video_result["frame_count"]
        else:
            # Create single combined video
            all_image_files = []
            for test_type, info in sorted(structure.items()):
                all_image_files.extend(info['files'])
            
            video_result = self._create_single_video(
                image_files=all_image_files,
                output_path=output_dir / f"{class_name}_combined.mp4",
                video_name=f"{dataset_name}_{class_name}_combined",
                fps=fps,
                resize_method=resize_method,
                enhance=enhance
            )
            
            if video_result["success"]:
                results["videos_created"].append(video_result)
                results["total_images"] = video_result["frame_count"]
        
        return results
    
    def _create_single_video(self, image_files: list, output_path: Path, 
                           video_name: str, fps: int, resize_method: str, 
                           enhance: bool) -> dict:
        """Create a single video from list of image files."""
        
        if not image_files:
            return {"success": False, "error": "No image files provided"}
        
        logger.info(f"Creating video: {video_name}")
        logger.info(f"Images: {len(image_files)}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Enhancement: {'enabled' if enhance else 'disabled'}")
        
        # Determine video dimensions
        if resize_method == 'largest':
            logger.info("Scanning images to find largest dimensions...")
            max_width, max_height = 0, 0
            for image_path in image_files:
                image = cv2.imread(image_path)
                if image is not None:
                    h, w = image.shape[:2]
                    max_width = max(max_width, w)
                    max_height = max(max_height, h)
            width, height = max_width, max_height
            logger.info(f"Using largest dimensions found: {width}x{height}")
        else:
            # Use first image dimensions
            first_image = cv2.imread(image_files[0])
            if first_image is None:
                return {"success": False, "error": f"Could not read first image: {image_files[0]}"}
            height, width = first_image.shape[:2]
            logger.info(f"Using first image dimensions: {width}x{height}")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            return {"success": False, "error": f"Failed to create video writer for {output_path}"}
        
        processed_count = 0
        
        # Process each image
        for i, image_path in enumerate(image_files):
            try:
                # Read image with OpenCV
                image = cv2.imread(image_path)
                if image is None:
                    logger.warning(f"Could not read image {image_path}, skipping...")
                    continue
                
                # Apply enhancement if requested
                if enhance:
                    # Convert to PIL for enhancement
                    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    pil_image = self.enhance_image(pil_image)
                    # Convert back to OpenCV format
                    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                
                # Handle different resize methods
                if resize_method == 'none':
                    # Keep original - only resize if dimensions don't match target
                    if image.shape[:2] != (height, width):
                        image = cv2.resize(image, (width, height))
                elif resize_method == 'largest':
                    # Resize to largest dimensions if needed
                    if image.shape[:2] != (height, width):
                        image = cv2.resize(image, (width, height))
                elif resize_method == 'preserve_aspect':
                    # Preserve aspect ratio with black padding
                    image = self.resize_with_aspect_ratio(image, width, height)
                elif resize_method == 'crop':
                    # Center crop to target dimensions
                    image = self.center_crop(image, width, height)
                
                # Write frame to video
                video_writer.write(image)
                processed_count += 1
                
                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1}/{len(image_files)} images")
                    
            except Exception as e:
                logger.warning(f"Error processing {image_path}: {e}")
                continue
        
        # Release video writer
        video_writer.release()
        
        duration = processed_count / fps
        
        logger.info(f"✅ Video created successfully!")
        logger.info(f"   Output: {output_path}")
        logger.info(f"   Frames: {processed_count}")
        logger.info(f"   Duration: {duration:.1f} seconds")
        logger.info(f"   Dimensions: {width}x{height}")
        
        return {
            "success": True,
            "output_path": str(output_path),
            "frame_count": processed_count,
            "duration": duration,
            "dimensions": (width, height),
            "video_name": video_name
        }


def main():
    parser = argparse.ArgumentParser(
        description="Create videos from dataset test images with preprocessing options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create combined video for grid class
  python preprocessing/create_dataset_video.py --dataset custom --class_name grid
  
  # Create separate videos for each defect type
  python preprocessing/create_dataset_video.py --dataset custom --class_name grid --separate
  
  # Create video with image enhancement
  python preprocessing/create_dataset_video.py --dataset wfdd --class_name yellow_cloth --enhance
  
  # Create video from specific test types only
  python preprocessing/create_dataset_video.py --dataset custom --class_name grid --test_types good hole spot
  
  # Create video with largest dimensions (no padding)
  python preprocessing/create_dataset_video.py --dataset custom --class_name grid --resize_method largest
  
  # Create enhanced video with aspect ratio preservation
  python preprocessing/create_dataset_video.py --dataset wfdd --class_name pink_flower --enhance --resize_method preserve_aspect
        """
    )
    
    parser.add_argument(
        '--dataset',
        required=True,
        help='Dataset name (e.g., custom, wfdd, mvtec)'
    )
    
    parser.add_argument(
        '--class_name',
        required=True,
        help='Class name (e.g., grid, yellow_cloth, bottle)'
    )
    
    parser.add_argument(
        '--test_types',
        nargs='+',
        help='Specific test types to include (e.g., good hole spot). If not specified, includes all available.'
    )
    
    parser.add_argument(
        '--dataset_base',
        default='datasets',
        help='Base dataset directory (default: datasets)'
    )
    
    parser.add_argument(
        '--output_base',
        default='test-video',
        help='Base output directory for videos (default: test-video)'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=1,
        help='Frames per second (default: 1 = 1 second per image)'
    )
    
    parser.add_argument(
        '--resize_method',
        choices=['none', 'largest', 'preserve_aspect', 'crop'],
        default='none',
        help='How to handle different image sizes (default: none)'
    )
    
    parser.add_argument(
        '--enhance',
        action='store_true',
        help='Apply image enhancement (contrast, sharpness, etc.)'
    )
    
    parser.add_argument(
        '--separate',
        action='store_true',
        help='Create separate videos for each test type instead of combined'
    )
    
    parser.add_argument(
        '--list_structure',
        action='store_true',
        help='List available test structure and exit (no video creation)'
    )
    
    args = parser.parse_args()
    
    # Create video creator
    creator = DatasetVideoCreator(
        dataset_base=args.dataset_base,
        output_base=args.output_base
    )
    
    if args.list_structure:
        # Just show structure and exit
        structure = creator.get_dataset_structure(args.dataset, args.class_name)
        if structure:
            print(f"\nAvailable test structure for {args.dataset}/{args.class_name}:")
            for test_type, info in structure.items():
                print(f"  {test_type}: {info['count']} images")
            print(f"\nTotal test types: {len(structure)}")
            print(f"Total images: {sum(info['count'] for info in structure.values())}")
        return
    
    # Create videos
    logger.info(f"Creating videos for {args.dataset}/{args.class_name}")
    logger.info(f"Settings: FPS={args.fps}, Resize={args.resize_method}, Enhance={args.enhance}, Separate={args.separate}")
    
    results = creator.create_video_from_test_data(
        dataset_name=args.dataset,
        class_name=args.class_name,
        test_types=args.test_types,
        fps=args.fps,
        resize_method=args.resize_method,
        enhance=args.enhance,
        separate_videos=args.separate
    )
    
    if results["success"]:
        print(f"\n✅ Video creation completed!")
        print(f"Videos created: {len(results['videos_created'])}")
        print(f"Total images processed: {results['total_images']}")
        
        for video_info in results["videos_created"]:
            print(f"  📹 {video_info['video_name']}: {video_info['frame_count']} frames, {video_info['duration']:.1f}s")
            print(f"     → {video_info['output_path']}")
    else:
        print(f"❌ Error: {results.get('error', 'Unknown error')}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
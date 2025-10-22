#!/usr/bin/env python3
"""
Enhanced Video Creation Script for GLASS Inference
Creates videos with defect distribution across entire duration and saves good frames as images.

Features:
- Distributes defect frames equally throughout video duration
- Saves only good frames as individual images for reference
- Removes info text overlays for clean inference
- Supports custom defect-to-good frame ratios
- Organized output structure: test-video/[video-name]/
"""

import os
import argparse
import cv2
import glob
import numpy as np
import random
from pathlib import Path
from PIL import Image, ImageEnhance
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedVideoCreator:
    """Creates videos with distributed defects and saves good frames as images."""
    
    def __init__(self, dataset_base: str = "datasets", output_base: str = "test-video"):
        self.dataset_base = Path(dataset_base)
        self.output_base = Path(output_base)
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Default enhancement parameters
        self.enhancement_params = {
            'contrast': 1.1,
            'brightness': 1.0,
            'sharpness': 1.05,
            'color': 1.0
        }
    
    def get_dataset_structure(self, dataset_name: str, class_name: str) -> dict:
        """Analyze dataset structure and categorize good vs defect images."""
        dataset_path = self.dataset_base / dataset_name / class_name / "test"
        
        if not dataset_path.exists():
            logger.error(f"Dataset path does not exist: {dataset_path}")
            return {}
        
        structure = {"good": [], "defect": []}
        
        # Scan all subdirectories in test folder
        for subdir in dataset_path.iterdir():
            if subdir.is_dir():
                # Get images in this subdirectory
                image_files = []
                for ext in [f"*{fmt}" for fmt in self.supported_formats]:
                    image_files.extend(glob.glob(str(subdir / ext)))
                    image_files.extend(glob.glob(str(subdir / f"{ext.upper()}")))
                
                if image_files:
                    # Categorize: 'good' folder contains normal images, others are defects
                    if subdir.name.lower() == 'good':
                        structure['good'].extend(sorted(image_files))
                    else:
                        structure['defect'].extend(sorted(image_files))
        
        logger.info(f"Found structure for {dataset_name}/{class_name}:")
        logger.info(f"  Good images: {len(structure['good'])}")
        logger.info(f"  Defect images: {len(structure['defect'])}")
        
        return structure
    
    def enhance_image(self, image: Image.Image) -> Image.Image:
        """Apply image enhancement."""
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
    
    def distribute_defects(self, good_images: list, defect_images: list, 
                          defect_ratio: float = 0.3, total_frames: int = None) -> list:
        """
        Create a frame sequence with defects distributed equally throughout.
        
        Args:
            good_images: List of good image paths
            defect_images: List of defect image paths
            defect_ratio: Ratio of defect frames (0.0 to 1.0)
            total_frames: Target total frames (if None, uses available images)
        
        Returns:
            List of tuples (image_path, is_defect)
        """
        if not good_images:
            logger.error("No good images available")
            return []
        
        if not defect_images:
            logger.warning("No defect images available, using only good images")
            return [(img, False) for img in good_images]
        
        # Determine total frames
        if total_frames is None:
            total_frames = len(good_images) + len(defect_images)
        
        # Calculate number of defect frames
        num_defect_frames = int(total_frames * defect_ratio)
        num_good_frames = total_frames - num_defect_frames
        
        logger.info(f"Creating sequence: {total_frames} total frames")
        logger.info(f"  Good frames: {num_good_frames} ({100*(1-defect_ratio):.1f}%)")
        logger.info(f"  Defect frames: {num_defect_frames} ({100*defect_ratio:.1f}%)")
        
        # Sample images (with repetition if needed)
        selected_good = random.choices(good_images, k=num_good_frames) if good_images else []
        selected_defect = random.choices(defect_images, k=num_defect_frames) if defect_images else []
        
        # Create frame sequence with distributed defects
        frame_sequence = []
        
        if num_defect_frames == 0:
            # No defects, just good frames
            frame_sequence = [(img, False) for img in selected_good]
        else:
            # Calculate defect distribution intervals
            interval = total_frames / num_defect_frames
            defect_positions = [int(i * interval + interval/2) for i in range(num_defect_frames)]
            
            # Create base sequence with good frames
            frame_sequence = [(None, False)] * total_frames
            
            # Place defect frames at calculated positions
            defect_idx = 0
            for pos in defect_positions:
                if pos < total_frames and defect_idx < len(selected_defect):
                    frame_sequence[pos] = (selected_defect[defect_idx], True)
                    defect_idx += 1
            
            # Fill remaining positions with good frames
            good_idx = 0
            for i, (img, is_defect) in enumerate(frame_sequence):
                if img is None and good_idx < len(selected_good):
                    frame_sequence[i] = (selected_good[good_idx], False)
                    good_idx += 1
        
        logger.info(f"Created sequence with {len(frame_sequence)} frames")
        return frame_sequence
    
    def create_enhanced_video(self, dataset_name: str, class_name: str, 
                            video_name: str = None, fps: int = 2,
                            defect_ratio: float = 0.3, total_frames: int = None,
                            enhance: bool = False, save_good_frames: bool = True) -> dict:
        """
        Create enhanced video with distributed defects and save good frames as images.
        
        Args:
            dataset_name: Name of the dataset
            class_name: Name of the class
            video_name: Custom video name (defaults to class_name)
            fps: Frames per second
            defect_ratio: Ratio of defect frames (0.0 to 1.0)
            total_frames: Target total frames
            enhance: Apply image enhancement
            save_good_frames: Save good frames as individual images
        
        Returns:
            dict: Results with video and image information
        """
        if video_name is None:
            video_name = f"{dataset_name}_{class_name}"
        
        # Get dataset structure
        structure = self.get_dataset_structure(dataset_name, class_name)
        if not structure:
            return {"success": False, "error": f"Dataset not found: {dataset_name}/{class_name}"}
        
        if not structure.get('good') and not structure.get('defect'):
            return {"success": False, "error": "No images found in dataset structure"}
        
        # Create output directories
        output_dir = self.output_base / video_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if save_good_frames:
            frames_dir = output_dir / "good_frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Create distributed frame sequence
        frame_sequence = self.distribute_defects(
            structure['good'], structure['defect'], 
            defect_ratio, total_frames
        )
        
        if not frame_sequence:
            return {"success": False, "error": "Could not create frame sequence"}
        
        # Determine video dimensions from first image
        first_image = cv2.imread(frame_sequence[0][0])
        if first_image is None:
            return {"success": False, "error": f"Could not read first image: {frame_sequence[0][0]}"}
        
        height, width = first_image.shape[:2]
        
        # Create video writer
        video_path = output_dir / f"{video_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            return {"success": False, "error": f"Failed to create video writer for {video_path}"}
        
        logger.info(f"Creating enhanced video: {video_name}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Video dimensions: {width}x{height}")
        logger.info(f"FPS: {fps}, Total frames: {len(frame_sequence)}")
        
        processed_count = 0
        good_frames_saved = 0
        defect_frames_used = 0
        
        # Process each frame in sequence
        for i, (image_path, is_defect) in enumerate(frame_sequence):
            try:
                # Read image
                image = cv2.imread(image_path)
                if image is None:
                    logger.warning(f"Could not read image {image_path}, skipping...")
                    continue
                
                # Resize to match video dimensions if needed
                if image.shape[:2] != (height, width):
                    image = cv2.resize(image, (width, height))
                
                # Apply enhancement if requested
                if enhance:
                    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    pil_image = self.enhance_image(pil_image)
                    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                
                # Write frame to video
                video_writer.write(image)
                processed_count += 1
                
                # Save good frames as individual images
                if save_good_frames and not is_defect:
                    frame_filename = f"frame_{i:06d}.jpg"
                    frame_path = frames_dir / frame_filename
                    cv2.imwrite(str(frame_path), image)
                    good_frames_saved += 1
                
                if is_defect:
                    defect_frames_used += 1
                
                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1}/{len(frame_sequence)} frames")
                    
            except Exception as e:
                logger.warning(f"Error processing {image_path}: {e}")
                continue
        
        # Release video writer
        video_writer.release()
        
        duration = processed_count / fps
        
        logger.info(f"✅ Enhanced video created successfully!")
        logger.info(f"   Video: {video_path}")
        logger.info(f"   Frames: {processed_count}")
        logger.info(f"   Duration: {duration:.1f} seconds")
        logger.info(f"   Defect frames: {defect_frames_used}")
        logger.info(f"   Good frames saved as images: {good_frames_saved}")
        
        return {
            "success": True,
            "video_path": str(video_path),
            "frames_dir": str(frames_dir) if save_good_frames else None,
            "frame_count": processed_count,
            "good_frames_saved": good_frames_saved,
            "defect_frames_used": defect_frames_used,
            "duration": duration,
            "dimensions": (width, height),
            "video_name": video_name,
            "defect_ratio_actual": defect_frames_used / processed_count if processed_count > 0 else 0
        }


def main():
    parser = argparse.ArgumentParser(
        description="Create enhanced videos with distributed defects and save good frames",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create video with 30% defects distributed throughout
  python preprocessing/create_enhanced_video.py --dataset custom --class_name grid --video_name grid_test
  
  # Create video with 50% defects and save good frames
  python preprocessing/create_enhanced_video.py --dataset custom --class_name grid --defect_ratio 0.5 --save_good_frames
  
  # Create enhanced video with specific frame count
  python preprocessing/create_enhanced_video.py --dataset wfdd --class_name yellow_cloth --total_frames 300 --enhance
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
        '--video_name',
        help='Custom video name (default: dataset_classname)'
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
        default=2,
        help='Frames per second (default: 2)'
    )
    
    parser.add_argument(
        '--defect_ratio',
        type=float,
        default=0.3,
        help='Ratio of defect frames (0.0 to 1.0, default: 0.3)'
    )
    
    parser.add_argument(
        '--total_frames',
        type=int,
        help='Target total frames (default: use all available images)'
    )
    
    parser.add_argument(
        '--enhance',
        action='store_true',
        help='Apply image enhancement (contrast, sharpness, etc.)'
    )
    
    parser.add_argument(
        '--save_good_frames',
        action='store_true',
        default=True,
        help='Save good frames as individual images (default: True)'
    )
    
    parser.add_argument(
        '--no_save_frames',
        action='store_true',
        help='Disable saving good frames as images'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible defect distribution (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Handle save_good_frames flag
    save_good_frames = args.save_good_frames and not args.no_save_frames
    
    # Validate defect ratio
    if not 0.0 <= args.defect_ratio <= 1.0:
        logger.error("Defect ratio must be between 0.0 and 1.0")
        return 1
    
    # Create video creator
    creator = EnhancedVideoCreator(
        dataset_base=args.dataset_base,
        output_base=args.output_base
    )
    
    # Create enhanced video
    logger.info(f"Creating enhanced video for {args.dataset}/{args.class_name}")
    logger.info(f"Settings: FPS={args.fps}, Defect ratio={args.defect_ratio}, Enhance={args.enhance}")
    logger.info(f"Save good frames: {save_good_frames}")
    
    results = creator.create_enhanced_video(
        dataset_name=args.dataset,
        class_name=args.class_name,
        video_name=args.video_name,
        fps=args.fps,
        defect_ratio=args.defect_ratio,
        total_frames=args.total_frames,
        enhance=args.enhance,
        save_good_frames=save_good_frames
    )
    
    if results["success"]:
        print(f"\n✅ Enhanced video creation completed!")
        print(f"Video: {results['video_path']}")
        print(f"Frames processed: {results['frame_count']}")
        print(f"Duration: {results['duration']:.1f} seconds")
        print(f"Defect frames: {results['defect_frames_used']} ({results['defect_ratio_actual']:.1%})")
        if results.get('frames_dir'):
            print(f"Good frames saved: {results['good_frames_saved']} images")
            print(f"Frames directory: {results['frames_dir']}")
    else:
        print(f"❌ Error: {results.get('error', 'Unknown error')}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
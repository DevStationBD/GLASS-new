#!/usr/bin/env python3
"""
Script to create videos from dataset images with 1 second per frame.
Each image is displayed for exactly 1 second in the output video.
"""

import os
import argparse
import cv2
import glob
from pathlib import Path
import sys

def create_video_from_images(input_path, output_path, fps=1):
    """
    Create video from images in a directory (recursively searches subfolders).
    
    Args:
        input_path (str): Path to directory containing images
        output_path (str): Output video file path
        fps (int): Frames per second (1 = 1 second per image)
    """
    # Get all image files recursively
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    
    for ext in image_extensions:
        # Search recursively using **/ pattern
        image_files.extend(glob.glob(os.path.join(input_path, '**', ext), recursive=True))
        image_files.extend(glob.glob(os.path.join(input_path, '**', ext.upper()), recursive=True))
    
    if not image_files:
        print(f"No images found in {input_path}")
        return False
    
    # Sort images by name for consistent ordering
    image_files.sort()
    print(f"Found {len(image_files)} images")
    
    # Read first image to get dimensions
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        print(f"Could not read image: {image_files[0]}")
        return False
    
    height, width, channels = first_image.shape
    print(f"Video dimensions: {width}x{height}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print(f"Failed to create video writer for {output_path}")
        return False
    
    # Process each image
    for i, image_path in enumerate(image_files):
        # Show relative path from input directory
        rel_path = os.path.relpath(image_path, input_path)
        print(f"Processing image {i+1}/{len(image_files)}: {rel_path}")
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}, skipping...")
            continue
        
        # Resize image if dimensions don't match
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height))
        
        # Write frame (since fps=1, each image will be shown for 1 second)
        video_writer.write(image)
    
    # Release video writer
    video_writer.release()
    
    total_duration = len(image_files)
    print(f"Video created successfully: {output_path}")
    print(f"Total duration: {total_duration} seconds ({len(image_files)} images)")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Create video from dataset images with 1 second per frame",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create video from grid class test images
  python preprocessing/create_video.py grid --output preprocessing/grid_test_video.mp4
  
  # Create video from custom path
  python preprocessing/create_video.py bottle --input_path datasets/custom/bottle/test/good --output preprocessing/bottle_good.mp4
  
  # Create video from defect images
  python preprocessing/create_video.py grid --defect hole --output preprocessing/grid_hole_video.mp4
        """
    )
    
    parser.add_argument(
        'class_name',
        help='Dataset class name (e.g., grid, bottle, cable, etc.)'
    )
    
    parser.add_argument(
        '--input_path',
        help='Custom input path (overrides default dataset path)'
    )
    
    parser.add_argument(
        '--output',
        required=True,
        help='Output video file path (e.g., preprocessing/output.mp4)'
    )
    
    parser.add_argument(
        '--defect',
        help='Defect type for test images (default: good images). If specified, uses test/defect_type/'
    )
    
    parser.add_argument(
        '--dataset_base',
        default='datasets/custom',
        help='Base dataset directory (default: datasets/custom)'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=1,
        help='Frames per second (default: 1 = 1 second per image)'
    )
    
    args = parser.parse_args()
    
    # Determine input path
    if args.input_path:
        input_path = args.input_path
    else:
        if args.defect:
            input_path = os.path.join(args.dataset_base, args.class_name, 'test', args.defect)
        else:
            input_path = os.path.join(args.dataset_base, args.class_name, 'test', 'good')
    
    # Validate input path
    if not os.path.exists(input_path):
        print(f"Error: Input path does not exist: {input_path}")
        return 1
    
    if not os.path.isdir(input_path):
        print(f"Error: Input path is not a directory: {input_path}")
        return 1
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Input path: {input_path}")
    print(f"Output path: {args.output}")
    print(f"FPS: {args.fps} (each image shown for {1/args.fps:.1f} seconds)")
    
    # Create video
    success = create_video_from_images(input_path, args.output, args.fps)
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
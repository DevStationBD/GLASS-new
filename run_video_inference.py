#!/usr/bin/env python3

import os
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Easy GLASS Video Inference')
    parser.add_argument('video_path', type=str, help='Path to input video')
    parser.add_argument('output_path', type=str, help='Path for output video')
    parser.add_argument('--class_name', type=str, default='wfdd_yellow_cloth',
                       choices=['wfdd_yellow_cloth', 'wfdd_pink_flower'],
                       help='Which trained model to use')
    parser.add_argument('--threshold', type=float, default=None,
                       help='Anomaly threshold (auto-calculated if not provided)')
    
    args = parser.parse_args()
    
    # Fixed paths based on your setup
    model_path = "results/models/backbone_0"
    image_size = 384  # Based on your training config
    device = "cuda:0"
    
    # Check if files exist
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)
    
    if not os.path.exists(model_path):
        print(f"Error: Model directory not found: {model_path}")
        sys.exit(1)
    
    # Build command
    cmd = [
        sys.executable, "video_inference_sidebyside.py",
        "--model_path", model_path,
        "--class_name", args.class_name,
        "--video_path", args.video_path,
        "--output_path", args.output_path,
        "--image_size", str(image_size),
        "--device", device
    ]
    
    if args.threshold is not None:
        cmd.extend(["--threshold", str(args.threshold)])
    
    print(f"Running inference with model: {args.class_name}")
    print(f"Input video: {args.video_path}")
    print(f"Output video: {args.output_path}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    # Run the command
    os.execv(sys.executable, cmd)

if __name__ == '__main__':
    main()
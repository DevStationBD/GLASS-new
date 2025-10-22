#!/usr/bin/env python3
"""
Frame Extraction Script for GLASS Project

Extract frames from video files at specified intervals for dataset creation or analysis.
"""

import cv2
import os
import argparse
from pathlib import Path
import sys
import glob

def get_video_files(video_paths):
    """
    Expand video paths to include glob patterns and validate files
    
    Args:
        video_paths (list): List of video paths (can include glob patterns)
    
    Returns:
        list: List of valid video file paths
    """
    all_videos = []
    
    for path in video_paths:
        if '*' in path or '?' in path:
            # Glob pattern
            matched_files = glob.glob(path)
            if matched_files:
                all_videos.extend(matched_files)
            else:
                print(f"⚠️ No files matched pattern: {path}")
        else:
            # Single file
            if os.path.exists(path):
                all_videos.append(path)
            else:
                print(f"⚠️ File not found: {path}")
    
    return sorted(all_videos)

def extract_frames(video_path, output_dir, save_frequency=30, start_frame=0, max_frames=None, video_prefix=None):
    """
    Extract frames from video at specified intervals
    
    Args:
        video_path (str): Path to input video file
        output_dir (str): Directory to save extracted frames
        save_frequency (int): Extract every Nth frame (default: 30)
        start_frame (int): Frame to start extraction from (default: 0)
        max_frames (int): Maximum number of frames to extract (default: None for all)
        video_prefix (str): Prefix to add to frame filenames (default: None)
    
    Returns:
        int: Number of frames extracted
    """
    
    # Validate input video
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    # Get video name for prefix
    if video_prefix is None:
        video_name = Path(video_path).stem
        video_prefix = video_name
    
    print(f"📹 Video Info: {Path(video_path).name}")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Total frames: {total_frames}")
    print(f"   Duration: {duration:.2f} seconds")
    print(f"   Save frequency: every {save_frequency} frames")
    print(f"   Start from frame: {start_frame}")
    print(f"   Frame prefix: {video_prefix}")
    
    # Set starting position
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = start_frame
    saved_count = 0
    
    print(f"\n🎬 Extracting frames to: {output_path}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if we should save this frame
        if (frame_count - start_frame) % save_frequency == 0:
            # Generate filename with prefix
            filename = f"{video_prefix}_frame_{frame_count:06d}.jpg"
            filepath = output_path / filename
            
            # Save frame
            success = cv2.imwrite(str(filepath), frame)
            if success:
                saved_count += 1
                if saved_count % 10 == 0:
                    print(f"   Extracted {saved_count} frames...")
            else:
                print(f"⚠️ Failed to save frame {frame_count}")
        
        frame_count += 1
        
        # Check if we've reached the maximum
        if max_frames and saved_count >= max_frames:
            print(f"📊 Reached maximum frame limit: {max_frames}")
            break
    
    cap.release()
    
    print(f"\n✅ Frame extraction completed!")
    print(f"   Total frames processed: {frame_count - start_frame}")
    print(f"   Frames saved: {saved_count}")
    print(f"   Output directory: {output_path}")
    
    # Calculate extraction statistics
    if saved_count > 0:
        time_per_frame = duration / total_frames if total_frames > 0 else 0
        time_span = saved_count * save_frequency * time_per_frame
        print(f"   Time span covered: {time_span:.2f} seconds")
        print(f"   Extraction ratio: {saved_count/total_frames*100:.2f}%")
    
    return saved_count

def process_multiple_videos(video_paths, output_dir, save_frequency=30, start_frame=0, max_frames=None, separate_folders=False):
    """
    Process multiple video files
    
    Args:
        video_paths (list): List of video file paths
        output_dir (str): Base output directory
        save_frequency (int): Extract every Nth frame
        start_frame (int): Frame to start extraction from
        max_frames (int): Maximum frames per video
        separate_folders (bool): Create separate folder for each video
    
    Returns:
        dict: Processing results for each video
    """
    results = {}
    total_extracted = 0
    
    print(f"\n🎭 Processing {len(video_paths)} videos...")
    print(f"📁 Base output directory: {output_dir}")
    print(f"📂 Separate folders: {separate_folders}")
    
    for i, video_path in enumerate(video_paths, 1):
        print(f"\n{'='*60}")
        print(f"🎬 Processing video {i}/{len(video_paths)}: {Path(video_path).name}")
        print(f"{'='*60}")
        
        try:
            # Determine output directory for this video
            if separate_folders:
                video_name = Path(video_path).stem
                video_output_dir = Path(output_dir) / video_name
            else:
                video_output_dir = output_dir
            
            # Extract frames
            extracted_count = extract_frames(
                video_path=video_path,
                output_dir=str(video_output_dir),
                save_frequency=save_frequency,
                start_frame=start_frame,
                max_frames=max_frames,
                video_prefix=Path(video_path).stem if not separate_folders else None
            )
            
            results[video_path] = {
                'status': 'success',
                'frames_extracted': extracted_count,
                'output_dir': str(video_output_dir)
            }
            total_extracted += extracted_count
            
        except Exception as e:
            print(f"❌ Failed to process {video_path}: {e}")
            results[video_path] = {
                'status': 'failed',
                'error': str(e),
                'frames_extracted': 0
            }
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 PROCESSING SUMMARY")
    print(f"{'='*60}")
    successful = sum(1 for r in results.values() if r['status'] == 'success')
    failed = len(results) - successful
    
    print(f"✅ Successful: {successful}/{len(video_paths)} videos")
    print(f"❌ Failed: {failed}/{len(video_paths)} videos")
    print(f"📸 Total frames extracted: {total_extracted}")
    
    if failed > 0:
        print(f"\n❌ Failed videos:")
        for video_path, result in results.items():
            if result['status'] == 'failed':
                print(f"   - {Path(video_path).name}: {result['error']}")
    
    return results

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Extract frames from video files (supports multiple videos and glob patterns)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single video - extract every 30th frame
  python extract_frames.py input.mp4 output_frames/
  
  # Multiple videos - explicit paths
  python extract_frames.py video1.mp4 video2.mp4 video3.mp4 output_frames/
  
  # Multiple videos - glob pattern
  python extract_frames.py "videos/*.mp4" output_frames/
  
  # Multiple videos with separate folders
  python extract_frames.py video1.mp4 video2.mp4 output_frames/ --separate-folders
  
  # Extract every 10th frame with maximum 100 frames per video
  python extract_frames.py "*.mp4" output_frames/ --frequency 10 --max-frames 100
  
  # Start from frame 500, extract every 60th frame
  python extract_frames.py input.mp4 output_frames/ --frequency 60 --start-frame 500
        """
    )
    
    parser.add_argument('video_paths', nargs='+', help='Path(s) to input video file(s) (supports glob patterns)')
    parser.add_argument('output_dir', help='Directory to save extracted frames')
    parser.add_argument('--frequency', '-f', type=int, default=30,
                       help='Extract every Nth frame (default: 30)')
    parser.add_argument('--start-frame', '-s', type=int, default=0,
                       help='Frame number to start extraction from (default: 0)')
    parser.add_argument('--max-frames', '-m', type=int, default=None,
                       help='Maximum number of frames to extract per video (default: no limit)')
    parser.add_argument('--separate-folders', action='store_true',
                       help='Create separate output folder for each video')
    
    args = parser.parse_args()
    
    try:
        # Get all video files (expand globs)
        all_video_paths = get_video_files(args.video_paths)
        
        if not all_video_paths:
            print("❌ No valid video files found.")
            sys.exit(1)
        
        if len(all_video_paths) == 1:
            # Single video processing
            saved_count = extract_frames(
                video_path=all_video_paths[0],
                output_dir=args.output_dir,
                save_frequency=args.frequency,
                start_frame=args.start_frame,
                max_frames=args.max_frames
            )
            
            if saved_count == 0:
                print("⚠️ No frames were extracted. Check your input parameters.")
                sys.exit(1)
        else:
            # Multiple video processing
            results = process_multiple_videos(
                video_paths=all_video_paths,
                output_dir=args.output_dir,
                save_frequency=args.frequency,
                start_frame=args.start_frame,
                max_frames=args.max_frames,
                separate_folders=args.separate_folders
            )
            
            # Check if any videos were processed successfully
            successful_count = sum(1 for r in results.values() if r['status'] == 'success')
            if successful_count == 0:
                print("❌ No videos were processed successfully.")
                sys.exit(1)
            
    except FileNotFoundError as e:
        print(f"❌ File Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Video Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
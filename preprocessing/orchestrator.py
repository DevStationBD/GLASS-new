#!/usr/bin/env python3
"""
Fabric Defect Dataset Creation and Video Generation Orchestrator
Main script that coordinates dataset creation pipeline and video generation from test images.

Usage:
    # Dataset creation pipeline with separate dataset and class names (recommended)
    python orchestrator.py --source raw-data/custom-grid/good-images --dataset custom --class_name grid --full_pipeline
    
    # Video creation from existing dataset
    python orchestrator.py --source raw-data/custom-grid/good-images --dataset custom --class_name grid --create_videos
    
    # Combined pipeline (dataset creation + video generation)
    python orchestrator.py --source raw-data/custom-grid/good-images --dataset custom --class_name grid --full_pipeline --create_videos
    
    # Backward compatible (legacy format still works)
    python orchestrator.py --source raw-data/custom-grid/good-images --dataset custom_grid --full_pipeline
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Optional
import time
import json

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetOrchestrator:
    """Orchestrates the complete fabric defect dataset creation pipeline."""
    
    def __init__(self, source_dir: str, dataset_name: str, class_name: str = None, base_output_dir: str = None):
        self.source_dir = Path(source_dir)
        self.dataset_name = dataset_name
        self.class_name = class_name or self._extract_class_from_dataset_name(dataset_name)
        self.base_output_dir = Path(base_output_dir) if base_output_dir else Path(__file__).parent.parent / "datasets"
        
        # Create proper nested structure: datasets/[dataset]/[class]/
        if class_name:
            # New mode: separate dataset and class  
            self.dataset_dir = self.base_output_dir / dataset_name / class_name
            self.full_dataset_name = f"{dataset_name}_{class_name}"
        else:
            # Legacy mode: extract from combined name
            parts = dataset_name.split('_', 1)
            if len(parts) >= 2:
                actual_dataset = parts[0]
                actual_class = parts[1]
                self.dataset_dir = self.base_output_dir / actual_dataset / actual_class
                self.full_dataset_name = dataset_name
            else:
                # Fallback for simple names
                self.dataset_dir = self.base_output_dir / dataset_name / dataset_name
                self.full_dataset_name = dataset_name
        self.scripts_dir = Path(__file__).parent
        
        # Pipeline configuration
        self.config = {
            "image_size": 384,
            "train_ratio": 0.8,
            "images_per_defect": 50,
            "defect_types": ["hole", "foreign_yarn", "missing_yarn", "slab", "spot"],
            "seed": 42
        }
        
        # Video configuration
        self.video_config = {
            "fps": 1,
            "resize_method": "none",
            "enhance": False,
            "separate_videos": False,  # Default to combined video
            "output_base": "test-video"
        }
        
        # Track pipeline progress
        self.pipeline_status = {
            "structure_created": False,
            "images_preprocessed": False,
            "defects_generated": False,
            "masks_validated": False,
            "videos_created": False
        }
    
    def validate_inputs(self) -> bool:
        """Validate input parameters and directories."""
        logger.info("Validating inputs...")
        
        # Check source directory
        if not self.source_dir.exists():
            logger.error(f"Source directory does not exist: {self.source_dir}")
            return False
        
        # Check for image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in self.source_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if len(image_files) == 0:
            logger.error(f"No image files found in source directory: {self.source_dir}")
            return False
        
        logger.info(f"Found {len(image_files)} image files in source directory")
        
        # Check scripts exist
        required_scripts = [
            "prepare_dataset_structure.py",
            "preprocess_images.py", 
            "defect_simulator.py",
            "mask_utils.py"
        ]
        
        for script in required_scripts:
            script_path = self.scripts_dir / script
            if not script_path.exists():
                logger.error(f"Required script not found: {script_path}")
                return False
        
        logger.info("✅ Input validation passed")
        return True
    
    def run_script(self, script_name: str, args: List[str]) -> bool:
        """Run a pipeline script with arguments."""
        script_path = self.scripts_dir / script_name
        cmd = ["python", str(script_path)] + args
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Log output
            if result.stdout:
                logger.info(f"Output: {result.stdout.strip()}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Script failed: {script_name}")
            logger.error(f"Error: {e.stderr}")
            return False
    
    def step_1_create_structure(self) -> bool:
        """Step 1: Create dataset directory structure."""
        logger.info("🏗️  Step 1: Creating dataset structure...")
        
        args = [
            "--dataset_name", self.full_dataset_name,
            "--base_path", str(self.base_output_dir)
        ]
        
        success = self.run_script("prepare_dataset_structure.py", args)
        if success:
            self.pipeline_status["structure_created"] = True
            logger.info("✅ Dataset structure created successfully")
        else:
            logger.error("❌ Failed to create dataset structure")
        
        return success
    
    def step_2_preprocess_images(self) -> bool:
        """Step 2: Preprocess and split images into train/test."""
        logger.info("🖼️  Step 2: Preprocessing images...")
        
        args = [
            "--source", str(self.source_dir),
            "--target", str(self.dataset_dir),
            "--image_size", str(self.config["image_size"]),
            "--train_ratio", str(self.config["train_ratio"]),
            "--seed", str(self.config["seed"])
        ]
        
        success = self.run_script("preprocess_images.py", args)
        if success:
            self.pipeline_status["images_preprocessed"] = True
            logger.info("✅ Image preprocessing completed successfully")
        else:
            logger.error("❌ Failed to preprocess images")
        
        return success
    
    def step_3_generate_defects(self) -> bool:
        """Step 3: Generate defective images and masks."""
        logger.info("🔧 Step 3: Generating defective images...")
        
        input_dir = self.dataset_dir / "test" / "good"
        output_dir = self.dataset_dir / "test"
        
        args = [
            "--input", str(input_dir),
            "--output", str(output_dir),
            "--defect_types"] + self.config["defect_types"] + [
            "--images_per_defect", str(self.config["images_per_defect"]),
            "--seed", str(self.config["seed"])
        ]
        
        success = self.run_script("defect_simulator.py", args)
        if success:
            self.pipeline_status["defects_generated"] = True
            logger.info("✅ Defect generation completed successfully")
        else:
            logger.error("❌ Failed to generate defects")
        
        return success
    
    def step_4_validate_masks(self) -> bool:
        """Step 4: Validate generated masks."""
        logger.info("🔍 Step 4: Validating generated masks...")
        
        ground_truth_dir = self.dataset_dir / "ground_truth"
        
        args = [
            "--validate", str(ground_truth_dir),
            "--plot_stats"
        ]
        
        success = self.run_script("mask_utils.py", args)
        if success:
            self.pipeline_status["masks_validated"] = True
            logger.info("✅ Mask validation completed successfully")
        else:
            logger.error("❌ Mask validation failed")
        
        return success
    
    def validate_video_inputs(self) -> bool:
        """Validate inputs for video creation."""
        logger.info("Validating video creation inputs...")
        
        # Check if dataset exists
        if not self.dataset_dir.exists():
            logger.error(f"Dataset directory does not exist: {self.dataset_dir}")
            return False
        
        # Check if test directory exists
        test_dir = self.dataset_dir / "test"
        if not test_dir.exists():
            logger.error(f"Test directory does not exist: {test_dir}")
            return False
        
        # Check if video creation script exists
        video_script = self.scripts_dir / "create_dataset_video.py"
        if not video_script.exists():
            logger.error(f"Video creation script not found: {video_script}")
            return False
        
        logger.info("✅ Video creation inputs validated")
        return True
    
    def _extract_class_from_dataset_name(self, dataset_name: str) -> str:
        """
        Extract class name from combined dataset name for backward compatibility.
        
        Args:
            dataset_name: Combined dataset name (e.g., 'custom_grid')
            
        Returns:
            str: Extracted class name (e.g., 'grid')
        """
        parts = dataset_name.split('_', 1)
        return parts[1] if len(parts) >= 2 else dataset_name
    
    def _parse_dataset_name(self, dataset_name: str) -> tuple:
        """
        Parse dataset name to extract dataset and class components.
        
        Expected formats:
        - custom_grid -> ('custom', 'grid')
        - wfdd_yellow_cloth -> ('wfdd', 'yellow_cloth')
        - mvtec_bottle -> ('mvtec', 'bottle')
        
        Args:
            dataset_name: Full dataset name with underscore separator
            
        Returns:
            tuple: (dataset_name, class_name)
        """
        # Split on first underscore to separate dataset from class
        parts = dataset_name.split('_', 1)
        
        if len(parts) >= 2:
            dataset = parts[0]
            class_name = parts[1]
        else:
            # Fallback: if no underscore, use the name as both dataset and class
            logger.warning(f"No underscore found in dataset name '{dataset_name}'. Using as both dataset and class.")
            dataset = dataset_name
            class_name = dataset_name
        
        return dataset, class_name
    
    def step_5_create_videos(self) -> bool:
        """Step 5: Create videos from test images."""
        logger.info("🎬 Step 5: Creating videos from test images...")
        
        # Validate video inputs
        if not self.validate_video_inputs():
            return False
        
        # Use the separate dataset and class names
        logger.info(f"Creating videos for dataset: '{self.dataset_name}', class: '{self.class_name}'")
        
        success = True
        created_videos = []
        
        # Create video for the class
        logger.info(f"Creating videos for class: {self.class_name}")
        
        args = [
            "--dataset", self.dataset_name,
            "--class_name", self.class_name,
            "--dataset_base", str(self.base_output_dir),
            "--output_base", self.video_config["output_base"],
            "--fps", str(self.video_config["fps"]),
            "--resize_method", self.video_config["resize_method"]
        ]
        
        # Add optional flags
        if self.video_config["enhance"]:
            args.append("--enhance")
        
        if self.video_config["separate_videos"]:
            args.append("--separate")
        
        # Run video creation script
        if self.run_script("create_dataset_video.py", args):
            created_videos.append(self.class_name)
            logger.info(f"✅ Videos created successfully for {self.class_name}")
            self.pipeline_status["videos_created"] = True
            logger.info(f"✅ Video creation completed successfully")
        else:
            logger.error(f"❌ Failed to create videos for {self.class_name}")
            success = False
        
        return success
    
    def run_video_creation_only(self) -> bool:
        """Run only video creation from existing dataset."""
        logger.info("🎬 Starting video creation from existing dataset...")
        start_time = time.time()
        
        # Create videos
        if not self.step_5_create_videos():
            return False
        
        # Print summary
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"🎉 Video creation completed successfully in {duration:.1f} seconds!")
        
        # Print video creation info
        self.print_video_summary()
        
        return True
    
    def print_video_summary(self):
        """Print video creation summary."""
        print("\n" + "="*70)
        print("VIDEO CREATION SUMMARY")
        print("="*70)
        
        print(f"\n📁 Dataset: {self.dataset_name}")
        print(f"📁 Class Name: {self.class_name}")
        print(f"📹 Video output: {self.video_config['output_base']}/{self.dataset_name}/{self.class_name}/")
        
        # Video configuration
        print(f"\n⚙️  Video Configuration:")
        print(f"   FPS: {self.video_config['fps']} (1 second per image)")
        print(f"   Resize method: {self.video_config['resize_method']}")
        print(f"   Enhancement: {'enabled' if self.video_config['enhance'] else 'disabled'}")
        print(f"   Separate videos: {'yes' if self.video_config['separate_videos'] else 'no (combined)'}")
        
        # Video status
        if self.pipeline_status["videos_created"]:
            print(f"\n✅ Video Status: Successfully created")
        else:
            print(f"\n❌ Video Status: Creation failed")
        
        print(f"\n💡 Videos location:")
        print(f"   {self.video_config['output_base']}/{self.dataset_name}/{self.class_name}/")
        print("\n" + "="*70)
    
    def generate_dataset_info(self) -> Dict:
        """Generate comprehensive dataset information."""
        dataset_info = {
            "dataset_name": self.dataset_name,
            "creation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_directory": str(self.source_dir),
            "dataset_directory": str(self.dataset_dir),
            "configuration": self.config.copy(),
            "pipeline_status": self.pipeline_status.copy(),
            "statistics": {}
        }
        
        # Collect file counts
        try:
            train_good_dir = self.dataset_dir / "train" / "good"
            test_good_dir = self.dataset_dir / "test" / "good"
            
            dataset_info["statistics"]["train_images"] = len(list(train_good_dir.glob("*.png"))) if train_good_dir.exists() else 0
            dataset_info["statistics"]["test_good_images"] = len(list(test_good_dir.glob("*.png"))) if test_good_dir.exists() else 0
            
            defect_stats = {}
            for defect_type in self.config["defect_types"]:
                defect_dir = self.dataset_dir / "test" / defect_type
                mask_dir = self.dataset_dir / "ground_truth" / defect_type
                
                defect_count = len(list(defect_dir.glob("*.png"))) if defect_dir.exists() else 0
                mask_count = len(list(mask_dir.glob("*_mask.png"))) if mask_dir.exists() else 0
                
                defect_stats[defect_type] = {
                    "defect_images": defect_count,
                    "mask_images": mask_count
                }
            
            dataset_info["statistics"]["defect_types"] = defect_stats
            
        except Exception as e:
            logger.warning(f"Could not collect full statistics: {e}")
        
        return dataset_info
    
    def save_dataset_info(self, dataset_info: Dict):
        """Save dataset information to JSON file."""
        info_path = self.dataset_dir / "dataset_info.json"
        
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        logger.info(f"Dataset information saved to: {info_path}")
    
    def print_summary(self, dataset_info: Dict):
        """Print dataset creation summary."""
        print("\n" + "="*70)
        print("FABRIC DEFECT DATASET CREATION SUMMARY")
        print("="*70)
        
        print(f"\n📁 Dataset: {dataset_info['dataset_name']}")
        print(f"📍 Location: {dataset_info['dataset_directory']}")
        print(f"📅 Created: {dataset_info['creation_date']}")
        
        # Configuration
        config = dataset_info['configuration']
        print(f"\n⚙️  Configuration:")
        print(f"   Image size: {config['image_size']}x{config['image_size']}")
        print(f"   Train ratio: {config['train_ratio']}")
        print(f"   Images per defect: {config['images_per_defect']}")
        print(f"   Defect types: {', '.join(config['defect_types'])}")
        
        # Statistics
        stats = dataset_info['statistics']
        if stats:
            print(f"\n📊 Statistics:")
            print(f"   Training images: {stats.get('train_images', 0)}")
            print(f"   Test good images: {stats.get('test_good_images', 0)}")
            
            if 'defect_types' in stats:
                print(f"   Defect images by type:")
                for defect_type, defect_stats in stats['defect_types'].items():
                    print(f"      {defect_type}: {defect_stats['defect_images']} images, {defect_stats['mask_images']} masks")
        
        # Pipeline status
        status = dataset_info['pipeline_status']
        print(f"\n✅ Pipeline Status:")
        for step, completed in status.items():
            status_icon = "✅" if completed else "❌"
            print(f"   {status_icon} {step.replace('_', ' ').title()}")
        
        print("\n" + "="*70)
    
    def run_full_pipeline(self) -> bool:
        """Run the complete dataset creation pipeline."""
        logger.info("🚀 Starting full fabric defect dataset creation pipeline...")
        start_time = time.time()
        
        # Step 1: Create structure
        # if not self.step_1_create_structure():
        #     return False
        
        # Step 2: Preprocess images
        if not self.step_2_preprocess_images():
            return False
        
        # Step 3: Generate defects
        if not self.step_3_generate_defects():
            return False
        
        # Step 4: Validate masks
        if not self.step_4_validate_masks():
            return False
        
        # Generate and save dataset info
        dataset_info = self.generate_dataset_info()
        self.save_dataset_info(dataset_info)
        
        # Print summary
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"🎉 Pipeline completed successfully in {duration:.1f} seconds!")
        self.print_summary(dataset_info)
        
        return True
    
    def run_partial_pipeline(self, steps: List[str]) -> bool:
        """Run specific steps of the pipeline."""
        step_functions = {
            # "structure": self.step_1_create_structure,
            "preprocess": self.step_2_preprocess_images,
            "defects": self.step_3_generate_defects,
            "validate": self.step_4_validate_masks,
            "videos": self.step_5_create_videos
        }
        
        success = True
        for step in steps:
            if step in step_functions:
                logger.info(f"Running step: {step}")
                if not step_functions[step]():
                    success = False
                    break
            else:
                logger.error(f"Unknown step: {step}")
                success = False
                break
        
        if success:
            dataset_info = self.generate_dataset_info()
            self.save_dataset_info(dataset_info)
            self.print_summary(dataset_info)
        
        return success


def main():
    parser = argparse.ArgumentParser(description="Fabric defect dataset creation orchestrator")
    parser.add_argument("--source", required=True,
                       help="Source directory containing good fabric images")
    parser.add_argument("--dataset", required=True,
                       help="Dataset name (e.g., custom, wfdd, mvtec)")
    parser.add_argument("--class_name", 
                       help="Class name (e.g., grid, yellow_cloth, bottle). If not provided, will be extracted from dataset name for backward compatibility.")
    parser.add_argument("--output_dir", 
                       help="Base output directory (default: ../datasets)")
    
    # Pipeline control
    parser.add_argument("--full_pipeline", action="store_true", default=True,
                       help="Run complete dataset creation pipeline (default: True)")
    parser.add_argument("--create_videos", action="store_true", default=True,
                       help="Create videos from test images (default: True)")
    parser.add_argument("--no_pipeline", action="store_true",
                       help="Disable full pipeline (use with --steps)")
    parser.add_argument("--no_videos", action="store_true",
                       help="Disable video creation")
    parser.add_argument("--steps", nargs="+", 
                       choices=["structure", "preprocess", "defects", "validate", "videos"],
                       help="Run specific pipeline steps (disables defaults)")
    
    # Configuration
    parser.add_argument("--image_size", type=int, default=384,
                       help="Target image size (default: 384)")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="Training split ratio (default: 0.8)")
    parser.add_argument("--images_per_defect", type=int, default=50,
                       help="Images to generate per defect type (default: 50)")
    parser.add_argument("--defect_types", nargs="+",
                       default=["hole", "foreign_yarn", "missing_yarn", "slab", "spot"],
                       help="Defect types to generate")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    
    # Video configuration
    parser.add_argument("--video_fps", type=int, default=1,
                       help="Video FPS (default: 1 = 1 second per image)")
    parser.add_argument("--video_resize_method", 
                       choices=["none", "largest", "preserve_aspect", "crop"],
                       default="none",
                       help="Video resize method (default: none)")
    parser.add_argument("--video_enhance", action="store_true",
                       help="Apply image enhancement to video frames")
    parser.add_argument("--video_separate", action="store_true",
                       help="Create separate videos for each defect type instead of combined single video")
    parser.add_argument("--video_output", default="test-video",
                       help="Video output directory (default: test-video)")
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = DatasetOrchestrator(
        source_dir=args.source,
        dataset_name=args.dataset,
        class_name=args.class_name,
        base_output_dir=args.output_dir
    )
    
    # Update configuration
    orchestrator.config.update({
        "image_size": args.image_size,
        "train_ratio": args.train_ratio,
        "images_per_defect": args.images_per_defect,
        "defect_types": args.defect_types,
        "seed": args.seed
    })
    
    # Update video configuration
    orchestrator.video_config.update({
        "fps": args.video_fps,
        "resize_method": args.video_resize_method,
        "enhance": args.video_enhance,
        "separate_videos": args.video_separate,  # Use the separate flag directly
        "output_base": args.video_output
    })
    
    # Handle disable flags
    if args.no_pipeline:
        args.full_pipeline = False
    if args.no_videos:
        args.create_videos = False
    
    # If steps are specified, disable defaults unless explicitly enabled
    if args.steps:
        if not args.full_pipeline:  # Only disable if not explicitly enabled
            args.full_pipeline = False
        if not args.create_videos:  # Only disable if not explicitly enabled
            args.create_videos = False
    
    # Determine execution mode
    success = False
    
    if args.steps:
        # Specific steps mode (takes precedence)
        if "videos" in args.steps:
            # If videos step is included, no need to validate dataset creation inputs
            # Just validate video creation inputs when the step runs
            pass
        else:
            # Validate dataset creation inputs for other steps
            if not orchestrator.validate_inputs():
                logger.error("Input validation failed!")
                sys.exit(1)
        
        success = orchestrator.run_partial_pipeline(args.steps)
        
        # If videos step was included, add video creation step
        if success and "videos" in args.steps:
            orchestrator.print_video_summary()
            
    elif args.full_pipeline and args.create_videos:
        # Combined mode: full pipeline + videos (default behavior)
        logger.info("🚀 Running combined pipeline (dataset creation + video generation)...")
        
        # First validate dataset creation inputs
        if not orchestrator.validate_inputs():
            logger.error("Input validation failed!")
            sys.exit(1)
        
        # Run full pipeline
        if orchestrator.run_full_pipeline():
            # Then create videos
            success = orchestrator.step_5_create_videos()
        else:
            success = False
            
    elif args.full_pipeline:
        # Dataset creation only mode
        logger.info("🚀 Running dataset creation pipeline...")
        if not orchestrator.validate_inputs():
            logger.error("Input validation failed!")
            sys.exit(1)
        success = orchestrator.run_full_pipeline()
        
    elif args.create_videos:
        # Video creation only mode
        logger.info("🚀 Running video creation from existing dataset...")
        success = orchestrator.run_video_creation_only()
        
    else:
        logger.error("❌ No operation specified. Use --help for usage information.")
        sys.exit(1)
    
    # Final result handling
    if success:
        if args.create_videos and not (args.full_pipeline or args.steps):
            # Video creation only
            logger.info("🎉 Video creation completed successfully!")
            print(f"\n💡 Videos location:")
            print(f"   {orchestrator.video_config['output_base']}/{orchestrator.dataset_name}/")
        else:
            # Dataset creation (with or without videos)
            logger.info("🎉 Operation completed successfully!")
            print(f"\n💡 Next steps:")
            print(f"   1. Review the generated content:")
            if args.full_pipeline or (args.steps and "preprocess" in args.steps):
                print(f"      Dataset: {orchestrator.dataset_dir}")
            if args.create_videos or (args.steps and "videos" in args.steps):
                print(f"      Videos: {orchestrator.video_config['output_base']}/{orchestrator.dataset_name}/")
            if args.full_pipeline or (args.steps and "preprocess" in args.steps):
                print(f"   2. Use the dataset with GLASS training:")
                print(f"      python main.py dataset -d {args.dataset} {orchestrator.base_output_dir}")
        sys.exit(0)
    else:
        logger.error("❌ Operation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
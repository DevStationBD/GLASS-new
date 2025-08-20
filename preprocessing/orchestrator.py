#!/usr/bin/env python3
"""
Fabric Defect Dataset Creation Orchestrator
Main script that coordinates the entire dataset creation pipeline.

Usage:
    python orchestrator.py --source raw-data/custom-grid/good-images --dataset custom_grid --full-pipeline
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
    
    def __init__(self, source_dir: str, dataset_name: str, base_output_dir: str = None):
        self.source_dir = Path(source_dir)
        self.dataset_name = dataset_name
        self.base_output_dir = Path(base_output_dir) if base_output_dir else Path(__file__).parent.parent / "datasets"
        self.dataset_dir = self.base_output_dir / dataset_name
        self.scripts_dir = Path(__file__).parent
        
        # Pipeline configuration
        self.config = {
            "image_size": 384,
            "train_ratio": 0.8,
            "images_per_defect": 50,
            "defect_types": ["hole", "foreign_yarn", "missing_yarn", "slab", "spot"],
            "seed": 42
        }
        
        # Track pipeline progress
        self.pipeline_status = {
            "structure_created": False,
            "images_preprocessed": False,
            "defects_generated": False,
            "masks_validated": False
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
            "--dataset_name", self.dataset_name,
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
        if not self.step_1_create_structure():
            return False
        
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
            "structure": self.step_1_create_structure,
            "preprocess": self.step_2_preprocess_images,
            "defects": self.step_3_generate_defects,
            "validate": self.step_4_validate_masks
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
                       help="Dataset name (e.g., custom_grid)")
    parser.add_argument("--output_dir", 
                       help="Base output directory (default: ../datasets)")
    
    # Pipeline control
    parser.add_argument("--full_pipeline", action="store_true",
                       help="Run complete pipeline")
    parser.add_argument("--steps", nargs="+", 
                       choices=["structure", "preprocess", "defects", "validate"],
                       help="Run specific pipeline steps")
    
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
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = DatasetOrchestrator(
        source_dir=args.source,
        dataset_name=args.dataset,
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
    
    # Validate inputs
    if not orchestrator.validate_inputs():
        logger.error("Input validation failed!")
        sys.exit(1)
    
    # Run pipeline
    success = False
    if args.full_pipeline:
        success = orchestrator.run_full_pipeline()
    elif args.steps:
        success = orchestrator.run_partial_pipeline(args.steps)
    else:
        logger.error("Please specify --full_pipeline or --steps")
        sys.exit(1)
    
    if success:
        logger.info("🎉 Dataset creation completed successfully!")
        print(f"\n💡 Next steps:")
        print(f"   1. Review the generated dataset in: {orchestrator.dataset_dir}")
        print(f"   2. Use the dataset with GLASS training:")
        print(f"      python main.py dataset -d {args.dataset} {orchestrator.base_output_dir}")
        sys.exit(0)
    else:
        logger.error("❌ Dataset creation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
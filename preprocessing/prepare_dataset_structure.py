#!/usr/bin/env python3
"""
Dataset Structure Preparation Script for Custom Grid Fabric Dataset
Creates WFDD-compatible directory structure for fabric defect detection.

Usage:
    python prepare_dataset_structure.py --dataset_name custom_grid
"""

import os
import argparse
from pathlib import Path


class DatasetStructureCreator:
    """Creates standardized dataset directory structure for fabric defect detection."""
    
    def __init__(self, dataset_name: str, class_name: str = None, base_path: str = None):
        self.dataset_name = dataset_name
        self.class_name = class_name or dataset_name  # Use dataset_name as class_name if not specified
        self.base_path = Path(base_path) if base_path else Path(__file__).parent.parent / "datasets"
        self.dataset_path = self.base_path / dataset_name
        self.class_path = self.dataset_path / self.class_name
        
        # Define defect types for fabric analysis
        self.defect_types = [
            "hole",           # Circular/irregular missing fabric areas
            "foreign_yarn",   # Different colored threads/yarns
            "missing_yarn",   # Linear gaps where threads should be
            "slab",          # Thick yarn sections
            "spot"           # Stains, contamination marks
        ]
    
    def create_directory_structure(self):
        """Create complete GLASS-compatible directory structure."""
        print(f"Creating GLASS dataset structure for '{self.dataset_name}' with class '{self.class_name}'...")
        
        # Create base dataset directory and class subdirectory
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        self.class_path.mkdir(parents=True, exist_ok=True)
        
        # Create train directory structure inside class directory
        train_path = self.class_path / "train"
        train_good_path = train_path / "good"
        train_good_path.mkdir(parents=True, exist_ok=True)
        
        # Create test directory structure inside class directory
        test_path = self.class_path / "test"
        test_good_path = test_path / "good"
        test_good_path.mkdir(parents=True, exist_ok=True)
        
        # Create test defect directories
        for defect_type in self.defect_types:
            defect_test_path = test_path / defect_type
            defect_test_path.mkdir(parents=True, exist_ok=True)
        
        # Create ground truth directory structure
        gt_path = self.class_path / "ground_truth"
        for defect_type in self.defect_types:
            defect_gt_path = gt_path / defect_type
            defect_gt_path.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ GLASS dataset structure created at: {self.class_path}")
        self._print_structure()
    
    def _print_structure(self):
        """Print the created directory structure."""
        print("\n📁 Created directory structure:")
        print(f"{self.dataset_name}/")
        print(f"└── {self.class_name}/")
        print("    ├── train/")
        print("    │   └── good/")
        print("    ├── test/")
        print("    │   ├── good/")
        for i, defect_type in enumerate(self.defect_types):
            prefix = "    │   ├──" if i < len(self.defect_types) - 1 else "    │   └──"
            print(f"{prefix} {defect_type}/")
        print("    └── ground_truth/")
        for i, defect_type in enumerate(self.defect_types):
            prefix = "        ├──" if i < len(self.defect_types) - 1 else "        └──"
            print(f"{prefix} {defect_type}/")
    
    def validate_structure(self):
        """Validate that all required directories exist."""
        required_dirs = [
            "train/good",
            "test/good"
        ]
        
        # Add defect directories
        for defect_type in self.defect_types:
            required_dirs.extend([
                f"test/{defect_type}",
                f"ground_truth/{defect_type}"
            ])
        
        missing_dirs = []
        for dir_path in required_dirs:
            full_path = self.class_path / dir_path
            if not full_path.exists():
                missing_dirs.append(dir_path)
        
        if missing_dirs:
            print(f"❌ Missing directories: {missing_dirs}")
            return False
        
        print("✅ GLASS dataset structure validation passed!")
        return True
    
    def get_structure_info(self):
        """Get information about the created structure."""
        info = {
            "dataset_name": self.dataset_name,
            "dataset_path": str(self.dataset_path),
            "defect_types": self.defect_types,
            "total_directories": 2 + (2 * len(self.defect_types))  # train/good, test/good + 2*defects
        }
        return info


def main():
    parser = argparse.ArgumentParser(description="Create dataset structure for fabric defect detection")
    parser.add_argument("--dataset_name", default="custom_grid", 
                       help="Name of the dataset (default: custom_grid)")
    parser.add_argument("--base_path", default=None,
                       help="Base path for datasets (default: ../datasets)")
    parser.add_argument("--validate", action="store_true",
                       help="Validate existing structure instead of creating")
    
    args = parser.parse_args()
    
    # Create structure creator
    creator = DatasetStructureCreator(args.dataset_name, args.base_path)
    
    if args.validate:
        creator.validate_structure()
    else:
        creator.create_directory_structure()
        creator.validate_structure()
        
        # Print summary
        info = creator.get_structure_info()
        print(f"\n📊 Summary:")
        print(f"Dataset: {info['dataset_name']}")
        print(f"Location: {info['dataset_path']}")
        print(f"Defect types: {len(info['defect_types'])}")
        print(f"Total directories: {info['total_directories']}")


if __name__ == "__main__":
    main()
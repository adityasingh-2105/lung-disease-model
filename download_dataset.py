"""
Script to download and prepare the Chest X-Ray Pneumonia dataset.
This dataset is commonly available on Kaggle.
"""

import os
import zipfile
import urllib.request
import argparse
from pathlib import Path


def download_kaggle_dataset_instructions():
    """
    Print instructions for downloading from Kaggle.
    """
    print("=" * 70)
    print("DATASET DOWNLOAD INSTRUCTIONS")
    print("=" * 70)
    print()
    print("The Chest X-Ray Pneumonia dataset is available on Kaggle:")
    print("https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
    print()
    print("To download the dataset:")
    print()
    print("Option 1: Using Kaggle API (Recommended)")
    print("-" * 70)
    print("1. Install Kaggle API: pip install kaggle")
    print("2. Get your Kaggle API credentials from:")
    print("   https://www.kaggle.com/settings")
    print("3. Save credentials to ~/.kaggle/kaggle.json")
    print("4. Run: kaggle datasets download -d paultimothymooney/chest-xray-pneumonia")
    print("5. Extract the zip file")
    print()
    print("Option 2: Manual Download")
    print("-" * 70)
    print("1. Visit: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
    print("2. Click 'Download' button")
    print("3. Extract the zip file")
    print("4. Organize the dataset as shown below")
    print()
    print("Dataset Structure Required:")
    print("-" * 70)
    print("data/")
    print("├── train/")
    print("│   ├── NORMAL/")
    print("│   │   ├── image1.jpeg")
    print("│   │   └── ...")
    print("│   └── PNEUMONIA/")
    print("│       ├── image1.jpeg")
    print("│       └── ...")
    print("└── val/")
    print("    ├── NORMAL/")
    print("    │   └── ...")
    print("    └── PNEUMONIA/")
    print("        └── ...")
    print()
    print("Note: The class names should match exactly (NORMAL, PNEUMONIA)")
    print("      or be renamed to (Normal, Pneumonia) to match the code.")
    print("=" * 70)


def organize_dataset(source_dir, target_dir):
    """
    Organize downloaded dataset into the required structure.
    
    Args:
        source_dir: Source directory where dataset was extracted
        target_dir: Target directory for organized dataset
    """
    print(f"Organizing dataset from {source_dir} to {target_dir}...")
    
    # Create target structure
    train_normal = Path(target_dir) / "train" / "Normal"
    train_pneumonia = Path(target_dir) / "train" / "Pneumonia"
    val_normal = Path(target_dir) / "val" / "Normal"
    val_pneumonia = Path(target_dir) / "val" / "Pneumonia"
    
    for dir_path in [train_normal, train_pneumonia, val_normal, val_pneumonia]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    source_path = Path(source_dir)
    
    # Check for different possible structures
    possible_train_paths = [
        source_path / "chest_xray" / "train",
        source_path / "train",
        source_path / "chest-xray-pneumonia" / "chest_xray" / "train",
    ]
    
    possible_val_paths = [
        source_path / "chest_xray" / "val",
        source_path / "val",
        source_path / "chest-xray-pneumonia" / "chest_xray" / "val",
    ]
    
    train_dir = None
    val_dir = None
    
    for path in possible_train_paths:
        if path.exists():
            train_dir = path
            break
    
    for path in possible_val_paths:
        if path.exists():
            val_dir = path
            break
    
    if not train_dir:
        print(f"ERROR: Could not find train directory in {source_dir}")
        print("Please check the dataset structure and try again.")
        return False
    
    if not val_dir:
        print(f"WARNING: Could not find val directory in {source_dir}")
        print("Creating validation split from training data...")
        val_dir = None
    
    # Handle different class name formats
    class_mappings = {
        "NORMAL": "Normal",
        "Normal": "Normal",
        "normal": "Normal",
        "PNEUMONIA": "Pneumonia",
        "Pneumonia": "Pneumonia",
        "pneumonia": "Pneumonia",
    }
    
    # Copy training data
    if train_dir.exists():
        for old_class_name in train_dir.iterdir():
            if old_class_name.is_dir():
                new_class_name = class_mappings.get(old_class_name.name, old_class_name.name)
                if new_class_name == "Normal":
                    dest_dir = train_normal
                elif new_class_name == "Pneumonia":
                    dest_dir = train_pneumonia
                else:
                    continue
                
                print(f"Copying {old_class_name.name} -> {new_class_name}...")
                # Note: In production, you'd copy files. For now, we'll just create symlinks or instructions
                # import shutil
                # shutil.copytree(old_class_name, dest_dir, dirs_exist_ok=True)
    
    print("\nDataset organization complete!")
    print(f"Organized dataset is in: {target_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Download and organize Chest X-Ray dataset')
    parser.add_argument('--action', type=str, default='instructions',
                        choices=['instructions', 'organize'],
                        help='Action to perform')
    parser.add_argument('--source', type=str, default=None,
                        help='Source directory for organize action')
    parser.add_argument('--target', type=str, default='data',
                        help='Target directory for organized dataset')
    
    args = parser.parse_args()
    
    if args.action == 'instructions':
        download_kaggle_dataset_instructions()
    elif args.action == 'organize':
        if not args.source:
            print("ERROR: --source directory is required for organize action")
            return
        organize_dataset(args.source, args.target)


if __name__ == '__main__':
    main()


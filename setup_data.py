"""
Quick setup script to help organize the dataset.
This script will help organize the dataset once it's downloaded.
"""

import os
import shutil
from pathlib import Path


def create_sample_structure():
    """Create the required directory structure."""
    base_dir = Path("data")
    dirs = [
        base_dir / "train" / "Normal",
        base_dir / "train" / "Pneumonia",
        base_dir / "val" / "Normal",
        base_dir / "val" / "Pneumonia",
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {dir_path}")
    
    print(f"\n✅ Directory structure created at: {base_dir.absolute()}")
    print("\nNext steps:")
    print("1. Download the dataset from Kaggle:")
    print("   https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
    print("2. Extract the zip file")
    print("3. Copy images from the extracted dataset to the directories above")
    print("4. Or use the organize_dataset function if you have the extracted dataset")


def organize_kaggle_dataset(source_path, target_path="data"):
    """
    Organize the Kaggle Chest X-Ray dataset into the required structure.
    
    Args:
        source_path: Path to the extracted Kaggle dataset
        target_path: Target directory (default: "data")
    """
    source = Path(source_path)
    target = Path(target_path)
    
    # Create target structure
    train_normal = target / "train" / "Normal"
    train_pneumonia = target / "train" / "Pneumonia"
    val_normal = target / "val" / "Normal"
    val_pneumonia = target / "val" / "Pneumonia"
    
    for dir_path in [train_normal, train_pneumonia, val_normal, val_pneumonia]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Find the chest_xray directory (common Kaggle structure)
    chest_xray_paths = [
        source / "chest_xray",
        source / "chest-xray-pneumonia" / "chest_xray",
        source,
    ]
    
    chest_xray_dir = None
    for path in chest_xray_paths:
        if path.exists() and (path / "train").exists():
            chest_xray_dir = path
            break
    
    if not chest_xray_dir:
        print(f"❌ Error: Could not find chest_xray directory in {source}")
        print("\nPlease check if the dataset is extracted correctly.")
        print("Expected structure: .../chest_xray/train/ and .../chest_xray/val/")
        return False
    
    print(f"✅ Found dataset at: {chest_xray_dir}")
    
    # Organize training data
    train_source = chest_xray_dir / "train"
    if train_source.exists():
        print("\nOrganizing training data...")
        for class_dir in train_source.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                # Normalize class names
                if class_name.upper() == "NORMAL":
                    dest = train_normal
                elif class_name.upper() == "PNEUMONIA":
                    dest = train_pneumonia
                else:
                    print(f"⚠️  Skipping unknown class: {class_name}")
                    continue
                
                # Copy files
                files = list(class_dir.glob("*.*"))
                print(f"  Copying {len(files)} files from {class_name}...")
                for file in files:
                    shutil.copy2(file, dest / file.name)
    
    # Organize validation data
    val_source = chest_xray_dir / "val"
    if val_source.exists():
        print("\nOrganizing validation data...")
        for class_dir in val_source.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                if class_name.upper() == "NORMAL":
                    dest = val_normal
                elif class_name.upper() == "PNEUMONIA":
                    dest = val_pneumonia
                else:
                    continue
                
                files = list(class_dir.glob("*.*"))
                print(f"  Copying {len(files)} files from {class_name}...")
                for file in files:
                    shutil.copy2(file, dest / file.name)
    
    print(f"\n✅ Dataset organized successfully!")
    print(f"   Training Normal: {len(list(train_normal.glob('*.*')))} files")
    print(f"   Training Pneumonia: {len(list(train_pneumonia.glob('*.*')))} files")
    print(f"   Validation Normal: {len(list(val_normal.glob('*.*')))} files")
    print(f"   Validation Pneumonia: {len(list(val_pneumonia.glob('*.*')))} files")
    
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        source = sys.argv[1]
        organize_kaggle_dataset(source)
    else:
        print("Creating sample directory structure...")
        create_sample_structure()
        print("\n" + "="*60)
        print("To organize an existing dataset, run:")
        print(f"  python3 setup_data.py /path/to/extracted/dataset")
        print("="*60)


"""
Data preprocessing utilities for chest X-ray images.
"""

import torch
from torchvision import transforms
from PIL import Image
import numpy as np


def get_train_transforms():
    """
    Get data augmentation transforms for training.
    
    Returns:
        Composition of transforms for training data
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # For grayscale images
    ])


def get_val_transforms():
    """
    Get transforms for validation/test data.
    
    Returns:
        Composition of transforms for validation data
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # For grayscale images
    ])


def preprocess_image(image_path, transform=None):
    """
    Preprocess a single image for prediction.
    
    Args:
        image_path: Path to the image file
        transform: Optional transform to apply (default: validation transform)
        
    Returns:
        Preprocessed image tensor ready for model input
    """
    if transform is None:
        transform = get_val_transforms()
    
    try:
        # Load image
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        
        # Apply transforms
        image_tensor = transform(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")


def preprocess_uploaded_image(uploaded_file, transform=None):
    """
    Preprocess an uploaded image file (e.g., from Streamlit).
    
    Args:
        uploaded_file: File-like object from Streamlit file uploader
        transform: Optional transform to apply (default: validation transform)
        
    Returns:
        Preprocessed image tensor ready for model input
    """
    if transform is None:
        transform = get_val_transforms()
    
    try:
        # Convert uploaded file to PIL Image
        image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
        
        # Apply transforms
        image_tensor = transform(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    
    except Exception as e:
        raise ValueError(f"Error preprocessing uploaded image: {str(e)}")


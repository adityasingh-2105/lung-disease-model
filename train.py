"""
Training script for the chest X-ray CNN model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os
import argparse
from tqdm import tqdm
from model import create_model
from preprocessing import get_train_transforms, get_val_transforms


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: CNN model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on (cuda or cpu)
        
    Returns:
        Average training loss and accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc='Training')
    
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': running_loss / len(train_loader),
            'acc': 100 * correct / total
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """
    Validate the model.
    
    Args:
        model: CNN model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run on (cuda or cpu)
        
    Returns:
        Average validation loss and accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation')
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': running_loss / len(val_loader),
                'acc': 100 * correct / total
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description='Train CNN for chest X-ray classification')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory (should contain train/val subdirectories)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save trained models (default: models)')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes (default: 2)')
    parser.add_argument('--input_channels', type=int, default=1,
                        help='Number of input channels (1 for grayscale, 3 for RGB)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Prepare data
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise ValueError(f"Data directories not found. Expected {train_dir} and {val_dir}")
    
    # Create datasets
    train_dataset = ImageFolder(train_dir, transform=get_train_transforms())
    val_dataset = ImageFolder(val_dir, transform=get_val_transforms())
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    print(f'Train samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    print(f'Classes: {train_dataset.classes}')
    
    # Create model
    model = create_model(num_classes=args.num_classes, input_channels=args.input_channels)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    best_val_acc = 0.0
    
    print(f'\nStarting training for {args.epochs} epochs...\n')
    
    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')
        print('-' * 40)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(args.model_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f'New best model saved! Validation accuracy: {val_acc:.2f}%\n')
        else:
            print()
    
    print(f'Training complete! Best validation accuracy: {best_val_acc:.2f}%')
    print(f'Best model saved to: {os.path.join(args.model_dir, "best_model.pth")}')


if __name__ == '__main__':
    main()


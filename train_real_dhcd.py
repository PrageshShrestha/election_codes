#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pickle

# Import the same model architecture
from pytorch_cnn_gru_model import CNN_GRU_Model

class DHCD_Dataset(Dataset):
    """Load real DHCD dataset from .npz file"""
    def __init__(self, data_path, transform=None, train=True):
        self.transform = transform
        self.train = train
        
        # Load the DHCD dataset
        print(f"Loading DHCD dataset from: {data_path}")
        data = np.load(data_path)
        
        if train:
            self.images = data['arr_0']  # Training images
            self.labels = data['arr_1']  # Training labels
            print(f"Training data: {len(self.images)} samples")
        else:
            self.images = data['arr_2']  # Test images 
            self.labels = data['arr_3']  # Test labels
            print(f"Test data: {len(self.images)} samples")
        
        # Convert to proper format
        self.images = self.images.astype(np.uint8)
        self.labels = self.labels.astype(np.int64)
        
        # Adjust labels from 1-46 to 0-45 for PyTorch
        self.labels = self.labels - 1
        
        print(f"Image shape: {self.images.shape}")
        print(f"Label shape: {self.labels.shape}")
        print(f"Unique labels: {np.unique(self.labels)}")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert numpy array to PIL Image
        if len(image.shape) == 2:  # If grayscale
            image = np.stack([image]*3, axis=-1)  # Convert to 3 channels
        
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cuda'):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0.0
    best_model_state = None
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"New best validation accuracy: {best_val_acc:.2f}%")
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation accuracy: {best_val_acc:.2f}%")
    
    return model, train_losses, val_losses, train_accs, val_accs

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('real_dhcd_training_history.png')
    print("Training history plot saved as 'real_dhcd_training_history.png'")
    plt.show()

def main():
    # Configuration
    data_path = './DHCD_Dataset/dataset/dataset.npz'
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # Check if dataset exists
    if not os.path.exists(data_path):
        print(f"❌ Dataset not found at: {data_path}")
        print("Please ensure the DHCD dataset is available.")
        return
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load real DHCD dataset
    try:
        train_dataset = DHCD_Dataset(data_path, transform=transform, train=True)
        test_dataset = DHCD_Dataset(data_path, transform=transform, train=False)
        
        print(f"\n✅ Real DHCD Dataset Loaded Successfully!")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize model (same architecture as before)
    num_classes = len(np.unique(train_dataset.labels))
    model = CNN_GRU_Model(input_shape=(3, 64, 64), num_classes=num_classes).to(device)
    
    print(f"\nModel initialized with {num_classes} classes")
    print("Model Architecture (same as original Keras):")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train model on real DHCD data
    print(f"\n🔥 TRAINING ON REAL DHCD DATASET 🔥")
    trained_model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, test_loader, criterion, optimizer, num_epochs, device
    )
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    # Save model
    torch.save(trained_model.state_dict(), 'pytorch_cnn_gru_real_dhcd.pth')
    print("\n✅ Model saved as 'pytorch_cnn_gru_real_dhcd.pth'")
    
    # Final evaluation on test set
    trained_model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = trained_model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print("\n=== FINAL EVALUATION ON REAL DHCD TEST SET ===")
    accuracy = accuracy_score(all_labels, all_preds) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Class names (Devanagari digits 0-9)
    class_names = [f'Digit_{i}' for i in range(num_classes)]
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    
    print(f"\n🎉 TRAINING COMPLETE ON REAL DHCD DATA!")
    print(f"Best validation accuracy: {max(val_accs):.2f}%")
    print(f"Final test accuracy: {accuracy:.2f}%")
    print("Model is now ready for real Devanagari digit recognition!")

if __name__ == "__main__":
    main()

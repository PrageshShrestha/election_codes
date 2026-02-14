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

# Copy exact same architecture from original model
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.9):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.bn(x)
        x = self.pool(x)
        return x

class CNN_GRU_Model(nn.Module):
    def __init__(self, input_shape=(3, 64, 64), num_classes=46):
        super(CNN_GRU_Model, self).__init__()
        
        # Exact same CNN architecture as original
        self.conv_block1 = ConvBlock(3, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        
        # Global Max Pooling
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Calculate the flattened feature size
        self._calculate_conv_output_size(input_shape)
        
        # GRU Layer
        self.gru = nn.GRU(self.conv_output_size, 64, batch_first=True)
        
        # Fully Connected Layers (same as original)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)  # 46 classes for DHCD
        
        self.dropout = nn.Dropout(0.5)
        
    def _calculate_conv_output_size(self, input_shape):
        # Create a dummy input to calculate the output size after conv layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_input = self.conv_block1(dummy_input)
            dummy_input = self.conv_block2(dummy_input)
            dummy_input = self.conv_block3(dummy_input)
            dummy_input = self.conv_block4(dummy_input)
            dummy_input = self.global_max_pool(dummy_input)
            dummy_input = dummy_input.view(dummy_input.size(0), -1)
            self.conv_output_size = dummy_input.size(1)
    
    def forward(self, x):
        # CNN Feature Extraction (exact same as original)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        
        # Global Max Pooling
        x = self.global_max_pool(x)
        
        # Reshape for GRU: (batch_size, sequence_length, features)
        x = x.view(x.size(0), -1)  # Flatten
        x = x.unsqueeze(1)  # Add sequence dimension: (batch, 1, features)
        
        # GRU Layer
        x, _ = self.gru(x)
        x = x[:, -1, :]  # Take the last output
        
        # Fully Connected Layers (exact same as original)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

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
        print(f"Number of classes: {len(np.unique(self.labels))}")
        
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

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30, device='cuda'):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0.0
    best_model_state = None
    
    print(f"Starting training for {num_epochs} epochs on real DHCD data...")
    
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
            
            if batch_idx % 100 == 0:
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
    plt.savefig('dhcd_46_classes_training.png')
    print("Training history plot saved as 'dhcd_46_classes_training.png'")
    plt.show()

def main():
    # Configuration
    data_path = './DHCD_Dataset/dataset/dataset.npz'
    batch_size = 32
    num_epochs = 30
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # Check if dataset exists
    if not os.path.exists(data_path):
        print(f"❌ Dataset not found at: {data_path}")
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
    
    # Initialize model with exact same architecture, 46 classes for DHCD
    num_classes = len(np.unique(train_dataset.labels))  # Should be 46
    model = CNN_GRU_Model(input_shape=(3, 64, 64), num_classes=num_classes).to(device)
    
    print(f"\n🔥 MODEL ARCHITECTURE (EXACT COPY FROM ORIGINAL) 🔥")
    print(f"Model initialized with {num_classes} classes (DHCD has 46 classes)")
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
    print(f"\n🚀 TRAINING ON REAL DHCD DATASET WITH 46 CLASSES 🚀")
    trained_model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, test_loader, criterion, optimizer, num_epochs, device
    )
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    # Save model
    torch.save(trained_model.state_dict(), 'pytorch_cnn_gru_dhcd_46.pth')
    print("\n✅ Model saved as 'pytorch_cnn_gru_dhcd_46.pth'")
    
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
    
    # Class names (DHCD classes 1-46)
    class_names = [f'Class_{i+1}' for i in range(num_classes)]
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    
    print(f"\n🎉 TRAINING COMPLETE ON REAL DHCD DATA!")
    print(f"Best validation accuracy: {max(val_accs):.2f}%")
    print(f"Final test accuracy: {accuracy:.2f}%")
    print(f"Model trained on {num_classes} DHCD classes with exact same architecture!")

if __name__ == "__main__":
    main()

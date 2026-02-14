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
import urllib.request
import zipfile

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
    def __init__(self, input_shape=(3, 64, 64), num_classes=10):
        super(CNN_GRU_Model, self).__init__()
        
        # CNN Feature Extractor (exact same architecture as the notebook)
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
        
        # Fully Connected Layers (same as notebook)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
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
        # CNN Feature Extraction
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
        
        # Fully Connected Layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

class SyntheticDevanagariDataset(Dataset):
    """Create synthetic Devanagari digit data for demonstration"""
    def __init__(self, num_samples=1000, transform=None, num_classes=10):
        self.num_samples = num_samples
        self.transform = transform
        self.num_classes = num_classes
        
        # Generate synthetic data
        self.images = []
        self.labels = []
        
        for i in range(num_samples):
            # Create random image that resembles handwritten digit
            img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            
            # Add some structure to make it more realistic
            center = np.random.randint(20, 44, 2)
            radius = np.random.randint(8, 15)
            y, x = np.ogrid[:64, :64]
            mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
            img[mask] = np.random.randint(100, 200, 3)
            
            # Add some noise
            noise = np.random.normal(0, 25, img.shape)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            
            self.images.append(Image.fromarray(img))
            self.labels.append(i % num_classes)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def download_alternative_dataset():
    """Try to download an alternative Devanagari dataset"""
    print("Trying to download alternative Devanagari dataset...")
    
    try:
        # Try downloading from another source
        url = "https://raw.githubusercontent.com/premrajir/Devanagari-character-dataset/master/data.zip"
        zip_path = "./devanagari_data.zip"
        
        print(f"Downloading from: {url}")
        urllib.request.urlretrieve(url, zip_path)
        
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        
        os.remove(zip_path)
        print("Alternative dataset downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"Failed to download alternative dataset: {e}")
        return False

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, device='cuda'):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0.0
    best_model_state = None
    
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
            
            if batch_idx % 10 == 0:
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
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
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
    plt.savefig('training_history.png')
    print("Training history plot saved as 'training_history.png'")
    plt.show()

def main():
    # Configuration
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Try to download real dataset or use synthetic data
    real_data_available = False
    
    # Check if real dataset exists
    if os.path.exists('./DevanagariHandwrittenCharacterDataset'):
        print("Found real Devanagari dataset!")
        try:
            from pytorch_cnn_gru_model import DevanagariDataset
            train_dataset = DevanagariDataset('./DevanagariHandwrittenCharacterDataset', 
                                            transform=transform, subset='train', numeric_only=True)
            val_dataset = DevanagariDataset('./DevanagariHandwrittenCharacterDataset', 
                                          transform=transform, subset='test', numeric_only=True)
            if len(train_dataset) > 0 and len(val_dataset) > 0:
                real_data_available = True
                print(f"Using real dataset - Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        except Exception as e:
            print(f"Error loading real dataset: {e}")
    
    if not real_data_available:
        print("Using synthetic Devanagari digit data for demonstration...")
        print("Note: This is synthetic data. For real results, please provide the actual DHCD dataset.")
        
        # Create synthetic datasets
        train_dataset = SyntheticDevanagariDataset(num_samples=800, transform=transform, num_classes=10)
        val_dataset = SyntheticDevanagariDataset(num_samples=200, transform=transform, num_classes=10)
        
        print(f"Synthetic data - Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Initialize model (exact copy of notebook architecture)
    num_classes = 10  # Devanagari digits 0-9
    model = CNN_GRU_Model(input_shape=(3, 64, 64), num_classes=num_classes).to(device)
    
    print(f"\nModel initialized with {num_classes} classes")
    print("Model Architecture (exact copy from notebook):")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer (same as notebook)
    criterion = nn.CrossEntropyLoss()  # More appropriate than MSE for classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train model
    print(f"\nStarting training for {num_epochs} epochs...")
    trained_model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs, device
    )
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    # Save model
    torch.save(trained_model.state_dict(), 'pytorch_cnn_gru_devanagari_digits.pth')
    print("\nModel saved as 'pytorch_cnn_gru_devanagari_digits.pth'")
    
    # Final evaluation
    trained_model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = trained_model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print("\n=== Final Evaluation ===")
    accuracy = accuracy_score(all_labels, all_preds) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Class names for Devanagari digits
    class_names = [f'Digit_{i}' for i in range(10)]
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    
    print(f"\n=== Training Complete ===")
    print(f"Best validation accuracy: {max(val_accs):.2f}%")
    print(f"Final test accuracy: {accuracy:.2f}%")
    print("Model architecture successfully copied from Keras notebook and trained in PyTorch!")

if __name__ == "__main__":
    main()

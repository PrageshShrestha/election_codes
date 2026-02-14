import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

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
        
        # CNN Feature Extractor (same architecture as the notebook)
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
        
        # Fully Connected Layers
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

class DevanagariDataset(Dataset):
    def __init__(self, data_dir, transform=None, subset='train', numeric_only=True):
        self.data_dir = data_dir
        self.transform = transform
        self.subset = subset
        self.numeric_only = numeric_only
        
        self.images = []
        self.labels = []
        self.class_names = []
        
        # Define numeric digit classes (0-9 in Devanagari)
        if numeric_only:
            self.numeric_classes = ['digit_0', 'digit_1', 'digit_2', 'digit_3', 'digit_4', 
                                   'digit_5', 'digit_6', 'digit_7', 'digit_8', 'digit_9']
        else:
            # All classes if needed
            self.numeric_classes = None
        
        self._load_data()
    
    def _load_data(self):
        subset_dir = os.path.join(self.data_dir, self.subset.capitalize())
        
        if not os.path.exists(subset_dir):
            print(f"Warning: {subset_dir} does not exist")
            return
        
        for class_name in sorted(os.listdir(subset_dir)):
            if self.numeric_only and not any(digit in class_name.lower() for digit in ['digit_0', 'digit_1', 'digit_2', 'digit_3', 'digit_4', 'digit_5', 'digit_6', 'digit_7', 'digit_8', 'digit_9']):
                continue
                
            class_dir = os.path.join(subset_dir, class_name)
            if os.path.isdir(class_dir):
                self.class_names.append(class_name)
                class_idx = len(self.class_names) - 1
                
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

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
    plt.show()

def main():
    # Configuration
    data_dir = './DevanagariHandwrittenCharacterDataset'  # Update this path
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    try:
        train_dataset = DevanagariDataset(data_dir, transform=transform, subset='train', numeric_only=True)
        val_dataset = DevanagariDataset(data_dir, transform=transform, subset='test', numeric_only=True)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Classes: {train_dataset.class_names}")
        
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print("No data found. Please check the dataset path.")
            return
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please make sure the DHCD dataset is available in the correct path.")
        return
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    num_classes = len(train_dataset.class_names)
    model = CNN_GRU_Model(input_shape=(3, 64, 64), num_classes=num_classes).to(device)
    
    print(f"Model initialized with {num_classes} classes")
    print(model)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train model
    print("Starting training...")
    trained_model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs, device
    )
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    # Save model
    torch.save(trained_model.state_dict(), 'pytorch_cnn_gru_devanagari_digits.pth')
    print("Model saved as 'pytorch_cnn_gru_devanagari_digits.pth'")
    
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
    
    print("\nFinal Evaluation:")
    print(f"Accuracy: {accuracy_score(all_labels, all_preds) * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_dataset.class_names))

if __name__ == "__main__":
    main()

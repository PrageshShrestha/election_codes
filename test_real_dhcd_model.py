#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.9):
        super(ConvBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = torch.nn.BatchNorm2d(out_channels, momentum=momentum)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.bn(x)
        x = self.pool(x)
        return x

class CNN_GRU_Model(torch.nn.Module):
    def __init__(self, input_shape=(3, 64, 64), num_classes=46):
        super(CNN_GRU_Model, self).__init__()
        self.conv_block1 = ConvBlock(3, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.global_max_pool = torch.nn.AdaptiveMaxPool2d((1, 1))
        
        self._calculate_conv_output_size(input_shape)
        self.gru = torch.nn.GRU(self.conv_output_size, 64, batch_first=True)
        
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, num_classes)
        self.dropout = torch.nn.Dropout(0.5)
        
    def _calculate_conv_output_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_input = self.conv_block1(dummy_input)
            dummy_input = self.conv_block2(dummy_input)
            dummy_input = self.conv_block3(dummy_input)
            dummy_input = self.conv_block4(dummy_input)
            dummy_input = self.global_max_pool(dummy_input)
            self.conv_output_size = dummy_input.view(1, -1).size(1)
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.global_max_pool(x)
        x = x.view(x.size(0), -1).unsqueeze(1)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)

def predict_image(image_path, model, device, transform):
    if not os.path.exists(image_path):
        return None
    from PIL import Image

    print("helloowrld")
    some_new = Image.open(image_path)
    image = Image.open(image_path).convert('RGB')
    image.show()
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        
    return {
        'class': predicted_class.item() + 1,
        'confidence': confidence.item(),
        'is_digit_one': (predicted_class.item() + 1 == 2)
    }

def main():
    model_path = 'pytorch_cnn_gru_dhcd_46.pth'
    test_images = ['test.png', 'test2.png']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # DHCD Class mapping
    class_mapping = {
        1: '० (0)', 2: '१ (1)', 3: '२ (2)', 4: '३ (3)', 5: '४ (4)', 6: '५ (5)', 7: '६ (6)', 8: '७ (7)', 9: '८ (8)', 10: '९ (9)',
        11: 'क (ka)', 12: 'ख (kha)', 13: 'ग (ga)', 14: 'घ (gha)', 15: 'ङ (ṅa)', 16: 'च (ca)', 17: 'छ (cha)', 18: 'ज (ja)', 19: 'झ (jha)', 20: 'ञ (ña)',
        21: 'ट (ṭa)', 22: 'ठ (ṭha)', 23: 'ड (ḍa)', 24: 'ढ (ḍha)', 25: 'ण (ṇa)', 26: 'त (ta)', 27: 'थ (tha)', 28: 'द (da)', 29: 'ध (dha)', 30: 'न (na)',
        31: 'प (pa)', 32: 'फ (pha)', 33: 'ब (ba)', 34: 'भ (bha)', 35: 'म (ma)', 36: 'य (ya)', 37: 'र (ra)', 38: 'ल (la)', 39: 'व (va)', 40: 'श (śa)',
        41: 'ष (ṣa)', 42: 'स (sa)', 43: 'ह (ha)', 44: 'क्ष (kṣa)', 45: 'त्र (tra)', 46: 'ज्ञ (jña)'
    }
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    model = CNN_GRU_Model()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device).eval()
        
        for img in test_images:
            result = predict_image(img, model, device, transform)
            if result:
                devanagari_char = class_mapping.get(result['class'], f'Class {result["class"]}')
                print(f"{img}: {devanagari_char} ({result['confidence']:.2%})")

if __name__ == "__main__":
    main()
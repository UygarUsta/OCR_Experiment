import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from tqdm import tqdm
from thop import profile
import openvino as ov

# Karakter setini tanımlama (Türk plakalarına uygun)
CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(CHARS)}  # 0 CTC blank için ayrıldı
IDX_TO_CHAR = {idx + 1: char for idx, char in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1  # +1 CTC blank için


class PlateOCR(nn.Module):
    def __init__(self, num_classes):
        super(PlateOCR, self).__init__()
        
        # MobileNetV3 backbone
        self.backbone = models.mobilenet_v3_small(pretrained=True)
        self.backbone.classifier = nn.Identity()  # Orijinal sınıflandırıcıyı kaldır
        
        # Özellik boyutunu hesapla
        self.cnn_output_height = 2  # 64/32
        self.cnn_output_width = 6   # 192/32
        self.feature_size = 576     # MobileNetV3-Small'ın son katmanındaki kanal sayısı
        
        # Bidirectional GRU
        self.rnn = nn.GRU(
            input_size=self.feature_size,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # Son fully connected katmanı
        self.fc = nn.Linear(512, num_classes)  # 512 = 256*2 (bidirectional)

    def forward(self, x):
        # Feature extraction (B, C, H, W)
        features = self.backbone.features(x)
        
        # Reshape for RNN
        batch_size = features.size(0)
        features = features.permute(0, 2, 3, 1)  # (B, H, W, C)
        features = features.reshape(batch_size, -1, self.feature_size)  # (B, T, F)
        
        # RNN
        rnn_output, _ = self.rnn(features)
        
        # Final prediction
        output = self.fc(rnn_output)
        output = output.permute(1, 0, 2)  # (T, B, C) CTC loss için
        
        return output
    
 
class ImprovedPlateOCR(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedPlateOCR, self).__init__()
        
        # MobileNetV3-Small backbone
        self.backbone = models.mobilenet_v3_small(pretrained=True)
        self.backbone.classifier = nn.Identity()
        
        self.feature_h = 2
        self.feature_w =  10  #6 192/32
        self.feature_channels = 576
        
        # Channel Attention (SE-like block)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.feature_channels, self.feature_channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feature_channels // 16, self.feature_channels, 1),
            nn.Sigmoid()
        )
        
        # Simple Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(self.feature_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Bidirectional GRU
        self.rnn = nn.LSTM(
            input_size=self.feature_channels,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            dropout=0.2,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        # Backbone features
        features = self.backbone.features(x)
        
        # Apply channel attention
        channel_weights = self.channel_attention(features)
        features = features * channel_weights
        
        # Apply spatial attention
        spatial_weights = self.spatial_attention(features)
        features = features * spatial_weights
        
        # Reshape for sequence processing
        batch_size = features.size(0)
        features = features.permute(0, 2, 3, 1)  # (B, H, W, C)
        features = features.reshape(batch_size, -1, self.feature_channels)  # (B, T, F)
        
        # RNN processing
        rnn_output, _ = self.rnn(features)
        
        # Classification
        output = self.classifier(rnn_output)
        
        # Prepare for CTC loss
        output = output.permute(1, 0, 2)  # (T, B, C)
        
        return output

    def get_seq_length(self):
        return self.feature_h * self.feature_w
    


class MixingBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion=4, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        # Local mixing (Depthwise Conv)
        self.local_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, 
                     padding=kernel_size//2, groups=channels),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        
        # Global mixing (Multi-head Attention)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.attention_norm = nn.LayerNorm(channels)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * expansion),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(channels * expansion, channels),
        )
        self.ffn_norm = nn.LayerNorm(channels)

    def forward(self, x, H, W):
        B, T, C = x.shape
        residual = x
        
        # Local mixing branch
        x_2d = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x_local = self.local_conv(x_2d).permute(0, 2, 3, 1).view(B, T, C)
        x = residual + x_local
        
        # Global mixing branch
        x = self.attention_norm(x)
        x_global, _ = self.attention(x, x, x)
        x = x + x_global
        
        # FFN
        x = self.ffn_norm(x)
        x = x + self.ffn(x)
        
        return x

class ImprovedPlateOCRSVTR(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedPlateOCRSVTR, self).__init__()
        
        # MobileNetV3-Small backbone
        self.backbone = models.mobilenet_v3_small(pretrained=True)
        self.backbone.classifier = nn.Identity()
        
        # Feature parameters
        self.feature_h = 2
        self.feature_w = 6
        in_channels = 576
        
        # Channel reduction
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(in_channels, 256, 1),
            nn.BatchNorm2d(256),
            nn.Hardswish(inplace=True))
        
        # SVTR Neck
        self.svtr_neck = nn.ModuleList([
            MixingBlock(256, num_heads=4),
            MixingBlock(256, num_heads=4)
        ])
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Backbone features
        features = self.backbone.features(x)
        
        # Channel reduction
        features = self.channel_reduce(features)
        
        # Prepare for SVTR
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1).reshape(B, H*W, -1)
        
        # Process through SVTR neck
        for block in self.svtr_neck:
            features = block(features, H, W)
        
        # Classification
        output = self.classifier(features)
        return output.permute(1, 0, 2)  # (T, B, C) for CTC

    def get_seq_length(self):
        return self.feature_h * self.feature_w
    

def decode_predictions(outputs):
    # Greedy decoding
    _, max_indices = torch.max(outputs.softmax(2), 2)
    decoded_preds = []
    
    for pred in max_indices.T:  # Batch içindeki her örnek için
        chars = []
        prev_char = None
        
        for p in pred:
            p = p.item()
            if p != 0 and p != prev_char:  # CTC blank ve tekrarları atla
                chars.append(IDX_TO_CHAR.get(p, ''))
            prev_char = p
            
        decoded_preds.append(''.join(chars))
    
    return decoded_preds

def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    return images, labels

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_model(model_path, device):
    """Eğitilmiş modeli yükler"""
    model = ImprovedPlateOCR(NUM_CLASSES).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    total_params = count_parameters(model)
    print(f'Toplam parametre sayısı: {total_params:,}')
    return model

def predict_single_image(model, image_path, device):
    """Tek bir görüntü için tahmin yapar"""
    # Görüntü önişleme
    transform = transforms.Compose([
        transforms.Resize((64, 320)), #192
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Görüntüyü yükle ve transform uygula
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Batch boyutu ekle
    image = image.to(device)
    
    # Tahmin
    with torch.no_grad():
        outputs = model(image)
        outputs = torch.tensor(outputs[0])
        outputs = outputs.softmax(2)  # Olasılıklara dönüştür
        
        # En yüksek olasılıklı karakterleri al
        confidence, predictions = torch.max(outputs, dim=2)
        
        # Decode et
        decoded_pred = []
        confidence_scores = []
        prev_char = None
        
        for pred, conf in zip(predictions.squeeze(), confidence.squeeze()):
            pred_idx = pred.item()
            if pred_idx != 0 and pred_idx != prev_char:  # CTC blank ve tekrarları atla
                decoded_pred.append(IDX_TO_CHAR.get(pred_idx, ''))
                confidence_scores.append(conf.item())
            prev_char = pred_idx
        
        result = ''.join(decoded_pred)
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        return {
            'prediction': result,
            'confidence': avg_confidence,
            'char_confidences': list(zip(decoded_pred, confidence_scores))
        }
    


device = "cpu" #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = "CPU"
model = load_model('best_plate_ocr_model_0.98_0.93.pth', device)
model = ov.convert_model(model,example_input=torch.rand(1, 3, 64, 320))
model = ov.compile_model(model, DEVICE)
image_path = r'E:\rec_derpet\rec_derpet\test\34FFS025.jpg'
result = predict_single_image(model, image_path, device)
print("\nSingle Image Prediction:")
print(f"Predicted Plate: {result['prediction']}")
print(f"Confidence: {result['confidence']:.4f}")
print("Character Confidences:")
for char, conf in result['char_confidences']:
    print(f"{char}: {conf:.4f}")

#Stress test
from time import time 
from glob import glob 
import os 
import cv2 
from tqdm import tqdm 

plate_data_path = r"E:\rec_derpet_modified\rec_derpet\test\*" #r"E:\Downloads\archive\x-anylabeling-crops\license plate\*.jpg" 
files = glob(os.path.join(plate_data_path))
acc = 0
for i in tqdm(files):
    image_path = i
    image = cv2.imread(i)
    f1 = time()
    result = predict_single_image(model, image_path, device)
    #print(f"Confidence: {result['confidence']:.4f}")
    #print(f"Predicted Plate: {result['prediction']}")
    f2 = time()
    #print(f"Single Image Prediction: {f2-f1}")
    if result['prediction'] == os.path.splitext(os.path.basename(image_path))[0]:
        acc += 1
        #continue
    #cv2.imshow(f"{result['prediction']}", image)
    # ch = cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # if ch == ord('q'):
    #     break

acc_avg = acc / len(files)
print(f"Accuracy: {acc_avg}")
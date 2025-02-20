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


# 3. Model Definitions (Modularized and using timm)
# ----------------------------------------------------------------------

class ChannelAttention(nn.Module): # Reusable Channel Attention Module
    def __init__(self, feature_channels, reduction_ratio=16):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_channels, feature_channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels // reduction_ratio, feature_channels, 1),
            nn.Sigmoid()
        )
    def forward(self, features):
        channel_weights = self.channel_attention(features)
        return features * channel_weights

class SpatialAttention(nn.Module): # Reusable Spatial Attention Module
    def __init__(self, feature_channels):
        super().__init__()
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(feature_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, features):
        spatial_weights = self.spatial_attention(features)
        return features * spatial_weights


class PlateOCRModel(nn.Module): # Base PlateOCR Model - more modular
    def __init__(self, num_classes, pretrained=True,
                 rnn_hidden_size=256, rnn_layers=2, bidirectional_rnn=True,
                 use_channel_attention=False, use_spatial_attention=False):
        super().__init__()

        self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
        self.backbone.classifier = nn.Identity()
        feature_channels = 576

        self.feature_h = 2 # Fixed feature grid size after backbone - adjust if needed based on backbone and input size
        self.feature_w = 10
        self.feature_channels = feature_channels

        attention_modules = []
        if use_channel_attention:
            attention_modules.append(ChannelAttention(self.feature_channels))
        if use_spatial_attention:
            attention_modules.append(SpatialAttention(self.feature_channels))
        self.attention = nn.Sequential(*attention_modules)


        self.rnn = nn.LSTM(
            input_size=self.feature_channels,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            bidirectional=bidirectional_rnn,
            dropout=0.2,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden_size * 2 if bidirectional_rnn else rnn_hidden_size, num_classes),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        features = self.backbone.features(x) # Get the last feature map

        features = self.attention(features) # Apply attention if any

        batch_size = features.size(0)
        features = features.permute(0, 2, 3, 1)  # (B, H, W, C)
        features = features.reshape(batch_size, -1, self.feature_channels)  # (B, T, F)

        rnn_output, _ = self.rnn(features)
        output = self.classifier(rnn_output)
        return output.permute(1, 0, 2)  # (T, B, C) for CTC

    def get_seq_length(self):
        return self.feature_h * self.feature_w


class OptimizedPlateOCR_Speed(PlateOCRModel): # Inherit from base and override/configure
    def __init__(self, num_classes):
        super().__init__(num_classes,
                         pretrained=True,
                         rnn_hidden_size=128,
                         rnn_layers=1,
                         bidirectional_rnn=False,
                         use_channel_attention=True,
                         use_spatial_attention=False) # Removed spatial, simpler channel, unidirectional RNN


class OptimizedPlateOCR_Accuracy(PlateOCRModel): # Inherit and configure for accuracy
    def __init__(self, num_classes):
        super().__init__(num_classes,
                         pretrained=True,
                         rnn_hidden_size=512,
                         rnn_layers=3,
                         bidirectional_rnn=True,
                         use_channel_attention=True,
                         use_spatial_attention=True) # Stronger backbone, deeper RNN, both attentions
    


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

# class ImprovedPlateOCRSVTR(nn.Module):
#     def __init__(self, num_classes):
#         super(ImprovedPlateOCRSVTR, self).__init__()
        
#         # MobileNetV3-Small backbone
#         self.backbone = models.mobilenet_v3_small(pretrained=True)
#         self.backbone.classifier = nn.Identity()
        
#         # Feature parameters
#         self.feature_h = 2
#         self.feature_w = 6
#         in_channels = 576
        
#         # Channel reduction
#         self.channel_reduce = nn.Sequential(
#             nn.Conv2d(in_channels, 256, 1),
#             nn.BatchNorm2d(256),
#             nn.Hardswish(inplace=True))
        
#         # SVTR Neck
#         self.svtr_neck = nn.ModuleList([
#             MixingBlock(256, num_heads=4),
#             MixingBlock(256, num_heads=4)
#         ])
        
#         # Final classifier
#         self.classifier = nn.Sequential(
#             nn.Linear(256, 256),
#             nn.GELU(),
#             nn.Dropout(0.1),
#             nn.Linear(256, num_classes)
#         )

#     def forward(self, x):
#         # Backbone features
#         features = self.backbone.features(x)
        
#         # Channel reduction
#         features = self.channel_reduce(features)
        
#         # Prepare for SVTR
#         B, C, H, W = features.shape
#         features = features.permute(0, 2, 3, 1).reshape(B, H*W, -1)
        
#         # Process through SVTR neck
#         for block in self.svtr_neck:
#             features = block(features, H, W)
        
#         # Classification
#         output = self.classifier(features)
#         return output.permute(1, 0, 2)  # (T, B, C) for CTC

#     def get_seq_length(self):
#         return self.feature_h * self.feature_w

import timm 
class ImprovedPlateOCRSVTR(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedPlateOCRSVTR, self).__init__()
        
        # MobileNetV3-Small backbone
        self.backbone = timm.create_model('mobilenetv4_conv_small_050', pretrained=True,features_only=True)
        self.backbone.classifier = nn.Identity()
        
        # Feature parameters
        self.feature_h = 2
        self.feature_w = 10
        in_channels = 480
        
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
        features = self.backbone(x)[-1]
        
        # Channel reduction
        features = self.channel_reduce(features)
        
        # Prepare for SVTR
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1).reshape(B, H*W, -1)
        
        # Process through SVTR neck
        for block in self.svtr_neck:
            features = block(features, torch.tensor(H), torch.tensor(W))
        
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
    model = OptimizedPlateOCR_Speed(NUM_CLASSES).to(device)
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
model = load_model('best_acc_plate_0.9227_ocr_model.pth', device)
# model = torch.ao.quantization.quantize_dynamic(
#     model,  # the original model
#     {torch.nn.Linear,torch.nn.Conv2d},  # a set of layers to dynamically quantize
#     dtype=torch.qint8)
# model = torch.jit.script(model,torch.randn(1,3,64,320))
# model = torch.jit.optimize_for_inference(model)

import openvino as ov
model = ov.convert_model(model,example_input=torch.randn(1,3,64,320))
model = ov.compile_model(model, DEVICE)
image_path = r'E:\rec_derpet\rec_derpet\test\34FFS025.jpg'
for i in range(4):
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
    print(f"Single Image Prediction: {f2-f1}")
    if result['prediction'] == os.path.splitext(os.path.basename(image_path))[0]:
        acc += 1
    #     continue
    # cv2.imshow(f"{result['prediction']}", image)
    # ch = cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # if ch == ord('q'):
    #     break

acc_avg = acc / len(files)
print(f"Accuracy: {acc_avg}")
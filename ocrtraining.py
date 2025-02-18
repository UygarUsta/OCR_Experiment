import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from tqdm import tqdm
import numpy as np

# Karakter setini tanımlama (Türk plakalarına uygun)
CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(CHARS)}  # 0 CTC blank için ayrıldı
IDX_TO_CHAR = {idx + 1: char for idx, char in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1  # +1 CTC blank için

class PlateDataset(Dataset):
    def __init__(self, img_dir,is_training=True):
        self.img_dir = img_dir
        self.is_training = is_training
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transforms.Compose([
            transforms.Resize((64, 192)),  # Plakalar için uygun boyut
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Eğitim için veri artırma
        self.train_transform = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
            ], p=0.5),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3)
            ], p=0.2),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=(-5, 5),  # Hafif rotasyon
                    translate=(0.05, 0.05),  # Hafif öteleme
                    scale=(0.95, 1.05),  # Hafif ölçekleme
                    shear=(-5, 5)  # Hafif kesme
                )
            ], p=0.5),
            transforms.RandomApply([
                transforms.RandomPerspective(distortion_scale=0.2)
            ], p=0.3),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Görüntüyü yükle
        image = Image.open(img_path).convert('RGB')
        if self.is_training:
            image = self.train_transform(image)
        image = self.transform(image)
        
        # Etiketi al (dosya adından)
        label = os.path.splitext(img_name)[0].upper()
        
        # Etiketi sayısal indekslere dönüştür
        label_indices = [CHAR_TO_IDX[c] for c in label if c in CHAR_TO_IDX]
        
        return image, torch.tensor(label_indices, dtype=torch.long)

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
        self.feature_w = 6
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
        self.rnn = nn.GRU(
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

def train_batch(model,criterion, optimizer, images, labels, device):
    model.train()
    images = images.to(device)
    
    # Forward pass
    outputs = model(images)
    
    # CTC Loss için input lengths hesapla
    input_lengths = torch.full(size=(images.size(0),),
                             fill_value=outputs.size(0),
                             dtype=torch.long,
                             device=device)
    
    # Target lengths
    target_lengths = torch.tensor([len(l) for l in labels],
                                dtype=torch.long,
                                device=device)
    
    # Labels'ı tek bir tensor'a birleştir
    labels = torch.cat(labels).to(device)
    
    # Loss hesapla
    loss = criterion(outputs.log_softmax(2), labels, input_lengths, target_lengths)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    
    return loss.item()

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

def load_model(model_path, device):
    """Eğitilmiş modeli yükler"""
    model = PlateOCR(NUM_CLASSES).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def predict_single_image(model, image_path, device):
    """Tek bir görüntü için tahmin yapar"""
    # Görüntü önişleme
    transform = transforms.Compose([
        transforms.Resize((64, 192)),
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

def evaluate_model(model, test_loader, criterion, device):
    """Model performansını değerlendirir"""
    model.eval()
    total_loss = 0
    correct_chars = 0
    total_chars = 0
    correct_plates = 0
    total_plates = 0
    
    results = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # CTC Loss hesapla
            input_lengths = torch.full(size=(images.size(0),),
                                    fill_value=outputs.size(0),
                                    dtype=torch.long,
                                    device=device)
            target_lengths = torch.tensor([len(l) for l in labels],
                                       dtype=torch.long,
                                       device=device)
            label_tensor = torch.cat(labels).to(device)
            
            loss = criterion(outputs.log_softmax(2), label_tensor, input_lengths, target_lengths)
            total_loss += loss.item()
            
            # Decode predictions
            decoded_preds = decode_predictions(outputs)
            
            # Metrikleri hesapla
            for pred, true_label in zip(decoded_preds, labels):
                true_str = ''.join([IDX_TO_CHAR.get(idx.item(), '') for idx in true_label])
                
                # Tam plaka doğruluğu
                if pred == true_str:
                    correct_plates += 1
                total_plates += 1
                
                # Karakter bazlı doğruluk
                for p_char, t_char in zip(pred, true_str):
                    if p_char == t_char:
                        correct_chars += 1
                total_chars += len(true_str)
                
                results.append({
                    'true': true_str,
                    'pred': pred,
                    'correct': pred == true_str
                })
    
    # Metrikleri hesapla
    avg_loss = total_loss / len(test_loader)
    char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
    plate_accuracy = correct_plates / total_plates if total_plates > 0 else 0
    
    return {
        'avg_loss': avg_loss,
        'char_accuracy': char_accuracy,
        'plate_accuracy': plate_accuracy,
        'results': results
    }

# Train/Val/Test split için yardımcı fonksiyon
def split_dataset(dataset, train_ratio=0.85, val_ratio=0.10, test_ratio=0.05):
    """Dataset'i train, validation ve test olarak böler"""
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size, test_size]
    )
    
    return train_dataset, val_dataset, test_dataset

def main():
    # Parametreler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    num_epochs = 200
    best_val_loss = float('inf')
    learning_rate = 0.001

    config = {
        'batch_size': batch_size,
        'epochs': num_epochs,
        'initial_lr': learning_rate,
        'min_lr': 1e-6,
        'weight_decay': 1e-4,
    }
    
    # Dataset ve DataLoader
    full_dataset = PlateDataset('E:/rec_derpet_modified/rec_derpet//train_val_test')
    train_dataset, val_dataset, test_dataset = split_dataset(full_dataset)
    train_loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Model, loss ve optimizer
    #model = PlateOCR(NUM_CLASSES).to(device)
    model = ImprovedPlateOCR(NUM_CLASSES).to(device)
    criterion = nn.CTCLoss(zero_infinity=True, reduction='mean')
    
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), 
                                lr=config['initial_lr'],
                                weight_decay=config['weight_decay'])
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # İlk restart cycle
        T_mult=2,  # Her cycle'ı 2 katına çıkar
        eta_min=config['min_lr']
    )
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    
    # Eğitim döngüsü
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            train_loss = train_batch(model, criterion, optimizer, images, labels, device)
            total_loss += train_loss
            
            # Her 100 batch'te bir örnek tahmin göster
            if batch_idx % 100 == 0:
                model.eval()
                with torch.no_grad():
                    outputs = model(images[:1].to(device))
                    decoded = decode_predictions(outputs)
                    print(f'\nSample: True: {labels[0]}, Pred: {decoded[0]}')
                model.train()
            
            progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1)})
        
        model.eval()
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f'Epoch {epoch+1}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Val Loss: {val_metrics["avg_loss"]:.4f}')
        print(f'Val Char Accuracy: {val_metrics["char_accuracy"]:.4f}')
        print(f'Val Plate Accuracy: {val_metrics["plate_accuracy"]:.4f}')
        
        # En iyi modeli kaydet
        if val_metrics["avg_loss"] < best_val_loss:
            best_val_loss = val_metrics["avg_loss"]
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, 'best_plate_ocr_model.pth')
        
        avg_loss = total_loss / len(train_loader)
        print(f'\nEpoch {epoch+1} Average Loss: {avg_loss:.4f}')
        
        # Modeli kaydet
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'plate_ocr_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
import timm  # Import timm for more models
import albumentations as A
from albumentations.pytorch import ToTensorV2


# 1. Configuration and Setup
# ----------------------------------------------------------------------

# Character set for Turkish license plates
CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(CHARS)}  # 0 reserved for CTC blank
IDX_TO_CHAR = {idx + 1: char for idx, char in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1  # +1 for CTC blank


# 2. Dataset Definition
# ----------------------------------------------------------------------

class PlateDataset(Dataset):
    def __init__(self, img_dir, is_training=True, image_size=(64, 320)):
        self.img_dir = img_dir
        self.is_training = is_training
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.image_size = image_size

        # Define augmentations using Albumentations for more flexibility and power
        if is_training:
            self.augmentations = A.Compose([
                A.Perspective(scale=(0.05, 0.1), p=0.3),
                A.Rotate(limit=(-10, 10), p=0.4),
                A.Affine(p=0.2,shear=(0, 10)),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Blur(blur_limit=3, p=0.2),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.ToGray(p=0.2),
                A.Resize(*self.image_size, interpolation=Image.Resampling.BICUBIC), # Resize after augmentation
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.augmentations = A.Compose([
                A.Resize(*self.image_size, interpolation=Image.Resampling.BICUBIC),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        image = np.array(image) # Albumentations works with numpy arrays
        augmented = self.augmentations(image=image)
        image = augmented['image']


        label = os.path.splitext(img_name)[0].upper()
        label_indices = [CHAR_TO_IDX[c] for c in label if c in CHAR_TO_IDX]

        return image, torch.tensor(label_indices, dtype=torch.long)


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


# Kept SVTR model as it is (you can modularize if needed, but looks fine)
class MixingBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion=4, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        # Local mixing (Depthwise Conv)
        self.local_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size,
                     padding=kernel_size // 2, groups=channels),
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
        super().__init__()

        # MobileNetV3-Small backbone
        self.backbone = timm.create_model('mobilenetv4_conv_small_050', pretrained=True, features_only=True)
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
        features = features.permute(0, 2, 3, 1).reshape(B, H * W, -1)

        # Process through SVTR neck
        for block in self.svtr_neck:
            features = block(features, H, W)

        # Classification
        output = self.classifier(features)
        return output.permute(1, 0, 2)  # (T, B, C) for CTC

    def get_seq_length(self):
        return self.feature_h * self.feature_w



# 4. Training and Evaluation Functions
# ----------------------------------------------------------------------

def train_batch(model, criterion, optimizer, scaler, images, labels, device):
    model.train()
    images = images.to(device)
    optimizer.zero_grad(set_to_none=True) # Gradient to None for potential memory/speed benefits

    with torch.cuda.amp.autocast(enabled=True): # Mixed Precision Training
        outputs = model(images)

        input_lengths = torch.full(size=(images.size(0),), fill_value=outputs.size(0),
                                    dtype=torch.long, device=device)
        target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long, device=device)
        label_tensor = torch.cat(labels).to(device)

        loss = criterion(outputs.log_softmax(2), label_tensor, input_lengths, target_lengths)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) # Gradient Clipping
    scaler.step(optimizer)
    scaler.update()

    return loss.item()


def decode_predictions(outputs):
    # Greedy decoding (you can replace with beam search for better accuracy)
    _, max_indices = torch.max(outputs.softmax(2), 2)
    decoded_preds = []
    for pred in max_indices.T:
        chars = []
        prev_char = None
        for p in pred:
            p_idx = p.item()
            if p_idx != 0 and p_idx != prev_char:
                chars.append(IDX_TO_CHAR.get(p_idx, ''))
            prev_char = p_idx
        decoded_preds.append(''.join(chars))
    return decoded_preds


def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    return images, labels


def load_model(model_path, device, model_class=PlateOCRModel, model_config=None): # Load any model class
    model = model_class(NUM_CLASSES, **(model_config or {})).to(device) # Pass config if needed
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def predict_single_image(model, image_path, device, image_size=(64, 320)): # Image size as argument
    transform_pipeline = transforms.Compose([
        transforms.Resize(image_size, interpolation=Image.Resampling.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform_pipeline(image).unsqueeze(0).to(device) # To tensor and add batch dimension

    model.eval() # Set model to eval mode for inference
    with torch.no_grad():
        outputs = model(image_tensor)
        outputs = outputs.softmax(2)

        confidence, predictions = torch.max(outputs, dim=2)

        decoded_pred = []
        confidence_scores = []
        prev_char = None

        for pred, conf in zip(predictions.squeeze(), confidence.squeeze()):
            pred_idx = pred.item()
            if pred_idx != 0 and pred_idx != prev_char:
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



def evaluate_model(model, data_loader, criterion, device):
    model.eval() # Set to evaluation mode
    total_loss = 0
    correct_chars = 0
    total_chars = 0
    correct_plates = 0
    total_plates = 0
    results = []

    with torch.no_grad(): # Disable gradient calculation during evaluation
        for images, labels in tqdm(data_loader, desc='Evaluating'): # Use data_loader argument
            images = images.to(device)

            outputs = model(images)

            input_lengths = torch.full(size=(images.size(0),), fill_value=outputs.size(0),
                                        dtype=torch.long, device=device)
            target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long, device=device)
            label_tensor = torch.cat(labels).to(device)

            loss = criterion(outputs.log_softmax(2), label_tensor, input_lengths, target_lengths)
            total_loss += loss.item()

            decoded_preds = decode_predictions(outputs)

            for pred, true_label in zip(decoded_preds, labels):
                true_str = ''.join([IDX_TO_CHAR.get(idx.item(), '') for idx in true_label])

                if pred == true_str:
                    correct_plates += 1
                total_plates += 1

                min_len = min(len(pred), len(true_str)) # Character accuracy should consider shorter length
                for p_char, t_char in zip(pred[:min_len], true_str[:min_len]): # Iterate up to min_len
                    if p_char == t_char:
                        correct_chars += 1
                total_chars += len(true_str) # Total chars are based on true label length

                results.append({
                    'true': true_str,
                    'pred': pred,
                    'correct': pred == true_str
                })

    avg_loss = total_loss / len(data_loader)
    char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
    plate_accuracy = correct_plates / total_plates if total_plates > 0 else 0

    return {
        'avg_loss': avg_loss,
        'char_accuracy': char_accuracy,
        'plate_accuracy': plate_accuracy,
        'results': results
    }



# 5. Main Training Function
# ----------------------------------------------------------------------

def main():
    # --- Configuration ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    num_epochs = 60  # Increased epochs for potentially better convergence
    initial_learning_rate = 0.001
    min_learning_rate = 1e-6
    weight_decay = 1e-4
    image_size = (64, 320) # Define image size once

    config = { # Configuration dictionary for better organization
        'batch_size': batch_size,
        'epochs': num_epochs,
        'initial_lr': initial_learning_rate,
        'min_lr': min_learning_rate,
        'weight_decay': weight_decay,
        'image_size': image_size,
        'T_0_ CosineAnnealingWarmRestarts': 10, # Cycle length for Cosine Annealing scheduler
        'T_mult_CosineAnnealingWarmRestarts': 2
    }

    # --- Dataset and DataLoaders ---
    train_dataset = PlateDataset('E:/rec_derpet_modified/rec_derpet/train', image_size=config['image_size']) # Pass image_size
    val_dataset = PlateDataset('E:/rec_derpet_modified/rec_derpet/valid', is_training=False, image_size=config['image_size']) # Pass image_size
    test_dataset = PlateDataset('E:/rec_derpet_modified/rec_derpet/test', is_training=False, image_size=config['image_size']) # Pass image_size

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=0) # num_workers=4 or more for faster loading if using CPU
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=0)


    # --- Model, Loss, Optimizer, Scheduler, Scaler ---
    model = OptimizedPlateOCR_Speed(NUM_CLASSES).to(device) # Choose your model here (Speed, Accuracy, SVTR, or PlateOCRModel with configs)
    # model = OptimizedPlateOCR_Accuracy(NUM_CLASSES).to(device)
    # model = ImprovedPlateOCRSVTR(NUM_CLASSES).to(device)
    # model = PlateOCRModel(NUM_CLASSES, backbone_name='efficientnet_b0', pretrained=True, rnn_hidden_size=256).to(device) # Example PlateOCRModel with EfficientNet backbone

    criterion = nn.CTCLoss(zero_infinity=True, reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['initial_lr'], weight_decay=config['weight_decay'])
    scaler = torch.cuda.amp.GradScaler() # Mixed Precision Scaler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts( # Cosine Annealing LR Scheduler
        optimizer,
        T_0=config['T_0_ CosineAnnealingWarmRestarts'],
        T_mult=config['T_mult_CosineAnnealingWarmRestarts'],
        eta_min=config['min_lr']
    )

    best_val_loss = float('inf')
    best_val_acc = 0.0 # Initialize best val accuracy

    # --- Training Loop ---
    for epoch in range(config['epochs']):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}')

        for batch_idx, (images, labels) in enumerate(progress_bar):
            train_loss = train_batch(model, criterion, optimizer, scaler, images, labels, device) # Pass scaler
            total_loss += train_loss

            if batch_idx % 100 == 0: # Sample prediction less frequently
                sample_images = images[:1].to(device)
                model.eval() # Set to eval for prediction sample
                with torch.no_grad():
                    outputs = model(sample_images)
                    decoded = decode_predictions(outputs)
                    true_label_sample = ''.join([IDX_TO_CHAR.get(idx.item(), '') for idx in labels[0]]) # Decode true label for sample
                    print(f'\nSample Prediction - True: {true_label_sample}, Pred: {decoded[0]}')
                model.train() # Back to train mode

            progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1)})


        val_metrics = evaluate_model(model, val_loader, criterion, device) # Evaluate each epoch
        scheduler.step() # Step scheduler every epoch


        print(f'Epoch {epoch+1}/{config["epochs"]}:')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}') # Last batch loss is not representative, use avg_loss for epoch
        print(f'  Val Loss: {val_metrics["avg_loss"]:.4f}')
        print(f'  Val Char Accuracy: {val_metrics["char_accuracy"]:.4f}')
        print(f'  Val Plate Accuracy: {val_metrics["plate_accuracy"]:.4f}')

        avg_loss = total_loss / len(train_loader) # Calculate average train loss for epoch
        print(f'  Avg Train Loss: {avg_loss:.4f}')


        # --- Save Best Model (based on val loss and val accuracy) ---
        if val_metrics["avg_loss"] < best_val_loss:
            best_val_loss = val_metrics["avg_loss"]
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, f'best_loss_{best_val_loss:.4f}_plate_ocr_model.pth')
            print(f'  > Best val_loss improved. Saved best loss model: best_loss_{best_val_loss:.4f}_plate_ocr_model.pth')


        if val_metrics["plate_accuracy"] > best_val_acc: # Save based on plate accuracy as well
            best_val_acc = val_metrics["plate_accuracy"]
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc, # Save val_acc in checkpoint as well
            }, f'best_acc_plate_{best_val_acc:.4f}_ocr_model.pth')
            print(f'  > Best val_acc improved. Saved best accuracy model: best_acc_plate_{best_val_acc:.4f}_ocr_model.pth')


        if (epoch + 1) % 10 == 0: # Save checkpoint every 10 epochs
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'plate_ocr_epoch_{epoch+1}.pth')
            print(f'  > Saved checkpoint for epoch {epoch+1}: plate_ocr_epoch_{epoch+1}.pth')



if __name__ == '__main__':
    main()
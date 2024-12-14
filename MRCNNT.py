import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from collections import Counter
import torch.nn.functional as F
import math
import copy
from copy import deepcopy
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# -----------------------
# Dataset Definition
# -----------------------
class EEGSleepDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)  # (N, 3000, 1)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# -----------------------
# Focal Loss Definition
# -----------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)  # Probability of the true class
        pt = torch.clamp(pt, min=1e-5, max=1 - 1e-5)  # Avoid small/large probabilities

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()

# -----------------------
# SE and MRCNN Modules
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride
        

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class GELU(nn.Module):
    # for older versions of PyTorch.  For new versions you can use nn.GELU() instead.
    def __init__(self):
        super(GELU, self).__init__()
        
    def forward(self, x):
        x = torch.nn.functional.gelu(x)
        return x
        
        
class MRCNN(nn.Module):
    def __init__(self, afr_reduced_cnn_size):
        super(MRCNN, self).__init__()
        drate = 0.5
        self.GELU = GELU()  # for older versions of PyTorch.  For new versions use nn.GELU() instead.
        self.features1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2)
        )

        self.features2 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=400, stride=50, bias=False, padding=200),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.Conv1d(128, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.dropout = nn.Dropout(drate)
        self.inplanes = 128
        self.AFR = self._make_layer(SEBasicBlock, afr_reduced_cnn_size, 1)

    def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x)
        x_concat = torch.cat((x1, x2), dim=2)
        x_concat = self.dropout(x_concat)
        x_concat = self.AFR(x_concat)
        return x_concat

##########################################################################################

class AttnSleep(nn.Module):
    def __init__(self):
        super(AttnSleep, self).__init__()

        d_model = 80  # Embedding dimension for the transformer
        nhead = 4  # Number of attention heads
        num_layers = 2  # Number of transformer layers
        dim_feedforward = 120  # Feedforward network dimension
        dropout = 0.1
        num_classes = 5
        afr_reduced_cnn_size = 30

        # Feature Extraction
        self.mrcnn = MRCNN(afr_reduced_cnn_size)

        # Project MRCNN output to d_model
        self.feature_projection = nn.Linear(afr_reduced_cnn_size, d_model)

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers)

        # Alignment Layer (Reduce size from 6400 to 2400)
        self.align_layer = nn.Linear(6400, 2400)

        # Classification Layer
        self.fc = nn.Linear(2400, num_classes)

    def forward(self, x):
        # Feature Extraction
        x_feat = self.mrcnn(x)  # Shape: (batch_size, afr_reduced_cnn_size, seq_len)
       

        # Project features to d_model size
        x_feat = x_feat.permute(0, 2, 1)  # Shape: (batch_size, seq_len, afr_reduced_cnn_size)
        x_feat = self.feature_projection(x_feat)  # Shape: (batch_size, seq_len, d_model)
       

        # Prepare input for Transformer
        x_feat = x_feat.permute(1, 0, 2)  # Shape: (seq_len, batch_size, d_model)
       

        # Transformer Encoding
        encoded_features = self.transformer(x_feat)  # Shape: (seq_len, batch_size, d_model)
        

        # Flatten Features
        encoded_features = encoded_features.permute(1, 0, 2).contiguous()  # Shape: (batch_size, seq_len, d_model)
        encoded_features = encoded_features.view(encoded_features.shape[0], -1)  # Shape: (batch_size, seq_len * d_model)
        

        # Alignment Layer
        aligned_features = self.align_layer(encoded_features)  # Shape: (batch_size, 2400)
        

        # Classification
        logits = self.fc(aligned_features)  # Shape: (batch_size, num_classes)
        return logits






# -----------------------
# Training & Evaluation Functions
# -----------------------
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for X, Y in dataloader:
        # Move data to device
        X, Y = X.to(device), Y.to(device)

        # Permute the input to match (batch_size, channels, sequence_length)
        X = X.permute(0, 2, 1)

        # Forward pass
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, Y)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == Y).sum().item()
        total += X.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(Y.cpu().numpy())

    avg_loss = total_loss / total
    acc = correct / total
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, acc, f1

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, Y in dataloader:
            # Move data to device
            X, Y = X.to(device), Y.to(device)

            # Permute the input to match (batch_size, channels, sequence_length)
            X = X.permute(0, 2, 1)

            # Forward pass
            logits = model(X)
            loss = criterion(logits, Y)

            total_loss += loss.item() * X.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == Y).sum().item()
            total += X.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(Y.cpu().numpy())

    avg_loss = total_loss / total
    acc = correct / total
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, acc, f1, all_preds, all_labels


# -----------------------
# Data Loading
# -----------------------
def load_data_from_npz(folder_path):
    all_x = []
    all_y = []
    for fpath in glob.glob(os.path.join(folder_path, "*.npz")):
        data = np.load(fpath)
        X = data['x']  # Expecting shape (N, 3000, 1)
        Y = data['y']  # (N,)
        all_x.append(X)
        all_y.append(Y)
    X = np.concatenate(all_x, axis=0)
    Y = np.concatenate(all_y, axis=0)
    return X, Y

# -----------------------
# Main Training Script
# -----------------------
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load your data
    data_folder = "/home/amyn/scratch/AIProject/output"
    X, Y = load_data_from_npz(data_folder)

    # Calculate class weights
    class_counts = Counter(Y)
    total_samples = len(Y)
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
    class_weights_tensor = torch.tensor([class_weights[i] for i in range(len(class_counts))], dtype=torch.float32).to(device)

    print("Class Weights:", class_weights)

    # Split data into training and validation sets
    dataset = EEGSleepDataset(X, Y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Initialize model
    model = AttnSleep()
    model.to(device)

    # Loss, optimizer
    criterion = FocalLoss(alpha=class_weights_tensor, gamma=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    #optimizer = optim.RMSprop(
    #model.parameters(), 
    #lr=1e-3, 
    #alpha=0.99, 
    #eps=1e-8, 
    #weight_decay=1e-3, 
    #momentum=0.9
#)

    # Training parameters
    num_epochs = 50
    train_losses, train_accuracies, train_f1s = [], [], []
    val_losses, val_accuracies, val_f1s = [], [], []

    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch + 1}/{num_epochs}...")
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, val_preds, val_labels = eval_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        train_f1s.append(train_f1)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_f1s.append(val_f1)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

    # Plotting
    epochs = range(1, num_epochs + 1)

    # Plot Loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_plot_MRCNN_final_100epoch.png")
    plt.close()

    # Plot Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_plot_MRCNN_final_100epoch.png")
    plt.close()

    # Plot F1 Score
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_f1s, label="Train F1 Score")
    plt.plot(epochs, val_f1s, label="Validation F1 Score")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.title("Training and Validation F1 Score")
    plt.legend()
    plt.grid(True)
    plt.savefig("f1_plot_MRCNN_final_100epoch.png")
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(val_labels, val_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix_MRCNN_final_100epoch.png")
    plt.close()

    # Classification Report
    report = classification_report(val_labels, val_preds, target_names=[f"Class {i}" for i in range(5)])
    print(report)

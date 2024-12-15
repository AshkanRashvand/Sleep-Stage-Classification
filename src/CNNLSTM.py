# -----------------------
# Import Libraries
# -----------------------
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

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
        pt = torch.clamp(pt, min=1e-7, max=1 - 1e-7)  # Avoid small/large probabilities

        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()

# -----------------------
# CNN + LSTM Model Definition
# -----------------------
class CNNSleepLSTMModel(nn.Module):
    def __init__(self, d_model=128, num_classes=5, lstm_layers=2, lstm_bidirectional=True, dropout=0.3):
        super(CNNSleepLSTMModel, self).__init__()

        # Updated CNN layers
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # Project CNN output (128 channels) to d_model
        self.input_proj = nn.Linear(128, d_model)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=lstm_layers,
            batch_first=False,
            dropout=dropout,
            bidirectional=lstm_bidirectional
        )

        # Adjusting for bidirectional LSTM
        lstm_output_dim = d_model * (2 if lstm_bidirectional else 1)

        # Classification head
        self.fc = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, 1, T)
        x = self.cnn(x)  # (B, 128, T_reduced)
        x = x.permute(2, 0, 1)  # (T_reduced, B, 128)
        x = self.input_proj(x)  # (T_reduced, B, d_model)
        lstm_out, (h_n, c_n) = self.lstm(x)
        final_h = h_n[-1, :, :] if not self.lstm.bidirectional else torch.cat((h_n[-2], h_n[-1]), dim=1)  # (B, d_model * directions)
        logits = self.fc(final_h)  # (B, num_classes)
        return logits

# -----------------------
# Training & Evaluation Functions
# -----------------------
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []

    for X, Y in dataloader:
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, Y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        all_preds.append(logits.argmax(dim=1))
        all_targets.append(Y)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    avg_loss = total_loss / len(dataloader.dataset)
    acc = (all_preds == all_targets).float().mean().item()
    f1 = f1_score(all_targets.cpu().numpy(), all_preds.cpu().numpy(), average='macro')
    return avg_loss, acc, f1

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)
            logits = model(X)
            loss = criterion(logits, Y)
            total_loss += loss.item() * X.size(0)
            all_preds.append(logits.argmax(dim=1))
            all_targets.append(Y)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    avg_loss = total_loss / len(dataloader.dataset)
    acc = (all_preds == all_targets).float().mean().item()
    f1 = f1_score(all_targets.cpu().numpy(), all_preds.cpu().numpy(), average='macro')
    return avg_loss, acc, f1, all_preds, all_targets

# -----------------------
# Data Loading
# -----------------------
def load_data_from_npz(folder_path):
    all_x = []
    all_y = []
    for fpath in glob.glob(os.path.join(folder_path, "*.npz")):
        data = np.load(fpath)
        X = data['x']
        Y = data['y']
        all_x.append(X)
        all_y.append(Y)
    X = np.concatenate(all_x, axis=0)
    Y = np.concatenate(all_y, axis=0)
    return X, Y

# -----------------------
# Main Training Script
# -----------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data_folder = "/home/amyn/scratch/AIProject/output"
    X, Y = load_data_from_npz(data_folder)

    # Split into train and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Calculate class weights
    class_counts = np.bincount(Y_train)
    total_samples = len(Y_train)
    class_weights = {cls: total_samples / count for cls, count in enumerate(class_counts)}
    class_weights_tensor = torch.tensor([class_weights[i] for i in range(len(class_counts))], dtype=torch.float32).to(device)

    # Create datasets and dataloaders
    train_dataset = EEGSleepDataset(X_train, Y_train)
    val_dataset = EEGSleepDataset(X_val, Y_val)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Initialize model
    model = CNNSleepLSTMModel(d_model=128, num_classes=5, lstm_layers=2, lstm_bidirectional=True, dropout=0.3)
    model.to(device)

    # Define loss and optimizer
    criterion = FocalLoss(alpha=class_weights_tensor, gamma=2)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_f1_scores, val_f1_scores = [], []

    # Training loop
    for epoch in range(50):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, val_preds, val_targets = eval_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        train_f1_scores.append(train_f1)
        val_f1_scores.append(val_f1)

        print(f"Epoch [{epoch + 1}/50] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

    # Save and plot results
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_plot_CNNLFW.png")
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("accuracy_plot_CNNLFW.png")
    plt.close()

    # F1 Score
    plt.figure()
    plt.plot(train_f1_scores, label="Train F1")
    plt.plot(val_f1_scores, label="Validation F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.savefig("f1_score_plot_CNNLFW.png")
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(val_targets.cpu().numpy(), val_preds.cpu().numpy())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Wake", "N1", "N2", "N3", "REM"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix_CNNLFW.png")
    plt.close()

    # Classification Report
    report = classification_report(val_targets.cpu().numpy(), val_preds.cpu().numpy(), target_names=["Wake", "N1", "N2", "N3", "REM"])
    print(report)

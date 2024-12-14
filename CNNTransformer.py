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
# Model Definition
# -----------------------
class SleepCNNTransformer(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_layers=1, num_classes=5, dropout=0.5):
        super(SleepCNNTransformer, self).__init__()

        # CNN layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: (B, T, 1)
        x = x.permute(0, 2, 1)  # Change to (B, 1, T) for CNN
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        # Flatten for Transformer
        x = x.permute(2, 0, 1)  # (T, B, d_model)
        x = self.transformer_encoder(x)

        # Pooling (mean over time)
        x = torch.mean(x, dim=0)  # (B, d_model)
        logits = self.fc(x)  # (B, num_classes)
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
        X, Y = X.to(device), Y.to(device)
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
            X, Y = X.to(device), Y.to(device)
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
    model = SleepCNNTransformer(d_model=32, nhead=4, num_layers=1, num_classes=5)
    model.to(device)

    # Loss, optimizer
    criterion = FocalLoss(alpha=class_weights_tensor, gamma=2)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

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
    plt.savefig("loss_plot_CTFWb64.png")
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
    plt.savefig("accuracy_plot_CTFWb64.png")
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
    plt.savefig("f1_plot_CTFWb64.png")
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(val_labels, val_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix_CTFWb64.png")
    plt.close()

    # Classification Report
    report = classification_report(val_labels, val_preds, target_names=[f"Class {i}" for i in range(5)])
    print(report)

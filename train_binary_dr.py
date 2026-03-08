import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from dataset import IDRiDDataset, get_transforms
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import numpy as np

TRAIN_CSV = "data/train/train_labels.csv"
TEST_CSV = "data/test/test_labels.csv"
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4

print("Using device:", DEVICE)

class BinaryDRDataset(IDRiDDataset):
    def __getitem__(self, idx):
        image, dr_label, _ = super().__getitem__(idx)
        binary_label = 0 if dr_label == 0 else 1
        return image, binary_label

train_dataset = BinaryDRDataset(
    TRAIN_CSV, TRAIN_DIR, transform=get_transforms(train=True)
)

test_dataset = BinaryDRDataset(
    TEST_CSV, TEST_DIR, transform=get_transforms(train=False)
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=2, pin_memory=True
)

test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=2, pin_memory=True
)

# EfficientNet-B4 (Stronger)
model = timm.create_model("efficientnet_b4", pretrained=True)
model.classifier = nn.Linear(model.num_features, 2)
model = model.to(DEVICE)

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        logp = self.ce(inputs, targets)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss

criterion = FocalLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler()

best_auc = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

    # Validation
    model.eval()
    preds, true, probs = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)

            probabilities = torch.softmax(outputs, dim=1)[:,1]
            predicted = torch.argmax(outputs, dim=1)

            preds.extend(predicted.cpu().numpy())
            true.extend(labels.numpy())
            probs.extend(probabilities.cpu().numpy())

    auc = roc_auc_score(true, probs)
    print("Validation AUC:", auc)

    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), "best_binary_dr_model.pth")
        print("🔥 Best model saved!")

print("Training Complete. Best AUC:", best_auc)

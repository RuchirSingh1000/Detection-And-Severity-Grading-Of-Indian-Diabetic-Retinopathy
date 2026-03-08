import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from dataset import IDRiDDataset, get_transforms
from sklearn.metrics import cohen_kappa_score
import pandas as pd

TRAIN_CSV = "data/train/train_labels.csv"
TEST_CSV = "data/test/test_labels.csv"
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Strong Augmentations
# -----------------------------
def get_strong_transforms(train=True):
    import torchvision.transforms as transforms
    if train:
        return transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.4, contrast=0.4),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

train_dataset = IDRiDDataset(TRAIN_CSV, TRAIN_DIR, transform=get_strong_transforms(True))
test_dataset = IDRiDDataset(TEST_CSV, TEST_DIR, transform=get_strong_transforms(False))

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

# -----------------------------
# Balanced Backbone (B2 is sweet spot)
# -----------------------------
base_model = timm.create_model("efficientnet_b2", pretrained=True)
base_model.classifier = nn.Identity()

num_features = base_model.num_features

class MultiTaskModel(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.dr_head = nn.Linear(num_features, 5)
        self.dme_head = nn.Linear(num_features, 3)

    def forward(self, x):
        features = self.base(x)
        return self.dr_head(features), self.dme_head(features)

model = MultiTaskModel(base_model).to(DEVICE)

# -----------------------------
# Label Smoothing Loss
# -----------------------------
criterion_dr = nn.CrossEntropyLoss(label_smoothing=0.1)
criterion_dme = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

scaler = torch.amp.GradScaler("cuda")

EPOCHS = 15
best_dr_qwk = 0
best_dme_qwk = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, dr_labels, dme_labels in train_loader:
        images = images.to(DEVICE)
        dr_labels = dr_labels.to(DEVICE)
        dme_labels = dme_labels.to(DEVICE)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            dr_out, dme_out = model(images)

            loss_dr = criterion_dr(dr_out, dr_labels)
            loss_dme = criterion_dme(dme_out, dme_labels)

            loss = loss_dr + loss_dme  # balanced

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    scheduler.step()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # -----------------------------
    # Evaluation
    # -----------------------------
    model.eval()
    dr_preds, dr_true = [], []
    dme_preds, dme_true = [], []

    with torch.no_grad():
        for images, dr_labels, dme_labels in test_loader:
            images = images.to(DEVICE)

            dr_out, dme_out = model(images)

            dr_pred = torch.argmax(dr_out, dim=1).cpu().numpy()
            dme_pred = torch.argmax(dme_out, dim=1).cpu().numpy()

            dr_preds.extend(dr_pred)
            dr_true.extend(dr_labels.numpy())

            dme_preds.extend(dme_pred)
            dme_true.extend(dme_labels.numpy())

    dr_qwk = cohen_kappa_score(dr_true, dr_preds, weights="quadratic")
    dme_qwk = cohen_kappa_score(dme_true, dme_preds, weights="quadratic")

    print("DR QWK:", dr_qwk)
    print("DME QWK:", dme_qwk)

    if dr_qwk > best_dr_qwk and dme_qwk > best_dme_qwk:
        best_dr_qwk = dr_qwk
        best_dme_qwk = dme_qwk
        torch.save(model.state_dict(), "best_grading_model.pth")
        print("Best balanced model saved!")

print("\nBest DR QWK:", best_dr_qwk)
print("Best DME QWK:", best_dme_qwk)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from dataset import IDRiDDataset, get_transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
TEST_CSV = "data/test/test_labels.csv"
TEST_DIR = "data/test"

# Dataset
test_dataset = IDRiDDataset(
    TEST_CSV,
    TEST_DIR,
    transform=get_transforms(train=False)
)

test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Model definition (same as training)
base_model = timm.create_model("efficientnet_b0", pretrained=False)
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
model.load_state_dict(torch.load("grading_model.pth", map_location=DEVICE))
model.eval()

dr_preds, dr_true = [], []

with torch.no_grad():
    for images, dr_labels, _ in test_loader:
        images = images.to(DEVICE)
        dr_out, _ = model(images)

        dr_pred = torch.argmax(dr_out, dim=1).cpu().numpy()
        dr_preds.extend(dr_pred)
        dr_true.extend(dr_labels.numpy())

cm = confusion_matrix(dr_true, dr_preds)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("DR Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

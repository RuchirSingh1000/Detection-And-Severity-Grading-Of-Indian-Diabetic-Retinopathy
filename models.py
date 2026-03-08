import torch.nn as nn
import timm

class DRDMEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = timm.create_model("efficientnet_b3", pretrained=True)
        in_features = self.base.classifier.in_features
        self.base.classifier = nn.Identity()

        self.dr_head = nn.Linear(in_features, 5)
        self.dme_head = nn.Linear(in_features, 3)

    def forward(self, x):
        features = self.base(x)
        dr_out = self.dr_head(features)
        dme_out = self.dme_head(features)
        return dr_out, dme_out

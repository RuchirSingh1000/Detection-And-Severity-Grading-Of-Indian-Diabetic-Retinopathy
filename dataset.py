import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class IDRiDDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    # Circular Crop
    def crop_image(self, image):
        image = np.array(image)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            image = image[y:y+h, x:x+w]

        return image

    # Ben Graham Enhancement (optimized sigma)
    def ben_graham_preprocess(self, image):
        blur = cv2.GaussianBlur(image, (0, 0), 10)
        enhanced = cv2.addWeighted(image, 4, blur, -4, 128)
        return enhanced

    # CLAHE
    def apply_clahe(self, image):
        green_channel = image[:, :, 1]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_green = clahe.apply(green_channel)
        image[:, :, 1] = enhanced_green
        return image

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0] + ".jpg"
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        # Resize early for efficiency
        image = image.resize((512, 512))
        image = np.array(image)

        image = self.crop_image(Image.fromarray(image))
        image = self.ben_graham_preprocess(image)
        image = self.apply_clahe(image)

        image = Image.fromarray(image)

        dr_label = int(self.data.iloc[idx, 1])
        dme_label = int(self.data.iloc[idx, 2])

        if self.transform:
            image = self.transform(image)

        return image, dr_label, dme_label


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
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

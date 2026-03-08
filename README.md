# Detection-And-Severity-Grading-Of-Indian-Diabetic-Retinopathy
Automated detection and grading of Diabetic Retinopathy (DR) and Diabetic Macular Edema (DME) using EfficientNet-based deep learning models on the IDRiD retinal fundus image dataset with advanced medical image preprocessing.
# Automated Detection and Grading of Diabetic Retinopathy and Diabetic Macular Edema

## Overview

Diabetic Retinopathy (DR) is one of the leading causes of vision impairment among diabetic patients worldwide. Early detection through retinal fundus imaging can significantly reduce the risk of blindness. However, manual screening by ophthalmologists is time-consuming and resource-intensive.

This project develops a deep learning-based automated system for detecting and grading Diabetic Retinopathy (DR) and Diabetic Macular Edema (DME) using retinal fundus images from the **IDRiD (Indian Diabetic Retinopathy Image Dataset)**.

The system uses transfer learning with EfficientNet architecture and incorporates medical-image-specific preprocessing techniques to enhance retinal features and improve classification performance.

---

## Key Features

- Binary classification for **DR detection (DR vs No DR)**
- Multi-class classification for **DR severity grading (0–4)**
- Multi-task learning for **simultaneous DR and DME prediction**
- Advanced medical image preprocessing pipeline
- Transfer learning using **EfficientNet**
- GPU optimized training
- Multiple evaluation metrics including **Accuracy, AUC, Sensitivity, Specificity, and Quadratic Weighted Kappa (QWK)**

---

## Dataset

This project uses the **IDRiD (Indian Diabetic Retinopathy Image Dataset)**.

Dataset includes:

- Retinal fundus images
- DR severity grading (5 classes)
- DME grading (3 classes)
- Lesion-level annotations

Dataset size:

- Training Images: 413
- Testing Images: 103

Source:
https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid

---

## Preprocessing Pipeline

Medical images require specialized preprocessing to highlight pathological features. The following techniques are applied:

### Circular Cropping
Removes black background borders from retinal fundus images to focus on the retinal region.

### Ben Graham Enhancement
Enhances vascular structures using Gaussian blur subtraction.

### CLAHE (Contrast Limited Adaptive Histogram Equalization)
Improves contrast of retinal vessels and lesions by applying CLAHE on the green channel.

### Data Augmentation
To improve generalization:

- Horizontal flipping
- Random rotation
- Color jitter (brightness and contrast)

### Normalization
Images are normalized using ImageNet statistics to match pretrained EfficientNet requirements.

---

## Model Architecture

The system uses **EfficientNet-B3** as the backbone network with transfer learning.

### Binary DR Detection Model

Input: Retinal fundus image  
Output:  
- Class 0 → No DR  
- Class 1 → DR Present

### Multi-Task DR + DME Model

The network has two classification heads:

- DR Head (5 classes)
- DME Head (3 classes)

This allows simultaneous prediction of DR severity and DME grade.

---

## Training Details

Model: EfficientNet-B3  
Input Resolution: 300 × 300  
Optimizer: Adam  
Loss Function: CrossEntropyLoss with class weighting  
Epochs: 20  
Batch Size: 16  

Training is performed using **GPU acceleration** for efficient learning.

---

## Evaluation Metrics

### Binary DR Detection
- Accuracy
- AUC (Area Under ROC Curve)
- Sensitivity (Recall)
- Specificity
- Confusion Matrix

### DR Severity Grading
- Accuracy
- Quadratic Weighted Kappa (QWK)

QWK is particularly important because DR grading is an **ordinal classification problem**, where prediction errors should be weighted based on severity differences.

---

## Project Structure

# Plant Disease Classification Using a Custom CNN

**Course:** Computer Vision and Pattern Recognition  
**Student ID:** 23-50578-1  
**Framework:** Python, PyTorch, Google Colab  
**Dataset:** [PlantVillage (Kaggle)](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

## Project Overview

This project implements a **custom Convolutional Neural Network (CNN)** in PyTorch to classify color images of plant leaves into **38 healthy and diseased categories**. The notebook covers dataset exploration, preprocessing, data augmentation, CNN design, training, validation, testing, and performance analysis.

## Dataset

- **Total images:** 54,305
- **Classes:** 38
- **Input size:** 128 × 128 RGB
- **Split:** Stratified, image-level train/validation/test split (70% / 15% / 15%; random seed 42)

| Subset | Number of images |
|---|---:|
| Training | 38,013 |
| Validation | 8,146 |
| Testing | 8,146 |

### Preprocessing and Augmentation

- Resize images to 128 × 128.
- Convert images to tensors and normalize with mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.
- **Training only:** random horizontal flip and random rotation (±15°).
- Validation and test images use deterministic preprocessing without augmentation.

## Custom CNN Architecture

The CNN is built from scratch and contains:

1. Convolution (3 → 32 channels), ReLU, max pooling
2. Convolution (32 → 64 channels), ReLU, max pooling
3. Convolution (64 → 128 channels), ReLU, max pooling
4. Flatten → fully connected layer (256 units) → ReLU → dropout (0.5)
5. Output fully connected layer (38 class logits)

**Trainable parameters:** 8,491,878

### Training Configuration

| Setting | Value |
|---|---|
| Loss function | CrossEntropyLoss |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 32 |
| Epochs | 5 |
| Best checkpoint selection | Highest validation accuracy |

## Results

The values below are from the **latest recorded training and evaluation run** in the project notebook.

| Metric | Result |
|---|---:|
| 
| Test accuracy | **95.00%** |
| Macro precision | **94.15%** |
| Macro recall | **92.41%** |
| Macro F1-score | **92.80%** |

### Class-wise Findings

- The highest class-level F1-scores include **Strawberry healthy** and **Squash Powdery Mildew** (approximately 99.27% each).
- **Corn Gray Leaf Spot** was challenging (F1-score approximately 57.89%, recall approximately 42.86%).
- The largest directional confusion was **Corn Gray Leaf Spot → Corn Northern Leaf Blight**, with **43** test images.

A high overall accuracy does not imply equally strong performance across all classes.

## Repository Files

```text
Plant-Disease-Classification-CNN/
├── CNN_23_50578_1.ipynb       # Complete project notebook (9 required sections)
├── README.md                  # Project documentation
├── accuracy_curve.png         # Training and validation accuracy
├── loss_curve.png             # Training and validation loss
├── confusion_matrix.png       # Confusion matrix for 38 classes
└── classification_report.csv  # Per-class evaluation metrics
```

The trained checkpoint (`best_plant_disease_cnn.pth`) is stored separately because of the assignment's model-file size guidance. Dataset images and private Kaggle credentials are **not** included in this repository.

## How to Run

1. Open `CNN_23_50578_1.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Select a GPU runtime if available: **Runtime → Change runtime type → GPU**.
3. Run the setup/import cells. You may need to supply your own private `kaggle.json` API credentials to download the dataset.
4. Run the cells in order to download and explore PlantVillage, create the train/validation/test split, and train the CNN.
5. Run the test-evaluation and visualization cells to reproduce metrics and graphs.

**Note:** Retraining may give slightly different results depending on random state, GPU environment, and library versions. The results reported above are the recorded results in the submitted notebook.

## Notebook Sections

1. Import Libraries
2. Load and Explore Dataset
3. Data Preprocessing & Augmentation
4. Define CNN Architecture
5. Training Loop with Validation
6. Evaluate Model on Test Set
7. Visualizations (Loss/Accuracy Curves)
8. Analysis & Discussion of Results
9. Conclusions & Future Work

## Limitations and Future Work

- The dataset is **imbalanced** across classes, and some diseases have considerably lower recall.
- An image-level split may include closely related images of the same plant/leaf in different subsets; a grouped split would provide a stronger generalization test.
- PlantVillage images may not represent real field conditions; evaluation on independent field images is needed.
- Future improvements include analyzing misclassified images, class-balanced training, tuning augmentation and hyperparameters, and comparing against other CNN architectures.

# 👁️ Human Age Prediction Using Deep Learning

This project predicts a person's age from facial images using convolutional neural networks and transfer learning.  
The goal is to estimate a continuous age value from a single face image.

The dataset originates from the ChaLearn Looking at People challenge  
(https://chalearnlap.cvc.uab.es/dataset/26/description/) and was adapted for the TripleTen assignment.

This repository focuses on two goals:

1. Complete clear EDA and analysis of the age labels and image samples.  
2. Build a modular deep learning pipeline using a `src/` folder for data loading, modeling, training, and evaluation.

The final model uses **ResNet50** as its backbone for age regression.

[![View Notebook](https://img.shields.io/badge/View%20Notebook-.ipynb-blue?style=for-the-badge&logo=jupyter&logoColor=white)](./notebooks/Human_Age_Prediction.ipynb)

---

## Dataset

The dataset consists of facial images with corresponding age labels.

Original assignment paths:

- Images: `/datasets/faces/final_files/`
- Labels: `/datasets/faces/labels.csv`

Local project layout:

- `data/labels.csv`  
- `data/final_files/`  

Each record includes:

- `file_name` – image filename  
- `real_age` – integer age in years  

> **Note:** The `final_files` image folder is **not** included in this repository because of size limits.  
> Download the original data from the ChaLearn website, then place it in: `data/final_files`

---

## Methodology

### 1. Exploratory Data Analysis (EDA)
The project begins with a detailed examination of the dataset:

- Inspect overall dataset size and label structure.
- Visualize the distribution of ages and age groups by decade.
- Identify outliers and class imbalance.
- Display sample facial images across various age ranges.
- Discuss potential challenges such as lighting variation, pose differences, and underrepresented age groups.

### 2. Data Loading and Preprocessing
A modular data pipeline handles all image preparation:

- Uses `ImageDataGenerator.flow_from_dataframe` with `labels.csv` + `final_files/`.
- Rescales images and applies light augmentation (rotation, zoom, horizontal flip).
- Splits the dataset into training and validation subsets using a configurable validation split.
- Outputs TensorFlow-compatible batches for efficient GPU training.

All loading logic is implemented in: `src/data_loader.py`

### 3. Model Architecture
The model uses **transfer learning** based on ResNet50:

- Base model: `ResNet50(weights="imagenet", include_top=False)`
- Global Average Pooling layer for feature aggregation.
- Dense regression head for continuous age prediction.

Compiled with:

- **Loss:** Mean Squared Error (MSE)
- **Metric:** Mean Absolute Error (MAE)
- **Optimizer:** Adam with a small learning rate

Architecture built in: `src/model_builder.py`

### 4. Training Setup
Training is managed with:

- **ModelCheckpoint**: saves the best model based on validation MAE.
- **EarlyStopping**: restores best weights after patience is exceeded.
- **Limited steps per epoch for local quick runs**, enabling lightweight functional testing.
- **Full benchmark training performed on GPU** (Google Colab) for proper convergence.

Training loop implemented in: `src/train.py`

### 5. Evaluation
Evaluation includes:

- Plotting **training vs. validation loss** over epochs.
- Plotting **training vs. validation MAE** over epochs.
- Comparing quick-run performance to the GPU benchmark results.
- Discussing reasons for MAE variability, such as:
  - age imbalance,
  - dataset diversity,
  - real-world facial variation.

Additional utilities in: `src/evaluate.py` 

---

## Key Insights

- Transfer learning significantly improves stability and convergence.  
- Age prediction accuracy varies greatly with lighting, pose, and expression.  
- Mid-range ages (20–50) yield the lowest prediction error.  
- Data augmentation helps generalization by broadening visual diversity.  

---

## Key Results

### Full GPU Benchmark (20 Epochs)

- Validation MAE stabilizes around **7–8 years**  
- Training MAE decreases steadily across epochs  
- Validation volatility reflects dataset imbalance (fewer young and elderly samples)

Most reliable predictions occur for ages **20–50**, where the dataset is most dense.

---

### Local Quick Run (3-Epoch Demo)

- Training MAE improved: **29 → 21 → 18**  
- Validation MAE improved: **23 → 16 → 13**  
- Checkpoints and early stopping worked as intended  
- Not expected to match GPU performance due to limited steps

The quick run verifies that the entire pipeline functions correctly from end to end.

---

## Final Conclusions

### 1. Can this model help the customer?

Yes, for use cases requiring **approximate age ranges**, such as:

- demographic analytics  
- audience segmentation  
- large-scale reporting  
- broad category classification (child, teen, adult, senior)  

However, MAE of **7–8 years** is not sufficient for strict identity verification or regulatory use cases.

---

### 2. Other Practical Applications

- **Ad Targeting:** approximate age-based personalization  
- **Content Moderation:** age gating or restricted content checks  
- **Retail Analytics:** in-store demographic estimation  
- **Hybrid Authentication:** supportive signal in multi-factor systems  

---
## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

---

## Author
**Rhiannon Fillingham** 

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rhiannonfilli)

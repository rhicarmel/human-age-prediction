# 👁️ Human Age Prediction Using Deep Learning

## Overview
This project develops a deep learning model to estimate human age from facial images.  
The task is framed as a **regression problem**, where the model predicts a continuous value (age) instead of a class label.

The goal is to build a reliable age-prediction pipeline using modern computer vision techniques, transfer learning, and structured experimentation.
**Best Model:** 🏆 ResNet50 (MAE ≈ 6.8 years on validation data)

🔗 [View the full notebook here](./[updated]HumanAge(ComputerVision).ipynb)

---

## Project Functionality

### 1. Data Loading and Preprocessing
The dataset includes facial images paired with age labels.  
Key steps:
- Parse directory structure and metadata
- Normalize pixel values
- Resize images for model compatibility
- Split data into training, validation, and test sets
- Use `ImageDataGenerator` for augmentation (rotation, flip, zoom)

### 2. Exploratory Data Analysis
The notebook includes:
- Age distribution histogram
- Random sample of face images with labels
- Basic feature review and dataset sanity checks

### 3. Model Architecture
The final model uses **ResNet50** as a pretrained backbone:
- Frozen convolutional layers
- `GlobalAveragePooling2D`
- Dense hidden layers with dropout
- Final dense output layer with linear activation for regression

This approach leverages transfer learning to improve accuracy with a relatively small dataset.

### 4. Training Workflow
- Loss: Mean Absolute Error (MAE)
- Optimizer: Adam
- Metrics: MAE, MSE
- Early stopping to reduce overfitting
- Checkpointing best-performing model

### 5. Evaluation
Model performance is measured on a held-out test set with:
- Final MAE score  
- Real vs predicted age comparisons  
- Scatterplot and residual analysis  

Example results are included in the notebook.

---

## Key Insights
- Transfer learning significantly improves model convergence and stability.
- Age prediction is sensitive to lighting, pose, and facial expression.
- The model performs best on mid-range ages, with slightly higher error for children and elderly subjects.
- Data augmentation improves generalization by broadening the face representations seen during training.

---

## Results
**Best Model:** ResNet50-based Regression Model  
**Final Performance:**  
- **MAE:** ~5.0–7.0 years (varies by dataset and split)
- **MSE:** Included in notebook  
- **Loss Curve:** Smooth convergence with early stopping  
- **Prediction Samples:** Displays true vs predicted ages for random test images

---

## Tech Stack
- Python  
- TensorFlow / Keras  
- NumPy, Pandas  
- Matplotlib, Seaborn  
- Jupyter Notebook  

---

## Author
Rhi Carmel

📎 [LinkedIn](www.linkedin.com/in/rhiannonfilli)

# 🧑‍💻 HUMAN AGE PREDICTION (COMPUTER VISION)

## Overview
A deep learning project that estimates a person’s **age from facial images** using convolutional neural networks and transfer learning.  
This regression task was built to explore how visual features correlate with age, using **ResNet50** for feature extraction and fine-tuning.

**Goal:** Predict human age with low Mean Absolute Error (MAE).  
**Best Model:** 🏆 ResNet50 (MAE ≈ 6.8 years on validation data)

🔗 [View the full notebook here](./[updated]HumanAge(ComputerVision).ipynb)

---

## Functionality
- Loads and preprocesses face image dataset.  
- Applies data augmentation to improve generalization.  
- Builds and fine-tunes a CNN model using **ResNet50**.  
- Trains the model with MSE loss and tracks **MAE** performance.  
- Evaluates predicted vs. actual ages on validation data.

---

## Key Insights
- **Transfer learning** significantly reduces training time and improves accuracy.  
- Model performs best on age ranges **20–60**, with slightly higher error for older groups.  
- **Data imbalance** (fewer elderly faces) affects model precision at extreme ages.

---

## Results
| Model | MAE (Validation) | Notes |
|--------|------------------|-------|
| Baseline CNN | ~13.2 | High bias, underfitting |
| ResNet50 (Pretrained) | ~6.8 | Excellent generalization |

---

## Tech Stack
**Python**, TensorFlow/Keras, OpenCV, NumPy, Matplotlib, Seaborn  
*Developed in Jupyter Notebook*

---

## Installing
```bash
# Clone repo
git clone https://github.com/rhicarmel/human-age-prediction.git
cd human-age-prediction

# Install dependencies
pip install -r requirements.txt

# Open notebook
jupyter notebook [updated]HumanAge(ComputerVision).ipynb
```

---

## Future Improvements
- Incorporate age group classification alongside regression.
- Add face detection preprocessing for raw photo input.
- Experiment with EfficientNet or Vision Transformers (ViT) for improved accuracy.

---

## Author
Rhi Carmel

📎 [LinkedIn](www.linkedin.com/in/rhiannonfilli)

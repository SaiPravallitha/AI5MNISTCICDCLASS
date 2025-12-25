# MNIST CI/CD Machine Learning Pipeline

![ML Pipeline Status](https://github.com/SaiPravallitha/AI5MNISTCICDCLASS/actions/workflows/pipeline.yml/badge.svg)

## 🎯 Project Overview
This project implements a lightweight 3-layer Convolutional Neural Network (CNN) trained on the MNIST dataset. It features a full CI/CD pipeline using GitHub Actions to enforce strict model constraints and quality standards.

## 🚀 Constraints & Requirements
- **Parameters:** Must be less than **25,000**.
- **Accuracy:** Must achieve **95% or higher** within just **1 Epoch**.
- **Input Size:** 28x28 grayscale images.
- **Output:** 10 classes (digits 0-9).

## 🛠️ Advanced Features
- **Data Augmentation:** Uses random rotations and affine transforms to improve model robustness.
- **Automated Validation:** Every push triggers tests for:
  1. Parameter count check.
  2. Input/Output tensor shape validation.
  3. Accuracy threshold verification.
  4. Output range (NaN/Inf) checks.
  5. Batch size invariance.

## 💻 Local Setup
To run this project locally, follow these steps:

1. **Clone the repo:**
   ```bash
   git clone [https://github.com/SaiPravallitha/AI5MNISTCICDCLASS.git](https://github.com/SaiPravallitha/AI5MNISTCICDCLASS.git)
   cd AI5MNISTCICDCLASS

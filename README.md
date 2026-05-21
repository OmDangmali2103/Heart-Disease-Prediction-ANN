# ❤️ Heart Disease Prediction System using ANN

An interactive, end-to-end Machine Learning web application that leverages an **Artificial Neural Network (ANN)** to predict the likelihood of heart disease in patients based on clinical parameters. 

Built with **TensorFlow/Keras** for deep learning and **Streamlit** for a smooth, user-friendly frontend interface.

---

## 🚀 Live Features
- **Deep Learning Core:** A sequential Artificial Neural Network trained on clinical patient features.
- **Interactive UI:** Dynamic numeric fields and drop-downs matching medical indicators.
- **Real-Time Inference:** Instantly scales patient data using a pre-trained `StandardScaler` pipeline and processes it through the ANN weights.

---

## 🛠️ Tech Stack & Architecture

- **Frontend Interface:** Streamlit
- **Deep Learning Framework:** TensorFlow 2.x & Keras
- **Data Engineering & Scaling:** Scikit-Learn (`StandardScaler`)
- **Data Manipulation:** Pandas, NumPy
- **Model Storage:** Serialization via Native Keras (`.keras`) & Legacy HDF5 (`.h5`) formats, Scaler pipeline via `joblib`.

### 🧠 Neural Network Structural Layout
1. **Input Layer:** Accepts 13 foundational patient clinical attributes.
2. **Hidden Layer 1:** Dense Layer (16 Neurons) with `ReLU` activation for learning complex feature combinations.
3. **Hidden Layer 2:** Dense Layer (8 Neurons) with `ReLU` activation.
4. **Output Layer:** Dense Layer (1 Neuron) with `Sigmoid` activation function, outputting a clear probability metric mapping to binary classification (0: Healthy, 1: At Risk).

---

## 📁 Repository Directory Structure

```text
├── app.py                 # Streamlit web application script 
├── experiment.ipynb       # Jupyter Notebook detailing EDA, ANN training, and evaluation
├── heart.csv              # Patient clinical dataset utilized for training/testing
├── heart_model.keras      # Production-ready trained ANN weights (Native Keras format)
├── heart_model.h5         # Legacy backup of trained ANN weights (HDF5 format)
├── scaler.pkl             # Serialized StandardScaler instance used for normalization
└── requirements.txt       # Directives for installing dependencies

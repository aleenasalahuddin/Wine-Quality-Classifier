# Wine Quality Prediction Web App

An end-to-end **machine learning web application** that predicts wine quality based on physicochemical properties using a neural network model. The app is built with **TensorFlow**, **Scikit-Learn**, and **Streamlit**, and deployed for real-time user interaction.

---

## Project Overview

This project predicts whether a wine is of **high quality** or **low quality** using its chemical characteristics such as acidity, sugar content, density, sulphates, and alcohol percentage.  
A neural network classifier is trained on the **UCI Wine Quality (White Wine) dataset** and deployed using Streamlit to allow users to interactively adjust inputs and receive instant predictions.

The project demonstrates the full machine learning lifecycle:  
**data preprocessing → model training → evaluation → deployment → real-time inference**.

---

## Live Demo

🔗 **Streamlit App:**  
https://wine-quality-classifier-gtsszwa7vashcyn2hp7ccd.streamlit.app/

---

## Machine Learning Approach

- **Problem Type:** Binary Classification (High vs Low Quality)
- **Model:** Feedforward Neural Network (TensorFlow / Keras)
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam
- **Evaluation Metric:** Accuracy
- **Imbalance Handling:** Class weights applied during training

---

## Dataset

- **Source:** UCI Machine Learning Repository – Wine Quality Dataset
- **Wine Type:** White Wine
- **Features:**
  - Fixed acidity
  - Volatile acidity
  - Citric acid
  - Residual sugar
  - Chlorides
  - Free sulfur dioxide
  - Total sulfur dioxide
  - Density
  - pH
  - Sulphates
  - Alcohol
- **Target Variable:** Wine quality (converted into binary classes)

---

## Web App Features

- 📈 Interactive data visualizations:
  - Correlation heatmap
  - Line, area, and bar charts
  - Wine quality distribution pie chart
- 🎛️ Sidebar sliders for adjusting wine chemical properties
- 🤖 Real-time predictions using a trained neural network
- 📊 Prediction probability visualization (High vs Low quality)
- ⚡ Cached data loading and model training for performance optimization

---

## Tech Stack

- Python
- TensorFlow / Keras
- Scikit-Learn
- Pandas & NumPy
- Matplotlib & Seaborn
- Streamlit

---

## 📁 Project Structure

├── app.py
├── winequality-white.csv
├── requirements.txt
└── README.md

# Key Learnings

Built and deployed a production-ready machine learning application

Addressed class imbalance using class weighting

Applied feature scaling consistently during training and inference

Integrated ML models with an interactive user interface

Demonstrated real-world application of data science concepts


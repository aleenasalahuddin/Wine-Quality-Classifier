import pandas as pd
import tensorflow as tf
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import Counter
import os

# Ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- App Title Section ---
st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="wide")

st.markdown(
    """
    <div style="text-align:center; padding:1rem 0;">
        <h1 style="color:#8E4585; font-size:2.5rem;">🍷 Wine Quality Prediction</h1>
        <p style="color:#555; font-size:1.1rem;">
            Analyze wine characteristics and predict its quality — powered by Machine Learning.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.image(
    "https://images.unsplash.com/photo-1601924638867-3ec6c8b1b2ce?auto=format&fit=crop&w=1200&q=80",
    use_column_width=True,
    caption="Explore how subtle chemical properties define wine excellence 🍇"
)

# --- Data Loading and Preprocessing ---
# Use st.cache_data to cache the data loading and preprocessing steps
@st.cache_data
def load_and_preprocess_data(file_path):
    # Adjust this path for your GitHub repository.
    # For example, if 'winequality-white.csv' is in a 'data' subfolder:
    # data_path = os.path.join(os.path.dirname(__file__), 'data', 'winequality-white.csv')
    
    # For this Colab context, we'll use the provided drive path.
    # Make sure this file exists in your deployed environment or adjust the path.
    try:
        df = pd.read_csv(file_path, delimiter=';')
    except FileNotFoundError:
        st.error(f"Error: Dataset not found at {file_path}. Please ensure the file is uploaded or the path is correct.")
        st.stop()

    # Store original quality for some visualizations if needed, and create encoded for model
    encoder = LabelEncoder()
    df['quality_encoded'] = encoder.fit_transform(df['quality'])

    # Prepare target variable (binary classification)
    y = pd.qcut(df['quality_encoded'], q=2, labels=[0, 1])
    
    # Features (dropping original and encoded quality columns)
    X = df.drop(['quality', 'quality_encoded'], axis=1)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.8, random_state=34)
    
    # Convert y_train and y_test to integer dtype
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    num_feat = X_scaled.shape[1]

    return df, X_train, y_train, X_scaled, scaler, num_feat

# Define the path to your dataset
# IMPORTANT: Change this path if you are not running in Google Colab 
# and your file is not in /content/drive/MyDrive/wine-quality-dataset/ 
DATA_FILE_PATH = '/content/drive/MyDrive/wine-quality-dataset/winequality-white.csv'
df, X_train, y_train, X_scaled_all, scaler, num_feat = load_and_preprocess_data(DATA_FILE_PATH)

# --- Model Training (cached to run only once) ---
# Use st.cache_resource to cache the model training so it only runs once
@st.cache_resource
def train_tf_model(X_train_data, y_train_data, num_features):
    inputs = tf.keras.Input(shape=(num_features,))
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.Dense(num_features, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Train with verbose=0 to avoid outputting training progress in Streamlit app logs
    model.fit(X_train_data, y_train_data, validation_split=0.2, batch_size=32, epochs=10, verbose=0)
    return model

model = train_tf_model(X_train, y_train, num_feat)

# --- Layout: Columns for Overview + Data Peek ---
col1, col2 = st.columns([1.5, 1])

with col1:
    st.markdown("### 📊 Dataset Overview")
    st.dataframe(df.head(10))

with col2:
    st.markdown("### 🔍 Key Statistics")
    st.metric("Rows", df.shape[0])
    st.metric("Columns", df.shape[1])
    # Using 'quality' for unique quality levels to show original range
    st.metric("Unique Original Quality Levels", len(df['quality'].unique()))

# --- Divider ---
st.markdown("---")

# --- Visualization Section Header ---
st.markdown(
    "<h2 style='text-align:center; color:#8E4585;'>Data Visualizations</h2>",
    unsafe_allow_html=True,
)

st.subheader('Raw Data Display')
st.dataframe(df)

st.markdown('<h3>Correlation Heatmap</h3>', unsafe_allow_html=True)
corr = df.corr(numeric_only=True) # Ensure correlation is computed only for numeric columns
fig_heatmap, ax_heatmap = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, annot=True, vmin=-1.0, vmax = 1.0, ax=ax_heatmap)
st.pyplot(fig_heatmap) # Pass the figure object to st.pyplot()

# Line chart
st.markdown('<h3>Line Chart: Quality vs Alcohol</h3>', unsafe_allow_html=True)
st.line_chart(df[['quality', 'alcohol']]) # Using original quality for visualization

# Area chart
st.markdown('<h3>Area Chart: Quality vs Alcohol</h3>', unsafe_allow_html=True)
st.area_chart(df[['quality', 'alcohol']])

# Bar chart
st.markdown('<h3>Bar Chart: Quality vs Alcohol</h3>', unsafe_allow_html=True)
st.bar_chart(df[['quality', 'alcohol']])

# Pie chart (Distribution of binary quality levels)
st.markdown('<h3>Pie Chart: Distribution of Wine Quality (Binary Classification)</h3>', unsafe_allow_html=True)
# Count the occurrences of each binary quality label (0 or 1)
binary_quality_y = pd.qcut(df['quality_encoded'], q=2, labels=[0, 1]) # Re-create y for pie chart if needed
binary_quality_counts = binary_quality_y.value_counts().sort_index()
fig_pie, ax_pie = plt.subplots()
ax_pie.pie(binary_quality_counts,
           labels=binary_quality_counts.index.map(lambda x: "High Quality" if x == 1 else "Low Quality"),
           autopct='%1.1f%%',
           startangle=90,
           colors=['#FF9999', '#66B2FF']) # Custom colors for better distinction
ax_pie.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
st.pyplot(fig_pie)

# --- Sidebar Input Widgets ---
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.info("Use sliders to adjust input values and predict wine quality below ↓")

    st.sidebar.title("Input Values")
    # Dynamic min/max for sliders based on the original DataFrame for user-friendliness
    fixed_acidity = st.sidebar.slider("Fixed acidity:", float(df['fixed acidity'].min()), float(df['fixed acidity'].max()), float(df['fixed acidity'].mean()))
    volatile_acidity = st.sidebar.slider("Volatile acidity:", float(df['volatile acidity'].min()), float(df['volatile acidity'].max()), float(df['volatile acidity'].mean()))
    citric_acid = st.sidebar.slider("Citric acid:", float(df['citric acid'].min()), float(df['citric acid'].max()), float(df['citric acid'].mean()))
    residual_sugar = st.sidebar.slider("Residual sugar:", float(df['residual sugar'].min()), float(df['residual sugar'].max()), float(df['residual sugar'].mean()))
    chlorides = st.sidebar.slider("Chlorides:", float(df['chlorides'].min()), float(df['chlorides'].max()), float(df['chlorides'].mean()))
    free_sulfur_dioxide = st.sidebar.slider("Free sulfur dioxide:", float(df['free sulfur dioxide'].min()), float(df['free sulfur dioxide'].max()), float(df['free sulfur dioxide'].mean()))
    total_sulfur_dioxide = st.sidebar.slider("Total sulfur dioxide:", float(df['total sulfur dioxide'].min()), float(df['total sulfur dioxide'].max()), float(df['total sulfur dioxide'].mean()))
    density = st.sidebar.slider("Density:", float(df['density'].min()), float(df['density'].max()), float(df['density'].mean()))
    pH = st.sidebar.slider("pH:", float(df['pH'].min()), float(df['pH'].max()), float(df['pH'].mean()))
    sulphates = st.sidebar.slider("Sulphates:", float(df['sulphates'].min()), float(df['sulphates'].max()), float(df['sulphates'].mean()))
    alcohol = st.sidebar.slider("Alcohol:", float(df['alcohol'].min()), float(df['alcohol'].max()), float(df['alcohol'].mean()))

# Create a data frame with the values from the sliders (original unscaled values)
input_data_original = pd.DataFrame({
    "fixed acidity": [fixed_acidity],
    "volatile acidity": [volatile_acidity],
    "citric acid": [citric_acid],
    "residual sugar": [residual_sugar],
    "chlorides": [chlorides],
    "free sulfur dioxide": [free_sulfur_dioxide],
    "total sulfur dioxide": [total_sulfur_dioxide],
    "density": [density],
    "pH": [pH],
    "sulphates": [sulphates],
    "alcohol": [alcohol]
})

# Scale the input data using the *same* scaler used for training
input_data_scaled = pd.DataFrame(scaler.transform(input_data_original), columns=X_scaled_all.columns)

# Define the map_prediction_to_label function (simplified based on binary classification)
def map_prediction_to_label(prediction_value):
    """Map a binary prediction (0 or 1) to a quality label."""
    if prediction_value == 1:
        return "High Quality"
    else:
        return "Low Quality"

# Make prediction when a button is clicked
if st.sidebar.button("Predict Wine Quality"):
    prediction = model.predict(input_data_scaled)
    prediction_binary = np.round(prediction).flatten()[0] # Get the single binary prediction
    quality_label = map_prediction_to_label(prediction_binary)

    st.subheader("Prediction Result")
    st.write(f"The predicted wine quality is: **{quality_label}**")

    # Display prediction probability as a bar chart
    st.markdown("<h3>Prediction Probability</h3>", unsafe_allow_html=True)
    probability_high = prediction[0][0]
    probability_low = 1 - probability_high

    prob_df = pd.DataFrame({
        'Quality Level': ['Low Quality', 'High Quality'],
        'Probability': [probability_low, probability_high]
    })
    st.bar_chart(prob_df.set_index('Quality Level'))

import pandas as pd
import tensorflow as tf
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Streamlit Page Config
st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="wide")

# Header
st.markdown("""
    <div style="text-align:center; padding:1rem 0;">
        <h1 style="color:#8E4585;">🍷 Wine Quality Prediction</h1>
        <p>Analyze wine characteristics and predict its quality — powered by Machine Learning.</p>
    </div>
""", unsafe_allow_html=True)

st.image(
    "https://images.unsplash.com/photo-1601924638867-3ec6c8b1b2ce?auto=format&fit=crop&w=1200&q=80",
    use_column_width=True,
    caption="Explore how subtle chemical properties define wine excellence 🍇"
)

# --- Data Loading ---
@st.cache_data
def load_data(path):
    df = pd.read_csv(path, delimiter=';')
    encoder = LabelEncoder()
    df['quality_encoded'] = encoder.fit_transform(df['quality'])
    y = pd.qcut(df['quality_encoded'], q=2, labels=[0, 1])
    X = df.drop(['quality', 'quality_encoded'], axis=1)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.astype(int), test_size=0.2, random_state=34)
    return df, X_train, y_train, scaler

df, X_train, y_train, scaler = load_data('winequality-white.csv')

# --- Model ---
@st.cache_resource
def train_model(X_train, y_train):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(X_train.shape[1], activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=32, verbose=0)
    return model

model = train_model(X_train, y_train)

# --- Data Overview ---
st.markdown("### 📊 Dataset Overview")
col1, col2 = st.columns(2)
with col1:
    st.dataframe(df.head(10))
with col2:
    st.metric("Rows", df.shape[0])
    st.metric("Columns", df.shape[1])
    st.metric("Unique Quality Levels", len(df['quality'].unique()))

st.divider()

# --- Visualization ---
st.markdown("<h2 style='color:#8E4585;text-align:center;'>Data Visualizations</h2>", unsafe_allow_html=True)
corr = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(corr, annot=True, vmin=-1, vmax=1, cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.line_chart(df[['quality', 'alcohol']])
st.area_chart(df[['quality', 'alcohol']])
st.bar_chart(df[['quality', 'alcohol']])

# --- Sidebar Inputs ---
st.sidebar.header("🔧 Adjust Wine Features")
input_features = {}
for col in df.columns[:-2]:  # exclude encoded columns
    input_features[col] = st.sidebar.slider(
        col.capitalize(),
        float(df[col].min()), float(df[col].max()), float(df[col].mean())
    )

# Convert and scale input
input_df = pd.DataFrame([input_features])
input_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)

# --- Prediction Button ---
st.sidebar.markdown("---")
if st.sidebar.button("🔮 Predict Wine Quality"):
    with st.spinner("Analyzing wine properties..."):
        prediction = np.array(model.predict(input_scaled, verbose=0))[0][0]
        pred_binary = int(np.round(prediction))
        quality_label = "High Quality 🍇" if pred_binary == 1 else "Low Quality 🍷"

        st.subheader("Prediction Result")
        st.markdown(f"<h3 style='color:#8E4585;'>Predicted Wine Quality: {quality_label}</h3>", unsafe_allow_html=True)

        # Probability chart (dynamic visualization)
        prob_df = pd.DataFrame({
            'Quality Level': ['Low Quality', 'High Quality'],
            'Probability': [1 - prediction, prediction]
        })
        st.bar_chart(prob_df.set_index('Quality Level'))

        # NEW pie chart for current prediction
        fig_pred, ax_pred = plt.subplots()
        ax_pred.pie(
            [1 - prediction, prediction],
            labels=['Low Quality', 'High Quality'],
            autopct='%1.1f%%',
            colors=['#FF9999', '#66B2FF']
        )
        ax_pred.axis('equal')
        st.pyplot(fig_pred)

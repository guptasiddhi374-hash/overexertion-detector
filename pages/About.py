import streamlit as st
import pandas as pd

# 1. Page Title
st.title("📊 Model Performance & Analysis")

# 2. Key Metrics (Your exact dissertation results)
st.subheader("Performance Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", "99.83%")
col2.metric("Precision", "99.85%")
col3.metric("Recall", "99.83%")
col4.metric("F1-Score", "99.82%")

st.divider()

# 3. Display the Images you uploaded to GitHub
st.subheader("Confusion Matrix")
st.image("confusion_matrix.png", caption="This matrix shows the model correctly identified almost all 'Stress' and 'Active' states.")

st.subheader("Learning Curve")
st.image("learning_curve.png", caption="The plateau indicates the model is stable and not overfitting.")

# 4. MBA Business Context
st.info("""
**Executive Summary:** The high accuracy and precision demonstrate that this Random Forest model is 
reliable for wearable healthcare deployment, significantly outperforming 
baseline Logistic Regression models.
""")

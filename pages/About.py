import streamlit as st
import pandas as pd

# 1. Professional Header
st.set_page_config(page_title="Model Analytics", page_icon="📊")
st.title("📊 Model Performance & Business Logic")

# 2. Key Metrics Summary
st.subheader("Final Evaluation Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", "99.83%", help="Overall correctness of the model")
col2.metric("Precision", "99.85%", help="Ability to avoid false alarms")
col3.metric("Recall", "99.83%", help="Ability to detect every stress event")
col4.metric("F1-Score", "99.82%", help="Balance between Precision and Recall")

st.divider()

# 3. Algorithm Comparison (The "MBA" Part)
st.subheader("Strategic Model Selection")
st.write("The Random Forest algorithm was selected after comparing performance against industry baselines.")

comparison_data = {
    "Algorithm": ["Logistic Regression", "Decision Tree", "Random Forest (Proposed)"],
    "Accuracy": ["98.43%", "99.12%", "99.83%"],
    "Reliability": ["Low", "Moderate", "Very High"],
    "Complexity": ["Simple", "Medium", "High (Ensemble)"]
}
df_comp = pd.DataFrame(comparison_data)
st.table(df_comp)

# 4. Technical Visualizations
st.subheader("Visual Validation")
col_img1, col_img2 = st.columns(2)

with col_img1:
    st.image("confusion_matrix.png", caption="Confusion Matrix: High Class Separation")
with col_img2:
    st.image("learning_curve.png", caption="Learning Curve: No Overfitting Detected")

# 5. Business Impact & Future Scope
st.divider()
st.subheader("💡 Strategic Value & Future Scope")
st.info("""
**Business Value:** This system provides a low-cost, high-accuracy monitoring solution for 
occupational health, potentially reducing workplace overexertion incidents by 40%.

**Future Roadmap:**
1. **Real-time API Integration:** Connecting to Garmin/Fitbit Cloud APIs.
2. **Edge Deployment:** Optimizing the model to run directly on wearable hardware.
3. **User Personalization:** Implementing 'Transfer Learning' to adapt to individual baselines.
""")

import streamlit as st
import joblib
import pandas as pd

# 1. Page Configuration (Wide mode is better for screenshots)
st.set_page_config(page_title="Wearable AI Monitor", page_icon="⌚", layout="wide")

# 2. Sidebar: Branding, Navigation & Ethics
st.sidebar.header("🛡️ System Information")
st.sidebar.info("Predicting physiological overexertion to improve occupational safety.")

st.sidebar.divider()
st.sidebar.subheader("🔒 Data Privacy & Ethics")
st.sidebar.caption("""
This prototype follows **Privacy-by-Design** principles. 
No Personal Identifiable Information (PII) is stored or 
transmitted. Data is processed locally for inference.
""")

# 3. Main Interface Header
st.title("⌚ Live Overexertion Detector")
st.markdown("### Interactive Decision Support Simulation")

# 4. User Inputs (Simulating Wearable Sensors)
# We use columns to make the UI look professional
col1, col2 = st.columns(2)

with col1:
    hr = st.slider("Heart Rate (BPM)", 40, 200, 80)
    eda = st.slider("EDA (Skin Conductance)", 0.0, 20.0, 1.0)

with col2:
    temp_val = st.slider("Body Temp (°C)", 30.0, 42.0, 36.5)
    acc = st.slider("Movement (Acc_Mag)", 0.0, 50.0, 1.0)

# 5. Load Model and Perform Inference
try:
    # Load the serialized Random Forest model
    model = joblib.load('overexertion_model.pkl')
    
    # CRITICAL FIX: Mapping slider 'temp_val' to model feature 'TEMP'
    input_data = pd.DataFrame([[hr, eda, temp_val, acc]], 
                              columns=['HR', 'EDA', 'TEMP', 'Acc_Mag'])
    
    # Generate Prediction
    prediction = model.predict(input_data)[0]

    st.divider()
    
    # 6. Dynamic Clinical Guidance & MBA Insights
    if prediction == "Stress":
        st.error(f"## Current State: {prediction}")
        st.write("""
        **⚠️ High Overexertion Detected:** * **Clinical Advice**: Immediate cessation of physical activity. 
        * **Intervention**: Perform guided recovery breathing (4-7-8 method).
        * **Business Impact**: High risk of workplace injury; intervention required.
        """)
    elif prediction == "Active":
        st.success(f"## Current State: {prediction}")
        st.write("""
        **🏃 Optimal Performance Zone:** * **Clinical Advice**: You are in the target aerobic training zone. 
        * **Monitoring**: Maintain current pace and ensure electrolyte hydration.
        """)
    else:
        st.info(f"## Current State: {prediction}")
        st.write("""
        **🧘 Recovery / Neutral State:** * **Clinical Advice**: Physiological markers indicate a stable resting baseline. 
        * **Status**: System standby.
        """)

except Exception as e:
    # Professional error handling for the Viva
    st.error("⚠️ **System Configuration Error**")
    st.write(f"Details: {e}")
    st.info("Check if 'overexertion_model.pkl' is in the main GitHub folder.")

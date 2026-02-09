import streamlit as st
import joblib
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Wearable AI Monitor", page_icon="⌚", layout="wide")

# 2. Sidebar Branding & Ethics
st.sidebar.header("🛡️ System Information")
st.sidebar.info("Predicting physiological overexertion to improve occupational safety.")
st.sidebar.divider()
st.sidebar.subheader("🔒 Data Privacy & Ethics")
st.sidebar.caption("Privacy-by-Design prototype. No PII is stored.")

# 3. Main Interface
st.title("⌚ Live Overexertion Detector")
st.markdown("### Interactive Decision Support Simulation")

# 4. User Inputs
col1, col2 = st.columns(2)
with col1:
    hr = st.slider("Heart Rate (BPM)", 40, 200, 80)
    eda = st.slider("EDA (Skin Conductance)", 0.0, 20.0, 1.0)
with col2:
    temp = st.slider("Body Temp (°C)", 30.0, 42.0, 36.5)
    acc = st.slider("Movement (Acc_Mag)", 0.0, 50.0, 1.0)

# 5. The Numerical Bypass Logic
try:
    model = joblib.load('overexertion_model.pkl')
    
    # We convert the inputs into a raw numpy array
    # This ignores the column names and just gives the model the numbers
    # Ensure this order matches your training: HR, EDA, TEMP, Acc_Mag
    input_data = np.array([[hr, eda, temp, acc]])
    
    prediction = model.predict(input_data)[0]

    st.divider()
    
    # 6. Dynamic Clinical Guidance
    if prediction == "Stress":
        st.error(f"## Current State: {prediction}")
        st.write("**⚠️ High Overexertion Detected.** Recommended: Immediate recovery protocol.")
    elif prediction == "Active":
        st.success(f"## Current State: {prediction}")
        st.write("**🏃 Optimal Activity.** You are in the target aerobic zone.")
    else:
        st.info(f"## Current State: {prediction}")
        st.write("**🧘 Neutral / Recovery.** Physiological markers are stable.")

except Exception as e:
    st.error("⚠️ System Configuration Error")
    st.write(f"**Details:** {e}")



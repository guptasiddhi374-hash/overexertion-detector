import streamlit as st
import joblib
import numpy as np

# 1. Page Configuration (Set to wide for a better layout in your dissertation)
st.set_page_config(page_title="Wearable AI Monitor", page_icon="⌚", layout="wide")

# 2. Sidebar with Ethics & Strategic Context
st.sidebar.header("🛡️ System Information")
st.sidebar.info("Predicting physiological overexertion to improve workplace safety.")
st.sidebar.divider()
st.sidebar.subheader("🔒 Data Privacy & Ethics")
st.sidebar.caption("Privacy-by-Design prototype. No PII is stored or transmitted.")

# 3. Main Header
st.title("⌚ Live Overexertion Detector")
st.markdown("### Prescriptive Decision Support Simulation")

# 4. User Inputs (Simulating Sensors)
col1, col2 = st.columns(2)
with col1:
    hr = st.slider("Heart Rate (BPM)", 40, 200, 80)
    eda = st.slider("EDA (Skin Conductance)", 0.0, 20.0, 1.0)
with col2:
    temp = st.slider("Body Temp (°C)", 30.0, 42.0, 36.5)
    acc = st.slider("Movement (Acc_Mag)", 0.0, 50.0, 1.0)

# 5. Robust Prediction Logic
try:
    # Load the model file from your GitHub main folder
    model = joblib.load('overexertion_model.pkl')
    
    # We use a raw numpy array to bypass naming/casing errors
    # ORDER MATTERS: HR, EDA, TEMP, Acc_Mag
    input_array = np.array([[hr, eda, temp, acc]])
    
    prediction = model.predict(input_array)[0]

    st.divider()
    
    # 6. Detailed & Alternative Interventions
    if prediction == "Stress":
        st.error(f"## Current Status: {prediction}")
        st.subheader("📋 Targeted Interventions")
        
        # Using tabs to offer alternatives
        p_tab, s_tab = st.tabs(["🚀 Primary Action", "🕶️ Stealth Alternative"])
        with p_tab:
            st.markdown("**Mandatory Recovery:** Immediate cessation of high-intensity tasks. Move to a quiet, cool environment for 10 minutes.")
        with s_tab:
            st.markdown("**Box Breathing:** If you are in a meeting, use the 4-4-4-4 breathing technique. This lowers EDA and Heart Rate without requiring physical movement.")
            
    elif prediction == "Active":
        st.success(f"## Current Status: {prediction}")
        st.subheader("🏃 Performance Optimization")
        
        m_tab, c_tab = st.tabs(["📈 Maintain Momentum", "📉 Safe Transition"])
        with m_tab:
            st.markdown("**Optimal Zone:** You are currently in the target aerobic zone. Keep this pace for maximum cardiovascular health.")
        with c_tab:
            st.markdown("**Active Recovery:** If fatigue sets in, transition to a brisk walk (3 mph). This clears lactic acid while maintaining heart health.")

    else:
        st.info(f"## Current Status: {prediction}")
        st.subheader("🧘 Recovery Management")
        st.write("Markers are at baseline. Ideal for strategic planning, cognitive work, or nutritional refueling.")

except Exception as e:
    # Clear error handling for your VIVA presentation
    st.error("⚠️ **System Configuration Error**")
    st.write(f"**Details:** {e}")
    st.info("Ensure 'overexertion_model.pkl' is in the main directory.")




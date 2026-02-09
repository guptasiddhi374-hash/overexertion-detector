import streamlit as st
import joblib
import pandas as pd

# 1. Page Configuration
st.set_page_config(page_title="Wearable AI Monitor", page_icon="⌚", layout="wide")

# 2. Sidebar Branding
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
    temp_input = st.slider("Body Temp (°C)", 30.0, 42.0, 36.5)
    acc = st.slider("Movement (Acc_Mag)", 0.0, 50.0, 1.0)

# 5. The "Fail-Proof" Inference Logic
try:
    model = joblib.load('overexertion_model.pkl')
    
    # We create a dictionary first to force the names to match exactly
    data_dict = {
        'HR': [hr],
        'EDA': [eda],
        'TEMP': [temp_input],
        'Acc_Mag': [acc]
    }
    input_df = pd.DataFrame(data_dict)
    
    # Make Prediction
    prediction = model.predict(input_df)[0]

    st.divider()
    
    # 6. Dynamic Clinical Guidance
    if prediction == "Stress":
        st.error(f"## Current State: {prediction}")
        st.write("**⚠️ High Overexertion Detected.** Immediate recovery protocol recommended.")
    elif prediction == "Active":
        st.success(f"## Current State: {prediction}")
        st.write("**🏃 Optimal Activity.** Maintaining target aerobic zone.")
    else:
        st.info(f"## Current State: {prediction}")
        st.write("**🧘 Neutral / Recovery.** Stable physiological baseline.")

except Exception as e:
    st.error("⚠️ System Configuration Error")
    st.write(f"**Details:** {e}")
    st.info("Tip: If the error persists, ensure 'TEMP' is capitalized in your training dataset.")


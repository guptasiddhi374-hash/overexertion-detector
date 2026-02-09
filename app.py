import streamlit as st
import joblib
import pandas as pd

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

# 4. User Inputs (Matching your model's training variables)
col1, col2 = st.columns(2)
with col1:
    HR = st.slider("Heart Rate (BPM)", 40, 200, 80)
    EDA = st.slider("EDA (Skin Conductance)", 0.0, 20.0, 1.0)
with col2:
    TEMP = st.slider("Body Temp (°C)", 30.0, 42.0, 36.5)
    Acc_Mag = st.slider("Movement (Acc_Mag)", 0.0, 50.0, 1.0)

# 5. The "Exact-Match" Inference Logic
try:
    model = joblib.load('overexertion_model.pkl')
    
    # We create the DataFrame using ALL CAPS keys to match your fit-time error
    input_df = pd.DataFrame({
        'HR': [HR],
        'EDA': [EDA],
        'TEMP': [TEMP],
        'Acc_Mag': [Acc_Mag]
    })
    
    # Force the column order just in case
    input_df = input_df[['HR', 'EDA', 'TEMP', 'Acc_Mag']]
    
    prediction = model.predict(input_df)[0]

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
    


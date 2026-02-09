import streamlit as st
import joblib
import pandas as pd

# 1. Page Configuration & Sidebar
st.set_page_config(page_title="Wearable AI Monitor", page_icon="⌚")

st.sidebar.header("🛡️ System Information")
st.sidebar.info("""
This AI model predicts physical states using physiological sensor data. 
Use the 'About' page in the sidebar for technical validation.
""")

# Ethics & Privacy Section
st.sidebar.divider()
st.sidebar.subheader("🔒 Data Privacy & Ethics")
st.sidebar.caption("""
This prototype follows **Privacy-by-Design** principles. 
No Personal Identifiable Information (PII) is stored or 
transmitted. Data is processed locally for inference.
""")

# 2. Main Interface
st.title("⌚ Live Overexertion Detector")
st.write("Adjust the sliders below to simulate real-time wearable sensor inputs.")

# 3. User Inputs (Simulating Sensors)
col1, col2 = st.columns(2)

with col1:
    hr = st.slider("Heart Rate (BPM)", 40, 200, 80)
    eda = st.slider("EDA (Skin Conductance)", 0.0, 20.0, 1.0)

with col2:
    temp = st.slider("Body Temp (°C)", 30.0, 42.0, 36.5)
    acc = st.slider("Movement (Acc_Mag)", 0.0, 50.0, 1.0)

# 4. Load Model and Make Prediction
try:
    model = joblib.load('overexertion_model.pkl')
    # Create feature dataframe (Ensure columns match your training data)
    input_data = pd.DataFrame([[hr, eda, temp, acc]], 
                              columns=['HR', 'EDA', 'TEMP', 'Acc_Mag'])
    
    prediction = model.predict(input_data)[0]

    # 5. Dynamic Prediction Display & Clinical Guidance
    st.divider()
    st.subheader(f"Current State: {prediction}")

    if prediction == "Stress":
        st.error("⚠️ **Action Required: High Overexertion Detected**")
        st.write("""
        **Clinical Guidance:**
        * **Immediate**: Cease strenuous physical activity.
        * **Intervention**: Practice 4-7-8 breathing for 2 minutes.
        * **Strategic**: Evaluate environmental triggers and workload.
        """)
    elif prediction == "Active":
        st.success("🏃 **Performance Status: Optimal Activity**")
        st.write("""
        **Clinical Guidance:**
        * **Goal**: You are within the target aerobic training zone.
        * **Monitoring**: Maintain current pace; ensure proper hydration.
        """)
    else:
        st.info("🧘 **Status: Neutral / Recovery**")
        st.write("Current physiological markers indicate a stable resting or recovery state.")

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.warning("Ensure 'overexertion_model.pkl' is uploaded to your main GitHub folder.")

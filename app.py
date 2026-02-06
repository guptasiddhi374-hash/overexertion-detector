import streamlit as st
import joblib
import pandas as pd

# 1. Load the "Brain" (The model you saved from Colab)
# Ensure your .pkl file is in the same folder as this script!
model = joblib.load('overexertion_model.pkl')

# 2. Design the Website Header
st.title("🏃 Overexertion & Stress Monitoring System")
st.markdown("""
This AI model predicts your physical state based on wearable sensor data. 
Adjust the sensors below to see the real-time prediction.
""")

# 3. Create the User Interface (Sliders)
st.sidebar.header("Wearable Sensor Inputs")
hr = st.sidebar.slider("Heart Rate (BPM)", 40, 200, 75)
eda = st.sidebar.slider("Skin Conductance (EDA)", 0.0, 20.0, 1.0)
temp = st.sidebar.slider("Skin Temp (°C)", 28.0, 40.0, 32.0)
acc = st.sidebar.slider("Movement (Acc_Mag)", 0.0, 2.0, 0.1)

# 4. Process data and Make Prediction
# The column names must match exactly what was used during training
input_df = pd.DataFrame([[hr, eda, temp, acc]], 
                        columns=['HR', 'EDA', 'Temp', 'Acc_Mag'])

prediction = model.predict(input_df)[0]

# 5. Display the Result with professional styling
st.subheader("AI Analysis Result:")
if prediction == "Active":
    st.info(f"The user is currently: **{prediction}**")
    if hr > 175:
        st.error("⚠️ WARNING: High intensity detected. Risk of overexertion!")
elif prediction == "Stress":
    st.warning(f"The user is currently: **{prediction}**")
else:
    st.success(f"The user is currently: **{prediction}**")

# 6. Show the raw data table (optional)
with st.expander("See Raw Sensor Data"):
    st.write(input_df)
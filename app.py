# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import gradio as gr
import joblib
import numpy as np

# Load the trained model
model = joblib.load('rf_model.joblib')

def predict_disease(age, sex, trestbps, chol, fbs, thalach, exang, oldpeak, slope,
                    cp_atypical, cp_non_anginal, cp_typical, restecg_normal, restecg_abnormal):
    
    # Create input array in the EXACT order your model expects
    input_data = np.array([[
        age, sex, trestbps, chol, fbs, thalach, exang, oldpeak, slope,
        cp_atypical, cp_non_anginal, cp_typical, restecg_normal, restecg_abnormal
    ]])
    
    # Get prediction and probability
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]
    
    # Determine risk level
    if prob < 0.3:
        risk = "🟢 Low Risk"
    elif prob < 0.6:
        risk = "🟡 Moderate Risk"
    else:
        risk = "🔴 High Risk"
    
    # Result message
    if pred == 1:
        return f"⚠️ HEART DISEASE DETECTED\nProbability: {prob:.1%}\nRisk Level: {risk}"
    else:
        return f"✅ NO HEART DISEASE\nProbability: {prob:.1%}\nRisk Level: {risk}"

# Create Gradio interface
demo = gr.Interface(
    fn=predict_disease,
    inputs=[
        gr.Slider(20, 100, value=50, label="Age"),
        gr.Radio([0, 1], value=0, label="Sex (0=Female, 1=Male)"),
        gr.Slider(80, 200, value=120, label="Blood Pressure"),
        gr.Slider(100, 400, value=200, label="Cholesterol"),
        gr.Radio([0, 1], value=0, label="Fasting Blood Sugar (0=No, 1=Yes)"),
        gr.Slider(70, 220, value=150, label="Max Heart Rate"),
        gr.Radio([0, 1], value=0, label="Exercise Angina (0=No, 1=Yes)"),
        gr.Slider(0.0, 6.2, value=1.0, label="ST Depression"),
        gr.Slider(1, 3, value=2, label="ST Slope (1-3)"),
        gr.Dropdown([0, 1], value=0, label="Chest Pain: Atypical Angina (0/1)"),
        gr.Dropdown([0, 1], value=0, label="Chest Pain: Non-anginal (0/1)"),
        gr.Dropdown([0, 1], value=0, label="Chest Pain: Typical Angina (0/1)"),
        gr.Dropdown([0, 1], value=0, label="Resting ECG: Normal (0/1)"),
        gr.Dropdown([0, 1], value=0, label="Resting ECG: ST-T Abnormality (0/1)")
    ],
    outputs=gr.Textbox(label="Diagnosis", lines=5),
    title="🫀 Heart Disease Predictor",
    description="Enter patient data to predict heart disease risk"
)

# Launch the app
demo.launch(share = True)

# %%

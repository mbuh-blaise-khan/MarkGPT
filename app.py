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
import pandas as pd
import numpy as np
# Load your trained model
model = joblib.load('rf_model.joblib')

# Define options for categorical features
sex_options = ['male', 'female']
smoker_options = ['yes', 'no']
region_options = ['northwest', 'southeast', 'southwest']

def predict_target(
    age=30, 
    sex='male', 
    bmi=25.0, 
    children=0, 
    smoker='no', 
    region='northwest'
):
    # Convert categorical variables to numerical
    sex_num = 0 if sex == 'male' else 1
    smoker_num = 1 if smoker == 'yes' else 0
    
    # One-hot encode region
    region_nw = 1 if region == 'northwest' else 0
    region_se = 1 if region == 'southeast' else 0
    region_sw = 1 if region == 'southwest' else 0
    
    # Create DataFrame with only the features used for prediction
    input_data = pd.DataFrame([{
        'age': age,
        'sex': sex_num,
        'bmi': bmi,
        'children': children,
        'smoker': smoker_num,
        'region_northwest': region_nw,
        'region_southeast': region_se,
        'region_southwest': region_sw
        
    }])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    return 2.7**prediction

# Explanation for users
description_text = (
    "This tool estimates your expected insurance charges based on your personal information.\n"
    "Please fill in the details below. The model uses features such as age, sex, BMI, "
    "number of children, smoking status, and region. Ensure that your inputs are realistic "
    "and within the specified ranges."
)

# Create the Gradio interface with default values and description
iface = gr.Interface(
    fn=predict_target,
    inputs=[
        gr.Number(
            label='Age (18-100)', 
            minimum=18, 
            maximum=100, 
            value=30
        ),
        gr.Dropdown(
            label='Sex',
            choices=sex_options,
            value='male'
        ),
        gr.Number(
            label='BMI (10-50)', 
            minimum=10, 
            maximum=50, 
            step=0.1,
            value=25.0
        ),
        gr.Number(
            label='Number of Children (0-10)', 
            minimum=0, 
            maximum=10, 
            value=0
        ),
        gr.Dropdown(
            label='Smoker',
            choices=smoker_options,
            value='no'
        ),
        gr.Dropdown(
            label='Region',
            choices=region_options,
            value='northwest'
        )
    ],
    outputs=gr.Number(label='Estimated Charges'),
    title='Insurance Charges Prediction',
    description=description_text
)

# Launch the app
iface.launch(share=True)

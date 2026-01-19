#gradio app 

import gradio as gr
import pandas as pd
import pickle
import numpy as np

# 1. Load the Model
with open("insurance_gb_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# 2. The Logic Function
def predict_cost(age, sex, bmi, children, smoker, region):
    
    # Pack inputs into a DataFrame
    # The column names must match your CSV file exactly
    input_df = pd.DataFrame([[
        age, sex, bmi, children, smoker, region

    ]],
      columns=[
        'age','sex','bmi','children','smoker','region'
    ])
    
    # Predict
    prediction = model.predict(input_df)[0]
    
    # Return formatted result (Clipped 0-5)
    return f"Predicted Insurance Cost: {np.clip(prediction, 0, None):.2f}"
    

# 3. The App Interface
# Defining inputs in a list to keep it clean
inputs = [
        gr.Slider(18, 64, step=1, label="Age"),
        gr.Dropdown(["male","female"], label="Sex"),
        gr.Slider(16, 53.1, step=0.1, label="BMI"),
        gr.Slider(0, 5, step=1, label="Children"),
        gr.Dropdown(["yes","no"], label="Smoker"),
        gr.Dropdown(["southwest","southeast","northwest","northeast"], label="Region")
]

app = gr.Interface(
    fn=predict_cost,
      inputs=inputs,
        outputs="text", 
        title="Medical Insurance Cost Predictor")

app.launch(share=True)
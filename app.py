# Gradio app 
import gradio as gr
import pandas as pd
import pickle
import numpy as np

# 1. Load the Model
with open("insurance_gb_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# 2. The Logic Function
def predict_insurance_cost(age, sex, bmi, children, smoker, region):
    
    # Pack inputs into a DataFrame
    input_df = pd.DataFrame([[
        age, sex, bmi, children, smoker, region
    ]],
      columns=[
        'age', 'sex', 'bmi', 'children', 'smoker', 'region'
    ])
    
    # Predict
    prediction = model.predict(input_df)[0]
    
    # Return formatted result
    return f"Predicted Insurance Cost: ${prediction:.2f}"

# 3. The App Interface
inputs = [
    gr.Number(label="Age", value=30),
    gr.Radio(["male", "female"], label="Sex"),
    gr.Number(label="BMI", value=25.0),
    gr.Slider(0, 5, step=1, label="Children"),
    gr.Radio(["yes", "no"], label="Smoker"),
    gr.Dropdown(["southwest", "southeast", "northwest", "northeast"], label="Region")
]

app = gr.Interface(
    fn=predict_insurance_cost,
      inputs=inputs,
        outputs="text", 
        title="Medical Insurance Cost Predictor")

app.launch(share=True)
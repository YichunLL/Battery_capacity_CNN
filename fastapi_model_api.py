from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import joblib
import os

app = FastAPI()

# ✅ Fix CORS Issue
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all domains (change this for security in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Load the trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get current directory
MODEL_PATH = os.path.join(BASE_DIR, "CNN_5x1_for-app.h5")  # Load model from project directory
model = tf.keras.models.load_model(MODEL_PATH)

# Load the saved scaler
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")  # Path to scaler
scaler = joblib.load(SCALER_PATH)

# Define the input format
class ImpedanceInput(BaseModel):
    impedance_values: list[float]  # Expecting a list of 5 floats

@app.post("/predict")
async def predict(data: ImpedanceInput):
    try:
        # Ensure input length is correct
        if len(data.impedance_values) != 5:
            raise HTTPException(status_code=400, detail="Input must have exactly 5 values.")

        # Convert input to NumPy array
        input_array = np.array(data.impedance_values, dtype=np.float32).reshape(1, -1)
        
        # Normalize input using the saved scaler
        normalized_input = scaler.transform(input_array)

        # Reshape to match the model input format
        processed_input = normalized_input.reshape(1, 5, 1, 1)

        # Make prediction
        prediction = model.predict(processed_input)
        predicted_capacity = float(prediction[0][0])  # Convert NumPy float to Python float

        return {"predicted_capacity": predicted_capacity}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np

app = FastAPI()
pp.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (change later for security)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)
# Load the trained model
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get current directory
MODEL_PATH = os.path.join(BASE_DIR, "CNN_5x1_for-app.h5")  # Load model from project directory
model = tf.keras.models.load_model(MODEL_PATH)

# Define the input format
class ImpedanceInput(BaseModel):
    impedance_values: list[float]  # Expecting a list of 5 floats

@app.post("/predict")
async def predict(data: ImpedanceInput):
    try:
        # Ensure input length is correct
        if len(data.impedance_values) != 5:
            raise HTTPException(status_code=400, detail="Input must have exactly 5 values.")

        # Convert input to NumPy array and reshape
        processed_input = np.array(data.impedance_values, dtype=np.float32).reshape(1, 5, 1, 1)

        # Make prediction
        prediction = model.predict(processed_input)
        predicted_capacity = float(prediction[0][0])  # Convert NumPy float to Python float

        return {"predicted_capacity": predicted_capacity}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

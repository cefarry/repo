# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import List
# import numpy as np
# import joblib  # Assuming you're using a pre-trained model saved as a .pkl file

# # Initialize FastAPI app
# app = FastAPI()

# # Define the Pydantic model for the input data
# class Feature(BaseModel):
#     fixed_acidity: float
#     volatile_acidity: float
#     citric_acid: float
#     residual_sugar: float
#     chlorides: float
#     free_sulfur_dioxide: float
#     total_sulfur_dioxide: float
#     density: float
#     pH: float
#     sulphates: float
#     alcohol: float

# class PredictionRequest(BaseModel):
#     features: List[Feature]

# # Load your pre-trained model (make sure it's saved as a .pkl file or another format that joblib can load)
# model_path = "C:/Users/clax2/Downloads/EAS503Project/random_forest_experiment_2_model.joblib"
# try:
#     model = joblib.load(model_path)
#     print("Model loaded successfully!")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     model = None

# @app.post("/predict")
# async def predict(request: PredictionRequest):
#     try:
#         if model is None:
#             raise HTTPException(status_code=500, detail="Model not loaded properly")

#         # Extract features from the request body (exclude 'quality' from the input features)
#         input_data = np.array([[feature.fixed_acidity, feature.volatile_acidity, feature.citric_acid, feature.residual_sugar,
#                                 feature.chlorides, feature.free_sulfur_dioxide, feature.total_sulfur_dioxide, feature.density,
#                                 feature.pH, feature.sulphates, feature.alcohol] for feature in request.features])

#         # Make predictions using the model
#         predictions = model.predict(input_data)

#         # Return the predictions as a JSON response
#         return {"predictions": predictions.tolist()[0]}
    
#     except Exception as e:
#         # Return error details
#         raise HTTPException(status_code=400, detail=f"Error during prediction: {str(e)}")

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import joblib  # Make sure your model is saved as .joblib or another format

# Initialize FastAPI
app = FastAPI()

# Define the Pydantic model for the input data
class Feature(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

class PredictionRequest(BaseModel):
    features: List[Feature]

# Load your pre-trained model (ensure the path is correct)
model = joblib.load("C:/Users/clax2/Downloads/EAS503Project/random_forest_experiment_2_model.joblib")

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # Extract features from the request body
        input_data = np.array([[feature.fixed_acidity, feature.volatile_acidity, feature.citric_acid, feature.residual_sugar,
                                feature.chlorides, feature.free_sulfur_dioxide, feature.total_sulfur_dioxide, feature.density,
                                feature.pH, feature.sulphates, feature.alcohol] for feature in request.features])

        # Make predictions using the model
        predictions = model.predict(input_data)

        # Return the predictions as a JSON response
        return {"predictions": predictions.tolist()}
    except Exception as e:
        # Handle errors and return error messages
        return {"error": str(e)}

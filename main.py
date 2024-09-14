from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
from pydantic import BaseModel
import joblib
import os

# Initialize the app
app = FastAPI()

# load the trined model using a relative path
model = None
model_path = os.path.join(os.path.dirname(__file__), 'ML_model', 'iris_model.joblib')

try:
    model = joblib.load(model_path)
    print(f"Model loaded successfully: {model_path}")
except FileNotFoundError:
    print(f"Model file not found: {model_path}")
except Exception as e:
    print(f"An error occurred while loading the model: {e}")

# Serve templates directory
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, sepal_length: float = Form(...), sepal_width: float = Form(...), petal_length: float = Form(...), petal_width: float = Form(...)):
    if model is None:
        return templates.TemplateResponse("index.html", {"request": request, "error": "Model not found"})

    data = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)

    try:
        prediction = model.predict(data)[0]
        species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        species = species_mapping.get(prediction, "Unknown")
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": f"An error occurred while making the prediction: {e}"})
    return templates.TemplateResponse("index.html", {"request": request, "species": species})
# Define the request model
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Define the response model
class IrisResponse(BaseModel):
    species: str


@app.post("/api/predict", response_model=IrisResponse)
async def api_predict(iris: IrisRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not found")

    # Prepare the data for prediction
    data = np.array([
        iris.sepal_length,
        iris.sepal_width,
        iris.petal_length,
        iris.petal_width
    ]).reshape(1, -1)

    # Predict the species
    try:
        prediction = model.predict(data)[0]
        species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        species = species_mapping.get(prediction, "Unknown")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while making the prediction: {e}")

    # Return the response model
    return IrisResponse(species=species)
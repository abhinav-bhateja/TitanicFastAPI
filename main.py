from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import joblib
import numpy as np




class Prediction(BaseModel):
    passengerId: int
    pClass: int
    Sex_Male: bool 
    Sex_Female: bool 
    Age: int
    SibSp: float
    Parch: float
    Fare: float
    Embarked_S: bool 
    Embarked_C: bool
    Embarked_Q: bool


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict/")
async def predict(prediction: Prediction):
    loaded_model = joblib.load("model_joblib")

    array = [
        prediction.passengerId,
        prediction.pClass,
        float(prediction.Sex_Male),
        float(prediction.Sex_Female),
        prediction.Age,
        prediction.SibSp,
        prediction.Parch,
        prediction.Fare,
        float(prediction.Embarked_S),
        float(prediction.Embarked_C),
        float(prediction.Embarked_Q),
    ]
    print(array)

    a = np.asarray(array).reshape(1,-1)
    predicted_value= loaded_model.predict(a)
    return {"predict": str(predicted_value[0])}
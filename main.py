from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import re
import logging

from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import Request

limiter = Limiter(key_func=get_remote_address)
# Load artifacts
model = joblib.load("model.pkl")
pp = joblib.load("preprocessor.pkl")
age_mean = joblib.load("age_mean.pkl")

app = FastAPI(title="Maritime Passenger Survival API")
app.state.limiter = limiter

# Define input
class Passenger(BaseModel):
    Age: float = None
    Gender: str
    BoardingPort: str
    Title: str
    TicketTier: float
    TicketCost: float
    FamilySize: float
    FarePerPerson: float
    RelativesAboard: float
    ParentsChildren: float
    CLass: str

def get_prefix(ticket):
    ticket = ticket.strip()
    prefix = re.sub(r'[^A-Za-z/]', '', ticket)
    return 'Numeric' if prefix == '' else prefix.upper()

@app.get("/")
def home():
    return {"message": "Maritime Survival Prediction API is running!"}

@app.post("/predict")
@limiter.limit("10/minute")
def predict(request:Request,passenger: Passenger):
    # Convert to dataframe
    try:
        data = pd.DataFrame([passenger.model_dump()])
        logging.info(f'Received data: {passenger.model_dump()}')
        
        # Fill age if missing
        data['Age'] = data['Age'].fillna(age_mean)
        
        # Ticket prefix
        data['TicketPrefix'] = data['CLass'].apply(get_prefix)
        data.drop('CLass', axis=1, inplace=True)
        
        # Preprocess
        data_pp = pp.transform(data)
        
        # Predict
        prediction = model.predict(data_pp)[0]
        probability = model.predict_proba(data_pp)[0][1]
        logging.info(f'Prediction: {prediction}, Probability: {probability}')
    except ValueError as e:
        raise HTTPException(status_code=422,detail=f"Error processing input: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {e}")
    
            
        
    return {
        "survived": bool(prediction),
        "survival_probability": round(float(probability), 2),
        "message": "Survived!" if prediction == 1 else "Did not survive"
    }
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from prediction import Prediction
import pandas as pd
from typing import List

# Initialize FastAPI
app = FastAPI()

# Path to your model
model_path = "/home/jampanasasank/Documents/Deployment/Deployment/telecom_churn_prediction/model"



@app.get("/ping")
async def ping():
    return {"message": "pong"}

@app.post("/inference")
def prediction(data: List[dict]):
    """
    Handle prediction requests.
    """
    try:
        obj = Prediction(model_path=model_path)
        preds, df = obj.make_prediction(data)  # Adjust the function to handle lists
        return df.to_json(orient='records')

    except Exception as e:
        # Catch and raise any errors
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

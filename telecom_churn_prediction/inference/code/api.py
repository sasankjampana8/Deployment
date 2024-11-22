from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import json
from prediction import Prediction
import pandas as pd
from typing import List

# Initialize FastAPI
app = FastAPI()

# Path to your model
model_path = os.environ['MODEL']
print(model_path)
# model_path = '/home/jampanasasank/Documents/Deployment/Deployment/telecom_churn_prediction/inference/model'



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
        data = pd.DataFrame(data)
        _, df = obj.make_prediction(data)  # Adjust the function to handle lists
        print(True)
        return df.to_dict(orient='records')

    except Exception as e:
        # Catch and raise any errors
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8080, reload=True)

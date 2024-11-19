import pandas as pd 
import os
import pickle as pkl 
from preprocess import Preprocess


class Prediction:
    def __init__(self, model_path) -> None:
        #load the model and estimators
        # self.model = pkl.load(open(model_path, 'rb'))
        self.model_path = model_path
        try:
            for file in os.listdir(self.model_path):
                if "model.pkl" == file:
                    self.model = pkl.load(open(os.path.join(self.model_path, file), 'rb'))
                    print("Model Loaded..")
                    break
        except Exception as e:
            print(f"Error loading model: {e}")
            
    
    def preprocess(self, data):
        #preprocess the data
        process = Preprocess(self.model_path)
        
    
    def make_prediction(self,data ):
        #make prediction
        processed_data = self.preprocess(data)
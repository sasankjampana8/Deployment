import pandas as pd
import os
import pickle as pkl
from preprocess import Preprocess


class Prediction:
    def __init__(self, model_path) -> None:
        """
        Initialize the Prediction class by loading the model and preprocessors.
        """
        self.model_path = model_path
        self.model = None
        
        # Load the model
        try:
            for file in os.listdir(self.model_path):
                if file == "model.pkl":
                    self.model = pkl.load(open(os.path.join(self.model_path, file), 'rb'))
                    print("Model Loaded..")
                    break
            if not self.model:
                raise FileNotFoundError("model.pkl not found in the provided path.")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def preprocess(self, data):
        """
        Preprocess the data using the Preprocess class.
        """
        process = Preprocess(self.model_path)
        processed_data = process.preprocess(data)
        return processed_data
    
    def make_prediction(self, data):
        """
        Make predictions on the preprocessed data.
        """
        try:
            processed_data = self.preprocess(data)
            preds = self.model.predict(processed_data)
            return preds
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None

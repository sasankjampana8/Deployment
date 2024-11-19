import pandas as pd
import numpy as np
import pickle as pkl
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

class Preprocess:
    def __init__(self, model_path):
        self.model_path = model_path  # Assign the argument to an instance variable
        self.estimators = dict()  # Dictionary of estimators
        self.msg = []
        try:
            for file in os.listdir(self.model_path):
                if "pkl" in file:
                    self.estimators[file.split(".")[0]] = pkl.load(open(os.path.join(self.model_path, file), 'rb'))
                elif "model" in file:
                    continue
            print("Loaded the estimators...")
        except Exception as e:
            print(f"Error loading the estimators: {e}")
    
    def process(self, data):
        pass

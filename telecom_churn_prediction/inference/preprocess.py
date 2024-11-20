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
            # print(self.estimators)
        except Exception as e:
            print(f"Error loading the estimators: {e}")
            
    def label_encode(self, data):
        try:
            cat_cols = [
            'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod'
        ]
            for col in cat_cols:
                col_name = f"{col}_label_encoder_estimator"
                encoder = self.estimators[col_name]
                data[col].fillna("missing", inplace=True)
                data[col] = data[col].map(
                    lambda s: "Others" if s not in encoder.classes_ else s
                ) 
                data[col] = encoder.transform(data[col])
            
            return data
                
        
        except Exception as e:
            print(f"Error label encoding categorical columns: {e}")
            
            
    def scaling(self, data):
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        
        num_cols = ['tenure','MonthlyCharges', 'TotalCharges' ]

        
        for col in num_cols:
            col_scaler = f"{col}_scaler_estimator"
            scaler = self.estimators[col_scaler]
            data[col] = scaler.fit_transform(data[[col]])
        
        return data
    
    def drop_columns(self, data):
        data.drop(['customerID'], axis=1, inplace=True)
        return data
    
    def preprocess(self, data):
        data = self.label_encode(data)
        data = self.scaling(data)
        data = self.drop_columns(data)
        print("Preprocessing done")
        return data

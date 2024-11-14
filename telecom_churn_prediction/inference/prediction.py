import pandas as pd 
import os
import pickle as pkl 



class Prediction:
    def __init__(self) -> None:
        #load the model and estimators
        # self.model = pkl.load(open(model_path, 'rb'))
        self.model = None
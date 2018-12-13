# pylint: disable=C0303

"""
Neural net predictor, does not deal nicely with prior probability shifts
"""

import os
import pickle
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import binarize

try:
    import matplotlib.pyplot as plt
    disable_graphics = False
except ImportError:
    disable_graphics = True

from measure_yields.basepredictors import BasePredictor, Prediction
from neuralnet.utilities import select_features


class NaiveNnPredictor(BasePredictor):
    """ 
    A simple neural net predictor
    
    Arguments:
        title: String
        saved_model: Pre-trained Keras model (.h5 file)
        saved_scaler: Pickled StandardScaler (.pkl file)
    """
    
    def __init__(self, title, saved_model, saved_scaler):
        
        super(NaiveNnPredictor, self).__init__(title)
        
        # Load the model
        self.model = load_model(saved_model)
        
        # Load the data scaler
        with open(saved_scaler) as hfile:
            self.scaler = pickle.load(hfile)

        
    
    
    def select_features(self, x, features, include_mass=True):
        """
        Pick out the correct columns to use
        """
        
        return select_features(x, features, include_mass)
    
        
    def predict(self, x):
        """
        Predict number of H, A, bgnd events. Predictions are made by summing up
        the NN outputs for each class, which is a decimal number. 
        """
        
        # Preprocess
        xs = self.scaler.transform(x)
        
        # NN output
        preds = self.model.predict(xs)   # (?, n_class) matrix, floats

        # Sum classes
        num = np.sum(preds, axis=0)
        
        return float(num[1])/len(preds)
        
        

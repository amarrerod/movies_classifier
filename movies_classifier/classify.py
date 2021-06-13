from movies_classifier.utilities import load_model
import numpy as np
from movies_classifier.keys import CLEAN_PIPE, FULL_PIPE, REG_MODEL
import pandas as pd


def classify(X: np.array) -> float:
    """
    Predice la clasificación de X con el modelo de Regresión Lineal
    """
    return load_model(REG_MODEL).predict(X)

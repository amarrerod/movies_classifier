from movies_classifier.utilities import load_model
import numpy as np
from movies_classifier.keys import CLEAN_PIPE, FULL_PIPE, REG_MODEL


def classify(X: np.array) -> float:
    """
    Predice la clasificación de X con el modelo de Regresión Lineal
    """
    return load_model(REG_MODEL).predict(X)


def prepare_data(X: dict) -> np.array:
    must_have = [
        "Year",
        "Age",
        "Rotten Tomatoes",
        "Netflix",
        "Hulu",
        "Prime Video",
        "Disney+",
        "Runtime",
        "Genres",
        "Country",
        "Language",
    ]
    if not all(key in X for key in must_have):
        raise RuntimeError(f"Error in X. Must contain the following keys: {must_have}")
    clean_pipe = load_model(CLEAN_PIPE)
    full_pipe = load_model(FULL_PIPE)
    X = clean_pipe.transform(X)
    X = full_pipe.transform(X)
    return X

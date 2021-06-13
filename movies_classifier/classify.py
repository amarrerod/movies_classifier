from movies_classifier.utilities import load_model
import numpy as np
from movies_classifier.keys import REG_MODEL


def classify(X: np.array) -> float:
    return load_model(REG_MODEL).predict(X)

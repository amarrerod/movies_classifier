from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np
from movies_classifier.classifier.utilities import save_model
from movies_classifier.classifier.keys import REG_MODEL


def create_and_train_model(X, y):
    """
    Crea un modelo Linear Regressor y lo entrena con los datos recibidos
    Además lo guarda en el directorio models para usos posteriores
    """
    reg = LinearRegression()
    reg.fit(X, y)
    scores = cross_val_score(reg, X, y, scoring="neg_mean_squared_error", cv=3)
    reg_rmse = np.sqrt(-scores)
    print(f"Training done with the following resuñts")
    print(f"Scores: {reg_rmse}")
    print(f"Mean: {scores.mean()}")
    print(f"STD: {scores.std()}")
    save_model(reg, REG_MODEL)
    return reg

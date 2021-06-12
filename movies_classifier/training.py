from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
from movies_classifier.utilities import save_model
from sklearn.linear_model import LinearRegression


def create_and_train_model(X, y):
    """
    Crea un modelo RandomForestRegressor y lo entrena con los datos recibidos
    Además lo guarda en el directorio models para usos posteriores
    """
    n_trees = 894
    forest = RandomForestRegressor(n_estimators=n_trees)
    scores = cross_val_score(forest, X, y, scoring="neg_mean_squared_error", cv=10)
    forest_rmse = np.sqrt(-scores)
    print(f"Training done with the following resuñts")
    print(f"Scores: {forest_rmse}")
    print(f"Mean: {scores.mean()}")
    print(f"STD: {scores.std()}")
    save_model(forest, "movies_classifier/models/forest_894.pkl")
    return forest

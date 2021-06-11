
from numpy.core.numeric import cross
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
import joblib


def save_model(model, model_name):
    """
        Guarda el modelo que hemos creado 
        dentro del fichero espeficado
    """
    joblib.dump(model, model_name)


def create_and_train_model(X, y):
    n_trees = 894
    forest = RandomForestRegressor(n_estimators=n_trees)
    scores = cross_val_score(
        forest, X, y, scoring='neg_mean_squared_error', cv=10)
    forest_rmse = np.sqrt(-scores)
    print(f'Training done with the following resuñts')
    print(f'Scores: {scores}')
    print(f'Mean: {scores.mean()}')
    print(f'STD: {scores.std()}')

    return forest

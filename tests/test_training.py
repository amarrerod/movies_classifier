from movies_classifier.training import create_and_train_model
from sklearn.ensemble import RandomForestRegressor
from movies_classifier.preprocessing import create_datasets
from os.path import isfile


def test_create_and_train_model():
    X_train, _, y_train, _ = create_datasets()
    forest = create_and_train_model(X_train, y_train)
    assert type(forest) == type(RandomForestRegressor())
    assert forest.n_estimators == 894
    assert isfile("movies_classifier/models/forest_894.pkl")

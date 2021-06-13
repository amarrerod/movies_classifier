from movies_classifier.classifier.keys import REG_MODEL
from movies_classifier.classifier.training import create_and_train_model
from sklearn.linear_model import LinearRegression
from movies_classifier.classifier.preprocessing import create_datasets
from os.path import isfile


def test_create_and_train_model():
    X_train, _, y_train, _ = create_datasets()
    model = create_and_train_model(X_train, y_train)
    assert type(model) == type(LinearRegression())
    assert isfile(REG_MODEL)

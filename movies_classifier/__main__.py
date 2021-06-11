"""Main module."""

from numpy import save
from movies_classifier.preprocessing import create_datasets
from movies_classifier.training import create_and_train_model, save_model

X_train, X_test, y_train, y_test = create_datasets()
forest = create_and_train_model(X_train, y_train)
save_model(forest, 'forest_894.pkl')

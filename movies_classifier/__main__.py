"""Main module."""

from movies_classifier.preprocessing import create_datasets

X_train, X_test, y_train, y_test = create_datasets()

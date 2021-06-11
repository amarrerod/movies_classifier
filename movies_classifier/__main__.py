"""Main module."""

from movies_classifier.preprocessing import *

df = load_dataset()
X_train, X_test, y_train, y_test = split_dataset(df)
X_train, y_train = preprocess_dataset(X_train, y_train)

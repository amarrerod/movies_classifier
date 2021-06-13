#!/usr/bin/env python

"""Tests for `movies_classifier` package."""

from movies_classifier.classifier.utilities import (
    load_preprocessed_data,
    save_model,
    load_model,
    save_data_to_files,
)
from movies_classifier.classifier.preprocessing import create_datasets
from sklearn.linear_model import LinearRegression
from os.path import isfile
from os import remove


def test_save_model():
    save_model(LinearRegression(), "linear.pkl")
    assert isfile("linear.pkl")
    remove("linear.pkl")


def test_load_model():
    save_model(LinearRegression(), "linear.pkl")
    assert isfile("linear.pkl")
    model = load_model("linear.pkl")
    assert type(model) == type(LinearRegression())
    remove("linear.pkl")


def test_save_data_to_files():
    data = [[i] for i in range(4)]
    names = [str(i) for i in range(4)]
    save_data_to_files(data, names)
    assert isfile("movies_classifier/data/0.npy")
    assert isfile("movies_classifier/data/1.npy")
    assert isfile("movies_classifier/data/2.npy")
    assert isfile("movies_classifier/data/3.npy")
    remove("movies_classifier/data/0.npy")
    remove("movies_classifier/data/1.npy")
    remove("movies_classifier/data/2.npy")
    remove("movies_classifier/data/3.npy")


def test_load_preprocessed_data():
    X_train, X_test, y_train, y_test = create_datasets()
    save_data_to_files([X_train, X_test, y_train, y_test])
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    assert X_train.shape == (13395, 384)
    assert X_test.shape == (3349, 384)
    assert y_train.shape == (13395,)
    assert y_test.shape == (3349,)

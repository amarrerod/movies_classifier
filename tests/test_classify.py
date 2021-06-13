from movies_classifier.keys import EXPECTED_COLS
from movies_classifier.preprocessing import preprocess_instance
import pandas as pd
import pytest
from movies_classifier.classify import classify


def test_classify_data():
    X = {
        "Year": 1999,
        "Age": "all",
        "Rotten Tomatoes": "87%",
        "Netflix": 0,
        "Hulu": 0,
        "Prime Video": 0,
        "Disney+": 0,
        "Runtime": 200,
        "Genres": "Music, Commedy, Drama",
        "Country": "USA, ESP",
        "Language": "Spanish",
    }
    X = pd.DataFrame(
        X,
        index=[
            "0",
        ],
    )
    X_prepared = preprocess_instance(X)
    assert len(X_prepared) == len(X)
    assert len(X_prepared[0]) == EXPECTED_COLS
    pred = classify(X_prepared)
    assert len(pred) == len(X_prepared)
    assert pred != 0

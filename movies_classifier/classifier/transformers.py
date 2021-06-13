import pandas as pd
from sklearn.preprocessing import QuantileTransformer, MultiLabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin


class Label(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.med = None

    def fit(self, X, y=None):
        self.med = X.median()
        return self

    def transform(self, X):
        """
        Return a pandas Series with the labels preprocessed
        """
        if self.med is None:
            raise RuntimeError("Calling transform without calling fit first")
        X.fillna(self.med, inplace=True)
        X_norm = X.values.reshape(len(X), 1)
        quantile = QuantileTransformer(output_distribution="normal")
        X_norm = quantile.fit_transform(X_norm)
        return X_norm.reshape(len(X_norm))


class Categorical(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns = ["Genres", "Country", "Language"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for column in self.columns:
            X[column].fillna(f"Unknown_{column}", inplace=True)
            X[column] = X[column].map(lambda x: x.split(","))
            X = self.__column_to_binary_encode(X, column)
        return X

    def __column_to_binary_encode(self, df, column):
        mlb = MultiLabelBinarizer(sparse_output=True)
        df = df.join(
            pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(df.pop(column)), index=df.index, columns=mlb.classes_
            )
        )
        return df


class Cat2Numeric(BaseEstimator, TransformerMixin):
    """
    Transformer which converts categorical attributes
    to the correspoding numeric representation
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.loc[X["Age"] == "all", "Age"] = "18+"
        X["Age"] = X["Age"].apply(
            lambda x: float(x[:-1]) if isinstance(x, str) else float(x)
        )
        X["Rotten Tomatoes"] = X["Rotten Tomatoes"].map(
            lambda x: x if type(x) is float else float(x[:-1])
        )
        return X

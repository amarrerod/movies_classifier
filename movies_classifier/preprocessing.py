import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from movies_classifier.transformers import Categorical, Cat2Numeric, Label
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from movies_classifier.utilities import load_model, save_model


_DF_PATH = "./movies_classifier/data/movies.csv"


def _load_dataset(path=_DF_PATH):
    """
    Carga el conjunto de datos desde el fichero
    """
    df = pd.read_csv(path, index_col="ID")
    df.drop(["Unnamed: 0", "Type", "Title", "Directors"], axis=1, inplace=True)
    return df


def _split_dataset(df: pd.DataFrame):
    """
    Divide el conjunto de datos en dos subconjuntos (entrenamiento y testeo)
    Además, crea las variables X e y para cada subconjunto
    Devuelve de la forma X_train, X_test, y_train, y_test
    """
    cleaning_pipeline = Pipeline(
        [("cat_to_num", Cat2Numeric()), ("categorical", Categorical())]
    )
    df = cleaning_pipeline.fit_transform(df)
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
    X_train, y_train = train_set.drop("IMDb", axis=1), train_set["IMDb"].copy()
    X_test, y_test = test_set.drop("IMDb", axis=1), test_set["IMDb"].copy()
    return X_train, X_test, y_train, y_test


def _preprocess_dataset(X_train, X_test, y_train, y_test):
    """
    Realiza el preprocesado de los atributos del dataset
    para poder aplicar distintos algoritmos de ML
    """
    num_attrbs = [
        "Year",
        "Age",
        "Rotten Tomatoes",
        "Netflix",
        "Hulu",
        "Prime Video",
        "Disney+",
        "Runtime",
    ]

    label_trans = Label()
    label_trans.fit(y_train)
    y_train = label_trans.transform(y_train)
    y_test = label_trans.transform(y_test)

    full_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    full_pipeline.fit(X_train)
    X_train_prepared = full_pipeline.transform(X_train)
    X_test_prepared = full_pipeline.transform(X_test)
    # Guardamos los modelos para usar en futuras tareas
    save_model(label_trans, "movies_classifier/models/label_transformer.pkl")
    save_model(full_pipeline, "movies_classifier/models/full_pipe.pkl")
    return X_train_prepared, X_test_prepared, y_train, y_test


def preprocess_instance(X, y=None):
    """
    Realiza el preprocesado de una instancia
    antes de ser pasada al clasificador
    """
    if y:
        label_trans = load_model("movies_classifier/models/label_transformer.pkl")
        y = label_trans.transform(y)

    full_pipeline = load_model("movies_classifier/models/full_pipe.pkl")
    X_prepared = full_pipeline.transform(X)
    return X_prepared, y


def create_datasets():
    """
    Crea los datasets con el preprocesado realizado
    """
    df = _load_dataset()
    X_train, X_test, y_train, y_test = _split_dataset(df)
    X_train, X_test, y_train, y_test = _preprocess_dataset(
        X_train, X_test, y_train, y_test
    )
    return X_train, X_test, y_train, y_test

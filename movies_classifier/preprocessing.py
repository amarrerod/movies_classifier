
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from movies_classifier.transformers import Categorical, Cat2Numeric, Label
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

_DF_PATH = './movies_classifier/data/movies.csv'


def load_dataset(path=_DF_PATH):
    """
        Carga el conjunto de datos desde el fichero
    """
    df = pd.read_csv(path, index_col='ID')
    df.drop(['Unnamed: 0', 'Type', 'Title', 'Directors'], axis=1, inplace=True)
    return df


def split_dataset(df: pd.DataFrame):
    """
        Divide el conjunto de datos en dos subconjuntos (entrenamiento y testeo)
        Además, crea las variables X e y para cada subconjunto
        Devuelve de la forma X_train, X_test, y_train, y_test
    """
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
    X_train, y_train = train_set.drop('IMDb', axis=1), train_set['IMDb'].copy()
    X_test, y_test = test_set.drop('IMDb', axis=1), test_set['IMDb'].copy()
    return X_train, X_test, y_train, y_test


def preprocess_dataset(X_train, X_test, y_train, y_test):
    """
        Realiza el preprocesado de los atributos del dataset
        para poder aplicar distintos algoritmos de ML
    """
    num_attrbs = ["Year", "Age", "Rotten Tomatoes", "Netflix",
                  "Hulu", "Prime Video", "Disney+", "Runtime"]
    cat_attrbs = ["Genres", "Country", "Language"]

    label_trans = Label()
    label_trans.fit(y_train)
    y_train = label_trans.transform(y_train)

    numeric_pipe = Pipeline([
        ('cat_to_num', Cat2Numeric()),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    full_pipeline = ColumnTransformer([
        ('numeric', numeric_pipe, num_attrbs),
        ('categorical', Categorical(), cat_attrbs)
    ])

    X_train_prepared = full_pipeline.fit_transform(X_train)

    y_test = label_trans.transform(y_test)
    X_test_prepated = full_pipeline.transform(X_test)

    return X_train_prepared, X_test_prepated, y_train, y_test


def save_data_to_files(datasets: list):
    """
       Guarda los subconjuntos en el directory de
       ejecución como ficheros NPY
    """
    names = ['X_train', 'X_test', 'y_train', 'y_test']
    for name, data in zip(names, datasets):
        np.save(name, data)


def create_datasets():
    """
       Crea los datasets con el preprocesado realizado
    """
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(df)
    X_train, X_test, y_train, y_test = preprocess_dataset(
        X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test

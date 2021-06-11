
import pandas as pd
from pandas.core.algorithms import quantile
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.prepocessing import QuantileTransformer, StandardScaler, MultiLabelBinarizer
from sklearn.impute import SimpleImputer


_DF_PATH = 'movies.csv'


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
    X_train, y_train = train_set['IMDb'].copy(), train_set.drop('IMDb', axis=1)
    X_test, y_test = test_set['IMDb'].copy(), test_set.drop('IMDb', axis=1)
    return X_train, X_test, y_train, y_test


def preprocess_dataset(X, y):
    labels_med = y.median()
    y.fillna(labels_med, inplace=True)
    y_norm = y.values.reshape(len(y), 1)
    quantile = QuantileTransformer(output_distribution='normal')
    y_norm = quantile.fit_transform(y_norm)
    y = pd.Series(y_norm.reshape(len(y_norm)))

    X.loc[X['Age'] == 'all', 'Age'] = '18+'
    X['Age'] = X['Age'].apply(lambda x: float(
        x[:-1]) if isinstance(x, str) else float(x))
    X['Rotten Tomatoes'] = X['Rotten Tomatoes'].map(
        lambda x: x if type(x) is float else float(x[:-1]))

    imputer = SimpleImputer(strategy='median')
    # OJO Separamos atributos de tipo object
    X_cat = X[["Genres", "Country", "Language"]].copy()
    X = X.drop(["Genres", "Country", "Language"], axis=1)
    # Entrenamos el modelo SimpleImputer para que calcule la mediana de cada atributo
    imputer.fit(X_cat)
    X_num = imputer.transform(X)
    X = pd.DataFrame(X_num, columns=X.columns, index=X.index)

    scaler = StandardScaler()
    cols_to_scale = ['Year', 'Age', 'Rotten Tomatoes', 'Runtime']
    X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

    X_cat['Genres'] = __split_column_to_list(X_cat['Genres'])
    X_cat['Country'] = __split_column_to_list(X_cat['Country'])
    X_cat['Language'] = __split_column_to_list(X_cat['Language'])
    X_cat = __column_to_binary_encode(X_cat, 'Genres')
    X_cat = __column_to_binary_encode(X_cat, 'Country')
    X_cat = __column_to_binary_encode(X_cat, 'Language')

    X.join(X_cat)
    return X, y


def __split_column_to_list(column):
    return column.map(lambda x: x.split(','))


def __column_to_binary_encode(df, column):
    mlb = MultiLabelBinarizer(sparse_output=True)
    df = df.join(pd.DataFrame.sparse.from_spmatrix(
                 mlb.fit_transform(column),
                 index=df.index,
                 columns=mlb.classes_))
    return df

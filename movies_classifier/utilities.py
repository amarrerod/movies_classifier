import joblib
import numpy as np


def save_model(model, model_name):
    """
    Guarda el modelo que hemos creado
    dentro del fichero espeficado
    """
    joblib.dump(model, model_name)


def load_model(model_name):
    """
    Carga el modelo con el nombre recibido
    """
    return joblib.load(model_name)


def save_data_to_files(datasets: list):
    """
    Guarda los subconjuntos en el directory de
    ejecución como ficheros NPY
    """
    root_dir = "movies_classifier/data/"
    names = ["X_train", "X_test", "y_train", "y_test"]
    for name, data in zip(names, datasets):
        full_name = f"{root_dir}{name}"
        np.save(full_name, data)


def load_preprocessed_data():
    """
    Carga los datasets guardados previamente dentro del
    directorio data
    """
    root_dir = "movies_classifier/data/"
    names = ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy"]
    try:
        datasets = [np.load(f"{root_dir}{name}") for name in names]
    except FileNotFoundError as err:
        print(
            f"File: {err.filename} not found. Try to run the program in preprocessing mode first"
        )
        exit(1)

    X_train, X_test, y_train, y_test = datasets
    return X_train, X_test, y_train, y_test

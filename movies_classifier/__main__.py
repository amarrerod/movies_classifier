"""Main module."""

from movies_classifier.utilities import save_data_to_files, load_preprocessed_data
from movies_classifier.preprocessing import create_datasets
from movies_classifier.training import create_and_train_model
from argparse import ArgumentParser

parser = ArgumentParser("Movies classifier")
parser.add_argument("mode", choices=["training", "preprocessing", "deploy"])

args = parser.parse_args()
if args.mode == "training":
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    forest = create_and_train_model(X_train, y_train)

elif args.mode == "preprocessing":
    X_train, X_test, y_train, y_test = create_datasets()
    save_data_to_files([X_train, X_test, y_train, y_test])

else:
    pass

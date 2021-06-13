"""Main module."""


from movies_classifier.classifier.utilities import (
    save_data_to_files,
    load_preprocessed_data,
)
from movies_classifier.classifier.preprocessing import create_datasets
from movies_classifier.classifier.training import create_and_train_model
from argparse import ArgumentParser
from movies_classifier.app import app

parser = ArgumentParser("Movies classifier")
parser.add_argument("mode", choices=["training", "preprocessing", "run"])
parser.add_argument("--hostname", help="Hostname to run the app")
parser.add_argument("--port", help="Port to run the app")

args = parser.parse_args()
if args.mode == "training":
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    forest = create_and_train_model(X_train, y_train)

elif args.mode == "preprocessing":
    X_train, X_test, y_train, y_test = create_datasets()
    save_data_to_files([X_train, X_test, y_train, y_test])

elif args.mode == "run":
    if args.hostname is None or args.port is None:
        raise RuntimeError(f"Hostname or port not set.\n{parser.print_help()}")
        exit(1)
    app.run(host=args.hostname, port=args.port)

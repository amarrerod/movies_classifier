"""Main module."""


from movies_classifier.classifier.utilities import (
    save_data_to_files,
    load_preprocessed_data,
)
from movies_classifier.classifier.preprocessing import create_datasets
from movies_classifier.classifier.training import create_and_train_model
from argparse import ArgumentParser
from movies_classifier.app import app
from os import environ

parser = ArgumentParser("Movies classifier")
parser.add_argument("mode", choices=["training", "preprocessing", "run"])
parser.add_argument("--hostname", help="Hostname to run the app")
parser.add_argument("--port", help="Port to run the app")
parser.add_argument("--heroku", help="Run the app in Heroku")

args = parser.parse_args()
if args.mode == "training":
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    forest = create_and_train_model(X_train, y_train)

elif args.mode == "preprocessing":
    X_train, X_test, y_train, y_test = create_datasets()
    save_data_to_files([X_train, X_test, y_train, y_test])

elif args.mode == "run":
    hostname = ""
    port = ""
    if args.heroku is not None:
        hostname = "0.0.0.0"
        port = int(environ.get("PORT", 5000))

    elif args.hostname is not None or args.port is not None:
        hostname = args.hostname
        port = int(args.port)
    else:
        raise RuntimeError(f"Hostname or port not set.\n{parser.print_help()}")
        exit(1)
    app.run(host=hostname, port=port)

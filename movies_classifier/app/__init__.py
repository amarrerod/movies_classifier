from flask import Flask, request
from movies_classifier.classifier.classify import classify
from movies_classifier.classifier.preprocessing import preprocess_instance_from_dict

app = Flask(__name__)


@app.route("/")
def hello():
    return {"status": 200, "msg": "hello world!"}


@app.route("/classify", methods=["POST"])
def classify_instance():
    data = request.get_json()
    try:
        X = preprocess_instance_from_dict(data)
    except RuntimeError as err:
        return str(f"Error in the data. {err}")

    pred = classify(X)
    return str(pred)

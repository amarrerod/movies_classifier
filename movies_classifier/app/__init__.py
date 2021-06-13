from flask import Flask

app = Flask(__name__)


@app.route("/")
def hello():
    return {"status": 200, "msg": "hello world!"}

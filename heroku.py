from movies_classifier.app import app
from os import environ

hostname = "0.0.0.0"
port = int(environ.get("PORT", 5000))
app.run(host=hostname, port=port)

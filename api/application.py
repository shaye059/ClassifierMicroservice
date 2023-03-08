from flask import Flask, jsonify, request
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.neural_network import MLPClassifier
from db import ConnectionPool
import pickle

application = Flask(__name__)
connection_pool = ConnectionPool(size=5)


@application.route("/health/")
def health():
    return jsonify({"message": "Ok"})


@application.route("/models/", methods=["POST"])
def create_model():
    # Extract parameters from the request body
    model_name = request.json["model"]
    params = request.json["params"]
    d = request.json["d"]
    n_classes = request.json["n_classes"]

    # Create the classifier instance
    if model_name == "SGDClassifier":
        clf = SGDClassifier(**params)
    elif model_name == "CategoricalNB":
        clf = CategoricalNB(**params)
    elif model_name == "MLPClassifier":
        clf = MLPClassifier(**params)
    else:
        return jsonify({"error": "Invalid model name"})

    # Serialize the classifier to binary
    # TODO: try joblib serializer here instead
    # import joblib
    # from io import BytesIO
    # import base64
    # with BytesIO() as tmp_bytes:
    #     joblib.dump({"test": "test"}, tmp_bytes)
    #     bytes_obj = tmp_bytes.getvalue()
    #     base64_obj = base64.b64encode(bytes_obj)
    clf_bytes = pickle.dumps(clf)

    # Insert the serialized classifier into the database
    conn = connection_pool.get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO models (model_name, clf_bytes, d, n_classes) VALUES (%s, %s, %s, %s)",
        (model_name, clf_bytes, d, n_classes),
    )
    model_id = cur.lastrowid
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({"id": model_id})


@application.route("/models/<int:model_id>/", methods=["GET"])
def get_model(model_id):
    # Retrieve the model from the database
    conn = connection_pool.get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT model_name, clf_bytes, d, n_classes, n_trained FROM models WHERE id=%s",
        (model_id,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()

    if row is None:
        return jsonify({"error": "Model not found"})

    model_name, clf_bytes, d, n_classes, n_trained = row

    # Deserialize the classifier from binary
    clf = pickle.loads(clf_bytes)

    # Return the classifier metadata
    return jsonify(
        {
            "model": model_name,
            "params": clf.get_params(),
            "d": d,
            "n_classes": n_classes,
            "n_trained": n_trained,
        }
    )

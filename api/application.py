from flask import Flask, jsonify, request
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.neural_network import MLPClassifier
from db import ConnectionPool, setup_db
import pickle
import logging, logging.handlers
import base64
import json

application = Flask(__name__)
connection_pool = ConnectionPool(size=10)
setup_db(connection_pool)

formatter = logging.Formatter(  # pylint: disable=invalid-name
    '%(asctime)s %(levelname)s %(process)d ---- %(threadName)s  '
    '%(module)s : %(funcName)s {%(pathname)s:%(lineno)d} %(message)s','%Y-%m-%dT%H:%M:%SZ')

handler = logging.StreamHandler()
handler.setFormatter(formatter)

application.logger.setLevel(logging.DEBUG)
application.logger.addHandler(handler)


@application.route("/health/")
def health():
    return jsonify({"status": "ok"})

@application.route('/models/', methods=['POST'])
def create_model():
    # Extract parameters from the request body
    model = request.json['model']
    params = request.json['params']
    d = request.json['d']
    n_classes = request.json['n_classes']

    # Create the classifier object
    clf = None
    if model == 'SGDClassifier':
        clf = SGDClassifier(**params)
    elif model == 'CategoricalNB':
        clf = CategoricalNB(**params)
    elif model == 'MLPClassifier':
        clf = MLPClassifier(**params)
    if clf is None:
        return jsonify({'error': 'Invalid model'}), 400

    # Serialize the classifier to binary
    conn = connection_pool.get_connection()
    cur = conn.cursor()
    cur.execute('INSERT INTO db.classifiers (model, params, d, n_classes, clf_bytes) VALUES (%s, %s, %s, %s, %s)', (model, str(params), d, n_classes, pickle.dumps(clf)))
    conn.commit()
    model_id = cur.lastrowid
    return jsonify({'id': model_id})


@application.route("/models/<int:model_id>/", methods=["GET"])
def get_model(model_id):
    # Retrieve the model from the database
    conn = connection_pool.get_connection()
    cur = conn.cursor()
    query = "SELECT model, params, d, n_classes, n_trained FROM db.classifiers WHERE id = %s"
    cur.execute(query, (model_id,))
    row = cur.fetchone()

    cur.close()
    conn.close()

    # Check if exists, if not return error
    if row is None:
        return jsonify({"error": "Model not found"}), 404

    # Return the classifier metadata
    model_name, params, d, n_classes, n_trained = row
    return jsonify(
        {
            "model": model_name,
            "params": eval(params),
            "d": d,
            "n_classes": n_classes,
            "n_trained": n_trained,
        }
    )

@application.route('/models/<int:model_id>/train/', methods=['POST'])
def train_model(model_id):
    # Check inputs
    x = request.json.get('x')
    y = request.json.get('y')
    application.logger.debug(str(model_id) + " " + str(x) + str(y))
    if x is None or y is None:
        application.logger.error("Bad input!!!!")
        return jsonify({'error': 'Invalid input data'}), 400

    conn = connection_pool.get_connection()
    cur = conn.cursor()
    query = "SELECT clf_bytes, n_trained, d, n_classes FROM db.classifiers WHERE id = %s"
    cur.execute(query, (model_id,))
    result = cur.fetchone()

    if result is None:
        cur.close()
        conn.close()
        return jsonify({'error': 'Model not found'}), 404

    model_data, n_trained, d, n_classes = result
    x = np.array(x)
    if len(x) != d or y > n_classes-1:
        application.logger.warning("Dimension of X incorrect")
        return jsonify({'error': 'Invalid input data'}), 400

    application.logger.error(f"TRAINING model {model_id} on {x} with expected value {y} and {n_classes} classes.")
    application.logger.debug("Training on classes: " + str(list(range(0, n_classes))))

    # If all is good unpickle the model
    clf = pickle.loads(model_data)
    clf.partial_fit([x], [y], classes=list(range(0, n_classes)))
    application.logger.debug("Model training successful!!!!")

    # Update the trained model in the database
    cur.execute("UPDATE db.classifiers SET n_trained = %s, clf_bytes = %s WHERE id = %s", (n_trained + 1, pickle.dumps(clf), model_id))
    conn.commit()
    cur.close()
    conn.close()
    application.logger.debug("Training model saved successfully!!!!")
    return jsonify({'message': 'Model trained successfully'})


@application.route('/models/<int:model_id>/predict/', methods=['GET'])
def predict(model_id):
    # Load and decode query string
    x = request.args.get('x')
    x = json.loads(base64.urlsafe_b64decode(x + '==').decode())
    application.logger.debug("Query string: " + str(x))
    if x is None:
        return jsonify({'error': 'Invalid input data'}), 400

    conn = connection_pool.get_connection()
    cur = conn.cursor()
    query = "SELECT clf_bytes, d FROM db.classifiers WHERE id = %s"
    cur.execute(query, (model_id,))
    result = cur.fetchone()
    cur.close()
    conn.close()

    if result is None:
        return jsonify({'error': 'Model not found'}), 404

    model_data, d = result

    if len(x) != d:
        application.logger.warning("Dimension of X incorrect")
        return jsonify({'error': 'Invalid input data'}), 400

    clf = pickle.loads(model_data)
    y = clf.predict([x])
    application.logger.debug("Prediction is:" + str(y))

    return jsonify({'y': int(y[0])})

def normalize(x):
    if len(x) == 0:
        return np.array([])
    if len(x) == 1:
        return np.array([1.0])
    return (x - np.min(x)) / (np.max(x) - np.min(x))

@application.route('/models/')
def get_models():
    # Connect to the database
    conn = connection_pool.get_connection()
    cursor = conn.cursor()

    # Query the database to get the model statistics
    cursor.execute('SELECT id, model, n_trained FROM db.classifiers')
    rows = cursor.fetchall()
    application.logger.debug("###### Models fetched #######")
    application.logger.debug(rows)


    # Initialize dictionaries to keep track of the number of times each model was trained
    sgd_counts = {}
    cat_counts = {}
    mlp_counts = {}

    # Count the number of times each model was trained
    for row in rows:
        id = row[0]
        model = row[1]
        n_trained = row[2]
        if model == 'SGDClassifier':
            sgd_counts[id] = n_trained
        elif model == 'CategoricalNB':
            cat_counts[id] = n_trained
        elif model == 'MLPClassifier':
            mlp_counts[id] = n_trained

    application.logger.debug("SGD COUNTS: " + str(sgd_counts) + str(list(sgd_counts.values())))
    application.logger.debug("CAT COUNTS: " + str(cat_counts)+ str(list(cat_counts.values())))
    application.logger.debug("MLP COUNTS: " + str(mlp_counts)+ str(list(mlp_counts.values())))

    # Compute the scores for each model type
    sgd_scores = normalize(np.array(list(sgd_counts.values())))
    cat_scores = normalize(np.array(list(cat_counts.values())))
    mlp_scores = normalize(np.array(list(mlp_counts.values())))

    # Re-assign the scores to their model IDs
    sgd_scores_dict = dict(zip(sgd_counts.keys(), sgd_scores.tolist()))
    cat_scores_dict = dict(zip(cat_counts.keys(), cat_scores.tolist()))
    mlp_scores_dict = dict(zip(mlp_counts.keys(), mlp_scores.tolist()))

    application.logger.debug("SGD dict: " + str(sgd_scores_dict))
    application.logger.debug(cat_scores_dict)
    application.logger.debug(mlp_scores_dict)

    # Query the database again to get the full model statistics
    cursor.execute('SELECT id, model, n_trained FROM classifiers')
    rows = cursor.fetchall()

    # Initialize a list to store the output
    output = []

    # Compute the score for each model instance and add it to the output
    for row in rows:
        id = row[0]
        model = row[1]
        n_trained = row[2]
        if model == 'SGDClassifier':
            score = sgd_scores_dict[id]
        elif model == 'CategoricalNB':
            score = cat_scores_dict[id]
        elif model == 'MLPClassifier':
            score = mlp_scores_dict[id]
        output.append({
            'id': id,
            'model': model,
            'n_trained': n_trained,
            'training_score': score
        })

    # Close the database connection
    conn.close()

    # Return the output as a JSON object
    return jsonify({'models': output})

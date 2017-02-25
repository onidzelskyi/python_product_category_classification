"""Flask REST service endpoints."""
from flask import Flask
from flask import json
from flask import jsonify
from flask import request

import pandas as pd

from prod_classify.core import ProductClassifyModel


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp'

model = ProductClassifyModel()


def fit() -> str:
    """Train model using by train dataset in input CSV-file.
    @:return JSON object with response body."""

    df = pd.read_csv(request.file)

    # Send data to the model and get back predicted results
    response = model.fit(df)

    return jsonify(response)


def predict() -> str:
    """Send product items to the model to predict theirs categories.
    @:return JSON object with response body."""

    data = json.loads(request.data)
    # Validate input data format
    assert 'products' in data

    # Send data to the model and get back predicted results
    response = model.predict(data['products'])

    return jsonify(response)


# Match routes with handlers
app.add_url_rule('/predict', 'predict', predict, methods=['POST'])
app.add_url_rule('/fit', 'fit', fit, methods=['POST'])

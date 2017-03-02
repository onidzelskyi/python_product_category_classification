"""Flask REST service endpoints."""
from flask import Flask
from flask import jsonify
from flask import request
from flask import send_from_directory
from werkzeug.utils import secure_filename

from datetime import datetime

from werkzeug.exceptions import BadRequest

import pandas as pd

import os

from prod_classify.core import ProductClassifyModel, logger, upload_folder


msg_data_not_found = 'Input CSV file or JSON data not found.'
msg_bad_data_format = 'Bad file format.'
msg_train_success = 'Model trained successfully.'
msg_train_fail = 'Model train fail.'
msg_predict_invalid_input = 'Invalid input data'
msg_predict_model_not_ready = 'Model not ready.'
msg_fail_load_model = 'Model cannot be loaded from file.'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '../{}'.format(upload_folder)

model = ProductClassifyModel()


def fit() -> str:
    """Train model using by train dataset in input CSV-file.
    @:return JSON object with response body."""
    # Validate input file
    if request.files:
        df = pd.read_csv(request.files['file'])

    elif request.data:
        try:
            df = pd.read_json(request.data)
        except ValueError as err:
            logger.debug(err)
            raise BadRequest(msg_data_not_found)

    else:
        logger.debug(msg_data_not_found)
        raise BadRequest(msg_data_not_found)

    # Check input file format
    if not {'label', 'text'}.issubset(df.columns):
        raise BadRequest(msg_bad_data_format)

    # Send train data to the model to train model and check operation result
    if model.fit(df):
        response_data = msg_train_success
        response_op_status = 'success'
        response_http_status = 200

    else:
        response_data = msg_train_fail
        response_op_status = 'failure'
        response_http_status = 400

    return jsonify(dict(status=response_op_status, date=datetime.today(), data=response_data)), response_http_status


def predict() -> str:
    """Send product items to the model to predict theirs categories.
    @:return JSON object with response body."""
    data = request.get_json()

    # Validate input data format
    if not data or 'products' not in data:
        raise BadRequest(msg_predict_invalid_input)

    # Send data to the model and get back predicted results
    try:
        response = model.predict(data['products'])
        response_data = response
        response_op_status = 'success'
        response_http_status = 200
    except NotImplementedError as err:
        logger.debug(err)
        response_data = msg_predict_model_not_ready
        response_op_status = 'failure'
        response_http_status = 400

    return jsonify(dict(status=response_op_status, date=datetime.today(), data=response_data)), response_http_status


def dump() -> str:
    """Download file with trained model
    @:return pickle file with model to download."""
    directory = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'])
    if 'file' in request.args:
        file_name = request.args['file']
    else:
        file_name = 'model_{}'.format(datetime.today().timestamp())

    model.dump_model('{}/{}'.format(directory, file_name))

    return send_from_directory(directory=directory, filename=file_name)


def load() -> str:
    """Load file with trained model
    @:return Status of operation."""
    if not request.files:
        raise BadRequest(msg_predict_invalid_input)

    # Save file on disk
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    try:
        response_data = model.load_model(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        response_op_status = 'success'
        response_http_status = 200
    except ValueError as err:
        logger.debug(err)
        response_data = msg_fail_load_model
        response_op_status = 'failure'
        response_http_status = 400

    return jsonify(dict(status=response_op_status, date=datetime.today(), data=response_data)), response_http_status


def statistics() -> str:
    """Get statistics for trained model
    @:return statistics of model."""
    if request.files:
        try:
            df = pd.read_csv(request.files['file'])
        except UnicodeDecodeError as err:
            logger.debug(err)
            raise BadRequest(msg_predict_invalid_input)

    elif request.data:
        try:
            df = pd.read_json(request.data)
        except ValueError as err:
            logger.debug(err)
            raise BadRequest(msg_data_not_found)

    else:
        logger.debug(msg_data_not_found)
        raise BadRequest(msg_data_not_found)

    # Check input file format
    if not {'label', 'text'}.issubset(df.columns):
        raise BadRequest(msg_bad_data_format)

    # Retrieve statistics
    try:
        response = model.get_statistics(df)
        response_data = response
        response_op_status = 'success'
        response_http_status = 200
    except NotImplementedError as err:
        logger.debug(err)
        response = msg_predict_model_not_ready
        response_data = response
        response_op_status = 'failure'
        response_http_status = 400

    return jsonify(dict(status=response_op_status, date=datetime.today(), data=response_data)), response_http_status


# Match routes with handlers
app.add_url_rule('/predict', 'predict', predict, methods=['POST'])
app.add_url_rule('/fit', 'fit', fit, methods=['POST'])
app.add_url_rule('/dump', 'dump', dump, methods=['GET'])
app.add_url_rule('/load', 'load', load, methods=['POST'])
app.add_url_rule('/statistics', 'statistics', statistics, methods=['POST'])

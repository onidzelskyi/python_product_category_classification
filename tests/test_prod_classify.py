"""Test product category classification model."""
import os
import mock

from flask_testing import TestCase

from flask import url_for
from flask import json

from prod_classify.endpoints import app


data = {"products": {"101": "Awesome product one", "102": "Awesome product two"}}


class ProductClassifyTest(TestCase):
    """Product classification test class."""
    train_csv_file = '{}/../resources/train_set.csv'.format(os.path.dirname(__file__))
    train_json_file = '{}/../resources/train_set.json'.format(os.path.dirname(__file__))
    test_csv_file = '{}/../resources/test_set.csv'.format(os.path.dirname(__file__))
    bad_json_content_file = '{}/../resources/bad_json_content.json'.format(os.path.dirname(__file__))
    pickle_model_file = '{}/../resources/model.pickle'.format(os.path.dirname(__file__))

    def create_app(self):
        """Create and return Flask app."""
        return app

    def test_fit_from_csv_file_success(self):
        """Test predict."""
        response = self.client.post(url_for('fit'),
                                    data={'file': (self.train_csv_file, 'train_set.csv')})

        self.assert200(response)

    def test_fit_from_json_data_success(self):
        """Test predict."""
        with open(self.train_json_file) as fp:
            data = fp.read()
        response = self.client.post(url_for('fit'),
                                    data=data)

        self.assert200(response)

    def test_fit_no_file(self):
        """Test predict."""
        response = self.client.post(url_for('fit'))

        self.assert400(response, 'Invalid input data or file not found.')

    def test_fit_bad_csv_file(self):
        """Test predict."""
        response = self.client.post(url_for('fit'),
                                    data={'file': (self.train_json_file, 'train_set.json')})

        self.assert400(response, 'Bad file format.')

    def test_fit_bad_csv_file_content(self):
        """Test predict."""
        response = self.client.post(url_for('fit'),
                                    data={'file': (self.train_json_file, 'bad_csv_content.csv')})

        self.assert400(response, 'Bad CSV file format.')

    def test_fit_bad_json_data_content(self):
        """Test predict."""
        with open(self.bad_json_content_file) as fp:
            data = fp.read()
        response = self.client.post(url_for('fit'),
                                    data=data)

        self.assert400(response, 'Bad CSV file format.')

    def test_predict_success(self):
        """Test predict."""
        response = self.client.post(url_for('fit'),
                                    data={'file': (self.train_csv_file, 'train_set.csv')})

        self.assert200(response)

        response = self.client.post(url_for('predict'),
                                    data=json.dumps(data),
                                    headers={'Content-Type': 'application/json'})

        self.assert200(response)

    def test_predict_empty_input(self):
        """Test predict."""
        response = self.client.post(url_for('predict'))

        self.assert400(response, 'Invalid input data')

    def test_predict_bad_json(self):
        """Test predict."""
        response = self.client.post(url_for('predict'), data=json.dumps(dict()))

        self.assert400(response, 'Invalid input data')

    def test_predict_model_not_ready(self):
        """Test predict."""
        response = self.client.post(url_for('predict'), data=json.dumps(dict()))

        self.assert400(response, 'Model not ready.')

    @mock.patch('__main__.open')
    def test_dump_model_default(self, mock_open):
        """Test dump pickle file with default file name."""
        response = self.client.get(url_for('dump'))

        self.assert200(response)

    @mock.patch('__main__.open')
    def test_dump_model_to_given_file(self, mock_open):
        """Test dump pickle file with given file name."""
        response = self.client.get(url_for('dump'), query_string=dict(file='test.txt'))

        self.assert200(response)

    def test_load_model_success(self):
        """Test dump pickle file with given file name."""
        response = self.client.post(url_for('load'),
                                    data={'file': (self.pickle_model_file, 'model.pickle')})

        self.assert200(response)

    def test_get_statistics_model_not_ready(self):
        """Test dump pickle file with given file name."""
        response = self.client.post(url_for('statistics'),
                                    data={'file': (self.pickle_model_file, 'model.pickle')})

        self.assert400(response, 'Model not ready.')

    def test_get_statistics_bad_csv_file_format(self):
        """Test dump pickle file with given file name."""
        response = self.client.post(url_for('statistics'),
                                    data={'file': (self.pickle_model_file, 'model.pickle')})

        self.assert400(response, 'Model not ready.')

    def test_get_statistics_from_csv_file_success(self):
        """Test dump pickle file with given file name."""
        response = self.client.post(url_for('statistics'),
                                    data={'file': (self.test_csv_file, 'test_set.csv')})

        self.assert200(response)

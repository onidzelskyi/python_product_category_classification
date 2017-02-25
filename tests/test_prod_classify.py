"""Test product category classification model."""

from flask_testing import TestCase

from flask import url_for
from flask import json

from prod_classify.endpoints import app


data = {"products": {"101": "Awesome product one", "102": "Awesome product two"}}


class ProductClassifyTest(TestCase):
    """Product classification test class."""

    def create_app(self):
        """Create and return Flask app."""
        return app

    def test_fit(self):
        """Test predict."""
        response = self.client.post(url_for('fit'), data=json.dumps(data))

        self.assert200(response)

    def test_predict(self):
        """Test predict."""
        response = self.client.post(url_for('predict'), data=json.dumps(data))

        self.assert200(response)


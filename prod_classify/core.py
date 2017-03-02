"""Core of product classification model."""
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

import pickle
import logging


# Logger
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='app.log',
                    filemode='w')
logger = logging.getLogger(__name__)

upload_folder = 'resources'


class ProductClassifyModel:
    """Product category classification model."""

    def __init__(self):
        """Initialization of model."""
        self.vectorizer = TfidfVectorizer(sublinear_tf=True,
                                          max_df=0.5,
                                          stop_words='english')
        self.clf = MultinomialNB(alpha=.01)
        self.ready = False

    def fit(self, train_data: list) -> bool:
        """Train model.
        @:arg train_data - list of tuples in (text, label) format.
        @:return bool flag operation status. True if success False else."""
        # Extract text data from train dataset
        x = train_data.text.values

        # Extract labels from train dataset
        y = train_data.label.values

        # Convert word2vec
        try:
            x = self.vectorizer.fit_transform(x)
        except ValueError as err:
            logger.debug(err)
            raise NotImplementedError('Bad CSV file format.')

        # Train model
        self.clf.fit(x, y)

        # Set flag that model is ready for use
        self.ready = True

        return self.ready

    def predict(self, product_items: dict) -> list:
        """Predict product category.
        @:arg data - list of product items.
        @:return list of product items with theirs predicted categories."""
        # Check if model loaded, trained, and ready for use.
        if not self.ready:
            raise NotImplementedError('Model not ready.')

        data_scores = self.vectorizer.transform(product_items.values())
        prob = self.clf.predict_proba(data_scores)
        result = {x: {k: v for k, v in zip(self.clf.classes_.tolist(), prob.tolist()[0])} for x in product_items}

        return result

    def dump_model(self, file_name: str) -> str:
        """Dump trained model to the pickle file.
        @:arg file_name - pickle file name with model to be dumped.
        @:return file_name - pickle file name with model to be download."""
        assert file_name != None, ValueError('No file name given.')

        with open(file_name, 'wb') as fp:
            pickle.dump(self.clf, fp)
            pickle.dump(self.vectorizer, fp)

        return file_name

    def load_model(self, file_name: str):
        """Load trained model from pickle file.
        @:arg file_name - name of pickle file to be upload."""
        try:
            with open(file_name, 'rb') as fp:
                self.clf = pickle.load(fp)
                self.vectorizer = pickle.load(fp)
        except ValueError as err:
            logger.debug(err)
            raise ValueError('Model cannot be loaded from given file.')

        return 'Successfully loaded model from file.'

    def get_statistics(self, test_data: dict) -> str:
        """Get statistics of trained model.
        @:return string of statistics"""
        if not self.ready:
            raise NotImplementedError('Model not ready.')

        x_test = self.vectorizer.transform(test_data.text.values)
        y_test = test_data.label.values

        pred = self.clf.predict(x_test)

        accuracy = metrics.accuracy_score(y_test, pred)
        f1_score = metrics.f1_score(y_test, pred, average='micro')
        precision = metrics.precision_score(y_test, pred, average='micro')
        recall = metrics.recall_score(y_test, pred, average='micro')

        return dict(accuracy=accuracy, f1_score=f1_score, precision=precision, recall=recall)

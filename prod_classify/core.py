"""Core of product classification model."""
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


class ProductClassifyModel:
    """Product category classification model."""

    def __init__(self):
        """Initialization of model."""
        self.vectorizer = TfidfVectorizer(sublinear_tf=True,
                                          max_df=0.5,
                                          stop_words='english')
        self.clf = MultinomialNB(alpha=.01)
        self.ready = False

    def fit(self, train_data: list) -> None:
        """Train model.
        @:arg train_data - list of tuples in (text, label) format."""
        # Extract text data from train dataset
        x = [x[0] for x in train_data]

        # Extract labels from train dataset
        y = [x[1] for x in train_data]

        # Convert word2vec
        x = self.vectorizer.fit_transform(x)

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
        assert self.ready == True, NotImplementedError('Model not ready.')

        data_scores = self.vectorizer.transform(product_items.values())
        prob = self.clf.predict_proba(data_scores)
        result = {x: {k: v for k, v in zip(self.clf.classes_.tolist(), prob.tolist()[0])} for x in product_items}

        return result

    def dump_model(self):
        """Dump trained model to the pickle file."""
        raise NotImplementedError('Not implemented yet.')

    def load_model(self):
        """Load trained model from pickle file."""
        raise NotImplementedError('Not implemented yet.')

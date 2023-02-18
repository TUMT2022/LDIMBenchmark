import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances


class LDIMMethodScikitAdapter:
    def __init__(self, method):
        self.method = method

    def get_params(self, deep=True):
        return self.method.hyperparameters

    def set_params(self, **params):
        # self.method.hyperparameters.update(params)
        return self

    def fit(self, X, y):
        self.method.train(X)

    def predict(self, X):
        return self.method.detect(X)

    def score(self, X, y):
        return 0


class ScikitLearnEstimatorAdapter(BaseEstimator, ClassifierMixin):
    def __init__(self, demo_param="demo"):
        self.demo_param = demo_param

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]


param_grid = [
    {"demo_param": [1, 10, 100, 1000]},
    {"demo_param": [1, 10, 100, 1000]},
]


from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

iris = datasets.load_iris()
clf = GridSearchCV(ScikitLearnEstimatorAdapter(), param_grid)
clf.fit([{}], [{}], cv=1)
print(clf.cv_results_)

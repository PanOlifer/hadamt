import numpy as np
from ..models.clf_base import train_clf


def train_baseline(X, y):
    clf = train_clf(X, y)
    acc = (clf.predict(X) == y).mean()
    return clf, acc

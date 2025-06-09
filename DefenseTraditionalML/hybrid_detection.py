import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, roc_auc_score


class HybridDetector:
    """Wrapper around IsolationForest for anomaly scoring."""

    def __init__(self, **kwargs):
        self.iforest = IsolationForest(**kwargs)

    def fit(self, X):
        self.iforest.fit(X)

    def score_samples(self, X):
        # higher score -> more anomalous
        return -self.iforest.score_samples(X)


def combine_features(other, errors):
    errors = np.asarray(errors).reshape(-1, 1)
    other = np.asarray(other)
    if other.ndim == 1:
        other = other.reshape(-1, 1)
    return np.hstack([other, errors])


def evaluate(scores, y_true, threshold=None):
    scores = np.asarray(scores)
    if threshold is None:
        threshold = np.percentile(scores, 95)
    y_pred = (scores >= threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, scores)
    return precision, recall, roc_auc

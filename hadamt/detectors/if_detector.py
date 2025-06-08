from sklearn.ensemble import IsolationForest


def fit_if(X):
    clf = IsolationForest(random_state=42)
    clf.fit(X)
    return clf


def score_if(clf, X):
    return -clf.score_samples(X)

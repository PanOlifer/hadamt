from sklearn.linear_model import LogisticRegression


def train_clf(X, y):
    clf = LogisticRegression(max_iter=200)
    clf.fit(X, y)
    return clf


def predict(clf, X):
    return clf.predict(X)

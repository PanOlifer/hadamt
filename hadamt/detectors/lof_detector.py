from sklearn.neighbors import LocalOutlierFactor


def fit_lof(X):
    lof = LocalOutlierFactor(novelty=True)
    lof.fit(X)
    return lof


def score_lof(model, X):
    return -model.score_samples(X)

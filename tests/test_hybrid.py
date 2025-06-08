import numpy as np
from hadamt.detectors.hybrid import HybridDetector


def test_hybrid_score():
    X_tab = np.random.rand(30, 10)
    X_img = np.random.rand(30, 3, 32, 32)
    y = np.random.randint(0, 2, size=30)
    hd = HybridDetector()
    hd.fit(X_tab, X_img, y)
    scores = hd.score(X_tab, X_img)
    assert scores.mean() > 0

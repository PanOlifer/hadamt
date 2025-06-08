import numpy as np
from hadamt.pipelines.defense import run_defense


def test_defense_pipeline():
    n = 2000
    X = np.random.rand(n, 10)
    X_img = np.random.rand(n, 3, 32, 32)
    y = np.zeros(n)
    poison_idx = np.random.choice(n, int(0.3 * n), replace=False)
    y[poison_idx] = 1
    X_f, y_f, scores = run_defense(X, X_img, y, thr_high=0.5)
    removed_poison = (y[~(scores <= 0.5)] == 1).mean()
    removed_clean = (y[~(scores <= 0.5)] == 0).mean()
    assert removed_poison >= 0.6
    assert removed_clean <= 0.15

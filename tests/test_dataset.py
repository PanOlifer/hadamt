import numpy as np
from hadamt.datasets.synthetic_ds import save_synthetic, load_synthetic


def test_synthetic(tmp_path):
    X = np.random.rand(10, 5)
    path = tmp_path / "synth.csv"
    save_synthetic(X, path)
    X2 = load_synthetic(path)
    assert X2.shape == X.shape

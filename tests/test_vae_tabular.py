import numpy as np
from hadamt.models.vae_tabular import train_vae, reconstruction_error


def test_vae_tabular():
    X = np.random.rand(20, 10)
    model = train_vae(X, epochs=1)
    err = reconstruction_error(model, X)
    assert err.mean() >= 0

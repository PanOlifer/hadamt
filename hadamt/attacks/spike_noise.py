import numpy as np


def add_spike_noise(data, ratio=0.05):
    n = len(data)
    idx = np.random.choice(n, int(ratio * n), replace=False)
    data[idx] *= 1.5
    return data

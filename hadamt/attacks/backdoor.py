import numpy as np


def add_backdoor(images, ratio=0.03):
    n = images.shape[0]
    idx = np.random.choice(n, int(ratio * n), replace=False)
    for i in idx:
        images[i, -2:, -2:, :] = 255
    return images

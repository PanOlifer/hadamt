import numpy as np


def label_flip(images, labels, ratio=0.07):
    n = len(labels)
    idx = np.random.choice(n, int(ratio * n), replace=False)
    for i in idx:
        labels[i] = np.random.randint(0, max(labels)+1)
    return images, labels

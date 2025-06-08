import torch
from torchvision import datasets, transforms
import numpy as np
from pathlib import Path
from ..attacks.label_flip import label_flip
from ..attacks.backdoor import add_backdoor


def load_cifar100(root="data/cifar100", train=True):
    Path(root).mkdir(parents=True, exist_ok=True)
    ds = datasets.CIFAR100(root=root, train=train, download=True,
                           transform=transforms.ToTensor())
    images = ds.data.astype(np.float32)
    labels = np.array(ds.targets)
    return images, labels


def poisoned_cifar(images, labels):
    images, labels = label_flip(images, labels, ratio=0.07)
    images = add_backdoor(images, ratio=0.03)
    return images, labels

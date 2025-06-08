import numpy as np
import pandas as pd
from pathlib import Path


def save_synthetic(X, path="data/synthetic/synth.csv"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X).to_csv(path, index=False)


def load_synthetic(path="data/synthetic/synth.csv"):
    return pd.read_csv(path).values

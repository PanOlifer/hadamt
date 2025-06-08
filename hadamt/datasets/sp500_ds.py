import pandas as pd
from pathlib import Path
import numpy as np
from ..attacks.spike_noise import add_spike_noise


def load_sp500(path="data/sp500.csv"):
    if not Path(path).exists():
        # fallback: create random data
        df = pd.DataFrame({"close": np.random.rand(1000)})
        df.to_csv(path, index=False)
    df = pd.read_csv(path)
    return df['close'].values.reshape(-1, 1)


def poisoned_sp500(data):
    return add_spike_noise(data, ratio=0.05)

import numpy as np
from ..detectors.hybrid import HybridDetector
from ..models.vae_tabular import reconstruction_error as tab_rec_err


def run_defense(X_tab, X_img, y, thr_high=0.7):
    hd = HybridDetector()
    hd.fit(X_tab, X_img, y)
    scores = hd.score(X_tab, X_img)
    mask_clean = scores <= thr_high
    X_filtered = X_tab[mask_clean]
    y_filtered = y[mask_clean]
    return X_filtered, y_filtered, scores

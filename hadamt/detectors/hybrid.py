import numpy as np
from ..models.vae_tabular import train_vae as train_tabular_vae, reconstruction_error as tab_rec_err
from ..models.vae_image import VAE as ImageVAE, reconstruction_error as img_rec_err
from ..models.gan_disc import train_gan, disc_score
from .if_detector import fit_if, score_if
from .lof_detector import fit_lof, score_lof
from sklearn.linear_model import LogisticRegression


class HybridDetector:
    def __init__(self):
        self.weights = {
            'vae': 0.2,
            'gan': 0.2,
            'if': 0.2,
            'lof': 0.2,
            'diva': 0.2,
        }
        self.meta = LogisticRegression(max_iter=500)

    def _norm(self, arr):
        arr = np.asarray(arr)
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)

    def fit(self, X_tab, X_img, y=None):
        self.tab_vae = train_tabular_vae(X_tab, epochs=1)
        self.img_vae = ImageVAE()
        # dummy init
        self.if_model = fit_if(X_tab)
        self.lof_model = fit_lof(X_tab)
        self.g_gen, self.g_disc = train_gan(X_tab, epochs=1)
        if y is not None:
            clean_idx = y == 0
            poison_idx = y == 1
            clean_vec = self._c_measures(X_tab[clean_idx])
            poison_vec = self._c_measures(X_tab[poison_idx])
            X_train = np.vstack([clean_vec, poison_vec])
            y_train = np.array([0, 1]) if poison_vec.ndim == 1 else np.hstack([
                np.zeros(len(clean_vec)), np.ones(len(poison_vec))])
            self.meta.fit(X_train, y_train)
        
    def score(self, X_tab, X_img):
        v_tab = tab_rec_err(self.tab_vae, X_tab)
        if X_img is not None and len(X_img) > 0:
            v_img = img_rec_err(self.img_vae, X_img)
        else:
            v_img = np.zeros(len(X_tab))
        gan_s = disc_score(self.g_disc, X_tab)
        if_s = score_if(self.if_model, X_tab)
        lof_s = score_lof(self.lof_model, X_tab)
        diva_vec = self._c_measures(X_tab)
        diva_score = self.meta.predict_proba(diva_vec.reshape(1, -1))[0, 1]
        scores = (
            self.weights['vae'] * self._norm(v_tab + v_img) +
            self.weights['gan'] * self._norm(gan_s) +
            self.weights['if'] * self._norm(if_s) +
            self.weights['lof'] * self._norm(lof_s) +
            self.weights['diva'] * diva_score
        )
        return scores

    @staticmethod
    def _c_measures(X):
        var = np.var(X)
        return np.array([var])

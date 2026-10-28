import numpy as np
from scipy.stats._continuous_distns import cosine_gen

class cand_gen(cosine_gen):
    def _pdf(self, x):
        return np.cos(x / 2)**2 / np.pi
    def _logpdf(self, x):
        with np.errstate(divide='ignore'):
            return 2 * np.log(np.cos(x / 2)) - np.log(np.pi)

cand = cand_gen(a=-np.pi, b=np.pi, name='cosine')

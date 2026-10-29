import numpy as np
from scipy.stats._continuous_distns import foldcauchy_gen

class cand_gen(foldcauchy_gen):
    def _cdf(self, x, c):
        # atan(x-c) + atan(x+c) = atan2(2x, 1 + (c-x)(c+x)) exactly
        return np.arctan2(2*x, 1 + (c - x)*(c + x))/np.pi

cand = cand_gen(a=0.0, name='foldcauchy')

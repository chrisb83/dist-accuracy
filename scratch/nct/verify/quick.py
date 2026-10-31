import sys; sys.path.insert(0, r'C:/Users/chbau/Documents/research/dist-accuracy/scratch/nct/verify')
import time, mpmath as mp
from vref import both
import scipy.stats as st
for kind, t, k, d in [('cdf', -5, 0.5, 5), ('cdf', 1, 14, 0.24045031331198066), ('pdf', 1, 14, 0.24045031331198066)]:
    t0 = time.time(); v, rel = both(kind, t, k, d)
    print(kind, t, k, d, mp.nstr(v, 20), mp.nstr(rel, 3), getattr(st.nct, kind)(t, k, d), round(time.time()-t0, 1))

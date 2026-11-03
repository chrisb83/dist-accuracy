"""Absolute error of beta._ppf(q, a, a) near q = 1/2 in units of ulp(0.5), and resulting rdist.ppf error."""
import sys; sys.path.insert(0, 'refs')
import mpmath as mp, numpy as np
import rdist_ref as r
from scipy import stats
mp.mp.dps = 120
for c in [0.5, 1.6, 3.0, 10.0, 1e3, 1e6]:
    row = []
    for q in [0.4999999999999999, 0.5000000000000001, 0.49999999999999, 0.4999999999, 0.49999999, 0.4]:
        ref = r.ppf(mp.mpf(q), c)
        p = stats.beta._ppf(np.float64(q), c/2, c/2)
        perr = float((mp.mpf(p) - (1 + ref)/2) / mp.mpf(2)**-53)
        got = stats.rdist.ppf(q, c)
        ulp = float(abs((mp.mpf(got) - ref)/ref))/2.22e-16
        row.append(f"q={q!r}: p err {perr:+.3g} ulp(0.5), x={got:.6g} ref={mp.nstr(ref,6)} {ulp:.3g} ulp")
    print(f"c={c:g}"); print("   " + "\n   ".join(row))

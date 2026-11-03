# Error of Boost ibeta_inv(a, a, q) near q = 1/2 in units of ulp(0.5) = 2**-53,
# against p_true = (1 + x_ref)/2 from the verifier's independent ppf (vref.ppf).
import sys
sys.path.insert(0, r'C:/Users/chbau/Documents/research/dist-accuracy/scratch/rdist/verify')
import mpmath as mp, numpy as np, scipy.stats as st
import vref
mp.mp.dps = 100
qs = [0.5-2**-54, 0.5+2**-53, 0.5-1e-15, 0.5+1e-15, 0.5-1e-12, 0.5+1e-12, 0.5-1e-8, 0.5+1e-8, 0.4999, 0.6]
for c in [0.5, 1.6, 3.0, 10.0, 1e3, 1e6]:
    errs = []
    for q in qs:
        x = vref.ppf(q, c)
        p_true = (1 + x)/2
        p = st.beta.ppf(q, c/2, c/2)
        errs.append(float((mp.mpf(p) - p_true)/mp.mpf(2)**-53))
    print(f"c={c:g} a={c/2:g}: " + " ".join(f"{q!r}:{e:+.3g}" for q, e in zip(qs, errs)))
    print(f"    max |err| = {max(abs(e) for e in errs):.3g}, max over |q-1/2|<=1e-8 = {max(abs(e) for e,q in zip(errs,qs) if abs(q-0.5)<=1e-8):.3g}")

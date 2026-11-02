# Verifier reference: pdf via c = exp(-a): (1-c^2)/(1+c^2-2c cos x) = sinh(a)/(cosh(a)-cos(x))
# cross-checked with the wrapped-Cauchy sum sum_k a/(pi(a^2+(x+2 pi k)^2)) for one point.
import mpmath as mp, numpy as np, warnings
from scipy import stats
warnings.simplefilter("ignore")

def ref(x, c, dps):
    with mp.workdps(dps):
        x = mp.mpf(float(x)); c = mp.mpf(float(c))
        a = -mp.log(c)
        return mp.sinh(a)/(2*mp.pi*(mp.cosh(a)-mp.cos(x)))

def wrapped(x, c, dps=40):
    with mp.workdps(dps):
        x = mp.mpf(float(x)); a = -mp.log(mp.mpf(float(c)))
        return mp.nsum(lambda k: a/(mp.pi*(a**2+(x+2*mp.pi*k)**2)), [-mp.inf, mp.inf])

pts = [
 ("F1", 0.0, 0.9999999999), ("F1", 1e-08, 0.9999999999), ("F1", 6.283185307179586, 0.9999999999),
 ("F1", 1e-08, 0.9999999999999998), ("F1", 0.0, 0.9999999870854271), ("F1", 1e-06, 0.9999999999),
 ("F1", 0.0, 0.999999), ("F1", 0.0001, 0.9999999999), ("F1", 0.0, 0.999),
 ("F2", 3.141592653589793, 0.9999999870854271), ("F2", 2.0, 0.9999999870854271),
 ("F2", 3.141592653589793, 0.9999999999), ("F2", 1.0, 0.999999),
]
for fid, x, c in pts:
    r100, r200 = ref(x, c, 100), ref(x, c, 200)
    with mp.workdps(200):
        agree = abs(r100 - r200)/abs(r200) < mp.mpf(10)**-30
    s = stats.wrapcauchy.pdf(x, c)
    rf = float(r200)
    ulp = abs(s - rf)/(rf*2.220446049250313e-16) if np.isfinite(s) else np.inf
    print(fid, repr(x), repr(c), repr(s), repr(rf), f"{ulp:.3g}", "agree30" if agree else "DISAGREE")
print("wrapped-sum check pdf(2.0, 0.9999999870854271):", wrapped(2.0, 0.9999999870854271))
print("wrapped-sum check pdf(0.0, 0.999):", wrapped(0.0, 0.999))

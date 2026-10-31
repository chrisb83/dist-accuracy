"""Separate S3 (D = Phi(z+w) - Phi(z) cancellation) from S2 (epsabs) in the
infinite-df 1-D integrand: integrate SciPy's float64 integrand with a tight
tolerance (epsabs=0) and compare with the mpmath value of R_k(w)."""
import sys, math, mpmath as mp
from scipy import integrate
from scipy.special import erfc
sys.path.insert(0, 'refs'); import studentized_range_ref as R
def Phi(z): return 0.5*erfc(-z*0.7071067811865475)
def phi(z): return math.exp(-z*z/2)/math.sqrt(2*math.pi)
def f_cdf(z, w, k): return k*phi(z)*(Phi(z+w)-Phi(z))**(k-1)
def f_pdf(z, w, k): return k*(k-1)*phi(z)*phi(z+w)*(Phi(z+w)-Phi(z))**(k-2)
mp.mp.dps = 30
for k in (3.0,):
    for w in (1e-2, 1e-4, 1e-6, 1e-8):
        for name, f, d in (("cdf", f_cdf, False), ("pdf", f_pdf, True)):
            ref = R._inner(mp.mpf(w), mp.mpf(k), d, mp.mpf('0.1'), mp.mpf(10)**-35)
            tight = integrate.quad(f, -40, 40, args=(w, k), epsabs=0, epsrel=2e-14, limit=500, points=[-w/2])[0]
            loose = integrate.quad(f, -math.inf, math.inf, args=(w, k), epsabs=1e-11, epsrel=1e-12)[0]
            print(f"k={k} w={w:g} {name}: ref={mp.nstr(ref,17)} rel err tight(float D)={abs(tight/ref-1):.2e}  scipy-tol(inf range)={abs(loose/ref-1):.2e}")

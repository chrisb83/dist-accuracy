import numpy as np, mpmath as mp, scipy.special as sc
from scipy.stats._continuous_distns import _argus_phi, _norm_pdf_logC
mp.mp.dps = 50
eps = 2.220446049250313e-16
print("gammainc(1.5, z) accuracy for small z")
for z in [1e-1, 1e-3, 1e-6, 1e-10, 1e-20, 1e-50, 5e-101, 1e-150, 1e-200, 1e-205, 1e-210]:
    v = sc.gammainc(1.5, z); r = mp.gammainc(1.5, 0, mp.mpf(z), regularized=True)
    print(f"z={z:g} scipy={v!r} ulp={float(abs(v-r)/r/eps) if r else 0:.3g}")
print("_argus_phi underflow vs chi")
for chi in [1e-100, 1e-102, 1e-103, 1e-104, 1e-105, 1e-106, 1e-107, 1e-108, 1e-109]:
    v = _argus_phi(chi); r = mp.gammainc(1.5, 0, mp.mpf(chi)**2/2, regularized=True)/2
    print(f"chi={chi:g} phi={v!r} ref={mp.nstr(r,6)} relerr={float(abs(v-r)/r):.3g}")
print("A term")
for chi in [1e-3, 1e-10, 1e-50, 1e-100]:
    A = 3*np.log(chi) - _norm_pdf_logC - np.log(_argus_phi(chi))
    c = mp.mpf(chi)
    Ar = 3*mp.log(c) - mp.log(mp.sqrt(2*mp.pi)) - mp.log(mp.gammainc(1.5,0,c**2/2,regularized=True)/2)
    print(chi, A, mp.nstr(Ar, 17), float(abs(A-Ar)/eps))

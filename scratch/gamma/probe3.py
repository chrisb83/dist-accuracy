import numpy as np, mpmath as mp, scipy.special as sc
mp.mp.dps = 50
def log1pmx(x): return mp.log1p(x) - x
for a_, k in [(1e6,-5.0),(1e6,-8.0),(3e5,-5.0),(1e5,-5.0),(1e6,-14.0)]:
    a = mp.mpf(a_); x = mp.mpf(float(a_ + k*np.sqrt(a_)))
    sigma = (x - a)/a
    eta = -mp.sqrt(-2*log1pmx(sigma))
    z = -eta*mp.sqrt(a/2)
    t1 = mp.erfc(z)/2                      # leading term of the Temme expansion
    Pex = mp.gammainc(a, 0, x, regularized=True)
    corr_exact = Pex - t1                  # what the correction series must deliver
    sp = mp.mpf(float(sc.gammainc(float(a), float(x))))
    corr_scipy = sp - t1
    pref = mp.exp(-a*eta**2/2)/mp.sqrt(2*mp.pi*a)
    print(f"a={a_:g} k={k:+.0f}  z={mp.nstr(z,8)}")
    print(f"   P exact      = {mp.nstr(Pex,17)}")
    print(f"   0.5*erfc(z)  = {mp.nstr(t1,17)}   (t1-P)/P = {mp.nstr((t1-Pex)/Pex,4)}")
    print(f"   corr exact   = {mp.nstr(corr_exact,10)}   /prefactor = {mp.nstr(corr_exact/pref,10)}")
    print(f"   corr scipy   = {mp.nstr(corr_scipy,10)}   /prefactor = {mp.nstr(corr_scipy/pref,10)}")
    print(f"   rel err scipy= {mp.nstr(abs(sp-Pex)/Pex,4)}   d(corr)/pref = {mp.nstr((corr_scipy-corr_exact)/pref,6)}")

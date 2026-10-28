import numpy as np, mpmath as mp, scipy.special as sc
mp.mp.dps = 50
def igam_series(a, x, maxiter):
    """cephes igam_series with a fixed iteration cap, in high precision."""
    a = mp.mpf(a); x = mp.mpf(x)
    ax = mp.exp(a*mp.log(x) - x - mp.loggamma(a))
    r = a; c = mp.mpf(1); ans = mp.mpf(1)
    used = maxiter
    for i in range(maxiter):
        r += 1
        c *= x/r
        ans += c
        if c/ans <= mp.mpf(2)**-53:
            used = i+1
            break
    return ans*ax/a, used

print(" a        k    scipy                     series(2000 terms)        rel(scipy,trunc)  rel(scipy,exact)  terms_needed")
for a, k in [(1e6,-5.0),(1e6,-6.0),(1e6,-8.0),(1e6,-10.0),(1e6,-12.0),(3e5,-5.0),(1e5,-5.0),(1e6,-3.0)]:
    x = float(a + k*np.sqrt(a))
    trunc, used = igam_series(a, x, 2000)
    full, nfull = igam_series(a, x, 200000)
    sp = mp.mpf(float(sc.gammainc(a, x)))
    print(f"{a:8.3g} {k:+5.1f}  {mp.nstr(sp,17):24} {mp.nstr(trunc,17):24} "
          f"{float(abs(sp-trunc)/trunc):.2e}          {float(abs(sp-full)/full):.2e}        {nfull}")

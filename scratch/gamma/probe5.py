import numpy as np, mpmath as mp, scipy.special as sc
mp.mp.dps = 50
def series(a, x, maxiter=400000):
    a = mp.mpf(a); x = mp.mpf(x)
    ax = mp.exp(a*mp.log(x) - x - mp.loggamma(a))
    r = a; c = mp.mpf(1); ans = mp.mpf(1)
    for i in range(maxiter):
        r += 1; c *= x/r; ans += c
        if c/ans <= mp.mpf(10)**-45: return ans*ax/a, i+1
    raise RuntimeError("no convergence")
# validation of the converged series against mpmath's own gammainc at a=1e6
a, x = 1e6, 995000.0
s, n = series(a, x)
g = mp.gammainc(mp.mpf(a), 0, mp.mpf(x), regularized=True)
print("validation a=1e6 x=995000: series =", mp.nstr(s, 20), " mp.gammainc =", mp.nstr(g, 20))
# Kummer/hyp1f1 route at a=1e7
a2, x2 = 1e7, float(1e7 - 4.6*np.sqrt(1e7))
alt = mp.exp(mp.mpf(a2)*mp.log(mp.mpf(x2)) - mp.mpf(x2) - mp.loggamma(mp.mpf(a2)+1))*mp.hyp1f1(1, mp.mpf(a2)+1, mp.mpf(x2))
s2, n2 = series(a2, x2)
print("validation a=1e7: series =", mp.nstr(s2, 20), " hyp1f1 =", mp.nstr(alt, 20))
print()
print("  a       k      x                P (exact)        scipy             rel.err    terms needed")
for a in [1e6, 1e7, 1e8, 1e9]:
    for k in [-4.6, -5.0, -7.0, -10.0]:
        x = float(a + k*np.sqrt(a))
        ref, n = series(a, x)
        sp = sc.gammainc(a, x)
        rel = float(abs(mp.mpf(float(sp)) - ref)/ref)
        print(f"{a:7.1g} {k:+5.1f}  {x:16.10g} {float(ref):.10e}  {sp:.10e}  {rel:.3e}   {n}")

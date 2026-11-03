# Cross-check (fast): Newton residual dp = (F(p) - q)/f(p) with F via I_p(a,a) = 1/2 + sign(2p-1)/2 I_{(2p-1)^2}(1/2, a), units 2**-53.
# dp = (I_p(a,a) - q) / beta_pdf(p), in units of 2**-53 (route independent of vref.ppf).
import mpmath as mp, numpy as np, scipy.stats as st
for dps in (100, 200):
    mp.mp.dps = dps
    out = []
    for c, q in [(3.0, 0.4999999999999999), (3.0, 0.49999999999999994), (1e3, 0.50000001), (1e3, 0.500000000000001),
                 (1e6, 0.500000000001), (1e6, 0.50000001), (1e6, 0.4999999999), (1e3, 0.4999999999), (1e6, 0.5000000001), (1e3, 0.5000000001)]:
        a = mp.mpf(c)/2; p = mp.mpf(st.beta.ppf(q, c/2, c/2))
        xx = 2*p - 1; F = mp.mpf(1)/2 + mp.sign(xx)*mp.betainc(mp.mpf(1)/2, a, 0, xx*xx, regularized=True)/2
        f = p**(a-1)*(1-p)**(a-1)/mp.beta(a, a)
        out.append(f"c={c:g} q={q!r}: {float((F - mp.mpf(q))/f/mp.mpf(2)**-53):+.4g}")
    print(f"dps={dps}: " + "; ".join(out))

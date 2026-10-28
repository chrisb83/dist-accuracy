"""Verify the pdf-integral self-check FAILs (a < 1) are quadrature artifacts:
integrate the reference pdf on [0, b] after the substitution x = t**(1/a),
which removes the x**(a-1) endpoint singularity, and compare with the cdf."""
import sys, mpmath as mp
sys.path.insert(0, "refs")
import dgamma_ref as r
mp.mp.dps = 40
for a, b in [(0.001, 6.91293435418913e-98), (0.1, 0.0693899)]:
    a = mp.mpf(a); b = mp.mpf(b)
    naive = mp.quad(lambda t: r.pdf(t, a), [0, b])
    sub = mp.quad(lambda t: r.pdf(t ** (1 / a), a) * t ** (1 / a - 1) / a, [0, b ** a])
    diff = r.cdf(b, a) - r.cdf(mp.mpf(0), a)
    print(float(a), "naive quad rel err", mp.nstr(naive / diff - 1, 3),
          "| substituted quad rel err", mp.nstr(sub / diff - 1, 3),
          "| -b side", mp.nstr(mp.quad(lambda t: r.pdf(-t ** (1 / a), a) * t ** (1 / a - 1) / a, [0, b ** a]) / (r.cdf(mp.mpf(0), a) - r.cdf(-b, a)) - 1, 3))

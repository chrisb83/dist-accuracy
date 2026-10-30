# Independent verification of F1 (gumbel_l ppf/isf near quantile 0).
# pdf(x) = exp(x - exp(x)); cdf via quad of pdf; quantile by findroot on the quad cdf,
# cross-checked against the closed-form log(-log(1-q)).
import mpmath as mp
import numpy as np
from scipy import stats

cases = [("ppf", 0.6321205588285577), ("isf", 0.36787944117144233),
         ("ppf", 0.6321205588285578), ("isf", 0.3678794411714424)]

def ref(method, q, dps):
    with mp.workdps(dps):
        q = mp.mpf(q)
        target = q if method == "ppf" else 1 - q
        pdf = lambda t: mp.exp(t - mp.exp(t))
        # cdf(x) = cdf(0) + int_0^x pdf ; cdf(0) = 1 - 1/e from quad over (-inf, 0]
        c0 = mp.quad(pdf, [-mp.inf, -20, -5, 0])
        f = lambda x: c0 + mp.quad(pdf, [0, x]) - target
        # Newton with pdf as derivative, starting at 0
        x = mp.mpf(0)
        for _ in range(8):
            x = x - f(x) / pdf(x)
        closed = mp.log(-mp.log(1 - target))
        return x, closed

for m, q in cases:
    a, ca = ref(m, q, 100)
    b, cb = ref(m, q, 200)
    agree = mp.nstr(abs(a - b) / abs(b), 5)
    s = getattr(stats.gumbel_l, m)(q)
    rel = abs(mp.mpf(s) - b) / abs(b)
    print(m, repr(q), "scipy", repr(float(s)), "ref", mp.nstr(b, 20),
          "closed-vs-quad", mp.nstr(abs(cb - b) / abs(b), 3), "100v200", agree,
          "ulp", mp.nstr(rel / 2.220446049250313e-16, 3))

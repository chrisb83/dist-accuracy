import warnings, numpy as np, mpmath as mp
from scipy import stats
warnings.simplefilter("ignore")

def pdf(x, c):   # docstring pdf, direct power form
    return c / (1 + x)**(c + 1)

def isf_ref(q, c):
    # solve sf(x) = q, sf from integrating the pdf numerically; starting guess closed form
    x0 = (q**(-1/c) - 1)
    sf = lambda x: mp.quad(lambda t: pdf(t, c), [x, x + 1, mp.inf])
    # Newton on sf(x) - q, sf' = -pdf
    x = x0
    for _ in range(3):
        x = x + (sf(x) - q) / pdf(x, c)
    return x, sf(x)

def logpdf_ref(x, c):
    return mp.log(pdf(x, c))   # log of power form (different from log(c)-(c+1)log1p(x))

cases = [
 ("isf", 0.99999999999999, 1000.0),
 ("isf", 0.9999999999, 1e6),
 ("isf", 0.999999, 1000.0),
 ("isf", 0.99, 1e6),
 ("isf", 0.5, 1e6),
 ("pdf", 1.032549079066965, 1000.0),
 ("pdf", 0.0007102513991588548, 1e6),
 ("logpdf", 0.24468229012549414, 1.8771398388773268),
 ("logpdf", 0.006924720186917837, 1000.0),
 ("pdf", 1e-15, 1e6),
 ("pdf", 1e-12, 1e6),
 ("pdf", 0.0007002443566872014, 1e6),
]

def ref(m, a, c, dps):
    mp.mp.dps = dps
    a, c = mp.mpf(a), mp.mpf(c)
    if m == "isf":
        x, sfx = isf_ref(a, c)
        return x, abs(sfx/a - 1)
    if m == "pdf":
        return pdf(a, c), 0
    return logpdf_ref(a, c), 0

for m, a, c in cases:
    r100, chk = ref(m, a, c, 100)
    r200, _ = ref(m, a, c, 200)
    mp.mp.dps = 200
    agree = abs(r200 - r100) <= abs(r200) * mp.mpf(10)**-30
    s = getattr(stats.lomax, m)(a, c)
    r = float(r200)
    ulp = abs(s - r200) / (abs(r200) * 2.220446049250313e-16) if r200 != 0 else None
    print(f"{m}({a!r}, {c!r}): scipy={s!r} ref={mp.nstr(r200,17)} ulp={mp.nstr(ulp,3)} agree30={agree} sfcheck={mp.nstr(chk,3)}")

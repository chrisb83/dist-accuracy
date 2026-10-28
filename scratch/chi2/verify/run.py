import sys, time, mpmath as mp, numpy as np
sys.path.insert(0, r'C:/Users/chbau/Documents/research/dist-accuracy/scratch/chi2/verify')
import vref as R
from scipy.stats import chi2
mp.mp.dps = int(sys.argv[1])
which = sys.argv[2:]
M = mp.mpf
cases = {
 'F1a': ('cdf', 1990800., 2e6), 'F1b': ('cdf', 19970907.045526452, 2e7),
 'F1c': ('cdf', 199908000., 2e8), 'F1d': ('sf', 199908000., 2e8),
 'F2a': ('pdf', 200000000., 2e8), 'F2b': ('pdf', 199400000., 2e8), 'F2c': ('pdf', 1980000., 2e6),
 'F3a': ('logpdf', 200000000., 2e8), 'F3b': ('logpdf', 1994000., 2e6),
 'F4a': ('ppf', 1e-6, 2e6), 'F4b': ('isf', 0.999999, 2e6),
 'F5a': ('ppf', 0.5000000000000001, 0.002), 'F5b': ('isf', 0.36, 0.002),
 'F6a': ('sf', 3897.3665961010274, 2000.), 'F6b': ('cdf', 1051.3167019494863, 2000.),
}
for key in which:
    m, a, k = cases[key]; t0 = time.time()
    sp = getattr(chi2, m)(a, k)
    if m in ('pdf', 'logpdf'): ref = getattr(R, m)(a, k)
    elif m == 'cdf': ref = R.cdf(a, k)
    elif m == 'sf': ref = R.sf(a, k) if a > k else 1 - R.cdf(a, k)
    elif k < 1:  # tiny df quantiles
        q = M(a) if m == 'ppf' else 1 - M(a)
        ref = R.newton_log(R.cdf_small, sp, q, k)
        chk = R.cdf_small(ref, k)/q - 1
    else:
        q = M(a) if m == 'ppf' else 1 - M(a)
        ref = R.newton(R.cdf, sp, q, k)
        chk = R.cdf(ref, k)/q - 1
    ulp = abs((M(sp) - ref)/ref)/M(2)**-52
    extra = f' cdfcheck={mp.nstr(chk,3)}' if m in ('ppf','isf') else ''
    print(f'{key} {m}({a!r},{k!r}) scipy={sp!r} ref={mp.nstr(ref,40)} ulp={mp.nstr(ulp,3)} relerr={mp.nstr(abs(M(sp)/ref-1),3)}{extra} t={time.time()-t0:.1f}s', flush=True)

"""Quantile check: at the analyst's x_ref evaluate my cdf (or sf) and pdf, take a
log-space Newton step to get my own quantile, and compare SciPy with it."""
import sys; sys.path.insert(0, r'C:/Users/chbau/Documents/research/dist-accuracy/scratch/nct/verify')
import mpmath as mp, scipy.stats as st
from vref import both
for item in sys.argv[1].split(';'):
    kind, q, k, d, xa = item.split()
    F = 'cdf' if kind == 'ppf' else 'sf'
    Fv, r1 = both(F, xa, k, d)
    fv, r2 = both('pdf', xa, k, d)
    with mp.workdps(200):
        qq = mp.mpf(q)
        step = (mp.log(Fv) - mp.log(qq))*Fv/fv
        x = mp.mpf(xa) - step if F == 'cdf' else mp.mpf(xa) + step
    try: s = float(getattr(st.nct, kind)(float(q), float(k), float(d)))
    except Exception as e: s = float('nan')
    ulp = abs(mp.mpf(s) - x)/abs(x)/mp.mpf(2)**-52 if s == s else mp.inf
    print(f"{kind}({q},{k},{d}) scipy={s!r} myref={mp.nstr(x,17)} {F}(x_a)/q={mp.nstr(Fv/qq,17)} ulp={mp.nstr(ulp,3)} stab={mp.nstr(max(r1,r2),2)}", flush=True)

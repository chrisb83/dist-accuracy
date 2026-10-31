import sys; sys.path.insert(0, r'C:/Users/chbau/Documents/research/dist-accuracy/scratch/nct/verify')
import mpmath as mp, numpy as np, scipy.stats as st
from vref import both
def sp(kind, *a):
    try: return float(getattr(st.nct, kind)(*a))
    except Exception as e: return type(e).__name__
pts = [l.split() for l in sys.argv[1].split(';')]
for p in pts:
    kind = p[0]; t, k, d = p[1], p[2], p[3]
    v, rel = both(kind, t, k, d)
    s = sp(kind, float(t), float(k), float(d))
    if isinstance(s, float):
        ulp = abs(mp.mpf(s) - v)/abs(v)/mp.mpf(2)**-52
        u = mp.nstr(ulp, 3)
    else: u = 'inf'
    print(f"{kind}({t}, {k}, {d}) scipy={s!r} ref={mp.nstr(v, 17)} ulp={u} stab={mp.nstr(rel,2)}", flush=True)

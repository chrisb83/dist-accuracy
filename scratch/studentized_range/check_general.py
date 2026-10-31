import sys, time, mpmath as mp
sys.path.insert(0, 'refs'); import studentized_range_ref as R
mp.mp.dps = 30
for x, k, df, d in [(1, 2, 10, False), (1, 2, 10, True), (0.01, 2, 3, False)]:
    x, k, df = map(mp.mpf, (x, k, df))
    t = time.time(); g = R._checked(x, k, df, d)
    e = R._pdf2(x, df) if d else R._cdf2(x, df)
    print(float(x), int(k), float(df), d, mp.nstr(g, 25), mp.nstr(e, 25), mp.nstr(abs(g/e-1), 3), round(time.time()-t, 1))
from scipy.stats import studentized_range as S
for x, k, df, d in [(1, 3, 10, False), (1, 3, 10, True), (1, 1.5, 10, True), (1, 1.5, 10, False)]:
    t = time.time(); g = R._checked(*map(mp.mpf, (x, k, df)), d)
    sv = (S.pdf if d else S.cdf)(x, k, df)
    print(x, k, df, d, mp.nstr(g, 20), sv, mp.nstr(abs(sv/g-1), 3), round(time.time()-t, 1))

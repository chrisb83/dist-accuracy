# F8 reference by numerical integration of the docstring pdf (different route from betainc / continued fraction)
import time, mpmath as mp
from scipy import stats
import importlib.util, sys
spec = importlib.util.spec_from_file_location('v', 'verify.py')
src = open('verify.py').read().split('CASES = [')[0]
ns = {}; exec(src, ns)
x, n, m = 0.9496708976337849, 1e5, 3e5
res = []
for dps in (100, 200):
    mp.mp.dps = dps
    t = time.time()
    X = mp.mpf(x)
    f = lambda u: ns['pdf'](u, n, m)
    pts = [X - mp.mpf(k)/100 for k in range(40, -1, -1)]   # [x-0.4, x] in 0.01 steps
    pts = [X - mp.mpf('0.4') + (mp.mpf('0.4')*j)/400 for j in range(401)]
    v = mp.quad(f, pts) + mp.quad(f, [0, X - mp.mpf('0.4')])
    res.append(v); print(dps, mp.nstr(v, 25), 'tail part', mp.nstr(mp.quad(f, [0, X-mp.mpf('0.4')]),5), f'{time.time()-t:.1f}s', flush=True)
mp.mp.dps = 200
print('100vs200', mp.nstr(abs(res[0]-res[1])/res[1], 3))
s = stats.f.cdf(x, n, m)
print('scipy', repr(s), 'ulp', mp.nstr(abs(mp.mpf(s)-res[1])/res[1]/mp.mpf(2.220446049250313e-16), 4))

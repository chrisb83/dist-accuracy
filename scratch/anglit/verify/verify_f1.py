import mpmath as mp, numpy as np, scipy, scipy.stats as st, inspect
print(scipy.__version__)
x = -0.7853981633974483
assert x == -np.pi/4
def ref_cdf(x):
    a = -mp.pi/4
    # route 1: integral of pdf cos(2t) from -pi/4 to x
    q = mp.quad(lambda t: mp.cos(2*t), [a, mp.mpf(x)])
    # route 2: (1 + sin 2x)/2
    c = (1 + mp.sin(2*mp.mpf(x)))/2
    return q, c
def ref_sf(x):
    b = mp.pi/4
    q = mp.quad(lambda t: mp.cos(2*t), [mp.mpf(x), b])
    c = (1 - mp.sin(2*mp.mpf(x)))/2
    return q, c
for dps in (100, 200):
    mp.mp.dps = dps
    print(dps, [mp.nstr(v, 40) for v in ref_cdf(x)], [mp.nstr(v, 40) for v in ref_sf(-x)])
mp.mp.dps = 100
print("fl(pi/4) - pi/4 =", mp.nstr(mp.mpf(np.pi/4) - mp.pi/4, 20))
print("scipy cdf", st.anglit.cdf(x), "sf", st.anglit.sf(-x))
print("_cdf", st.anglit._cdf(np.float64(x)), "_sf", st.anglit._sf(np.float64(-x)))
print("support", st.anglit.support())
src, start = inspect.getsourcelines(type(st.anglit))
for i, l in enumerate(src):
    if 'def _cdf' in l or 'def _sf' in l or 'pi/4' in l or 'pi / 4' in l:
        print(start+i, l.rstrip())

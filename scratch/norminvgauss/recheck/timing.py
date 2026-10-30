import sys, time
sys.path.insert(0, r'C:/Users/chbau/Documents/research/dist-accuracy/refs')
import mpmath as mp
import norminvgauss_ref as R
mp.mp.dps = 100
orig = R._mix_tail1
calls = []
def wrap(x, a, b, sgn, hfac=20):
    calls.append(hfac); return orig(x, a, b, sgn, hfac)
R._mix_tail1 = wrap
for a, b in [(1, 0.999), (10, 9.99999)]:
    for x in [-1e5, -300, -10, -1, 0, 1, 10, 22, 100, 1e3, 1e4, 1e5, 6e5, 1e8]:
        calls.clear(); t = time.time()
        R._tail0.cache_clear()
        c = R.cdf(mp.mpf(x), mp.mpf(a), mp.mpf(b))
        print(a, b, x, mp.nstr(c, 12), 'hfac', calls, f'{time.time()-t:.1f}s', flush=True)

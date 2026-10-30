import sys, time, warnings
warnings.simplefilter('ignore')
sys.path.insert(0, r'C:/Users/chbau/Documents/research/dist-accuracy/refs')
import mpmath as mp
import norminvgauss_ref as R
for m, x in [('isf', 1e-200), ('isf', 0.3), ('isf', 1 - 1e-12)]:
    for d in (100, 200):
        t = time.time()
        with mp.workdps(d):
            v = R.REFS[m](mp.mpf(x), mp.mpf(10), mp.mpf(9.99999))
        print(m, x, d, mp.nstr(v, 20), f'{time.time()-t:.1f}s', flush=True)

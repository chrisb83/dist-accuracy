import sys, os, time
sys.path.insert(0, r"C:/Users/chbau/Documents/research/dist-accuracy/refs")
import mpmath as mp
import pearson3_ref as R
k = mp.mpf(-1.59e-5)
for dps in (100, 200, 300):
    mp.mp.dps = dps
    t0 = time.time()
    v = R._tail_quad(mp.mpf(-30), k, -1)
    print("scaled", dps, mp.nstr(v, 20), round(time.time()-t0, 1))

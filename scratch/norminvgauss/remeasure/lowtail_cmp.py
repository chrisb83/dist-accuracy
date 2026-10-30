# Recompute the isf low-tail (q < 1e-3) references with the fixed refs module
# and compare with the old grid values in points.json (pre-fix reference).
import sys, json, time, warnings
warnings.simplefilter('ignore')
sys.path.insert(0, r'C:/Users/chbau/Documents/research/dist-accuracy/refs')
import mpmath as mp
import norminvgauss_ref as R
mp.mp.dps = 100
old = json.load(open(r'C:/Users/chbau/Documents/research/dist-accuracy/scratch/norminvgauss/points.json'))
pts = [p for p in old if p['method'] == 'isf' and p['region'] == 'low tail'
       and tuple(p['shapes']) != (10.0, 9.99999)]
start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
pts = pts[start:]
print(len(pts), "points from", start, flush=True)
t0 = time.time()
for p in pts:
    t = time.time()
    v = float(R.isf(mp.mpf(p['arg']), *[mp.mpf(x) for x in p['shapes']]))
    rel = abs(v - p['ref']) / abs(v) if v else float('inf')
    flag = 'CHANGED' if rel > 1e-15 else 'same'
    print(flag, p['shapes'], repr(p['arg']), p['ref'], v, f'{rel:.2e}',
          p['impl']['scipy 1.18.1']['severity'], f'{time.time()-t:.0f}s', flush=True)
print('done', f'{time.time()-t0:.0f}s')

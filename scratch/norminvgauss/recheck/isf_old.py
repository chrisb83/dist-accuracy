import sys, json, time, warnings
warnings.simplefilter('ignore')
sys.path.insert(0, r'C:/Users/chbau/Documents/research/dist-accuracy/refs')
import mpmath as mp
import norminvgauss_ref as R
mp.mp.dps = 100
old = json.load(open(r'C:/Users/chbau/Documents/research/dist-accuracy/scratch/norminvgauss/points.json'))
pts = [p for p in old if p['method'] == 'isf' and tuple(p['shapes']) in {(10.0, 9.99999), (1.0, 0.999)} and 1e-6 <= p['arg'] <= 1 - 1e-6]
print(len(pts), 'points', flush=True)
for p in pts:
    t = time.time()
    v = float(R.isf(mp.mpf(p['arg']), *[mp.mpf(x) for x in p['shapes']]))
    rel = abs(v - p['ref']) / abs(v) if v else float('inf')
    sc = p['impl']['scipy 1.18.1']
    flag = 'CHANGED' if rel > 1e-15 else ''
    print(flag, p['shapes'], repr(p['arg']), 'old', p['ref'], 'new', v, f'{rel:.2e}', 'scipy', sc['value'], sc['severity'], f'{time.time()-t:.0f}s', flush=True)

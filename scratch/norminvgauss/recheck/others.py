import sys, json, time, warnings
warnings.simplefilter('ignore')
sys.path.insert(0, r'C:/Users/chbau/Documents/research/dist-accuracy/refs')
import mpmath as mp
import norminvgauss_ref as R
mp.mp.dps = 100
old = json.load(open(r'C:/Users/chbau/Documents/research/dist-accuracy/scratch/norminvgauss/points.json'))
skip = {(10.0, 9.99999), (1.0, 0.999)}
pts = sorted({(tuple(p['shapes']), p['arg']) for p in old if p['method'] == 'cdf' and tuple(p['shapes']) not in skip})
O = {(p['method'], tuple(p['shapes']), p['arg']): p for p in old if p['method'] in ('cdf', 'sf')}
t0 = time.time(); n = 0
for sh, x in pts:
    c, s = R._tail(mp.mpf(x), *[mp.mpf(v) for v in sh])
    n += 1
    for m, v in (('cdf', c), ('sf', s)):
        o = O.get((m, sh, x))
        if o is None or v == 0: continue
        rel = abs(o['ref'] - float(v)) / abs(float(v)) if o['ref'] != 0 else float('inf')
        if rel > 1e-15:
            print('CHANGED', m, sh, x, o['ref'], float(v), f'{rel:.2e}', flush=True)
print('checked', n, 'points', f'{time.time()-t0:.0f}s', flush=True)

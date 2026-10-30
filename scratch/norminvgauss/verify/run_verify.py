import sys, time, warnings
import mpmath as mp
import numpy as np
import scipy.stats as st
import vref
warnings.simplefilter('ignore')
D = st.norminvgauss
# (finding, method, arg, a, b, analyst_ref)  for isf: x0 = analyst ref
P = [
 ('F1','sf',-1e-10,1e6,5e5,1.0),
 ('F1','sf',179088.24039927218,10,9.99999,0.00017082512756882697),
 ('F1','cdf',0.54012607482554176,1e6,5e5,3.380440201163224e-201),
 ('F1','cdf',179088.24039927218,10,9.99999,0.9998291748724312),
 ('F1','cdf',-0.0001,1e8,0,0.15865525332653024),
 ('F2','isf',0.29,10,9.99999,67.44589577432565),
 ('F2','isf',0.57,1e6,5e5,0.5771311001290594),
 ('F2','isf',0.3,1e8,0,5.244005109217989e-05),
 ('F3','pdf',0.0,1.4e154,0,4.720348719413148e+76),
 ('F3','cdf',-1e-80,1e160,0,0.15865525393145705),
 ('F4','isf',0.3,1e-170,0,0.72654252800536094),
 ('F5','isf',0.49999999999999994,1,0,1.0658744816094324e-16),
 ('F5','isf',0.49999999949999741,1,0,9.600593660978633e-10),
 ('F6','cdf',-30,1,0,5.860566997067416e-16),
 ('F6','sf',10,1,0,1.3409847489184868e-06),
 ('F6','cdf',-600,1,0,1.9503586841127083e-265),
 ('F7','isf',1e-10,1.25,0.5,24.982927090751097),
 ('F7','isf',1e-300,1.25,0.5,908.2422209188078),
 ('F8','pdf',60000000.002271466,10,9.99999,7.296857453961407e-273),
 ('F8','pdf',0.001,1e6,0,241.97066402645464),
]
sel = sys.argv[1:] 
for fid, m, x, a, b, aref in P:
    if sel and fid not in sel: continue
    t = time.time()
    try:
        sp = float(getattr(D, m)(x, a, b))
    except Exception as e:
        sp = f'{type(e).__name__}: {e}'
    vals = []
    for dps in (100, 200):
        mp.mp.dps = dps
        if m == 'isf':
            v = vref.isf(x, a, b, aref)
        else:
            v = getattr(vref, m)(x, a, b)
        vals.append(v)
    mp.mp.dps = 200
    agree = mp.nstr(abs(vals[0]-vals[1])/abs(vals[1]), 3) if vals[1] != 0 else 'abs %s' % mp.nstr(abs(vals[0]-vals[1]),3)
    r = vals[1]
    ulp = (lambda c: mp.nstr(abs(mp.mpf(c)-r)/abs(r)/mp.mpf(2.220446049250313e-16), 3)) 
    out = [fid, f'{m}({x!r}, {a!r}, {b!r})', repr(sp), mp.nstr(r, 17), 'rel100v200=' + agree,
           'scipy_ulp=' + (ulp(sp) if isinstance(sp, float) and np.isfinite(sp) else 'n/a'),
           'analyst_rel=' + mp.nstr(abs(mp.mpf(aref)-r)/abs(r), 3), f'{time.time()-t:.0f}s']
    if m == 'isf':
        mp.mp.dps = 60
        out.append('check sf(x)/q-1=' + mp.nstr(vref.sf(r, a, b)/mp.mpf(x)-1, 3))
    print(' | '.join(out), flush=True)

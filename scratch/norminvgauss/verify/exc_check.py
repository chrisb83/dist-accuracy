import warnings, mpmath as mp, scipy.stats as st
warnings.simplefilter('ignore')
D = st.norminvgauss
for args in [(0.3, 1e160, 0), (0.3, 1e-170, 0), (0.3, 1e-170, 5e-171)]:
    try:
        print(args, D.isf(*args))
    except Exception as e:
        print(args, type(e).__name__, str(e)[:90])
print('pdf(0,1.4e154,0)', D.pdf(0, 1.4e154, 0), 'pdf(0,1.3e154,0)', D.pdf(0, 1.3e154, 0))
print('cdf(-1e-80,1e160,0)', D.cdf(-1e-80, 1e160, 0), 'sf(1e-80,1e160,0)', D.sf(1e-80, 1e160, 0))
mp.mp.dps = 100
print('Cauchy isf(0.3)', mp.nstr(mp.tan(mp.pi*(mp.mpf(1)/2 - mp.mpf(0.3))), 20))
# F5 linearization: b=0 symmetric, sf(x) ~ 1/2 - f(0) x
f0 = mp.besselk(1, 1) * mp.e / mp.pi
for q in [0.49999999999999994, 0.49999999949999741]:
    print('F5 lin', q, mp.nstr((mp.mpf(1)/2 - mp.mpf(q)) / f0, 17), D.isf(q, 1, 0))

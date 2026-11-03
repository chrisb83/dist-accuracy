import mpmath as mp, numpy as np, warnings
from scipy import stats
warnings.simplefilter("ignore")
xs = [1.4e154, 1e155, 1e200, 1e250, 1.7e308, 1.3e154, 1.35e154]
def ref(x, dps):
    # route: log of numerical derivative of cdf F(t) = (2/pi) atan(t), in t = log x
    # (dF/dt = x f(x)) -> log f = log(dF/dt) - log x
    with mp.workdps(dps):
        X = mp.mpf(x)
        g = lambda t: 2/mp.pi*mp.atan(mp.exp(t))
        d = mp.diff(g, mp.log(X))
        return mp.log(d) - mp.log(X)
def ref2(x, dps):
    with mp.workdps(dps):
        X = mp.mpf(x)
        return mp.log(mp.quad(lambda t: 2/mp.pi/(1+t*t), [X, mp.inf])) * 0 + mp.log(2/mp.pi) - mp.log(1 + X*X)
for x in xs:
    a = ref(x, 100); b = ref(x, 200); c = ref2(x, 200)
    with mp.workdps(200):
        agree = mp.nstr(abs(a-b)/abs(b), 3); agree2 = mp.nstr(abs(c-b)/abs(b), 3)
    s = stats.halfcauchy.logpdf(x)
    r = float(b)
    ulp = abs(s - r)/(abs(r)*2.2e-16) if np.isfinite(s) else np.inf
    print(f"{x:g} scipy={s!r} ref={mp.nstr(b,20)} rel100v200={agree} closedform_vs_diff={agree2} ulp={ulp:.3g}")

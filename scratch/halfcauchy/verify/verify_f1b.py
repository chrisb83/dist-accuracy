import mpmath as mp, numpy as np, warnings
from scipy import stats
warnings.simplefilter("ignore")
xs = [1.3e154, 1.35e154, 1.4e154, 1e155, 1e200, 1e250, 1.7e308]
def ref_u(x, dps):
    # route: u = 1/x, f = (2/pi) u^2 / (1 + u^2)
    with mp.workdps(dps):
        u = 1/mp.mpf(x)
        return mp.log(2/mp.pi) + 2*mp.log(u) - mp.log1p(u*u)
def ref_d(x, dps):
    # route: f = -d/dx sf, sf = (2/pi) atan(1/x); symbolic derivative via mp.diff at high precision
    with mp.workdps(dps + 700):
        X = mp.mpf(x)
        d = -mp.diff(lambda t: 2/mp.pi*mp.atan(1/t), X)
        return mp.log(d)
for x in xs:
    a = ref_u(x, 100); b = ref_u(x, 200); c = ref_d(x, 100)
    with mp.workdps(200):
        e1 = mp.nstr(abs(a-b)/abs(b), 3); e2 = mp.nstr(abs(c-b)/abs(b), 3)
    s = stats.halfcauchy.logpdf(x); r = float(b)
    ulp = abs(s - r)/(abs(r)*2.2e-16) if np.isfinite(s) else np.inf
    print(f"{x:g} scipy={s!r} ref={r!r} 100v200={e1} diffroute={e2} ulp={ulp:.3g}")

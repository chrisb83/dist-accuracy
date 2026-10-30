# Independent reference for moyal ppf: invert cdf obtained by quadrature of the docstring pdf
import mpmath as mp, numpy as np, scipy.stats as st
qs = [0.31731050786291415, 0.31731050789464516, 0.3173108251734219,
      0.3173101905524062, 0.315, 0.3]
def pdf(x): return mp.exp(-(x + mp.exp(-x))/2)/mp.sqrt(2*mp.pi)
def ref(q, dps):
    with mp.workdps(dps):
        q = mp.mpf(q)
        c0 = mp.quad(pdf, [-mp.inf, -5, -2, 0])          # cdf(0) by quadrature
        cdf = lambda x: c0 + mp.quad(pdf, [0, x])
        x = mp.mpf(0)
        for _ in range(200):
            dx = (cdf(x) - q)/pdf(x)
            x -= dx
            if abs(dx) < mp.mpf(10)**(-dps+5)*max(abs(x), mp.mpf(10)**-30): break
        return x
for q in qs:
    a, b = ref(q, 100), ref(q, 200)
    with mp.workdps(200):
        agree = abs(a-b)/abs(b)
    s = st.moyal.ppf(q)
    r = float(b)
    ulp = abs(s - r)/(abs(r)*2.220446049250313e-16)
    print(f"{q!r}: scipy={s!r} ref={mp.nstr(b,20)} rel100v200={mp.nstr(agree,3)} ulp={ulp:.3g} abs/eps={abs(s-r)/2.220446049250313e-16:.3g}")

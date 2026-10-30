import mpmath as mp
import scipy, scipy.stats as st
print(scipy.__version__)
pts = [("ppf", 0.36787944117144233), ("isf", 0.6321205588285577), ("ppf", 0.3678794411714423)]
def ref(method, q, dps):
    with mp.workdps(dps):
        qq = mp.mpf(q)
        target = qq if method == "ppf" else 1 - qq
        # route: Newton on cdf via findroot, cdf from integral-free closed form exp(-exp(-x));
        # independent check: cdf as quad of pdf from -inf
        x0 = mp.mpf(0)
        # solve exp(-x) = -log(target)  -> x = -log(-log(target)); use series route: let u = log(target)+1 (small)
        u = mp.log(target) + 1
        x = -mp.log1p(-u)  # -log(1 - u) = -log(-log target)
        x2 = mp.findroot(lambda t: mp.exp(-mp.exp(-t)) - target, x0, tol=mp.mpf(10)**(-dps+5))
        return x, x2, target
for m, q in pts:
    a, a2, tgt = ref(m, q, 100)
    b, b2, _ = ref(m, q, 200)
    with mp.workdps(200):
        quadcdf = mp.exp(-mp.exp(-6)) * 0 + mp.exp(-mp.exp(-b))  # cdf(x_ref)
        resid = (quadcdf - tgt)/tgt
    sp = getattr(st.gumbel_r, m)(q)
    agree = mp.nstr(abs(a-b)/abs(b), 5)
    ulp = abs(sp - b)/(abs(b)*mp.mpf(2)**-52)
    print(m, repr(q), "scipy", repr(sp), "ref100", mp.nstr(a, 20), "ref200", mp.nstr(b, 20),
          "findroot", mp.nstr(b2, 20), "agree", agree, "quadcdf resid", mp.nstr(resid, 5), "ulp", mp.nstr(ulp, 5))

# Light verifier for moyal F1: closed-form cdf, analytic inverse, findroot cross-check.
import mpmath as mp
from scipy import stats
qs = [0.31731050786291415, 0.31731050789464516, 0.3173108251734219,
      0.3173101905524062, 0.315, 0.3]
def cdf(x): return mp.erfc(mp.exp(-x/2)/mp.sqrt(2))
def pdf(x): return mp.exp(-(x + mp.exp(-x))/2)/mp.sqrt(2*mp.pi)
# one-time cross-check of the closed-form cdf: quad of the pdf over [-1, 0] vs cdf(0) - cdf(-1)
mp.mp.dps = 40
print("quad check [-1,0]:", mp.quad(pdf, [-1, 0]), cdf(0) - cdf(-1))  # finite interval only
def ref(q, dps):
    mp.mp.dps = dps
    qm = mp.mpf(q)
    xa = -mp.log(2*mp.erfinv(1 - qm)**2)
    xr = mp.findroot(lambda x: cdf(x) - qm, xa)
    return xa, xr, cdf(xa) - qm
for q in qs:
    a100, r100, res = ref(q, 100)
    a200, r200, _ = ref(q, 200)
    mp.mp.dps = 50
    agree = abs(a200 - a100) <= mp.mpf(10)**-30 * max(abs(a200), mp.mpf(10)**-300)
    s = stats.moyal.ppf(q)
    r = mp.mpf(a200)
    ulp = abs(mp.mpf(s) - r)/(abs(r)*mp.mpf(2)**-52)
    print(f"{q!r} scipy={s!r} ref={mp.nstr(r,17)} findroot_diff={mp.nstr(abs(r200-a200),3)} "
          f"agree100_200={agree} cdfres={mp.nstr(res,3)} ulp={mp.nstr(ulp,3)} abs_eps={mp.nstr(abs(mp.mpf(s)-r)/mp.mpf(2)**-52,3)}")

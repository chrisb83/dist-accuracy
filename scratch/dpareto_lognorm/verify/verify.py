# Independent reference: log X = u + s*N + T, T = E1/a - E2/b (asymmetric Laplace),
# density of T: ab/(a+b) e^{-a t} (t>0), ab/(a+b) e^{b t} (t<0).
# pdf(x) = (1/x) * ab/(a+b) * Int phi(v) h(d - s v)/s ... computed via convolution over v.
import sys, mpmath as mp, numpy as np, scipy.stats as st, warnings
warnings.simplefilter('ignore')

def parts(d, s, a, b):
    z = d / s
    pts = sorted(set([z, a*s, -b*s, 0]))
    lo = [-mp.inf] + pts
    hi = pts + [mp.inf]
    return list(zip(lo, hi))

def integ(f, d, s, a, b):
    z = d / s
    pts = set()
    for c in (z, a*s, -b*s, 0):
        for k in (0, 0.5, 1, 2, 4, 8, 16, 40):
            pts.add(c + k); pts.add(c - k)
    pts = sorted(pts)
    # mp.quad's tolerance is effectively absolute: normalise the integrand
    S = max(abs(f(p)) for p in pts)
    return S * mp.quad(lambda v: f(v) / S, [-mp.inf] + pts + [mp.inf])

def ref(method, x, u, s, a, b):
    x, u, s, a, b = map(mp.mpf, (x, u, s, a, b))
    d = mp.log(x) - u          # log x - u = s*N + T
    C = a*b/(a+b)
    phi = lambda v: mp.exp(-v**2/2)/mp.sqrt(2*mp.pi)
    # pdf of W=log X - u at d: Int phi(v) h(d - s v) dv, h density of T
    def h(t): return C*mp.exp(-a*t) if t > 0 else C*mp.exp(b*t)
    # cdf of T, and its complement
    def H(t): return 1 - b/(a+b)*mp.exp(-a*t) if t > 0 else a/(a+b)*mp.exp(b*t)
    def Hc(t): return b/(a+b)*mp.exp(-a*t) if t > 0 else 1 - a/(a+b)*mp.exp(b*t)
    # P(sN+T<=d) = E[ H(d - s N) ]
    if method in ('pdf', 'logpdf'):
        g = integ(lambda v: phi(v)*h(d - s*v), d, s, a, b)
        val = g / x
    else:
        F = integ(lambda v: phi(v)*H(d - s*v), d, s, a, b)
        S = integ(lambda v: phi(v)*Hc(d - s*v), d, s, a, b)   # direct, not 1 - F
        if method == 'cdf': val = F
        elif method == 'sf': val = S
        elif method == 'logcdf': return mp.log(F) if F < 0.5 else mp.log1p(-S)
        else: return mp.log(S) if S < 0.5 else mp.log1p(-F)
    return mp.log(val) if method.startswith('log') else val

calls = [
 ('logpdf', 1e-290, 0, 0.01, 1, 1),
 ('logpdf', 1e-100, 0, 0.01, 1, 1),
 ('pdf', 1e-290, 0, 0.01, 1, 1),
 ('cdf', 1e-290, 0, 0.01, 1, 1),
 ('logcdf', 1e290, 0, 0.01, 1, 1),
 ('sf', 1e290, 0, 0.01, 1, 1),
 ('logsf', 1e290, 0, 0.01, 1, 1),
 ('cdf', 1e120, 500, 0.1, 2, 3),
 ('pdf', 10, 0, 0.01, 1, 1),
 ('pdf', 9.5, 0, 1, 100, 100),
 ('pdf', 1e-270, 3, 1.2, 1.5, 2),
 ('sf', 1e200, 3, 1.2, 1.5, 2),
 ('logpdf', 1e-170, 500, 0.1, 2, 3),
 ('logsf', 1e220, 3, 1.2, 1.5, 2),
 ('logsf', 1e20, 0, 1, 100, 100),
 ('logsf', 1.886180808490652e+64, -2, 0.5, 5, 0.5),
 ('logsf', 1e300, 0, 10, 2, 3),
 ('logsf', 1e215, 3, 1.2, 1.5, 2),
 ('logsf', 1e210, 3, 1.2, 1.5, 2),
]
def main():
  sel = [int(i) for i in sys.argv[1:]] or range(len(calls))
  for i in sel:
      m, x, *sh = calls[i]
      mp.mp.dps = 100; r1 = ref(m, x, *sh)
      mp.mp.dps = 200; r2 = ref(m, x, *sh)
      agree = mp.nstr(abs(r1-r2)/abs(r2), 3) if r2 != 0 else 'n/a'
      sp = getattr(st.dpareto_lognorm, m)(x, *sh)
      ulp = abs((mp.mpf(float(sp)) - r2)/r2)/mp.mpf(2.0**-52) if np.isfinite(sp) else mp.inf
      print(f"{i}\t{m}({x!r}, {', '.join(map(repr, sh))})\t{sp!r}\t{mp.nstr(r2, 17)}\t{mp.nstr(ulp, 3)}\trel100v200={agree}", flush=True)

if __name__=="__main__": main()

# genhyperbolic hypotheses (SciPy 1.18.1)

Kernel `_stats.genhyperbolic_logpdf` is compiled (.pyx not shipped). From SciPy's
public source (scipy/stats/_stats.pyx), structure as recalled, to be confirmed by output:
  logpdf = logC(p,a,b) + (p-1/2) log s + log(kve(p-1/2, a s)) - a s + b x,  s = sqrt(1 + x^2)
  logC   = (p/2) log((a+b)(a-b)) - log(2pi)/2 - (p-1/2) log a - (log kve(p, g) - g),  g = sqrt((a+b)(a-b))
  pdf    = exp(logpdf)
cdf/sf: integrate.quad of the LowLevelCallable pdf over (-inf, x] / [x, inf), epsrel=1e-10,
  epsabs=0, split at mean = b/g kv(p+1,g)/kv(p,g); result clipped with max(0, min(1, I)).

H1 (p16 / item 16, pattern 3/10) logpdf, pdf: kve(p-1/2, a s) nan for a s >= 2**30 - 0.5 -> nan
   for |x| >= ~1.07e9/a. logpdf is a normal double there -> loss. pdf normal only if a - |b| tiny.
H2 (pattern 3) logpdf, pdf: kve(p, g) overflows (large |p|, small g) -> nan for all x. Also kve(p-1/2, a s).
H3 (pattern 10/9) |b| = a, p < 0 (allowed by _argcheck): g = 0, log(0) - log(kve(p,0)=inf) -> nan.
   pdf nan for all x. cdf: mean nan, quad nan -> min(1, nan) = 1.0 -> cdf = sf = 1.0 everywhere.
H4 (pattern 5) logpdf/pdf: -a s + b x cancels when b ~ a, x >> 1 (abs error ~ a|x| eps in log).
   Also large a: -a s + g in the bulk, abs error ~ a eps.
H5 (pattern 3) a > 1.34e154: (a+b)(a-b) overflows; a >= 2**30: kve(p, g) nan already.
H6 (pattern 10, item 15) cdf/sf: quad epsrel 1e-10 -> up to ~5e5 ulp; tails may lose more;
   nan from integrand -> clipped to 1.0 silently (min(1.0, nan) = 1.0 in Python).
H7 |x| > 1.34e154: x**2 overflows -> s = inf (moot given H1).

# alpha hypotheses (SciPy 1.18.1, lines 546-557)

z = a - 1/x.
- H1 logpdf (pattern 3): _norm_logpdf forms z**2; overflows for |z| > 1.34e154 although
  -z**2/2 is finite up to |z| ~ 1.9e154. Window x in ~[5.3e-155, 7.5e-155] (any moderate a).
- H2 pdf (pattern 4): 1/(x**2) with x**2 subnormal (x < 1.49e-154) while pdf ~ 0.4/x**2
  is normal. Needs a ~ 1/x ~ 1e154 (peak near x = 1/a). Expected small loss (<~10 ulp).
  Also x**2 underflow to 0 -> 1/0 = inf, inf*phi -> inf/nan for x < ~1e-162 if phi(z) != 0.
- H3 pdf/cdf/logpdf: rounding of 1/x inside a - 1/x when a ~ 1/x large (cancellation);
  expected within conditioning wrt x.
- H4 ppf (pattern 8): a - ndtri(q*Phi(a)) cancels as q -> 1; expected ~cond 1/(1-q).
  q*Phi(a) rounding when Phi(a) close to 1 (large a).
- H5 large a: Phi(a) == 1, phi(a) underflow; pdf for large x underflows (true too).
- Inherited sf = 1 - cdf: cancels for large x (sf ~ phi(a)/(Phi(a) x)). logcdf: log(cdf)
  underflow for x < ~1/(a+38). isf = ppf(1-q).

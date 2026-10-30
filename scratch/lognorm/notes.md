# lognorm hypotheses

Code: _lognorm_logpdf (6852-6857): -log(x)**2/(2*s**2) - log(s*x*sqrt(2pi)); pdf = exp(logpdf);
cdf/sf/logcdf/logsf = ndtr/log_ndtr(+-log(x)/s); ppf/isf = exp(+-s*ndtri(q)).

H1 (pattern 3/4) logpdf, pdf: s**2 underflows for s < ~1.5e-154 (0) and is subnormal for
   1e-162 < s < 1.5e-154. At x == 1, log(x)**2 = 0 -> 0/0 = nan (true logpdf = -log(s sqrt(2pi))).
   Near x = 1 with subnormal s**2 -> severe relative error in the quadratic term.
H2 (pattern 3) logpdf: s*x*sqrt(2pi) overflows for s*x > 7e307 -> log = inf -> -inf
   (true logpdf finite, ~ -710 ... ). Needs large s or s >~ 1 with x ~ 1e308.
H3 (pattern 6-like) pdf = exp(logpdf): absolute error of logpdf (~eps*|logpdf|) becomes
   relative error -> up to ~700 ulp where |logpdf| ~ 700 (minor).
H4 logpdf near its zero crossing: cancellation between -L^2/2s^2 and -L - log(s sqrt2pi).
H5 cdf/sf: rounding of z = log(x)/s amplified by z**2 in ndtr tails (as johnsonsb F2);
   cond wrt x is ~|z|/s, so excess ~ |z|*s ... up to ~1e3 ulp near |z| ~ 37.
H6 ppf/isf: exp(s*ndtri(q)) amplifies rounding of s*ndtri by |log x| (up to ~709 ulp).
H7 logpdf with s tiny & x tiny: s*x subnormal -> negligible relative to quadratic term (expect fine).

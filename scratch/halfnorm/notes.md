# halfnorm hypotheses
- H1 ppf (pattern 8): `(1+q)/2` rounds q away for q < eps -> ppf = ndtri(0.5) = 0; partial loss for q < ~1e-8 (relative error ~ eps/q).
- H2 ppf (pattern 8/1): q near 1, 1+q in [1,2) rounds (spacing 2^-52): q = 1-2^-53 -> 1+q = 2 -> inf.
- H3 logpdf (pattern 3): x*x overflows for x in (1.34e154, 1.9e154) while the true logpdf ~ -x^2/2 is finite.
- H4 cdf: x/sqrt(2) subnormal for x < ~3e-308 (only last bits); erf accurate otherwise.
- H5 pdf: exp(-x*x/2) rounding of x*x within cond (x^2). Expected ok.
- isf = -ndtri(p/2), sf = 2*ndtr(-x): expected ok.
- Inherited logcdf = log(erf): loses for large x (cdf rounds to 1); logsf = log(2 ndtr(-x)) underflows x > ~38.6.

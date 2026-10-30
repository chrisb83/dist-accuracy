# loggamma hypotheses

Source lines 6675-6725. _LOGXMIN = log(2.2e-308) = -708.396, _XMIN = 2.2e-308.

1. pdf/logpdf (6677, 6680): `c*x - exp(x) - gammaln(c)`; for large c the three
   terms are ~c log c and cancel to O(log sqrt c) near the mode x ~ log c
   (pattern 5). Expect ~c*log(c)*eps absolute error -> severe for c = 1e6.
2. logpdf for subnormal c: gammaln(c) = inf (INDEX, dgamma F5) -> -inf (pattern 10).
3. cdf/sf (6700, 6716): `gammainc(c, exp(x))`. For large c (>~2e5) gammainc's
   unconverged series (gamma F1) -> severe in left part of bulk (pattern 10, shared).
   Also prefactor rounding (gamma F6) for c = 1e3.
4. cdf/sf: rounding of exp(x) adds ~ (f/F)*eps error, cond = |x| f/F, so beyond
   conditioning where |x| < 1 (x near 0, large c left tail; c small?) - minor.
5. Branch x < _LOGXMIN: one-term series, relative error ~ c e^x/(c+1); fine.
   Check both sides of -708.396.
6. ppf (6705-6709): log(gammaincinv(c, q)); gammaincinv inaccurate for c=1e-3
   (gamma F5) and c=1e6 (gamma F3); log reduces relative error by |log g|.
   Branch g < _XMIN: (log q + gammaln(c+1))/c; check for small c both sides.
7. isf (6721-6725): gammainccinv(c, q); branch with log1p(-q). For q close to 1 and
   small c, gammainccinv(c, q) may be imprecise (it inverts a value near 1).
8. Large c, ppf: (log q + gammaln(c+1))/c branch for g < XMIN never used for c>=1
   unless q tiny: c=1, q=1e-300 -> g = 1e-300 not < XMIN. ok.

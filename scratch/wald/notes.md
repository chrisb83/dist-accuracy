# wald hypotheses (wald = invgauss with mu = 1; wrappers lines 11415-11438 just call invgauss._xxx(x, 1.0))
H0 wrappers: no arithmetic of their own; mu = 1.0 exact -> 2/mu = 2 exact, x/mu = x exact. Expect no extra error.
H1 (p1/p5, invgauss F1) sf/logsf/logcdf right tail: log1p(-exp(b-a)) cancels, error ~x ulp; sf normal up to x ~ 1416.
H2 (invgauss F2) isf: Boost _invgauss_isf returns search bound for q <~ 1e-76; ppf for q>0.5 uses isf(1-q) >= 1.1e-16, fine.
H3 (p3, invgauss F6) logpdf: (x/1 - 1)**2 overflows for x > 1.34e154 -> -inf, true -x/2.
H4 (p3, invgauss F7) pdf: x**3 overflow at x > 5.6e102 - pdf already underflows there for mu = 1, not reachable.
H5 pdf x < ~1e-103: x**3 underflows -> inf*0 = nan, true pdf underflows (exp(-1/(2x))) -> not measurable.
H6 cdf left tail: exp(logcdf) with |logcdf| up to 709 -> up to ~700 ulp, within cond.

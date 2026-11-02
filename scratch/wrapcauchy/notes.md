# wrapcauchy hypotheses

- H1 pdf (line 11480), pattern 5/1: denominator 1+c*c-2*c*cos(x) cancels to (1-c)^2 + 4c sin^2(x/2)
  when c -> 1 and x near 0 or 2pi (the peak). Expect severe/loss for 1-c <= 1e-6.
- H2 pdf (line 11480), pattern 1: 1.0-c*c with c near 1: rounding of c*c, ~1/(1-c) ulp.
- H3 cdf f2 (line 11490): uses 2*np.pi - x; constant error 2.4e-16 absolute only hurts 1-cdf part,
  cdf ~ 1 there, so expect fine. sf = 1 - cdf inherited -> opportunity.
- H4 ppf (lines 11497-11498): tan(pi*q) near q=1/2 (pi*q rounding), q just below 1/2, c near 1.
  Probably within conditioning. q near 1: 1-q exact.
- H5 cdf f1 near x=pi (tan(x/2) huge): arctan saturates; fine expected.
- H6 tiny c: cr, val round to 1; expect fine.

# levy_l hypotheses
- H1 (pattern 1) cdf line 6494: 2*ndtr(1/sqrt(ax)) - 1 cancels for large |x| (cdf -> 0); total loss for |x| > ~1e32.
- H2 (pattern 8/1) ppf line 6501: (q + 1.0)/2 rounds away small q; ppf -> -inf for q < 1.1e-16, large errors for q < 1e-3.
- H3 (pattern 4) pdf line 6490: exp(-1/(2ax)) subnormal window ax ~ 7e-4 (as in levy).
- H4 isf line 6505: p/2 subnormal for p < 4.45e-308; ndtri near 0.5 for p -> 1.
- H5 ppf q -> 1: q+1 rounding loses half ulp of 1-q, within conditioning expected.

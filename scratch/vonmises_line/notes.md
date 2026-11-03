# vonmises_line hypotheses
- cdf series branch (kappa < 50): 0.5 + x/(2pi) + V/pi, absolute error ~eps -> left tail (x near -pi, large kappa) loses relative accuracy beyond cond; clipped to 0 when negative (patterns 1/5, 10).
- cdf normal approx (kappa >= 50): approximation, not exact; O(1/kappa) relative error, worse in tails (pattern 7). Discontinuity at kappa=50 (seen: cdf(-1, 49.99)=6.94e-12, cdf(-1,50)=6.76e-12).
- cdf right half: result near 1 is fine (absolute), sf inherited 1 - cdf fails.
- pdf: exp(kappa*cosm1(x)) error ~|kappa*cosm1(x)| ulp (pattern 6-like), minor vs cond.
- logpdf: cancellation where logpdf ~ 0 (pattern 5), covered by cond.
- kappa = 0: series with R = k/(...)?

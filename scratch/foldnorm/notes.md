# foldnorm hypotheses
- H1 (pattern 5/10) cdf line 2703: erf((x-c)/s2) + erf((x+c)/s2) cancels when x << c (erf(x-c) ~ -erf(c)); cdf ~ 2 x phi(c). Small x any c>0, and moderate x with large c (both erf -> +-1 => 0).
- H2 cdf with tiny c, tiny x: terms ~c, result ~x -> loss ~c/x ulp.
- H3 pdf line 2699: exp(-(x-c)**2/2) for large x,c: z rounding; check vs conditioning (expected within cond).
- H4 sf line 2706: ndtr(c-x)+ndtr(-x-c) both positive; expect fine. Near x small: sf ~1 fine.
- Inherited: logpdf underflow for x > c+38; logsf underflow; logcdf from cdf; ppf/isf numerical.

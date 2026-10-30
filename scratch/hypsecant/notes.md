# hypsecant hypotheses
- pdf 1/(pi*cosh(x)): cosh overflows |x|>710.5, but true pdf < 2.2e-308 already for |x|>~708 -> expect nothing (pattern 3, unmeasurable).
- cdf/sf arctan(exp(+-x)): exp accurate, arctan accurate; expect ok. exp underflow x<-745 -> 0 where true subnormal anyway.
- ppf log(tan(pi*q/2)): q near 0.5: tan ~ 1, log(1+u) cancellation (pattern 2) -- but ppf is ill-conditioned near median (cond ~ 0.5/|q-0.5|) so probably within cond.
  q near 1: pi*q/2 near pole, np.pi rounding (pattern 11/8). cond ~ 1/((1-q) |x|). Check vs. cond.
  q tiny: tan(pi q/2) ~ pi q/2, fine; q < ~1e-308*2/pi subnormal product? pi*q/2 for q=1e-300 fine.
- isf: same as ppf mirrored; isf near q=1 pole.
- inherited logpdf=log(pdf): -inf for |x|>~708 (pattern 12).
- inherited logcdf (median switch): log(cdf) x<0 -> -inf x<-708; logsf mirrored.

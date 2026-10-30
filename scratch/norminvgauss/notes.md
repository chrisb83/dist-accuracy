# norminvgauss hypotheses

- H1 (pattern 5) pdf line 5589: `b*x - a*sq + gamma` cancels for large a (bulk, x ~ 1/sqrt(a)):
  abs err ~ a*eps in exponent -> ~a ulp, cond ~ a x^2 ~ 1.
- H2 (pattern 5) pdf: b close to a, large x>0: b*x - a*sq ~ (b-a)x, abs err ~ a|x| eps; cond ~ (a-b)|x|.
- H3 (pattern 3) pdf: (a+b)*(a-b) overflows for a > ~1.3e154 -> gamma = inf -> inf/nan.
- H4 (pattern 4) gamma underflows for a < ~1e-162 -> mean = b/0 -> isf start point nan/inf.
- H5 (pattern 10) cdf/sf: integrate.quad default epsabs=epsrel=1.49e-8 -> bulk errors up to ~1e7 ulp,
  tails (values < 1.5e-8) possibly relative error ~1.
- H6 cdf/sf far tails: quad over [x, inf) for x far from peak; check 0 returns.
- H7 isf: brentq xtol=1e-14 absolute -> results with |x| << 1 (q near median) lose relative accuracy;
  plus propagates sf quad error (H5).
- H8 isf: bracket loop doubling from mean; huge results / q tiny -> many quad calls; q near 1 uses
  sf = 1 - quad(-inf, x) only for x < -|mean|.

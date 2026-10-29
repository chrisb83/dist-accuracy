# genextreme hypotheses

- H1 (pattern 3): `cx = c*x` overflows to -inf for |c x| > 1.8e308 (|c| > 1, x in the
  long tail). logpdf -> fill -inf (cx == -inf condition), loglogcdf = log1p(inf)/c = +-inf
  -> cdf 0 or 1, sf 1 or 0. True values are normal when (|c|*|x|)**(1/c) is moderate,
  i.e. |c| > ~108. Methods: logpdf, pdf, cdf, logcdf, sf.
- H2 (pattern 3/8): ppf/isf `-expm1(-c*x)/c` overflows for -c*x in (709.78, 709.78+log|c|)
  while the result is finite. Needs |c| > ~1; window in q is narrow.
- H3 (pattern 5): logpdf `-pex2 + lpex2 - lex2`: lpex2 - lex2 = L(1/c - 1) cancels for c near 1
  near the upper endpoint (L = log1p(-cx) very negative). Probably within cond(x).
- H4 (pattern 9): c -> 0 handled by log1p/expm1; c*x subnormal for |c| < ~1e-300 (not tested
  heavily; meaningless shapes).
- H5 (pattern 7): near endpoint 1/c, cx rounds to 1 -> log1p(-1) = -inf; conditioning huge.
- Opportunity: logsf = log(sf) underflows when exp(ll) < 1e-308 (right tail);
  logsf = ll + log(-expm1(-t)/t), t = exp(ll).

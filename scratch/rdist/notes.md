# rdist hypotheses (a = c/2, y = (x+1)/2)

All methods go through beta with a = b = c/2.

1. pdf/logpdf, large c (pattern 5 + betaln error, beta F4): logpdf = -log2 + xlog1py(a-1,-y) + xlogy(a-1,y) - betaln(a,a);
   near x = 0 the terms are ~1.39*a and cancel to O(log c). pdf = exp(logpdf) turns the absolute error into relative error.
   Regime: c >= 1e3, x near 0 (bulk).
2. logpdf/pdf/sf near x = 1 (pattern 10): y = (x+1)/2 rounds for x in (0.5, 1) (x+1 in [1.5, 2) loses one bit);
   1-y then has relative error ~eps/(1-x). Probably within cond (cond ~ (a-1)/(1-x) for pdf). Check c = 1.6 and c = 1000.
   Also x in (-0.5, 0): 1 + x rounds relative eps/2, cond O(1): fine.
3. sf(x) = betaincc(a,a,(x+1)/2): near x = 1 same argument rounding; compare against cdf(-x) by symmetry.
4. ppf (pattern 5/10): 2*p - 1 with p from Boost ibeta_inv; near q = 1/2 (all c) and for large c everywhere (x ~ 1/sqrt(c), lose sqrt(c)).
   Boost ibeta_inv failure bands at small q (beta F1/F7), visible only when p is not tiny (large c).
5. ppf, tiny c: p is extremely close to 0/1; x = 2p - 1 near +1 loses relative nothing. Check q around 1/2 (bimodal).
6. Inherited isf = ppf(1-q): 1 - q rounds (pattern 8); better -ppf(q) or 2*beta._isf(q,a,a)-1.
7. Inherited logcdf/logsf (median split, log(cdf)/log1p(-sf)): fine unless cdf underflows (large c, x near -1); a log-space beta tail would help.

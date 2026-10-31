# t hypotheses (SciPy 1.18.1, lines 8022-8131)

1. logpdf/pdf, pattern 3: `x * x/df` (line 8070) overflows for |x| > 1.34e154
   (any df) or x*x/df > 1.8e308 for tiny df (df=1e-3: |x| > 4.2e152) ->
   log1p(inf)=inf -> logpdf -inf, pdf 0 where true value is normal (small df).
2. logpdf, pattern 5: `log(poch(df/2, 1/2)) - 0.5*log(df)` cancels for huge df
   (both ~0.5 log df); loss ~ log(df)/0.92 ulp - minor/moderate only for df >~1e100.
3. pdf = exp(logpdf): error ~ |logpdf| ulp absolute -> relative in pdf (pattern 5/6-ish), up to ~700 ulp; check against cond.
4. logpdf, pattern 6-like: (df+1)/2*log1p(x*x/df) for large df: x*x/df rounding
   amplified by (df+1)/2 * u/(1+u)... error relative to log1p result: fine (product of rounded terms).
5. df == inf branch: norm._logpdf uses _norm_logpdf `-x**2/2` (INDEX item 2).
6. cdf/sf/ppf/isf: sc.stdtr / sc.stdtrit (special functions). Check tiny df (1e-3, 1e-300),
   huge df (1e10, 1e300), far tails, q extremes. stdtrit tiny q / tiny df -> overflow region.
7. sf = stdtr(df, -x), isf = -stdtrit(df, q): symmetric, no cancellation.
8. ppf near q=0.5: stdtrit accuracy for result near 0.

## Outcome (run3, scratch/t/run3.txt / run3.json)
1 -> F2 confirmed (loss). 2 -> F10 minor (29 ulp at 1e100). 3 -> F9 minor (554 ulp).
4 -> fine. 5 -> F3 confirmed (INDEX item 2). 6 -> F1 (stdtr t*t overflow), F4 (stdtrit cap),
F5 (stdtrit tail factor 2 / +inf), F6 (stdtr df==1 near 0), F7 (stdtrit 2<df<3), F8 (df=0.001).
Note: run1 cache was corrupted (values ~1e-17 off) - probably a harness timeout during the
ppf(0.1, 0.001) self-check search; cache deleted, ppf now returns +-inf when the root > 1.8e308.

# anglit hypotheses
- H1 (pattern 11/7-ish) cdf line 594: x + np.pi/4 near x=-pi/4; np.pi/4 off by ~3.1e-17. Error ~0.28/t ulp, cond 1.57/t -> expect within cond.
- H2 sf line 597: x + np.pi/4 near x = pi/4 rounds (ulp of 1.57); cos near pi/2. Expect within cond (~1 ulp perturbation of x).
- H3 ppf line 600: arcsin(sqrt(q)) - pi/4 cancels near q=0.5 (pattern 5/8). Abs err ~1e-16 vs cond 0.5/|y| -> maybe minor.
- H4 ppf q->1: sqrt(q) rounding amplified by arcsin' -> ~cond. q->0 fine.
- H5 pdf: cos(2x) near +-pi/4: 2x exact, libm cos accurate -> fine.
- Inherited: isf = ppf(1-q) loses for small q (pattern 8); logpdf = log(cos 2x) loses for |x|<1e-8 (pattern 2); logcdf near x->pi/4, logsf near x->-pi/4 (log of ~1) fine-ish but loses relative accuracy when result tiny (pattern 1/2).

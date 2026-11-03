# norm hypotheses
- H1 logpdf (P3): `-x**2/2.0` line 367: x**2 overflows for |x| > 1.34e154 though -x**2/2 is finite up to |x| ~ 1.896e154 -> -inf (loss). Same as halfnorm F3.
- H2 pdf (P3/P11): `exp(-x**2/2)` line 363: x**2 rounded before exp; error ~ x**2/4 ulp; cond(arg)=x**2 so expected within conditioning. Check x in 10..38.6.
- H3 constants (P11): _norm_pdf_C = sqrt(2pi), _norm_pdf_logC = log(sqrt(2pi)) (two roundings) -> <= ~1 ulp off; check in bulk/near x=0.
- H4 logcdf/logsf (P10): log_ndtr branches at x = -1 (log(erfcx(-t)/2) - t*t vs log1p(-erfc(t)/2)); t = x/sqrt2 rounded; huge |x| where t*t overflows.
- H5 cdf/sf: ndtr far left tail (-37.5 .. -38.4 subnormal), tiny |x|.
- H6 ppf/isf (P8): ndtri near q=0.5 (cond large), q -> 1 (uses 1-q, exact), tiny q 1e-300; isf = -ndtri(q).

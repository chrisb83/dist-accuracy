# levy hypotheses
- H1 pdf (l.6386): exp(-1/(2x)) subnormal for 1/(2x) in (708, 745) while the pdf is normal (prefactor x^-1.5 ~ 1e4.7): pattern 4. x ~ 6.7e-4 .. 7.06e-4.
- H2 sf (l.6393): 0.5/x subnormal for x > 2.2e307 (pattern 4); sf ~ 1e-154 normal -> precision loss, then 0 for x > ~1.1e308*... .
- H3 ppf (l.6397): q/2 subnormal for q < 4.45e-308 (pattern 4); ndtri of subnormal.
- H4 isf (l.6401): erfinv(p)**2 underflows/subnormal for p < ~1.5e-154 while isf ~ 2/(pi p^2) may still be normal near p~1.5e-154 (pattern 3/4).
- H5 cdf (l.6390): erfc large arg (small x) - rounding of sqrt(0.5/x) amplified by 2 z^2; this is within cond wrt x. Check.
- H6 ppf near q->1: _norm_isf(q/2) near 0.5 -> small val, 1/val^2; cond large anyway.
- Inherited: logpdf, logcdf (log(erfc) underflow small x), logsf (log(erf) fine?).

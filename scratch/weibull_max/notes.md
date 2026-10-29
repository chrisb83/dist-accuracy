# weibull_max hypotheses (t = (-x)**c)
- H1 pdf (l.3048) pattern 4: exp(-t) subnormal/0 for t > 708 while c*(-x)**(c-1)*exp(-t) normal (large c). Likely within cond (cond ~ c*t).
- H2 pdf/cdf/logpdf pattern 6: rounding of pow(-x,c) amplified by t in exp(-t): up to ~t ulp; cond(x) = c*t, so beyond cond only for c < 1.
- H3 logpdf (l.3051) pattern 5: log(c) + (c-1)log(-x) - t near zero crossing; small c with large |log(-x)|.
- H4 ppf (l.3063) pattern 6/8: rounding of log(q) raised to 1/c: ~1/(2c) ulp for small c, cond = 1/(c|log q|) -> beyond cond for small q, small c.
- H5 sf: -expm1(-t) fine; t underflow for x near 0 only when true sf subnormal.
- Inherited: isf = ppf(1-q) -> loses small q (pattern 8). logsf = log(sf): -inf when t < 1e-308.

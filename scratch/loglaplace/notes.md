# loglaplace hypotheses
- H1 pdf (l.6793-6794): exponent c-1 / -c-1 rounded (pattern 4 of INDEX / pattern 6-like); error ~ |log x| * rounding(c+-1); c not dyadic (0.3, 0.7), x far from 1.
- H2 pdf: x**(c-1) subnormal for large c near x<1 but times c/2 lifts it (pattern 4); c=1e6.
- H3 ppf/isf (l.6803, 6806): 1/c rounded, amplified by |log(2q)|; c=0.3, 3, q tiny or near 1.
- H4 cdf/sf branches at x=1 and q=0.5: complement side 1 - 0.5*x**+-c >= 0.5, no cancellation; 1-q exact for q>=0.5 (Sterbenz). Expected fine.
- H5 tiny c (1e-3): ppf (2q)**(1/c) over/underflow only where the result does too.
- Inherited logcdf/logsf/logpdf: log(cdf) underflows for x<1 when c*log x < -745; logsf for x>1. Closed forms exist.

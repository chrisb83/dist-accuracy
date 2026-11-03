# halfcauchy hypotheses
- H1 logpdf (l.4542) `log(2/pi) - log1p(x*x)`: x*x overflows for x > ~1.34e154 -> -inf, true ~ log(2/pi) - 2 log x (normal). Pattern 3. Expect loss for x in (1.34e154, 1.8e308].
- H2 pdf (l.4539) `2/pi/(1+x*x)`: overflow of x*x only where true pdf < 3.5e-309 (subnormal) -> not measurable. Pattern 3.
- H3 ppf (l.4548) `tan(pi/2*q)` q->1: np.pi/2 rounding (6e-17) vs gap to pole; cond ~1/(1-q) should cover. Pattern 8/11.
- H4 isf (l.4554) `1/tan(pi*p/2)`: p->1 result small, cond ~1/(1-p). p tiny fine. Pattern 8.
- H5 cdf arctan(x) / sf arctan2(1,x): expected fine both tails.
- Inherited logcdf=log(cdf), logsf=log(sf): sf subnormal only for x>~1.8e308 -> fine; logcdf fine.

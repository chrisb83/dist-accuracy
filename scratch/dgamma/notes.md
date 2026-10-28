# dgamma hypotheses

1. pdf (line 1800) `1.0/(2*sc.gamma(a))*ax**(a-1.0)*np.exp(-ax)`:
   - pattern 3: sc.gamma(a) overflows for a > 171.62 -> 1/inf = 0 -> pdf 0 (or 0*inf = nan) everywhere. Loss, all x.
   - pattern 3: ax**(a-1) overflows when (a-1)*log|x| > 709.78 (e.g. a=150, |x| > 117) although pdf normal -> inf.
   - pattern 4: exp(-ax) underflows for |x| > 745 while x**(a-1)/Gamma(a) large (needs a ~> 100s; overflow of pow comes first though).
   - a < 1, x -> 0: ax**(a-1) up to 1e300, fine.
   - a tiny (< 5.6e-309): gamma(a) overflows (subnormal shape; edge).
2. logpdf (line 1804): cancellation of xlogy(a-1,|x|), |x|, gammaln(a) for large a near |x|~a (pattern 5; same as gamma F4).
3. cdf/sf (1807-1814): 0.5 + 0.5*gammainc: fine (result >= 0.5). Large a: gammainc F1 (gamma report) enters only additively -> diluted.
   x<=0 branch 0.5*gammaincc(a,-x): accurate as gammaincc; large a F1 again diluted (Q ~ 1 there).
4. ppf/isf (1819-1827): 2q-1 exact for q in (0.5,1) (Sterbenz); 2q exact. Near q = 0.5 the quantile is gammaincinv(a, small) -> gamma F3/F5 apply undiluted: a<<1 (F5) and a=1e6 (F3).
5. np.where evaluates both branches -> warnings possible (gammainc with negative arg returns nan silently?).

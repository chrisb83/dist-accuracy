# irwinhall hypotheses
Code: pdf = BSpline.basis_element(arange(n+1))(x) (de Boor, k = n-1); cdf = antiderivative spline (splantider: cumsum of coeffs*dt/(k+1)), sf = cdf(n - x).
No explicit alternating sum in the scipy code (comb/stirling2 are only in _munp).
H1 (pattern 6/accumulation): de Boor recursion is a convex combination, error grows ~ O(n) or O(n^2) ulp for large n. Test n = 100, 300, 1000 bulk.
H2 (pattern 4): left tail pdf ~ x^(n-1)/(n-1)!: de Boor products of small ratios -> subnormal intermediate near underflow edge. Test x where pdf ~ 1e-300.
H3 (pattern 1): sf = cdf(n - x): n - x rounds for small x but sf ~ 1 there, well conditioned. cdf near 1 for x near n: spline sum near 1 -> fine. Check sf at x tiny (1e-300) and cdf right tail.
H4 (pattern 7): knot points x = integer k exactly (interval selection), x = n endpoint, x = 0.
H5: n = 1 (uniform), n = 2 (triangle) exactness.
H6: antiderivative padding knots (t0-1) with extrapolate=True: evaluation exactly at x = n uses last interval?

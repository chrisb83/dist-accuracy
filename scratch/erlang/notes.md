# erlang hypotheses
erlang_gen (3794-3846) overrides only _argcheck, _shape_info, _fitstart, fit.
All numerical methods are gamma_gen's -> gamma findings F1-F6 apply where their
regime contains integer a.
- F1/F3 (gammainc unconverged, a >~ 2e5): a = 1e6, 1e8 are integers -> reach erlang.
- F2/F4 (logpdf cancellation, large a): a = 1e3, 1e6 integers -> reach.
- F5 (gammaincinv for a << 1): outside a >= 1; but _argcheck returns a > 0 (only warns),
  so non-integer a < 1 is accepted -> reachable only via the warned path.
- F6 (igam_fac prefactor): a = 1e3 x=500/2000, and x < 1e-50 for a=1 -> reach.
- a = 1: xlogy(0,x) exact; logpdf = -x exact.
- _fitstart: int(4/(1e-8+skew^2)) -> 0 when |skew| > 2 -> invalid start a=0 (not accuracy).
- _argcheck: np.all(np.floor(a)==a) on the whole array; a=inf? floor(inf)==inf True.

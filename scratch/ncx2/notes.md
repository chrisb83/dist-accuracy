# ncx2 hypotheses

Code: lines 7811-7824 (_ncx2_log_pdf), 7874-7901 (methods). All non-log methods
are Boost non_central_chi_squared (pdf, cdf=chndtr, sf, ppf=chndtrix, isf);
nc == 0 dispatches to chi2 (so chi2 findings apply there; no ncf-F1 analogue
expected because SciPy, not Boost, handles nc == 0).

Boost structure (from memory of Boost source, not inspected here):
- cdf/sf: x > k + l -> Q series started at the Poisson mode (k = round(l/2)),
  else l < 200 -> Ding's method from j = 0, else P series from the mode.
- pdf: l <= 50 -> exp(log(x/l)(k/4-1/2) - (x+l)/2) * I_{k/2-1}(sqrt(lx)) / 2
  unless |r| >= log_max/4 -> series started at the Poisson mode,
  `if (pois == 0) return 0`.

H1 (pattern 4/10) pdf, l > 50 (or |r| large): the mode-start term
  gamma_p_derivative(k + df/2, x/2) underflows while the sum is normal ->
  returns 0. Regime: left tail x << nc, large nc; right tail x >> nc.
H2 (pattern 4/10) sf right tail, x > df + nc: gamma_q(df/2 + k, x/2) at the
  Poisson mode underflows while sf ~ exp(-(sqrt x - sqrt nc)^2/2) is normal.
  e.g. nc = 1000, x = 4000. Same for cdf left tail via P series (nc >= 200).
H3 (pattern 4) logpdf: ive(df/2 - 1, sqrt(nc x)) underflows for small
  sqrt(nc x) with order > 0 (tiny x, or large df) -> fill -inf.
  df = 1000, nc = 1 in the bulk: ive(499, 31.6) ~ e^-1233 -> -inf (NOTE).
H4 (pattern 3) logpdf: x/nc overflows (x = 1e300, nc = 1e-10) -> xlogy = inf
  -> +inf; or underflows (tiny x, big nc).
H5 (pattern 5) logpdf: xlogy((df/2-1)/2, x/nc) and log(ive) large, opposite
  sign, cancel (large df, small nc).
H6 (pattern 8/10) ppf/isf: Boost root finder tolerance; isf for small q
  inherits H2 (sf -> 0).
H7 Boost series max-iteration ("Series did not converge") for large nc or x.
H8 tiny nc (1e-10): continuity with chi2; Boost pdf l<=50 path:
  log(x/l) * (k/4 - 1/2) with x/l huge -> |r| large -> series path.
H9 df < 2 near x -> 0 (pdf pole), df tiny.

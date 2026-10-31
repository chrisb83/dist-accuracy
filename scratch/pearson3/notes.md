# pearson3 hypotheses (SciPy 1.18.1, lines 8432-8625)

Map: beta = 2/skew, alpha = 4/skew**2, zeta = -2/skew, t = beta*(x - zeta) = alpha + beta*x.
skew>0: cdf = P(alpha, t), sf = Q(alpha, t); skew<0: cdf = Q(alpha, t), sf = P(alpha, t).

H1 (pattern 7/9) |skew| < 1.6e-5 -> normal branch. Not an accuracy-preserving limit:
   relative error ~ |skew|/6*He3(z) in bulk (~1e-6 => ~1e9 ulp), grows like
   exp(skew z^3/6) in tails. pdf, logpdf, cdf, sf, ppf all affected.
H2 (pattern 10, gamma F1) just above the threshold alpha ~ 1.6e10; also skew ~1e-3 (alpha 4e6):
   sc.gammainc unconverged series for lower-gamma-tail with 4.5 < -z < 15 (in t sd units).
   skew>0: cdf left tail; skew<0: sf right tail (gamma.cdf). Also ppf via gammaincinv.
H3 (pattern 5) t = beta*(x - zeta): x - zeta = x + 2/skew rounds with abs error (2/|skew|)eps
   -> cdf/sf/pdf error ~ (2/|skew|) * |dlogF/dx| ulp, beyond cond (|x| dlogF/dx) when |x| << 2/|skew|.
H4 (pattern 5/8) ppf: gammaincinv(alpha,q)/beta + zeta cancels: two terms ~2/|skew| -> result O(1).
   error ~ (2/|skew|)/|ppf| ulp. Small skew, bulk q.
H5 (pattern 1/8) ppf for skew<0: q -> 1 - q (line 8611): small q rounded away -> ppf(1e-20, -2) = -inf?
H6 (gamma F2/F4) logpdf/pdf for large alpha: gamma.logpdf cancels alpha*log(alpha) terms.
H7 (gamma F5) huge |skew| -> alpha << 1: ppf via gammaincinv loses ~|log x| ulp.
H8 inherited isf = ppf(1-q): skew>0 small q loses; logcdf/logsf = log(cdf/sf) underflow.

## Restart (2026-10-31)
- Main grid hung in the condition-number step (harness timeouts are Unix-only;
  cond is uncached and the bisection ppf reference is evaluated ~8x per point).
  Now: PARAMS without +-0.01; pdf/logpdf/cdf/sf at 400 digits (run3_pdfcdf.txt),
  ppf at --dps 100 (run3_ppf.txt). Tiny |skew| only via --at (at_batch.sh -> at_H*.txt).
- pearson3_tiny_ref.py: sf now integrates the right tail directly (1 - cdf at 100
  digits gave 0 for sf < 1e-100).
- H1 confirmed (at_H1.txt), H2 confirmed (at_H2.txt, rel. err 0.92 at alpha=1.56e10),
  H3/H4 confirmed at skew 1.7e-5 (at_H3.txt; skew 1.6e-5 / 1e-3 give exact zeta = -1.25e5 / -2000,
  so x - zeta is often exact there), H6 confirmed (pdf/logpdf at alpha 4e6, 1.56e10).
- Opportunities (opp.py): logsf(200,0.1) -inf vs -3047.06; logcdf(-19.9,0.1) -inf vs -1725.24;
  isf(1e-20,0.1) inf vs 10.72; isf(1e-20,-0.1) 20.0 vs 7.90.

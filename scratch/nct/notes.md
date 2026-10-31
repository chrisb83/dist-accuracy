# nct hypotheses (SciPy 1.18.1, all methods wrap Boost non_central_t)

All five implemented methods are thin wrappers (lines 8183-8198); no SciPy-side
arithmetic except np.clip(sf, 0, 1). Hypotheses come from the probe (probe.py):

H1 cdf, t < 0, large nc > 0 (30, 20), (3, 38), (10, 40): cdf(-10, 30, 20) = 8.2e-24 > Phi(-20) = 2.8e-89,
   impossible (cdf is increasing and cdf(0) = Phi(-nc)). Left tail for large nc is wrong (Boost reflection
   t<0 -> 1 - F(-t; -nc) or series start at Poisson mode). (3, 38) cdf(-1e20) = nan.
H2 pdf, OverflowError from ibeta_derivative for |x| >~ 1e154 (t*t overflow) -- small df (1, 0.5) and also
   (30, 20) at -1e154 (pattern 3, t.md F1 analogue). Also pdf/cdf/sf at |x| >= 1e154 return 0/1 where
   tail values are normal for small df (t.md F1 analogue).
H3 df = 1e6: OverflowError (tgamma too large) for |x| >= 10 in some method (pattern 3).
H4 pdf large nc: pdf returns 0 in wide regions (10, 40) at t = 0, 1 -- true value normal? Poisson-mode
   start underflow (ncx2 F1 analogue). gh-19348 (pdf inf for large t and nc).
H5 ppf/isf: clamp at 1.34e154 for small df (stdtrit cap analogue, t.md F4); quantiles for small q with
   large nc (ppf(1e-100, 30, 20) = -52.4 vs cdf(-10) = 8e-24 inconsistent) inherit H1.
H6 sf right tail: for (30, 20) sf(1e5) normal; check pattern 1 (1 - cdf) where Boost uses complement.
H7 nc = 0: delegates to students_t? check continuity at tiny nc (1e-300) vs nc = 0.
H8 negative nc: reflection t -> -t, nc -> -nc; check symmetry.

## Outcome (see reports/nct.md)
H1 confirmed (F1, also mirrored sf for nc<0, pdf F8, quantiles F12). H2 confirmed (F2 cdf/sf, F3 pdf OverflowError).
H3 confirmed (F6: tgamma OverflowError / nan in pdf for df >= 200, left tail). Additional: F5 pdf bulk error ~0.2*df ulp.
H4: pdf zeros only in far right tail (F9); gh-19348 inf not reproduced. H5 confirmed (F11 saturation, F12).
H6: sf right tail fine for nc > 0. H7: nc = 1e-8 fine (cdf/sf/pdf <= 5 ulp), except ppf nan at q=1e-300 (F13).
New: F4 (df == 1, tiny |x| nan), F7 (pdf t~0 shortcut), F10 (df=1e6 bulk moderate).
Runs: runx (pdf/cdf/sf, 12 sets; pdf at df>1e5 superseded by runpdf after a reference fix), runpdf, runq (quantiles, 6 sets).
run1 (all methods) was aborted: cond computation for quantiles at large nc too slow without timeouts on Windows.

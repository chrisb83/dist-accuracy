# mielke: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `mielke_gen`, lines 7193-7252
  (`_pdf` 7232-7233, `_logpdf` 7235-7238, `_cdf` 7240-7241, `_ppf` 7243-7245)
- **Implemented:** pdf, logpdf, cdf, ppf (all written out in `mielke_gen`; no delegation)
- **Inherited (not measured):** logcdf = log(cdf), sf = 1 - cdf, logsf = log(sf), isf = ppf(1 - q)
- **Related:** mielke(k, s) is burr(c = s, d = k/s) (Dagum / Burr III), and the same
  `x**power` / `(1+x**s)**power` idioms appear in burr, burr12 and fisk; helpers: none;
  special functions: none in the measured methods (`sc.gamma` only in `_munp`)
- **Reference:** `refs/mielke_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** (10.4, 4.6) [inventory], (2, 3), (1, 1), (0.5, 2), (1e-3, 1e-3),
  (1e6, 1e-3), (1e-3, 1e6), (300, 4.6), (1000, 1), (1, 1e-3), (1e6, 1)
- **Worst finding:** loss in cdf, pdf, logpdf and ppf

## Summary

mielke is written with *positive* powers of x (`x**k`, `x**s`, `(1+x**s)**(k/s)`),
which makes it very accurate for x < 1 (over 332 such cdf points the worst error
is 279 ulp, and 11.5 ulp for logpdf) but destroys the right tail: `cdf` and `pdf`
return `nan` or `0.0` as soon as the intermediate `(1+x**s)**(k/s)` passes
1.8e308, and `logpdf` returns `-inf` as soon as `x**s` does, although the true
values are ordinary doubles. This is not a far-tail curiosity: `mielke.cdf(1e10,
1e6, 1.0)` is `nan` where the true value is 0.9999, `mielke.cdf(2.0, 1000, 1.0)`
is `0.0` where the true value is 8.1e-177, and for k >= 300 with s = 4.6 the cdf
is already `nan` at x = 11. 474 of 1236 measurable cdf points, 207 of 930 pdf
points and 222 of 1779 logpdf points are lost. `ppf` fails at both ends of the q
range: `1.0 - q**(s/k)` cancels when `s/k` is small (relative error ~ eps·k/(s²(1-q)),
up to 4.5e14 ulp, `inf` once `q**(s/k)` rounds to 1, and already 9.1e5 ulp at the
central value q = 0.85 for k/s = 1e6), and `q**(s/k)` underflows for small q,
giving `0.0` where the quantile is a normal double (1.5e15 ulp just above the
cliff). Three further mechanisms are smaller: log-space cancellation in `logpdf`
for k >> s (3.9e5 ulp), the `1/s` power amplifying rounding in `ppf` (1.1e3 ulp),
and the inexact exponent `k/s` amplified by `log(1+x**s)` in cdf/pdf (673 ulp,
occasionally returning a cdf > 1). The four inherited methods (sf, logsf, logcdf,
isf) fail at their first demonstration point each.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | cdf | x above min((exp(709.8·s/k)-1)^(1/s), exp(709.8/s)); all shapes | `(1+x**s)**(k/s)` (and `x**k`, `x**s`) overflow -> `inf/inf = nan` or `finite/inf = 0` (pattern 3) | nan for 1.0; 0.0 for 5.0e-294 | loss |
| F2 | pdf | x above min((exp(709.8·s/(s+k))-1)^(1/s), exp(709.8/s)) | same, with exponent `1+k/s`: denominator overflows first -> 0.0, or nan when `x**(k-1)` overflows too (pattern 3) | 0.0 for 8.6e-115; nan for 4.4e-4 | loss |
| F3 | logpdf | x > exp(709.8/s), any k | `x**s` overflows inside `log1p` -> `log1p(inf) = inf` -> `-inf` (pattern 3) | -inf for -900.27 | loss |
| F4 | ppf | q -> 1, worst for s/k << 1 (reaches q = 0.85 at k/s = 1e6) | `1.0 - q**(s/k)` cancels; = 0 once (s/k)(1-q) < eps/2 -> `inf` (patterns 1, 8) | inf; 4.5e14 ulp | loss |
| F5 | ppf | q < exp(-708·k/s), i.e. `q**(s/k)` subnormal or 0, while the quantile is normal | underflow of `pow(q, s/k)`, then `**(1/s)` lifts it back (patterns 4, 8) | 0.0 for 1e-150; 1.5e15 ulp | loss |
| F6 | pdf | k < 1 and x subnormal (x < 1.6e-309) | `x**(k-1)` overflows while `k*x**(k-1)` is normal (pattern 3) | inf for 1.7e305 | loss |
| F7 | logpdf | k >> s, x >> 1 (e.g. k = 1e6, s = 1) | `(k-1)log x` and `(1+k/s)log1p(x**s)` cancel to -(s+1)log x (pattern 5) | 3.9e5 ulp | moderate |
| F8 | ppf | s < 1 and \|log q\| large (e.g. s = 1e-3) | rounding of `qsk/(1-qsk)` raised to the power `1.0/s` (pattern 6) | 1.1e3 ulp | moderate |
| F9 | cdf, pdf | k/s >~ 100 or \|log(1+x**s)\| large; x in the bulk | 1/2-ulp rounding of `1+x**s` and of the inexact exponent `k/s` amplified by the power (patterns 6, 11) | 673 ulp; cdf = 1.0000000000000262 | minor |
| F10 | logpdf | x = 0 with k = 1 exactly (established by code reading plus a direct call, **not** by the harness: x = 0 is outside the grid and the true value 0.0 is not a normal double, so it falls outside the tool's literal loss definition and has no ulp figure) | `log(0)*(k-1)` = `-inf*0` = nan; true value `log(k)` = 0.0 (pattern 7) | nan vs 0.0 (no ulp figure: the reference is exactly 0.0; severity assigned by inspection) | loss |

### F1: cdf returns nan or 0 in the right tail (overflow of `(1+x**s)**(k/s)`)

- **Code:** `_cdf`, lines 7240-7241: `return x**k / (1.0+x**s)**(k*1.0/s)`
- **Mechanism:** pattern 3. The identity used is
  `F = x**k / (1+x**s)**(k/s)`, i.e. the quotient of two quantities that are both
  approximately `x**k`, while the answer is at most 1. Three intermediates can
  leave the double range before the result does:
  `(1+x**s)**(k/s)` (overflow for `(k/s)·log1p(x**s) > 709.8` — always the first
  one for x > 1, because `log1p(x**s) > s·log x`), `x**k` (overflow for
  `k·log x > 709.8`) and `x**s` (overflow for `s·log x > 709.8`, which makes
  `1+x**s = inf` and the denominator `inf`). If only the denominator is `inf` the
  result is `0.0`; if the numerator overflows as well it is `inf/inf = nan`.
  The mathematically equivalent `(1 + x**(-s))**(-k/s)` never leaves the range.
  This finding has **no counterpart in burr or fisk**: `burr_gen._cdf` (line 1218)
  is already written in that bounded form, `(1 + x**(-c))**(-d)`, which stays in
  range in both tails and is the one method `reports/fisk.md` found flawless
  (max 1.0 ulp over the whole grid). The cdf collapse is therefore specific to
  `mielke_gen._cdf`, not an instance of an inherited family pattern.
- **Regime:** `x >= min((exp(709.8·s/k) - 1)**(1/s), exp(709.8/s))`; measured onsets
  match: 4.37e29 (10.4, 4.6, -> nan), ~1.6e102 (2, 3, -> 0.0), 10.65 (300, 4.6, -> nan),
  1.034 (1000, 1, -> 0.0), 2.3e14 (1, 1e-3, -> 0.0), 1.00071 (1e-3, 1e6, -> 0.0).
  For k/s >= 1e6 the whole range in which the true cdf is a normal double is lost
  (nan from x ~ 1.4e3 upward at k = 1e6, s = 1). 354 of the high-tail, 107 of the
  low-tail (small cdf at large x) and 13 bulk points are lost, 474 in total.
- **Reproducer:** `python accuracy.py refs/mielke_ref.py --at cdf 1e10 1e6 1.0`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `cdf(1e10, 1e6, 1.0)` | nan | 0.9999000049998383 | inf | 1.0e-4 | loss |
  | `cdf(4.371176973533702e+29, 10.4, 4.6)` | nan | 1.0 | inf | 4.68e-136 | loss |
  | `cdf(11.0, 300.0, 4.6)` | nan | 0.9989438536252409 | inf | 0.00486 | loss |
  | `cdf(2.0, 1000.0, 1.0)` | 0.0 | 8.104774656527566e-177 | inf | 333 | loss |
  | `cdf(1.036583355609761, 1000.0, 1.0)` | 0.0 | 5.035481661066546e-294 | inf | 491 | loss |
  | `cdf(1.001, 1e-3, 1e6)` | 0.0 | 1.0 | inf | 5.5e-312 | loss |

- **Fix direction:** evaluate the burr III form `(1 + x**(-s))**(-k/s)` for x >= 1
  (equivalently `exp(-(k/s)*log1p(x**(-s)))`) and keep the present form only for
  x < 1, where it is accurate.

### F2: pdf returns 0 or nan in the right tail (overflow of `(1+x**s)**(1+k/s)`)

- **Code:** `_pdf`, lines 7232-7233: `return k*x**(k-1.0) / (1.0+x**s)**(1.0+k*1.0/s)`
- **Mechanism:** pattern 3, as F1 but with the exponent `1+k/s`, so the denominator
  overflows slightly earlier (at `(1+k/s)·log1p(x**s) > 709.8`). The numerator
  `k*x**(k-1)` is still finite over a wide window, so the quotient is `0.0`; for
  larger k the numerator overflows too and the result is `nan`. The true pdf,
  `k x**(-s-1) (1 + x**(-s))**(-1-k/s)`, is a normal double over tens of decades
  beyond the break point.
- **Regime:** `x >= min((exp(709.8·s/(s+k)) - 1)**(1/s), exp(709.8/s))`; measured
  onsets 3.55e20 (10.4, 4.6), 1.3e61 (2, 3), 10.28 (300, 4.6; nan from x = 11),
  1.032 (1000, 1), and the entire normal-pdf range for (1e6, 1). 121 high-tail,
  63 low-tail and 23 bulk points lost, 207 in total.
- **Reproducer:** `python accuracy.py refs/mielke_ref.py --at pdf 1e10 1e6 1.0`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `pdf(1e10, 1e6, 1.0)` | nan | 9.999000048998483e-15 | inf | 2.0 | loss |
  | `pdf(3.554797644010525e+20, 10.4, 4.6)` | 0.0 | 8.559925616567759e-115 | inf | 5.6 | loss |
  | `pdf(11.0, 300.0, 4.6)` | nan | 4.414243235026728e-4 | inf | 5.6 | loss |
  | `pdf(2.0, 1000.0, 1.0)` | 0.0 | 1.3507957760879278e-174 | inf | 332 | loss |
  | `pdf(1.036583355609761, 1000.0, 1.0)` | 0.0 | 2.3852538056799023e-291 | inf | 490 | loss |

- **Fix direction:** for x >= 1 use
  `k*x**(-s-1) * (1 + x**(-s))**(-1-k/s)`, or `exp(logpdf)` with a repaired
  `logpdf` (F3/F7).

### F3: logpdf returns -inf in the right tail (`x**s` overflows inside log1p)

- **Code:** `_logpdf`, lines 7235-7238:
  `np.log(k) + np.log(x)*(k - 1) - np.log1p(x**s)*(1 + k/s)`
- **Mechanism:** pattern 3. `x**s` is formed explicitly; for `s·log x > 709.8` it
  is `inf`, `log1p(inf) = inf`, and the whole logpdf becomes `-inf` although it is
  an ordinary number of size a few hundred. Nothing in the result is out of range —
  the overflow is purely an artefact, exactly as in fisk F3 (`reports/fisk.md`),
  with the sign of the exponent flipped.
- **Regime:** `x > exp(709.8/s)`, independent of k: 1.03e67 for s = 4.6, 1.6e102
  for s = 3, 1.3e154 for s = 2, 1.00071 for s = 1e6; unreachable for s <= 1 inside
  the double range (the sets with s = 1, 1e-3 show no loss). 222 high-tail points lost.
- **Reproducer:** `python accuracy.py refs/mielke_ref.py --at logpdf 1e70 10.4 4.6`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `logpdf(1e70, 10.4, 4.6)` | -inf | -900.2715506475186 | inf | 0.00622 | loss |
  | `logpdf(1.001, 1e-3, 1e6)` | -inf | -1006.4090878627384 | inf | 994 | loss |
  | `logpdf(1.35242013341704e+154, 0.5, 2.0)` | -inf | -1065.39314718056 | inf | 0.00282 | loss |

- **Fix direction:** for x >= 1 use `log1p(x**s) = s*log(x) + log1p(x**(-s))`, i.e.
  `log(k) - (s+1)*log(x) - (1+k/s)*log1p(x**(-s))`, which also removes the
  cancellation of F7.

### F4: ppf cancels in `1.0 - q**(s/k)` and returns inf near q = 1

- **Code:** `_ppf`, lines 7243-7245:
  `qsk = pow(q, s*1.0/k)` ; `return pow(qsk/(1.0-qsk), 1.0/s)`
- **Mechanism:** patterns 1 and 8. For `(s/k)·|log q| << 1` the intermediate `qsk`
  is `1 - (s/k)|log q|` and `1.0 - qsk` keeps only the digits beyond the leading 1:
  the relative error of the difference is about `eps/((s/k)|log q|)`, and the final
  `**(1/s)` divides it by s, so the quantile has relative error
  `~ eps·k/(s²(1-q))`. Once `(s/k)(1-q) < eps/2`, `1.0 - qsk = 0` and the result is
  `inf`; in between, `1.0-qsk` is quantized in multiples of eps/2, so the returned
  quantile is quantized too (`ppf(1-1e-10, 1e6, 1)` returns exactly 2**53 - 1).
  The intrinsic conditioning of the quantile is only `~ 1/(s(1-q))`, so the
  implementation loses an extra factor `~ k/s`; the errors below are all beyond
  the tool's conditioning allowance.
- **Regime:** `1 - q <~ (k/s)·eps` gives `inf` (1-q < 2.5e-16 for (10.4, 4.6),
  < 7e-15 for (300, 4.6), < 1e-10 for (1e6, 1)); severe and moderate errors extend
  down to `1 - q ~ k·eps/s²`, which for k/s = 1e6 reaches the middle of the
  distribution (9.1e5 ulp at q = 0.85). 24 loss + 39 severe + 4 moderate high-tail
  points and 4 severe + 23 moderate bulk points.
- **Reproducer:** `python accuracy.py refs/mielke_ref.py --at ppf 0.85 1e6 1.0`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `ppf(0.9999999999999999, 10.4, 4.6)` | inf | 3510.7951940437956 | inf | 2.0e15 | loss |
  | `ppf(0.9999999999522942, 1e6, 1.0)` | inf | 2.096179451176214e+16 | inf | 2.1e10 | loss |
  | `ppf(0.9999999999, 1e6, 1.0)` | 9007199254740991.0 | 9999999172096358.0 | 4.5e14 | 1.0e10 | severe |
  | `ppf(0.9999999999, 1000.0, 1.0)` | 9996891514694.885 | 9999999172095.86 | 1.4e12 | 1.0e10 | severe |
  | `ppf(0.85, 1e6, 1.0)` | 6153128.879372721 | 6153128.880622048 | 9.1e5 | 6.15 | moderate |
  | `ppf(0.99999280314327, 300.0, 4.6)` | 32.54160045704271 | 32.54160045452515 | 3.5e5 | 3.0e4 | moderate |
  | `ppf(0.9991516571017559, 1000.0, 1.0)` | 1178268.0641121154 | 1178268.0640683856 | 1.7e5 | 1.2e3 | moderate |

  (cond (shapes) is 8.6-40 at these points, so the loss is not shape sensitivity.)
- **Fix direction:** form the small quantity directly, `1 - q**(s/k) =
  -expm1((s/k)*log(q))`, and finish in log space: with `t = (s/k)*log(q)`,
  `x = exp((t - log(-expm1(t)))/s)`. A custom `_isf` should use
  `t = (s/k)*log1p(-q)`.

### F5: ppf returns 0 (or loses all digits) when `q**(s/k)` underflows

- **Code:** `_ppf`, line 7244: `qsk = pow(q, s*1.0/k)`
- **Mechanism:** patterns 4 and 8. `q**(s/k)` is computed before the `1/s` root is
  taken, so for `s/k > 1` it can be subnormal or zero while the quantile
  `(qsk/(1-qsk))**(1/s)` is a perfectly normal double: the root undoes the
  underflow. A subnormal `qsk` keeps only the leading bits (1.5e15 ulp measured
  just above the cliff), and `qsk = 0` makes the result exactly `0.0`.
- **Regime:** `(s/k)·|log q| > 708` (subnormal) resp. `> 745` (zero): q < 1e-205
  resp. 1e-216 for (2, 3); q < 1e-77 resp. 1e-81 for (0.5, 2); every q <= 0.5 for
  (1e-3, 1e6), where s/k = 1e9. 20 low-tail + 12 bulk loss points, 6 + severe
  points in the band just above the cliff. Also visible at the other end: for
  (1e-3, 1e6) `1-q**(s/k)` is fine but `qsk` itself is subnormal, giving 2.5e9 ulp
  at q = 1 - 7.4e-7.
- **Reproducer:** `python accuracy.py refs/mielke_ref.py --at ppf 1e-300 2.0 3.0`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-300, 2.0, 3.0)` | 0.0 | 1e-150 | inf | 0.5 | loss |
  | `ppf(0.5, 1e-3, 1e6)` | 0.0 | 9.332636185032323e-302 | inf | 1.0e3 | loss |
  | `ppf(3.162277660168379e-151, 0.5, 2.0)` | 0.0 | 9.999999999999999e-302 | inf | 2.0 | loss |
  | `ppf(1.2961424830573168e-81, 0.5, 2.0)` | 2.2227587494850775e-162 | 1.6799853363859867e-162 | 1.5e15 | 2.0 | severe |
  | `ppf(1.997134533661588e-216, 2.0, 3.0)` | 1.7031839360032837e-108 | 1.4132001038995107e-108 | 9.2e14 | 0.5 | severe |
  | `ppf(0.9999992550002775, 1e-3, 1e6)` | 0.9992558369548414 | 0.9992552774436163 | 2.5e9 | 1.0e3 | severe |

- **Fix direction:** stay in log space: with `t = (s/k)*log(q)`,
  `x = exp((t - log(-expm1(t)))/s)`; no intermediate then leaves the double range.

### F6: pdf returns inf for k < 1 at subnormal x

- **Code:** `_pdf`, line 7233: `k*x**(k-1.0)`
- **Mechanism:** pattern 3. For k < 1 the numerator power `x**(k-1)` overflows once
  `(1-k)·|log x| > 709.8`, but the result `k*x**(k-1)/(1+x**s)**(1+k/s)` is smaller
  by the factor k, so for small k it is still a normal double over about
  `log10(1/k)` decades.
- **Regime:** narrow: `709.8/(1-k) >= 709.8` forces `x < exp(-709.8) = 1.6e-309`, so
  only subnormal arguments are affected (2 points for (1e-3, 1e-3), 1 for
  (1e-3, 1e6); nothing for k >= 1).
- **Reproducer:** `python accuracy.py refs/mielke_ref.py --at pdf 1.31830766936289e-309 0.001 0.001`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `pdf(1.31830766936289e-309, 1e-3, 1e-3)` | inf | 1.67541568698601e+305 | inf | 1.0 | loss |
  | `pdf(2.686618191037897e-309, 1e-3, 1e6)` | inf | 1.8290402172717744e+305 | inf | 0.999 | loss |

- **Fix direction:** `exp(logpdf)`, or `k * x**k / x / (1+x**s)**(1+k/s)`.

### F7: logpdf loses digits to log-space cancellation when k >> s

- **Code:** `_logpdf`, lines 7237-7238:
  `np.log(k) + np.log(x)*(k - 1) - np.log1p(x**s)*(1 + k/s)`
- **Mechanism:** pattern 5. For x >> 1 the two terms are
  `(k-1)log x` and `(1+k/s)log1p(x**s) ~ (s+k)log x + (k/s)x**(-s)`, both of size
  `k·log x`, while the result is only `log k - (s+1)log x - (k/s)x**(-s)`. The
  relative loss is the ratio of the terms to the result, about
  `k·log x / ((s+1)log x) = k/(s+1)` ulp. Combining the terms analytically removes it.
- **Regime:** k/s large and x far from 1: 3.9e5 ulp at k = 1e6, s = 1 (37 moderate
  points, x from 1e10 to 3e307), 346 ulp at k = 1000, s = 1, 199 ulp at (300, 4.6);
  the same mechanism produces the 41 minor bulk points (max 199 ulp).
- **Reproducer:** `python accuracy.py refs/mielke_ref.py --at logpdf 1e290 1e6 1.0`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `logpdf(1e290, 1e6, 1.0)` | -1321.6838434934616 | -1321.6838433785822 | 3.9e5 | 0.00151 | moderate |
  | `logpdf(1.8087545456327052e+16, 1000.0, 1.0)` | -67.96024472101271 | -67.96024472101793 | 3.5e2 | 0.0294 | minor |
  | `logpdf(2.5, 300.0, 4.6)` | -0.39853595735314684 | -0.3985359573531292 | 2.0e2 | 2.92 | minor |

- **Fix direction:** the x >= 1 form of F3,
  `log(k) - (s+1)*log(x) - (1+k/s)*log1p(x**(-s))`, in which no term is larger
  than the result by more than `log1p(x**-s)·k/s` and the subtraction disappears.

### F8: ppf amplifies rounding by the outer power `1.0/s`

- **Code:** `_ppf`, line 7245: `pow(qsk/(1.0-qsk), 1.0/s)`
- **Mechanism:** pattern 6 (plus 11: `1.0/s` is itself a rounded double). The
  argument of the outer power carries at least a 1/2-ulp rounding from `pow(q, s/k)`
  and from the division; raising it to the power `1/s` multiplies that relative
  error by `1/s`, so for s = 1e-3 about `1000·eps/2 = 5.5e-14` (250 ulp) is
  unavoidable in this formulation, and the measured error is 1.1e3 ulp.
- **Regime:** s < 1 with `|log q|` large: (1, 1e-3) gives 2 moderate and 9 minor
  points (q from 1e-300 to 4e-176); the (1e-3, 1e-3) set reaches 500 ulp at
  q = 0.5 (same mechanism, 1/s = 1000), but there cond (arg) = 2.0e3, so the tool
  classifies it as **ok**.
- **Reproducer:** `python accuracy.py refs/mielke_ref.py --at ppf 1.6155980984399093e-213 1.0 0.001`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `ppf(1.6155980984399093e-213, 1.0, 1e-3)` | 1.2539529091979406e+199 | 1.2539529091976361e+199 | 1.1e3 | 2.58 | moderate |
  | `ppf(2.3713737056616554e-263, 1.0, 1e-3)` | 3.4558856566676243e+80 | 3.4558856566682604e+80 | 8.3e2 | 2.2 | minor |

- **Fix direction:** one `exp(.../s)` instead of a `pow(., 1/s)` of a rounded
  quotient: `x = exp((t - log(-expm1(t)))/s)` with `t = (s/k)*log(q)` (as in F4/F5),
  which keeps a single final rounding and no reciprocal of s.

### F9: cdf and pdf amplify the rounding of `1+x**s` and of the exponent `k/s`

- **Code:** `_cdf`, line 7241: `(1.0+x**s)**(k*1.0/s)`; `_pdf`, line 7233:
  `(1.0+x**s)**(1.0+k*1.0/s)`
- **Mechanism:** patterns 6 and 11. Two effects, both proportional to the exponent:
  the 1/2-ulp rounding of `1+x**s` is multiplied by `k/s` (about
  `(k/s)·(x**s/(1+x**s))·eps/2` relative), and the exponent `k*1.0/s` is itself a
  rounded double whose absolute error `ulp(k/s)/2` is multiplied by
  `log(1+x**s)`. For (1, 1e-3) at x = 8 the second effect predicts
  `1000·1.1e-16·0.69/(2·...) ~ 3e-13`, and 671 ulp (1.5e-13) is measured. A
  side effect is that the cdf can exceed 1 (`cdf(1.4e102, 2, 3) =
  1.0000000000000262`), i.e. the returned value is not a probability.
- **Regime:** k/s >~ 100, or any shapes with `|log(1+x**s)|` of order 100 or more:
  35 minor bulk and 74 minor high-tail cdf points, 28 + 30 for pdf; max 673 ulp.
- **Reproducer:** `python accuracy.py refs/mielke_ref.py --at cdf 8.0 1.0 0.001`

  | call | scipy | reference | ulp | cond (arg) | severity |
  |---|---|---|---|---|---|
  | `cdf(8.0, 1.0, 1e-3)` | 2.6382417528167117e-301 | 2.638241752817105e-301 | 6.7e2 | 0.499 | minor |
  | `pdf(8.0, 1.0, 1e-3)` | 1.6471866994101466e-302 | 1.6471866994103927e-302 | 6.7e2 | 0.501 | minor |
  | `pdf(3.133997440868717e+61, 2.0, 3.0)` | 2.0731725297882713e-246 | 2.0731725297881407e-246 | 2.8e2 | 4.0 | minor |
  | `cdf(10.102672572677047, 300.0, 4.6)` | 0.9984382482891926 | 0.9984382482891532 | 1.8e2 | 0.00719 | minor |
  | `cdf(1.399787605293446e+102, 2.0, 3.0)` | 1.0000000000000262 | 1.0 | 1.2e2 | 7.3e-307 | minor |

  (cond (shapes) is 1.4e3 at the first two points: the cdf really is that
  sensitive to the last bit of s, but s is an exact user input, so the error
  remains avoidable.)
- **Fix direction:** `exp(-(k/s)*log1p(x**(-s)))` for x >= 1 and
  `exp(k*log(x) - (k/s)*log1p(x**s))` for x < 1, so the rounding of `1+u` enters
  through `log1p` and the division by s happens once, at the end.

### F10: logpdf(0, k, s) is nan for k = 1

- **Code:** `_logpdf`, line 7238: `np.log(x)*(k - 1)`
- **Mechanism:** pattern 7. At x = 0 with k exactly 1 the product is
  `-inf * 0.0 = nan` (the `errstate(divide='ignore')` only silences the warning).
  The true value is `log(k) = 0.0`, the finite limit of the logpdf at the left
  endpoint. For k < 1 / k > 1 the same expression correctly yields `+inf` / `-inf`,
  and `pdf(0, k, s)` is correct in all three cases (inf, 1.0, 0.0).
- **Regime:** x = 0 and k = 1 exactly. **Severity assigned by inspection, not by
  the tool:** `accuracy.py` cannot score this point (x = 0 is not part of the
  default grid, and the exact value 0.0 is not a normal double, so the harness
  would skip it even if it were). It is listed as `loss` because a nan is returned
  where the correct result is a finite number — the same criterion the tool applies
  to normal-double cases.
- **Reproducer:** `python -c "from scipy.stats import mielke; print(mielke.logpdf(0.0, 1.0, 1.0))"`
  -> `nan`, true value `0.0` (and `mielke.logpdf(0.0, 0.5, 2.0)` -> `inf`,
  `mielke.logpdf(0.0, 2.0, 3.0)` -> `-inf`, both correct).
- **Fix direction:** `sc.xlogy(k - 1, x)` instead of `np.log(x)*(k-1)`.

## RuntimeWarnings

- `pdf`: `overflow encountered in power` (F2/F6 regimes), `invalid value
  encountered in divide` (the `inf/inf = nan` of F2), `overflow/underflow
  encountered in multiply`, `underflow encountered in divide/power` — in 11 of 11
  parameter sets. The underflow warnings in the far tails accompany correct
  (subnormal or zero) results and are spurious; the overflow and invalid-divide
  warnings mark the F2/F6 losses.
- `cdf`: `overflow encountered in power`, `invalid value encountered in divide`
  (F1), plus spurious `underflow encountered in power/divide` far out in the left
  tail where 0 is the right answer.
- `logpdf`: `overflow encountered in power` exactly in the F3 regime;
  `underflow encountered in power/multiply` on correct values.
- `ppf`: `divide by zero encountered in divide` exactly where F4 returns `inf`;
  `underflow encountered in power` where F5 returns 0; `overflow encountered in
  power` for extreme q with s < 1, where the true quantile is outside the double
  range (no accuracy issue there).
- `logpdf` at x = 0 is explicitly guarded by `np.errstate(divide='ignore')`
  (line 7236-7237), which hides the warning but not the nan of F10.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| cdf | x < 1, all 11 sets, 332 points | 279 ulp (minor, the F9 mechanism at (1, 1e-3)); median 0. Note that the tool's "low tail" region is labelled by the cdf *value*, so its 107 loss points are F1 cases at large x, not small-x cases |
| logpdf | x < 1, 700 points | 11.5 ulp; over the tool's whole "low tail" region 27.8 ulp (4 minor points). No left-tail overflow or cancellation (the burr/fisk F3/F4 mechanisms cannot occur: `x**s`, `x**(k-1)` underflow instead of overflowing, and `log1p(x**s) -> 0`) |
| pdf | left tail with normal x, all sets | <= 1.5 ulp for (0.5, 2) and (1, 1) over their whole grids; only the subnormal-x corner fails (F6) |
| pdf, logpdf, cdf | the whole grid for (1, 1) and (1e-3, 1e-3) | 1.5 / 0.99 / 1.0 ulp — no loss at all |
| ppf | q in [1e-16, 0.9] for (10.4, 4.6), (2, 3), (1, 1), (1e-3, 1e-3) | <= 9 ulp for (10.4, 4.6); the F4/F5/F8 mechanisms need extreme s/k or extreme q |
| pdf | pattern 4 (subnormal `x**(k-1)` lifted back by the factor k): x = exp(-720/(k-1)) for k = 300 and k = 1e6 | not a finding — the denominator `(1+x**s)**(1+k/s) >= 1` means the true pdf is subnormal (6.1e-311) or far below (1e-301180) wherever `x**(k-1)` is subnormal, so the regime has no normal-double result |
| cdf, pdf, ppf | (1e6, 1e-3) | no measurable point: with k/s = 1e9 the true cdf is below 2.2e-308 for every double x (it would need x > e**14200), so this set only confirms that nothing is reported; the shape corner k/s -> inf is covered by (1e6, 1) instead |
| pdf(0, k, s) | k = 0.5, 1, 2 | correct limits inf, 1.0, 0.0 |

## Suspected, not confirmed

- `_munp` (line 7247-7251) computes the moments as
  `sc.gamma((k+n)/s)*sc.gamma(1-n/s)/sc.gamma(k/s)`. The individual gammas
  overflow for `k/s >~ 171` while the ratio stays finite (pattern 3), and
  `sc.gamma(1-n/s)` has poles at `n = s`, `2s`, ... — `apply_where(n < s, ...)`
  only guards `n >= s`. Not measured (moments are outside the eight distribution
  functions); a `gammaln`-based form would be the fix.
- F4 for very small s/k (s/k <= 1e-9, e.g. (1e6, 1e-3)): `q**(s/k)` rounds to 1 for
  every q in the grid, so `_ppf` must return `inf` there, but the true quantile is
  also outside the double range for that set, so no measurable point exists.
- Possible `x**k` overflow with a *finite* result in `_cdf` (numerator overflowing
  while the denominator does not): not reachable, because for x > 1
  `(1+x**s)**(k/s) > x**k`, so the denominator always overflows first; for x < 1
  `x**k` cannot overflow.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| sf | `1 - cdf` | `-expm1(-(k/s)*log1p(x**(-s)))`, i.e. `1 - (1+x**-s)**(-k/s)`; for x >> 1 ~ `(k/s)*x**(-s)` | right tail: the cdf rounds to 1 once `(k/s)x**(-s) < eps`, so the sf loses all digits and then returns 0; it is already `nan` wherever F1 applies. `sf(1e10, 10.4, 4.6)` returns 2.220446049250313e-15 for a true 2.26086956521741e-46 (4.4e46 ulp, **loss**) |
| logsf | `log(sf)` | `log1mexp(-(k/s)*log1p(x**(-s)))` (`sc._ufuncs._log1mexp`) | same regime, one step worse: `logsf(1e10, 10.4, 4.6)` returns -33.74106829612311 for a true -105.10316477507382 (3.1e15 ulp, **severe**), and -inf further out |
| logcdf | `log(cdf)` | `-(k/s)*log1p(x**(-s))` for x >= 1, `k*log(x) - (k/s)*log1p(x**s)` for x < 1 | wherever F1 breaks the cdf, and near cdf = 1 where `log(1.0) = 0` instead of a small negative number: `logcdf(1e10, 1e6, 1.0)` returns nan for a true -9.9999999995e-05 (**loss**) |
| isf | `ppf(1 - q)` | `t = (s/k)*log1p(-q)`; `exp((t - log(-expm1(t)))/s)` | small q: `1 - q` rounds to 1, so `_ppf` divides by `1.0 - 1.0 = 0`. `isf(1e-300, 10.4, 4.6)` returns inf for a true 1.9697346241853196e+65 (**loss**); it also inherits F4/F5/F8 |

## Shared code

- Nothing in the measured methods comes from a helper or from `scipy.special`:
  `mielke_gen` uses only `np.log`, `np.log1p`, `pow` and `**`, and all of them
  return correct values for the arguments they are given — the arguments, and the
  absence of complementary/log-space forms, are the problem. `sc.gamma` appears
  only in `_munp` (see "Suspected, not confirmed").
- mielke is algebraically burr with `c = s`, `d = k/s`, but it has an independent
  implementation, so none of the burr/fisk code is executed here and no fix carries
  over automatically. Some *mechanisms* do recur, with the tails exchanged because
  mielke writes positive powers of x where burr writes negative ones:
  - **F2 <-> fisk F2** and **F3 <-> fisk F3**: the right-tail mirror of
    "`x**(-c-1)` / `x**(-c)` overflows although the true result is a normal
    double" (`reports/fisk.md`), here `(1+x**s)**(1+k/s)` resp. `x**s`.
  - **F9 <-> fisk F5** and **F8 <-> fisk F6**: the mielke instances of a rounded
    exponent (`k*1.0/s`, `1.0/s`) amplified by a large logarithm.
  - **F7: same pattern as fisk F4, different regime.** Both are the pattern-5
    cancellation of the two large log terms in `_logpdf`, but the conditions
    differ: fisk F4 needs `c*d ~ 1` (c close to 1), whereas F7 needs only
    `k >> s` and x far from 1, with no special relation between the shapes.
  - **F1 has no fisk or burr counterpart** — `burr_gen._cdf` uses the bounded
    `(1 + x**(-c))**(-d)`, so the cdf failure is mielke's own (see F1).
  - F6 is the left-tail analogue of the same pattern-3 numerator overflow as
    fisk F2, but far narrower: mielke's numerator exponent is `k-1` with k < 1
    (so `|k-1| < 1` and only subnormal x can trigger it), whereas fisk's is
    `-c-1` with no such bound.
  - F4, F5 and F10 were not reported for fisk at all (fisk's `_ppf` is accurate
    because d = 1 makes `q**(-1/d)` a single rounding).

  The inherited-`sf` collapse quoted under
  "Opportunities" is the same defect as fisk F1, except that in fisk it lives in
  an explicit `_sf`/`_logsf` while mielke has none at all. A single fix applied to
  all four Burr-family classes (burr, burr12, fisk, mielke) would be the economical
  route; SciPy issue #17832 already tracks the missing `_sf`/`_logsf`/`_logcdf`
  overrides for this family.

## Existing GitHub issues
- #15195 (closed): "BUGS: stats: Tracking issue for distributions that warn
  and/or return nan at support endpoints" - relates to F10; `mielke` is named
  in its list of distributions that warn or return nan at an endpoint, though
  the issue does not say which method. F10 (`logpdf(0, 1, s)` = nan where the
  true value is 0.0) is exactly that defect, and it is still present.
- #17832 (open): Tracking issue: custom methods for continuous distributions -
  relates to the inherited `sf`, `logsf`, `logcdf` and `isf`, all four of which
  collapse at their demonstration points; the issue names the `sf = 1 - _cdf`
  construction as a "risk of catastrophic cancellations in the right tail" and
  tracks which distributions still need their own implementations.
- #11465 (closed, implemented in PR #21050): "Numerical stability of
  distributions: Add log_p as argument to all distributions?" - same scope as
  above for the inherited `logcdf`/`logsf`, which take `log` of an already
  rounded value.
- F1-F5 and F7-F9: none found. The right-tail collapse of `cdf`, `pdf`,
  `logpdf` and `ppf` measured here does not appear to be reported for mielke.
  Note that F1 in particular is specific to `mielke_gen._cdf` and is not
  covered by any burr/fisk issue, since `burr_gen._cdf` already uses the
  bounded form.
- Context for the suspected `_munp` overflow (not a scored finding): PR #10154
  ("MAINT: add moments and improve doc of mielke in scipy.stats") is the change
  that introduced the current moment code whose gamma ratio overflows for
  k/s >~ 171.

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `cdf(1e10, 1e6, 1.0)` | nan | 0.99990000499983834 | verified (loss) | intermediates confirmed: `x**k = inf`, `(1+x**s)**(k/s) = inf` -> `inf/inf` |
| F1 | `cdf(2.0, 1000.0, 1.0)` | 0.0 | 8.1047746565275667e-177 | verified (loss) | `x**k = 1.07e301` finite, denominator `3**1000 = inf` -> `finite/inf` |
| F1 | `cdf(11.0, 300.0, 4.6)`; `cdf(1.001, 1e-3, 1e6)` | nan; 0.0 | 0.99894385362524082; 1.0 | verified (loss) | both reproduced; mechanism as stated |
| F2 | `pdf(1e10, 1e6, 1.0)` | nan | 9.9990000489984834e-15 | verified (loss) | numerator `k*x**(k-1) = inf` and denominator `inf` |
| F2 | `pdf(3.554797644010525e+20, 10.4, 4.6)` | 0.0 | 8.5599256165677582e-115 | verified (loss) | numerator 1.57e194 finite, denominator inf -> 0.0, exactly as described |
| F3 | `logpdf(1e70, 10.4, 4.6)` | -inf | -900.27155064751852 | verified (loss) | `x**s = inf` -> `log1p(inf) = inf`; onset `exp(709.8/s) = 1.03e67` confirmed |
| F3 | `logpdf(1.35242013341704e+154, 0.5, 2.0)`; `logpdf(1.001, 1e-3, 1e6)` | -inf; -inf | -1065.3931471805599; -1006.4090878627384 | verified (loss) | k-independent threshold confirmed |
| F4 | `ppf(0.9999999999999999, 10.4, 4.6)` | inf | 3510.7951940437958 | verified (loss) | `qsk` rounds to 1.0, `1.0-qsk = 0.0` -> division by zero |
| F4 | `ppf(0.9999999999, 1e6, 1.0)` | 9007199254740991.0 | 9999999172096357.97 | verified (severe, 4.5e14 ulp) | `1-qsk = 1.1102230246251565e-16` = eps/2 exactly; cond (arg) 1.0e10 << error |
| F4 | `ppf(0.85, 1e6, 1.0)` | 6153128.879372721 | 6153128.880622048 | verified (moderate, 9.1e5 ulp) | cond (arg) 6.15; mid-distribution loss confirmed |
| F5 | `ppf(1e-300, 2.0, 3.0)` | 0.0 | 1.0000000000000000125e-150 | verified (loss) | `q**(s/k) = q**1.5` underflows to 0 before the `1/s` root |
| F5 | `ppf(0.5, 1e-3, 1e6)`; `ppf(3.162277660168379e-151, 0.5, 2.0)` | 0.0; 0.0 | 9.3326361850323235e-302; 9.9999999999999985e-302 | verified (loss) | both normal doubles; `s/k` = 1e9 resp. 4 |
| F5 | `ppf(1.2961424830573168e-81, 0.5, 2.0)` | 2.2227587494850775e-162 | 1.6799853363859867e-162 | verified (severe, 1.5e15 ulp) | `qsk = 5e-324` (one bit of mantissa left), as described |
| F6 | `pdf(1.31830766936289e-309, 1e-3, 1e-3)` | inf | 1.67541568698601e+305 | verified (loss) | argument round-trips as a subnormal double; `x**(k-1) = inf`, denominator 2.223 |
| F6 | `pdf(2.686618191037897e-309, 1e-3, 1e6)` | inf | 1.8290402172717744e+305 | verified (loss) | same mechanism, denominator 1.0 |
| F10 | `logpdf(0.0, 1.0, 1.0)` | nan | 0.0 (`= log k`) | verified (nan for a finite value) | `np.log(0.0)*(k-1) = -inf*0 = nan` confirmed; also nan for (k, s) = (1, 2), so the regime is k = 1, any s. Severity "loss" is justified in spirit (no digits of a finite value), but note the harness definition of `loss` quotes a *normal double* true value and the true value here is exactly 0.0, so no ulp figure exists |
| F10 | `logpdf(0.0, 0.5, 2.0)`; `logpdf(0.0, 2.0, 3.0)` | inf; -inf | inf; -inf | verified correct | true limits `log k + (k-1)log x` -> +-inf for k < 1 / k > 1 |
| F7-F9 (context) | `logpdf(1e290, 1e6, 1.0)`; `ppf(1.6155980984399093e-213, 1.0, 1e-3)`; `cdf(8.0, 1.0, 1e-3)` | -1321.6838434934616; 1.2539529091979406e+199; 2.6382417528167117e-301 | -1321.6838433785822; 1.2539529091976361e+199; 2.6382417528171052e-301 | spot-checked, 3.9e5 / 1.1e3 / 672 ulp | out of scope, but the quoted numbers reproduce |

Method: an independent reference was built from the docstring pdf
`f(x,k,s) = k x**(k-1)/(1+x**s)**(1+k/s)` (`scratch/mielke/verify/ref_check.py`),
with the cdf taken in the complementary-form antiderivative `(1+x**(-s))**(-k/s)`
(verified analytically as the antiderivative of the pdf, numerically against
`mp.diff` of the cdf to >=50 digits, and against `mp.quad` of the pdf) and the ppf
by closed-form inversion `x = (q**(-s/k) - 1)**(-1/s)`, re-checked by
`cdf(x_ref) == q` to 1e-56 at every ppf point. Every value was evaluated at 100
and 200 digits and agreed to >=90 digits. All reference values agree with the
report's (and with `refs/mielke_ref.py` via `--at`) to every printed digit; no
reference disagreement. Cited code lines (7232-7233 `_pdf`, 7235-7238 `_logpdf`,
7240-7241 `_cdf`, 7243-7245 `_ppf`) match the installed SciPy 1.18.1 source and
each stated mechanism was confirmed by evaluating the intermediates in float64.

Two caveats on the fisk correspondence claimed under "Shared code": (i) fisk/burr
has no cdf counterpart to F1 - `burr_gen._cdf` is already the bounded form
`(1 + x**(-c))**(-d)` and is accurate in both tails (fisk cdf max 1.0 ulp), so
only F2 (<-> fisk F2) and F3 (<-> fisk F3, which there affects logpdf *and*
logcdf) are true mirrors; mielke's cdf defect is extra, not mirrored. (ii) F7 is
the same pattern-5 "wrong tail of the chosen form" cancellation as fisk F4, but
not a literal mirror: fisk F4 needs the fine-tuned shape relation `c*d ~ 1`,
whereas F7 occurs for every k >> s with size ~ k/(s+1). The F8 <-> fisk F6 and
F9 <-> fisk F5 correspondences (rounded exponent amplified by a large logarithm)
are accurate.

Verified by dist-accuracy-verifier: 7 of 7 verified.

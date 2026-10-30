# halfnorm: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `halfnorm_gen`, lines 4736-4816
- **Implemented:** pdf, logpdf, cdf, sf, ppf, isf
- **Inherited (not measured):** logcdf = log(cdf) / log1p(-sf) split at the median (rv_continuous default), logsf = log(sf) / log1p(-cdf) split at the median
- **Related:** foldnorm (c = 0 is halfnorm, separate code); chi with df = 1 (separate code); helpers: `_norm_ppf` (line 378, `sc.ndtri`), `_norm_sf` (line 382, `ndtr(-x)`), `_norm_isf` (line 390, `-ndtri(q)`); special functions: sc.erf, sc.ndtr, sc.ndtri
- **Reference:** `refs/halfnorm_ref.py`, 400 digits; self-checks: PASS
- **Parameter sets:** () (no shapes)
- **Worst finding:** loss in ppf and logpdf

## Summary

pdf, cdf, sf and isf are accurate to about 1 ulp everywhere (sf up to 512 ulp in
the far right tail, within conditioning). ppf is badly wrong for small q: it
evaluates `ndtri((1 + q)/2)`, and forming `1 + q` throws away the digits of q,
so the relative error is about eps/q (moderate from q ~ 1e-6, severe from
q ~ 1e-8) and the result is 0.0 for q < ~1.1e-16, for all q down to 1e-300.
ppf also returns inf at q = 1 - 2**-53 because `1 + q` rounds to 2. logpdf
returns -inf for x in about (1.34e154, 1.9e154), where `x*x` overflows
although the true value is still a finite double.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | ppf | q < ~1e-4 (loss for q < ~1.1e-16) | `(1+q)/2` rounds q away (pattern 8 / 1) | 0.0 for 1.2533141373155002e-300 | loss |
| F2 | ppf | q = 1 - 2**-53 (largest double below 1) | `1 + q` rounds to 2, ndtri(1) = inf (pattern 8) | inf for 8.292361075813595 | loss |
| F3 | logpdf | x in (~1.34e154, ~1.9e154) | `x*x` overflows (pattern 3) | -inf for -1.1250000000000002e+308 | loss |

### F1: ppf loses q in (1 + q)/2 for small q

- **Code:** `_ppf`, line 4775: `_norm_ppf((1+q)/2.0)` with `_norm_ppf = sc.ndtri` (line 379).
- **Mechanism:** pattern 8 (and 1): `1 + q` keeps only the bits of q above
  2**-52 relative to 1, so the argument of ndtri is 0.5 + q/2 with an absolute
  error up to eps/2. ppf(q) ~ sqrt(pi/2) q for small q, so the relative error
  is about eps/q; for q < ~1.1e-16, `1 + q == 1`, ndtri(0.5) = 0 and the result
  is exactly 0.0. The quantile function is well conditioned here (cond (arg) = 1).
- **Regime:** all q < ~1e-4: minor at q = 1e-4 (496 ulp), moderate at
  q = 1e-6, severe for 1e-15 <= q <= 1e-8, loss (0.0) for every q <= 1e-16
  down to 1e-300.
- **Reproducer:** `python accuracy.py refs/halfnorm_ref.py --at ppf 1e-20`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(1e-300)` | 0.0 | 1.2533141373155002e-300 | inf | 1 | loss |
  | `ppf(1e-20)` | 0.0 | 1.2533141373155002e-20 | inf | 1 | loss |
  | `ppf(1e-16)` | 0.0 | 1.2533141373155002e-16 | inf | 1 | loss |
  | `ppf(1e-15)` | 1.3914582123358837e-15 | 1.2533141373155003e-15 | 5.0e+14 | 1 | severe |
  | `ppf(1e-12)` | 1.2534255576721639e-12 | 1.2533141373155002e-12 | 4.0e+11 | 1 | severe |
  | `ppf(1e-08)` | 1.25331412969852e-08 | 1.2533141373155004e-08 | 2.7e+07 | 1 | severe |
  | `ppf(1e-06)` | 1.2533141372127224e-06 | 1.2533141373158284e-06 | 3.7e+05 | 1 | moderate |
  | `ppf(0.0001)` | 0.0001253314140596531 | 0.0001253314140596669 | 496 | 1 | minor |

  (36 ppf points flagged in total, 27 of them loss; all with cond (arg) = 1.)
- **Fix direction:** `sqrt(2) * sc.erfinv(q)` (accurate for small q; e.g.
  gives 1.2533141373155004e-20 at q = 1e-20), or `-ndtri((1 - q)/2)` for q
  below 0.5 and the current form above.

### F2: ppf returns inf at q = 1 - 2**-53

- **Code:** `_ppf`, line 4775: `_norm_ppf((1+q)/2.0)`.
- **Mechanism:** pattern 8: in [1, 2) the double spacing is 2**-52, so
  `1 + (1 - 2**-53)` rounds (to even) to 2.0 and `ndtri(1.0) = inf`. For other
  q close to 1, `1 + q` rounds 1 - q by up to half its own ulp; the resulting
  error stays inside the (huge) conditioning there, so only this point is flagged.
- **Regime:** q = 0.9999999999999999 (1 - 2**-53) only; q = 1 - 2**-52 gives
  8.209536151601387, 0 ulp.
- **Reproducer:** `python accuracy.py refs/halfnorm_ref.py --at ppf 0.9999999999999999`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `ppf(0.9999999999999999)` | inf | 8.292361075813595 | inf | 1.3e+14 | loss |

- **Fix direction:** for q > 0.5 use `-ndtri((1 - q)/2)` (`1 - q` is exact for
  q in [0.5, 1]), i.e. `_norm_isf((1 - q)/2)`.

### F3: logpdf overflows in x*x before the result does

- **Code:** `_logpdf`, line 4769: `0.5 * np.log(2.0/np.pi) - x*x/2.0`.
- **Mechanism:** pattern 3: `x*x` overflows to inf for x > ~1.34e154, while
  `x*x/2` stays below 1.8e308 until x ~ 1.9e154; the result is -inf where the
  true logpdf is a finite double.
- **Regime:** 1.34e154 < x < ~1.896e154 (narrow, extreme argument range).
- **Reproducer:** `python accuracy.py refs/halfnorm_ref.py --at logpdf 1.5e154`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `logpdf(1.35e+154)` | -inf | -9.1125e+307 | inf | 2 | loss |
  | `logpdf(1.5e+154)` | -inf | -1.1250000000000002e+308 | inf | 2 | loss |
  | `logpdf(1.8e+154)` | -inf | -1.62e+308 | inf | 2 | loss |
  | `logpdf(1.89e+154)` | -inf | -1.78605e+308 | inf | 2 | loss |

- **Fix direction:** `(x/2)*x` or `0.5*x*x` evaluated as `x*(0.5*x)`.

## RuntimeWarnings

- pdf: "overflow encountered in multiply" (`x*x` for x > 1.34e154) and
  "underflow encountered in exp / multiply" in the right tail. The returned
  0.0 is correct there (true value below the normal range); spurious.
- logpdf: "overflow encountered in multiply" accompanies the wrong -inf of F3;
  "underflow encountered in multiply" for tiny x (x*x underflows, result
  correct).
- cdf: "underflow encountered in divide" for x ~ 1e-308 (`x/np.sqrt(2)` is
  subnormal); result accurate (max 1.02 ulp in the low tail); spurious.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| pdf | full grid 1e-300 ... 1e300, x near 37-38.6 and near 1.3e154 | 1.16 ulp |
| logpdf | full grid except F3 | 0.713 ulp |
| cdf | full grid incl. x = 2.5e-308 (subnormal x/sqrt(2)) and large x | 1.02 ulp |
| sf | full grid incl. far right tail up to underflow (x ~ 38.5) | 512 ulp, within conditioning (cond 1.37e3); bulk 4.34 ulp |
| isf | q in [1e-300, 1 - 3e-16] incl. q near 1 | 1.63 ulp |
| ppf | bulk (q in ~[1e-2, 1 - 1e-6]) and high tail except F2 | 6.91 ulp bulk |

## Suspected, not confirmed

none. (`--plot` of `accuracy.py` terminated the process silently with exit
code 127 on this machine, also on a rerun; the measurements were done without
the plot. This does not affect the numbers.)

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logcdf | `log(cdf)` below the median, `log1p(-sf)` above (rv_continuous) | `log(erf(x/sqrt(2)))` / `log1p(-erfc(x/sqrt(2)))` is what the default already does | none found: `logcdf(10)` = -1.523970604832094e-23, true -1.523970604832105e-23 (~33 ulp, from erf/ndtr rounding); no custom method needed |
| logsf | `log(sf)` above the median, `log1p(-cdf)` below | `log(2) + sc.log_ndtr(-x)` | sf underflows for x > ~38.6: `logsf(40)` = -inf, true -803.9152948331938 (`log(2) + log_ndtr(-40)` = -803.915294833194) |

## Shared code

none. `sc.ndtri`, `sc.ndtr` and `sc.erf` are accurate for their own results;
F1/F2 come from the argument transformation `(1 + q)/2` in `halfnorm._ppf`,
not from the `_norm_ppf` helper itself.

## Existing GitHub issues

none found (web search; `gh` unavailable). PR #18760 (ENH: override halfnorm fit) touches halfnorm but not these methods.

## Verification

| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `ppf(1e-20)` | 0.0 | 1.2533141373155002e-20 | verified | ref `sqrt(2)*erfinv(q)`, checked by `mp.quad` of pdf on [0, x]; also 1e-300 -> 0.0 (ref 1.2533141373155002e-300), 1e-15 5e14 ulp, 1e-8 2.7e7 ulp, 1e-6 3.7e5 ulp, 1e-4 496 ulp, all as reported; mechanism matches line 4775 |
| F2 | `ppf(0.9999999999999999)` | inf | 8.292361075813595 | verified | quad of pdf on [x, inf) equals 1 - q; `1 + (1 - 2**-53)` is a tie, rounds to even 2.0; 1 - 2**-52 gives 8.209536151601387, 0 ulp, as reported |
| F3 | `logpdf(1.5e154)` | -inf | -1.1250000000000002e+308 | verified | direct `log(2/pi)/2 - x*x/2` at 100/200 digits; 1.35e154, 1.8e154, 1.89e154 also -inf vs finite refs as reported; `x*x` at line 4769 confirmed |

Verified by dist-accuracy-verifier: 3 of 3 verified.

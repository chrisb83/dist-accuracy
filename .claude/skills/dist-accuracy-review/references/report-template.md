# Report template: `reports/<dist>.md`

Keep the section headings exactly as below (the orchestrator checks them).
Numbers in tables come from `accuracy.py` output; do not retype them by hand.

```markdown
# <dist>: numerical accuracy review

- **SciPy version:** 1.18.1
- **Source:** `scipy/stats/_continuous_distns.py`, class `<dist>_gen`, lines <a>-<b>
- **Implemented:** pdf, logpdf, cdf, sf, ppf, isf  (note any defined in another class, e.g. "via burr")
- **Inherited (not measured):** logcdf = log(cdf), logsf = log(sf)
- **Related:** <distributions sharing code>; helpers: <helpers>; special functions: <sc.*>
- **Reference:** `refs/<dist>_ref.py`, <dps> digits; self-checks: PASS
- **Parameter sets:** (…), (…)
- **Worst finding:** <severity> in <method>, or "none beyond conditioning"

## Summary

Two to five sentences: what is accurate, what fails, where, how badly.

## Findings

| ID | Method | Regime (shapes, argument) | Mechanism | Worst error | Severity |
|---|---|---|---|---|---|
| F1 | sf | x**-c < eps, any c, d | 1 - cdf cancels (pattern 1) | 0.0 for 3e-16 | loss |

### F1: <one-line title>

- **Code:** `_sf`, line <n>: `<the expression>`
- **Mechanism:** what goes wrong and why (refer to patterns.md by number).
- **Regime:** where it happens, as conditions on shapes and argument.
- **Reproducer:** `python accuracy.py refs/<dist>_ref.py --at sf 1e8 2 3`

  | call | scipy | reference | ulp | cond | severity |
  |---|---|---|---|---|---|
  | `sf(1e8, 2, 3)` | 0.0 | 3.0e-16 | inf | 1.0 | loss |

- **Fix direction:** one line, e.g. `-expm1(logcdf)`.

(one subsection per finding, worst first)

## RuntimeWarnings

Warnings emitted, for which methods/regimes, and whether the returned value
is nevertheless accurate (spurious warning) or not.

## Checked and fine

| Method | Regimes checked | Max error |
|---|---|---|
| cdf | full grid, all parameter sets | 3.1 ulp |

## Suspected, not confirmed

Code-reading hypotheses that could not be demonstrated, with what was tried.
Write "none" if empty.

## Opportunities: inherited methods

| Method | Default | Better formula | Regime where the default fails |
|---|---|---|---|
| logsf | log(sf) | `sc.gammaincc` in log space via … | sf underflows for x > ~… |

## Shared code

Problems located in helpers, base classes or `scipy.special` functions that
other distributions also use. Write "none" if empty.

## Existing GitHub issues

Filled in by the orchestrator; leave "not checked".

## Verification

Filled in by the verifier; leave "pending".
```

## Severity scale (from `accuracy.py`)

| Severity | Meaning |
|---|---|
| loss | 0, inf, nan or wrong sign where the true value is a normal double; or relative error >= 1 |
| severe | more than 1e6 ulp (relative error > 2.2e-10) and beyond conditioning |
| moderate | more than 1e3 ulp and beyond conditioning |
| minor | beyond conditioning, at most 1e3 ulp |
| ok | within conditioning: ulp <= 8 + cond |

Report every loss, severe and moderate case as a finding. Report minor cases as
findings only if they have a clear mechanism in the code; otherwise mention
them under "Checked and fine" with the maximum error.

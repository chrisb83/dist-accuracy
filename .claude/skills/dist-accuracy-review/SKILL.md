---
name: dist-accuracy-review
description: How to find and document floating-point accuracy problems in one scipy.stats continuous distribution - reading the implemented methods for known failure patterns, writing an mpmath reference, measuring errors in ulp with the bundled accuracy.py, and writing reports/<dist>.md. Use when reviewing or verifying the numerical accuracy of a scipy.stats distribution.
---

# Numerical accuracy review of a scipy.stats continuous distribution

Goal: find where the **implemented** methods of one distribution lose accuracy,
prove each problem with a measured example against mpmath, and document it in
`reports/<dist>.md`. Nothing is fixed at this stage.

Bundled files (in this skill's directory, `${CLAUDE_SKILL_DIR}`):

- `scripts/accuracy.py` - measurement tool (run with `--help` once)
- `references/patterns.md` - failure patterns and how to spot them
- `references/report-template.md` - required report format and severity scale
- `references/burr_ref_example.py` - a complete reference module (worked example)
- `references/burr-example.md` - a complete report (worked example)

Read `patterns.md` and `report-template.md` before starting, and skim the two
burr examples to calibrate depth and style.

## Scope

- Measure only methods the distribution implements itself (or gets from a
  related distribution class): any of pdf, logpdf, cdf, logcdf, sf, logsf,
  ppf, isf that is not the generic `rv_continuous` default. The inventory says
  which.
- Do **not** benchmark inherited generic methods (`sf = 1 - cdf`,
  `logcdf = log(cdf)`, numerical ppf, ...). They are slow and expected to be
  inaccurate. Instead, under "Opportunities", say whether a better custom
  method could be written, with the formula and the regime where the default
  fails (pattern 12).
- Standard form only: loc = 0, scale = 1. SciPy applies loc/scale generically,
  so the underscore methods always work in standard form.
- Do not modify SciPy. Write only `refs/<dist>_ref.py`, `reports/<dist>.md`
  and scratch files under `scratch/<dist>/`.

## Procedure

### 1. Inventory and source

You receive the inventory entry (implemented/inherited methods with line
numbers, shapes and domains, example parameters, helpers, special functions,
related distributions). If you did not, run
`python .claude/skills/dist-accuracy-orchestrate/scripts/inventory.py <dist>`.

Read the class source and every helper or related-distribution method it
calls. Note:
- the docstring pdf and parameter ranges (this defines the parametrization),
- every branch (`np.where`, `xpx.apply_where`, `_lazywhere`, `if`) and its threshold,
- every special function and whether its complementary/log variant is used,
- expressions matching `patterns.md`.

### 2. Hypotheses

Write a short list (in `scratch/<dist>/notes.md`): for each suspicious
expression, the pattern number, the method, and the regime (conditions on
shapes and argument) where it should fail. This drives parameter and point choice.

### 3. Reference module `refs/<dist>_ref.py`

Follow `references/burr_ref_example.py`. Rules:

- **Parametrization:** standard form, SciPy's shapes, as in the docstring pdf.
  The cdf/sf/ppf/isf formulas may be derived from the code or a textbook;
  either is fine because the harness checks them against each other, against
  the integral of the pdf, and against SciPy in the bulk.
- **Use the simplest defining formula** (e.g. `1 - cdf`, `(1 + t)**(-d)`).
  At 400 digits cancellation is harmless; clever rewrites only add risk.
- **mpmath functions only** (`mp.gammainc`, `mp.betainc`, `mp.erfc`,
  `mp.hyp2f1`, `mp.quad` as a last resort). Regularized incomplete functions:
  `mp.gammainc(a, 0, x, regularized=True)`, `mp.betainc(a, b, 0, x, regularized=True)`.
- **No closed-form inverse:** `from accuracy import invert`, then
  `ppf = invert(lambda x: cdf(x, *s), q)` and
  `isf = invert(lambda x: cdf(x, *s), 1 - q)` (`1 - q` is exact at 400
  digits; `invert` works on log(F) or log(1 - F), so tail targets keep full
  relative accuracy). For supports that include negative values pass
  `positive=False`; for decreasing functions `increasing=False`.
- **PARAMS:** the example parameters from the inventory, plus values aimed at
  your hypotheses: tiny and huge shapes (1e-3 ... 1e6 or what the domain
  allows), special relations between shapes, both sides of shape thresholds
  in the code. 4 to 8 sets is typical.
- **x_points / q_points:** add both sides of every threshold in the code and
  the arguments where intermediates overflow or underflow. The default grids
  already reach 1e-300 ... 1e300 and q in [1e-300, 1 - 3e-16].

**Speed.** Every reference value is computed at 400 digits, and the harness
also evaluates references for condition numbers and self-checks, so a slow
reference makes the run very slow. Rules:

- Build everything from the **fastest** function, usually the cdf:
  `sf = 1 - cdf` (exact enough at 400 digits), `isf = invert(cdf, 1 - q)`
  (`1 - q` is exact in mpmath). Example: `mp.gammainc(a, x, mp.inf)` takes
  seconds for small x, `1 - mp.gammainc(a, 0, x)` milliseconds.
- Time one reference call per method before the full run
  (`python -c` with `mp.mp.dps = 400`); anything over ~0.1 s per call needs a
  faster formula.
- If the run takes more than ~15 minutes, reduce PARAMS or lower `DPS`
  (never below 100; the harness checks stability).
- Results are cached in `.cache/dist-accuracy/` per reference file
  version, so reruns after adding points are cheap.

Points whose reference fails or times out are skipped and listed at the end
of the output; say so in the report if a regime could not be measured.

### 4. Run and read

```bash
python ${CLAUDE_SKILL_DIR}/scripts/accuracy.py refs/<dist>_ref.py --verbose --plot scratch/<dist>/errors.png
```

1. **Self-checks first.**
   - **FAIL** means the reference is wrong: it disagrees with SciPy at every
     bulk point (parametrization), the integral of the pdf does not match the
     cdf, a quantile does not invert its cdf, or a value changes with
     precision. Fix the reference before reading anything else.
   - **WARN** (disagreement at some bulk points) can be either side: check
     the points with `--at` and the docstring pdf before deciding.
   - **NOTE** (SciPy returns 0/inf/nan in the bulk) is almost always a real
     SciPy problem; confirm with `--at` and report it as a finding.
2. **Summary table:** per method and region, max/median ulp and counts per severity.
3. **Flagged points:** markdown tables, ready to paste into the report.
4. Use `--at METHOD ARG SHAPES...` to study single points and to produce the
   reproducer lines for the report.

Severity uses the conditioning with respect to the argument only
(shapes are exact user choices). `cond (shapes)` is context: mention it if an
error is smaller than it, but it does not downgrade a finding.

### 5. Explain every flagged regime

For each group of flagged points, find the responsible expression and line
and the pattern. Do not report a finding without a mechanism; if you cannot
find one after a real attempt, report it with "mechanism unclear". Add
points or parameter sets to map the regime's boundaries, then rerun.

Hypotheses from step 2 that did not show up: test them deliberately
(targeted PARAMS or points). If they still do not show, list them under
"Checked and fine" (if the regime was covered) or "Suspected, not confirmed"
(if it could not be reached).

### 6. Opportunities for inherited methods

For each inherited method: is there a closed form or a better formula, and
where does the default fail? At most one cheap demonstration point each
(e.g. `sf` of the generic default vs. your reference at one far-tail x).

### 7. Write `reports/<dist>.md`

Use `references/report-template.md` exactly (headings are checked).
Copy numbers from the tool output. Keep the reference module in `refs/`
so the verifier and the later fixing step can reuse the harness.

## Conventions

- ulp = |computed - reference| / (|reference| * 2.2e-16). 1 ulp is the
  spacing of doubles relative to the value: 100 ulp = 2.2e-14 relative,
  1e6 ulp = 2.2e-10.
- Only results whose true value is a normal double (2.2e-308 ... 1.8e308) are
  measured. Returning 0 where the true value is subnormal is not a finding.
- RuntimeWarnings are recorded separately from accuracy. A warning on an
  accurate result (e.g. from `np.where` evaluating both branches) is worth
  reporting under "RuntimeWarnings" but is not an accuracy finding.
- Errors in `scipy.special` functions: if the distribution calls a special
  function correctly but the special function itself is inaccurate, report it
  under "Shared code" as well.

## Light review mode

For distributions without a practical mpmath reference (the inventory marks
them, e.g. `levy_stable`, `studentized_range`): read the code against
`patterns.md`, list hypotheses under "Suspected, not confirmed", and write
the rest of the template with "not measured". Do not spend time building a
reference by numerical integration of an integral representation.

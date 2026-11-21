# Draft issue for scipy/xsf

**Title:** BUG: `gammainc(a, x)` loses up to ~1e3 ulp for small `x` (prefactor `exp(a*log(x) - x - lgam(a))`)

### Describe your issue

For moderate `a` and small `x`, `gammainc(a, x)` has relative errors of hundreds of ulp, growing
as `x -> 0`. The power series in `igam_series` is fine (all terms positive); the error comes from
the prefactor in `igam_fac`, which for `|a - x| > 0.4*a` computes

```c++
ax = a * log(x) - x - lgam(a);
return exp(ax);
```

For small `x`, `a*log(x)` is large in magnitude (e.g. -173 for a = 1.5, x = 1e-50), so the
absolute rounding error of `ax` turns into a relative error of about `|ax|*eps` after `exp`.
The result is far from underflow here, so this is purely a loss of accuracy, not a range issue.

Worst relative error over 400 log-uniform `x` in [1e-200, 1e-1] per `a` (reference: mpmath,
50 digits; points with P < 1e-300 skipped):

| a    | `gammainc` (xsf) | direct prefactor (below) |
|------|-----------------:|-------------------------:|
| 0.5  | 187 ulp          | 1.9 ulp                  |
| 1.5  | 716 ulp          | 2.4 ulp                  |
| 3    | 1303 ulp         | 1.5 ulp                  |
| 10   | 676 ulp          | 1.7 ulp                  |
| 50   | 370 ulp          | 1.1 ulp                  |

The same branch causes the tiny-`a` errors in #456 (via `lgam(a)`) and the large-`a`
lower-tail errors in scipy/scipy#26535; this report is the third regime (moderate `a`, small
`x`), where the large term is `a*log(x)`.

### Why a few hundred ulp matter

On its own, a relative error of a few hundred ulp (~1e-14) looks harmless, but it propagates
through the formulas built on `gammainc`. A ratio adds the errors of numerator and denominator,
and a subtraction like `1 - ratio` turns them into a much larger relative error when the result
is small. `scipy.stats.argus` (SciPy 1.18.1) is an example: with `a = chi**2/2`,
`b = a*(1 - x**2)` and `P(z) = gammainc(1.5, z)`, it computes `sf = P(b)/P(a)` and
`cdf = 1 - sf`. For small `chi` both arguments are tiny. The table compares the 1.18.1 formulas
with `gammainc` and with an accurate `P` (the `hyp1f1` form below) and nothing else changed:

| argus (1.18.1 formulas) | with `gammainc` | with accurate P |
|---|---:|---:|
| `sf(0.1, 1e-50)`  | 248 ulp     | 1.1 ulp |
| `cdf(0.1, 1e-50)` | 1.6e4 ulp   | 69 ulp  |
| `cdf(0.03, 1e-50)`| 1.8e5 ulp   | 320 ulp |

In the cdf, the error of ~250 ulp in sf is amplified by `sf/cdf` (66 at x = 0.1, 740 at
x = 0.03), so `cdf(0.03, 1e-50)` has only ~11 correct digits. The remaining 69 and 320 ulp are
the unavoidable cancellation of `1 - sf` itself. (scipy.stats now avoids `gammainc` for small
`chi` in argus; `maxwell`, `chi`, `chi2`, `gamma` and `nakagami` go through the same
`gammainc` path for small `x`.)

### Reproducing Code Example

Against mpmath, and against the Kummer form
`P(a, x) = x**a exp(-x) M(1, a + 1, x) / Gamma(a + 1)` (DLMF 8.5.1 divided by `Gamma(a)`), which
needs only SciPy and whose series has positive terms only:

```python
import numpy as np, scipy.special as sc, mpmath as mp
mp.mp.dps = 50

def ulp(got, ref):
    return float(abs(mp.mpf(got) - ref) / np.spacing(float(ref)))

for a, x in [(1.5, 1e-50), (3.0, 1e-50), (10.0, 1e-20)]:
    ref = mp.gammainc(a, 0, x, regularized=True)
    got = sc.gammainc(a, x)
    kummer = x**a * np.exp(-x) * sc.hyp1f1(1, a + 1, x) / sc.gamma(a + 1)
    print(f"a={a}, x={x}: gammainc {ulp(got, ref):.0f} ulp, "
          f"Kummer form {ulp(kummer, ref):.1f} ulp, "
          f"gammainc/Kummer - 1 = {got/kummer - 1:.1e}")
```

```
a=1.5, x=1e-50: gammainc 172 ulp, Kummer form 0.2 ulp, gammainc/Kummer - 1 = -2.8e-14
a=3.0, x=1e-50: gammainc 230 ulp, Kummer form 0.1 ulp, gammainc/Kummer - 1 = -4.7e-14
a=10.0, x=1e-20: gammainc 46 ulp, Kummer form 0.3 ulp, gammainc/Kummer - 1 = 5.8e-15
```

### Possible fix

When `x` is small compared with `a` and `pow(x, a)` does not underflow, compute the prefactor
directly instead of through a log:

```c++
ax = pow(x, a) * exp(-x) / Gamma(a);   // or pow(x, a) * exp(-x) * rgamma(a)
```

`pow` is correctly rounded or close to it in common libms, so the error no longer scales with
`|a*log(x)|`. A Python model of `igam_series` with this prefactor gives the right-hand column
of the table above (<= 2.4 ulp). The log form would still be needed where `pow(x, a)` underflows
or `Gamma(a)` overflows (a > ~171), and for large `a` the Lanczos branch may be the better
choice (cf. scipy/scipy#26535). The Kummer form from the reproducer is an alternative
(spot checks at x = 1e-1, 1e-5, 1e-20, 1e-50, 1e-100, 1e-200: <= 1.6 ulp for a in {1.5, 3, 10},
<= 3.1 ulp for a = 0.5), but adds a
dependency on `hyp1f1`; it is also a convenient mpmath-free reference for tests.

### SciPy/NumPy/Python version and system information

SciPy 1.18.1, NumPy 2.5.4, Windows 11. Same code path on xsf main
(`include/xsf/cephes/igam.h`, `igam_fac`).

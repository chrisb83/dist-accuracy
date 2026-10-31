| Finding | Reproducer | SciPy | Verifier reference | Result | Note |
|---|---|---|---|---|---|
| F1 | `pdf(1.0, 1.5, 10)` | nan | 0.2959488274035536 | verified (loss) | The 2-D mp.quad was too slow, so this reference is in double precision: a log-space integrand with D taken from log_ndtr on the far tail, and an adaptive quad and a trapezoid rule that agree to 1.6e-13. Matches the analyst to 2e-15. When the Cython integrand is called directly at k = 1.5, it gives inf at z = 9 and at s = 1e-20, and nan at z = -40. So the D == 0 mechanism holds, but the nodes give inf as well as nan (inf where exp has not underflowed, and also at small s from cancellation). |
| F1 | `pdf(3.0, 1.9, 10)` | nan | 0.0634752556831439 | verified (loss) | Same method; agrees with the analyst to 2e-15. |
| F1 | `pdf(1.0, 1.5, 1e6)` | nan | 0.303474114 (double; ~1e-9 accuracy) | verified (loss) | At df = 1e6 the double log-constant limits my reference to ~1e-9 (my code is off by 1.2e-9 for k = 2 there). The analyst's 0.3034741138986432 is consistent with this. |
| F2 | `pdf(1e5, 2, 0.1)` | 0.0 | 2.7328352859798695e-07 | verified (loss) | k = 2 references: route A integrates erf / the normal density over u = log s with mp.quad; route B is the Student-t form via hyp2f1 for the cdf and the closed form for the pdf. Routes A and B agree; A at 100 and 200 digits agrees to < 1e-93 unless noted. Outer s range confirmed: nquad's ranges[1] = (0, inf), lines 12424/12451. |
| F2 | `pdf(1e4, 2, 1)` | 3.3771865619410243e-57 | 9.0031629815078011e-09 | verified (loss) | 4.5e15 ulp |
| F2 | `cdf(1e6, 2, 0.1)` | 0.9999999992757298 | 0.78292317714715163 | verified (severe) | 1.25e15 ulp |
| F2 | `cdf(1e5, 2, 1)` | 1.0 | 0.99999099683683903 | verified (severe) | 4.05e10 ulp |
| F3 | `cdf(1e-18, 2, 10)` | 0.0 | 5.5028235383783891e-19 | verified (loss) | Route B (hyp2f1) has no cancellation at small x. |
| F3 | `cdf(1e-8, 2, 10)` | 5.5028235130033e-09 | 5.5028235383783888e-09 | verified (severe) | 2.08e7 ulp |
| F3 | `cdf(1e-6, 2, 10)` | 5.50282353834932e-07 | 5.502823538377884e-07 | verified (moderate) | 2.34e4 ulp |
| F3 | `cdf(1e-4, 3, 10)` | 2.7566444648004866e-09 | 2.7566444725145531e-09 | verified (severe) | 1.26e7 ulp. k = 3 reference uses Owen's T: R_3(w) = 1 - 12 T(w/sqrt2, 1/sqrt3). Averaging over S then gives 1-D integrals for the pdf and cdf; the cdf is taken as int_0^x pdf. Cross-checked with the closed form r_3(w) = (3/sqrt pi) e^{-w^2/4} erf(w/(2 sqrt3)) to 6e-60. |
| F3 | `cdf(1e-8, 3, 10)` | 2.7566793994803397e-17 | 2.7566444771089603e-17 | verified (severe) | 5.71e10 ulp |
| F3 | `pdf(1e-9, 3, 10)` | 5.513288569888619e-10 | 5.5132889542179208e-10 | verified (severe) | 3.14e8 ulp; `pdf(1e-6, 3, 10)` 1.16e5 ulp also reproduced |
| F4 | `pdf(1e10, 2, 10)` | 0.0 | 7.875e-104 | verified (loss) | At 100 digits route A was not converged (1e-5); at 200 digits it agrees with the closed form to 6e-104. |
| F4 | `pdf(100, 2, 10)` | 6.223183851362113e-16 | 7.7889352591072694e-16 | verified (severe) | 9.05e14 ulp |
| F4 | `pdf(10, 2, 10)` | 2.8890410275144366e-05 | 2.8890410265782959e-05 | verified (severe) | 1.46e6 ulp |
| F5 | `pdf(30, 2, 100000)` | 3.9140532599010535e-98 | 1.7922343384951665e-98 | verified (loss) | 5.33e15 ulp. Route A at 100 digits agrees only to 4e-14; at 200 digits it agrees with the closed form to 3e-128. |
| F5 | `pdf(1, 2, 100000)` | 0.4393912894677224 | 0.43938936713686688 | verified (severe) | 1.97e10 ulp |
| F5 | `cdf(1, 2, 100000)` | 0.5204998778130465 | 0.52049823009885773 | verified (severe) | 1.43e10 ulp. SciPy's value equals erf(1/2) = R_2(1) at S = 1, which confirms the asymptotic switch (lines 12419/12446). |
| F6 | `pdf(9.5, 2, 99999)` | 5.378034985538992e-13 | 9.010875057797595e-11 | verified (severe) | 4.48e15 ulp |
| F6 | `cdf(1e-10, 2, 99999)` | 3.1169341624381133e-13 | 5.6418817306145571e-11 | verified (severe) | 4.48e15 ulp |
| F7 | `cdf(1, 2, 99999)` | 0.5204982300875673 | 0.52049823008238046 | verified (moderate) | 4.49e4 ulp. The mechanism is consistent, with one correction: `_studentized_range_cdf_logconst(3, 99999)` is too large by 1.24e-11 against the exact 50005.78270341152019, not 9.8e-12. The sign and size match the +1.0e-11 relative bias. |
| F7 | `pdf(1, 2, 99999)` | 0.4393893671219536 | 0.43938936711764344 | verified (moderate) | 4.42e4 ulp |

Verified by dist-accuracy-verifier: 7 of 7 verified.

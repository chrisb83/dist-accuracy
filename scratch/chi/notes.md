# chi hypotheses (a = df/2, y = x**2/2)
H1 (p5/p3) logpdf L1568-1569: terms 0.5*log2*df, gammaln(df/2), (df-1)log x, x**2/2 cancel for large df near mode -> relative error ~ df*log(df)*eps/|logpdf|.
H2 (p3) pdf L1565: exp(logpdf) turns abs error of logpdf into relative error -> ~ df log df ulp for large df.
H3 (p3) logpdf: x**2 overflows for 1.34e154 < x < 1.9e154 where true logpdf ~ -x^2/2 is still a finite double -> -inf.
H4 (p4/p10) cdf L1572: x**2 underflows (x < ~1.5e-162) / subnormal (x < ~1.5e-154) while cdf ~ x**df/(2^(df/2) Gamma(df/2+1)) is normal for small df -> 0 or large error (subnormal error * df/2).
H5 (p4/p8) ppf L1578: gammaincinv(a, q) underflows/subnormal for small q, small df although sqrt(2y) is normal -> 0. isf L1581 same for q -> 1.
H6 (p10, shared) cdf/sf/ppf/isf: cephes igam 2000-term cap (reports/gamma.md F1) for a > ~2e5 (df > ~4e5) and 4.5 sqrt(a) < a - y < ~15 sqrt(a), i.e. x**2 in df - (9..30)*sqrt(df/2).
H7 (p10, shared) ppf/isf for tiny df: igami exp(exponent/a) (gamma F5).
H8 (p10, shared) cdf/sf prefactor igam_fac exp(a log y - y - lgam a) for |y-a|>0.4a (gamma F6).

## Outcome
H1 F6, H2 F5, H3 F3, H4 F1, H5 F2, H6 F4+F7 (df=2e6 main run, df=2e8 aux run scratch/chi/chi_df2e8_ref.py --no-checks), H7 F8, H8 F9 (minor).
Runs: scratch/chi/run1.txt (+ run1.json), scratch/chi/run_df2e8.txt.

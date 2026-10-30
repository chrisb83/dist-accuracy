# maxwell hypotheses (SciPy 1.18.1, lines 7158-7177)

- H1 (pattern 3) logpdf line 7165: `0.5*x*x` overflows for x > 1.34e154; true logpdf finite
  (> -1.8e308) until x ~ 1.9e154 -> -inf (as chi F3).
- H2 (pattern 4) pdf line 7160: `exp(-x*x/2)` subnormal for x > 37.64 while pdf (x**2 factor ~1400)
  stays normal up to x ~ 37.83 -> up to ~1e3 ulp in that window.
- H3 (pattern 4/10) cdf/sf lines 7168/7174: `x*x/2` subnormal/zero for x < 2e-154; but cdf ~ x**3 then
  < 1e-460, not measurable; sf = 1. Expect nothing.
- H4 pdf small x: x*x subnormal for x < 1.5e-154 but pdf < 1.8e-308 there too. Edge only.
- H5 (pattern 10) gammainc/gammaincc at a = 1.5: igam prefactor (gamma F6/chi F9, minor ~187 ulp at df ~ 1-3,
  small x). Test x in 1e-100 ... 1e-5.
- H6 ppf/isf via gammaincinv/gammainccinv(1.5, q): small q, q near 1; chi report saw 88 ulp at df=3 q~1e-288.
- Inherited logcdf/logsf: logsf -inf once sf underflows (x > ~38.6); closed form
  logsf = -x**2/2 + log(erfcx(x/sqrt2) + sqrt(2/pi)*x). logcdf -inf once cdf underflows (x < ~1e-103);
  closed form for small x: log(sqrt(2/pi)/3) + 3 log x + log1p(-3x^2/10 ...), or log(gammainc).

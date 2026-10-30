# invweibull hypotheses (t = x**-c)
- H1 pattern 4, pdf: large c, x > 1: pow(x, -c-1) subnormal while c*x**(-c-1) normal ((c+1)log x in [708, 708+log c]). Expect up to ~c ulp.
- H2 pattern 3, pdf: c in ~[0.002, 0.0093], x near 2.2e-308: pow(x, -c-1) = t/x overflows while exp(-t) not small -> inf.
- H3 pattern 6-like, pdf/cdf: rounding of pow(x,-c) amplified by t in exp(-t); beyond cond c*t when c < 1.
- H4 ppf/isf: rounding of -log(q) amplified by 1/c; beyond cond when |log q| large (small q, small c). Minor.
- H5 pdf: rounded shape -c-1 raised: x**(-c-1) with c small inexact? -c-1 rounding: error |log x| * eps*(c+1) relative -> up to 700 ulp for x -> 0 or inf.
- logpdf/logcdf/logsf inherited: log(pdf) fails when exp(-t) underflows (t>745) or for x->inf large c; logcdf = -t closed form; logsf = log(-expm1(-t)) fails where sf underflows.

## Self-check FAIL for c = 0.005 (integral of pdf vs cdf)
Harness uses plain mp.quad on [ppf(0.1), ppf(0.5)] = [3.6e-73, 6.8e31]; with x = exp(u)
substitution (20 subintervals) the integral matches cdf difference to 1e-30 (0.0 and -3.7e-30).
Artifact of the quadrature, not the reference. PARAMS reordered so self-checks use 10.58, 0.1, 1.

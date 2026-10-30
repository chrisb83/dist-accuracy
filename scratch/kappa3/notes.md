# kappa3 hypotheses
- H1 (p6/p9) pdf l7570, cdf l7573: (a + x**a)**(-1/a[-1]) amplifies rounding of a + x**a by ~1/a -> small a (1e-3: ~1e3 ulp, 1e-6: ~1e6 ulp, 1e-300: loss).
- H2 (p3) cdf l7573: x**a overflows for a*log(x) > 709.8 -> cdf = x*0 = 0 while true ~1. Loss (a=2, x=1e155; a=10, x=1e31; a=1e3, x=3).
- H3 (p1/p7) sf l7577-7588: 1 - cdf for sf >= 0.01 fine; sf2 branch: x**-a overflow? only for x<1 where sf not small. Check large a near x~1.
- H4 (p8) ppf l7592: q**-a - 1 cancels for small a (loss ~1/(a|log q|) then *1/a), and q->1; q**-a overflows for a*log(1/q) > 709.8 -> ppf 0 (a=1e3, q=0.4 in bulk!).
- H5 (p8/p3) isf l7595-7597: expm1(-a*log1p(-q)) overflows for large a / q->1 -> isf 0 (a=1e3, q=0.6). Small a: ^(1/a) amplification but cond also 1/a.
- H6 sf branch selection when sf from 1-cdf is computed from an overflowed cdf (cdf=0 -> sf=1): only when sf subnormal? check.

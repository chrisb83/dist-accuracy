# gausshyper hypotheses

Only `_pdf` is implemented (lines 4915-4918):

    normalization_constant = sc.beta(a, b) * sc.hyp2f1(c, a, a + b, -z)
    return 1./nc * x**(a-1) * (1-x)**(b-1) / (1 + z*x)**c

- H1 (pattern 3/4): large a, b: sc.beta(a, b) is tiny (or underflows to 0 for
  a = b >~ 540); x**(a-1) underflows to 0 while the pdf (x**(a-1)/B) is normal.
  Regime: a = b = 300, x <~ 0.08 (x**299 < 1e-308). Same for (1-x)**(b-1) near 1.
- H2 (pattern 3): c < 0 large, z large: hyp2f1(c, a, a+b, -z) ~ z**(-c) overflows
  (or returns nan); (1+zx)**c overflows too. pdf -> nan/0 for all x.
- H3 (pattern 10 / shared): hyp2f1 inaccurate for large c with small z, and for
  hyp2f1(c, a; a+b; w) with c == a (equal first parameters) at w << -1
  (degenerate 1/w transformation). Constant relative error at every x.
- H4 (pattern 6): (1-x)**(b-1) with x < 0.5 and large b: rounding of 1 - x
  amplified by b; cond(x) = (b-1) x/(1-x) small for small x.
- H5 (pattern 6): (1 + z*x)**c with large c, small z*x: rounding of 1 + zx
  amplified by c; cond(x) ~ c z x.
- H6: z near -1, x near 1: 1 + z*x cancels; rounding of z*x relative to 1+zx;
  expected within cond(x).
- H7: tiny shapes a = b = 0.01: beta large, pdf ~ x**-0.99 near 0; probably fine.
- H8: x**(a-1) overflow for a < 1 and subnormal x (not on grid; harness only
  normal x).

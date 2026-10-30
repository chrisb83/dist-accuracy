# lognorm recheck (F6 mechanism, subnormal x in logpdf)
- f6_split.py / f6_all.py: split of F6 error into ndtr(z_double) error vs. argument rounding F(z_double) - F(z_exact).
- ndtr_sweep.py: sc.ndtr at exact double z in [-37.5, -5]; log_ndtr(z > 0) = ndtr(-z) error.
- subnormal.txt / repro.txt: logpdf/pdf at subnormal x, s in {1, 0.954, 1e-3, 10, 30, 100, 1e3, 1e10}.

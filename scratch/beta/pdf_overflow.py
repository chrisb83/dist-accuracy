"""beta.pdf raises OverflowError for tiny x (accuracy.py aborts on it)."""
import sys
sys.path.insert(0, "refs")
import mpmath as mp
import scipy.stats as st
import beta_ref as R
mp.mp.dps = 120
for x, a, b in [(1e-310, 0.5, 5.0), (5e-324, 0.5, 0.5), (1e-310, 1e-3, 1e-3),
                (5e-309, 1.0, 3.0), (1e-308, 1.5, 5.0), (1e-306, 20.0, 50.0)]:
    r = R.pdf(x, a, b)
    try:
        v = repr(float(st.beta.pdf(x, a, b)))
    except Exception as e:
        v = f"{type(e).__name__}"
    print(f"pdf({x!r}, {a}, {b}) scipy={v} ref={mp.nstr(r, 17)}")
    print(f"   logpdf scipy={st.beta.logpdf(x, a, b)!r} ref={mp.nstr(R.logpdf(x, a, b), 17)}")

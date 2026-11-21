"""Recompute the argus test reference values and compare with SciPy.

Usage:
    python argus.py                          # scipy found on sys.path
    python argus.py PATH/TO/site-packages    # a specific scipy build

First, as a separate sanity check of the mpmath references, it compares them
with SciPy for a few easy cases (moderate x and chi, see check_easy), where
SciPy is accurate with and without the fix.

Then, for every case in CASES (the same values as in
the new rows and tests in
scipy/stats/tests/test_distributions.py::TestArgus), this script
  * recomputes the reference value with mpmath (closed forms, mp.dps = 100),
  * marks it with (!) if the hard-coded test value differs from the
    recomputed one by more than 1 ulp,
  * evaluates the given SciPy version and reports the relative error and
    whether it is within the test tolerance.
"""
import sys
import warnings

if len(sys.argv) > 1:
    sys.path.insert(0, sys.argv[1])

import mpmath
import numpy as np
import scipy
from mpmath import mp
from scipy import stats

mp.dps = 100


# ---------------------------------------------------------------------------
# mpmath reference: closed forms of the standard argus distribution
#   pdf(x) = chi**3/(sqrt(2*pi)*Psi(chi)) * x*sqrt(1 - x**2)*exp(-a*(1 - x**2))
# with a = chi**2/2 and Psi(chi) = Phi(chi) - chi*phi(chi) - 1/2
# = P(3/2, a)/2, where P(s, t) is the regularized lower incomplete gamma
# function. Substituting t = a*(1 - s**2) in the integral of the pdf gives,
# with b = a*(1 - x**2),
#   sf(x)  = P(3/2, b)/P(3/2, a)  = int_0^b  t**0.5 exp(-t) dt / int_0^a (...)
#   cdf(x) = int_b^a t**0.5 exp(-t) dt / int_0^a t**0.5 exp(-t) dt
# mpmath evaluates the incomplete gamma integrals over [0, b], [b, a] and
# [0, a] directly, so no difference of two rounded numbers is formed.
# Psi is written via P(3/2, a) and not via Phi, phi; that is a different
# formula from the one in scipy (and the identity is checked in main()).
# ---------------------------------------------------------------------------

def _gi(lo, hi):
    """Regularized int_lo^hi t**0.5 exp(-t) dt / Gamma(3/2)."""
    return mp.gammainc(mp.mpf(1.5), lo, hi, regularized=True)


def _ab(x, chi):
    x, chi = mp.mpf(x), mp.mpf(chi)
    a = chi**2/2
    return x, chi, a, a*(1 - x**2)


def pdf(x, chi):
    x, chi, a, b = _ab(x, chi)
    psi = _gi(0, a)/2
    return (chi**3/(mp.sqrt(2*mp.pi)*psi) * x*mp.sqrt(1 - x**2)
            * mp.exp(-b))


def logpdf(x, chi):
    x, chi, a, b = _ab(x, chi)
    psi = _gi(0, a)/2
    return (3*mp.log(chi) - mp.log(mp.sqrt(2*mp.pi)*psi) + mp.log(x)
            + mp.log(1 - x**2)/2 - b)


def sf(x, chi):
    x, chi, a, b = _ab(x, chi)
    return _gi(0, b)/_gi(0, a)


def cdf(x, chi):
    x, chi, a, b = _ab(x, chi)
    return _gi(b, a)/_gi(0, a)


def phi(x, chi):
    """_argus_phi(chi) = Psi(chi) = P(3/2, chi**2/2)/2 (x is unused)."""
    chi = mp.mpf(chi)
    return _gi(0, chi**2/2)/2


REFS = {"pdf": pdf, "logpdf": logpdf, "cdf": cdf, "sf": sf, "phi": phi}


def _scipy(meth, x, chi):
    if meth == "phi":
        from scipy.stats._continuous_distns import _argus_phi
        return float(_argus_phi(chi))
    return float(getattr(stats.argus, meth)(x, chi))


# ---------------------------------------------------------------------------
# (method, x, chi, value in the test, rtol in the test)
# ---------------------------------------------------------------------------

CASES = [
    # test_pdf_small_chi (new rows; x = 0.1, 0.5, 0.9, test rtol 5e-15)
    ('pdf', 0.1, 1e-103, 0.298496231131986, 5e-15),
    ('pdf', 0.5, 1e-103, 1.299038105676658, 5e-15),
    ('pdf', 0.9, 1e-103, 1.1769027147559818, 5e-15),
    ('pdf', 0.1, 1e-200, 0.298496231131986, 5e-15),
    ('pdf', 0.5, 1e-200, 1.299038105676658, 5e-15),
    ('pdf', 0.9, 1e-200, 1.1769027147559818, 5e-15),
    ('pdf', 0.1, 1.414, 0.19515544088652562, 5e-15),
    ('pdf', 0.5, 1.414, 1.0796000612831684, 5e-15),
    ('pdf', 0.9, 1.414, 1.7120365668684747, 5e-15),
    ('pdf', 0.1, 1.4143, 0.19511729382031046, 5e-15),
    ('pdf', 0.5, 1.4143, 1.07949893937232, 5e-15),
    ('pdf', 0.9, 1.4143, 1.712282958177796, 5e-15),
    # test_sf_small_chi (new rows; x = 0.1, 0.5, 0.9, test rtol 5e-15)
    ('sf', 0.1, 1e-103, 0.9850375627355538, 5e-15),
    ('sf', 0.5, 1e-103, 0.649519052838329, 5e-15),
    ('sf', 0.9, 1e-103, 0.08281907992727278, 5e-15),
    ('sf', 0.1, 1e-200, 0.9850375627355538, 5e-15),
    ('sf', 0.5, 1e-200, 0.649519052838329, 5e-15),
    ('sf', 0.9, 1e-200, 0.08281907992727278, 5e-15),
    ('sf', 0.1, 1.414, 0.9902664037438805, 5e-15),
    ('sf', 0.5, 1.414, 0.7430384738016715, 5e-15),
    ('sf', 0.9, 1.414, 0.130148572830973, 5e-15),
    ('sf', 0.1, 1.4143, 0.990268326997523, 5e-15),
    ('sf', 0.5, 1.4143, 0.7430759260374723, 5e-15),
    ('sf', 0.9, 1.4143, 0.13017163822838482, 5e-15),
    # test_cdf_small_chi (new rows; x = 0.1, 0.5, 0.9, test rtol 1e-13)
    ('cdf', 0.1, 1e-103, 0.014962437264446246, 1e-13),
    ('cdf', 0.5, 1e-103, 0.350480947161671, 1e-13),
    ('cdf', 0.9, 1e-103, 0.9171809200727272, 1e-13),
    ('cdf', 0.1, 1e-200, 0.014962437264446246, 1e-13),
    ('cdf', 0.5, 1e-200, 0.350480947161671, 1e-13),
    ('cdf', 0.9, 1e-200, 0.9171809200727272, 1e-13),
    ('cdf', 0.1, 1.414, 0.009733596256119468, 1e-13),
    ('cdf', 0.5, 1.414, 0.2569615261983284, 1e-13),
    ('cdf', 0.9, 1.414, 0.869851427169027, 1e-13),
    ('cdf', 0.1, 1.4143, 0.00973167300247699, 1e-13),
    ('cdf', 0.5, 1.4143, 0.25692407396252764, 1e-13),
    ('cdf', 0.9, 1.4143, 0.8698283617716152, 1e-13),
    # test_sf_near_1 (new rows)
    ('sf', 0.999999, 1e-100, 2.8284250035481114e-09, 5e-15),
    ('sf', 0.9999999999999999, 1e-100, 3.3087224502121103e-24, 5e-15),
    ('sf', 0.999999, 10.0, 7.522070804082667e-07, 5e-15),
    # test_cdf_near_1_small_chi
    ('cdf', 0.999999, 1e-100, 0.999999997171575, 5e-15),
    # test_argus_phi_small_chi (all rows; x is unused)
    ('phi', None, 0.9, 0.07646314974436118, 1e-14),
    ('phi', None, 0.5, 0.015429797891863365, 1e-14),
    ('phi', None, 0.1, 0.00013258252932780493, 1e-14),
    ('phi', None, 0.01, 1.3297677078224565e-07, 1e-14),
    ('phi', None, 1e-3, 1.3298072023958999e-10, 1e-14),
    ('phi', None, 1e-4, 1.3298075973486864e-13, 1e-14),
    ('phi', None, 1e-6, 1.3298076013377098e-19, 1e-14),
    ('phi', None, 1e-9, 1.3298076013381093e-28, 1e-14),
    # test_cdf_large_chi
    ('cdf', 0.5, 10.0, 3.623303813850349e-16, 1e-14),
    ('cdf', 0.8, 3.0, 0.3366511316738839, 5e-15),
    ('cdf', 0.95, 100.0, 4.767900552974467e-211, 5e-14),
    ('cdf', 0.999999, 1e4, 4.218750983153304e-43, 2e-14),
    ('cdf', 0.8819171028062798, 3.0, 0.5595041887383695, 5e-15),
    ('cdf', 0.8819171045701141, 3.0, 0.5595041947252353, 5e-15),
    # test_logpdf
    ('logpdf', 0.35, 1e-100, -0.01654899765162614, 5e-15),
    ('logpdf', 0.35, 0.001, -0.01654913640163471, 1e-14),
    ('logpdf', 0.5, 1e-102, 0.26162407188227393, 5e-15),
    ('logpdf', 0.5, 1e-103, 0.26162407188227393, 5e-15),
    ('logpdf', 1e-300, 1e-110, -689.6769156095456, 5e-15),
    ('logpdf', 0.99, 1e155, -9.950000000000008e+307, 5e-15),
    ('logpdf', 0.99999999, 0.5, -7.690692157986852, 5e-15),
    ('logpdf', 0.99999999, 10.0, -2.181803865346421, 5e-15),
    ('logpdf', 0.999999999999, 1.0, -12.079022009773738, 5e-15),
    ('logpdf', 0.5, 1.414, 0.07659065892640939, 5e-15),
    ('logpdf', 0.5, 1.4143, 0.07649698845470652, 5e-15),
    # test_pdf_tiny_x
    ('pdf', 1e-280, 0.001, 2.999999400000034e-280, 5e-15),
    ('pdf', 1e-290, 1.0, 2.4349495043718023e-290, 5e-15),
    ('pdf', 1e-300, 1e-110, 3e-300, 5e-15),
    # test_pdf_near_1
    ('pdf', 0.99999999, 0.5, 0.00045706170331619064, 5e-15),
    ('pdf', 0.99999999, 10.0, 0.11283780274470981, 5e-15),
    ('pdf', 0.999999999999, 1.0, 5.677372559683347e-06, 5e-15),
    # test_large_chi_at_mode (x = double nearest to the mode for chi = 1e7)
    ('pdf', 0.999999999999995, 1e7, 48394137170617.998, 5e-15),
    ('cdf', 0.999999999999995, 1e7, 0.80144535872829902, 5e-15),
    ('sf', 0.999999999999995, 1e7, 0.19855464127170098, 5e-15),
]


# ---------------------------------------------------------------------------
# Sanity check of the references: "easy" cases (moderate x and chi, no
# underflow, overflow or cancellation), where SciPy was already accurate
# before the fix. Agreement there shows that the mpmath formulas above
# describe the same distribution as scipy.stats.argus.
# ---------------------------------------------------------------------------

EASY_X = [0.3, 0.6, 0.9]
EASY_CHI = [0.5, 1.0, 3.0]
EASY_RTOL = 1e-13


def check_easy():
    print(f"Sanity check: mpmath references vs SciPy for easy cases "
          f"(x in {EASY_X}, chi in {EASY_CHI}, rtol {EASY_RTOL:.0e})")
    hdr = f"{'method':8s} {'max rel err':>12s}  {'at (x, chi)':s}"
    print(hdr)
    print("-"*len(hdr))
    n_fail = 0
    for meth in ["pdf", "logpdf", "cdf", "sf", "phi"]:
        worst, where = 0.0, None
        for chi in EASY_CHI:
            for x in ([None] if meth == "phi" else EASY_X):
                ref = REFS[meth](x, chi)
                rel = float(abs((mp.mpf(_scipy(meth, x, chi)) - ref)/ref))
                if rel >= worst:
                    worst, where = rel, (x, chi)
        ok = worst <= EASY_RTOL
        n_fail += not ok
        print(f"{meth:8s} {worst:12.2e}  {where}  {'ok' if ok else 'FAIL'}")
    print(f"{n_fail} of 5 methods outside rtol {EASY_RTOL:.0e}\n")


def main():
    print(f"scipy {scipy.__version__} ({scipy.__file__})\n"
          f"mpmath {mpmath.__version__}, mp.dps = {mp.dps}\n")

    # self-check: Psi(chi) via Phi, phi equals P(3/2, chi**2/2)/2
    for chi in ["0.001", "1", "10"]:
        chi = mp.mpf(chi)
        psi = mp.ncdf(chi) - chi*mp.npdf(chi) - mp.mpf(1)/2
        assert abs(psi/(_gi(0, chi**2/2)/2) - 1) < mp.mpf(10)**-90

    check_easy()

    print("Test cases (the new and changed rows in TestArgus)")
    hdr = (f"{'case':40s} {'scipy':>24s} {'reference':>24s} "
           f"{'rel err':>9s} {'rtol':>7s}")
    print(hdr)
    print("-"*len(hdr))
    n_fail = n_stale = 0
    for meth, x, chi, test_ref, rtol in CASES:
        ref = REFS[meth](x, chi)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = _scipy(meth, x, chi)
        if np.isfinite(got):
            rel = float(abs((mp.mpf(got) - ref)/ref))
        else:
            rel = np.inf
        ok = rel <= rtol
        n_fail += not ok
        stale = abs(test_ref - float(ref)) > 2.3e-16*abs(float(ref))
        n_stale += stale
        label = f"{meth}({x!r}, {chi!r})"
        print(f"{label:40s} {got!r:>24} {float(ref)!r:>24} {rel:9.2e} "
              f"{rtol:7.0e} {'ok' if ok else 'FAIL'}{' (!)' if stale else ''}")
    print(f"\n{n_fail} of {len(CASES)} outside the test tolerance; "
          f"{n_stale} hard-coded values differ from the recomputed ones "
          "by > 1 ulp (marked (!))")


if __name__ == "__main__":
    main()

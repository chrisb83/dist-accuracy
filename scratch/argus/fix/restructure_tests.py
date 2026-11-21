p = 'scipy/stats/tests/test_distributions.py'
s = open(p, 'rb').read().decode('utf-8')
assert '\r\n' in s
s = s.replace('\r\n', '\n')

def sub(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)

# --- pdf small chi: new rows
sub("""         (1e-9, (0.298496231131986, 1.299038105676658, 1.1769027147559818))])
    def test_pdf_small_chi(self, chi, expected):""",
"""         (1e-9, (0.298496231131986, 1.299038105676658, 1.1769027147559818)),
         # _argus_phi(chi) underflows for chi < ~1e-102.5 (inf before)
         (1e-100, (0.298496231131986, 1.299038105676658, 1.1769027147559818)),
         (1e-103, (0.298496231131986, 1.299038105676658, 1.1769027147559818)),
         (1e-200, (0.298496231131986, 1.299038105676658, 1.1769027147559818)),
         # both sides of the switch at chi = sqrt(2)
         (1.414, (0.19515544088652562, 1.0796000612831684, 1.7120365668684747)),
         (1.4143, (0.19511729382031046, 1.07949893937232, 1.712282958177796))])
    def test_pdf_small_chi(self, chi, expected):""")

# --- sf small chi: new rows
sub("""         (1e-9, (0.9850375627355538, 0.649519052838329, 0.0828190799272728))])
    def test_sf_small_chi(self, chi, expected):""",
"""         (1e-9, (0.9850375627355538, 0.649519052838329, 0.0828190799272728)),
         # gammainc(1.5, z) inaccurate for tiny z (~400 ulp before), and
         # _argus_phi(chi) underflows for chi < ~1e-102.5 (nan before)
         (1e-100, (0.9850375627355538, 0.649519052838329, 0.08281907992727278)),
         (1e-103, (0.9850375627355538, 0.649519052838329, 0.08281907992727278)),
         (1e-200, (0.9850375627355538, 0.649519052838329, 0.08281907992727278)),
         # both sides of the switch at chi = sqrt(2)
         (1.414, (0.9902664037438805, 0.7430384738016715, 0.130148572830973)),
         (1.4143, (0.990268326997523, 0.7430759260374723, 0.13017163822838482))])
    def test_sf_small_chi(self, chi, expected):""")

# --- sf near 1: new rows
sub("""         (0.999999999, 10.0, 2.3788319094393724e-11)])
    def test_sf_near_1(self, x, chi, expected):""",
"""         (0.999999999, 10.0, 2.3788319094393724e-11),
         # numerator gammainc(1.5, chi**2*(1 - x**2)/2) underflowed (0 before)
         (0.999999, 1e-100, 2.8284250035481114e-09),
         (0.9999999999999999, 1e-100, 3.3087224502121103e-24)])
    def test_sf_near_1(self, x, chi, expected):""")

# --- cdf small chi: new rows
sub("""         (1e-9, (0.014962437264446245, 0.350480947161671, 0.9171809200727272))])
    def test_cdf_small_chi(self, chi, expected):""",
"""         (1e-9, (0.014962437264446245, 0.350480947161671, 0.9171809200727272)),
         # _argus_phi(chi) underflows for chi < ~1e-102.5 (nan before)
         (1e-100, (0.014962437264446246, 0.350480947161671, 0.9171809200727272)),
         (1e-103, (0.014962437264446246, 0.350480947161671, 0.9171809200727272)),
         (1e-200, (0.014962437264446246, 0.350480947161671, 0.9171809200727272)),
         # both sides of the switch at chi = sqrt(2)
         (1.414, (0.009733596256119468, 0.2569615261983284, 0.869851427169027)),
         (1.4143, (0.00973167300247699, 0.25692407396252764, 0.8698283617716152))])
    def test_cdf_small_chi(self, chi, expected):""")

# --- replace test_accuracy by focused tests
i = s.index("    # Reference values computed with mpmath (mp.dps = 100) from the closed")
j = s.index("        assert_allclose(res, ref, rtol=rtol)\n", i) + len("        assert_allclose(res, ref, rtol=rtol)\n")
s = s[:i] + '''    # Expected values were computed with mpmath (mp.dps = 100) as
    # cdf = gammainc(1.5, b, a)/gammainc(1.5, 0, a) (regularized), with
    # a = chi**2/2 and b = a*(1 - x**2), see gh-XXXXX.
    @pytest.mark.parametrize(
        'x, chi, expected, rtol',
        # cdf = 1 - sf cancelled when the cdf is small (0 or no digits before)
        [(0.5, 10.0, 3.623303813850349e-16, 1e-14),  # cond ~25
         (0.8, 3.0, 0.3366511316738839, 5e-15),
         (0.95, 100.0, 4.767900552974467e-211, 5e-14),  # cond ~9e3
         (0.999999, 1e4, 4.218750983153304e-43, 2e-14),  # cond ~1e8
         # both sides of the switch at chi**2*(1 - x**2)/2 = 1
         (0.8819171028062798, 3.0, 0.5595041887383695, 5e-15),
         (0.8819171045701141, 3.0, 0.5595041947252353, 5e-15)])
    def test_cdf_large_chi(self, x, chi, expected, rtol):
        assert_allclose(stats.argus.cdf(x, chi), expected, rtol=rtol)

    # Expected values were computed with mpmath (mp.dps = 100) as
    # 3*log(chi) - log(sqrt(2*pi)*gammainc(1.5, 0, a)/2) + log(x)
    # + log(1 - x**2)/2 - a*(1 - x**2) with a = chi**2/2, see gh-XXXXX.
    @pytest.mark.parametrize(
        'x, chi, expected, rtol',
        # 3*log(chi) - log(_argus_phi(chi)) cancelled for small chi
        [(0.35, 1e-100, -0.01654899765162614, 5e-15),
         (0.35, 0.001, -0.01654913640163471, 1e-14),  # cond ~52
         (0.5, 1e-102, 0.26162407188227393, 5e-15),
         # _argus_phi(chi) underflowed (inf before)
         (0.5, 1e-103, 0.26162407188227393, 5e-15),
         (1e-300, 1e-110, -689.6769156095456, 5e-15),
         # chi**2 overflowed (-inf before)
         (0.99, 1e155, -9.950000000000008e+307, 5e-15),
         # both sides of the switch at chi = sqrt(2)
         (0.5, 1.414, 0.07659065892640939, 5e-15),
         (0.5, 1.4143, 0.07649698845470652, 5e-15)])
    def test_logpdf(self, x, chi, expected, rtol):
        assert_allclose(stats.argus.logpdf(x, chi), expected, rtol=rtol)

    # pdf was exp(logpdf), which turns the absolute error of log(x) into a
    # relative error of the pdf. Expected values were computed with mpmath.
    @pytest.mark.parametrize(
        'x, chi, expected',
        [(1e-280, 0.001, 2.999999400000034e-280),
         (1e-290, 1.0, 2.4349495043718023e-290),
         (1e-300, 1e-110, 3e-300)])
    def test_pdf_tiny_x(self, x, chi, expected):
        assert_allclose(stats.argus.pdf(x, chi), expected, rtol=5e-15)
''' + s[j:]
open(p, 'wb').write(s.replace('\n', '\r\n').encode('utf-8'))
